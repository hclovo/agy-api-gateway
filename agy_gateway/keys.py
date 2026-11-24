"""Database-backed API key management, verification, usage accounting, and RPM rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from agy_gateway.db import DatabaseBackend


class KeyAuthError(Exception):
    """Raised when API key authentication, authorization, or rate limiting fails."""

    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ApiKeyRecord:
    """Metadata and usage counters for an issued gateway API key."""

    id: str
    name: str
    key_prefix: str
    rpm_limit: int
    allowed_models: List[str]
    enabled: bool
    created_at: str
    last_used_at: Optional[str]
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    allowed_accounts: List[str] = ()  # type: ignore[assignment]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _mask_key(raw_key: str) -> str:
    if len(raw_key) <= 14:
        return f"{raw_key[:4]}...{raw_key[-2:]}"
    return f"{raw_key[:10]}...{raw_key[-4:]}"


class KeyStore:
    """Thread-safe PostgreSQL/SQLite store for issued API keys and sliding-window RPM checks."""

    def __init__(
        self,
        db_path: Path | str = "./data/gateway.db",
        database_url: str = "",
    ) -> None:
        self.db_path = Path(db_path)
        self.database_url = database_url
        self._db = DatabaseBackend(db_path=self.db_path, database_url=database_url)
        self._lock = threading.Lock()
        self._rpm_windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._init_db()

    @property
    def backend_name(self) -> str:
        return self._db.backend_name

    def _connect(self):
        return self._db.connect()

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.init_pragmas()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    key_hash TEXT UNIQUE NOT NULL,
                    key_prefix TEXT NOT NULL,
                    rpm_limit INTEGER NOT NULL DEFAULT 30,
                    allowed_models TEXT NOT NULL DEFAULT '[]',
                    allowed_accounts TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    total_requests BIGINT NOT NULL DEFAULT 0,
                    total_input_tokens BIGINT NOT NULL DEFAULT 0,
                    total_output_tokens BIGINT NOT NULL DEFAULT 0
                )
                """
            )
            cols = conn.column_names("api_keys")
            if "allowed_accounts" not in cols:
                conn.execute(
                    "ALTER TABLE api_keys ADD COLUMN allowed_accounts TEXT NOT NULL DEFAULT '[]'"
                )

    @staticmethod
    def _row_to_record(row: Any) -> ApiKeyRecord:
        keys = row.keys()
        raw_accs = row["allowed_accounts"] if "allowed_accounts" in keys else "[]"
        return ApiKeyRecord(
            id=row["id"],
            name=row["name"],
            key_prefix=row["key_prefix"],
            rpm_limit=int(row["rpm_limit"]),
            allowed_models=json.loads(row["allowed_models"]),
            allowed_accounts=json.loads(raw_accs or "[]"),
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            total_requests=int(row["total_requests"]),
            total_input_tokens=int(row["total_input_tokens"]),
            total_output_tokens=int(row["total_output_tokens"]),
        )

    def create_key(
        self,
        name: str,
        rpm_limit: int = 30,
        allowed_models: Optional[List[str]] = None,
        allowed_accounts: Optional[List[str]] = None,
    ) -> Tuple[ApiKeyRecord, str]:
        """Generate a new API key, persist only its SHA-256 hash, and return (record, raw_key)."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("API key name must not be empty")
        if rpm_limit < 0:
            raise ValueError("rpm_limit must be >= 0 (0 means unlimited)")

        key_id = f"key_{secrets.token_hex(6)}"
        raw_key = f"sk-agy-{secrets.token_urlsafe(24)}"
        key_hash = _hash_key(raw_key)
        key_prefix = _mask_key(raw_key)
        created_at = _utc_now_iso()
        models_json = json.dumps(allowed_models or [])
        accounts_json = json.dumps(allowed_accounts or [])

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (
                    id, name, key_hash, key_prefix, rpm_limit, allowed_models,
                    allowed_accounts, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    key_id,
                    clean_name,
                    key_hash,
                    key_prefix,
                    rpm_limit,
                    models_json,
                    accounts_json,
                    created_at,
                ),
            )
            row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        assert row is not None
        return self._row_to_record(row), raw_key

    def list_keys(self) -> List[ApiKeyRecord]:
        """Return all issued API keys ordered newest-first."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_key(self, key_id: str) -> Optional[ApiKeyRecord]:
        """Fetch a single API key record by its ID."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def update_key(
        self,
        key_id: str,
        *,
        name: Optional[str] = None,
        rpm_limit: Optional[int] = None,
        allowed_models: Optional[List[str]] = None,
        allowed_accounts: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[ApiKeyRecord]:
        """Update mutable fields of an API key and return the updated record (or None if missing)."""
        updates: List[str] = []
        params: List[object] = []

        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise ValueError("API key name must not be empty")
            updates.append("name = ?")
            params.append(clean_name)
        if rpm_limit is not None:
            if rpm_limit < 0:
                raise ValueError("rpm_limit must be >= 0")
            updates.append("rpm_limit = ?")
            params.append(int(rpm_limit))
        if allowed_models is not None:
            updates.append("allowed_models = ?")
            params.append(json.dumps(allowed_models))
        if allowed_accounts is not None:
            updates.append("allowed_accounts = ?")
            params.append(json.dumps(allowed_accounts))
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(1 if enabled else 0)

        with self._lock, self._connect() as conn:
            if updates:
                params.append(key_id)
                cur = conn.execute(
                    f"UPDATE api_keys SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
                if cur.rowcount == 0:
                    return None
            row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
            return self._row_to_record(row) if row else None

    def set_enabled(self, key_id: str, enabled: bool) -> bool:
        """Enable or disable an existing API key."""
        return self.update_key(key_id, enabled=enabled) is not None

    def delete_key(self, key_id: str) -> bool:
        """Permanently delete an API key record."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            self._rpm_windows.pop(key_id, None)
            return cur.rowcount > 0

    def authenticate(
        self,
        raw_key: Optional[str],
        model: Optional[str] = None,
        check_rate_limit: bool = True,
    ) -> ApiKeyRecord:
        """Validate key existence, enabled status, optional model scope, and RPM limit."""
        if not raw_key or not raw_key.strip():
            raise KeyAuthError("Missing API key", status_code=401)

        digest = _hash_key(raw_key.strip())
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM api_keys WHERE key_hash = ?", (digest,)
                ).fetchone()
            if row is None:
                raise KeyAuthError("Invalid API key", status_code=401)
            # Constant-time equality verification on digest for defense in depth.
            if not hmac.compare_digest(row["key_hash"], digest):
                raise KeyAuthError("Invalid API key", status_code=401)

            record = self._row_to_record(row)
            if not record.enabled:
                raise KeyAuthError("API key is disabled", status_code=401)

            if model and record.allowed_models and model not in record.allowed_models:
                raise KeyAuthError(
                    f"API key is not authorized for model '{model}'",
                    status_code=403,
                )

            if check_rate_limit and record.rpm_limit > 0:
                now = time.monotonic()
                window = self._rpm_windows[record.id]
                cutoff = now - 60.0
                while window and window[0] <= cutoff:
                    window.popleft()
                if len(window) >= record.rpm_limit:
                    raise KeyAuthError(
                        f"Rate limit exceeded ({record.rpm_limit} requests/minute)",
                        status_code=429,
                    )
                window.append(now)

            return record

    def record_usage(
        self,
        key_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Increment request and token accounting for an API key."""
        now_iso = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE api_keys
                SET last_used_at = ?,
                    total_requests = total_requests + 1,
                    total_input_tokens = total_input_tokens + ?,
                    total_output_tokens = total_output_tokens + ?
                WHERE id = ?
                """,
                (now_iso, max(0, int(input_tokens)), max(0, int(output_tokens)), key_id),
            )
