"""Multi-account store, isolated HOME management, and interactive `agy acp --login` coordinator."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agy_gateway.db import DatabaseBackend


VALID_ROUTING_STRATEGIES = ("least_busy", "round_robin", "priority")
_OAUTH_URL_RE = re.compile(r"https://accounts\.google\.com/o/oauth2/auth\S+")
_EMAIL_LOG_RE = re.compile(r"applyAuthResult:\s*email=([^\s,]+)")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class AccountRecord:
    """Persisted metadata and usage stats for one Antigravity CLI account profile."""

    id: str
    name: str
    email: Optional[str]
    home_dir: str  # Empty string means inherit container/process default $HOME
    enabled: bool
    priority: int
    max_concurrency: int
    status: str  # unknown | logged_in | needs_login | cooldown | error
    cooldown_until: float
    last_error: Optional[str]
    created_at: str
    last_used_at: Optional[str]
    total_requests: int
    total_errors: int
    total_input_tokens: int
    total_output_tokens: int
    quota: Optional[Dict[str, Any]] = None
    quota_updated_at: Optional[str] = None

    def resolved_home(self) -> Path:
        if self.home_dir:
            return Path(self.home_dir)
        return Path.home()

    def build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.home_dir:
            home = Path(self.home_dir)
            (home / ".config").mkdir(parents=True, exist_ok=True)
            (home / ".cache").mkdir(parents=True, exist_ok=True)
            (home / ".local" / "share").mkdir(parents=True, exist_ok=True)
            env["HOME"] = str(home)
            env["XDG_CONFIG_HOME"] = str(home / ".config")
            env["XDG_CACHE_HOME"] = str(home / ".cache")
            env["XDG_DATA_HOME"] = str(home / ".local" / "share")
        return env


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def detect_email_from_logs(home_dir: Path) -> Optional[str]:
    """Read the most recent `agy` CLI log under home_dir to extract signed-in account email."""
    log_dir = home_dir / ".gemini" / "antigravity-cli" / "log"
    if not log_dir.is_dir():
        return None
    logs = sorted(log_dir.glob("cli-*.log"), key=_safe_mtime, reverse=True)
    for log_path in logs[:5]:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        matches = _EMAIL_LOG_RE.findall(text)
        if matches:
            return matches[-1].strip()
    return None


class AccountStore:
    """Thread-safe PostgreSQL/SQLite store for Antigravity accounts and gateway routing settings."""

    def __init__(
        self,
        db_path: Path | str,
        accounts_root: Path | str,
        database_url: str = "",
    ) -> None:
        self.db_path = Path(db_path)
        self.database_url = database_url
        self.accounts_root = Path(accounts_root)
        self.accounts_root.mkdir(parents=True, exist_ok=True)
        self._db = DatabaseBackend(db_path=self.db_path, database_url=database_url)
        self._lock = threading.Lock()
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
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    home_dir TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    priority INTEGER NOT NULL DEFAULT 100,
                    max_concurrency INTEGER NOT NULL DEFAULT 6,
                    status TEXT NOT NULL DEFAULT 'unknown',
                    cooldown_until DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    total_requests BIGINT NOT NULL DEFAULT 0,
                    total_errors BIGINT NOT NULL DEFAULT 0,
                    total_input_tokens BIGINT NOT NULL DEFAULT 0,
                    total_output_tokens BIGINT NOT NULL DEFAULT 0,
                    quota_json TEXT,
                    quota_updated_at TEXT
                )
                """
            )
            cols = conn.column_names("accounts")
            if "quota_json" not in cols:
                conn.execute("ALTER TABLE accounts ADD COLUMN quota_json TEXT")
            if "quota_updated_at" not in cols:
                conn.execute("ALTER TABLE accounts ADD COLUMN quota_updated_at TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gateway_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            # Ensure default account exists and inherits container/host default $HOME
            existing = conn.execute("SELECT id FROM accounts WHERE id = 'default'").fetchone()
            if existing is None:
                detected_email = detect_email_from_logs(Path.home())
                initial_status = "logged_in" if detected_email else "unknown"
                conn.execute(
                    """
                    INSERT INTO accounts (
                        id, name, email, home_dir, enabled, priority, max_concurrency,
                        status, created_at
                    ) VALUES ('default', '默认主账号', ?, '', 1, 100, 12, ?, ?)
                    """,
                    (detected_email, initial_status, _utc_now_iso()),
                )

    @staticmethod
    def _row_to_record(row: Any) -> AccountRecord:
        keys = row.keys()
        quota_obj: Optional[Dict[str, Any]] = None
        if "quota_json" in keys and row["quota_json"]:
            with contextlib.suppress(Exception):
                parsed = json.loads(row["quota_json"])
                if isinstance(parsed, dict):
                    quota_obj = parsed
        quota_updated_at = row["quota_updated_at"] if "quota_updated_at" in keys else None
        return AccountRecord(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            home_dir=row["home_dir"],
            enabled=bool(row["enabled"]),
            priority=int(row["priority"]),
            max_concurrency=int(row["max_concurrency"]),
            status=row["status"],
            cooldown_until=float(row["cooldown_until"] or 0.0),
            last_error=row["last_error"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            total_requests=int(row["total_requests"]),
            total_errors=int(row["total_errors"]),
            total_input_tokens=int(row["total_input_tokens"]),
            total_output_tokens=int(row["total_output_tokens"]),
            quota=quota_obj,
            quota_updated_at=quota_updated_at,
        )

    def get_routing_strategy(self, default: str = "least_busy") -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM gateway_settings WHERE key = 'routing_strategy'"
            ).fetchone()
        val = row["value"] if row else default
        return val if val in VALID_ROUTING_STRATEGIES else default

    def set_routing_strategy(self, strategy: str) -> str:
        if strategy not in VALID_ROUTING_STRATEGIES:
            raise ValueError(f"Unsupported routing strategy: {strategy}")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO gateway_settings (key, value) VALUES ('routing_strategy', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (strategy,),
            )
        return strategy

    def create_account(
        self,
        name: str,
        priority: int = 100,
        max_concurrency: int = 6,
    ) -> AccountRecord:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Account name must not be empty")
        if max_concurrency < 0:
            raise ValueError("max_concurrency must be >= 0")

        acc_id = f"acc_{secrets.token_hex(6)}"
        home_path = (self.accounts_root / acc_id).resolve()
        home_path.mkdir(parents=True, exist_ok=True)
        try:
            home_path.chmod(0o700)
        except OSError:
            pass

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO accounts (
                    id, name, email, home_dir, enabled, priority, max_concurrency,
                    status, created_at
                ) VALUES (?, ?, NULL, ?, 1, ?, ?, 'needs_login', ?)
                """,
                (acc_id, clean_name, str(home_path), int(priority), int(max_concurrency), _utc_now_iso()),
            )
            row = conn.execute("SELECT * FROM accounts WHERE id = ?", (acc_id,)).fetchone()
        assert row is not None
        return self._row_to_record(row)

    def list_accounts(self) -> List[AccountRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM accounts ORDER BY priority ASC, created_at ASC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_account(self, account_id: str) -> Optional[AccountRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def update_account(
        self,
        account_id: str,
        *,
        name: Optional[str] = None,
        email: Optional[str] = None,
        enabled: Optional[bool] = None,
        priority: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        status: Optional[str] = None,
        cooldown_until: Optional[float] = None,
        last_error: Optional[str] = None,
        clear_error: bool = False,
    ) -> Optional[AccountRecord]:
        updates: List[str] = []
        params: List[object] = []

        if name is not None:
            clean = name.strip()
            if not clean:
                raise ValueError("Account name must not be empty")
            updates.append("name = ?")
            params.append(clean)
        if email is not None:
            updates.append("email = ?")
            params.append(email.strip() or None)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(1 if enabled else 0)
        if priority is not None:
            updates.append("priority = ?")
            params.append(int(priority))
        if max_concurrency is not None:
            if max_concurrency < 0:
                raise ValueError("max_concurrency must be >= 0")
            updates.append("max_concurrency = ?")
            params.append(int(max_concurrency))
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if cooldown_until is not None:
            updates.append("cooldown_until = ?")
            params.append(float(cooldown_until))
        if clear_error:
            updates.append("last_error = NULL")
        elif last_error is not None:
            updates.append("last_error = ?")
            params.append(last_error[:500])

        with self._lock, self._connect() as conn:
            if updates:
                params.append(account_id)
                cur = conn.execute(
                    f"UPDATE accounts SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
                if cur.rowcount == 0:
                    return None
            row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
            return self._row_to_record(row) if row else None

    def update_quota(
        self,
        account_id: str,
        quota: Dict[str, Any],
    ) -> Optional[AccountRecord]:
        now_iso = _utc_now_iso()
        quota_json = json.dumps(quota, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE accounts
                SET quota_json = ?,
                    quota_updated_at = ?
                WHERE id = ?
                """,
                (quota_json, now_iso, account_id),
            )
            if cur.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
            return self._row_to_record(row) if row else None

    def delete_account(self, account_id: str) -> bool:
        if account_id == "default":
            raise ValueError("Cannot delete the built-in default account")
        rec = self.get_account(account_id)
        if rec is None:
            return False
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
        if cur.rowcount > 0 and rec.home_dir:
            home_path = Path(rec.home_dir)
            with contextlib.suppress(OSError):
                if home_path.is_dir() and self.accounts_root.resolve() in home_path.resolve().parents:
                    shutil.rmtree(home_path, ignore_errors=True)
        return cur.rowcount > 0

    def record_success(
        self,
        account_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        detected_email: Optional[str] = None,
    ) -> None:
        now_iso = _utc_now_iso()
        with self._lock, self._connect() as conn:
            if detected_email:
                conn.execute(
                    """
                    UPDATE accounts
                    SET status = 'logged_in',
                        email = COALESCE(?, email),
                        cooldown_until = 0.0,
                        last_error = NULL,
                        last_used_at = ?,
                        total_requests = total_requests + 1,
                        total_input_tokens = total_input_tokens + ?,
                        total_output_tokens = total_output_tokens + ?
                    WHERE id = ?
                    """,
                    (
                        detected_email,
                        now_iso,
                        max(0, int(input_tokens)),
                        max(0, int(output_tokens)),
                        account_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE accounts
                    SET status = 'logged_in',
                        cooldown_until = 0.0,
                        last_error = NULL,
                        last_used_at = ?,
                        total_requests = total_requests + 1,
                        total_input_tokens = total_input_tokens + ?,
                        total_output_tokens = total_output_tokens + ?
                    WHERE id = ?
                    """,
                    (
                        now_iso,
                        max(0, int(input_tokens)),
                        max(0, int(output_tokens)),
                        account_id,
                    ),
                )

    def record_failure(
        self,
        account_id: str,
        error_message: str,
        *,
        needs_login: bool = False,
        cooldown_seconds: float = 0.0,
    ) -> None:
        new_status = "needs_login" if needs_login else ("cooldown" if cooldown_seconds > 0 else "error")
        cooldown_until = (time.time() + cooldown_seconds) if cooldown_seconds > 0 else 0.0
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE accounts
                SET status = ?,
                    cooldown_until = ?,
                    last_error = ?,
                    last_used_at = ?,
                    total_errors = total_errors + 1
                WHERE id = ?
                """,
                (new_status, cooldown_until, error_message[:500], _utc_now_iso(), account_id),
            )


@dataclass
class _PendingLogin:
    session_id: str
    account_id: str
    auth_url: str
    proc: asyncio.subprocess.Process
    created_at: float
    expires_at: float


class OAuthLoginManager:
    """Manages interactive `agy acp --login` subprocesses for browser-based account sign-in."""

    def __init__(self, agy_bin: str, workdir: Path, ttl_seconds: float = 600.0) -> None:
        self.agy_bin = agy_bin
        self.workdir = Path(workdir)
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, _PendingLogin] = {}
        self._lock = asyncio.Lock()

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()

    async def cleanup_expired(self) -> None:
        now = time.time()
        expired: List[_PendingLogin] = []
        async with self._lock:
            for sid, sess in list(self._sessions.items()):
                if sess.expires_at <= now or sess.proc.returncode is not None:
                    expired.append(self._sessions.pop(sid))
        for sess in expired:
            await self._terminate(sess.proc)

    async def start_login(self, account: AccountRecord) -> Tuple[str, str, float]:
        """Start `agy acp --login` for account and return (session_id, auth_url, expires_at)."""
        await self.cleanup_expired()
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Cancel any prior pending session for the same account
        async with self._lock:
            for sid, existing in list(self._sessions.items()):
                if existing.account_id == account.id:
                    self._sessions.pop(sid, None)
                    await self._terminate(existing.proc)

        proc = await asyncio.create_subprocess_exec(
            self.agy_bin,
            "acp",
            "--login",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(self.workdir),
            env=account.build_env(),
        )

        auth_url = ""
        captured_lines: List[str] = []
        deadline = time.monotonic() + 25.0
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                captured_lines.append(line)
            match = _OAUTH_URL_RE.search(line)
            if match:
                auth_url = match.group(0)
                break

        if not auth_url:
            await self._terminate(proc)
            detail = " | ".join(captured_lines[-5:]) or "No OAuth URL received from `agy acp --login`"
            raise RuntimeError(detail)

        now = time.time()
        session_id = f"login_{secrets.token_hex(8)}"
        expires_at = now + self.ttl_seconds
        async with self._lock:
            self._sessions[session_id] = _PendingLogin(
                session_id=session_id,
                account_id=account.id,
                auth_url=auth_url,
                proc=proc,
                created_at=now,
                expires_at=expires_at,
            )
        return session_id, auth_url, expires_at

    async def submit_code(self, session_id: str, code_or_url: str) -> Tuple[str, str]:
        """Submit pasted OAuth authorization code/URL to the waiting `agy acp --login` process."""
        clean_code = code_or_url.strip()
        if not clean_code:
            raise ValueError("Authorization code or callback URL must not be empty")

        async with self._lock:
            sess = self._sessions.pop(session_id, None)
        if sess is None:
            raise KeyError("Login session expired or not found; please generate a new login link.")

        proc = sess.proc
        try:
            assert proc.stdin is not None
            proc.stdin.write((clean_code + "\n").encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=35.0)
            output = stdout_bytes.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                raise RuntimeError(output or f"agy acp --login exited with code {proc.returncode}")
            return sess.account_id, output
        finally:
            await self._terminate(proc)

    async def cancel_login(self, session_id: str) -> bool:
        async with self._lock:
            sess = self._sessions.pop(session_id, None)
        if sess is None:
            return False
        await self._terminate(sess.proc)
        return True
