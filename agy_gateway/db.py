"""Unified dual-backend database abstraction (PostgreSQL + SQLite) and migration helpers."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Set


def is_postgres_url(url: str) -> bool:
    """Return True if `url` is a PostgreSQL connection URI."""
    cleaned = (url or "").strip().lower()
    return cleaned.startswith(("postgresql://", "postgres://"))


def _translate_qmarks_to_pg(sql: str) -> str:
    """Translate SQLite `?` parameter placeholders to psycopg `%s` outside string literals."""
    out: list[str] = []
    in_single = False
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            out.append(ch)
            if in_single and i + 1 < n and sql[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            in_single = not in_single
        elif not in_single and ch == "?":
            out.append("%s")
        elif ch == "%":
            out.append("%%")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_POOLS_LOCK = threading.Lock()
_PG_POOLS: Dict[str, Any] = {}


def _get_pg_pool(database_url: str) -> Any:
    """Return (or lazily create) a shared psycopg ConnectionPool for `database_url`."""
    with _POOLS_LOCK:
        pool = _PG_POOLS.get(database_url)
        if pool is not None:
            return pool

        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        conn_kwargs: Dict[str, Any] = {
            "row_factory": dict_row,
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 3,
        }
        if "sslmode=" not in database_url.lower():
            conn_kwargs["sslmode"] = "prefer"

        pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=12,
            open=True,
            check=ConnectionPool.check_connection,
            kwargs=conn_kwargs,
        )
        _PG_POOLS[database_url] = pool
        return pool


def close_all_pools() -> None:
    """Close all open PostgreSQL connection pools (called on graceful shutdown)."""
    with _POOLS_LOCK:
        pools = list(_PG_POOLS.values())
        _PG_POOLS.clear()
    for pool in pools:
        with contextlib.suppress(Exception):
            pool.close()


class DatabaseConnection:
    """Unified thin wrapper around either a sqlite3.Connection or psycopg.Connection."""

    def __init__(self, raw_conn: Any, *, is_postgres: bool) -> None:
        self._conn = raw_conn
        self.is_postgres = is_postgres

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        if self.is_postgres:
            pg_sql = _translate_qmarks_to_pg(sql)
            return self._conn.execute(pg_sql, tuple(params) if params else None)
        return self._conn.execute(sql, tuple(params))

    def init_pragmas(self) -> None:
        if not self.is_postgres:
            self._conn.execute("PRAGMA journal_mode=WAL;")

    def column_names(self, table_name: str) -> Set[str]:
        if self.is_postgres:
            rows = self._conn.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                """,
                (table_name,),
            ).fetchall()
            return {str(r["column_name"]) for r in rows}
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(r["name"]) for r in rows}


class DatabaseBackend:
    """Selects PostgreSQL (pooled) when `database_url` is configured, else local SQLite."""

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        database_url: str = "",
    ) -> None:
        self.database_url = (database_url or "").strip()
        self.is_postgres = is_postgres_url(self.database_url)
        self.db_path = Path(db_path) if db_path is not None else Path("./data/gateway.db")
        if not self.is_postgres:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def backend_name(self) -> str:
        return "postgresql" if self.is_postgres else "sqlite"

    @contextlib.contextmanager
    def connect(self) -> Iterator[DatabaseConnection]:
        if self.is_postgres:
            pool = _get_pg_pool(self.database_url)
            with pool.connection() as raw_conn:
                wrapped = DatabaseConnection(raw_conn, is_postgres=True)
                try:
                    yield wrapped
                    raw_conn.commit()
                except Exception:
                    raw_conn.rollback()
                    raise
            return

        raw_conn = sqlite3.connect(
            self.db_path,
            timeout=15.0,
            check_same_thread=False,
        )
        raw_conn.row_factory = sqlite3.Row
        raw_conn.execute("PRAGMA busy_timeout=15000;")
        raw_conn.execute("PRAGMA synchronous=NORMAL;")
        wrapped = DatabaseConnection(raw_conn, is_postgres=False)
        try:
            yield wrapped
            raw_conn.commit()
        except Exception:
            raw_conn.rollback()
            raise
        finally:
            raw_conn.close()


def migrate_sqlite_to_postgres(
    sqlite_path: Path | str,
    database_url: str,
    accounts_root: Path | str = "./data/accounts",
) -> Dict[str, Dict[str, int]]:
    """Idempotently migrate all gateway tables from a SQLite file into PostgreSQL."""
    src_path = Path(sqlite_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source SQLite database not found: {src_path}")
    if not is_postgres_url(database_url):
        raise ValueError("Target database_url must start with postgresql:// or postgres://")

    # Initialize target PostgreSQL schema across all stores first
    from agy_gateway.accounts import AccountStore
    from agy_gateway.context_manager import ContextCheckpointStore
    from agy_gateway.keys import KeyStore

    KeyStore(db_path=src_path, database_url=database_url)
    AccountStore(db_path=src_path, accounts_root=accounts_root, database_url=database_url)
    ContextCheckpointStore(db_path=src_path, database_url=database_url)

    src_conn = sqlite3.connect(src_path)
    src_conn.row_factory = sqlite3.Row
    pg = DatabaseBackend(database_url=database_url)
    report: Dict[str, Dict[str, int]] = {}

    try:
        src_tables = {
            row["name"]
            for row in src_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        with pg.connect() as dst:
            # 1. accounts
            if "accounts" in src_tables:
                rows = src_conn.execute("SELECT * FROM accounts").fetchall()
                migrated = 0
                for r in rows:
                    keys = set(r.keys())
                    dst.execute(
                        """
                        INSERT INTO accounts (
                            id, name, email, home_dir, enabled, priority, max_concurrency,
                            status, cooldown_until, last_error, created_at, last_used_at,
                            total_requests, total_errors, total_input_tokens, total_output_tokens,
                            quota_json, quota_updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            email = EXCLUDED.email,
                            home_dir = EXCLUDED.home_dir,
                            enabled = EXCLUDED.enabled,
                            priority = EXCLUDED.priority,
                            max_concurrency = EXCLUDED.max_concurrency,
                            status = EXCLUDED.status,
                            cooldown_until = EXCLUDED.cooldown_until,
                            last_error = EXCLUDED.last_error,
                            created_at = EXCLUDED.created_at,
                            last_used_at = EXCLUDED.last_used_at,
                            total_requests = EXCLUDED.total_requests,
                            total_errors = EXCLUDED.total_errors,
                            total_input_tokens = EXCLUDED.total_input_tokens,
                            total_output_tokens = EXCLUDED.total_output_tokens,
                            quota_json = EXCLUDED.quota_json,
                            quota_updated_at = EXCLUDED.quota_updated_at
                        """,
                        (
                            r["id"],
                            r["name"],
                            r["email"],
                            r["home_dir"],
                            int(r["enabled"]),
                            int(r["priority"]),
                            int(r["max_concurrency"]),
                            r["status"],
                            float(r["cooldown_until"] or 0.0),
                            r["last_error"],
                            r["created_at"],
                            r["last_used_at"],
                            int(r["total_requests"] or 0),
                            int(r["total_errors"] or 0),
                            int(r["total_input_tokens"] or 0),
                            int(r["total_output_tokens"] or 0),
                            r["quota_json"] if "quota_json" in keys else None,
                            r["quota_updated_at"] if "quota_updated_at" in keys else None,
                        ),
                    )
                    migrated += 1
                cnt = int(dst.execute("SELECT COUNT(*) AS c FROM accounts").fetchone()["c"])
                report["accounts"] = {"source": len(rows), "migrated": migrated, "target_total": cnt}

            # 2. api_keys
            if "api_keys" in src_tables:
                rows = src_conn.execute("SELECT * FROM api_keys").fetchall()
                migrated = 0
                for r in rows:
                    keys = set(r.keys())
                    dst.execute(
                        """
                        INSERT INTO api_keys (
                            id, name, key_hash, key_prefix, rpm_limit, allowed_models,
                            allowed_accounts, enabled, created_at, last_used_at,
                            total_requests, total_input_tokens, total_output_tokens
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            key_hash = EXCLUDED.key_hash,
                            key_prefix = EXCLUDED.key_prefix,
                            rpm_limit = EXCLUDED.rpm_limit,
                            allowed_models = EXCLUDED.allowed_models,
                            allowed_accounts = EXCLUDED.allowed_accounts,
                            enabled = EXCLUDED.enabled,
                            created_at = EXCLUDED.created_at,
                            last_used_at = EXCLUDED.last_used_at,
                            total_requests = EXCLUDED.total_requests,
                            total_input_tokens = EXCLUDED.total_input_tokens,
                            total_output_tokens = EXCLUDED.total_output_tokens
                        """,
                        (
                            r["id"],
                            r["name"],
                            r["key_hash"],
                            r["key_prefix"],
                            int(r["rpm_limit"]),
                            r["allowed_models"],
                            r["allowed_accounts"] if "allowed_accounts" in keys else "[]",
                            int(r["enabled"]),
                            r["created_at"],
                            r["last_used_at"],
                            int(r["total_requests"] or 0),
                            int(r["total_input_tokens"] or 0),
                            int(r["total_output_tokens"] or 0),
                        ),
                    )
                    migrated += 1
                cnt = int(dst.execute("SELECT COUNT(*) AS c FROM api_keys").fetchone()["c"])
                report["api_keys"] = {"source": len(rows), "migrated": migrated, "target_total": cnt}

            # 3. gateway_settings
            if "gateway_settings" in src_tables:
                rows = src_conn.execute("SELECT * FROM gateway_settings").fetchall()
                migrated = 0
                for r in rows:
                    dst.execute(
                        """
                        INSERT INTO gateway_settings (key, value) VALUES (?, ?)
                        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                        """,
                        (r["key"], r["value"]),
                    )
                    migrated += 1
                cnt = int(dst.execute("SELECT COUNT(*) AS c FROM gateway_settings").fetchone()["c"])
                report["gateway_settings"] = {
                    "source": len(rows),
                    "migrated": migrated,
                    "target_total": cnt,
                }

            # 4. context_checkpoints
            if "context_checkpoints" in src_tables:
                rows = src_conn.execute("SELECT * FROM context_checkpoints").fetchall()
                migrated = 0
                for r in rows:
                    dst.execute(
                        """
                        INSERT INTO context_checkpoints (
                            session_fingerprint, prefix_hash, summary_text, mode,
                            source_messages, source_bytes, created_at, last_used_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (session_fingerprint, prefix_hash) DO UPDATE SET
                            summary_text = EXCLUDED.summary_text,
                            mode = EXCLUDED.mode,
                            source_messages = EXCLUDED.source_messages,
                            source_bytes = EXCLUDED.source_bytes,
                            created_at = EXCLUDED.created_at,
                            last_used_at = EXCLUDED.last_used_at
                        """,
                        (
                            r["session_fingerprint"],
                            r["prefix_hash"],
                            r["summary_text"],
                            r["mode"],
                            int(r["source_messages"] or 0),
                            int(r["source_bytes"] or 0),
                            float(r["created_at"]),
                            float(r["last_used_at"]),
                        ),
                    )
                    migrated += 1
                cnt = int(
                    dst.execute("SELECT COUNT(*) AS c FROM context_checkpoints").fetchone()["c"]
                )
                report["context_checkpoints"] = {
                    "source": len(rows),
                    "migrated": migrated,
                    "target_total": cnt,
                }
    finally:
        src_conn.close()

    return report
