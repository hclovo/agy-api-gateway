"""Unit tests for dual-backend DB helpers and portable SQLite/PostgreSQL operations."""

from __future__ import annotations

from pathlib import Path

from agy_gateway.context_manager import ContextCheckpointStore
from agy_gateway.db import DatabaseBackend, _translate_qmarks_to_pg, is_postgres_url


def test_postgres_url_detection_and_placeholder_translation() -> None:
    assert is_postgres_url("postgresql://user:pass@1.2.3.4:5432/agy_gateway") is True
    assert is_postgres_url("postgres://user:pass@host/db") is True
    assert is_postgres_url("") is False
    assert is_postgres_url("./data/gateway.db") is False

    sql = "SELECT * FROM api_keys WHERE id = ? AND name = 'what?' AND pct = '100%' AND rpm > ?"
    translated = _translate_qmarks_to_pg(sql)
    assert translated == (
        "SELECT * FROM api_keys WHERE id = %s AND name = 'what?' AND pct = '100%%' AND rpm > %s"
    )


def test_sqlite_backend_and_portable_checkpoint_lru_eviction(tmp_path: Path) -> None:
    db_path = tmp_path / "gw.db"
    backend = DatabaseBackend(db_path=db_path)
    assert backend.is_postgres is False
    assert backend.backend_name == "sqlite"

    # Verify portable tuple-IN LRU eviction works cleanly on SQLite
    store = ContextCheckpointStore(db_path=db_path, ttl_hours=24, max_entries=50)
    for i in range(60):
        store.put(
            f"sess_{i // 5}",
            f"hash_{i}",
            f"summary {i}",
            mode="deterministic",
            source_messages=i + 1,
            source_bytes=1024 * (i + 1),
        )
    assert store.count() == 50
    # Oldest entries evicted, newest preserved
    assert store.get("sess_11", "hash_59") == "summary 59"
