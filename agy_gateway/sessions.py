"""Persistent multi-turn `agy --conversation` session cursor store (PostgreSQL + SQLite)."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from agy_gateway.context_manager import (
    ChatMessage,
    NormalizedRequest,
    extract_clean_human_intent,
    parse_tool_calls,
    strip_tool_calls,
)
from agy_gateway.db import DatabaseBackend

_STAGED_MEDIA_RE = re.compile(r"/media_[0-9a-f]{16}/")
_ISO_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
_THINKING_RE = re.compile(r"<(?:thinking|thought)>.*?</(?:thinking|thought)>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class SessionRecord:
    """Persisted mapping from an external client conversation to an `agy --conversation` ID."""

    session_key: str
    session_fingerprint: str
    account_id: str
    agy_conversation_id: str
    model: str
    sys_tools_hash: str
    sent_message_count: int
    sent_prefix_hash: str
    last_assistant_fingerprint: str
    turn_count: int
    total_delta_bytes: int
    created_at: float
    last_active_at: float


def normalize_stable_text(text: str) -> str:
    """Normalize per-request ephemeral staging paths so identical turns hash identically."""
    if not text:
        return ""
    return _STAGED_MEDIA_RE.sub("/media_STAGED/", text)


def compute_messages_prefix_hash(messages: Sequence[ChatMessage]) -> str:
    """Compute a deterministic, staging-path-independent hash of a message prefix."""
    h = hashlib.sha256()
    for msg in messages:
        h.update(msg.role.encode("utf-8"))
        h.update(b"\x00")
        h.update((msg.tool_call_id or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(normalize_stable_text(msg.content).encode("utf-8", errors="ignore"))
        h.update(b"\x01")
        if msg.tool_calls:
            # Normalize tool calls by name + arguments (ignore client-regenerated call IDs)
            norm_calls = []
            for tc in msg.tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                raw_args = fn.get("arguments") if "arguments" in fn else tc.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        pass
                norm_calls.append({"name": fn.get("name") or tc.get("name"), "arguments": raw_args})
            h.update(json.dumps(norm_calls, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\x02")
        for tr in msg.tool_results:
            h.update((tr.name or "").encode("utf-8"))
            h.update(b":")
            h.update(normalize_stable_text(tr.content).encode("utf-8", errors="ignore"))
            h.update(b"\x03")
    return h.hexdigest()[:32]


def compute_sys_tools_hash(req: NormalizedRequest) -> str:
    """Compute a drift-tolerant hash of system prompt + tool definitions.

    Strips volatile ISO timestamps so harnesses injecting current time each turn do not
    break session reuse, while real toolset or instruction changes cleanly reset the session.
    """
    sys_clean = _ISO_TS_RE.sub("<TS>", (req.system_prompt or "").strip())
    sys_head = sys_clean[:4096]
    tool_sigs: List[str] = []
    for raw_tool in req.tools:
        if not isinstance(raw_tool, dict):
            continue
        fn = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else raw_tool
        name = str(fn.get("name") or "")
        params = fn.get("parameters") or fn.get("input_schema") or {}
        props = sorted((params.get("properties") or {}).keys()) if isinstance(params, dict) else []
        tool_sigs.append(f"{name}({','.join(props)})")
    raw = f"sys:{sys_head}\nlen:{len(sys_clean) // 512}\ntools:{'|'.join(tool_sigs)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def compute_session_key(
    *,
    api_key_id: str,
    session_fingerprint: str,
    target_model: str,
) -> str:
    """Scope session reuse strictly to (api_key_id, session_fingerprint, target_model)."""
    raw = f"key:{api_key_id or 'anon'}\nfp:{session_fingerprint}\nmodel:{target_model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def compute_assistant_fingerprint_from_output(raw_text: str, *, has_tools: bool) -> str:
    """Compute a cross-protocol fingerprint of the assistant output produced by `agy`."""
    calls = parse_tool_calls(raw_text) if has_tools else []
    if calls:
        names = [str(c.get("function", {}).get("name") or "") for c in calls]
        leading = " ".join(strip_tool_calls(raw_text).split())[:256]
        return f"tools:{','.join(names)}:{hashlib.sha256(leading.encode('utf-8')).hexdigest()[:12]}"
    cleaned = " ".join(_THINKING_RE.sub("", raw_text).split())
    return f"text:{hashlib.sha256(cleaned.encode('utf-8')).hexdigest()[:20]}"


def compute_assistant_fingerprint_from_message(msg: ChatMessage) -> str:
    """Compute the matching cross-protocol fingerprint from a client-echoed assistant message."""
    if msg.tool_calls:
        names: List[str] = []
        for tc in msg.tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            names.append(str(fn.get("name") or tc.get("name") or ""))
        leading = " ".join(_THINKING_RE.sub("", msg.content or "").split())[:256]
        return f"tools:{','.join(names)}:{hashlib.sha256(leading.encode('utf-8')).hexdigest()[:12]}"
    # Some clients echo tool calls inline inside content text as <tool_call>
    inline_calls = parse_tool_calls(msg.content or "")
    if inline_calls:
        names = [str(c.get("function", {}).get("name") or "") for c in inline_calls]
        leading = " ".join(strip_tool_calls(msg.content or "").split())[:256]
        return f"tools:{','.join(names)}:{hashlib.sha256(leading.encode('utf-8')).hexdigest()[:12]}"
    cleaned = " ".join(_THINKING_RE.sub("", msg.content or "").split())
    return f"text:{hashlib.sha256(cleaned.encode('utf-8')).hexdigest()[:20]}"


def assistant_fingerprints_compatible(expected_fp: str, echoed_msg: ChatMessage) -> bool:
    """Return True if a client-echoed assistant turn matches what `agy` previously produced."""
    if echoed_msg.role != "assistant":
        return False
    if not expected_fp:
        return True
    actual_fp = compute_assistant_fingerprint_from_message(echoed_msg)
    if actual_fp == expected_fp:
        return True
    # Allow minor leading-text formatting differences when tool call sequence matches identically
    if expected_fp.startswith("tools:") and actual_fp.startswith("tools:"):
        exp_names = expected_fp.split(":", 2)[1]
        act_names = actual_fp.split(":", 2)[1]
        if exp_names and exp_names == act_names:
            return True
    return False


def agy_conversation_db_exists(account_home: Path, agy_conversation_id: str) -> bool:
    """Verify the `agy` conversation SQLite DB exists on disk before calling `--conversation`."""
    if not agy_conversation_id:
        return False
    db_file = (
        Path(account_home)
        / ".gemini"
        / "antigravity-cli"
        / "conversations"
        / f"{agy_conversation_id}.db"
    )
    return db_file.is_file()


class SessionStore:
    """Thread-safe PostgreSQL/SQLite store for active `agy --conversation` reuse cursors."""

    def __init__(
        self,
        db_path: Path | str = "./data/gateway.db",
        ttl_hours: int = 24,
        max_entries: int = 1000,
        database_url: str = "",
    ) -> None:
        self.db_path = Path(db_path)
        self.database_url = database_url
        self.ttl_seconds = max(300, int(ttl_hours * 3600))
        self.max_entries = max(50, int(max_entries))
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
                CREATE TABLE IF NOT EXISTS gateway_sessions (
                    session_key TEXT PRIMARY KEY,
                    session_fingerprint TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    agy_conversation_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    sys_tools_hash TEXT NOT NULL,
                    sent_message_count INTEGER NOT NULL DEFAULT 0,
                    sent_prefix_hash TEXT NOT NULL,
                    last_assistant_fingerprint TEXT NOT NULL DEFAULT '',
                    turn_count INTEGER NOT NULL DEFAULT 1,
                    total_delta_bytes BIGINT NOT NULL DEFAULT 0,
                    created_at DOUBLE PRECISION NOT NULL,
                    last_active_at DOUBLE PRECISION NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_gw_sessions_last_active
                ON gateway_sessions (last_active_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_gw_sessions_conv_id
                ON gateway_sessions (agy_conversation_id)
                """
            )

    @staticmethod
    def _row_to_record(row: Any) -> SessionRecord:
        return SessionRecord(
            session_key=str(row["session_key"]),
            session_fingerprint=str(row["session_fingerprint"]),
            account_id=str(row["account_id"]),
            agy_conversation_id=str(row["agy_conversation_id"]),
            model=str(row["model"]),
            sys_tools_hash=str(row["sys_tools_hash"]),
            sent_message_count=int(row["sent_message_count"]),
            sent_prefix_hash=str(row["sent_prefix_hash"]),
            last_assistant_fingerprint=str(row["last_assistant_fingerprint"] or ""),
            turn_count=int(row["turn_count"]),
            total_delta_bytes=int(row["total_delta_bytes"]),
            created_at=float(row["created_at"]),
            last_active_at=float(row["last_active_at"]),
        )

    def get(self, session_key: str) -> Optional[SessionRecord]:
        cutoff = time.time() - self.ttl_seconds
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM gateway_sessions
                WHERE session_key = ? AND last_active_at >= ?
                """,
                (session_key, cutoff),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def upsert(
        self,
        *,
        session_key: str,
        session_fingerprint: str,
        account_id: str,
        agy_conversation_id: str,
        model: str,
        sys_tools_hash: str,
        sent_message_count: int,
        sent_prefix_hash: str,
        last_assistant_fingerprint: str,
        delta_bytes: int = 0,
        increment_turn: bool = False,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT turn_count, total_delta_bytes, created_at FROM gateway_sessions WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if existing is not None and increment_turn:
                turn_count = int(existing["turn_count"]) + 1
                total_bytes = int(existing["total_delta_bytes"]) + max(0, int(delta_bytes))
                created_at = float(existing["created_at"])
            else:
                turn_count = 1
                total_bytes = max(0, int(delta_bytes))
                created_at = now

            conn.execute(
                """
                INSERT INTO gateway_sessions (
                    session_key, session_fingerprint, account_id, agy_conversation_id,
                    model, sys_tools_hash, sent_message_count, sent_prefix_hash,
                    last_assistant_fingerprint, turn_count, total_delta_bytes,
                    created_at, last_active_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_key) DO UPDATE SET
                    session_fingerprint = EXCLUDED.session_fingerprint,
                    account_id = EXCLUDED.account_id,
                    agy_conversation_id = EXCLUDED.agy_conversation_id,
                    model = EXCLUDED.model,
                    sys_tools_hash = EXCLUDED.sys_tools_hash,
                    sent_message_count = EXCLUDED.sent_message_count,
                    sent_prefix_hash = EXCLUDED.sent_prefix_hash,
                    last_assistant_fingerprint = EXCLUDED.last_assistant_fingerprint,
                    turn_count = EXCLUDED.turn_count,
                    total_delta_bytes = EXCLUDED.total_delta_bytes,
                    created_at = EXCLUDED.created_at,
                    last_active_at = EXCLUDED.last_active_at
                """,
                (
                    session_key,
                    session_fingerprint,
                    account_id,
                    agy_conversation_id,
                    model,
                    sys_tools_hash,
                    int(sent_message_count),
                    sent_prefix_hash,
                    last_assistant_fingerprint,
                    turn_count,
                    total_bytes,
                    created_at,
                    now,
                ),
            )

            count_row = conn.execute("SELECT COUNT(*) AS c FROM gateway_sessions").fetchone()
            total = int(count_row["c"]) if count_row else 0
            if total > self.max_entries:
                excess = total - self.max_entries
                conn.execute(
                    """
                    DELETE FROM gateway_sessions
                    WHERE session_key IN (
                        SELECT session_key FROM gateway_sessions
                        ORDER BY last_active_at ASC
                        LIMIT ?
                    )
                    """,
                    (excess,),
                )

    def delete(self, session_key: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM gateway_sessions WHERE session_key = ?", (session_key,)
            )
            return int(cur.rowcount or 0) > 0

    def list_active_conversation_ids(self) -> Set[str]:
        """Return all `agy_conversation_id`s currently protected from housekeeping pruning."""
        cutoff = time.time() - self.ttl_seconds
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT agy_conversation_id FROM gateway_sessions WHERE last_active_at >= ?",
                (cutoff,),
            ).fetchall()
            return {str(r["agy_conversation_id"]) for r in rows if r["agy_conversation_id"]}

    def prune_expired(self) -> int:
        cutoff = time.time() - self.ttl_seconds
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM gateway_sessions WHERE last_active_at < ?", (cutoff,)
            )
            return int(cur.rowcount or 0)

    def clear(self) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM gateway_sessions")
            return int(cur.rowcount or 0)

    def count(self) -> int:
        cutoff = time.time() - self.ttl_seconds
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM gateway_sessions WHERE last_active_at >= ?",
                (cutoff,),
            ).fetchone()
            return int(row["c"]) if row else 0
