"""Tests for native `agy --conversation` multi-turn session reuse and production bug fixes."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from agy_gateway.accounts import AccountStore
from agy_gateway.agy_client import (
    AgyChunk,
    AgyModelInfo,
    AgyRunner,
    AgySessionMismatchError,
    UsageInfo,
)
from agy_gateway.app import create_app
from agy_gateway.config import Settings
from agy_gateway.context_manager import (
    ChatMessage,
    ContextCheckpointStore,
    ContextManager,
    NormalizedRequest,
    compute_prefix_hash,
    compute_session_fingerprint,
    extract_clean_human_intent,
)
from agy_gateway.dispatcher import AgyDispatcher
from agy_gateway.housekeeping import prune_agy_artifacts
from agy_gateway.keys import KeyStore


class ResumableFakeRunner:
    """Fake runner supporting `--conversation` multi-turn resume and optional mismatch simulation."""

    supports_conversation_resume = True

    def __init__(self, fail_conv_ids: Optional[set[str]] = None) -> None:
        self.fail_conv_ids = set(fail_conv_ids or set())
        self.calls: List[Tuple[Optional[str], str]] = []
        self._counter = 0
        self.next_reply: Optional[str] = None

    async def list_models(self, force_refresh: bool = False, env=None, cache_key: str = "default"):
        return [AgyModelInfo("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)")]

    async def probe_account(self, env, cache_key=None):
        return True, await self.list_models(), ""

    async def stream_prompt(
        self,
        prompt: str,
        model: str,
        effort: Optional[str] = None,
        timeout_seconds: float = 180.0,
        env=None,
        cwd=None,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[AgyChunk]:
        self.calls.append((conversation_id, prompt))
        if conversation_id and conversation_id in self.fail_conv_ids:
            raise AgySessionMismatchError(f"Requested conversation '{conversation_id}' not found")
        if conversation_id:
            cid = conversation_id
        else:
            self._counter += 1
            cid = f"agy-conv-{self._counter}"
        reply = self.next_reply if self.next_reply is not None else f"reply:{cid}:turn{len(self.calls)}"
        self.next_reply = None
        yield AgyChunk(text_delta=reply, done=False, conversation_id=cid)
        yield AgyChunk(
            done=True,
            usage=UsageInfo(input_tokens=120, output_tokens=10, cache_read_tokens=80 if conversation_id else 0),
            conversation_id=cid,
            finish_reason="stop",
        )


def _make_reuse_client(tmp_path: Path, runner: ResumableFakeRunner):
    settings = Settings(
        workdir=tmp_path / "work",
        db_path=tmp_path / "gw.db",
        accounts_dir=tmp_path / "accounts",
        admin_key="adm-reuse",
        session_reuse_enabled=True,
    )
    ks = KeyStore(settings.db_path)
    acc_store = AccountStore(settings.db_path, settings.accounts_dir)
    disp = AgyDispatcher(settings=settings, runner=runner, account_store=acc_store)  # type: ignore[arg-type]
    app = create_app(settings=settings, key_store=ks, dispatcher=disp, account_store=acc_store)
    return TestClient(app), ks, disp, acc_store


def test_multi_turn_session_reuse_sends_only_delta_and_tool_loops(tmp_path: Path) -> None:
    runner = ResumableFakeRunner()
    client, ks, disp, _ = _make_reuse_client(tmp_path, runner)
    _, key_a = ks.create_key(name="client-a", rpm_limit=60)
    _, key_b = ks.create_key(name="client-b", rpm_limit=60)
    hdr_a = {"Authorization": f"Bearer {key_a}"}
    hdr_b = {"Authorization": f"Bearer {key_b}"}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]

    # Turn 1 (new session) -> creates agy-conv-1 with full prompt
    runner.next_reply = "I remembered BLUE_DOLPHIN_42."
    r1 = client.post(
        "/v1/chat/completions",
        headers=hdr_a,
        json={
            "model": "gemini-3.8-flash-high",
            "tools": tools,
            "messages": [
                {"role": "system", "content": "SYS_PROMPT_STABLE_123"},
                {"role": "user", "content": "Remember secret BLUE_DOLPHIN_42."},
            ],
        },
    )
    assert r1.status_code == 200
    assert r1.headers["X-Agy-Session-Reused"] == "false"
    assert runner.calls[0][0] is None
    assert "SYS_PROMPT_STABLE_123" in runner.calls[0][1]

    # Turn 2 (same session) -> resumes agy-conv-1 with ONLY delta message!
    runner.next_reply = '<tool_call>{"name": "read_file", "arguments": {"path": "src/app.py"}}</tool_call>'
    r2 = client.post(
        "/v1/chat/completions",
        headers=hdr_a,
        json={
            "model": "gemini-3.8-flash-high",
            "tools": tools,
            "messages": [
                {"role": "system", "content": "SYS_PROMPT_STABLE_123"},
                {"role": "user", "content": "Remember secret BLUE_DOLPHIN_42."},
                {"role": "assistant", "content": "I remembered BLUE_DOLPHIN_42."},
                {"role": "user", "content": "Now read src/app.py"},
            ],
        },
    )
    assert r2.status_code == 200
    assert r2.headers["X-Agy-Session-Reused"] == "true"
    assert r2.headers["X-Agy-Context-Stage"] == "session_delta"
    assert r2.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert runner.calls[1][0] == "agy-conv-1"
    assert "Now read src/app.py" in runner.calls[1][1]
    assert "BLUE_DOLPHIN_42" not in runner.calls[1][1]  # NOT resent!
    assert "SYS_PROMPT_STABLE_123" not in runner.calls[1][1]  # NOT resent!

    # Turn 3 (tool_result echo) -> matches assistant tool_call fingerprint and sends ONLY tool_result delta!
    runner.next_reply = "Done reading src/app.py!"
    r3 = client.post(
        "/v1/chat/completions",
        headers=hdr_a,
        json={
            "model": "gemini-3.8-flash-high",
            "tools": tools,
            "messages": [
                {"role": "system", "content": "SYS_PROMPT_STABLE_123"},
                {"role": "user", "content": "Remember secret BLUE_DOLPHIN_42."},
                {"role": "assistant", "content": "I remembered BLUE_DOLPHIN_42."},
                {"role": "user", "content": "Now read src/app.py"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "client_custom_id_99",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "src/app.py"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "client_custom_id_99", "content": "FILE_BODY_OK"},
            ],
        },
    )
    assert r3.status_code == 200
    assert r3.headers["X-Agy-Session-Reused"] == "true"
    assert runner.calls[2][0] == "agy-conv-1"
    assert "FILE_BODY_OK" in runner.calls[2][1]
    assert "Now read src/app.py" not in runner.calls[2][1]

    # Isolation check: API Key B sending identical history MUST NOT touch Key A's agy-conv-1!
    r_b = client.post(
        "/v1/chat/completions",
        headers=hdr_b,
        json={
            "model": "gemini-3.8-flash-high",
            "tools": tools,
            "messages": [
                {"role": "system", "content": "SYS_PROMPT_STABLE_123"},
                {"role": "user", "content": "Remember secret BLUE_DOLPHIN_42."},
                {"role": "assistant", "content": "I remembered BLUE_DOLPHIN_42."},
                {"role": "user", "content": "Now read src/app.py"},
            ],
        },
    )
    assert r_b.status_code == 200
    assert r_b.headers["X-Agy-Session-Reused"] == "false"
    assert runner.calls[3][0] is None  # fresh conversation for Key B!

    # History edit / client-side /compact check: editing Turn 1 invalidates reuse and starts fresh cleanly
    r_edited = client.post(
        "/v1/chat/completions",
        headers=hdr_a,
        json={
            "model": "gemini-3.8-flash-high",
            "tools": tools,
            "messages": [
                {"role": "system", "content": "SYS_PROMPT_STABLE_123"},
                {"role": "user", "content": "Remember secret EDITED_BY_USER."},
                {"role": "assistant", "content": "I remembered BLUE_DOLPHIN_42."},
                {"role": "user", "content": "Continue"},
            ],
        },
    )
    assert r_edited.status_code == 200
    assert r_edited.headers["X-Agy-Session-Reused"] == "false"
    assert runner.calls[4][0] is None

    stats = disp.context_stats_snapshot()
    assert stats["session_reuse_hits"] == 2
    assert stats["active_sessions"] >= 2


def test_session_mismatch_fallback_and_real_runner_init_guard(tmp_path: Path) -> None:
    """When `--conversation <id>` is missing, runner aborts before emitting and dispatcher retries full context."""
    # 1. Dispatcher transparent fallback without account cooldown
    runner = ResumableFakeRunner()
    client, ks, disp, acc_store = _make_reuse_client(tmp_path, runner)
    _, raw_key = ks.create_key(name="fallback-client", rpm_limit=60)
    hdr = {"Authorization": f"Bearer {raw_key}"}

    runner.next_reply = "A1"
    client.post(
        "/v1/chat/completions",
        headers=hdr,
        json={"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "Turn 1"}]},
    )
    # Simulate agy-conv-1 disappearing before Turn 2!
    runner.fail_conv_ids.add("agy-conv-1")
    runner.next_reply = "Recovered cleanly"
    r2 = client.post(
        "/v1/chat/completions",
        headers=hdr,
        json={
            "model": "gemini-3.8-flash-high",
            "messages": [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Turn 2"},
            ],
        },
    )
    assert r2.status_code == 200
    assert r2.json()["choices"][0]["message"]["content"] == "Recovered cleanly"
    assert r2.headers["X-Agy-Session-Reused"] == "false"
    # Attempted agy-conv-1 first, then automatically retried with full context (conversation_id=None)
    assert runner.calls[1][0] == "agy-conv-1"
    assert runner.calls[2][0] is None
    assert "Turn 1" in runner.calls[2][1] and "Turn 2" in runner.calls[2][1]
    assert disp.context_stats_snapshot()["session_reuse_fallbacks"] == 1
    assert acc_store.get_account("default").total_errors == 0  # type: ignore[union-attr]

    # 2. Subprocess AgyRunner `init` event guard catches silent new-session creation!
    fake_agy = tmp_path / "fake_agy_mismatch.py"
    fake_agy.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "sys.stdin.readline()\n"
        'print(json.dumps({"event": "init", "conversation_id": "brand-new-unexpected-id"}), flush=True)\n'
        'print(json.dumps({"event": "result", "result": {"conversation_id": "brand-new-unexpected-id", "status": "SUCCESS", "response": "oops"}}))\n',
        encoding="utf-8",
    )
    fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IXUSR)
    real_runner = AgyRunner(agy_bin=str(fake_agy), workdir=tmp_path / "work2")
    with pytest.raises(AgySessionMismatchError):
        asyncio.run(
            real_runner.complete_prompt(
                prompt="delta",
                model="gemini-3.8-flash-high",
                conversation_id="expected-old-id",
            )
        )


def test_housekeeping_preserves_active_reusable_sessions(tmp_path: Path) -> None:
    home = tmp_path / "acc_home"
    cli = home / ".gemini" / "antigravity-cli"
    (cli / "conversations").mkdir(parents=True)
    (cli / "brain" / "active-conv-1").mkdir(parents=True)
    (cli / "brain" / "stale-conv-2").mkdir(parents=True)
    (cli / "presence").mkdir(parents=True)

    active_db = cli / "conversations" / "active-conv-1.db"
    active_wal = cli / "conversations" / "active-conv-1.db-wal"
    active_lock = cli / "presence" / "active-conv-1.lock"
    stale_db = cli / "conversations" / "stale-conv-2.db"
    stale_lock = cli / "presence" / "stale-conv-2.lock"

    for p in (active_db, active_wal, active_lock, stale_db, stale_lock):
        p.write_bytes(b"data" * 100)
    (cli / "brain" / "active-conv-1" / "t.jsonl").write_text("ok", encoding="utf-8")
    (cli / "brain" / "stale-conv-2" / "t.jsonl").write_text("old", encoding="utf-8")

    # Age everything by 3 hours (past 30-minute default retention)
    old_ts = time.time() - 10800
    for p in (
        active_db,
        active_wal,
        active_lock,
        stale_db,
        stale_lock,
        cli / "brain" / "active-conv-1",
        cli / "brain" / "stale-conv-2",
    ):
        os.utime(p, (old_ts, old_ts))

    report = prune_agy_artifacts(
        [home],
        retention_minutes=30,
        protected_conversation_ids={"active-conv-1"},
    )
    assert report.removed_entries == 3  # stale db + stale brain + stale lock
    assert active_db.exists()
    assert active_wal.exists()
    assert active_lock.exists()
    assert (cli / "brain" / "active-conv-1").exists()
    assert not stale_db.exists()
    assert not stale_lock.exists()
    assert not (cli / "brain" / "stale-conv-2").exists()


def test_production_bug_fixes_boilerplate_windows_paths_and_media_prefix_hash(tmp_path: Path) -> None:
    """Regression tests for all 4 real-world bugs found in production `context_checkpoints`."""
    # 1. 25KB <system-reminder> + <identity_context> boilerplate must NOT crowd out real user prompt
    boilerplate = (
        "<system-reminder data-role=\"user-context\">\n"
        "<user_info>\nOS Version: win32\nWorkspace Folder: D:/gta7\n</user_info>\n"
        "<identity_context>\n# SOUL.md\n" + ("Template personality paragraph.\n" * 800) + "</identity_context>\n"
        "</system-reminder>\n\n"
    )
    real_goal_a = "把车辆模型改成红色川A出租车，并优化玉林路口转弯手感"
    real_goal_b = "给网关接入 Redis 分布式限流"

    cleaned = extract_clean_human_intent(boilerplate + real_goal_a)
    assert cleaned == real_goal_a

    # Two sessions with identical 25KB boilerplate must produce DIFFERENT session fingerprints!
    req_a = NormalizedRequest(
        model="gemini-3.8-flash-high",
        system_prompt="Same system prompt",
        messages=[ChatMessage(role="user", content=boilerplate + real_goal_a)],
    )
    req_b = NormalizedRequest(
        model="gemini-3.8-flash-high",
        system_prompt="Same system prompt",
        messages=[ChatMessage(role="user", content=boilerplate + real_goal_b)],
    )
    assert compute_session_fingerprint(req_a) != compute_session_fingerprint(req_b)

    # 2. Windows drive-letter path `D:\gta7\vehicles.js` stays intact (NOT truncated to `D`)
    msgs = [
        ChatMessage(role="user", content=boilerplate + real_goal_a),
        ChatMessage(
            role="assistant",
            content="读取车辆模块代码。",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "D:\\gta7\\vehicles.js", "offset": 1, "limit": 200}),
                    },
                }
            ],
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="const car = {};\n" * 600),
    ]
    for i in range(2, 16):
        msgs.append(ChatMessage(role="assistant", content=f"step {i} " + ("W" * 1000)))
        msgs.append(ChatMessage(role="user", content=f"continue {i}"))

    store = ContextCheckpointStore(tmp_path / "gw.db")
    mgr = ContextManager(mode="deterministic", keep_recent_bytes=4_000, checkpoint_max_bytes=4_000, checkpoint_store=store)
    res = mgr.build_context(NormalizedRequest(model="gemini-3.8-flash-high", messages=msgs), max_bytes=12_000)

    assert real_goal_a in res.prompt
    assert "D:\\gta7\\vehicles.js" in res.prompt
    assert "Files inspected: `D`" not in res.prompt

    # 3. Ephemeral per-request `/media_<uuid>/` paths must NOT bust `compute_prefix_hash`!
    m_turn_1 = [
        ChatMessage(
            role="user",
            content="Check screenshot [Attached image #1: /app/data/agy-workdir/media_1111222233334444/attachment_01.webp]",
        )
    ]
    m_turn_2 = [
        ChatMessage(
            role="user",
            content="Check screenshot [Attached image #1: /app/data/agy-workdir/media_aaaabbbbccccdddd/attachment_01.webp]",
        )
    ]
    assert compute_prefix_hash(m_turn_1) == compute_prefix_hash(m_turn_2)


def test_agy_runner_uses_per_turn_step_usage_over_cumulative_result_usage(tmp_path: Path) -> None:
    """`agy --conversation` emits session-cumulative `result.usage`; runner must report per-turn `step_update.usage`."""
    # 1. Resumed single-step turn: result.usage is inflated across prior turns
    fake_single = tmp_path / "fake_agy_cum_single.py"
    fake_single.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "sys.stdin.readline()\n"
        'print(json.dumps({"event": "init", "conversation_id": "conv-abc"}), flush=True)\n'
        'print(json.dumps({"event": "step_update", "step_update": {"conversation_id": "conv-abc", "step_index": 7, "text_delta": "GAMMA", "usage": {"input_tokens": 2239, "output_tokens": 1, "thinking_tokens": 0, "cache_read_tokens": 8203, "total_tokens": 2240}}}), flush=True)\n'
        'print(json.dumps({"event": "result", "result": {"conversation_id": "conv-abc", "status": "SUCCESS", "response": "GAMMA", "usage": {"input_tokens": 14521, "output_tokens": 4, "thinking_tokens": 0, "cache_read_tokens": 16197, "total_tokens": 14525}}}), flush=True)\n',
        encoding="utf-8",
    )
    fake_single.chmod(fake_single.stat().st_mode | stat.S_IXUSR)
    runner = AgyRunner(agy_bin=str(fake_single), workdir=tmp_path / "w1")
    text, usage, cid = asyncio.run(
        runner.complete_prompt(prompt="next", model="gemini-3.8-flash-low", conversation_id="conv-abc")
    )
    assert text == "GAMMA"
    assert cid == "conv-abc"
    assert usage.input_tokens == 2239
    assert usage.cache_read_tokens == 8203
    assert usage.total_input_tokens == 10442
    assert usage.output_tokens == 1
    assert usage.total_tokens == 10443

    # 2. Multi-step internal turn (e.g. read_file on staged image + reply): take LAST step input/cache, sum THIS turn outputs
    fake_multi = tmp_path / "fake_agy_cum_multi.py"
    fake_multi.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "sys.stdin.readline()\n"
        'print(json.dumps({"event": "init", "conversation_id": "conv-img"}), flush=True)\n'
        'print(json.dumps({"event": "step_update", "step_update": {"conversation_id": "conv-img", "step_index": 13, "usage": {"input_tokens": 60699, "output_tokens": 342, "thinking_tokens": 100, "cache_read_tokens": 7853, "total_tokens": 61041}}}), flush=True)\n'
        'print(json.dumps({"event": "step_update", "step_update": {"conversation_id": "conv-img", "step_index": 19, "text_delta": "Image inspected.", "usage": {"input_tokens": 6214, "output_tokens": 962, "thinking_tokens": 300, "cache_read_tokens": 68553, "total_tokens": 7176}}}), flush=True)\n'
        'print(json.dumps({"event": "result", "result": {"conversation_id": "conv-img", "status": "SUCCESS", "response": "Image inspected.", "usage": {"input_tokens": 123099, "output_tokens": 2784, "thinking_tokens": 900, "cache_read_tokens": 195061, "total_tokens": 125883}}}), flush=True)\n',
        encoding="utf-8",
    )
    fake_multi.chmod(fake_multi.stat().st_mode | stat.S_IXUSR)
    runner2 = AgyRunner(agy_bin=str(fake_multi), workdir=tmp_path / "w2")
    _, usage2, _ = asyncio.run(
        runner2.complete_prompt(prompt="inspect", model="gemini-4-neon-high", conversation_id="conv-img")
    )
    assert usage2.input_tokens == 6214
    assert usage2.cache_read_tokens == 68553
    assert usage2.total_input_tokens == 74767
    assert usage2.output_tokens == 342 + 962
    assert usage2.thinking_tokens == 100 + 300
    assert usage2.total_tokens == 74767 + 342 + 962

    # 3. Fallback to result.usage when no step_update.usage is present
    fake_fallback = tmp_path / "fake_agy_result_only.py"
    fake_fallback.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "sys.stdin.readline()\n"
        'print(json.dumps({"event": "result", "result": {"conversation_id": "conv-fb", "status": "SUCCESS", "response": "ok", "usage": {"input_tokens": 55, "output_tokens": 7, "total_tokens": 62}}}), flush=True)\n',
        encoding="utf-8",
    )
    fake_fallback.chmod(fake_fallback.stat().st_mode | stat.S_IXUSR)
    runner3 = AgyRunner(agy_bin=str(fake_fallback), workdir=tmp_path / "w3")
    _, usage3, _ = asyncio.run(runner3.complete_prompt(prompt="hi", model="gemini-3.8-flash-low"))
    assert usage3.input_tokens == 55
    assert usage3.output_tokens == 7


def test_codex_boilerplate_stripping_and_session_hints(tmp_path: Path) -> None:
    """Codex AGENTS.md/<INSTRUCTIONS>/<environment_context> boilerplate must not collide sessions or pollute goals."""
    codex_prefix = (
        "# AGENTS.md instructions for C:\\workspace\\demo-3d\n\n"
        "<INSTRUCTIONS>\n"
        + ("Repo rule line for sample-workspace-skill.\n" * 60)
        + "</INSTRUCTIONS><environment_context>\n"
        "  <cwd>C:\\workspace\\demo-3d</cwd>\n"
        "  <shell>powershell</shell>\n"
        "</environment_context>"
        "<permissions instructions>\nFilesystem unrestricted.\n</permissions instructions>"
    )
    task_a = "用threejs纯代码的形式优化美术和画质，让其像3a游戏大作"
    task_b = "给地图加上夜间霓虹招牌和雨天路面反射"

    # 1. Combined in msg #1
    assert extract_clean_human_intent(codex_prefix + task_a) == task_a
    req_a = NormalizedRequest(
        model="gemini-4-neon-high",
        system_prompt="Codex system prompt",
        messages=[ChatMessage(role="user", content=codex_prefix + task_a)],
    )
    req_b = NormalizedRequest(
        model="gemini-4-neon-high",
        system_prompt="Codex system prompt",
        messages=[ChatMessage(role="user", content=codex_prefix + task_b)],
    )
    assert compute_session_fingerprint(req_a) != compute_session_fingerprint(req_b)

    # 2. Split across separate pure-boilerplate leading user messages
    req_split_a = NormalizedRequest(
        model="gemini-4-neon-high",
        system_prompt="Codex system prompt",
        messages=[
            ChatMessage(role="user", content=codex_prefix),
            ChatMessage(role="user", content=task_a),
        ],
    )
    req_split_b = NormalizedRequest(
        model="gemini-4-neon-high",
        system_prompt="Codex system prompt",
        messages=[
            ChatMessage(role="user", content=codex_prefix),
            ChatMessage(role="user", content=task_b),
        ],
    )
    assert compute_session_fingerprint(req_split_a) != compute_session_fingerprint(req_split_b)

    # 3. Checkpoint Primary Goal captures real task (not AGENTS.md boilerplate)
    msgs = [
        ChatMessage(role="user", content=codex_prefix),
        ChatMessage(role="user", content=task_a),
    ]
    for i in range(1, 14):
        msgs.append(ChatMessage(role="assistant", content=f"step {i} " + ("X" * 1000)))
        msgs.append(ChatMessage(role="user", content=f"next {i}"))
    mgr = ContextManager(mode="deterministic", keep_recent_bytes=3_000, checkpoint_max_bytes=3_000)
    res = mgr.build_context(NormalizedRequest(model="gemini-4-neon-high", messages=msgs), max_bytes=10_000)
    assert task_a in res.prompt
    assert "sample-workspace-skill" not in res.prompt

    # 4. Codex `prompt_cache_key` and `session-id` header isolate sessions even with identical messages
    runner = ResumableFakeRunner()
    client, ks, _, _ = _make_reuse_client(tmp_path, runner)
    _, raw_key = ks.create_key(name="codex-client", rpm_limit=60)
    hdr = {"Authorization": f"Bearer {raw_key}"}

    runner.next_reply = "reply-1"
    r1 = client.post(
        "/v1/chat/completions",
        headers={**hdr, "session-id": "codex-thread-aaa"},
        json={
            "model": "gemini-3.8-flash-high",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r1.status_code == 200
    assert r1.headers["X-Agy-Session-Reused"] == "false"

    # Same messages + prompt_cache_key for a DIFFERENT thread must NOT reuse thread-aaa
    runner.next_reply = "reply-2"
    r2 = client.post(
        "/v1/chat/completions",
        headers=hdr,
        json={
            "model": "gemini-3.8-flash-high",
            "prompt_cache_key": "codex-thread-bbb",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "reply-1"},
                {"role": "user", "content": "Follow up"},
            ],
        },
    )
    assert r2.status_code == 200
    assert r2.headers["X-Agy-Session-Reused"] == "false"

    # Continuing thread-aaa with session-id header MUST reuse agy-conv-1
    runner.next_reply = "reply-3"
    r3 = client.post(
        "/v1/chat/completions",
        headers={**hdr, "session-id": "codex-thread-aaa"},
        json={
            "model": "gemini-3.8-flash-high",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "reply-1"},
                {"role": "user", "content": "Follow up"},
            ],
        },
    )
    assert r3.status_code == 200
    assert r3.headers["X-Agy-Session-Reused"] == "true"
    assert r3.json()["usage"]["prompt_tokens"] == 200  # 120 uncached + 80 cached
    assert r3.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 80
    assert r3.json()["usage"]["total_tokens"] == 210  # 200 + 10 output
