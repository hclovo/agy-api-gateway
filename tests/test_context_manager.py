"""Comprehensive tests for long-session context management, compaction, and overflow recovery."""

from __future__ import annotations

import asyncio
import json
import stat
import sys
from pathlib import Path
from typing import AsyncIterator, List, Optional

import pytest
from fastapi.testclient import TestClient

from agy_gateway.accounts import AccountStore
from agy_gateway.adapters import (
    format_anthropic_response,
    format_gemini_response,
    format_openai_chat_completion,
    parse_anthropic_request,
    parse_gemini_request,
    parse_openai_request,
)
from agy_gateway.agy_client import AgyChunk, AgyExecutionError, AgyModelInfo, AgyRunner, UsageInfo
from agy_gateway.app import create_app
from agy_gateway.config import Settings
from agy_gateway.context_manager import (
    ChatMessage,
    ContextCheckpointStore,
    ContextManager,
    NormalizedRequest,
    clip_utf8_middle,
    estimate_tokens,
    has_truncated_tool_call,
    is_context_overflow_error,
    parse_tool_calls,
    strip_tool_calls,
    utf8_len,
)
from agy_gateway.dispatcher import AgyDispatcher
from agy_gateway.keys import KeyStore
from agy_gateway.media import StagedMedia


def test_cjk_token_estimation_and_utf8_semantic_clipping() -> None:
    ascii_text = "hello world " * 100
    cjk_text = "优化上下文管理防止长会话异常截断" * 80
    # CJK characters carry higher token density than ASCII words
    assert estimate_tokens(cjk_text) > len(cjk_text) * 0.6
    assert estimate_tokens(ascii_text) > len(ascii_text) * 0.2

    multiline = "\n".join(f"第{i}行：测试长会话截断保护 🚀 payload={i * 12345}" for i in range(200))
    clipped = clip_utf8_middle(multiline, 2000)
    assert utf8_len(clipped) <= 2000
    assert "第0行" in clipped
    assert "第199行" in clipped
    assert "omitted" in clipped


def test_atomic_tool_call_pair_integrity_and_material_progress() -> None:
    """Pi `findCutPoint` invariant: never orphan a tool_result, preserve turn-0 goal & material progress."""
    raw_messages = [
        {"role": "user", "content": "PRIMARY_GOAL_AUTH_REFACTOR: Refactor OAuth refresh and fix failing tests."},
    ]
    for step in range(1, 22):
        cid = f"call_{step}"
        if step % 3 == 1:
            tc = {
                "id": cid,
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/mod_{step}.py"})},
            }
            tool_out = f"def fn_{step}():\n" + ("    x = 1234567890\n" * 120)
        elif step % 3 == 2:
            tc = {
                "id": cid,
                "type": "function",
                "function": {"name": "edit_file", "arguments": json.dumps({"path": f"src/edited_{step}.py"})},
            }
            tool_out = f"Successfully updated src/edited_{step}.py\n" + ("diff line\n" * 60)
        else:
            tc = {
                "id": cid,
                "type": "function",
                "function": {"name": "run_command", "arguments": json.dumps({"command": f"pytest tests/test_{step}.py"})},
            }
            tool_out = f"Running pytest {step}\n" + ("ok\n" * 80) + f"AssertionError: mismatch in step {step}\n1 failed\n"

        raw_messages.append(
            {
                "role": "assistant",
                "content": f"Step {step}: analyzing component {step}.",
                "tool_calls": [tc],
            }
        )
        raw_messages.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "content": tool_out,
            }
        )
        if step == 10:
            raw_messages.append(
                {"role": "user", "content": "CONSTRAINT_KEEP_COMPAT: Never break v1 token schema."}
            )

    req = parse_openai_request(
        {"model": "gemini-3.8-flash-high", "messages": raw_messages},
        default_model="gemini-3.8-flash-high",
    )
    mgr = ContextManager(mode="deterministic", keep_recent_bytes=9_000, checkpoint_max_bytes=5_000)
    res = mgr.build_context(req, max_bytes=18_000)

    assert res.compacted is True
    assert res.final_bytes <= 18_000
    assert res.compacted_messages > 0
    assert res.kept_messages > 0

    # Turn-0 goal, mid-session user constraint, and material progress survive compaction!
    assert "PRIMARY_GOAL_AUTH_REFACTOR" in res.prompt
    assert "CONSTRAINT_KEEP_COMPAT" in res.prompt
    assert "src/edited_2.py" in res.prompt
    assert "pytest tests/test_3.py" in res.prompt
    assert "AssertionError" in res.prompt

    # Strict Pi invariant: in the verbatim tail after checkpoint, NO `[TOOL(...)]` appears before its assistant call!
    conv_tail = res.prompt.split("[Conversation]\n", 1)[1]
    # Find the first `[ASSISTANT]` or `[TOOL(` after the checkpoint header
    first_tool_pos = conv_tail.find("\n[TOOL(")
    first_assistant_pos = conv_tail.find("\n[ASSISTANT]")
    assert first_tool_pos != -1
    assert first_assistant_pos != -1
    assert first_assistant_pos < first_tool_pos


def test_anthropic_and_gemini_structured_tools_and_split_turn() -> None:
    """Both Anthropic and Gemini preserve structured tool calls, strip stale thinking, and detect split turns."""
    # Anthropic payload with thinking blocks + 15 tool loops in a single active turn
    ant_messages = [
        {"role": "user", "content": [{"type": "text", "text": "ANTHROPIC_GOAL_999: Implement streaming parser."}]},
    ]
    for i in range(1, 16):
        ant_messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "SECRET_STALE_THOUGHT " * 200},
                    {"type": "text", "text": f"Inspecting file {i}"},
                    {"type": "tool_use", "id": f"toolu_{i}", "name": "read_file", "input": {"path": f"pkg/f{i}.py"}},
                ],
            }
        )
        ant_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{i}",
                        "is_error": (i == 4),
                        "content": (f"RuntimeError: broken invariant in f{i}\n" if i == 4 else "") + ("code_line()\n" * 180),
                    }
                ],
            }
        )

    ant_req = parse_anthropic_request(
        {"model": "gemini-3.8-flash-high", "messages": ant_messages},
        default_model="gemini-3.8-flash-high",
    )
    assert len(ant_req.messages[1].tool_calls) == 1
    assert len(ant_req.messages[2].tool_results) == 1
    assert "SECRET_STALE_THOUGHT" not in ant_req.messages[1].content

    mgr = ContextManager(mode="deterministic", keep_recent_bytes=8_000, checkpoint_max_bytes=5_000)
    ant_res = mgr.build_context(ant_req, max_bytes=16_000)
    assert ant_res.final_bytes <= 16_000
    assert ant_res.split_turn is True
    assert "ANTHROPIC_GOAL_999" in ant_res.prompt
    assert "RuntimeError: broken invariant in f4" in ant_res.prompt
    assert "SECRET_STALE_THOUGHT" not in ant_res.prompt

    # Gemini payload with thought parts + functionCall / functionResponse
    gem_contents = [
        {"role": "user", "parts": [{"text": "GEMINI_GOAL_777: Debug database lock."}]},
    ]
    for i in range(1, 14):
        gem_contents.append(
            {
                "role": "model",
                "parts": [
                    {"text": "STALE_GEMINI_THOUGHT " * 150, "thought": True},
                    {"text": f"Checking table {i}"},
                    {"functionCall": {"id": f"fc_{i}", "name": "read_file", "args": {"path": f"db/t{i}.sql"}}},
                ],
            }
        )
        gem_contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": f"fc_{i}",
                            "name": "read_file",
                            "response": {"output": f"SQL_BODY_{i} " * 400},
                        }
                    }
                ],
            }
        )

    gem_req = parse_gemini_request(model="gemini-4-neon-high", body={"contents": gem_contents})
    assert len(gem_req.messages[1].tool_calls) == 1
    assert len(gem_req.messages[2].tool_results) == 1
    assert "STALE_GEMINI_THOUGHT" not in gem_req.messages[1].content

    gem_res = mgr.build_context(gem_req, max_bytes=16_000)
    assert gem_res.final_bytes <= 16_000
    assert gem_res.split_turn is True
    assert "GEMINI_GOAL_777" in gem_res.prompt


def test_duplicate_file_read_superseding_and_observation_masking() -> None:
    """Repeated reads of the same file are superseded; large tool logs keep diagnostics."""
    messages = [{"role": "user", "content": "Fix parser bug in src/core.py"}]
    for idx in range(1, 5):
        cid = f"read_{idx}"
        messages.append(
            {
                "role": "assistant",
                "content": f"Reading core.py pass {idx}",
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "src/core.py"}'},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "content": f"# Version {idx}\n" + (f"line_{idx} = 'payload'\n" * 600),
            }
        )
    # Add an old noisy pytest failure
    messages.append(
        {
            "role": "assistant",
            "content": "Running tests",
            "tool_calls": [
                {
                    "id": "cmd_test",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "pytest"}'},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "cmd_test",
            "content": ("collecting...\n" * 500) + "TypeError: expected str got None\n1 failed, 12 passed\n",
        }
    )
    messages.append({"role": "user", "content": "Now apply the fix."})

    req = parse_openai_request(
        {"model": "gemini-3.8-flash-high", "messages": messages},
        default_model="gemini-3.8-flash-high",
    )
    mgr = ContextManager(mode="deterministic", tool_output_max_bytes=2_000)
    res = mgr.build_context(req, max_bytes=32_000)

    assert res.deduped_tool_results == 3
    assert res.masked_tool_results >= 1
    assert "Superseded read_file(src/core.py)" in res.prompt
    assert "# Version 4" in res.prompt  # latest read preserved!
    assert "TypeError: expected str got None" in res.prompt
    assert "Now apply the fix." in res.prompt
    assert res.final_bytes <= 32_000


def test_oversized_latest_tool_result_never_starves_user_goal() -> None:
    """A 250KB latest tool dump must NOT evict the active human prompt or assistant tool call."""
    huge_dump = ("INFO noise log line 1234567890\n" * 8000) + "FATAL: OutOfMemoryError at worker.py:42\n"
    req = parse_openai_request(
        {
            "model": "gemini-3.8-flash-high",
            "messages": [
                {"role": "user", "content": "USER_GOAL_MUST_SURVIVE: Investigate why worker crashed."},
                {
                    "role": "assistant",
                    "content": "Let me inspect the crash log.",
                    "tool_calls": [
                        {
                            "id": "call_log",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "logs/worker.log"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_log", "content": huge_dump},
            ],
        },
        default_model="gemini-3.8-flash-high",
    )
    mgr = ContextManager(mode="deterministic")
    res = mgr.build_context(req, max_bytes=24_000)

    assert res.final_bytes <= 24_000
    assert "USER_GOAL_MUST_SURVIVE" in res.prompt
    assert "Let me inspect the crash log." in res.prompt
    assert "FATAL: OutOfMemoryError at worker.py:42" in res.prompt


def test_sticky_checkpoint_prefix_stability_across_turns(tmp_path: Path) -> None:
    """Consecutive turns reuse the exact sticky checkpoint prefix for prompt-cache stability."""
    store = ContextCheckpointStore(tmp_path / "gw.db", ttl_hours=24)
    mgr = ContextManager(
        mode="deterministic",
        keep_recent_bytes=6_000,
        checkpoint_max_bytes=4_000,
        checkpoint_store=store,
    )

    msgs: List[ChatMessage] = [
        ChatMessage(role="user", content="STICKY_GOAL: Build distributed cache."),
    ]
    for i in range(1, 20):
        msgs.append(ChatMessage(role="assistant", content=f"Turn {i} detailed analysis " + ("A" * 900)))
        msgs.append(ChatMessage(role="user", content=f"Turn {i} feedback " + ("B" * 400)))

    req1 = NormalizedRequest(model="gemini-3.8-flash-high", messages=list(msgs), session_id="sess-sticky-1")
    res1 = mgr.build_context(req1, max_bytes=16_000)
    assert res1.compacted_messages > 0
    assert res1.cache_hit is False
    assert store.count() == 1

    # Turn N+1 adds a small message; tail still fits within tail_budget -> sticky prefix hit!
    msgs2 = list(msgs) + [ChatMessage(role="assistant", content="Short follow-up step.")]
    req2 = NormalizedRequest(model="gemini-3.8-flash-high", messages=msgs2, session_id="sess-sticky-1")
    res2 = mgr.build_context(req2, max_bytes=16_000)
    assert res2.cache_hit is True
    assert res2.compacted_messages == res1.compacted_messages
    assert res2.prefix_hash == res1.prefix_hash
    # The prompt prefix up through the checkpoint is byte-for-byte identical!
    cp_marker = "[Compacted Conversation Checkpoint"
    assert cp_marker in res1.prompt and cp_marker in res2.prompt
    p1_prefix = res1.prompt.split("\n\n[ASSISTANT]\nTurn ", 1)[0]
    p2_prefix = res2.prompt.split("\n\n[ASSISTANT]\nTurn ", 1)[0]
    assert p1_prefix == p2_prefix


def test_json_aware_tool_call_parser_nested_tags_fences_and_truncation() -> None:
    # Literal `</tool_call>` inside a JSON string argument must NOT break parsing!
    tricky = (
        "Updating parser:\n"
        '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", '
        '"content": "pat = r\'<tool_call>(.*?)</tool_call>\'"}}</tool_call>\n'
        "And running tests:\n"
        '<tool_call>\n```json\n{"name": "bash", "arguments": {"command": "pytest"}}\n```\n</tool_call>'
    )
    calls = parse_tool_calls(tricky)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "edit_file"
    assert "</tool_call>" in json.loads(calls[0]["function"]["arguments"])["content"]
    assert calls[1]["function"]["name"] == "bash"
    assert strip_tool_calls(tricky) == "Updating parser:\n\nAnd running tests:"
    assert has_truncated_tool_call(tricky) is False

    # Truncated/unclosed tool call must be detected and mapped to length / max_tokens / MAX_TOKENS
    truncated = 'Sure, editing now:\n<tool_call>{"name": "write_file", "arguments": {"path": "a.py", "content": "incomplete'
    assert parse_tool_calls(truncated) == []
    assert has_truncated_tool_call(truncated) is True

    oai = format_openai_chat_completion(truncated, "m", UsageInfo(), has_tools=True)
    assert oai["choices"][0]["finish_reason"] == "length"
    ant = format_anthropic_response(truncated, "m", UsageInfo(), has_tools=True)
    assert ant["stop_reason"] == "max_tokens"
    gem = format_gemini_response(truncated, UsageInfo(), has_tools=True)
    assert gem["candidates"][0]["finishReason"] == "MAX_TOKENS"


def test_overflow_self_healing_without_account_cooldown_and_protocol_native_400(tmp_path: Path) -> None:
    """Context overflow triggers progressive compaction on the SAME account without cooling it down."""

    class SizeLimitedRunner:
        def __init__(self, byte_ceiling: int) -> None:
            self.byte_ceiling = byte_ceiling
            self.observed_sizes: List[int] = []

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
        ) -> AsyncIterator[AgyChunk]:
            sz = utf8_len(prompt)
            self.observed_sizes.append(sz)
            if sz > self.byte_ceiling:
                raise AgyExecutionError("400 CONTEXT_WINDOW_EXCEEDED: prompt is too long for model")
            yield AgyChunk(text_delta="recovered-ok", done=False)
            yield AgyChunk(
                done=True,
                usage=UsageInfo(input_tokens=300, output_tokens=10, total_tokens=310),
                conversation_id="conv-rec",
                finish_reason="stop",
            )

    assert is_context_overflow_error("CONTEXT_WINDOW_EXCEEDED: prompt is too long") is True
    assert is_context_overflow_error("HTTP 429 quota exceeded") is False

    # Case 1: Self-heals on retry (initial prompt ~22KB > ceiling 15KB -> shrinks to <= 15KB -> succeeds!)
    settings = Settings(
        workdir=tmp_path / "work1",
        db_path=tmp_path / "gw1.db",
        accounts_dir=tmp_path / "acc1",
        admin_key="adm-ctx",
        max_prompt_bytes=25_000,
        compaction_keep_recent_bytes=12_000,
        overflow_max_retries=2,
    )
    key_store = KeyStore(settings.db_path)
    _, raw_key = key_store.create_key(name="ctx-client", rpm_limit=60)
    acc_store = AccountStore(settings.db_path, settings.accounts_dir)
    runner = SizeLimitedRunner(byte_ceiling=15_500)
    dispatcher = AgyDispatcher(settings=settings, runner=runner, account_store=acc_store)  # type: ignore[arg-type]
    app = create_app(settings=settings, key_store=key_store, dispatcher=dispatcher, account_store=acc_store)
    client = TestClient(app)

    big_msgs = [{"role": "user", "content": "GOAL_SELF_HEAL: Keep going!"}]
    for i in range(12):
        big_msgs.append({"role": "assistant", "content": f"step {i} " + ("X" * 1000)})
        big_msgs.append({"role": "user", "content": f"reply {i} " + ("Y" * 700)})

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={"model": "gemini-3.8-flash-high", "messages": big_msgs},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "recovered-ok"
    assert resp.headers["X-Agy-Context-Compacted"] == "true"
    assert resp.headers["X-Agy-Context-Stage"] == "overflow_recovery"
    assert len(runner.observed_sizes) == 2
    assert runner.observed_sizes[0] > 15_500
    assert runner.observed_sizes[1] <= 15_500

    # Account MUST remain healthy (zero cooldown, zero account errors!)
    acc = acc_store.get_account("default")
    assert acc is not None
    assert acc.status != "cooldown"
    assert acc.total_errors == 0
    assert dispatcher.context_stats_snapshot()["overflow_recoveries"] == 1

    # Admin context stats & cache clear endpoints work
    ctx_stats = client.get("/admin/context/stats", headers={"Authorization": "Bearer adm-ctx"})
    assert ctx_stats.status_code == 200
    assert ctx_stats.json()["overflow_recoveries"] == 1
    cleared = client.post("/admin/context/cache/clear", headers={"Authorization": "Bearer adm-ctx"})
    assert cleared.status_code == 200

    # Case 2: Unrecoverable overflow -> returns protocol-native HTTP 400 (OpenAI/Anthropic/Gemini) + SSE error
    impossible_runner = SizeLimitedRunner(byte_ceiling=10)
    disp2 = AgyDispatcher(settings=settings, runner=impossible_runner, account_store=acc_store)  # type: ignore[arg-type]
    app2 = create_app(settings=settings, key_store=key_store, dispatcher=disp2, account_store=acc_store)
    client2 = TestClient(app2)

    oai_400 = client2.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert oai_400.status_code == 400
    assert oai_400.json()["error"]["code"] == "context_length_exceeded"

    ant_400 = client2.post(
        "/v1/messages",
        headers={"x-api-key": raw_key},
        json={"model": "gemini-3.8-flash-high", "max_tokens": 128, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert ant_400.status_code == 400
    assert ant_400.json()["error"]["type"] == "invalid_request_error"
    assert "prompt is too long" in ant_400.json()["error"]["message"]

    gem_400 = client2.post(
        f"/v1beta/models/gemini-3.8-flash-high:generateContent?key={raw_key}",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    assert gem_400.status_code == 400
    assert gem_400.json()["error"]["status"] == "INVALID_ARGUMENT"

    # Anthropic SSE overflow error
    with client2.stream(
        "POST",
        "/v1/messages",
        headers={"x-api-key": raw_key},
        json={
            "model": "gemini-3.8-flash-high",
            "stream": True,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as sse_resp:
        lines = list(sse_resp.iter_lines())
    assert "event: error" in lines
    assert any("prompt is too long" in ln for ln in lines)


def test_bounded_map_reduce_llm_compaction_caching_and_deadlock_free_fallback(tmp_path: Path) -> None:
    """`llm` mode uses bounded map/reduce, caches checkpoints in SQLite, and falls back without deadlock."""

    class LlmCompactingRunner:
        def __init__(self, fail_summary: bool = False) -> None:
            self.fail_summary = fail_summary
            self.summary_calls = 0
            self.max_summary_prompt_bytes = 0
            self.last_main_prompt = ""

        async def stream_prompt(
            self,
            prompt: str,
            model: str,
            effort: Optional[str] = None,
            timeout_seconds: float = 180.0,
            env=None,
            cwd=None,
        ) -> AsyncIterator[AgyChunk]:
            if "precision context-compaction engine" in prompt:
                self.summary_calls += 1
                self.max_summary_prompt_bytes = max(self.max_summary_prompt_bytes, utf8_len(prompt))
                if self.fail_summary:
                    raise AgyExecutionError("CONTEXT_WINDOW_EXCEEDED in summarizer")
                yield AgyChunk(
                    text_delta="## Goal & Constraints\n- LLM_SUMMARY_CHECKPOINT_OK\n## Next Steps\n1. Continue task.",
                    done=False,
                )
                yield AgyChunk(done=True, usage=UsageInfo(input_tokens=100, output_tokens=30))
                return
            self.last_main_prompt = prompt
            yield AgyChunk(text_delta="main-reply", done=False)
            yield AgyChunk(done=True, usage=UsageInfo(input_tokens=200, output_tokens=15))

    msgs = [ChatMessage(role="user", content="BUILD_SYSTEM_X")]
    for i in range(1, 25):
        msgs.append(ChatMessage(role="assistant", content=f"turn {i} " + ("M" * 1500)))
        msgs.append(ChatMessage(role="user", content=f"user {i} " + ("U" * 800)))
    req = NormalizedRequest(model="gemini-3.8-flash-high", messages=msgs, session_id="llm-sess-1")

    settings = Settings(
        workdir=tmp_path / "work_llm",
        db_path=tmp_path / "gw_llm.db",
        accounts_dir=tmp_path / "acc_llm",
        max_prompt_bytes=22_000,
        compaction_mode="llm",
        compaction_keep_recent_bytes=7_000,
        compaction_checkpoint_max_bytes=4_000,
    )
    runner = LlmCompactingRunner(fail_summary=False)
    disp = AgyDispatcher(settings=settings, runner=runner)  # type: ignore[arg-type]

    text1, _, _ = asyncio.run(disp.complete(req))
    assert text1 == "main-reply"
    assert runner.summary_calls >= 2  # bounded map/reduce split across multiple chunks!
    assert runner.max_summary_prompt_bytes < 22_000  # strictly bounded well under limit
    assert "LLM_SUMMARY_CHECKPOINT_OK" in runner.last_main_prompt

    # Second call with same prefix reuses cached LLM summary with ZERO extra summary calls!
    prev_calls = runner.summary_calls
    req2 = NormalizedRequest(
        model="gemini-3.8-flash-high",
        messages=msgs + [ChatMessage(role="assistant", content="tiny step")],
        session_id="llm-sess-1",
    )
    text2, _, _ = asyncio.run(disp.complete(req2))
    assert text2 == "main-reply"
    assert runner.summary_calls == prev_calls
    assert "LLM_SUMMARY_CHECKPOINT_OK" in runner.last_main_prompt

    # Deadlock-free fallback when LLM summarizer fails
    fail_settings = Settings(
        workdir=tmp_path / "work_fail",
        db_path=tmp_path / "gw_fail.db",
        accounts_dir=tmp_path / "acc_fail",
        max_prompt_bytes=22_000,
        compaction_mode="llm",
        compaction_keep_recent_bytes=7_000,
    )
    fail_runner = LlmCompactingRunner(fail_summary=True)
    fail_disp = AgyDispatcher(settings=fail_settings, runner=fail_runner)  # type: ignore[arg-type]
    text_fb, _, _ = asyncio.run(fail_disp.complete(req))
    assert text_fb == "main-reply"
    assert "BUILD_SYSTEM_X" in fail_runner.last_main_prompt
    assert fail_disp.context_stats_snapshot()["llm_summary_failures"] == 1


def test_compacted_historical_media_marked_archived_not_forced(tmp_path: Path) -> None:
    """Media from compacted old turns should not force re-reading; active-turn media stays mandatory."""
    p1 = tmp_path / "attachment_01.png"
    p2 = tmp_path / "attachment_02.png"
    staged = [
        StagedMedia(index=1, kind="image", mime_type="image/png", path=p1, size_bytes=100),
        StagedMedia(index=2, kind="image", mime_type="image/png", path=p2, size_bytes=100),
    ]
    msgs = [
        ChatMessage(role="user", content="Old screenshot [Attached image #1 (image/png): __AGY_MEDIA_PATH_1__]"),
    ]
    for i in range(1, 15):
        msgs.append(ChatMessage(role="assistant", content=f"step {i} " + ("Z" * 1200)))
    msgs.append(
        ChatMessage(role="user", content="Latest screenshot [Attached image #2 (image/png): __AGY_MEDIA_PATH_2__]")
    )
    req = NormalizedRequest(model="gemini-3.8-flash-high", messages=msgs)
    mgr = ContextManager(mode="deterministic", keep_recent_bytes=4_000, checkpoint_max_bytes=3_000)
    res = mgr.build_context(req, max_bytes=10_000, staged_media=staged)

    assert res.active_media_indices == (2,)
    assert str(p2) in res.prompt
    assert "archived attachment(s) #1" in res.prompt


def test_anthropic_sse_usage_and_agy_premature_eof_handling(tmp_path: Path) -> None:
    """Anthropic SSE reports real input_tokens, and premature agy EOF never silently succeeds."""
    # 1. Anthropic SSE input_tokens reporting (both non-tool and tool modes)
    class UsageRunner:
        async def list_models(self, force_refresh: bool = False, env=None, cache_key: str = "default"):
            return [AgyModelInfo("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)")]

        async def stream_prompt(
            self,
            prompt: str,
            model: str,
            effort: Optional[str] = None,
            timeout_seconds: float = 180.0,
            env=None,
            cwd=None,
        ) -> AsyncIterator[AgyChunk]:
            yield AgyChunk(text_delta="Hello ", done=False)
            yield AgyChunk(
                done=True,
                usage=UsageInfo(input_tokens=4321, output_tokens=77, cache_read_tokens=1000, total_tokens=4398),
                conversation_id="conv-u",
                finish_reason="stop",
            )

    settings = Settings(
        workdir=tmp_path / "work_sse",
        db_path=tmp_path / "gw_sse.db",
        accounts_dir=tmp_path / "acc_sse",
    )
    ks = KeyStore(settings.db_path)
    _, raw_key = ks.create_key(name="ant-sse", rpm_limit=60)
    disp = AgyDispatcher(settings=settings, runner=UsageRunner())  # type: ignore[arg-type]
    app = create_app(settings=settings, key_store=ks, dispatcher=disp)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/messages",
        headers={"x-api-key": raw_key},
        json={
            "model": "gemini-3.8-flash-high",
            "stream": True,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Analyze this codebase carefully."}],
        },
    ) as resp:
        assert resp.status_code == 200
        data_frames = [json.loads(ln[6:]) for ln in resp.iter_lines() if ln.startswith("data: ")]

    start_frame = next(f for f in data_frames if f.get("type") == "message_start")
    delta_frame = next(f for f in data_frames if f.get("type") == "message_delta")
    assert start_frame["message"]["usage"]["input_tokens"] > 0
    assert delta_frame["usage"]["input_tokens"] == 4321
    assert delta_frame["usage"]["output_tokens"] == 77
    assert delta_frame["usage"]["cache_read_input_tokens"] == 1000

    # 2. Subprocess AgyRunner premature EOF without `result` event
    empty_eof = tmp_path / "agy_empty_eof.py"
    empty_eof.write_text(f"#!{sys.executable}\nimport sys\nsys.stdin.readline()\n", encoding="utf-8")
    empty_eof.chmod(empty_eof.stat().st_mode | stat.S_IXUSR)
    r_empty = AgyRunner(agy_bin=str(empty_eof), workdir=tmp_path / "w1")
    with pytest.raises(AgyExecutionError, match="prematurely"):
        asyncio.run(r_empty.complete_prompt("hi", "gemini-3.8-flash-high"))

    partial_eof = tmp_path / "agy_partial_eof.py"
    partial_eof.write_text(
        f"#!{sys.executable}\nimport json, sys\nsys.stdin.readline()\n"
        'print(json.dumps({"event": "step_update", "step_update": {"text_delta": "half answer"}}))\n',
        encoding="utf-8",
    )
    partial_eof.chmod(partial_eof.stat().st_mode | stat.S_IXUSR)
    r_partial = AgyRunner(agy_bin=str(partial_eof), workdir=tmp_path / "w2")

    async def _collect():
        return [c async for c in r_partial.stream_prompt("hi", "gemini-3.8-flash-high")]

    collected = asyncio.run(_collect())
    assert collected[0].text_delta == "half answer"
    assert collected[-1].done is True
    assert collected[-1].finish_reason == "length"


def test_hybrid_mode_serves_immediately_and_upgrades_checkpoint_in_background(tmp_path: Path) -> None:
    """`hybrid` mode (pi-async-compaction style) never blocks Turn 1 and upgrades Turn 2 via SQLite cache."""

    class HybridFakeRunner:
        def __init__(self) -> None:
            self.bg_called = 0
            self.prompts: List[str] = []

        async def stream_prompt(
            self,
            prompt: str,
            model: str,
            effort: Optional[str] = None,
            timeout_seconds: float = 180.0,
            env=None,
            cwd=None,
        ) -> AsyncIterator[AgyChunk]:
            if "precision context-compaction engine" in prompt:
                self.bg_called += 1
                await asyncio.sleep(0.02)
                yield AgyChunk(
                    text_delta="## Goal & Constraints\n- HYBRID_BG_UPGRADED_SUMMARY\n## Next Steps\n1. Ship it.",
                    done=False,
                )
                yield AgyChunk(done=True, usage=UsageInfo(input_tokens=50, output_tokens=20))
                return
            self.prompts.append(prompt)
            yield AgyChunk(text_delta="ok", done=False)
            yield AgyChunk(done=True, usage=UsageInfo(input_tokens=100, output_tokens=5))

    settings = Settings(
        workdir=tmp_path / "work_hyb",
        db_path=tmp_path / "gw_hyb.db",
        accounts_dir=tmp_path / "acc_hyb",
        max_prompt_bytes=20_000,
        compaction_mode="hybrid",
        compaction_keep_recent_bytes=6_000,
        compaction_checkpoint_max_bytes=4_000,
    )
    runner = HybridFakeRunner()
    disp = AgyDispatcher(settings=settings, runner=runner)  # type: ignore[arg-type]

    msgs = [ChatMessage(role="user", content="HYBRID_INITIAL_GOAL")]
    for i in range(1, 20):
        msgs.append(ChatMessage(role="assistant", content=f"step {i} " + ("D" * 1200)))
        msgs.append(ChatMessage(role="user", content=f"note {i} " + ("E" * 600)))

    async def _run():
        # Turn 1: served immediately with deterministic checkpoint while BG task upgrades cache
        req1 = NormalizedRequest(model="gemini-3.8-flash-high", messages=list(msgs), session_id="hyb-1")
        res1 = await disp.complete_with_meta(req1)
        assert res1.text == "ok"
        assert "HYBRID_INITIAL_GOAL" in runner.prompts[0]
        assert "HYBRID_BG_UPGRADED_SUMMARY" not in runner.prompts[0]

        # Wait for scheduled background compaction task to finish
        if disp._bg_compaction_tasks:
            await asyncio.gather(*list(disp._bg_compaction_tasks.values()))
        assert runner.bg_called >= 1

        # Turn 2: immediately hits upgraded LLM summary in SQLite cache!
        req2 = NormalizedRequest(
            model="gemini-3.8-flash-high",
            messages=list(msgs) + [ChatMessage(role="assistant", content="tiny next step")],
            session_id="hyb-1",
        )
        res2 = await disp.complete_with_meta(req2)
        assert res2.text == "ok"
        assert res2.context_result is not None
        assert res2.context_result.cache_hit is True
        assert "HYBRID_BG_UPGRADED_SUMMARY" in runner.prompts[1]
        await disp.shutdown()

    asyncio.run(_run())


def test_extreme_200_turn_2mb_cjk_and_code_session_stress() -> None:
    """Stress test a ~2MB 200-turn bilingual coding session with 80 tools under 191,500 bytes."""
    import time

    tools = [
        {
            "type": "function",
            "function": {
                "name": f"agent_tool_{i}",
                "description": f"Detailed tool {i} documentation " * 25,
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string", "description": "Target file path"},
                        "mode": {"type": "string", "enum": ["read", "write", "diff"]},
                    },
                },
            },
        }
        for i in range(80)
    ]

    messages = [
        {"role": "system", "content": "你是一个资深系统架构师。请严格遵守规范。\n" * 600},
        {"role": "user", "content": "核心任务目标_9988：重构订单撮合引擎并解决高并发死锁问题。"},
    ]
    for turn in range(1, 100):
        cid = f"call_stress_{turn}"
        if turn % 4 == 0:
            path = "engine/matcher.py"  # repeatedly read to test superseding!
            t_name = "read_file"
            args = json.dumps({"path": path})
            output = f"# matcher.py rev {turn}\n" + ("def match_order(a, b): return a + b\n" * 1200)
        elif turn % 4 == 1:
            t_name = "edit_file"
            args = json.dumps({"path": f"engine/module_{turn}.py"})
            output = f"Edited engine/module_{turn}.py successfully.\n" + ("+ changed line payload\n" * 300)
        elif turn % 4 == 2:
            t_name = "run_command"
            args = json.dumps({"command": f"pytest -q tests/test_matcher_{turn}.py"})
            output = ("....\n" * 600) + (f"AssertionError: lock timeout in turn {turn}\n1 failed\n" if turn == 42 else "10 passed\n")
        else:
            t_name = "grep_search"
            args = json.dumps({"path": "engine/", "query": "acquire_lock"})
            output = "\n".join(f"engine/f{k}.py:{k}: acquire_lock()" for k in range(600))

        messages.append(
            {
                "role": "assistant",
                "content": f"第{turn}轮分析：正在处理模块 {turn} 的锁竞争问题。",
                "tool_calls": [{"id": cid, "type": "function", "function": {"name": t_name, "arguments": args}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": cid, "content": output})
        if turn == 50:
            messages.append({"role": "user", "content": "重要约束_7766：不要修改外部公开 API 签名。"})

    messages.append({"role": "user", "content": "最终指令_FINAL_123：请给出最终回归测试结论。"})

    req = parse_openai_request(
        {"model": "gemini-4-neon-high", "messages": messages, "tools": tools},
        default_model="gemini-4-neon-high",
    )
    mgr = ContextManager(mode="deterministic")
    t0 = time.perf_counter()
    res = mgr.build_context(req, max_bytes=191_500)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert res.original_bytes > 1_500_000
    assert res.final_bytes <= 191_500
    assert elapsed_ms < 500.0  # fast deterministic path
    assert res.deduped_tool_results >= 20
    assert "agent_tool_0(" in res.prompt
    assert "agent_tool_79(" in res.prompt
    assert "核心任务目标_9988" in res.prompt
    assert "重要约束_7766" in res.prompt
    assert "engine/module_1.py" in res.prompt
    assert "AssertionError: lock timeout in turn 42" in res.prompt
    assert res.prompt.endswith("最终指令_FINAL_123：请给出最终回归测试结论。")

