"""Unit tests for protocol adapters and prompt/tool conversions."""

from __future__ import annotations

from agy_gateway.adapters import (
    format_anthropic_response,
    format_gemini_response,
    format_openai_chat_completion,
    parse_anthropic_request,
    parse_gemini_request,
    parse_openai_request,
)
from agy_gateway.agy_client import UsageInfo
from agy_gateway.dispatcher import MAX_PROMPT_BYTES, build_prompt, parse_tool_calls


def test_openai_request_and_tool_call_parsing() -> None:
    req = parse_openai_request(
        {
            "model": "gemini-2.5-flash",
            "stream": True,
            "reasoning_effort": "low",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": [{"type": "text", "text": "Check weather"}]},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object"}},
                }
            ],
        },
        default_model="gemini-3.8-flash-high",
    )
    assert req.system_prompt == "Be concise."
    assert len(req.messages) == 1
    assert req.messages[0].content == "Check weather"
    assert req.effort == "low"

    prompt = build_prompt(req)
    assert "[System Instructions]\nBe concise." in prompt
    assert "get_weather" in prompt

    raw_reply = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
    calls = parse_tool_calls(raw_reply)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"

    resp = format_openai_chat_completion(
        text=raw_reply,
        model="gemini-2.5-flash",
        usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
        has_tools=True,
    )
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert resp["choices"][0]["message"]["content"] is None
    assert resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_gemini_and_anthropic_adapters() -> None:
    gem_req = parse_gemini_request(
        model="gemini-2.5-pro",
        body={
            "systemInstruction": {"parts": [{"text": "You are helpful."}]},
            "contents": [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {"role": "model", "parts": [{"text": "Hello"}]},
                {"role": "user", "parts": [{"text": "2+2?"}]},
            ],
        },
    )
    assert gem_req.system_prompt == "You are helpful."
    assert [m.role for m in gem_req.messages] == ["user", "assistant", "user"]

    gem_resp = format_gemini_response("4", UsageInfo(input_tokens=8, output_tokens=1, total_tokens=9))
    assert gem_resp["candidates"][0]["content"]["parts"][0]["text"] == "4"
    assert gem_resp["usageMetadata"]["totalTokenCount"] == 9

    ant_req = parse_anthropic_request(
        {
            "model": "gemini-3.8-flash-high",
            "system": [{"type": "text", "text": "Respond in JSON"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
        },
        default_model="gemini-3.8-flash-high",
    )
    assert ant_req.system_prompt == "Respond in JSON"
    ant_resp = format_anthropic_response(
        '{"ok": true}',
        model="gemini-3.8-flash-high",
        usage=UsageInfo(input_tokens=20, output_tokens=4),
    )
    assert ant_resp["stop_reason"] == "end_turn"
    assert ant_resp["content"][0]["text"] == '{"ok": true}'


def test_oversized_500kb_tools_and_history_never_truncate_latest_user_message() -> None:
    huge_tools = []
    for idx in range(120):
        huge_tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"mcp_tool_{idx}",
                    "description": ("Very detailed MCP tool documentation. " * 90),
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": ("Verbose parameter instructions. " * 30),
                            },
                            "limit": {"type": "integer", "enum": [10, 20, 50]},
                            "filters": {
                                "type": "object",
                                "properties": {"tag": {"type": "string"}},
                            },
                        },
                    },
                },
            }
        )

    messages = [
        {"role": "system", "content": "System guide. " * 4000},
        {"role": "user", "content": "Old message 1 " * 3000},
        {"role": "assistant", "content": "Old reply 1 " * 3000},
        {"role": "user", "content": "Old huge tool dump " * 6000},
        {"role": "user", "content": "FINAL_USER_QUESTION_MUST_STAY_INTACT_999"},
    ]
    req = parse_openai_request(
        {"model": "gemini-3.8-flash-high", "messages": messages, "tools": huge_tools},
        default_model="gemini-3.8-flash-high",
    )
    prompt = build_prompt(req)
    encoded_len = len(prompt.encode("utf-8"))

    # Strictly below agy CLI's 191,515-byte truncation cliff
    assert encoded_len <= MAX_PROMPT_BYTES
    assert encoded_len < 190_000
    # All 120 tools remain callable by name and latest user turn is completely intact at the tail
    assert "mcp_tool_0(" in prompt
    assert "mcp_tool_119(" in prompt
    assert "[Conversation]" in prompt
    assert prompt.endswith("FINAL_USER_QUESTION_MUST_STAY_INTACT_999")


def test_protocol_accurate_usage_translation_and_session_hint_scoping() -> None:
    """Verify OpenAI, Gemini, and Anthropic usage contracts for cached & thinking tokens."""
    # `agy` reports input_tokens (uncached=2245), cache_read_tokens=7994, output_tokens=649 (includes 369 thinking)
    usage = UsageInfo(
        input_tokens=2245,
        output_tokens=649,
        thinking_tokens=369,
        cache_read_tokens=7994,
        total_tokens=2894,
    )

    # 1. OpenAI: prompt_tokens INCLUDES cached_tokens; completion_tokens INCLUDES reasoning_tokens
    oa = format_openai_chat_completion("ok", model="gpt-4o", usage=usage)
    assert oa["usage"]["prompt_tokens"] == 2245 + 7994
    assert oa["usage"]["prompt_tokens_details"]["cached_tokens"] == 7994
    assert oa["usage"]["prompt_tokens"] >= oa["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert oa["usage"]["completion_tokens"] == 649
    assert oa["usage"]["completion_tokens_details"]["reasoning_tokens"] == 369
    assert oa["usage"]["total_tokens"] == (2245 + 7994) + 649

    # 2. Gemini: promptTokenCount INCLUDES cached; candidatesTokenCount EXCLUDES thoughts (no double count!)
    gem = format_gemini_response("ok", usage=usage)
    um = gem["usageMetadata"]
    assert um["promptTokenCount"] == 2245 + 7994
    assert um["cachedContentTokenCount"] == 7994
    assert um["candidatesTokenCount"] == 649 - 369
    assert um["thoughtsTokenCount"] == 369
    assert um["totalTokenCount"] == um["promptTokenCount"] + um["candidatesTokenCount"] + um["thoughtsTokenCount"]

    # 3. Anthropic: input_tokens and cache_read_input_tokens remain disjoint
    ant = format_anthropic_response("ok", model="claude-opus-5-5-high", usage=usage)
    assert ant["usage"]["input_tokens"] == 2245
    assert ant["usage"]["cache_read_input_tokens"] == 7994
    assert ant["usage"]["output_tokens"] == 649

    # 4. Anthropic 100% cache hit (input_tokens=0, cache_read_tokens=8000) must NOT inject fallback_input_tokens
    from agy_gateway.adapters import _anthropic_usage_dict

    full_cache_u = _anthropic_usage_dict(
        UsageInfo(input_tokens=0, output_tokens=12, cache_read_tokens=8000),
        fallback_input_tokens=9999,
    )
    assert full_cache_u["input_tokens"] == 0
    assert full_cache_u["cache_read_input_tokens"] == 8000

    # 5. Session hints: accept prompt_cache_key & Claude Code session user_id; ignore generic OpenAI `user`
    r_codex = parse_openai_request(
        {
            "model": "gpt-4o",
            "prompt_cache_key": "thread-xyz",
            "user": "generic-alice",
            "messages": [{"role": "user", "content": "hi"}],
        },
        default_model="gemini-3.8-flash-high",
    )
    assert r_codex.session_id == "thread-xyz"

    r_generic_user = parse_openai_request(
        {
            "model": "gpt-4o",
            "user": "generic-alice",
            "messages": [{"role": "user", "content": "hi"}],
        },
        default_model="gemini-3.8-flash-high",
    )
    assert r_generic_user.session_id is None

    r_claude_code = parse_anthropic_request(
        {
            "model": "claude-opus-5-5-high",
            "metadata": {"user_id": "user_abc_account_123_session_550e8400-e29b-41d4"},
            "messages": [{"role": "user", "content": "hi"}],
        },
        default_model="claude-opus-5-5-high",
    )
    assert r_claude_code.session_id == "user_abc_account_123_session_550e8400-e29b-41d4"

