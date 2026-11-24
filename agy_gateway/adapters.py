"""Protocol adapters for OpenAI, Gemini, and Anthropic HTTP APIs."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from agy_gateway.agy_client import AgyChunk, AgyModelInfo, UsageInfo
from agy_gateway.dispatcher import (
    ChatMessage,
    NormalizedRequest,
    ToolResultItem,
    extract_text_content,
    has_truncated_tool_call,
    parse_tool_calls,
    strip_tool_calls,
)
from agy_gateway.media import MediaAttachment, extract_media_block


def sse_data(payload: Dict[str, Any]) -> str:
    """Format a standard JSON SSE data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_event(event_name: str, payload: Dict[str, Any]) -> str:
    """Format a named SSE event frame (used by Anthropic Messages API)."""
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _openai_usage_dict(usage: UsageInfo) -> Dict[str, Any]:
    total = usage.total_tokens or (usage.input_tokens + usage.output_tokens)
    out: Dict[str, Any] = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": total,
    }
    if usage.cache_read_tokens > 0:
        out["prompt_tokens_details"] = {"cached_tokens": usage.cache_read_tokens}
    if usage.thinking_tokens > 0:
        out["completion_tokens_details"] = {"reasoning_tokens": usage.thinking_tokens}
    return out


def _gemini_usage_dict(usage: UsageInfo) -> Dict[str, int]:
    total = usage.total_tokens or (usage.input_tokens + usage.output_tokens)
    return {
        "promptTokenCount": usage.input_tokens,
        "candidatesTokenCount": usage.output_tokens,
        "thoughtsTokenCount": usage.thinking_tokens,
        "totalTokenCount": total,
    }


def _anthropic_usage_dict(
    usage: UsageInfo,
    *,
    fallback_input_tokens: int = 0,
    include_input: bool = True,
) -> Dict[str, int]:
    in_tok = usage.input_tokens if usage.input_tokens > 0 else fallback_input_tokens
    out: Dict[str, int] = {"output_tokens": usage.output_tokens}
    if include_input:
        out["input_tokens"] = in_tok
    if usage.cache_read_tokens > 0:
        out["cache_read_input_tokens"] = usage.cache_read_tokens
    return out


def _extract_session_hint(body: Dict[str, Any], header_session_id: Optional[str] = None) -> Optional[str]:
    if header_session_id and header_session_id.strip():
        return header_session_id.strip()
    meta = body.get("metadata")
    if isinstance(meta, dict):
        for key in ("session_id", "conversation_id", "thread_id", "user_id"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    user_val = body.get("user")
    if isinstance(user_val, str) and user_val.strip():
        return user_val.strip()
    return None


def _parse_anthropic_content_blocks(
    content: Any,
    attachments: List[MediaAttachment],
) -> Tuple[str, List[Dict[str, Any]], List[ToolResultItem]]:
    """Extract text/media, structured tool_use calls, and structured tool_result items."""
    if content is None:
        return "", [], []
    if isinstance(content, str):
        return content, [], []
    if not isinstance(content, list):
        return str(content), [], []

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[ToolResultItem] = []

    for idx, item in enumerate(content):
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in ("thinking", "redacted_thinking"):
            # Drop stale client-replayed thinking blocks to save context window
            continue
        if item_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(item.get("id") or f"toolu_hist_{idx}"),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": item.get("input", {}),
                    },
                }
            )
            continue
        if item_type == "tool_result":
            tool_use_id = str(item.get("tool_use_id") or "")
            inner = extract_text_content(item.get("content"), attachments=attachments)
            tool_results.append(
                ToolResultItem(
                    tool_call_id=tool_use_id,
                    content=inner,
                    is_error=bool(item.get("is_error", False)),
                )
            )
            continue
        if "text" in item and isinstance(item["text"], str):
            text_parts.append(item["text"])
            continue
        media_marker = extract_media_block(item, attachments)
        if media_marker:
            text_parts.append(media_marker)

    return "\n".join(p for p in text_parts if p), tool_calls, tool_results


def _parse_gemini_parts(
    parts: Any,
    attachments: List[MediaAttachment],
) -> Tuple[str, List[Dict[str, Any]], List[ToolResultItem]]:
    """Extract text/media, structured functionCall blocks, and structured functionResponse items."""
    if parts is None:
        return "", [], []
    if isinstance(parts, str):
        return parts, [], []
    if not isinstance(parts, list):
        return str(parts), [], []

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[ToolResultItem] = []

    for idx, item in enumerate(parts):
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("thought") is True:
            # Skip historical model thoughts replayed in contents[]
            continue
        if "functionCall" in item and isinstance(item["functionCall"], dict):
            fc = item["functionCall"]
            tool_calls.append(
                {
                    "id": str(fc.get("id") or f"gfc_{idx}"),
                    "type": "function",
                    "function": {
                        "name": str(fc.get("name") or ""),
                        "arguments": fc.get("args", {}),
                    },
                }
            )
            continue
        if "functionResponse" in item and isinstance(item["functionResponse"], dict):
            fr = item["functionResponse"]
            resp_obj = fr.get("response", {})
            if isinstance(resp_obj, str):
                resp_text = resp_obj
            else:
                resp_text = json.dumps(resp_obj, ensure_ascii=False)
            tool_results.append(
                ToolResultItem(
                    tool_call_id=str(fr.get("id") or ""),
                    name=str(fr.get("name") or ""),
                    content=resp_text,
                )
            )
            continue
        if "text" in item and isinstance(item["text"], str):
            text_parts.append(item["text"])
            continue
        media_marker = extract_media_block(item, attachments)
        if media_marker:
            text_parts.append(media_marker)

    return "\n".join(p for p in text_parts if p), tool_calls, tool_results


def parse_openai_request(
    body: Dict[str, Any],
    default_model: str,
    session_id: Optional[str] = None,
) -> NormalizedRequest:
    """Convert an OpenAI /v1/chat/completions payload into NormalizedRequest."""
    system_parts: List[str] = []
    messages: List[ChatMessage] = []
    attachments: List[MediaAttachment] = []
    for raw_msg in body.get("messages") or []:
        if not isinstance(raw_msg, dict):
            continue
        role = str(raw_msg.get("role") or "user").lower()
        text = extract_text_content(raw_msg.get("content"), attachments=attachments)
        if role in ("system", "developer"):
            if text:
                system_parts.append(text)
            continue
        messages.append(
            ChatMessage(
                role=role,
                content=text,
                name=raw_msg.get("name"),
                tool_call_id=raw_msg.get("tool_call_id"),
                tool_calls=list(raw_msg.get("tool_calls") or []),
            )
        )

    return NormalizedRequest(
        model=str(body.get("model") or default_model),
        messages=messages,
        system_prompt="\n\n".join(system_parts) if system_parts else None,
        stream=bool(body.get("stream", False)),
        effort=body.get("reasoning_effort"),
        tools=list(body.get("tools") or []),
        tool_choice=body.get("tool_choice"),
        max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
        temperature=body.get("temperature"),
        attachments=attachments,
        session_id=_extract_session_hint(body, header_session_id=session_id),
    )


def format_openai_models(
    models: List[AgyModelInfo], aliases: Dict[str, str]
) -> Dict[str, Any]:
    """Build OpenAI /v1/models response including configured aliases."""
    now_ts = int(time.time())
    seen: set[str] = set()
    data: List[Dict[str, Any]] = []
    for m in models:
        if m.id not in seen:
            seen.add(m.id)
            data.append(
                {
                    "id": m.id,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": "google-antigravity",
                    "display_name": m.display_name,
                }
            )
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            data.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": "agy-gateway-alias",
                }
            )
    return {"object": "list", "data": data}


def format_openai_chat_completion(
    text: str,
    model: str,
    usage: UsageInfo,
    has_tools: bool = False,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an OpenAI /v1/chat/completions non-streaming JSON response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    tool_calls = parse_tool_calls(text) if has_tools else []
    if tool_calls:
        leading = strip_tool_calls(text)
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": leading or None,
            "tool_calls": tool_calls,
        }
        resolved_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        if finish_reason == "length" or (has_tools and has_truncated_tool_call(text)):
            resolved_reason = "length"
        else:
            resolved_reason = "stop"

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": resolved_reason,
            }
        ],
        "usage": _openai_usage_dict(usage),
    }


async def iter_openai_sse(
    chunks: AsyncIterator[AgyChunk],
    model: str,
    has_tools: bool = False,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    """Translate AgyChunk stream into OpenAI SSE chunks."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def make_chunk(
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        empty_choices: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": []
            if empty_choices
            else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        return sse_data(payload)

    yield make_chunk({"role": "assistant"})

    buffered_parts: List[str] = []
    final_usage = UsageInfo()
    stream_finish_reason: Optional[str] = None

    async for chunk in chunks:
        if chunk.finish_reason:
            stream_finish_reason = chunk.finish_reason
        if chunk.done:
            final_usage = chunk.usage
            continue
        if has_tools:
            if chunk.thinking_delta:
                # Forward reasoning deltas live even while buffering text for <tool_call> parsing
                yield make_chunk({"reasoning_content": chunk.thinking_delta})
            if chunk.text_delta:
                buffered_parts.append(chunk.text_delta)
            continue

        delta: Dict[str, Any] = {}
        if chunk.thinking_delta:
            delta["reasoning_content"] = chunk.thinking_delta
        if chunk.text_delta:
            delta["content"] = chunk.text_delta
        if delta:
            yield make_chunk(delta)

    if has_tools:
        full_text = "".join(buffered_parts)
        tool_calls = parse_tool_calls(full_text)
        if tool_calls:
            leading = strip_tool_calls(full_text)
            if leading:
                yield make_chunk({"content": leading})
            indexed_calls = [
                {
                    "index": idx,
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
                for idx, tc in enumerate(tool_calls)
            ]
            yield make_chunk({"tool_calls": indexed_calls})
            yield make_chunk({}, finish_reason="tool_calls")
        else:
            if full_text:
                yield make_chunk({"content": full_text})
            end_reason = (
                "length"
                if (stream_finish_reason == "length" or has_truncated_tool_call(full_text))
                else "stop"
            )
            yield make_chunk({}, finish_reason=end_reason)
    else:
        end_reason = "length" if stream_finish_reason == "length" else "stop"
        yield make_chunk({}, finish_reason=end_reason)

    if include_usage:
        yield make_chunk({}, empty_choices=True, usage=_openai_usage_dict(final_usage))

    yield "data: [DONE]\n\n"


def parse_gemini_request(
    model: str,
    body: Dict[str, Any],
    stream: bool = False,
    session_id: Optional[str] = None,
) -> NormalizedRequest:
    """Convert a Gemini :generateContent payload into NormalizedRequest with structured tools."""
    attachments: List[MediaAttachment] = []
    system_instruction = body.get("systemInstruction") or body.get("system_instruction")
    system_prompt = None
    if isinstance(system_instruction, dict):
        system_prompt = extract_text_content(
            system_instruction.get("parts"), attachments=attachments
        )
    elif isinstance(system_instruction, str):
        system_prompt = system_instruction

    messages: List[ChatMessage] = []
    for item in body.get("contents") or []:
        if not isinstance(item, dict):
            continue
        raw_role = str(item.get("role") or "user").lower()
        role = "assistant" if raw_role == "model" else raw_role
        text, tool_calls, tool_results = _parse_gemini_parts(
            item.get("parts"), attachments=attachments
        )
        if text or tool_calls or tool_results:
            messages.append(
                ChatMessage(
                    role=role,
                    content=text,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                )
            )

    tools: List[Dict[str, Any]] = []
    for tool_group in body.get("tools") or []:
        if isinstance(tool_group, dict) and "functionDeclarations" in tool_group:
            for fn in tool_group["functionDeclarations"]:
                tools.append({"type": "function", "function": fn})

    gen_cfg = body.get("generationConfig") or {}
    return NormalizedRequest(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        stream=stream,
        tools=tools,
        max_tokens=gen_cfg.get("maxOutputTokens"),
        temperature=gen_cfg.get("temperature"),
        attachments=attachments,
        session_id=_extract_session_hint(body, header_session_id=session_id),
    )


def format_gemini_models(
    models: List[AgyModelInfo], aliases: Dict[str, str]
) -> Dict[str, Any]:
    """Build Gemini /v1beta/models response."""
    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for m in models:
        seen.add(m.id)
        entries.append(
            {
                "name": f"models/{m.id}",
                "baseModelId": m.id,
                "displayName": m.display_name,
                "supportedGenerationMethods": [
                    "generateContent",
                    "streamGenerateContent",
                ],
            }
        )
    for alias in aliases:
        if alias not in seen:
            entries.append(
                {
                    "name": f"models/{alias}",
                    "baseModelId": alias,
                    "displayName": f"{alias} (Gateway Alias)",
                    "supportedGenerationMethods": [
                        "generateContent",
                        "streamGenerateContent",
                    ],
                }
            )
    return {"models": entries}


def format_gemini_response(
    text: str,
    usage: UsageInfo,
    finish_reason: Optional[str] = "STOP",
    has_tools: bool = False,
) -> Dict[str, Any]:
    """Build a Gemini generateContent JSON payload."""
    parts: List[Dict[str, Any]] = []
    tool_calls = parse_tool_calls(text) if has_tools else []
    resolved_finish = finish_reason
    if tool_calls:
        leading = strip_tool_calls(text)
        if leading:
            parts.append({"text": leading})
        for tc in tool_calls:
            fn = tc["function"]
            try:
                args_obj = json.loads(fn["arguments"])
            except json.JSONDecodeError:
                args_obj = {"raw": fn["arguments"]}
            parts.append({"functionCall": {"name": fn["name"], "args": args_obj}})
    else:
        parts.append({"text": text})
        if resolved_finish is not None and (
            resolved_finish == "length" or (has_tools and has_truncated_tool_call(text))
        ):
            resolved_finish = "MAX_TOKENS"
        elif resolved_finish == "stop":
            resolved_finish = "STOP"

    candidate: Dict[str, Any] = {
        "content": {"role": "model", "parts": parts},
        "index": 0,
    }
    if resolved_finish is not None:
        candidate["finishReason"] = resolved_finish
    payload: Dict[str, Any] = {"candidates": [candidate]}
    if resolved_finish is not None:
        payload["usageMetadata"] = _gemini_usage_dict(usage)
    return payload


async def iter_gemini_sse(
    chunks: AsyncIterator[AgyChunk],
    has_tools: bool = False,
) -> AsyncIterator[str]:
    """Translate AgyChunk stream into Gemini SSE chunks."""
    final_usage = UsageInfo()
    stream_finish_reason: Optional[str] = None
    if has_tools:
        buffered: List[str] = []
        async for chunk in chunks:
            if chunk.finish_reason:
                stream_finish_reason = chunk.finish_reason
            if chunk.done:
                final_usage = chunk.usage
                continue
            if chunk.text_delta:
                buffered.append(chunk.text_delta)
        full_text = "".join(buffered)
        gem_reason = (
            "MAX_TOKENS"
            if (stream_finish_reason == "length" or has_truncated_tool_call(full_text))
            else "STOP"
        )
        yield sse_data(
            format_gemini_response(
                full_text,
                usage=final_usage,
                finish_reason=gem_reason,
                has_tools=True,
            )
        )
        return

    async for chunk in chunks:
        if chunk.finish_reason:
            stream_finish_reason = chunk.finish_reason
        if chunk.done:
            final_usage = chunk.usage
            continue
        if chunk.text_delta:
            yield sse_data(
                format_gemini_response(
                    chunk.text_delta, usage=final_usage, finish_reason=None
                )
            )
    gem_reason = "MAX_TOKENS" if stream_finish_reason == "length" else "STOP"
    yield sse_data(format_gemini_response("", usage=final_usage, finish_reason=gem_reason))


def parse_anthropic_request(
    body: Dict[str, Any],
    default_model: str,
    session_id: Optional[str] = None,
) -> NormalizedRequest:
    """Convert an Anthropic /v1/messages payload into NormalizedRequest with structured tools."""
    attachments: List[MediaAttachment] = []
    system_prompt = extract_text_content(body.get("system"), attachments=attachments) or None
    messages: List[ChatMessage] = []
    for raw_msg in body.get("messages") or []:
        if not isinstance(raw_msg, dict):
            continue
        role = str(raw_msg.get("role") or "user").lower()
        text, tool_calls, tool_results = _parse_anthropic_content_blocks(
            raw_msg.get("content"), attachments=attachments
        )
        if text or tool_calls or tool_results:
            messages.append(
                ChatMessage(
                    role=role,
                    content=text,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                )
            )

    tools: List[Dict[str, Any]] = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )

    return NormalizedRequest(
        model=str(body.get("model") or default_model),
        messages=messages,
        system_prompt=system_prompt,
        stream=bool(body.get("stream", False)),
        tools=tools,
        max_tokens=body.get("max_tokens"),
        temperature=body.get("temperature"),
        attachments=attachments,
        session_id=_extract_session_hint(body, header_session_id=session_id),
    )


def format_anthropic_response(
    text: str,
    model: str,
    usage: UsageInfo,
    has_tools: bool = False,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an Anthropic /v1/messages non-streaming JSON response."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    tool_calls = parse_tool_calls(text) if has_tools else []
    if tool_calls:
        content: List[Dict[str, Any]] = []
        leading = strip_tool_calls(text)
        if leading:
            content.append({"type": "text", "text": leading})
        for idx, tc in enumerate(tool_calls):
            fn = tc["function"]
            try:
                tool_input = json.loads(fn["arguments"])
            except json.JSONDecodeError:
                tool_input = {"raw": fn["arguments"]}
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:16]}_{idx}",
                    "name": fn["name"],
                    "input": tool_input,
                }
            )
        stop_reason = "tool_use"
    else:
        content = [{"type": "text", "text": text}]
        if finish_reason == "length" or (has_tools and has_truncated_tool_call(text)):
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _anthropic_usage_dict(usage),
    }


async def iter_anthropic_sse(
    chunks: AsyncIterator[AgyChunk],
    model: str,
    has_tools: bool = False,
    estimated_input_tokens: int = 0,
) -> AsyncIterator[str]:
    """Translate AgyChunk stream into Anthropic Messages SSE events with accurate token usage."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def make_message_start(usage: UsageInfo) -> str:
        start_usage = _anthropic_usage_dict(
            UsageInfo(
                input_tokens=usage.input_tokens,
                output_tokens=0,
                cache_read_tokens=usage.cache_read_tokens,
            ),
            fallback_input_tokens=estimated_input_tokens,
            include_input=True,
        )
        return sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": start_usage,
                },
            },
        )

    final_usage = UsageInfo()
    stop_reason = "end_turn"
    stream_finish_reason: Optional[str] = None

    if has_tools:
        buffered: List[str] = []
        async for chunk in chunks:
            if chunk.finish_reason:
                stream_finish_reason = chunk.finish_reason
            if chunk.done:
                final_usage = chunk.usage
                continue
            if chunk.usage.input_tokens > 0:
                final_usage = chunk.usage
            if chunk.text_delta:
                buffered.append(chunk.text_delta)

        # Emit message_start AFTER buffering so exact input_tokens are reported & pre-stream
        # errors never leave an unclosed message_start frame.
        yield make_message_start(final_usage)

        full_text = "".join(buffered)
        tool_calls = parse_tool_calls(full_text)
        if tool_calls:
            stop_reason = "tool_use"
            block_idx = 0
            leading = strip_tool_calls(full_text)
            if leading:
                yield sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": leading},
                    },
                )
                yield sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block_idx},
                )
                block_idx += 1
            for idx, tc in enumerate(tool_calls):
                cur_idx = block_idx + idx
                fn = tc["function"]
                try:
                    tool_input = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    tool_input = {"raw": fn["arguments"]}
                yield sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": cur_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": f"toolu_{uuid.uuid4().hex[:16]}_{idx}",
                            "name": fn["name"],
                            "input": {},
                        },
                    },
                )
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": cur_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tool_input, ensure_ascii=False),
                        },
                    },
                )
                yield sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": cur_idx},
                )
        else:
            if stream_finish_reason == "length" or has_truncated_tool_call(full_text):
                stop_reason = "max_tokens"
            yield sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            if full_text:
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": full_text},
                    },
                )
            yield sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 0}
            )
    else:
        started = False
        async for chunk in chunks:
            if chunk.finish_reason:
                stream_finish_reason = chunk.finish_reason
            if chunk.usage.input_tokens > 0 or chunk.done:
                final_usage = chunk.usage
            if not started:
                yield make_message_start(final_usage)
                yield sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                started = True
            if chunk.done:
                continue
            if chunk.text_delta:
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": chunk.text_delta},
                    },
                )
        if not started:
            yield make_message_start(final_usage)
            yield sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
        if stream_finish_reason == "length":
            stop_reason = "max_tokens"

    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": _anthropic_usage_dict(
                final_usage,
                fallback_input_tokens=estimated_input_tokens,
                include_input=True,
            ),
        },
    )
    yield sse_event("message_stop", {"type": "message_stop"})
