"""Advanced context management, observation masking, atomic turn slicing, and compaction.

Synthesizes proven long-session patterns from open-source coding agent harnesses:
- Pi (`pi-mono` / `pi-vcc`): `findCutPoint` with strict `tool_call`/`tool_result` pair
  integrity, `isSplitTurn` active-turn preservation, structured handoff checkpoints,
  and deterministic algorithmic compression.
- DeepSeek Harness (`deepseek-harness`): CJK-aware token estimation, deadlock-free
  bounded chronological map/reduce summarization, and provider-agnostic overflow sniffing.
- Paseo (`getpaseo/paseo`): durable material-progress tracking (files read/modified,
  commands executed, open blockers), prefix-stable prompt layout, and safe cache reuse.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from agy_gateway.db import DatabaseBackend
from agy_gateway.media import MediaAttachment, StagedMedia, media_placeholder

# `agy` CLI hard-truncates USER_INPUT at exactly 192,000 UTF-8 bytes.
# Default ceiling leaves a 500-byte safety margin below the physical cliff.
MAX_PROMPT_BYTES = 191_500
MAX_SYSTEM_BYTES = 96_000
MAX_TOOLS_BYTES = 64_000
MAX_HISTORICAL_TURN_BYTES = 64_000

VALID_COMPACTION_MODES = ("deterministic", "hybrid", "llm")

_THINKING_BLOCK_RE = re.compile(r"<(?:thinking|thought)>.*?</(?:thinking|thought)>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)
_CODE_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$")
_BOILERPLATE_TAG_RE = re.compile(
    r"<(system-reminder|identity_context|user_info|environment_details|attached_files)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_NUMBERED_FILE_LINE_RE = re.compile(r"^\s*\d+→")
_EPHEMERAL_MEDIA_PATH_RE = re.compile(r"(?:/[^\s\]\"']*?)?/media_[0-9a-f]{16}/[^\s\]\"']+")
_TRAILING_LINE_RANGE_RE = re.compile(r":\d*-\d*$")

_ERROR_LINE_RE = re.compile(
    r"(?:^|\b)(?:error|exception|traceback|failed|fatal|panic|assertionerror|syntaxerror|"
    r"typeerror|valueerror|keyerror|attributeerror|importerror|modulenotfounderror|"
    r"permissionerror|timeout|exit code [1-9]\d*|exited with code [1-9]\d*|"
    r"enoent|econnrefused|segmentation fault)\b",
    re.IGNORECASE,
)
_TEST_SUMMARY_RE = re.compile(
    r"(?:\b\d+\s+(?:passed|failed|errors?|skipped|warnings?)\b|"
    r"^(?:PASS|FAIL|OK|ERROR)\b|tests?\s+(?:passed|failed))",
    re.IGNORECASE,
)
_FILE_PATH_RE = re.compile(
    r"(?:^|[\s\"'(`])((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\."
    r"(?:py|ts|tsx|js|jsx|mjs|cjs|go|rs|java|kt|c|cc|cpp|h|hpp|rb|php|sh|zsh|bash|"
    r"json|yaml|yml|toml|md|html|css|scss|sql|Dockerfile|env|txt|ini|cfg))(?=$|[\s\"'`),:;])"
)

_READ_TOOL_NAMES = frozenset(
    {
        "read",
        "read_file",
        "view_file",
        "view_file_outline",
        "view_code_item",
        "open_file",
        "cat",
        "get_file",
        "file_read",
        "inspect_file",
    }
)
_WRITE_TOOL_NAMES = frozenset(
    {
        "write",
        "write_file",
        "write_to_file",
        "create_file",
        "edit",
        "edit_file",
        "replace_file_content",
        "multi_replace_file_content",
        "str_replace",
        "str_replace_editor",
        "patch",
        "apply_patch",
        "apply_diff",
        "notebook_edit",
    }
)
_LIST_TOOL_NAMES = frozenset(
    {
        "ls",
        "list_dir",
        "list_files",
        "find_by_name",
        "glob",
        "grep",
        "grep_search",
        "rg",
        "search_files",
    }
)
_EXEC_TOOL_NAMES = frozenset(
    {
        "bash",
        "shell",
        "exec",
        "execute",
        "run_command",
        "run_shell_command",
        "terminal",
        "command",
    }
)

_OVERFLOW_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"context[_\s-]*length[_\s-]*exceeded", re.IGNORECASE),
    re.compile(r"context[_\s-]*window[_\s-]*(?:exceeded|limit|full|overflow)", re.IGNORECASE),
    re.compile(r"maximum\s+context\s+length", re.IGNORECASE),
    re.compile(r"prompt\s+(?:is\s+)?too\s+(?:long|large)", re.IGNORECASE),
    re.compile(r"input\s+(?:is\s+)?too\s+(?:long|large)", re.IGNORECASE),
    re.compile(r"(?:request|payload|content|message)\s+(?:is\s+)?too\s+(?:large|long)", re.IGNORECASE),
    re.compile(r"(?:input\s+)?token\s+count\s+exceeds", re.IGNORECASE),
    re.compile(r"exceeds?\s+(?:the\s+)?maximum\s+(?:number\s+of\s+)?tokens", re.IGNORECASE),
    re.compile(r"too\s+many\s+(?:input\s+)?tokens", re.IGNORECASE),
    re.compile(r"prompt_too_long|request_too_large|content_too_large|max_prompt_tokens", re.IGNORECASE),
    re.compile(r"reduce\s+the\s+length\s+of\s+the\s+messages", re.IGNORECASE),
    re.compile(r"context\s+size\s+has\s+been\s+exceeded", re.IGNORECASE),
    re.compile(r"exceeded\s+model\s+token\s+limit", re.IGNORECASE),
)

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a precision context-compaction engine for an autonomous coding agent. "
    "Compress the provided conversation segment into a durable, actionable handoff checkpoint "
    "so another agent turn can seamlessly continue executing the task without losing goals, "
    "constraints, file paths, exact error messages, or progress. "
    "Do NOT invent facts or omit unresolved errors. Be dense and structured."
)

SUMMARIZATION_TEMPLATE = """Produce a concise Markdown handoff checkpoint using EXACTLY these headings:

## Goal & Constraints
- [Core user objective(s) and explicit constraints/preferences]

## Progress
### Done
- [x] [Completed steps, verified edits, passed checks]
### In Progress
- [ ] [Active unfinished task items]
### Blocked / Open Errors
- [ ] [Exact error messages, failing tests, or unresolved issues; or "(none)"]

## Key Decisions & Touched Files
- [Architectural decisions, exact file paths, function/symbol names, commands run]

## Next Steps
1. [Immediate concrete actions required to continue]
"""


class ContextOverflowError(RuntimeError):
    """Raised when a prompt still exceeds upstream context limits after compaction recovery."""


@dataclass(frozen=True)
class ToolResultItem:
    """Structured tool execution result preserved across OpenAI, Anthropic, and Gemini."""

    tool_call_id: str = ""
    name: str = ""
    content: str = ""
    is_error: bool = False


@dataclass(frozen=True)
class ChatMessage:
    """Protocol-agnostic normalized chat turn preserving structured tool metadata."""

    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[ToolResultItem] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizedRequest:
    """Protocol-agnostic generation request forwarded to AgyDispatcher."""

    model: str
    messages: List[ChatMessage]
    system_prompt: Optional[str] = None
    stream: bool = False
    effort: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    attachments: List[MediaAttachment] = field(default_factory=list)
    session_id: Optional[str] = None


@dataclass(frozen=True)
class ContextBuildResult:
    """Rendered prompt plus rich observability metadata for diagnostics and stats."""

    prompt: str
    original_bytes: int
    final_bytes: int
    original_tokens_est: int
    final_tokens_est: int
    compacted: bool = False
    compaction_stage: str = "none"  # none | tools_and_masking | turn_compaction | overflow_recovery
    compaction_mode: str = "deterministic"
    cache_hit: bool = False
    total_messages: int = 0
    kept_messages: int = 0
    compacted_messages: int = 0
    masked_tool_results: int = 0
    deduped_tool_results: int = 0
    split_turn: bool = False
    session_fingerprint: str = ""
    prefix_hash: str = ""
    compacted_prefix_text: str = ""
    compacted_prefix_bytes: int = 0
    active_media_indices: Tuple[int, ...] = ()
    llm_summary_cached: bool = False
    session_reused: bool = False


@dataclass
class _TurnUnit:
    """Atomic sequence of messages that must NEVER be split across a compaction boundary."""

    start_idx: int
    end_idx: int  # exclusive
    kind: str  # "human" | "tool_exchange" | "assistant" | "orphan_tool"
    messages: List[ChatMessage]
    is_human_turn_start: bool = False


@dataclass(frozen=True)
class _CutPointResult:
    """Pi-style cut point locating the safe boundary between compacted prefix and verbatim tail."""

    cut_unit_index: int
    cut_message_index: int
    is_split_turn: bool
    turn_start_message_index: Optional[int]


def utf8_len(text: str) -> int:
    """Return exact UTF-8 byte length of a string."""
    return len(text.encode("utf-8"))


def estimate_tokens(text: str) -> int:
    """CJK-aware conservative token estimator (avoids Pi & DSH CJK undercount bugs)."""
    if not text:
        return 0
    ascii_chars = 0
    non_ascii_chars = 0
    punct_chars = 0
    for ch in text:
        code = ord(ch)
        if code < 128:
            ascii_chars += 1
            if ch in "{}[]\":,;=()<>":
                punct_chars += 1
        else:
            non_ascii_chars += 1
    # Conservative weights:
    # - ASCII text/code: ~3.7 chars/token + slight JSON/code punctuation overhead
    # - CJK & wide Unicode: ~1.4 chars/token (~0.72 tokens/char) so CJK is never underestimated
    est = (ascii_chars / 3.7) + (punct_chars * 0.08) + (non_ascii_chars * 0.72)
    return max(1, int(est + 0.99))


def is_context_overflow_error(error_text: str) -> bool:
    """Return True if an upstream/agy error text indicates a context-window or prompt-size overflow."""
    if not error_text:
        return False
    for pattern in _OVERFLOW_PATTERNS:
        if pattern.search(error_text):
            return True
    return False


def clip_utf8_middle(text: str, max_bytes: int, head_ratio: float = 0.65) -> str:
    """Clip the middle of a UTF-8 string on line boundaries when possible, never exceeding max_bytes."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    if max_bytes <= 0:
        return ""
    if max_bytes < 160:
        return raw[:max_bytes].decode("utf-8", errors="ignore")

    omitted_est = max(1, len(raw) - max_bytes + 80)
    marker = f"\n...[omitted ~{omitted_est} bytes]...\n"
    marker_bytes = marker.encode("utf-8")
    budget = max(0, max_bytes - len(marker_bytes))
    head_budget = max(32, int(budget * head_ratio))
    tail_budget = max(0, budget - head_budget)

    head = raw[:head_budget].decode("utf-8", errors="ignore")
    tail = raw[-tail_budget:].decode("utf-8", errors="ignore") if tail_budget > 0 else ""

    # Snap to clean line boundaries if doing so keeps at least 75% of each half.
    if "\n" in head:
        nl = head.rfind("\n")
        if nl >= int(len(head) * 0.75):
            head = head[:nl]
    if "\n" in tail:
        nl = tail.find("\n")
        if nl != -1 and (len(tail) - nl) >= int(len(tail) * 0.75):
            tail = tail[nl + 1 :]

    combined = f"{head}{marker}{tail}"
    while utf8_len(combined) > max_bytes and len(head) > 16:
        head = head[:-max(16, len(head) // 12)]
        combined = f"{head}{marker}{tail}"
    if utf8_len(combined) > max_bytes:
        return combined.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
    return combined


def _one_line(text: Any, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(1, max_chars - 1)] + "…"


def _compact_schema_type(schema: Any, depth: int = 0) -> str:
    """Render a JSON Schema property into a concise TypeScript-like type expression."""
    if not isinstance(schema, dict):
        return "any"

    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        vals = schema["enum"]
        preview = "|".join(json.dumps(v, ensure_ascii=False) for v in vals[:6])
        return f"enum[{preview}{'|…' if len(vals) > 6 else ''}]"

    for union_key in ("anyOf", "oneOf"):
        if isinstance(schema.get(union_key), list) and schema[union_key]:
            sub = [_compact_schema_type(s, depth + 1) for s in schema[union_key][:3]]
            dedup = list(dict.fromkeys(sub))
            return "|".join(dedup)

    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        return "|".join(str(t) for t in raw_type)

    if raw_type == "string":
        return "str"
    if raw_type == "integer":
        return "int"
    if raw_type == "number":
        return "num"
    if raw_type == "boolean":
        return "bool"
    if raw_type == "array":
        item_type = _compact_schema_type(schema.get("items") or {}, depth + 1)
        return f"arr<{item_type}>"
    if raw_type == "object" or "properties" in schema:
        props = schema.get("properties")
        if depth < 2 and isinstance(props, dict) and props:
            req_set = set(schema.get("required") or [])
            items: List[str] = []
            for idx, (k, v) in enumerate(props.items()):
                if idx >= 10:
                    items.append("…")
                    break
                opt = "" if k in req_set else "?"
                items.append(f"{k}{opt}:{_compact_schema_type(v, depth + 1)}")
            return "{" + ", ".join(items) + "}"
        return "obj"
    return str(raw_type or "any")


def _dedupe_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for raw_tool in tools:
        if not isinstance(raw_tool, dict):
            continue
        fn = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else raw_tool
        name = str(fn.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(raw_tool)
    return out


def _render_tools_tier(
    tools: Sequence[Dict[str, Any]],
    desc_chars: int,
    param_desc_chars: int,
    include_types: bool = True,
) -> str:
    lines: List[str] = []
    for raw_tool in tools:
        fn = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else raw_tool
        name = fn.get("name")
        if not name:
            continue
        desc = _one_line(fn.get("description"), desc_chars)
        params_schema = fn.get("parameters") or fn.get("input_schema") or {}
        props = params_schema.get("properties") if isinstance(params_schema, dict) else {}
        req_set = set(params_schema.get("required") or []) if isinstance(params_schema, dict) else set()

        param_parts: List[str] = []
        if isinstance(props, dict):
            for p_name, p_schema in props.items():
                opt = "" if p_name in req_set else "?"
                if not include_types:
                    param_parts.append(f"{p_name}{opt}")
                    continue
                p_type = _compact_schema_type(p_schema)
                p_hint = (
                    _one_line(p_schema.get("description"), param_desc_chars)
                    if isinstance(p_schema, dict) and param_desc_chars > 0
                    else ""
                )
                if p_hint:
                    param_parts.append(f"{p_name}{opt}: {p_type} ({p_hint})")
                else:
                    param_parts.append(f"{p_name}{opt}: {p_type}")

        sig = f"- {name}({', '.join(param_parts)})"
        if desc:
            sig = f"{sig} — {desc}"
        lines.append(sig)
    return "\n".join(lines)


def format_compact_tools(tools: Sequence[Dict[str, Any]], max_bytes: int = MAX_TOOLS_BYTES) -> str:
    """Compile verbose JSON Schema tool definitions into compact signatures within max_bytes."""
    deduped = _dedupe_tools(tools)
    header = (
        "[External Harness Tools]\n"
        "If you need to invoke one or more external tools instead of giving a final text answer, "
        'output ONLY <tool_call>{"name": "...", "arguments": {...}}</tool_call> blocks '
        "(valid JSON arguments matching the parameter names below).\n"
    )
    body_budget = max(512, max_bytes - utf8_len(header))

    # Progressive compaction tiers: Detailed -> Concise -> Bare signatures -> Names only -> Hard cap.
    tiers = [
        (160, 50, True),
        (80, 0, True),
        (40, 0, False),
        (0, 0, False),
    ]
    rendered = ""
    for desc_chars, param_desc_chars, include_types in tiers:
        rendered = _render_tools_tier(
            deduped,
            desc_chars=desc_chars,
            param_desc_chars=param_desc_chars,
            include_types=include_types,
        )
        if utf8_len(rendered) <= body_budget:
            return header + rendered

    return header + clip_utf8_middle(rendered, body_budget)


# --- JSON-Aware Tool Call Extraction (Resilient to `</tool_call>` Inside Code Strings) ---


def _scan_tool_call_spans(text: str) -> Tuple[List[Tuple[int, int, Dict[str, Any]]], bool]:
    """Scan `<tool_call>{...}</tool_call>` blocks using JSONDecoder.raw_decode.

    Returns (parsed_blocks, has_unclosed_or_truncated_block).
    Handles nested braces, escaped quotes, markdown code fences, and literal `</tool_call>`
    appearing inside JSON string values (e.g., when editing agent/gateway code).
    """
    if "<tool_call>" not in text:
        return [], False

    decoder = json.JSONDecoder()
    results: List[Tuple[int, int, Dict[str, Any]]] = []
    has_truncated = False
    pos = 0
    n = len(text)
    open_tag = "<tool_call>"
    close_tag = "</tool_call>"

    while pos < n:
        start = text.find(open_tag, pos)
        if start == -1:
            break
        cursor = start + len(open_tag)
        # Skip whitespace and optional opening markdown fence (e.g. ```json)
        while cursor < n and text[cursor] in " \t\r\n":
            cursor += 1
        fence_match = _CODE_FENCE_OPEN_RE.match(text[cursor:])
        if fence_match:
            cursor += fence_match.end()
            while cursor < n and text[cursor] in " \t\r\n":
                cursor += 1

        parsed_obj: Optional[Dict[str, Any]] = None
        end_of_json = -1
        if cursor < n and text[cursor] == "{":
            try:
                obj, end_of_json = decoder.raw_decode(text, cursor)
                if isinstance(obj, dict) and obj.get("name"):
                    parsed_obj = obj
            except json.JSONDecodeError:
                parsed_obj = None

        if parsed_obj is not None and end_of_json != -1:
            tail_pos = end_of_json
            while tail_pos < n and text[tail_pos] in " \t\r\n":
                tail_pos += 1
            if text.startswith("```", tail_pos):
                tail_pos += 3
                while tail_pos < n and text[tail_pos] in " \t\r\n":
                    tail_pos += 1
            if text.startswith(close_tag, tail_pos):
                block_end = tail_pos + len(close_tag)
                results.append((start, block_end, parsed_obj))
                pos = block_end
                continue
            # JSON parsed cleanly, look shortly ahead for closing tag
            close_idx = text.find(close_tag, tail_pos)
            if close_idx != -1 and (close_idx - tail_pos) <= 16:
                block_end = close_idx + len(close_tag)
                results.append((start, block_end, parsed_obj))
                pos = block_end
                continue
            # Unclosed closing tag at EOF -> likely truncated right at the end
            has_truncated = True
            pos = end_of_json
            continue

        # Fallback: try slicing up to the next `</tool_call>` if raw_decode failed
        close_idx = text.find(close_tag, cursor)
        if close_idx == -1:
            has_truncated = True
            break
        inner = text[cursor:close_idx].strip()
        inner = _CODE_FENCE_OPEN_RE.sub("", inner)
        inner = _CODE_FENCE_CLOSE_RE.sub("", inner).strip()
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and obj.get("name"):
                results.append((start, close_idx + len(close_tag), obj))
        except json.JSONDecodeError:
            pass
        pos = close_idx + len(close_tag)

    return results, has_truncated


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract `<tool_call>{...}</tool_call>` blocks returned by the model."""
    spans, _ = _scan_tool_call_spans(text)
    calls: List[Dict[str, Any]] = []
    for idx, (_, _, parsed) in enumerate(spans):
        args = parsed.get("arguments", {})
        if isinstance(args, str):
            try:
                json.loads(args)
                args_str = args
            except json.JSONDecodeError:
                args_str = json.dumps({"raw": args}, ensure_ascii=False)
        else:
            args_str = json.dumps(args, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_agy_{idx + 1}",
                "type": "function",
                "function": {"name": str(parsed["name"]), "arguments": args_str},
            }
        )
    return calls


def strip_tool_calls(text: str) -> str:
    """Return any non-tool-call explanation text surrounding valid `<tool_call>` blocks."""
    spans, _ = _scan_tool_call_spans(text)
    if not spans:
        return text.strip()
    parts: List[str] = []
    last = 0
    for start, end, _ in spans:
        if start > last:
            parts.append(text[last:start])
        last = end
    if last < len(text):
        parts.append(text[last:])
    return "".join(parts).strip()


def has_truncated_tool_call(text: str) -> bool:
    """Return True if the model output ends with an unclosed/truncated `<tool_call>` block."""
    _, truncated = _scan_tool_call_spans(text)
    return truncated


# --- Structured Tool-Call Metadata & Observation Masking ---


def _safe_parse_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
    return {}


def _extract_tool_target(tool_name: str, args: Dict[str, Any]) -> Tuple[str, str]:
    """Return (kind, normalized_target) for deduplication and progress tracking."""
    lower = tool_name.lower()
    # Normalize common path/command keys across OpenCode, Claude Code, Pi, Paseo, DSH, Codex
    path_val = (
        args.get("path")
        or args.get("file_path")
        or args.get("filePath")
        or args.get("Path")
        or args.get("TargetFile")
        or args.get("AbsolutePath")
        or args.get("filename")
        or args.get("file")
        or ""
    )
    cmd_val = (
        args.get("command")
        or args.get("CommandLine")
        or args.get("cmd")
        or args.get("script")
        or ""
    )
    dir_val = args.get("dir") or args.get("directory") or args.get("DirectoryPath") or ""
    query_val = args.get("query") or args.get("Query") or args.get("pattern") or ""

    if lower in _WRITE_TOOL_NAMES or any(k in lower for k in ("write", "edit", "replace", "patch")):
        return "write", str(path_val).strip()
    if lower in _READ_TOOL_NAMES or any(k in lower for k in ("read_file", "view_file", "open_file")):
        start_ln = args.get("start_line") or args.get("StartLine") or args.get("offset") or ""
        end_ln = args.get("end_line") or args.get("EndLine") or args.get("limit") or ""
        suffix = f":{start_ln}-{end_ln}" if (start_ln or end_ln) else ""
        return "read", f"{str(path_val).strip()}{suffix}" if path_val else ""
    if lower in _EXEC_TOOL_NAMES or any(k in lower for k in ("bash", "shell", "command", "exec")):
        return "exec", _one_line(cmd_val, 140)
    if lower in _LIST_TOOL_NAMES or any(k in lower for k in ("list", "glob", "grep", "search", "find")):
        target = str(path_val or dir_val or "").strip()
        if query_val:
            return "search", f"{target}:{_one_line(query_val, 80)}".strip(":")
        return "list", target
    return "other", str(path_val or cmd_val or query_val or "").strip()[:120]


@dataclass(frozen=True)
class _ToolCallMeta:
    call_id: str
    name: str
    kind: str  # read | write | exec | list | search | other
    target: str
    turn_index: int


def _index_tool_calls(messages: Sequence[ChatMessage]) -> Dict[str, _ToolCallMeta]:
    """Map tool_call_id -> metadata across all messages."""
    index: Dict[str, _ToolCallMeta] = {}
    for idx, msg in enumerate(messages):
        for tc_pos, tc in enumerate(msg.tool_calls):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            name = str(fn.get("name") or tc.get("name") or "tool")
            cid = str(tc.get("id") or f"__anon_{idx}_{tc_pos}")
            args = _safe_parse_args(fn.get("arguments") if "arguments" in fn else tc.get("arguments"))
            kind, target = _extract_tool_target(name, args)
            index[cid] = _ToolCallMeta(
                call_id=cid,
                name=name,
                kind=kind,
                target=target,
                turn_index=idx,
            )
    return index


def _is_pure_tool_result_message(msg: ChatMessage) -> bool:
    if msg.role == "tool" or msg.tool_call_id:
        return True
    if msg.role == "user" and msg.tool_results and not msg.content.strip():
        return True
    return False


def _is_genuine_human_message(msg: ChatMessage) -> bool:
    if msg.role != "user":
        return False
    if msg.tool_call_id:
        return False
    if msg.content.strip():
        return True
    return not bool(msg.tool_results)


def _strip_stale_thinking(text: str) -> str:
    if not text or ("<thinking" not in text.lower() and "<thought" not in text.lower()):
        return text
    cleaned = _THINKING_BLOCK_RE.sub("", text).strip()
    return cleaned if cleaned else text


def _normalize_ephemeral_media_paths(text: str) -> str:
    if not text or "/media_" not in text:
        return text
    return _EPHEMERAL_MEDIA_PATH_RE.sub("[archived media attachment]", text)


def extract_clean_human_intent(text: str) -> str:
    """Strip injected harness boilerplate (`<system-reminder>`, `SOUL.md`, numbered dumps)
    so the user's real task instruction is never crowded out of checkpoints or fingerprints.
    """
    if not text:
        return ""
    cleaned = text
    for _ in range(4):
        updated = _BOILERPLATE_TAG_RE.sub("", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = _normalize_ephemeral_media_paths(cleaned)

    # Collapse long runs of numbered file lines (` 123→...`) injected by editor harnesses
    raw_lines = cleaned.splitlines()
    kept_lines: List[str] = []
    numbered_run: List[str] = []

    def _flush_run() -> None:
        if len(numbered_run) > 10:
            kept_lines.append(f"[inline file snippet: {len(numbered_run)} lines omitted]")
        else:
            kept_lines.extend(numbered_run)
        numbered_run.clear()

    for ln in raw_lines:
        if _NUMBERED_FILE_LINE_RE.match(ln):
            numbered_run.append(ln)
        else:
            if numbered_run:
                _flush_run()
            kept_lines.append(ln)
    if numbered_run:
        _flush_run()

    result = "\n".join(kept_lines).strip()
    # Collapse excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    if result:
        return result
    # If everything was wrapped inside tags, keep the tail where user instructions usually appear
    return _normalize_ephemeral_media_paths(text.strip())


def _extract_error_lines(text: str, max_lines: int = 8, max_line_chars: int = 180) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _ERROR_LINE_RE.search(line) or _TEST_SUMMARY_RE.search(line):
            compact = _one_line(line, max_line_chars)
            if compact not in seen:
                seen.add(compact)
                found.append(compact)
                if len(found) >= max_lines:
                    break
    return found


def summarize_tool_output_deterministic(
    text: str,
    max_bytes: int,
    *,
    tool_name: str = "",
    target_hint: str = "",
    is_error: bool = False,
    aggressive: bool = False,
) -> str:
    """Compress a large/stale tool output while preserving status, error lines, head, and tail."""
    raw_bytes = utf8_len(text)
    if raw_bytes <= max_bytes:
        return text

    lines = text.splitlines()
    line_count = len(lines)
    errors = _extract_error_lines(text, max_lines=6 if not aggressive else 4)
    has_err = is_error or bool(errors)

    header_parts = ["[Compacted tool output"]
    if tool_name:
        header_parts.append(f"tool={tool_name}")
    if target_hint:
        header_parts.append(f"target={_one_line(target_hint, 80)}")
    header_parts.append(f"{line_count} lines, {raw_bytes:,} bytes")
    if has_err:
        header_parts.append("contains errors/diagnostics")
    header = " | ".join(header_parts) + "]"

    if max_bytes < 260:
        err_hint = f" Errors: {errors[0]}" if errors else ""
        return clip_utf8_middle(f"{header}{err_hint}", max_bytes)

    sections: List[str] = [header]
    err_section = ""
    if errors:
        err_section = "Key diagnostics:\n" + "\n".join(f"  ! {e}" for e in errors)

    footer = (
        "[Full output condensed to fit context budget; re-read a targeted slice/range if exact lines are needed]"
    )
    overhead = utf8_len(header) + utf8_len(err_section) + utf8_len(footer) + 16
    preview_budget = max(0, max_bytes - overhead)

    if preview_budget >= 200 and not aggressive:
        head_budget = int(preview_budget * 0.55)
        tail_budget = preview_budget - head_budget
        head = clip_utf8_middle(text, head_budget, head_ratio=1.0) if head_budget > 80 else ""
        # Take clean head lines and tail lines directly instead of double markers
        head_lines: List[str] = []
        used = 0
        for ln in lines:
            ln_cost = utf8_len(ln) + 1
            if used + ln_cost > head_budget:
                break
            head_lines.append(ln)
            used += ln_cost
        tail_lines_rev: List[str] = []
        used_t = 0
        for ln in reversed(lines[len(head_lines) :]):
            ln_cost = utf8_len(ln) + 1
            if used_t + ln_cost > tail_budget:
                break
            tail_lines_rev.append(ln)
            used_t += ln_cost
        tail_lines = list(reversed(tail_lines_rev))
        if head_lines:
            sections.append("Head:\n" + "\n".join(head_lines))
        if err_section:
            sections.append(err_section)
        if tail_lines:
            sections.append("Tail:\n" + "\n".join(tail_lines))
        if not head_lines and not tail_lines and head:
            sections.append(head)
    else:
        if err_section:
            sections.append(err_section)
        if preview_budget >= 120:
            snippet = clip_utf8_middle(text, preview_budget, head_ratio=0.5)
            sections.append(snippet)

    sections.append(footer)
    combined = "\n".join(s for s in sections if s)
    if utf8_len(combined) > max_bytes:
        return clip_utf8_middle(combined, max_bytes, head_ratio=0.6)
    return combined


# --- Atomic Turn-Unit Grouping & Pi `findCutPoint` ---


def _group_into_turn_units(messages: Sequence[ChatMessage]) -> List[_TurnUnit]:
    """Group messages into atomic units so tool_call <-> tool_result pairs are NEVER split."""
    units: List[_TurnUnit] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        # Assistant with tool calls + all immediately following tool_result turns = 1 atomic unit!
        if msg.role == "assistant" and msg.tool_calls:
            end = i + 1
            while end < n and _is_pure_tool_result_message(messages[end]):
                end += 1
            # Also attach a mixed user message that carries tool_results for this assistant call
            # ONLY if it has no human text (already handled by _is_pure_tool_result_message).
            # If the next user message has BOTH tool_results AND human text, group it as part
            # of the exchange so tool_results are never orphaned, while marking human start.
            if end < n and messages[end].role == "user" and messages[end].tool_results:
                end += 1
            units.append(
                _TurnUnit(
                    start_idx=i,
                    end_idx=end,
                    kind="tool_exchange",
                    messages=list(messages[i:end]),
                    is_human_turn_start=False,
                )
            )
            i = end
            continue

        if _is_pure_tool_result_message(msg):
            # Orphan tool result (e.g., client already trimmed earlier assistant message):
            # group contiguous orphan tool results together and avoid cutting directly onto them
            # if there is a preceding assistant message.
            end = i + 1
            while end < n and _is_pure_tool_result_message(messages[end]):
                end += 1
            units.append(
                _TurnUnit(
                    start_idx=i,
                    end_idx=end,
                    kind="orphan_tool",
                    messages=list(messages[i:end]),
                    is_human_turn_start=False,
                )
            )
            i = end
            continue

        if _is_genuine_human_message(msg):
            units.append(
                _TurnUnit(
                    start_idx=i,
                    end_idx=i + 1,
                    kind="human",
                    messages=[msg],
                    is_human_turn_start=True,
                )
            )
            i += 1
            continue

        units.append(
            _TurnUnit(
                start_idx=i,
                end_idx=i + 1,
                kind="assistant",
                messages=[msg],
                is_human_turn_start=False,
            )
        )
        i += 1
    return units


def _format_single_message(msg: ChatMessage) -> str:
    """Render one ChatMessage identically to gateway conversation format."""
    if msg.tool_call_id:
        role_label = f"TOOL({msg.tool_call_id})"
    elif msg.role == "tool":
        role_label = f"TOOL({msg.name})" if msg.name else "TOOL"
    else:
        role_label = msg.role.upper()

    body_parts: List[str] = []
    if msg.content:
        body_parts.append(msg.content)
    if msg.tool_calls:
        for tc in msg.tool_calls:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") if "arguments" in fn else tc.get("arguments", {})
            # Preserve dict arguments cleanly when parsed from Anthropic/Gemini
            parsed_args: Any = raw_args
            if isinstance(raw_args, str):
                with contextlib.suppress(json.JSONDecodeError):
                    parsed_args = json.loads(raw_args)
            call_obj = {
                "name": fn.get("name") or tc.get("name"),
                "arguments": parsed_args if parsed_args is not None else {},
            }
            body_parts.append(
                f"<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>"
            )
    if msg.tool_results:
        for tr in msg.tool_results:
            if tr.tool_call_id:
                tag = f"[tool_result id={tr.tool_call_id}]"
            elif tr.name:
                tag = f"[tool_result name={tr.name}]"
            else:
                tag = "[tool_result]"
            if tr.is_error:
                tag = f"{tag} (ERROR)"
            body_parts.append(f"{tag}\n{tr.content}".strip())

    body = "\n".join(p for p in body_parts if p).strip()
    return f"[{role_label}]\n{body}" if body else ""


def _render_unit(unit: _TurnUnit) -> str:
    rendered = [_format_single_message(m) for m in unit.messages]
    return "\n\n".join(r for r in rendered if r)


def _find_cut_point(
    units: Sequence[_TurnUnit],
    unit_sizes: Sequence[int],
    keep_recent_bytes: int,
) -> _CutPointResult:
    """Pi-inspired `findCutPoint` walking backward over atomic units.

    Guarantees:
    - Never cuts inside a tool_call <-> tool_result exchange.
    - Never cuts directly on an orphan_tool unit if a cleaner boundary exists.
    - Always keeps at least the newest atomic unit.
    - Detects `is_split_turn` and locates the originating human user message (`turn_start_message_index`).
    """
    n = len(units)
    if n <= 1:
        return _CutPointResult(
            cut_unit_index=0,
            cut_message_index=0,
            is_split_turn=False,
            turn_start_message_index=None,
        )

    accumulated = 0
    cut_unit = 0
    # Always keep at least the newest unit (n - 1).
    accumulated += unit_sizes[n - 1]
    cut_unit = n - 1

    for idx in range(n - 2, -1, -1):
        cost = unit_sizes[idx] + 2
        if accumulated + cost <= keep_recent_bytes:
            accumulated += cost
            cut_unit = idx
        else:
            break

    # Avoid starting the kept tail on an orphan_tool unit if we can step back or forward cleanly
    if cut_unit > 0 and units[cut_unit].kind == "orphan_tool":
        # Prefer including the preceding assistant/human unit if small, else advance past orphan
        if cut_unit + 1 < n:
            cut_unit += 1

    # Detect Pi's `isSplitTurn`: did we cut in the middle of a multi-step turn?
    is_split = False
    turn_start_msg: Optional[int] = None
    if cut_unit > 0 and not units[cut_unit].is_human_turn_start:
        # Walk back to find the human user turn that initiated this ongoing agentic loop
        for back in range(cut_unit - 1, -1, -1):
            if units[back].is_human_turn_start:
                is_split = True
                turn_start_msg = units[back].start_idx
                break

    return _CutPointResult(
        cut_unit_index=cut_unit,
        cut_message_index=units[cut_unit].start_idx,
        is_split_turn=is_split,
        turn_start_message_index=turn_start_msg,
    )


# --- Observation Masking & Duplicate Tool-Output Superseding ---


def _apply_observation_masking(
    messages: Sequence[ChatMessage],
    *,
    tool_index: Dict[str, _ToolCallMeta],
    protected_tail_start: int,
    stale_tool_max_bytes: int,
    latest_tool_max_bytes: int,
    aggressive: bool = False,
) -> Tuple[List[ChatMessage], int, int]:
    """Supersede duplicate older file/dir reads and mask oversized/stale tool outputs.

    Returns (transformed_messages, masked_count, deduped_count).
    """
    # 1. Identify the latest turn index for every deduplicable read/list target
    latest_target_turn: Dict[Tuple[str, str], int] = {}
    for idx, msg in enumerate(messages):
        if msg.role == "tool" or msg.tool_call_id:
            meta = tool_index.get(msg.tool_call_id or "")
            if meta and meta.kind in ("read", "list") and meta.target:
                latest_target_turn[(meta.kind, meta.target)] = idx
        for tr in msg.tool_results:
            meta = tool_index.get(tr.tool_call_id)
            if meta and meta.kind in ("read", "list") and meta.target:
                latest_target_turn[(meta.kind, meta.target)] = idx

    masked_count = 0
    deduped_count = 0
    transformed: List[ChatMessage] = []
    n = len(messages)

    for idx, msg in enumerate(messages):
        is_in_protected_tail = idx >= protected_tail_start
        is_very_last = idx == n - 1

        # Strip stale <thinking> blocks from historical assistant turns
        content = msg.content
        if msg.role == "assistant" and not is_in_protected_tail and content:
            content = _strip_stale_thinking(content)

        # Case A: OpenAI-style standalone `role="tool"` message
        if msg.role == "tool" or msg.tool_call_id:
            meta = tool_index.get(msg.tool_call_id or "")
            t_name = (meta.name if meta else "") or (msg.name or "")
            t_target = meta.target if meta else ""
            raw_bytes = utf8_len(content)

            if (
                meta
                and meta.kind in ("read", "list")
                and meta.target
                and latest_target_turn.get((meta.kind, meta.target), -1) > idx
                and raw_bytes > 512
            ):
                newer_idx = latest_target_turn[(meta.kind, meta.target)]
                content = (
                    f"[Superseded {t_name}({t_target}): {raw_bytes:,} bytes — "
                    f"updated output appears later at message #{newer_idx + 1}]"
                )
                deduped_count += 1
            else:
                cap = latest_tool_max_bytes if is_very_last else (
                    stale_tool_max_bytes * 2 if is_in_protected_tail else stale_tool_max_bytes
                )
                if raw_bytes > cap:
                    content = summarize_tool_output_deterministic(
                        content,
                        cap,
                        tool_name=t_name,
                        target_hint=t_target,
                        aggressive=aggressive and not is_in_protected_tail,
                    )
                    masked_count += 1

            transformed.append(
                ChatMessage(
                    role=msg.role,
                    content=content,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    tool_calls=msg.tool_calls,
                    tool_results=msg.tool_results,
                )
            )
            continue

        # Case B: Anthropic / Gemini structured `tool_results` inside a user message
        if msg.tool_results:
            new_results: List[ToolResultItem] = []
            for tr in msg.tool_results:
                meta = tool_index.get(tr.tool_call_id)
                t_name = (meta.name if meta else "") or tr.name
                t_target = meta.target if meta else ""
                tr_text = tr.content
                raw_bytes = utf8_len(tr_text)

                if (
                    meta
                    and meta.kind in ("read", "list")
                    and meta.target
                    and latest_target_turn.get((meta.kind, meta.target), -1) > idx
                    and raw_bytes > 512
                ):
                    newer_idx = latest_target_turn[(meta.kind, meta.target)]
                    tr_text = (
                        f"[Superseded {t_name}({t_target}): {raw_bytes:,} bytes — "
                        f"updated output appears later at message #{newer_idx + 1}]"
                    )
                    deduped_count += 1
                else:
                    # Share budget across multiple tool results in the same message
                    per_item_divisor = max(1, len(msg.tool_results))
                    base_cap = (
                        max(2048, latest_tool_max_bytes // per_item_divisor)
                        if is_very_last
                        else (
                            max(2048, (stale_tool_max_bytes * 2) // per_item_divisor)
                            if is_in_protected_tail
                            else max(1024, stale_tool_max_bytes // per_item_divisor)
                        )
                    )
                    if raw_bytes > base_cap:
                        tr_text = summarize_tool_output_deterministic(
                            tr_text,
                            base_cap,
                            tool_name=t_name,
                            target_hint=t_target,
                            is_error=tr.is_error,
                            aggressive=aggressive and not is_in_protected_tail,
                        )
                        masked_count += 1

                new_results.append(
                    ToolResultItem(
                        tool_call_id=tr.tool_call_id,
                        name=tr.name,
                        content=tr_text,
                        is_error=tr.is_error,
                    )
                )

            transformed.append(
                ChatMessage(
                    role=msg.role,
                    content=content,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    tool_calls=msg.tool_calls,
                    tool_results=new_results,
                )
            )
            continue

        # Case C: Plain text message (user / assistant)
        if content != msg.content:
            transformed.append(
                ChatMessage(
                    role=msg.role,
                    content=content,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    tool_calls=msg.tool_calls,
                    tool_results=msg.tool_results,
                )
            )
        else:
            transformed.append(msg)

    return transformed, masked_count, deduped_count


# --- Deterministic Algorithmic Structured Checkpoint (Pi Handoff + Paseo Material Progress) ---


def build_deterministic_checkpoint(
    prefix_messages: Sequence[ChatMessage],
    *,
    tool_index: Dict[str, _ToolCallMeta],
    max_bytes: int,
    active_turn_user_msg: Optional[ChatMessage] = None,
    active_turn_index: Optional[int] = None,
    cached_llm_summary: Optional[str] = None,
) -> str:
    """Build a structured Pi/Paseo-style durable checkpoint from compacted prefix turns."""
    total_compacted = len(prefix_messages)
    header = (
        f"[Compacted Conversation Checkpoint — {total_compacted} earlier message(s) condensed; "
        "most recent turns preserved verbatim below]"
    )

    if cached_llm_summary and cached_llm_summary.strip():
        parts = [header, cached_llm_summary.strip()]
        if active_turn_user_msg and active_turn_user_msg.content.strip():
            turn_label = f" (message #{active_turn_index + 1})" if active_turn_index is not None else ""
            clean_active = extract_clean_human_intent(active_turn_user_msg.content)
            parts.append(
                f"## Active In-Progress User Request{turn_label}\n"
                + clip_utf8_middle(clean_active, min(4000, max_bytes // 3))
            )
        combined = "\n\n".join(parts)
        return clip_utf8_middle(combined, max_bytes, head_ratio=0.65)

    # Extract durable structured state from prefix_messages
    human_msgs: List[Tuple[int, str]] = []
    assistant_milestones: List[Tuple[int, str]] = []
    files_read: List[str] = []
    files_written: List[str] = []
    commands_run: List[str] = []
    tool_counts: Counter[str] = Counter()
    open_errors: List[str] = []
    seen_read: set[str] = set()
    seen_write: set[str] = set()
    seen_cmds: set[str] = set()
    seen_errs: set[str] = set()

    for idx, msg in enumerate(prefix_messages):
        if _is_genuine_human_message(msg) and msg.content.strip():
            clean_user = extract_clean_human_intent(msg.content)
            if clean_user:
                human_msgs.append((idx + 1, clean_user))
        elif msg.role == "assistant":
            clean_txt = _normalize_ephemeral_media_paths(_strip_stale_thinking(msg.content)).strip()
            # Filter out leaked raw tool wrapper blobs from milestones
            if (
                clean_txt
                and not clean_txt.startswith(("[USER]", "[TOOL", "[tool_result", "<tool_call>"))
            ):
                assistant_milestones.append((idx + 1, clean_txt))
            for tc in msg.tool_calls:
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                t_name = str(fn.get("name") or tc.get("name") or "tool")
                tool_counts[t_name] += 1
                args = _safe_parse_args(
                    fn.get("arguments") if "arguments" in fn else tc.get("arguments")
                )
                kind, target = _extract_tool_target(t_name, args)
                if kind == "read" and target:
                    # Strip ONLY trailing `:start-end` range; preserve Windows drive letters like `D:\...`!
                    base_path = _TRAILING_LINE_RANGE_RE.sub("", target)
                    if base_path and base_path not in seen_read:
                        seen_read.add(base_path)
                        files_read.append(base_path)
                elif kind == "write" and target:
                    if target not in seen_write:
                        seen_write.add(target)
                        files_written.append(target)
                elif kind == "exec" and target:
                    if target not in seen_cmds:
                        seen_cmds.add(target)
                        commands_run.append(target)

        # Scan tool outputs & errors for file paths and diagnostics
        result_texts: List[Tuple[str, bool]] = []
        if msg.role == "tool" or msg.tool_call_id:
            result_texts.append((msg.content, False))
        for tr in msg.tool_results:
            result_texts.append((tr.content, tr.is_error))

        for r_text, is_err in result_texts:
            if not r_text:
                continue
            errs = _extract_error_lines(r_text, max_lines=3, max_line_chars=160)
            if is_err and not errs:
                errs = [_one_line(r_text, 160)]
            for e in errs:
                if e not in seen_errs:
                    seen_errs.add(e)
                    open_errors.append(f"[msg #{idx + 1}] {e}")

        # Also harvest explicit file path mentions from early human/assistant text if tool metadata was absent
        if not msg.tool_calls and len(files_read) < 20 and msg.content:
            for match in _FILE_PATH_RE.finditer(msg.content[:2000]):
                fp = match.group(1)
                if fp not in seen_read and fp not in seen_write and "/" in fp:
                    seen_read.add(fp)
                    files_read.append(fp)
                    if len(files_read) >= 20:
                        break

    sections: List[str] = [header]

    # 1. Goal & Original Request (Turn-0 anchor + follow-up user directives)
    skip_separate_active = (
        active_turn_user_msg is not None
        and active_turn_index is not None
        and len(human_msgs) == 1
        and human_msgs[0][0] == active_turn_index + 1
    )
    if human_msgs:
        goal_lines: List[str] = []
        first_idx, first_text = human_msgs[0]
        primary_budget = min(2800, max(500, max_bytes // 4))
        goal_lines.append(
            f"- Primary goal (msg #{first_idx}): {clip_utf8_middle(first_text, primary_budget)}"
        )
        if len(human_msgs) > 1:
            followups = human_msgs[1:]
            # Keep first 2 + last 6 follow-up human directives so user constraints are never lost
            if len(followups) > 8:
                selected = followups[:2] + [(-1, f"... ({len(followups) - 8} intermediate user messages) ...")] + followups[-6:]
            else:
                selected = followups
            goal_lines.append("- Follow-up user directives & constraints:")
            for f_idx, f_text in selected:
                if f_idx == -1:
                    goal_lines.append(f"  - {f_text}")
                else:
                    goal_lines.append(f"  - [msg #{f_idx}] {_one_line(f_text, 360)}")
        heading = "## Goal & Active Turn Request" if skip_separate_active else "## Goal & User Directives"
        sections.append(f"{heading}\n" + "\n".join(goal_lines))

    # 2. Active Turn Context (Pi `isSplitTurn` safeguard)
    if not skip_separate_active and active_turn_user_msg and active_turn_user_msg.content.strip():
        t_lbl = f" (msg #{active_turn_index + 1})" if active_turn_index is not None else ""
        active_budget = min(3200, max(600, max_bytes // 4))
        clean_active = extract_clean_human_intent(active_turn_user_msg.content)
        sections.append(
            f"## Active In-Progress Turn Request{t_lbl}\n"
            + clip_utf8_middle(clean_active, active_budget)
        )

    # 3. Material Progress & Touched Files (Paseo Material Progress + Pi-VCC)
    prog_lines: List[str] = []
    if files_written:
        prog_lines.append(
            "- Files written/edited: " + ", ".join(f"`{p}`" for p in files_written[:25])
            + (f" (+{len(files_written) - 25} more)" if len(files_written) > 25 else "")
        )
    if files_read:
        only_read = [p for p in files_read if p not in seen_write]
        if only_read:
            prog_lines.append(
                "- Files inspected: " + ", ".join(f"`{p}`" for p in only_read[:25])
                + (f" (+{len(only_read) - 25} more)" if len(only_read) > 25 else "")
            )
    if commands_run:
        prog_lines.append(
            "- Recent commands run:\n"
            + "\n".join(f"  - `{c}`" for c in commands_run[-10:])
        )
    if tool_counts:
        top_tools = ", ".join(f"{name}×{cnt}" for name, cnt in tool_counts.most_common(12))
        prog_lines.append(f"- Completed tool calls ({sum(tool_counts.values())} total): {top_tools}")
    if prog_lines:
        sections.append("## Material Progress & Touched Files\n" + "\n".join(prog_lines))

    # 4. Key Assistant Decisions & Milestones
    if assistant_milestones:
        # Keep earliest milestone + last 8 milestones so both initial plan and latest progress survive
        if len(assistant_milestones) > 9:
            selected_a = [assistant_milestones[0]] + assistant_milestones[-8:]
        else:
            selected_a = assistant_milestones
        mile_lines = [f"- [msg #{a_idx}] {_one_line(a_text, 340)}" for a_idx, a_text in selected_a]
        sections.append("## Key Decisions & Assistant Milestones\n" + "\n".join(mile_lines))

    # 5. Open Errors / Blockers Detected
    if open_errors:
        err_lines = [f"- {e}" for e in open_errors[-8:]]
        sections.append("## Recent Errors / Diagnostics in Compacted Turns\n" + "\n".join(err_lines))

    combined = "\n\n".join(sections)
    if utf8_len(combined) > max_bytes:
        return clip_utf8_middle(combined, max_bytes, head_ratio=0.65)
    return combined


# --- Fingerprinting & Durable SQLite Checkpoint Cache ---


def compute_session_fingerprint(req: NormalizedRequest) -> str:
    """Compute a stable account-agnostic fingerprint identifying a multi-turn session.

    Uses `extract_clean_human_intent` so harnesses injecting identical `<system-reminder>`
    or `SOUL.md` blocks at the top of msg #1 do not collide across separate sessions.
    """
    if req.session_id and req.session_id.strip():
        return hashlib.sha256(f"sid:{req.session_id.strip()}".encode("utf-8")).hexdigest()[:24]

    sys_head = (req.system_prompt or "")[:1024]
    first_human = ""
    for msg in req.messages:
        if _is_genuine_human_message(msg) and msg.content.strip():
            first_human = extract_clean_human_intent(msg.content)[:1024]
            break
    if not first_human and req.messages:
        first_human = extract_clean_human_intent(req.messages[0].content)[:1024]

    raw = f"sys:{sys_head}\nfirst:{first_human}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def compute_prefix_hash(prefix_messages: Sequence[ChatMessage]) -> str:
    """Compute a deterministic content hash independent of ephemeral media staging UUIDs."""
    h = hashlib.sha256()
    for msg in prefix_messages:
        h.update(msg.role.encode("utf-8"))
        h.update(b"\x00")
        h.update((msg.tool_call_id or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(
            _normalize_ephemeral_media_paths(msg.content).encode("utf-8", errors="ignore")
        )
        h.update(b"\x01")
        if msg.tool_calls:
            h.update(json.dumps(msg.tool_calls, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\x02")
        for tr in msg.tool_results:
            h.update(tr.tool_call_id.encode("utf-8"))
            h.update(b":")
            h.update(
                _normalize_ephemeral_media_paths(tr.content).encode("utf-8", errors="ignore")
            )
            h.update(b"\x03")
    return h.hexdigest()[:32]


class ContextCheckpointStore:
    """Thread-safe PostgreSQL/SQLite cache for compacted session checkpoints."""

    def __init__(
        self,
        db_path: Path | str = "./data/gateway.db",
        ttl_hours: int = 24,
        max_entries: int = 500,
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
                CREATE TABLE IF NOT EXISTS context_checkpoints (
                    session_fingerprint TEXT NOT NULL,
                    prefix_hash TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'llm',
                    source_messages INTEGER NOT NULL DEFAULT 0,
                    source_bytes BIGINT NOT NULL DEFAULT 0,
                    created_at DOUBLE PRECISION NOT NULL,
                    last_used_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (session_fingerprint, prefix_hash)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ctx_cp_last_used
                ON context_checkpoints (last_used_at)
                """
            )

    def get_with_mode(
        self, session_fingerprint: str, prefix_hash: str
    ) -> Optional[Tuple[str, str]]:
        """Return (summary_text, mode) for an exact cached checkpoint if not expired."""
        now = time.time()
        cutoff = now - self.ttl_seconds
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT summary_text, mode, last_used_at FROM context_checkpoints
                WHERE session_fingerprint = ? AND prefix_hash = ? AND last_used_at >= ?
                """,
                (session_fingerprint, prefix_hash, cutoff),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE context_checkpoints SET last_used_at = ?
                WHERE session_fingerprint = ? AND prefix_hash = ?
                """,
                (now, session_fingerprint, prefix_hash),
            )
            return str(row["summary_text"]), str(row["mode"])

    def get(self, session_fingerprint: str, prefix_hash: str) -> Optional[str]:
        found = self.get_with_mode(session_fingerprint, prefix_hash)
        return found[0] if found is not None else None

    def get_latest_for_session(
        self, session_fingerprint: str
    ) -> Optional[Tuple[str, str, int, str]]:
        """Return (prefix_hash, summary_text, source_messages, mode) for the freshest checkpoint."""
        now = time.time()
        cutoff = now - self.ttl_seconds
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT prefix_hash, summary_text, source_messages, mode
                FROM context_checkpoints
                WHERE session_fingerprint = ? AND last_used_at >= ?
                ORDER BY last_used_at DESC
                LIMIT 1
                """,
                (session_fingerprint, cutoff),
            ).fetchone()
            if row is None:
                return None
            return (
                str(row["prefix_hash"]),
                str(row["summary_text"]),
                int(row["source_messages"]),
                str(row["mode"]),
            )

    def put(
        self,
        session_fingerprint: str,
        prefix_hash: str,
        summary_text: str,
        *,
        mode: str = "llm",
        source_messages: int = 0,
        source_bytes: int = 0,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO context_checkpoints (
                    session_fingerprint, prefix_hash, summary_text, mode,
                    source_messages, source_bytes, created_at, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_fingerprint, prefix_hash) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    mode = excluded.mode,
                    source_messages = excluded.source_messages,
                    source_bytes = excluded.source_bytes,
                    last_used_at = excluded.last_used_at
                """,
                (
                    session_fingerprint,
                    prefix_hash,
                    summary_text,
                    mode,
                    source_messages,
                    source_bytes,
                    now,
                    now,
                ),
            )
            # Cap table size via LRU eviction (portable across PostgreSQL & SQLite)
            count_row = conn.execute("SELECT COUNT(*) AS c FROM context_checkpoints").fetchone()
            total = int(count_row["c"]) if count_row else 0
            if total > self.max_entries:
                excess = total - self.max_entries
                conn.execute(
                    """
                    DELETE FROM context_checkpoints
                    WHERE (session_fingerprint, prefix_hash) IN (
                        SELECT session_fingerprint, prefix_hash
                        FROM context_checkpoints
                        ORDER BY last_used_at ASC
                        LIMIT ?
                    )
                    """,
                    (excess,),
                )

    def prune_expired(self) -> int:
        cutoff = time.time() - self.ttl_seconds
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM context_checkpoints WHERE last_used_at < ?",
                (cutoff,),
            )
            return int(cur.rowcount or 0)

    def clear(self) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM context_checkpoints")
            return int(cur.rowcount or 0)

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM context_checkpoints").fetchone()
            return int(row["c"]) if row else 0


class ContextStatsTracker:
    """Thread-safe runtime telemetry for long-session context protection."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests = 0
        self.compacted_requests = 0
        self.total_bytes_saved = 0
        self.overflow_recoveries = 0
        self.overflow_failures = 0
        self.masked_tool_results = 0
        self.deduped_tool_results = 0
        self.split_turn_compactions = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_summaries_generated = 0
        self.llm_summary_failures = 0
        self.session_reuse_hits = 0
        self.session_reuse_misses = 0
        self.session_reuse_fallbacks = 0
        self.last_compaction_stage = "none"

    def record_build(self, res: ContextBuildResult) -> None:
        with self._lock:
            self.total_requests += 1
            if res.session_reused:
                self.session_reuse_hits += 1
                self.total_bytes_saved += max(0, res.original_bytes - res.final_bytes)
                self.last_compaction_stage = res.compaction_stage
            elif res.compacted:
                self.compacted_requests += 1
                self.total_bytes_saved += max(0, res.original_bytes - res.final_bytes)
                self.last_compaction_stage = res.compaction_stage
            self.masked_tool_results += res.masked_tool_results
            self.deduped_tool_results += res.deduped_tool_results
            if res.split_turn:
                self.split_turn_compactions += 1
            if res.compacted_messages > 0:
                if res.cache_hit:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1

    def record_session_miss(self) -> None:
        with self._lock:
            self.session_reuse_misses += 1

    def record_session_fallback(self) -> None:
        with self._lock:
            self.session_reuse_fallbacks += 1

    def record_overflow_recovery(self) -> None:
        with self._lock:
            self.overflow_recoveries += 1

    def record_overflow_failure(self) -> None:
        with self._lock:
            self.overflow_failures += 1

    def record_llm_summary(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self.llm_summaries_generated += 1
            else:
                self.llm_summary_failures += 1

    def snapshot(
        self,
        *,
        mode: str,
        cached_checkpoints: int = 0,
        session_reuse_enabled: bool = True,
        active_sessions: int = 0,
    ) -> Dict[str, Any]:
        with self._lock:
            return {
                "compaction_mode": mode,
                "session_reuse_enabled": session_reuse_enabled,
                "active_sessions": active_sessions,
                "session_reuse_hits": self.session_reuse_hits,
                "session_reuse_misses": self.session_reuse_misses,
                "session_reuse_fallbacks": self.session_reuse_fallbacks,
                "total_requests": self.total_requests,
                "compacted_requests": self.compacted_requests,
                "total_bytes_saved": self.total_bytes_saved,
                "overflow_recoveries": self.overflow_recoveries,
                "overflow_failures": self.overflow_failures,
                "masked_tool_results": self.masked_tool_results,
                "deduped_tool_results": self.deduped_tool_results,
                "split_turn_compactions": self.split_turn_compactions,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cached_checkpoints": cached_checkpoints,
                "llm_summaries_generated": self.llm_summaries_generated,
                "llm_summary_failures": self.llm_summary_failures,
                "last_compaction_stage": self.last_compaction_stage,
            }


# --- Deadlock-Free Bounded Chronological Map/Reduce LLM Summarizer ---


def _chunk_text_bounded(text: str, chunk_bytes: int) -> List[str]:
    """Split a serialized transcript into UTF-8 chunks strictly <= chunk_bytes on line boundaries."""
    if utf8_len(text) <= chunk_bytes:
        return [text]
    chunks: List[str] = []
    current_lines: List[str] = []
    current_bytes = 0
    for line in text.splitlines():
        line_bytes = utf8_len(line) + 1
        if line_bytes > chunk_bytes:
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_bytes = 0
            chunks.append(clip_utf8_middle(line, chunk_bytes))
            continue
        if current_bytes + line_bytes > chunk_bytes and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_bytes = line_bytes
        else:
            current_lines.append(line)
            current_bytes += line_bytes
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


async def summarize_prefix_map_reduce(
    prefix_text: str,
    llm_complete: Callable[[str], Awaitable[str]],
    *,
    chunk_bytes: int = 64_000,
    max_summary_bytes: int = 16_000,
    max_chunks: int = 6,
) -> str:
    """Deadlock-free iterative bounded map/reduce summarization (solves DSH overflow deadlock).

    Each LLM call is strictly bounded well below the physical `USER_INPUT` limit, folding the
    prior checkpoint summary into the next chronological slice (Pi iterative chaining).
    """
    raw_chunks = _chunk_text_bounded(prefix_text, chunk_bytes=chunk_bytes)
    if len(raw_chunks) > max_chunks:
        # Preserve earliest goal chunk + newest chunks to keep latency & quota bounded
        kept_chunks = [raw_chunks[0]] + raw_chunks[-(max_chunks - 1) :]
    else:
        kept_chunks = raw_chunks

    rolling_summary = ""
    for idx, chunk in enumerate(kept_chunks):
        if rolling_summary:
            prev_block = clip_utf8_middle(rolling_summary, min(max_summary_bytes, 12_000))
            prompt = (
                f"[System Instructions]\n{SUMMARIZATION_SYSTEM_PROMPT}\n\n"
                f"Update the existing handoff checkpoint with the next chronological conversation slice "
                f"(part {idx + 1}/{len(kept_chunks)}). Preserve all still-relevant goals, constraints, "
                f"file paths, decisions, and unresolved errors.\n\n"
                f"{SUMMARIZATION_TEMPLATE}\n\n"
                f"[Previous Checkpoint]\n{prev_block}\n\n"
                f"[New Conversation Slice]\n{chunk}"
            )
        else:
            prompt = (
                f"[System Instructions]\n{SUMMARIZATION_SYSTEM_PROMPT}\n\n"
                f"{SUMMARIZATION_TEMPLATE}\n\n"
                f"[Conversation Slice (part {idx + 1}/{len(kept_chunks)})]\n{chunk}"
            )
        summary = (await llm_complete(prompt)).strip()
        if summary:
            rolling_summary = clip_utf8_middle(summary, max_summary_bytes)
    return rolling_summary


# --- Main Progressive Context Compiler ---


def _substitute_media_paths(text: str, staged_media: Sequence[StagedMedia]) -> str:
    if not staged_media or not text:
        return text
    result = text
    for item in staged_media:
        result = result.replace(media_placeholder(item.index), str(item.path))
    return result


def _substitute_media_in_message(msg: ChatMessage, staged_media: Sequence[StagedMedia]) -> ChatMessage:
    if not staged_media:
        return msg
    new_content = _substitute_media_paths(msg.content, staged_media)
    new_results = [
        ToolResultItem(
            tool_call_id=tr.tool_call_id,
            name=tr.name,
            content=_substitute_media_paths(tr.content, staged_media),
            is_error=tr.is_error,
        )
        for tr in msg.tool_results
    ]
    return ChatMessage(
        role=msg.role,
        content=new_content,
        name=msg.name,
        tool_call_id=msg.tool_call_id,
        tool_calls=msg.tool_calls,
        tool_results=new_results,
    )


def _build_policy_section(
    active_media: Sequence[StagedMedia],
    archived_media: Sequence[StagedMedia] = (),
) -> str:
    if active_media:
        manifest_lines = [
            f"- #{m.index} ({m.kind}, {m.mime_type}): {m.path}" for m in active_media
        ]
        archived_note = ""
        if archived_media:
            archived_ids = ", ".join(f"#{m.index} ({m.kind})" for m in archived_media)
            archived_note = (
                f"\nEarlier compacted turns also referenced archived attachment(s) {archived_ids}; "
                "you do NOT need to re-read archived attachments unless explicitly requested."
            )
        return (
            "[Gateway Execution Policy]\n"
            "Answer directly as the requested model. The user provided the following local media "
            "attachment(s) staged exclusively for this request:\n"
            + "\n".join(manifest_lines)
            + "\nYou MUST call the built-in `read_file` tool on each attachment path listed above "
            "BEFORE answering so you can directly inspect the image, video, audio, or document contents. "
            "Do NOT invoke any other Antigravity built-in workspace, filesystem, terminal, or browser tools "
            "(only `read_file` on the exact attachment paths listed above). If external harness tools are "
            "listed below, use <tool_call> blocks for them after inspecting the media."
            + archived_note
        )
    return (
        "[Gateway Execution Policy]\n"
        "Answer directly as the requested model. Do NOT invoke Antigravity built-in workspace, "
        "filesystem, terminal, or browser tools; if external harness tools are listed below, "
        "use <tool_call> blocks instead."
    )


def _find_referenced_media(
    texts: Sequence[str],
    staged_media: Sequence[StagedMedia],
) -> Tuple[List[StagedMedia], List[StagedMedia]]:
    if not staged_media:
        return [], []
    joined = "\n".join(texts)
    active: List[StagedMedia] = []
    archived: List[StagedMedia] = []
    for item in staged_media:
        if str(item.path) in joined or media_placeholder(item.index) in joined:
            active.append(item)
        else:
            archived.append(item)
    return active, archived


class ContextManager:
    """Progressive multi-stage context manager preventing abnormal long-session truncation."""

    def __init__(
        self,
        *,
        mode: str = "deterministic",
        keep_recent_bytes: int = 96_000,
        checkpoint_max_bytes: int = 24_000,
        tool_output_max_bytes: int = 12_000,
        checkpoint_store: Optional[ContextCheckpointStore] = None,
        stats: Optional[ContextStatsTracker] = None,
    ) -> None:
        self.mode = mode if mode in VALID_COMPACTION_MODES else "deterministic"
        self.keep_recent_bytes = max(8_000, int(keep_recent_bytes))
        self.checkpoint_max_bytes = max(2_000, int(checkpoint_max_bytes))
        self.tool_output_max_bytes = max(1_024, int(tool_output_max_bytes))
        self.checkpoint_store = checkpoint_store
        self.stats = stats or ContextStatsTracker()

    def build_context(
        self,
        req: NormalizedRequest,
        *,
        max_bytes: int = MAX_PROMPT_BYTES,
        staged_media: Optional[Sequence[StagedMedia]] = None,
        override_summary: Optional[str] = None,
        is_overflow_recovery: bool = False,
        record_stats: bool = True,
    ) -> ContextBuildResult:
        """Compile a NormalizedRequest into a strictly bounded, continuity-preserving prompt."""
        media_list = list(staged_media or [])
        session_fp = compute_session_fingerprint(req)
        sub_messages = [_substitute_media_in_message(m, media_list) for m in req.messages]
        raw_turns = [t for t in (_format_single_message(m) for m in sub_messages) if t]

        sys_prompt = (
            _substitute_media_paths(req.system_prompt.strip(), media_list)
            if req.system_prompt and req.system_prompt.strip()
            else ""
        )

        # Compute baseline uncompressed payload size for observability & Stage 0 fast path
        full_policy = _build_policy_section(media_list)
        fast_sections: List[str] = [full_policy]
        if sys_prompt:
            fast_sections.append(f"[System Instructions]\n{sys_prompt}")
        if req.tools:
            fast_sections.append(format_compact_tools(req.tools, max_bytes=max_bytes))

        if raw_turns:
            fast_candidate = (
                "\n\n".join(fast_sections) + "\n\n[Conversation]\n" + "\n\n".join(raw_turns)
            )
        else:
            fast_candidate = "\n\n".join(fast_sections)

        original_bytes = utf8_len(fast_candidate)
        original_tokens = estimate_tokens(fast_candidate)

        # Stage 0: Fast Path — zero compaction needed!
        if original_bytes <= max_bytes and not is_overflow_recovery:
            res = ContextBuildResult(
                prompt=fast_candidate,
                original_bytes=original_bytes,
                final_bytes=original_bytes,
                original_tokens_est=original_tokens,
                final_tokens_est=original_tokens,
                compacted=False,
                compaction_stage="none",
                compaction_mode=self.mode,
                total_messages=len(req.messages),
                kept_messages=len(req.messages),
                compacted_messages=0,
                session_fingerprint=session_fp,
                active_media_indices=tuple(m.index for m in media_list),
            )
            if record_stats:
                self.stats.record_build(res)
            return res

        # --- Progressive Compaction Path (Stages 1 -> 4) ---
        # Stage 1: Adaptive budgets for System Instructions & Compact Tools
        # Give conversation priority when huge instead of rigidly starving it.
        has_tools = bool(req.tools)
        has_sys = bool(sys_prompt)

        sys_cap = min(MAX_SYSTEM_BYTES, int(max_bytes * (0.34 if has_tools else 0.45)))
        tools_cap = min(MAX_TOOLS_BYTES, int(max_bytes * (0.26 if has_sys else 0.35)))
        if is_overflow_recovery:
            sys_cap = min(sys_cap, int(max_bytes * 0.28))
            tools_cap = min(tools_cap, int(max_bytes * 0.24))

        sys_section = ""
        if has_sys:
            sys_body = clip_utf8_middle(sys_prompt, sys_cap, head_ratio=0.6)
            sys_section = f"[System Instructions]\n{sys_body}"

        tools_section = ""
        if has_tools:
            tools_section = format_compact_tools(req.tools, max_bytes=tools_cap)

        def _assemble_fixed(active_m: Sequence[StagedMedia], archived_m: Sequence[StagedMedia]) -> str:
            parts: List[str] = [_build_policy_section(active_m, archived_m)]
            if sys_section:
                parts.append(sys_section)
            if tools_section:
                parts.append(tools_section)
            return "\n\n".join(parts)

        fixed_est = _assemble_fixed(media_list, ())
        conv_header = "\n\n[Conversation]\n"
        conv_budget = max(2048, max_bytes - utf8_len(fixed_est) - utf8_len(conv_header) - 128)

        if not sub_messages:
            final_text = clip_utf8_middle(fixed_est, max_bytes)
            res = ContextBuildResult(
                prompt=final_text,
                original_bytes=original_bytes,
                final_bytes=utf8_len(final_text),
                original_tokens_est=original_tokens,
                final_tokens_est=estimate_tokens(final_text),
                compacted=True,
                compaction_stage="overflow_recovery" if is_overflow_recovery else "tools_and_masking",
                compaction_mode=self.mode,
                total_messages=0,
                kept_messages=0,
                session_fingerprint=session_fp,
            )
            if record_stats:
                self.stats.record_build(res)
            return res

        # Stage 2: Observation Masking + Duplicate File-Read Superseding + Stale Thinking Strip
        tool_index = _index_tool_calls(sub_messages)
        # Protect the last 4 messages (typically the latest assistant->tool->assistant exchange)
        # from aggressive masking while still capping single giant tool dumps.
        protected_tail_start = max(0, len(sub_messages) - 4)
        stale_tool_cap = min(self.tool_output_max_bytes, max(1500, conv_budget // 8))
        # Even the latest single tool result must leave room for the user's goal & recent steps
        # when prior turns exist!
        if len(sub_messages) > 1:
            latest_tool_cap = max(4096, int(conv_budget * 0.55))
        else:
            latest_tool_cap = conv_budget

        masked_messages, masked_cnt, deduped_cnt = _apply_observation_masking(
            sub_messages,
            tool_index=tool_index,
            protected_tail_start=protected_tail_start,
            stale_tool_max_bytes=stale_tool_cap,
            latest_tool_max_bytes=latest_tool_cap,
            aggressive=is_overflow_recovery,
        )

        units = _group_into_turn_units(masked_messages)
        rendered_units = [_render_unit(u) for u in units]
        unit_sizes = [utf8_len(r) for r in rendered_units]
        total_conv_bytes = sum(unit_sizes) + max(0, (len(units) - 1) * 2)

        # Check if Stage 1 + Stage 2 (tool compaction + dedup + observation masking) ALREADY fits!
        if total_conv_bytes <= conv_budget:
            active_m, archived_m = _find_referenced_media(rendered_units, media_list)
            fixed_text = _assemble_fixed(active_m, archived_m)
            conv_body = "\n\n".join(r for r in rendered_units if r)
            final_prompt = fixed_text + conv_header + conv_body
            if utf8_len(final_prompt) <= max_bytes:
                res = ContextBuildResult(
                    prompt=final_prompt,
                    original_bytes=original_bytes,
                    final_bytes=utf8_len(final_prompt),
                    original_tokens_est=original_tokens,
                    final_tokens_est=estimate_tokens(final_prompt),
                    compacted=True,
                    compaction_stage="overflow_recovery" if is_overflow_recovery else "tools_and_masking",
                    compaction_mode=self.mode,
                    total_messages=len(req.messages),
                    kept_messages=len(req.messages),
                    compacted_messages=0,
                    masked_tool_results=masked_cnt,
                    deduped_tool_results=deduped_cnt,
                    split_turn=False,
                    session_fingerprint=session_fp,
                    active_media_indices=tuple(m.index for m in active_m),
                )
                if record_stats:
                    self.stats.record_build(res)
                return res

        # Stage 3: Atomic Turn-Unit Slicing (Pi `findCutPoint` + `isSplitTurn` + Sticky Prefix Hysteresis)
        # Reserve guaranteed budget for the structured compaction checkpoint!
        checkpoint_budget = min(self.checkpoint_max_bytes, max(2400, int(conv_budget * 0.28)))
        tail_budget = max(2048, conv_budget - checkpoint_budget - 64)
        # Keep `keep_recent_bytes` below `tail_budget` when room permits so consecutive turns
        # can reuse the same sticky checkpoint prefix (maximizing prompt-cache hits, Paseo #8809).
        effective_keep_bytes = min(tail_budget, self.keep_recent_bytes)

        sticky_cut: Optional[_CutPointResult] = None
        sticky_summary: Optional[str] = None
        sticky_mode: str = ""
        if (
            override_summary is None
            and not is_overflow_recovery
            and self.checkpoint_store is not None
        ):
            latest_cp = self.checkpoint_store.get_latest_for_session(session_fp)
            if latest_cp is not None:
                c_hash, c_text, c_src_msgs, c_mode = latest_cp
                if 0 < c_src_msgs < len(masked_messages):
                    matching_unit_idx = next(
                        (i for i, u in enumerate(units) if u.start_idx == c_src_msgs),
                        None,
                    )
                    if matching_unit_idx is not None and matching_unit_idx > 0:
                        tail_cost = sum(unit_sizes[matching_unit_idx:]) + max(
                            0, (len(units) - matching_unit_idx - 1) * 2
                        )
                        if (
                            tail_cost <= tail_budget
                            and compute_prefix_hash(masked_messages[:c_src_msgs]) == c_hash
                        ):
                            is_split = False
                            turn_start_msg: Optional[int] = None
                            if not units[matching_unit_idx].is_human_turn_start:
                                for back in range(matching_unit_idx - 1, -1, -1):
                                    if units[back].is_human_turn_start:
                                        is_split = True
                                        turn_start_msg = units[back].start_idx
                                        break
                            sticky_cut = _CutPointResult(
                                cut_unit_index=matching_unit_idx,
                                cut_message_index=c_src_msgs,
                                is_split_turn=is_split,
                                turn_start_message_index=turn_start_msg,
                            )
                            sticky_summary = c_text
                            sticky_mode = c_mode

        cut = sticky_cut or _find_cut_point(units, unit_sizes, keep_recent_bytes=effective_keep_bytes)

        # Single-unit / giant-latest-unit edge case: if cut == 0, every message stays in tail
        # and we tighten per-message sizes while protecting human intent & tail end!
        if cut.cut_unit_index == 0:
            active_m, archived_m = _find_referenced_media(rendered_units, media_list)
            fixed_text = _assemble_fixed(active_m, archived_m)
            actual_conv_budget = max(1024, max_bytes - utf8_len(fixed_text) - utf8_len(conv_header) - 64)
            trimmed_units = self._fit_tail_units(units, actual_conv_budget)
            final_prompt = fixed_text + conv_header + "\n\n".join(trimmed_units)
            if utf8_len(final_prompt) > max_bytes:
                # Re-clip ONLY the conversation body — NEVER slice system/tools!
                rem = max(512, max_bytes - utf8_len(fixed_text) - utf8_len(conv_header))
                final_prompt = fixed_text + conv_header + clip_utf8_middle(
                    "\n\n".join(trimmed_units), rem, head_ratio=0.45
                )
            res = ContextBuildResult(
                prompt=final_prompt,
                original_bytes=original_bytes,
                final_bytes=utf8_len(final_prompt),
                original_tokens_est=original_tokens,
                final_tokens_est=estimate_tokens(final_prompt),
                compacted=True,
                compaction_stage="overflow_recovery" if is_overflow_recovery else "turn_compaction",
                compaction_mode=self.mode,
                total_messages=len(req.messages),
                kept_messages=len(req.messages),
                compacted_messages=0,
                masked_tool_results=masked_cnt,
                deduped_tool_results=deduped_cnt,
                split_turn=False,
                session_fingerprint=session_fp,
                active_media_indices=tuple(m.index for m in active_m),
            )
            if record_stats:
                self.stats.record_build(res)
            return res

        # Stage 4: Structured Rolling Checkpoint for Prefix (`messages[:cut.cut_message_index]`)
        prefix_msgs = masked_messages[: cut.cut_message_index]
        tail_units = units[cut.cut_unit_index :]
        prefix_hash = compute_prefix_hash(prefix_msgs)

        # Look up cached checkpoint (sticky hit -> exact hit -> fresh deterministic build)
        cached_summary: Optional[str] = override_summary
        cached_mode: str = "llm" if override_summary else ""
        cache_hit = False
        if cached_summary is None and sticky_summary is not None:
            cached_summary = sticky_summary
            cached_mode = sticky_mode
            cache_hit = True
        elif cached_summary is None and self.checkpoint_store is not None:
            found = self.checkpoint_store.get_with_mode(session_fp, prefix_hash)
            if found is not None:
                cached_summary, cached_mode = found
                cache_hit = True

        active_turn_user: Optional[ChatMessage] = None
        if cut.is_split_turn and cut.turn_start_message_index is not None:
            active_turn_user = masked_messages[cut.turn_start_message_index]

        # Only require active `read_file` for media still referenced in the verbatim tail
        tail_rendered_raw = [_render_unit(u) for u in tail_units]
        active_m, archived_m = _find_referenced_media(tail_rendered_raw, media_list)
        fixed_text = _assemble_fixed(active_m, archived_m)
        actual_conv_budget = max(2048, max_bytes - utf8_len(fixed_text) - utf8_len(conv_header) - 64)

        checkpoint_budget = min(checkpoint_budget, max(1800, int(actual_conv_budget * 0.32)))
        if cached_summary is not None and cached_mode == "deterministic":
            # Reuse cached deterministic checkpoint verbatim for byte-for-byte prefix stability
            checkpoint_text = clip_utf8_middle(cached_summary, checkpoint_budget, head_ratio=0.65)
        else:
            checkpoint_text = build_deterministic_checkpoint(
                prefix_msgs,
                tool_index=tool_index,
                max_bytes=checkpoint_budget,
                active_turn_user_msg=active_turn_user,
                active_turn_index=cut.turn_start_message_index,
                cached_llm_summary=cached_summary if cached_mode == "llm" else None,
            )

        # Serialize compacted prefix text (for optional hybrid/llm summarization & cache accounting)
        prefix_serialized = "\n\n".join(
            t for t in (_format_single_message(m) for m in prefix_msgs) if t
        )
        prefix_serialized_bytes = utf8_len(prefix_serialized)

        # Persist deterministic checkpoint on cache miss so subsequent turns hit the sticky prefix
        if (
            not cache_hit
            and override_summary is None
            and not is_overflow_recovery
            and self.checkpoint_store is not None
        ):
            self.checkpoint_store.put(
                session_fp,
                prefix_hash,
                checkpoint_text,
                mode="deterministic",
                source_messages=len(prefix_msgs),
                source_bytes=prefix_serialized_bytes,
            )

        remaining_tail_budget = max(1024, actual_conv_budget - utf8_len(checkpoint_text) - 4)
        fitted_tail = self._fit_tail_units(tail_units, remaining_tail_budget)

        conv_body = checkpoint_text + "\n\n" + "\n\n".join(fitted_tail)
        final_prompt = fixed_text + conv_header + conv_body

        # Final hard ceiling safeguard: ONLY tighten checkpoint/tail, NEVER slice System/Tools!
        if utf8_len(final_prompt) > max_bytes:
            allowed_conv = max(512, max_bytes - utf8_len(fixed_text) - utf8_len(conv_header))
            # Prefer shrinking checkpoint slightly before touching the latest turn
            overflow = utf8_len(conv_body) - allowed_conv
            if overflow > 0 and utf8_len(checkpoint_text) > 1200:
                new_cp_budget = max(900, utf8_len(checkpoint_text) - overflow - 32)
                checkpoint_text = clip_utf8_middle(checkpoint_text, new_cp_budget, head_ratio=0.7)
                conv_body = checkpoint_text + "\n\n" + "\n\n".join(fitted_tail)
            if utf8_len(conv_body) > allowed_conv:
                # Re-fit tail strictly within remaining budget so latest user text stays intact
                rem_tail = max(400, allowed_conv - utf8_len(checkpoint_text) - 4)
                fitted_tail = self._fit_tail_units(tail_units, rem_tail)
                conv_body = checkpoint_text + "\n\n" + "\n\n".join(fitted_tail)
            if utf8_len(conv_body) > allowed_conv:
                conv_body = clip_utf8_middle(conv_body, allowed_conv, head_ratio=0.35)
            final_prompt = fixed_text + conv_header + conv_body

        res = ContextBuildResult(
            prompt=final_prompt,
            original_bytes=original_bytes,
            final_bytes=utf8_len(final_prompt),
            original_tokens_est=original_tokens,
            final_tokens_est=estimate_tokens(final_prompt),
            compacted=True,
            compaction_stage="overflow_recovery" if is_overflow_recovery else "turn_compaction",
            compaction_mode=self.mode,
            cache_hit=cache_hit,
            total_messages=len(req.messages),
            kept_messages=len(req.messages) - len(prefix_msgs),
            compacted_messages=len(prefix_msgs),
            masked_tool_results=masked_cnt,
            deduped_tool_results=deduped_cnt,
            split_turn=cut.is_split_turn,
            session_fingerprint=session_fp,
            prefix_hash=prefix_hash,
            compacted_prefix_text=prefix_serialized,
            compacted_prefix_bytes=prefix_serialized_bytes,
            active_media_indices=tuple(m.index for m in active_m),
            llm_summary_cached=(cached_mode == "llm"),
        )
        if record_stats:
            self.stats.record_build(res)
        return res

    def _fit_tail_units(self, tail_units: Sequence[_TurnUnit], budget_bytes: int) -> List[str]:
        """Render tail atomic units strictly within budget_bytes, always preserving the latest turn."""
        if not tail_units:
            return []
        rendered = [_render_unit(u) for u in tail_units]
        total = sum(utf8_len(r) for r in rendered) + max(0, (len(rendered) - 1) * 2)
        if total <= budget_bytes:
            return [r for r in rendered if r]

        # Keep newest unit intact (or clipped to budget if single unit exceeds budget),
        # then pack earlier tail units from newest to oldest.
        kept_rev: List[str] = []
        used = 0
        for idx in range(len(rendered) - 1, -1, -1):
            is_latest = idx == len(rendered) - 1
            text = rendered[idx]
            if not text:
                continue
            if is_latest:
                # If the latest unit alone exceeds budget, preserve tail end (human question / latest result)
                if utf8_len(text) > budget_bytes:
                    text = clip_utf8_middle(text, budget_bytes, head_ratio=0.45)
                kept_rev.append(text)
                used += utf8_len(text)
                continue
            cost = utf8_len(text) + 2
            if used + cost <= budget_bytes:
                kept_rev.append(text)
                used += cost
            else:
                rem = budget_bytes - used - 2
                if rem >= 512:
                    clipped = clip_utf8_middle(text, rem, head_ratio=0.55)
                    kept_rev.append(clipped)
                    used += utf8_len(clipped) + 2
                break
        return list(reversed(kept_rev))

    def build_delta_context(
        self,
        req: NormalizedRequest,
        delta_messages: Sequence[ChatMessage],
        *,
        max_bytes: int = MAX_PROMPT_BYTES,
        staged_media: Optional[Sequence[StagedMedia]] = None,
        record_stats: bool = True,
    ) -> ContextBuildResult:
        """Build an incremental delta prompt for resuming an active `agy --conversation` session.

        Only the newly arrived delta turns are transmitted, allowing `agy` to leverage its
        native 1M-token multi-turn memory, prompt caching, and native compaction.
        """
        media_list = list(staged_media or [])
        session_fp = compute_session_fingerprint(req)

        # Estimate full-request byte footprint for accurate savings telemetry
        full_bytes = utf8_len(req.system_prompt or "")
        for m in req.messages:
            full_bytes += utf8_len(m.content) + 16
            for tr in m.tool_results:
                full_bytes += utf8_len(tr.content) + 32

        sub_delta = [_substitute_media_in_message(m, media_list) for m in delta_messages]
        raw_delta_texts = [t for t in (_format_single_message(m) for m in sub_delta) if t]
        active_m, _ = _find_referenced_media(raw_delta_texts, media_list)

        header_parts: List[str] = []
        if active_m:
            manifest_lines = [
                f"- #{m.index} ({m.kind}, {m.mime_type}): {m.path}" for m in active_m
            ]
            header_parts.append(
                "[New Media Attachment(s) for This Turn]\n"
                + "\n".join(manifest_lines)
                + "\nCall the built-in `read_file` tool ONLY on the new attachment path(s) listed above before answering."
            )
        if req.tools:
            header_parts.append(
                "[Continued Session — New Turn(s)]\n"
                "Continue the ongoing conversation using the exact same system instructions, "
                "external harness tools (<tool_call> blocks), and no-built-in-tools policy."
            )
        else:
            header_parts.append("[Continued Session — New Turn(s)]")

        header_text = "\n\n".join(header_parts) + "\n\n"
        body_budget = max(2048, max_bytes - utf8_len(header_text) - 64)

        body = "\n\n".join(raw_delta_texts)
        masked_cnt = 0
        deduped_cnt = 0
        compacted_flag = False

        if utf8_len(body) > body_budget:
            compacted_flag = True
            full_tool_index = _index_tool_calls(req.messages)
            latest_cap = (
                max(4096, int(body_budget * 0.65)) if len(sub_delta) > 1 else body_budget
            )
            masked_delta, masked_cnt, deduped_cnt = _apply_observation_masking(
                sub_delta,
                tool_index=full_tool_index,
                protected_tail_start=max(0, len(sub_delta) - 2),
                stale_tool_max_bytes=min(self.tool_output_max_bytes, max(2048, body_budget // 4)),
                latest_tool_max_bytes=latest_cap,
            )
            delta_units = _group_into_turn_units(masked_delta)
            fitted = self._fit_tail_units(delta_units, body_budget)
            body = "\n\n".join(fitted)
            if utf8_len(body) > body_budget:
                body = clip_utf8_middle(body, body_budget, head_ratio=0.45)

        final_prompt = header_text + body
        if utf8_len(final_prompt) > max_bytes:
            allowed = max(512, max_bytes - utf8_len(header_text))
            final_prompt = header_text + clip_utf8_middle(body, allowed, head_ratio=0.45)

        final_bytes = utf8_len(final_prompt)
        res = ContextBuildResult(
            prompt=final_prompt,
            original_bytes=max(full_bytes, final_bytes),
            final_bytes=final_bytes,
            original_tokens_est=estimate_tokens(req.system_prompt or "") + sum(
                estimate_tokens(m.content) for m in req.messages
            ),
            final_tokens_est=estimate_tokens(final_prompt),
            compacted=compacted_flag,
            compaction_stage="session_delta",
            compaction_mode=self.mode,
            cache_hit=True,
            total_messages=len(req.messages),
            kept_messages=len(delta_messages),
            compacted_messages=0,
            masked_tool_results=masked_cnt,
            deduped_tool_results=deduped_cnt,
            split_turn=False,
            session_fingerprint=session_fp,
            active_media_indices=tuple(m.index for m in active_m),
            session_reused=True,
        )
        if record_stats:
            self.stats.record_build(res)
        return res
