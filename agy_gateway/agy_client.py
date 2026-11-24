"""Subprocess wrapper around the official Google Antigravity CLI (agy)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_LINE_LIMIT = 16 * 1024 * 1024  # 16 MiB per NDJSON event line


def _normalize_finish_reason(raw: Any) -> Optional[str]:
    """Normalize provider/agy stop/finish reasons into canonical 'stop' or 'length'."""
    if not raw:
        return None
    val = str(raw).strip().lower()
    if val in ("max_tokens", "length", "truncated", "max_output_tokens", "token_limit"):
        return "length"
    if val in ("stop", "end_turn", "completed", "finish_reason_stop"):
        return "stop"
    return val


class AgyExecutionError(RuntimeError):
    """Raised when an agy CLI execution fails or returns an error event."""


class AgySessionMismatchError(AgyExecutionError):
    """Raised when `--conversation <id>` was not found and agy silently opened a fresh session."""


@dataclass(frozen=True)
class AgyModelInfo:
    """Model entry reported by `agy models`."""

    id: str
    display_name: str


@dataclass(frozen=True)
class UsageInfo:
    """Per-turn token accounting normalized from agy stream-json events.

    Semantics match `agy`'s per-step events:
    - `input_tokens`: non-cached prompt tokens (excludes `cache_read_tokens`)
    - `cache_read_tokens`: cached prompt tokens
    - `output_tokens`: total generated tokens (includes `thinking_tokens`)
    - `thinking_tokens`: reasoning subset of `output_tokens`
    """

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    total_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """Total effective prompt tokens (non-cached + cached)."""
        return self.input_tokens + self.cache_read_tokens


@dataclass(frozen=True)
class AgyChunk:
    """Normalized streaming delta or completion chunk from agy."""

    text_delta: str = ""
    thinking_delta: str = ""
    done: bool = False
    usage: UsageInfo = field(default_factory=UsageInfo)
    conversation_id: str = ""
    finish_reason: Optional[str] = None


FALLBACK_MODELS: List[AgyModelInfo] = [
    AgyModelInfo("gemini-4-neon-high", "Gemini 4 Neon (High)"),
    AgyModelInfo("gemini-4-neon-xhigh", "Gemini 4 Neon (XHigh)"),
    AgyModelInfo("gemini-4-neon-medium", "Gemini 4 Neon (Medium)"),
    AgyModelInfo("gemini-4-neon-low", "Gemini 4 Neon (Low)"),
    AgyModelInfo("gemini-4-hydrogen-high", "Gemini 4 Hydrogen (High)"),
    AgyModelInfo("gemini-4-hydrogen-xhigh", "Gemini 4 Hydrogen (XHigh)"),
    AgyModelInfo("gemini-4-hydrogen-medium", "Gemini 4 Hydrogen (Medium)"),
    AgyModelInfo("gemini-4-hydrogen-low", "Gemini 4 Hydrogen (Low)"),
    AgyModelInfo("gemini-4-carbon-high", "Gemini 4 Carbon (High)"),
    AgyModelInfo("gemini-4-carbon-medium", "Gemini 4 Carbon (Medium)"),
    AgyModelInfo("gemini-4-carbon-low", "Gemini 4 Carbon (Low)"),
    AgyModelInfo("gemini-4-argon-high", "Gemini 4 Argon (High)"),
    AgyModelInfo("gemini-4-argon-medium", "Gemini 4 Argon (Medium)"),
    AgyModelInfo("gemini-4-argon-low", "Gemini 4 Argon (Low)"),
    AgyModelInfo("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
    AgyModelInfo("gemini-3.8-flash-medium", "Gemini 3.8 Flash (Medium)"),
    AgyModelInfo("gemini-3.8-flash-low", "Gemini 3.8 Flash (Low)"),
    AgyModelInfo("claude-opus-5-5-high", "Claude Opus 5.5 (High)"),
    AgyModelInfo("claude-opus-5-5-medium", "Claude Opus 5.5 (Medium)"),
    AgyModelInfo("claude-opus-5-5-low", "Claude Opus 5.5 (Low)"),
    AgyModelInfo("claude-sonnet-5-5-high", "Claude Sonnet 5.5 (High)"),
    AgyModelInfo("claude-sonnet-5-5-medium", "Claude Sonnet 5.5 (Medium)"),
    AgyModelInfo("claude-sonnet-5-5-low", "Claude Sonnet 5.5 (Low)"),
    AgyModelInfo("gpt-oss-120b-medium", "GPT-OSS 120B (Medium)"),
]


def parse_models_output(raw_output: str) -> List[AgyModelInfo]:
    """Parse `agy models` stdout after stripping spinner carriage returns and ANSI codes."""
    cleaned = _ANSI_RE.sub("", raw_output)
    models: List[AgyModelInfo] = []
    seen: set[str] = set()
    for raw_line in cleaned.splitlines():
        # Spinner frames use '\r'; keep only the final segment on the line.
        segment = raw_line.split("\r")[-1].strip()
        if not segment or segment.endswith("Fetching available models..."):
            continue
        match = re.match(r"^([a-zA-Z0-9._-]+)(?:\t+|\s{2,})(.+)$", segment)
        if not match:
            continue
        model_id, display_name = match.group(1), match.group(2).strip()
        if model_id not in seen:
            seen.add(model_id)
            models.append(AgyModelInfo(id=model_id, display_name=display_name))
    return models


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON object from `agy -p '/quota' --output-format json` stdout."""
    cleaned = _ANSI_RE.sub("", raw_text).strip()
    if not cleaned:
        return None
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(cleaned.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            with contextlib.suppress(json.JSONDecodeError):
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    return obj
    return None


def parse_quota_payload(
    quota_raw: str,
    credits_raw: str = "",
) -> Tuple[bool, Dict[str, Any], str]:
    """Normalize official `agy -p '/quota'` and `'/credits'` JSON output into a compact summary."""
    q_obj = _extract_json_object(quota_raw)
    if not q_obj:
        err = _ANSI_RE.sub("", quota_raw).strip()[:300]
        return False, {}, err or "Empty response from `agy -p /quota`"

    status = str(q_obj.get("status") or "")
    if status and status != "SUCCESS":
        err = str(q_obj.get("error") or q_obj.get("response") or f"status={status}")
        return False, {}, err[:300]

    cmd_data = ((q_obj.get("command") or {}).get("data") or {})
    raw_groups = cmd_data.get("groups")
    if not isinstance(raw_groups, list):
        err = str(q_obj.get("error") or q_obj.get("response") or "No quota groups returned")
        return False, {}, err[:300]

    groups: List[Dict[str, Any]] = []
    all_percents: List[float] = []
    five_h_percents: List[float] = []
    weekly_percents: List[float] = []

    for grp in raw_groups:
        if not isinstance(grp, dict):
            continue
        buckets_out: List[Dict[str, Any]] = []
        for b in grp.get("buckets") or []:
            if not isinstance(b, dict):
                continue
            raw_frac = b.get("remaining_fraction")
            try:
                frac = max(0.0, min(1.0, float(raw_frac)))
            except (TypeError, ValueError):
                continue
            pct = round(frac * 100.0, 1)
            window = str(b.get("window") or "").lower()
            all_percents.append(pct)
            if window == "5h":
                five_h_percents.append(pct)
            elif window == "weekly":
                weekly_percents.append(pct)
            buckets_out.append(
                {
                    "id": str(b.get("id") or ""),
                    "name": str(b.get("name") or ""),
                    "window": window,
                    "remaining_fraction": round(frac, 4),
                    "remaining_percent": pct,
                    "reset_time": str(b.get("reset_time") or ""),
                    "description": str(b.get("description") or ""),
                }
            )
        groups.append(
            {
                "name": str(grp.get("name") or "Unknown"),
                "description": str(grp.get("description") or ""),
                "buckets": buckets_out,
            }
        )

    remaining_credits: Optional[int] = None
    if credits_raw:
        c_obj = _extract_json_object(credits_raw)
        if c_obj and str(c_obj.get("status") or "SUCCESS") == "SUCCESS":
            c_data = (c_obj.get("command") or {}).get("data") or {}
            if c_data.get("remaining_credits") is not None:
                with contextlib.suppress(TypeError, ValueError):
                    remaining_credits = int(c_data["remaining_credits"])

    summary: Dict[str, Any] = {
        "groups": groups,
        "remaining_credits": remaining_credits,
        "min_remaining_percent": min(all_percents) if all_percents else None,
        "min_5h_percent": min(five_h_percents) if five_h_percents else None,
        "min_weekly_percent": min(weekly_percents) if weekly_percents else None,
        "error": None,
    }
    return True, summary, ""


def _parse_usage(raw_usage: dict | None) -> UsageInfo:
    if not isinstance(raw_usage, dict):
        return UsageInfo()
    return UsageInfo(
        input_tokens=int(raw_usage.get("input_tokens") or 0),
        output_tokens=int(raw_usage.get("output_tokens") or 0),
        thinking_tokens=int(raw_usage.get("thinking_tokens") or 0),
        cache_read_tokens=int(raw_usage.get("cache_read_tokens") or 0),
        total_tokens=int(raw_usage.get("total_tokens") or 0),
    )


def _has_nonzero_usage(usage: UsageInfo) -> bool:
    return any(
        (
            usage.input_tokens,
            usage.output_tokens,
            usage.thinking_tokens,
            usage.cache_read_tokens,
            usage.total_tokens,
        )
    )


def _aggregate_turn_usage(step_usages: Dict[Any, UsageInfo]) -> UsageInfo:
    """Compute accurate per-turn usage from `step_update.usage` entries for THIS turn only.

    `agy --conversation` reports session-cumulative totals in the final `result.usage`
    event (summing every turn since Turn 1), whereas each `step_update.usage` reports
    that single inference step. Within one turn (which may involve multiple internal steps,
    e.g., reading staged media via `read_file`), the active context window size is the
    final step's `(input_tokens, cache_read_tokens)` while generated tokens sum across
    this turn's steps only.
    """
    ordered = list(step_usages.values())
    if not ordered:
        return UsageInfo()
    last = ordered[-1]
    out_tok = sum(u.output_tokens for u in ordered)
    think_tok = sum(u.thinking_tokens for u in ordered)
    total_tok = last.input_tokens + last.cache_read_tokens + out_tok
    return UsageInfo(
        input_tokens=last.input_tokens,
        output_tokens=out_tok,
        thinking_tokens=think_tok,
        cache_read_tokens=last.cache_read_tokens,
        total_tokens=total_tok,
    )


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
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


async def _drain_stderr(
    stream: Optional[asyncio.StreamReader], max_bytes: int = 32768
) -> str:
    """Continuously drain stderr into a bounded tail buffer so the child never pipe-deadlocks."""
    if stream is None:
        return ""
    buf = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            del buf[: len(buf) - max_bytes]
    return buf.decode("utf-8", errors="replace").strip()


class AgyRunner:
    """Executes headless single-turn completions via the official `agy` CLI."""

    def __init__(
        self,
        agy_bin: str,
        workdir: Path,
        sandbox: bool = True,
        disable_slash_commands: bool = True,
        default_effort: str = "",
        model_cache_ttl_seconds: float = 300.0,
    ) -> None:
        self.agy_bin = agy_bin
        self.workdir = Path(workdir)
        self.sandbox = sandbox
        self.disable_slash_commands = disable_slash_commands
        self.default_effort = default_effort
        self.model_cache_ttl_seconds = model_cache_ttl_seconds
        self._models_cache: Dict[str, Tuple[float, List[AgyModelInfo]]] = {}
        self._models_lock = asyncio.Lock()

    def _build_command(
        self,
        model: str,
        effort: Optional[str],
        conversation_id: Optional[str] = None,
    ) -> List[str]:
        cmd = [
            self.agy_bin,
            "-p=",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--model",
            model,
        ]
        if conversation_id:
            cmd.extend(["--conversation", conversation_id])
        effective_effort = effort or self.default_effort
        if effective_effort:
            cmd.extend(["--effort", effective_effort])
        if self.disable_slash_commands:
            cmd.append("--disable-slash-commands")
        if self.sandbox:
            cmd.append("--sandbox")
        return cmd

    def _get_cached_models(self, cache_key: str) -> Optional[List[AgyModelInfo]]:
        entry = self._models_cache.get(cache_key)
        if entry is None:
            return None
        cached_at, models = entry
        if models and (time.monotonic() - cached_at) < self.model_cache_ttl_seconds:
            return list(models)
        return None

    async def list_models(
        self,
        force_refresh: bool = False,
        env: Optional[Dict[str, str]] = None,
        cache_key: str = "default",
    ) -> List[AgyModelInfo]:
        """Query available models via `agy models`, caching results per account key."""
        if not force_refresh:
            cached = self._get_cached_models(cache_key)
            if cached is not None:
                return cached

        async with self._models_lock:
            if not force_refresh:
                cached = self._get_cached_models(cache_key)
                if cached is not None:
                    return cached

            if not Path(self.agy_bin).exists():
                return list(FALLBACK_MODELS)

            self.workdir.mkdir(parents=True, exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                self.agy_bin,
                "models",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.workdir),
                env=env if env is not None else os.environ.copy(),
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
            except asyncio.TimeoutError:
                await _terminate_process(proc)
                stale = self._models_cache.get(cache_key)
                return list(stale[1] if stale else FALLBACK_MODELS)

            parsed = parse_models_output(stdout_bytes.decode("utf-8", errors="replace"))
            if parsed:
                self._models_cache[cache_key] = (time.monotonic(), parsed)
                return list(parsed)
            stale = self._models_cache.get(cache_key)
            return list(stale[1] if stale else FALLBACK_MODELS)

    async def probe_account(
        self,
        env: Dict[str, str],
        cache_key: Optional[str] = None,
    ) -> Tuple[bool, List[AgyModelInfo], str]:
        """Run `agy models` under an account's isolated env to verify sign-in status."""
        if not Path(self.agy_bin).exists():
            return False, [], f"agy binary not found at {self.agy_bin}"
        self.workdir.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            self.agy_bin,
            "models",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(self.workdir),
            env=env,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=25.0)
        except asyncio.TimeoutError:
            await _terminate_process(proc)
            return False, [], "Timed out checking `agy models`"

        output = _ANSI_RE.sub("", stdout_bytes.decode("utf-8", errors="replace")).strip()
        models = parse_models_output(output)
        if models and proc.returncode == 0:
            if cache_key:
                self._models_cache[cache_key] = (time.monotonic(), models)
            return True, models, ""
        # Strip spinner noise for clean error text
        lines = [ln.split("\r")[-1].strip() for ln in output.splitlines()]
        clean_err = " ".join(ln for ln in lines if ln and not ln.endswith("Fetching available models..."))
        return False, [], clean_err or f"agy models exited with code {proc.returncode}"

    async def _run_readonly_slash(
        self,
        slash_cmd: str,
        env: Dict[str, str],
        timeout_seconds: float = 20.0,
    ) -> Tuple[int, str]:
        """Run a zero-cost read-only `agy -p '/...' --output-format json` command."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            self.agy_bin,
            "-p",
            slash_cmd,
            "--output-format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(self.workdir),
            env=env,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            await _terminate_process(proc)
            return -1, f"Timed out running `agy -p {slash_cmd}`"
        return int(proc.returncode or 0), stdout_bytes.decode("utf-8", errors="replace")

    async def fetch_account_quota(
        self,
        env: Dict[str, str],
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Fetch remaining quota buckets & credits via official read-only `/quota` and `/credits`."""
        if not Path(self.agy_bin).exists():
            return False, {}, f"agy binary not found at {self.agy_bin}"
        (q_rc, q_out), (_, c_out) = await asyncio.gather(
            self._run_readonly_slash("/quota", env=env),
            self._run_readonly_slash("/credits", env=env),
        )
        ok, payload, err = parse_quota_payload(q_out, c_out)
        if ok:
            return True, payload, ""
        return False, {}, err or f"agy /quota exited with code {q_rc}"

    async def stream_prompt(
        self,
        prompt: str,
        model: str,
        effort: Optional[str] = None,
        timeout_seconds: float = 180.0,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Path] = None,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[AgyChunk]:
        """Run one prompt through `agy` stream-json mode (optionally resuming `conversation_id`)."""
        effective_cwd = Path(cwd) if cwd is not None else self.workdir
        effective_cwd.mkdir(parents=True, exist_ok=True)
        expected_conv_id = (conversation_id or "").strip() or None
        cmd = self._build_command(
            model=model,
            effort=effort,
            conversation_id=expected_conv_id,
        )
        run_env = env if env is not None else os.environ.copy()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(effective_cwd),
            env=run_env,
            limit=_LINE_LIMIT,
        )

        emitted_text = False
        got_result = False
        latest_usage = UsageInfo()
        step_usages: Dict[Any, UsageInfo] = {}
        anon_step_seq = 0
        latest_finish_reason: Optional[str] = None
        active_conv_id = expected_conv_id or ""
        deadline = time.monotonic() + timeout_seconds
        stderr_task = asyncio.create_task(_drain_stderr(proc.stderr))

        try:
            assert proc.stdin is not None
            payload = (
                json.dumps({"event": "user", "message": {"content": prompt}}, ensure_ascii=False)
                + "\n"
            )
            proc.stdin.write(payload.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            assert proc.stdout is not None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AgyExecutionError(
                        f"agy request timed out after {timeout_seconds:.1f}s"
                    )
                try:
                    raw_line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=remaining
                    )
                except asyncio.TimeoutError as exc:
                    raise AgyExecutionError(
                        f"agy request timed out after {timeout_seconds:.1f}s"
                    ) from exc
                except ValueError as exc:
                    raise AgyExecutionError(
                        f"agy stream event exceeded line buffer limit ({_LINE_LIMIT} bytes)"
                    ) from exc

                if not raw_line:
                    break

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("{"):
                    continue

                try:
                    event_obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_name = event_obj.get("event")
                if event_name == "init":
                    returned_cid = str(event_obj.get("conversation_id") or "")
                    if expected_conv_id and returned_cid and returned_cid != expected_conv_id:
                        raise AgySessionMismatchError(
                            f"Requested conversation '{expected_conv_id}' not found; "
                            f"agy opened fresh '{returned_cid}'"
                        )
                    active_conv_id = returned_cid or active_conv_id
                elif event_name == "step_update":
                    step = event_obj.get("step_update") or {}
                    step_cid = str(step.get("conversation_id") or "")
                    if expected_conv_id and step_cid and step_cid != expected_conv_id and not emitted_text:
                        raise AgySessionMismatchError(
                            f"Requested conversation '{expected_conv_id}' mismatched step '{step_cid}'"
                        )
                    active_conv_id = step_cid or active_conv_id
                    if isinstance(step.get("usage"), dict):
                        parsed_u = _parse_usage(step.get("usage"))
                        if _has_nonzero_usage(parsed_u):
                            raw_step_idx = step.get("step_index")
                            if raw_step_idx is not None:
                                step_key: Any = raw_step_idx
                            else:
                                step_key = f"__anon_{anon_step_seq}"
                                anon_step_seq += 1
                            step_usages[step_key] = parsed_u
                            latest_usage = _aggregate_turn_usage(step_usages)
                    raw_fr = (
                        step.get("finish_reason")
                        or step.get("stop_reason")
                        or step.get("stopReason")
                    )
                    if raw_fr:
                        latest_finish_reason = _normalize_finish_reason(raw_fr)
                    text_delta = step.get("text_delta") or ""
                    thinking_delta = step.get("thinking_delta") or ""
                    if text_delta or thinking_delta:
                        if text_delta:
                            emitted_text = True
                        yield AgyChunk(
                            text_delta=text_delta,
                            thinking_delta=thinking_delta,
                            done=False,
                            usage=latest_usage,
                            conversation_id=active_conv_id,
                            finish_reason=latest_finish_reason,
                        )
                elif event_name == "result":
                    result = event_obj.get("result") or {}
                    res_cid = str(result.get("conversation_id") or "")
                    if expected_conv_id and res_cid and res_cid != expected_conv_id and not emitted_text:
                        raise AgySessionMismatchError(
                            f"Requested conversation '{expected_conv_id}' mismatched result '{res_cid}'"
                        )
                    active_conv_id = res_cid or active_conv_id
                    if step_usages:
                        # Prefer per-turn aggregated `step_update.usage` over session-cumulative `result.usage`
                        latest_usage = _aggregate_turn_usage(step_usages)
                    elif isinstance(result.get("usage"), dict):
                        latest_usage = _parse_usage(result.get("usage"))
                    raw_fr = (
                        result.get("finish_reason")
                        or result.get("stop_reason")
                        or result.get("stopReason")
                    )
                    if raw_fr:
                        latest_finish_reason = _normalize_finish_reason(raw_fr)
                    status = str(result.get("status") or "")
                    if status != "SUCCESS":
                        err_msg = result.get("error") or f"agy returned status={status}"
                        raise AgyExecutionError(str(err_msg))
                    full_response = str(result.get("response") or "")
                    if not emitted_text and full_response:
                        emitted_text = True
                        yield AgyChunk(
                            text_delta=full_response,
                            done=False,
                            usage=latest_usage,
                            conversation_id=active_conv_id,
                        )
                    got_result = True
                    yield AgyChunk(
                        done=True,
                        usage=latest_usage,
                        conversation_id=active_conv_id,
                        finish_reason=latest_finish_reason or "stop",
                    )
                    break

            try:
                wait_timeout = 5.0 if got_result else max(1.0, deadline - time.monotonic())
                rc = await asyncio.wait_for(proc.wait(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                await _terminate_process(proc)
                rc = 0 if got_result else -1
            if not got_result:
                stderr_text = ""
                with contextlib.suppress(Exception):
                    stderr_text = await asyncio.wait_for(stderr_task, timeout=2.0)
                if rc != 0 or not emitted_text:
                    raise AgyExecutionError(
                        stderr_text
                        or f"agy stream ended prematurely without a result event (exit code {rc})"
                    )
                # Partial text was already streamed before stdout closed cleanly without `result`:
                # mark finish_reason="length" so clients know output was abnormally truncated.
                yield AgyChunk(
                    done=True,
                    usage=latest_usage,
                    conversation_id=active_conv_id,
                    finish_reason="length",
                )
        finally:
            await _terminate_process(proc)
            if not stderr_task.done():
                stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)

    async def complete_prompt(
        self,
        prompt: str,
        model: str,
        effort: Optional[str] = None,
        timeout_seconds: float = 180.0,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Path] = None,
        conversation_id: Optional[str] = None,
    ) -> Tuple[str, UsageInfo, str]:
        """Collect all streamed chunks into a single response tuple."""
        chunks: List[str] = []
        final_usage = UsageInfo()
        conv_id = ""
        async for chunk in self.stream_prompt(
            prompt=prompt,
            model=model,
            effort=effort,
            timeout_seconds=timeout_seconds,
            env=env,
            cwd=cwd,
            conversation_id=conversation_id,
        ):
            if chunk.text_delta:
                chunks.append(chunk.text_delta)
            if chunk.done:
                final_usage = chunk.usage
                conv_id = chunk.conversation_id
        return "".join(chunks), final_usage, conv_id
