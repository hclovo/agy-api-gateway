"""Concurrency dispatcher, multi-account routing, and context-protected execution for Antigravity gateway."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from agy_gateway.accounts import AccountRecord, AccountStore, detect_email_from_logs
from agy_gateway.agy_client import (
    AgyChunk,
    AgyExecutionError,
    AgyModelInfo,
    AgyRunner,
    AgySessionMismatchError,
    UsageInfo,
)
from agy_gateway.config import Settings
from agy_gateway.context_manager import (
    MAX_HISTORICAL_TURN_BYTES,
    MAX_PROMPT_BYTES,
    MAX_SYSTEM_BYTES,
    MAX_TOOLS_BYTES,
    ChatMessage,
    ContextBuildResult,
    ContextCheckpointStore,
    ContextManager,
    ContextOverflowError,
    ContextStatsTracker,
    NormalizedRequest,
    ToolResultItem,
    clip_utf8_middle,
    compute_session_fingerprint,
    estimate_tokens,
    format_compact_tools,
    has_truncated_tool_call,
    is_context_overflow_error,
    parse_tool_calls,
    strip_tool_calls,
    summarize_prefix_map_reduce,
)
from agy_gateway.media import (
    MediaAttachment,
    StagedMedia,
    cleanup_staging_dir,
    extract_media_block,
    media_placeholder,
    stage_media_attachments,
)
from agy_gateway.sessions import (
    SessionRecord,
    SessionStore,
    agy_conversation_db_exists,
    assistant_fingerprints_compatible,
    compute_assistant_fingerprint_from_output,
    compute_messages_prefix_hash,
    compute_session_key,
    compute_sys_tools_hash,
)

__all__ = [
    "MAX_PROMPT_BYTES",
    "MAX_SYSTEM_BYTES",
    "MAX_TOOLS_BYTES",
    "MAX_HISTORICAL_TURN_BYTES",
    "QueueTimeoutError",
    "ContextOverflowError",
    "ChatMessage",
    "ToolResultItem",
    "NormalizedRequest",
    "ContextBuildResult",
    "CompletionResult",
    "clip_utf8_middle",
    "estimate_tokens",
    "format_compact_tools",
    "extract_text_content",
    "build_prompt",
    "parse_tool_calls",
    "strip_tool_calls",
    "has_truncated_tool_call",
    "is_context_overflow_error",
    "AgyDispatcher",
]


class QueueTimeoutError(RuntimeError):
    """Raised when all agy execution slots remain busy past the queue timeout."""


@dataclass(frozen=True)
class CompletionResult:
    """Full non-streaming completion output with finish reason and context diagnostics."""

    text: str
    usage: UsageInfo
    conversation_id: str
    finish_reason: str = "stop"
    context_result: Optional[ContextBuildResult] = None


def extract_text_content(
    content: Any,
    attachments: Optional[List[MediaAttachment]] = None,
) -> str:
    """Flatten OpenAI/Gemini/Anthropic content strings or part arrays into text, optionally extracting media."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif item.get("type") == "tool_result":
                    tool_use_id = item.get("tool_use_id", "")
                    inner = extract_text_content(item.get("content"), attachments=attachments)
                    parts.append(f"[tool_result id={tool_use_id}]\n{inner}".strip())
                elif item.get("type") == "tool_use":
                    parts.append(
                        f"<tool_call>{json.dumps({'name': item.get('name'), 'arguments': item.get('input', {})}, ensure_ascii=False)}</tool_call>"
                    )
                elif "functionCall" in item and isinstance(item["functionCall"], dict):
                    fc = item["functionCall"]
                    parts.append(
                        f"<tool_call>{json.dumps({'name': fc.get('name'), 'arguments': fc.get('args', {})}, ensure_ascii=False)}</tool_call>"
                    )
                elif "functionResponse" in item and isinstance(item["functionResponse"], dict):
                    fr = item["functionResponse"]
                    parts.append(
                        f"[tool_result name={fr.get('name', '')}]\n{json.dumps(fr.get('response', {}), ensure_ascii=False)}"
                    )
                elif attachments is not None:
                    media_marker = extract_media_block(item, attachments)
                    if media_marker:
                        parts.append(media_marker)
        return "\n".join(p for p in parts if p)
    return str(content)


def build_prompt(
    req: NormalizedRequest,
    max_bytes: int = MAX_PROMPT_BYTES,
    staged_media: Optional[Sequence[StagedMedia]] = None,
) -> str:
    """Render normalized request using progressive context management strictly within max_bytes."""
    mgr = ContextManager(mode="deterministic")
    return mgr.build_context(
        req,
        max_bytes=max_bytes,
        staged_media=staged_media,
        record_stats=False,
    ).prompt


def _classify_account_error(error_text: str, base_cooldown: float) -> Tuple[bool, float]:
    """Return (needs_login, cooldown_seconds) based on an `agy` error string."""
    lower = error_text.lower()
    if any(
        needle in lower
        for needle in (
            "not logged in",
            "please sign in",
            "authentication failed",
            "invalid_grant",
            "unauthenticated",
        )
    ):
        return True, 0.0
    if any(
        needle in lower
        for needle in (
            "quota",
            "resource_exhausted",
            "429",
            "rate limit",
            "too many requests",
        )
    ):
        return False, max(base_cooldown, 180.0)
    if any(
        needle in lower
        for needle in (
            "no capacity",
            "unavailable",
            "code 503",
            "overloaded",
            "stream was interrupted",
        )
    ):
        return False, min(base_cooldown, 20.0)
    return False, min(base_cooldown, 45.0)


class AgyDispatcher:
    """Queues, routes, compacts context, self-heals overflows, and fails over across accounts."""

    def __init__(
        self,
        settings: Settings,
        runner: Optional[AgyRunner] = None,
        account_store: Optional[AccountStore] = None,
        checkpoint_store: Optional[ContextCheckpointStore] = None,
        session_store: Optional[SessionStore] = None,
    ) -> None:
        self.settings = settings
        self.runner = runner or AgyRunner(
            agy_bin=settings.agy_bin,
            workdir=settings.workdir,
            sandbox=settings.sandbox,
            disable_slash_commands=settings.disable_slash_commands,
            default_effort=settings.default_effort,
            model_cache_ttl_seconds=settings.model_cache_ttl_seconds,
        )
        self.account_store = account_store or AccountStore(
            db_path=settings.db_path,
            accounts_root=settings.accounts_dir,
            database_url=settings.database_url,
        )
        self.checkpoint_store = checkpoint_store or ContextCheckpointStore(
            db_path=settings.db_path,
            ttl_hours=settings.compaction_cache_ttl_hours,
            database_url=settings.database_url,
        )
        self.session_store = session_store or SessionStore(
            db_path=settings.db_path,
            ttl_hours=settings.session_ttl_hours,
            database_url=settings.database_url,
        )
        self.context_stats = ContextStatsTracker()
        self.context_manager = ContextManager(
            mode=settings.compaction_mode,
            keep_recent_bytes=settings.compaction_keep_recent_bytes,
            checkpoint_max_bytes=settings.compaction_checkpoint_max_bytes,
            tool_output_max_bytes=settings.compaction_tool_output_max_bytes,
            checkpoint_store=self.checkpoint_store,
            stats=self.context_stats,
        )
        self._semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))
        self._account_cond = asyncio.Condition()
        self._rr_cursor = 0
        self._active_by_account: Dict[str, int] = defaultdict(int)
        self._active_session_keys: set[str] = set()
        self._bg_compaction_tasks: Dict[str, asyncio.Task[None]] = {}
        self.active_requests = 0
        self.queued_requests = 0
        self.total_dispatched = 0
        self.total_errors = 0

    def _runner_supports_resume(self) -> bool:
        return isinstance(self.runner, AgyRunner) or bool(
            getattr(self.runner, "supports_conversation_resume", False)
        )

    def _filter_attachments_for_messages(
        self,
        attachments: Sequence[MediaAttachment],
        messages: Sequence[ChatMessage],
    ) -> List[MediaAttachment]:
        if not attachments:
            return []
        haystack_parts: List[str] = []
        for m in messages:
            if m.content:
                haystack_parts.append(m.content)
            for tr in m.tool_results:
                if tr.content:
                    haystack_parts.append(tr.content)
        joined = "\n".join(haystack_parts)
        return [att for att in attachments if media_placeholder(att.index) in joined]

    def _match_reusable_session(
        self,
        req: NormalizedRequest,
        *,
        target_model: str,
        api_key_id: str,
        allowed_accounts: Optional[Sequence[str]],
    ) -> Optional[Tuple[str, SessionRecord, List[ChatMessage], AccountRecord]]:
        """Check if an active `agy --conversation` session cleanly matches the incoming prefix."""
        if (
            not self.settings.session_reuse_enabled
            or not self._runner_supports_resume()
            or not req.messages
        ):
            return None

        session_fp = compute_session_fingerprint(req)
        session_key = compute_session_key(
            api_key_id=api_key_id,
            session_fingerprint=session_fp,
            target_model=target_model,
        )
        # Never contend concurrently on the same agy conversation lockfile
        if session_key in self._active_session_keys:
            return None

        sess = self.session_store.get(session_key)
        if sess is None:
            return None

        if sess.sys_tools_hash != compute_sys_tools_hash(req):
            return None

        sent_n = sess.sent_message_count
        if sent_n <= 0 or len(req.messages) <= sent_n:
            return None

        if compute_messages_prefix_hash(req.messages[:sent_n]) != sess.sent_prefix_hash:
            return None

        next_msg = req.messages[sent_n]
        if next_msg.role == "assistant":
            if not assistant_fingerprints_compatible(sess.last_assistant_fingerprint, next_msg):
                return None
            delta_messages = list(req.messages[sent_n + 1 :])
        else:
            delta_messages = list(req.messages[sent_n:])

        if not delta_messages:
            return None

        acc = self.account_store.get_account(sess.account_id)
        if acc is None or not acc.enabled or acc.status == "needs_login":
            return None
        if acc.cooldown_until > time.time():
            return None
        allowed_set = set(allowed_accounts or [])
        if allowed_set and acc.id not in allowed_set:
            return None

        # Pre-flight disk check prevents `agy --conversation <missing>` from silently opening a blank chat
        if isinstance(self.runner, AgyRunner) and not agy_conversation_db_exists(
            acc.resolved_home(), sess.agy_conversation_id
        ):
            self.session_store.delete(session_key)
            return None

        return session_key, sess, delta_messages, acc

    def _select_candidates(
        self,
        allowed_accounts: Optional[Sequence[str]] = None,
        excluded_ids: Optional[set[str]] = None,
        mutate_cursor: bool = True,
    ) -> Tuple[List[AccountRecord], bool]:
        """Return (ordered_candidates, any_at_per_account_capacity)."""
        now = time.time()
        excluded = excluded_ids or set()
        allowed_set = set(allowed_accounts or [])
        all_accounts = self.account_store.list_accounts()

        eligible: List[AccountRecord] = []
        cooldown_fallback: List[AccountRecord] = []
        any_at_capacity = False

        for acc in all_accounts:
            if not acc.enabled or acc.id in excluded:
                continue
            if allowed_set and acc.id not in allowed_set:
                continue
            if acc.status == "needs_login":
                continue
            if acc.max_concurrency > 0 and self._active_by_account[acc.id] >= acc.max_concurrency:
                any_at_capacity = True
                continue
            if acc.cooldown_until > now:
                cooldown_fallback.append(acc)
                continue
            eligible.append(acc)

        pool = eligible if eligible else cooldown_fallback
        if not pool:
            return [], any_at_capacity

        strategy = self.account_store.get_routing_strategy(self.settings.routing_strategy)
        if not mutate_cursor:
            return (
                sorted(pool, key=lambda a: (a.priority, a.id)),
                any_at_capacity,
            )
        if strategy == "round_robin":
            start = self._rr_cursor % len(pool)
            self._rr_cursor += 1
            return pool[start:] + pool[:start], any_at_capacity
        if strategy == "priority":
            return (
                sorted(
                    pool,
                    key=lambda a: (a.priority, self._active_by_account[a.id], a.id),
                ),
                any_at_capacity,
            )
        # Default: least_busy (tie-break by priority and round-robin rotation)
        start = self._rr_cursor % len(pool)
        self._rr_cursor += 1
        rotated = pool[start:] + pool[:start]
        return (
            sorted(
                rotated,
                key=lambda a: (self._active_by_account[a.id], a.priority),
            ),
            any_at_capacity,
        )

    def _ordered_candidates(
        self,
        allowed_accounts: Optional[Sequence[str]] = None,
        excluded_ids: Optional[set[str]] = None,
        mutate_cursor: bool = True,
    ) -> List[AccountRecord]:
        candidates, _ = self._select_candidates(
            allowed_accounts=allowed_accounts,
            excluded_ids=excluded_ids,
            mutate_cursor=mutate_cursor,
        )
        return candidates

    async def list_models(self, force_refresh: bool = False) -> List[AgyModelInfo]:
        candidates = self._ordered_candidates(mutate_cursor=False)
        if candidates:
            acc = candidates[0]
            env = acc.build_env() if acc.home_dir else None
            try:
                return await self.runner.list_models(
                    force_refresh=force_refresh,
                    env=env,
                    cache_key=acc.id,
                )
            except TypeError:
                try:
                    return await self.runner.list_models(env=env)
                except TypeError:
                    return await self.runner.list_models()
        return await self.runner.list_models()

    async def probe_and_update_account(self, account_id: str) -> Tuple[AccountRecord, List[AgyModelInfo], str]:
        acc = self.account_store.get_account(account_id)
        if acc is None:
            raise KeyError(f"Account '{account_id}' not found")
        try:
            ok, models, err = await self.runner.probe_account(
                acc.build_env(), cache_key=account_id
            )
        except TypeError:
            ok, models, err = await self.runner.probe_account(acc.build_env())
        email = detect_email_from_logs(acc.resolved_home())
        if ok:
            updated = self.account_store.update_account(
                account_id,
                email=email if email else None,
                status="logged_in",
                cooldown_until=0.0,
                clear_error=True,
            )
            fetch_quota = getattr(self.runner, "fetch_account_quota", None)
            if callable(fetch_quota) and updated is not None:
                try:
                    q_ok, q_data, _ = await fetch_quota(acc.build_env())
                    if q_ok and q_data:
                        refreshed = self.account_store.update_quota(account_id, q_data)
                        if refreshed is not None:
                            updated = refreshed
                except Exception:
                    pass
        else:
            needs_login, _ = _classify_account_error(err, self.settings.account_cooldown_seconds)
            updated = self.account_store.update_account(
                account_id,
                email=email if email else None,
                status="needs_login" if needs_login else "error",
                last_error=err,
            )
        assert updated is not None
        return updated, models, err

    async def refresh_account_quota(
        self, account_id: str
    ) -> Tuple[AccountRecord, Optional[Dict[str, Any]], str]:
        """Query official read-only `/quota` + `/credits` for one account and persist snapshot."""
        acc = self.account_store.get_account(account_id)
        if acc is None:
            raise KeyError(f"Account '{account_id}' not found")
        fetch_quota = getattr(self.runner, "fetch_account_quota", None)
        if not callable(fetch_quota):
            return acc, acc.quota, "Quota querying is not supported by active runner"
        ok, data, err = await fetch_quota(acc.build_env())
        if ok and data:
            updated = self.account_store.update_quota(account_id, data)
            assert updated is not None
            return updated, data, ""
        fallback = dict(acc.quota or {})
        fallback["error"] = err or "Failed to fetch quota"
        updated = self.account_store.update_quota(account_id, fallback)
        assert updated is not None
        return updated, fallback, err or "Failed to fetch quota"

    async def refresh_all_quotas(self) -> List[Tuple[AccountRecord, Optional[Dict[str, Any]], str]]:
        """Concurrently refresh quota snapshots for all enabled signed-in accounts."""
        targets = [
            a
            for a in self.account_store.list_accounts()
            if a.enabled and a.status != "needs_login"
        ]
        if not targets:
            return []
        results = await asyncio.gather(
            *(self.refresh_account_quota(a.id) for a in targets),
            return_exceptions=True,
        )
        out: List[Tuple[AccountRecord, Optional[Dict[str, Any]], str]] = []
        for item in results:
            if isinstance(item, tuple):
                out.append(item)
        return out

    async def _acquire_slot(self, timeout_seconds: Optional[float] = None) -> None:
        wait_limit = (
            self.settings.queue_timeout_seconds
            if timeout_seconds is None
            else max(0.05, timeout_seconds)
        )
        self.queued_requests += 1
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=wait_limit,
            )
        except asyncio.TimeoutError as exc:
            raise QueueTimeoutError(
                "Gateway concurrency queue is full; please retry shortly."
            ) from exc
        finally:
            self.queued_requests -= 1

    async def _wait_for_account_slot(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise QueueTimeoutError(
                "Account concurrency limit reached; please retry shortly."
            )
        self.queued_requests += 1
        try:
            async with self._account_cond:
                await asyncio.wait_for(self._account_cond.wait(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise QueueTimeoutError(
                "Account concurrency limit reached; please retry shortly."
            ) from exc
        finally:
            self.queued_requests -= 1

    async def _notify_account_slot_freed(self) -> None:
        async with self._account_cond:
            self._account_cond.notify_all()

    async def _run_auxiliary_summary_call(
        self,
        prompt: str,
        model: str,
        allowed_accounts: Optional[Sequence[str]] = None,
    ) -> str:
        """Run one bounded auxiliary summarization prompt without mutating account cooldowns."""
        candidates = self._ordered_candidates(
            allowed_accounts=allowed_accounts, mutate_cursor=False
        )
        acc = candidates[0] if candidates else None
        stream_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "effort": "low",
            "timeout_seconds": min(45.0, self.settings.request_timeout_seconds),
        }
        if acc is not None and acc.home_dir:
            stream_kwargs["env"] = acc.build_env()
        try:
            chunk_iter = self.runner.stream_prompt(**stream_kwargs)
        except TypeError:
            stream_kwargs.pop("env", None)
            chunk_iter = self.runner.stream_prompt(**stream_kwargs)
        parts: List[str] = []
        async for chunk in chunk_iter:
            if chunk.text_delta:
                parts.append(chunk.text_delta)
        return "".join(parts).strip()

    async def _generate_and_cache_llm_checkpoint(
        self,
        ctx_res: ContextBuildResult,
        target_model: str,
        allowed_accounts: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        """Run deadlock-free bounded map/reduce summarization and store it in SQLite cache."""
        if not ctx_res.compacted_prefix_text or not ctx_res.prefix_hash:
            return None
        summary_model = self.settings.resolve_model(
            self.settings.compaction_model or target_model
        )

        async def _complete(chunk_prompt: str) -> str:
            return await self._run_auxiliary_summary_call(
                chunk_prompt,
                model=summary_model,
                allowed_accounts=allowed_accounts,
            )

        try:
            summary = await summarize_prefix_map_reduce(
                ctx_res.compacted_prefix_text,
                _complete,
                chunk_bytes=min(64_000, max(16_000, self.settings.max_prompt_bytes // 3)),
                max_summary_bytes=self.settings.compaction_checkpoint_max_bytes,
            )
            if summary:
                self.checkpoint_store.put(
                    ctx_res.session_fingerprint,
                    ctx_res.prefix_hash,
                    summary,
                    mode="llm",
                    source_messages=ctx_res.compacted_messages,
                    source_bytes=ctx_res.compacted_prefix_bytes,
                )
                self.context_stats.record_llm_summary(True)
                return summary
            self.context_stats.record_llm_summary(False)
            return None
        except Exception:
            self.context_stats.record_llm_summary(False)
            return None

    def _schedule_background_compaction(
        self,
        ctx_res: ContextBuildResult,
        target_model: str,
        allowed_accounts: Optional[Sequence[str]] = None,
    ) -> None:
        """Schedule deduplicated async LLM checkpoint upgrade (`hybrid` mode, pi-async-compaction)."""
        key = f"{ctx_res.session_fingerprint}:{ctx_res.prefix_hash}"
        existing = self._bg_compaction_tasks.get(key)
        if existing is not None and not existing.done():
            return

        async def _worker() -> None:
            try:
                await self._generate_and_cache_llm_checkpoint(
                    ctx_res,
                    target_model=target_model,
                    allowed_accounts=allowed_accounts,
                )
            finally:
                self._bg_compaction_tasks.pop(key, None)

        task = asyncio.create_task(_worker())
        self._bg_compaction_tasks[key] = task

    async def shutdown(self) -> None:
        """Cancel any in-flight background compaction tasks cleanly."""
        tasks = list(self._bg_compaction_tasks.values())
        self._bg_compaction_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _stage_attachments_helper(
        self, attachments: Sequence[MediaAttachment]
    ) -> Tuple[Optional[Path], List[StagedMedia]]:
        if not attachments:
            return None, []
        return await stage_media_attachments(
            attachments,
            workdir=self.settings.workdir,
            max_files=self.settings.max_media_files,
            max_file_bytes=self.settings.max_media_file_bytes,
            max_total_bytes=self.settings.max_media_total_bytes,
            download_timeout_seconds=self.settings.media_download_timeout_seconds,
            allow_remote_urls=self.settings.allow_remote_media_urls,
        )

    async def _build_full_context_with_optional_llm(
        self,
        req: NormalizedRequest,
        *,
        max_bytes: int,
        staged_media: Sequence[StagedMedia],
        target_model: str,
        allowed_accounts: Optional[Sequence[str]],
        record_stats: bool = True,
    ) -> ContextBuildResult:
        ctx_res = self.context_manager.build_context(
            req,
            max_bytes=max_bytes,
            staged_media=staged_media,
            record_stats=record_stats,
        )
        if (
            ctx_res.compacted_messages > 0
            and not ctx_res.llm_summary_cached
            and ctx_res.compacted_prefix_text
        ):
            if self.context_manager.mode == "llm":
                llm_summary = await self._generate_and_cache_llm_checkpoint(
                    ctx_res,
                    target_model=target_model,
                    allowed_accounts=allowed_accounts,
                )
                if llm_summary:
                    ctx_res = self.context_manager.build_context(
                        req,
                        max_bytes=max_bytes,
                        staged_media=staged_media,
                        override_summary=llm_summary,
                        record_stats=False,
                    )
            elif (
                self.context_manager.mode == "hybrid"
                and ctx_res.compacted_prefix_bytes >= 8_000
            ):
                self._schedule_background_compaction(
                    ctx_res,
                    target_model=target_model,
                    allowed_accounts=allowed_accounts,
                )
        return ctx_res

    async def stream(
        self,
        req: NormalizedRequest,
        allowed_accounts: Optional[Sequence[str]] = None,
        context_out: Optional[List[ContextBuildResult]] = None,
        api_key_id: str = "",
    ) -> AsyncIterator[AgyChunk]:
        """Stage media, resume native `agy --conversation` when possible, or compact & failover."""
        target_model = self.settings.resolve_model(req.model)
        session_fp = compute_session_fingerprint(req) if req.messages else ""
        session_key = (
            compute_session_key(
                api_key_id=api_key_id,
                session_fingerprint=session_fp,
                target_model=target_model,
            )
            if session_fp
            else ""
        )
        sys_tools_hash = compute_sys_tools_hash(req) if req.messages else ""

        matched = self._match_reusable_session(
            req,
            target_model=target_model,
            api_key_id=api_key_id,
            allowed_accounts=allowed_accounts,
        )
        resumed_mode = matched is not None
        resumed_sess: Optional[SessionRecord] = matched[1] if matched else None
        resumed_account: Optional[AccountRecord] = matched[3] if matched else None
        delta_messages: List[ChatMessage] = matched[2] if matched else []

        # Track whether full attachments have been staged (in resumed mode we only stage delta media!)
        staging_dir: Optional[Path] = None
        staged_media: List[StagedMedia] = []
        full_media_staged = False

        if resumed_mode:
            delta_atts = self._filter_attachments_for_messages(req.attachments, delta_messages)
            staging_dir, staged_media = await self._stage_attachments_helper(delta_atts)
            full_media_staged = len(delta_atts) == len(req.attachments)
        else:
            staging_dir, staged_media = await self._stage_attachments_helper(req.attachments)
            full_media_staged = True
            if self.settings.session_reuse_enabled and self._runner_supports_resume() and len(req.messages) > 1:
                self.context_stats.record_session_miss()

        has_global_slot = False
        locked_session_key = False
        completed_cleanly = False
        try:
            if session_key and session_key not in self._active_session_keys:
                self._active_session_keys.add(session_key)
                locked_session_key = True

            current_max_bytes = self.settings.max_prompt_bytes
            if resumed_mode:
                ctx_res = self.context_manager.build_delta_context(
                    req,
                    delta_messages,
                    max_bytes=current_max_bytes,
                    staged_media=staged_media,
                )
            else:
                ctx_res = await self._build_full_context_with_optional_llm(
                    req,
                    max_bytes=current_max_bytes,
                    staged_media=staged_media,
                    target_model=target_model,
                    allowed_accounts=allowed_accounts,
                )

            if context_out is not None:
                context_out.clear()
                context_out.append(ctx_res)

            prompt = ctx_res.prompt
            queue_deadline = time.monotonic() + self.settings.queue_timeout_seconds

            await self._acquire_slot()
            has_global_slot = True
            self.active_requests += 1
            self.total_dispatched += 1
            tried_ids: set[str] = set()
            last_error: Optional[Exception] = None
            overflow_attempts = 0

            async def _fallback_to_full_context() -> None:
                nonlocal resumed_mode, resumed_sess, resumed_account
                nonlocal staging_dir, staged_media, full_media_staged, ctx_res, prompt, current_max_bytes
                resumed_mode = False
                resumed_sess = None
                resumed_account = None
                if session_key:
                    self.session_store.delete(session_key)
                self.context_stats.record_session_fallback()
                if not full_media_staged and req.attachments:
                    cleanup_staging_dir(staging_dir)
                    staging_dir, staged_media = await self._stage_attachments_helper(req.attachments)
                    full_media_staged = True
                current_max_bytes = self.settings.max_prompt_bytes
                ctx_res = await self._build_full_context_with_optional_llm(
                    req,
                    max_bytes=current_max_bytes,
                    staged_media=staged_media,
                    target_model=target_model,
                    allowed_accounts=allowed_accounts,
                    record_stats=False,
                )
                prompt = ctx_res.prompt
                if context_out is not None:
                    context_out.clear()
                    context_out.append(ctx_res)

            while True:
                if resumed_mode and resumed_account is not None and resumed_sess is not None:
                    # Wait for pinned account slot if busy, otherwise route directly to pinned account
                    if (
                        resumed_account.max_concurrency > 0
                        and self._active_by_account[resumed_account.id] >= resumed_account.max_concurrency
                    ):
                        self.active_requests -= 1
                        self._semaphore.release()
                        has_global_slot = False
                        await self._wait_for_account_slot(queue_deadline)
                        remaining = queue_deadline - time.monotonic()
                        if remaining <= 0:
                            raise QueueTimeoutError(
                                "Account concurrency limit reached; please retry shortly."
                            )
                        await self._acquire_slot(timeout_seconds=remaining)
                        has_global_slot = True
                        self.active_requests += 1
                        continue
                    acc = resumed_account
                    target_conv_id: Optional[str] = resumed_sess.agy_conversation_id
                else:
                    candidates, any_at_capacity = self._select_candidates(
                        allowed_accounts=allowed_accounts,
                        excluded_ids=tried_ids,
                    )
                    if not candidates:
                        if any_at_capacity:
                            # Release global slot while waiting for a busy per-account slot
                            self.active_requests -= 1
                            self._semaphore.release()
                            has_global_slot = False
                            await self._wait_for_account_slot(queue_deadline)
                            remaining = queue_deadline - time.monotonic()
                            if remaining <= 0:
                                raise QueueTimeoutError(
                                    "Account concurrency limit reached; please retry shortly."
                                )
                            await self._acquire_slot(timeout_seconds=remaining)
                            has_global_slot = True
                            self.active_requests += 1
                            continue
                        if last_error is not None:
                            raise last_error
                        raise AgyExecutionError(
                            "No available logged-in Antigravity accounts for this request."
                        )
                    acc = candidates[0]
                    target_conv_id = None

                tried_ids.add(acc.id)
                emitted_any = False
                emitted_text_parts: List[str] = []
                final_usage = UsageInfo()
                final_conv_id = target_conv_id or ""
                self._active_by_account[acc.id] += 1

                try:
                    stream_kwargs: Dict[str, Any] = {
                        "prompt": prompt,
                        "model": target_model,
                        "effort": req.effort,
                        "timeout_seconds": self.settings.request_timeout_seconds,
                    }
                    if acc.home_dir:
                        stream_kwargs["env"] = acc.build_env()
                    if staging_dir is not None:
                        stream_kwargs["cwd"] = staging_dir
                    if target_conv_id is not None:
                        stream_kwargs["conversation_id"] = target_conv_id
                    try:
                        chunk_iter = self.runner.stream_prompt(**stream_kwargs)
                    except TypeError:
                        stream_kwargs.pop("cwd", None)
                        stream_kwargs.pop("conversation_id", None)
                        chunk_iter = self.runner.stream_prompt(**stream_kwargs)

                    async for chunk in chunk_iter:
                        if chunk.text_delta:
                            emitted_any = True
                            emitted_text_parts.append(chunk.text_delta)
                        elif chunk.thinking_delta:
                            emitted_any = True
                        if chunk.conversation_id:
                            final_conv_id = chunk.conversation_id
                        if chunk.done:
                            final_usage = chunk.usage
                        yield chunk

                    completed_cleanly = True
                    detected_email = None if acc.email else detect_email_from_logs(acc.resolved_home())
                    self.account_store.record_success(
                        acc.id,
                        input_tokens=final_usage.input_tokens,
                        output_tokens=final_usage.output_tokens,
                        detected_email=detected_email,
                    )

                    # Persist updated multi-turn session cursor for next turn's delta reuse!
                    if (
                        self.settings.session_reuse_enabled
                        and self._runner_supports_resume()
                        and session_key
                        and final_conv_id
                        and req.messages
                    ):
                        assistant_fp = compute_assistant_fingerprint_from_output(
                            "".join(emitted_text_parts),
                            has_tools=bool(req.tools),
                        )
                        with contextlib.suppress(Exception):
                            self.session_store.upsert(
                                session_key=session_key,
                                session_fingerprint=session_fp,
                                account_id=acc.id,
                                agy_conversation_id=final_conv_id,
                                model=target_model,
                                sys_tools_hash=sys_tools_hash,
                                sent_message_count=len(req.messages),
                                sent_prefix_hash=compute_messages_prefix_hash(req.messages),
                                last_assistant_fingerprint=assistant_fp,
                                delta_bytes=ctx_res.final_bytes,
                                increment_turn=resumed_mode,
                            )
                    return
                except AgySessionMismatchError:
                    # `--conversation <id>` was not found on disk / expired -> seamless full-context retry on same account!
                    if emitted_any:
                        raise
                    await _fallback_to_full_context()
                    tried_ids.discard(acc.id)
                    continue
                except AgyExecutionError as exc:
                    if resumed_mode and not emitted_any:
                        # Pre-first-byte error on resumed session: if overflow, retry full compacted context on same account;
                        # otherwise mark account failure and failover with full context!
                        is_ovf = is_context_overflow_error(str(exc))
                        if not is_ovf:
                            needs_login, cooldown = _classify_account_error(
                                str(exc), self.settings.account_cooldown_seconds
                            )
                            self.account_store.record_failure(
                                acc.id,
                                str(exc),
                                needs_login=needs_login,
                                cooldown_seconds=cooldown,
                            )
                            last_error = exc
                        else:
                            tried_ids.discard(acc.id)
                        await _fallback_to_full_context()
                        continue

                    if is_context_overflow_error(str(exc)):
                        # Context overflow is a prompt-size issue, NOT an account health failure!
                        if not emitted_any and overflow_attempts < self.settings.overflow_max_retries:
                            overflow_attempts += 1
                            current_max_bytes = max(8_192, int(current_max_bytes * 0.6))
                            ctx_res = self.context_manager.build_context(
                                req,
                                max_bytes=current_max_bytes,
                                staged_media=staged_media,
                                is_overflow_recovery=True,
                                record_stats=False,
                            )
                            prompt = ctx_res.prompt
                            if context_out is not None:
                                context_out.clear()
                                context_out.append(ctx_res)
                            self.context_stats.record_overflow_recovery()
                            tried_ids.discard(acc.id)
                            continue
                        self.context_stats.record_overflow_failure()
                        raise ContextOverflowError(
                            f"Context length exceeded after {overflow_attempts} automatic compaction "
                            f"recovery attempt(s): {exc}"
                        ) from exc

                    needs_login, cooldown = _classify_account_error(
                        str(exc), self.settings.account_cooldown_seconds
                    )
                    self.account_store.record_failure(
                        acc.id,
                        str(exc),
                        needs_login=needs_login,
                        cooldown_seconds=cooldown,
                    )
                    last_error = exc
                    if emitted_any:
                        raise
                    # Pre-first-byte failure: seamlessly failover to the next eligible account
                    continue
                except Exception:
                    self.total_errors += 1
                    raise
                finally:
                    self._active_by_account[acc.id] = max(
                        0, self._active_by_account[acc.id] - 1
                    )
                    await self._notify_account_slot_freed()
        except (AgyExecutionError, ContextOverflowError):
            self.total_errors += 1
            raise
        finally:
            if resumed_mode and not completed_cleanly and session_key:
                # If client disconnected mid-turn during a resumed session, invalidate cursor
                # so the next turn never hits a half-written conversation state.
                with contextlib.suppress(Exception):
                    self.session_store.delete(session_key)
            if locked_session_key and session_key:
                self._active_session_keys.discard(session_key)
            if has_global_slot:
                self.active_requests -= 1
                self._semaphore.release()
            cleanup_staging_dir(staging_dir)

    async def complete_with_meta(
        self,
        req: NormalizedRequest,
        allowed_accounts: Optional[Sequence[str]] = None,
        api_key_id: str = "",
    ) -> CompletionResult:
        """Acquire a concurrency slot and return full response, usage, finish reason, and context info."""
        chunks: List[str] = []
        usage = UsageInfo()
        conv_id = ""
        finish_reason = "stop"
        ctx_holder: List[ContextBuildResult] = []
        async for chunk in self.stream(
            req,
            allowed_accounts=allowed_accounts,
            context_out=ctx_holder,
            api_key_id=api_key_id,
        ):
            if chunk.text_delta:
                chunks.append(chunk.text_delta)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.done:
                usage = chunk.usage
                conv_id = chunk.conversation_id
        return CompletionResult(
            text="".join(chunks),
            usage=usage,
            conversation_id=conv_id,
            finish_reason=finish_reason,
            context_result=ctx_holder[0] if ctx_holder else None,
        )

    async def complete(
        self,
        req: NormalizedRequest,
        allowed_accounts: Optional[Sequence[str]] = None,
        api_key_id: str = "",
    ) -> Tuple[str, UsageInfo, str]:
        """Acquire a concurrency slot and return full response text, usage, and conversation ID."""
        res = await self.complete_with_meta(
            req, allowed_accounts=allowed_accounts, api_key_id=api_key_id
        )
        return res.text, res.usage, res.conversation_id

    def active_counts_by_account(self) -> Dict[str, int]:
        return dict(self._active_by_account)

    def context_stats_snapshot(self) -> Dict[str, Any]:
        cached_count = 0
        active_sessions = 0
        with contextlib.suppress(Exception):
            cached_count = self.checkpoint_store.count()
        with contextlib.suppress(Exception):
            active_sessions = self.session_store.count()
        return self.context_stats.snapshot(
            mode=self.context_manager.mode,
            cached_checkpoints=cached_count,
            session_reuse_enabled=self.settings.session_reuse_enabled,
            active_sessions=active_sessions,
        )

    def stats(self) -> Dict[str, Any]:
        accounts = self.account_store.list_accounts()
        now = time.time()
        healthy_count = sum(
            1
            for a in accounts
            if a.enabled and a.status != "needs_login" and a.cooldown_until <= now
        )
        min_quotas = [
            float(a.quota["min_remaining_percent"])
            for a in accounts
            if a.enabled
            and isinstance(a.quota, dict)
            and a.quota.get("min_remaining_percent") is not None
        ]
        low_quota_count = sum(1 for pct in min_quotas if pct < 20.0)
        return {
            "database_backend": self.account_store.backend_name,
            "max_concurrency": self.settings.max_concurrency,
            "queue_timeout_seconds": self.settings.queue_timeout_seconds,
            "request_timeout_seconds": self.settings.request_timeout_seconds,
            "routing_strategy": self.account_store.get_routing_strategy(
                self.settings.routing_strategy
            ),
            "accounts_total": len(accounts),
            "accounts_healthy": healthy_count,
            "lowest_quota_percent": min(min_quotas) if min_quotas else None,
            "accounts_low_quota": low_quota_count,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "total_dispatched": self.total_dispatched,
            "total_errors": self.total_errors,
            "context": self.context_stats_snapshot(),
        }
