"""FastAPI application exposing OpenAI, Gemini, Anthropic, multi-account admin, and UI routes."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from agy_gateway import __version__
from agy_gateway.accounts import (
    AccountStore,
    OAuthLoginManager,
    VALID_ROUTING_STRATEGIES,
)
from agy_gateway.adapters import (
    format_anthropic_response,
    format_gemini_models,
    format_gemini_response,
    format_openai_chat_completion,
    format_openai_models,
    iter_anthropic_sse,
    iter_gemini_sse,
    iter_openai_sse,
    parse_anthropic_request,
    parse_gemini_request,
    parse_openai_request,
    sse_data,
    sse_event,
)
from agy_gateway.agy_client import AgyChunk, AgyExecutionError, UsageInfo
from agy_gateway.config import Settings, get_settings
from agy_gateway.db import close_all_pools
from agy_gateway.dispatcher import (
    AgyDispatcher,
    CompletionResult,
    ContextBuildResult,
    ContextOverflowError,
    NormalizedRequest,
    QueueTimeoutError,
    estimate_tokens,
)
from agy_gateway.housekeeping import prune_agy_artifacts
from agy_gateway.keys import ApiKeyRecord, KeyAuthError, KeyStore
from agy_gateway.media import MediaError


class CreateKeyBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    rpm_limit: int = Field(default=30, ge=0)
    allowed_models: List[str] = Field(default_factory=list)
    allowed_accounts: List[str] = Field(default_factory=list)


class UpdateKeyBody(BaseModel):
    enabled: Optional[bool] = None
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    rpm_limit: Optional[int] = Field(default=None, ge=0)
    allowed_models: Optional[List[str]] = None
    allowed_accounts: Optional[List[str]] = None


class CreateAccountBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    priority: int = Field(default=100, ge=0, le=10000)
    max_concurrency: int = Field(default=6, ge=0, le=64)


class UpdateAccountBody(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    email: Optional[str] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = Field(default=None, ge=0, le=10000)
    max_concurrency: Optional[int] = Field(default=None, ge=0, le=64)
    clear_cooldown: bool = False


class SubmitLoginBody(BaseModel):
    session_id: str = Field(..., min_length=1)
    code_or_url: str = Field(..., min_length=1)


class CancelLoginBody(BaseModel):
    session_id: str = Field(..., min_length=1)


class UpdateRoutingBody(BaseModel):
    routing_strategy: str


def extract_api_key(request: Request) -> Optional[str]:
    """Extract client API key from Authorization Bearer, x-api-key, x-goog-api-key, or ?key=."""
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    if header_key := request.headers.get("x-api-key"):
        return header_key.strip()
    if goog_key := request.headers.get("x-goog-api-key"):
        return goog_key.strip()
    if query_key := request.query_params.get("key"):
        return query_key.strip()
    return None


def _auth_or_raise(
    request: Request,
    key_store: KeyStore,
    model: Optional[str] = None,
    check_rate_limit: bool = True,
) -> ApiKeyRecord:
    raw_key = extract_api_key(request)
    try:
        return key_store.authenticate(
            raw_key=raw_key,
            model=model,
            check_rate_limit=check_rate_limit,
        )
    except KeyAuthError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error": {"message": exc.message, "type": "auth_error"}},
        ) from exc


def _require_admin(request: Request, settings: Settings) -> None:
    if not settings.admin_key:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": "Admin HTTP API is disabled (set AGY_GATEWAY_ADMIN_KEY to enable).",
                    "type": "admin_disabled",
                }
            },
        )
    raw = extract_api_key(request) or ""
    if not hmac.compare_digest(raw, settings.admin_key):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid admin key", "type": "auth_error"}},
        )


def _extract_session_header(request: Request) -> Optional[str]:
    for hdr in ("x-session-id", "x-conversation-id", "x-thread-id"):
        val = request.headers.get(hdr)
        if val and val.strip():
            return val.strip()
    return None


def _context_headers(ctx_res: Optional[ContextBuildResult]) -> Dict[str, str]:
    if ctx_res is None:
        return {}
    return {
        "X-Agy-Session-Reused": "true" if ctx_res.session_reused else "false",
        "X-Agy-Context-Compacted": "true" if ctx_res.compacted else "false",
        "X-Agy-Context-Stage": ctx_res.compaction_stage,
        "X-Agy-Context-Original-Bytes": str(ctx_res.original_bytes),
        "X-Agy-Context-Final-Bytes": str(ctx_res.final_bytes),
        "X-Agy-Context-Compacted-Turns": str(ctx_res.compacted_messages),
        "X-Agy-Context-Kept-Turns": str(ctx_res.kept_messages),
    }


def _overflow_json_response(exc: ContextOverflowError, protocol: str) -> JSONResponse:
    """Build a protocol-native HTTP 400 overflow payload recognized by Pi/DSH/Paseo sniffers."""
    msg = str(exc)
    if protocol == "anthropic":
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"prompt is too long: {msg}",
                },
            },
        )
    if protocol == "gemini":
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": 400,
                    "message": msg,
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": msg,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        },
    )


def _estimate_request_input_tokens(norm_req: NormalizedRequest) -> int:
    total = estimate_tokens(norm_req.system_prompt or "")
    for m in norm_req.messages:
        total += estimate_tokens(m.content)
        for tr in m.tool_results:
            total += estimate_tokens(tr.content)
    return total


async def _complete_with_accounting(
    request: Request,
    dispatcher: AgyDispatcher,
    key_store: KeyStore,
    key_record: ApiKeyRecord,
    norm_req: NormalizedRequest,
) -> CompletionResult:
    """Run one non-streaming request, cancel if client drops, map errors, and log usage."""
    task = asyncio.create_task(
        dispatcher.complete_with_meta(
            norm_req,
            allowed_accounts=key_record.allowed_accounts,
            api_key_id=key_record.id,
        )
    )
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=1.0)
            if task in done:
                break
            if await request.is_disconnected():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise HTTPException(status_code=499, detail="Client disconnected")
        result = task.result()
    except ContextOverflowError:
        raise
    except MediaError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
        ) from exc
    except QueueTimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": str(exc), "type": "queue_timeout"}},
        ) from exc
    except AgyExecutionError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": {"message": str(exc), "type": "upstream_error"}},
        ) from exc
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    key_store.record_usage(
        key_record.id, result.usage.input_tokens, result.usage.output_tokens
    )
    return result


async def _stream_with_accounting(
    dispatcher: AgyDispatcher,
    key_store: KeyStore,
    key_record: ApiKeyRecord,
    norm_req: NormalizedRequest,
) -> AsyncIterator[AgyChunk]:
    """Forward chunks from AgyDispatcher and persist usage accounting when stream ends."""
    final_usage = UsageInfo()
    try:
        async for chunk in dispatcher.stream(
            norm_req,
            allowed_accounts=key_record.allowed_accounts,
            api_key_id=key_record.id,
        ):
            if chunk.done:
                final_usage = chunk.usage
            yield chunk
    finally:
        key_store.record_usage(
            key_record.id,
            input_tokens=final_usage.input_tokens,
            output_tokens=final_usage.output_tokens,
        )


async def _safe_sse_stream(
    sse_iter_factory: Callable[[], AsyncIterator[str]],
    keepalive_seconds: float = 10.0,
    protocol: str = "openai",
) -> AsyncIterator[str]:
    """Stream SSE frames with periodic keepalive comments and protocol-aware error handling."""
    it = sse_iter_factory()
    next_task: Optional[asyncio.Task[str]] = None
    try:
        while True:
            next_task = asyncio.create_task(anext(it))  # type: ignore[arg-type]
            while not next_task.done():
                wait_timeout = keepalive_seconds if keepalive_seconds > 0 else None
                done, _ = await asyncio.wait({next_task}, timeout=wait_timeout)
                if next_task in done:
                    break
                yield ": keepalive\n\n"
            try:
                frame = next_task.result()
            except StopAsyncIteration:
                break
            next_task = None
            yield frame
    except (AgyExecutionError, QueueTimeoutError, MediaError, ContextOverflowError) as exc:
        is_queue = isinstance(exc, QueueTimeoutError)
        is_media = isinstance(exc, MediaError)
        is_overflow = isinstance(exc, ContextOverflowError)
        if protocol == "anthropic":
            err_type = (
                "invalid_request_error"
                if (is_media or is_overflow)
                else ("overloaded_error" if is_queue else "api_error")
            )
            err_msg = f"prompt is too long: {exc}" if is_overflow else str(exc)
            yield sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": err_type,
                        "message": err_msg,
                    },
                },
            )
        elif protocol == "gemini":
            yield sse_data(
                {
                    "error": {
                        "code": 400 if (is_media or is_overflow) else (503 if is_queue else 502),
                        "message": str(exc),
                        "status": (
                            "INVALID_ARGUMENT"
                            if (is_media or is_overflow)
                            else ("UNAVAILABLE" if is_queue else "INTERNAL")
                        ),
                    }
                }
            )
        else:
            err_obj: Dict[str, Any] = {
                "message": str(exc),
                "type": (
                    "invalid_request_error"
                    if (is_media or is_overflow)
                    else ("queue_timeout" if is_queue else "gateway_error")
                ),
            }
            if is_overflow:
                err_obj["code"] = "context_length_exceeded"
            yield sse_data({"error": err_obj})
            yield "data: [DONE]\n\n"
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
        aclose = getattr(it, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await aclose()


def _serialize_account(
    acc: Any, active_counts: Dict[str, int], now: float
) -> Dict[str, Any]:
    data = asdict(acc)
    cooldown_remaining = max(0, int(acc.cooldown_until - now))
    data["active_requests"] = active_counts.get(acc.id, 0)
    data["cooldown_remaining_seconds"] = cooldown_remaining
    if acc.status == "cooldown" and cooldown_remaining == 0:
        data["status"] = "logged_in"
    return data


def create_app(
    settings: Optional[Settings] = None,
    key_store: Optional[KeyStore] = None,
    dispatcher: Optional[AgyDispatcher] = None,
    account_store: Optional[AccountStore] = None,
    login_manager: Optional[OAuthLoginManager] = None,
) -> FastAPI:
    """Create configured FastAPI app (supports injecting fakes in tests)."""
    cfg = settings or get_settings()
    cfg.ensure_dirs()
    store = key_store or KeyStore(cfg.db_path, database_url=cfg.database_url)
    acc_store = account_store or (
        dispatcher.account_store
        if dispatcher is not None
        else AccountStore(cfg.db_path, cfg.accounts_dir, database_url=cfg.database_url)
    )
    dispatch = dispatcher or AgyDispatcher(cfg, account_store=acc_store)
    oauth_mgr = login_manager or OAuthLoginManager(cfg.agy_bin, cfg.workdir)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async def _housekeeper() -> None:
            while True:
                await asyncio.sleep(max(30.0, cfg.housekeeping_interval_seconds))
                with contextlib.suppress(Exception):
                    homes = [a.resolved_home() for a in acc_store.list_accounts()]
                    await asyncio.to_thread(dispatch.session_store.prune_expired)
                    protected_cids = await asyncio.to_thread(
                        dispatch.session_store.list_active_conversation_ids
                    )
                    await asyncio.to_thread(
                        prune_agy_artifacts,
                        homes,
                        cfg.artifact_retention_minutes,
                        cfg.workdir,
                        protected_cids,
                    )
                    await asyncio.to_thread(dispatch.checkpoint_store.prune_expired)
                    await oauth_mgr.cleanup_expired()

        async def _quota_poller() -> None:
            if cfg.quota_poll_interval_seconds <= 0:
                return
            if not callable(getattr(dispatch.runner, "fetch_account_quota", None)):
                return
            await asyncio.sleep(3.0)
            while True:
                with contextlib.suppress(Exception):
                    await dispatch.refresh_all_quotas()
                await asyncio.sleep(max(30.0, cfg.quota_poll_interval_seconds))

        bg_task = asyncio.create_task(_housekeeper())
        quota_task = asyncio.create_task(_quota_poller())
        try:
            yield
        finally:
            bg_task.cancel()
            quota_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await bg_task
            with contextlib.suppress(asyncio.CancelledError):
                await quota_task
            await dispatch.shutdown()
            close_all_pools()

    app = FastAPI(title="Antigravity API Gateway", version=__version__, lifespan=lifespan)
    app.state.settings = cfg
    app.state.key_store = store
    app.state.account_store = acc_store
    app.state.dispatcher = dispatch
    app.state.login_manager = oauth_mgr

    ui_html = Path(__file__).with_name("ui.html").read_text(encoding="utf-8")

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui", status_code=302)

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def key_admin_ui() -> HTMLResponse:
        return HTMLResponse(ui_html)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", **dispatch.stats()}

    # --- OpenAI Compatible Endpoints ---

    @app.get("/v1/models")
    async def openai_list_models(request: Request) -> Dict[str, Any]:
        _auth_or_raise(request, store, check_rate_limit=False)
        models = await dispatch.list_models()
        return format_openai_models(models, cfg.model_aliases)

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Any:
        body = await request.json()
        requested_model = str(body.get("model") or cfg.default_model)
        resolved_model = cfg.resolve_model(requested_model)
        key_record = _auth_or_raise(request, store, model=resolved_model)

        try:
            norm_req = parse_openai_request(
                body,
                default_model=cfg.default_model,
                session_id=_extract_session_header(request),
            )
        except MediaError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
            ) from exc
        has_tools = bool(norm_req.tools)

        if norm_req.stream:
            stream_opts = body.get("stream_options") or {}
            include_usage = bool(stream_opts.get("include_usage", False))

            def make_sse() -> AsyncIterator[str]:
                chunks = _stream_with_accounting(dispatch, store, key_record, norm_req)
                return iter_openai_sse(
                    chunks=chunks,
                    model=requested_model,
                    has_tools=has_tools,
                    include_usage=include_usage,
                )

            return StreamingResponse(
                _safe_sse_stream(
                    make_sse,
                    keepalive_seconds=cfg.sse_keepalive_seconds,
                    protocol="openai",
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            comp = await _complete_with_accounting(
                request, dispatch, store, key_record, norm_req
            )
        except ContextOverflowError as exc:
            return _overflow_json_response(exc, protocol="openai")

        return JSONResponse(
            format_openai_chat_completion(
                text=comp.text,
                model=requested_model,
                usage=comp.usage,
                has_tools=has_tools,
                finish_reason=comp.finish_reason,
            ),
            headers=_context_headers(comp.context_result),
        )

    # --- Anthropic Messages Compatible Endpoint ---

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Any:
        body = await request.json()
        requested_model = str(body.get("model") or cfg.default_model)
        resolved_model = cfg.resolve_model(requested_model)
        key_record = _auth_or_raise(request, store, model=resolved_model)

        try:
            norm_req = parse_anthropic_request(
                body,
                default_model=cfg.default_model,
                session_id=_extract_session_header(request),
            )
        except MediaError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
            ) from exc
        has_tools = bool(norm_req.tools)

        if norm_req.stream:
            est_in = _estimate_request_input_tokens(norm_req)

            def make_sse() -> AsyncIterator[str]:
                chunks = _stream_with_accounting(dispatch, store, key_record, norm_req)
                return iter_anthropic_sse(
                    chunks=chunks,
                    model=requested_model,
                    has_tools=has_tools,
                    estimated_input_tokens=est_in,
                )

            return StreamingResponse(
                _safe_sse_stream(
                    make_sse,
                    keepalive_seconds=cfg.sse_keepalive_seconds,
                    protocol="anthropic",
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            comp = await _complete_with_accounting(
                request, dispatch, store, key_record, norm_req
            )
        except ContextOverflowError as exc:
            return _overflow_json_response(exc, protocol="anthropic")

        return JSONResponse(
            format_anthropic_response(
                text=comp.text,
                model=requested_model,
                usage=comp.usage,
                has_tools=has_tools,
                finish_reason=comp.finish_reason,
            ),
            headers=_context_headers(comp.context_result),
        )

    # --- Gemini Native REST Compatible Endpoints ---

    @app.get("/v1beta/models")
    async def gemini_list_models(request: Request) -> Dict[str, Any]:
        _auth_or_raise(request, store, check_rate_limit=False)
        models = await dispatch.list_models()
        return format_gemini_models(models, cfg.model_aliases)

    @app.post("/v1beta/models/{model_action:path}")
    async def gemini_generate_content(model_action: str, request: Request) -> Any:
        if ":" not in model_action:
            raise HTTPException(status_code=404, detail="Unsupported Gemini model action")
        raw_model, action = model_action.rsplit(":", 1)
        if action not in ("generateContent", "streamGenerateContent"):
            raise HTTPException(status_code=404, detail=f"Unsupported action: {action}")

        resolved_model = cfg.resolve_model(raw_model)
        key_record = _auth_or_raise(request, store, model=resolved_model)
        body = await request.json()
        is_stream = action == "streamGenerateContent"
        try:
            norm_req = parse_gemini_request(
                model=raw_model,
                body=body,
                stream=is_stream,
                session_id=_extract_session_header(request),
            )
        except MediaError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
            ) from exc
        has_tools = bool(norm_req.tools)

        if is_stream and request.query_params.get("alt") == "sse":

            def make_sse() -> AsyncIterator[str]:
                chunks = _stream_with_accounting(dispatch, store, key_record, norm_req)
                return iter_gemini_sse(chunks=chunks, has_tools=has_tools)

            return StreamingResponse(
                _safe_sse_stream(
                    make_sse,
                    keepalive_seconds=cfg.sse_keepalive_seconds,
                    protocol="gemini",
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            comp = await _complete_with_accounting(
                request, dispatch, store, key_record, norm_req
            )
        except ContextOverflowError as exc:
            return _overflow_json_response(exc, protocol="gemini")

        gem_finish = "MAX_TOKENS" if comp.finish_reason == "length" else "STOP"
        payload = format_gemini_response(
            text=comp.text,
            usage=comp.usage,
            finish_reason=gem_finish,
            has_tools=has_tools,
        )
        headers = _context_headers(comp.context_result)
        if is_stream:
            return JSONResponse([payload], headers=headers)
        return JSONResponse(payload, headers=headers)

    # --- Admin Stats, Keys & Models ---

    @app.get("/admin/stats")
    async def admin_stats(request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        return {
            "dispatcher": dispatch.stats(),
            "keys_count": len(store.list_keys()),
        }

    @app.get("/admin/models")
    async def admin_models(request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        force_refresh = request.query_params.get("refresh") in ("1", "true", "yes")
        models = await dispatch.list_models(force_refresh=force_refresh)
        return {
            "models": [asdict(m) for m in models],
            "aliases": cfg.model_aliases,
        }

    @app.get("/admin/keys")
    async def admin_list_keys(request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        return {"keys": [asdict(k) for k in store.list_keys()]}

    @app.post("/admin/keys")
    async def admin_create_key(body: CreateKeyBody, request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        record, raw_key = store.create_key(
            name=body.name,
            rpm_limit=body.rpm_limit,
            allowed_models=body.allowed_models,
            allowed_accounts=body.allowed_accounts,
        )
        return {"key": asdict(record), "api_key": raw_key}

    @app.patch("/admin/keys/{key_id}")
    async def admin_update_key(
        key_id: str, body: UpdateKeyBody, request: Request
    ) -> Dict[str, Any]:
        _require_admin(request, cfg)
        record = store.update_key(
            key_id,
            name=body.name,
            rpm_limit=body.rpm_limit,
            allowed_models=body.allowed_models,
            allowed_accounts=body.allowed_accounts,
            enabled=body.enabled,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="API key not found")
        return {"key": asdict(record)}

    @app.delete("/admin/keys/{key_id}")
    async def admin_delete_key(key_id: str, request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        deleted = store.delete_key(key_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="API key not found")
        return {"deleted": key_id}

    # --- Admin Multi-Account Pool, Routing & Interactive OAuth Login ---

    @app.get("/admin/accounts")
    async def admin_list_accounts(request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        now = time.time()
        active_counts = dispatch.active_counts_by_account()
        return {
            "routing_strategy": acc_store.get_routing_strategy(cfg.routing_strategy),
            "valid_strategies": list(VALID_ROUTING_STRATEGIES),
            "accounts": [
                _serialize_account(a, active_counts, now)
                for a in acc_store.list_accounts()
            ],
        }

    @app.post("/admin/accounts")
    async def admin_create_account(
        body: CreateAccountBody, request: Request
    ) -> Dict[str, Any]:
        _require_admin(request, cfg)
        acc = acc_store.create_account(
            name=body.name,
            priority=body.priority,
            max_concurrency=body.max_concurrency,
        )
        return {
            "account": _serialize_account(acc, dispatch.active_counts_by_account(), time.time())
        }

    @app.patch("/admin/accounts/{account_id}")
    async def admin_update_account(
        account_id: str, body: UpdateAccountBody, request: Request
    ) -> Dict[str, Any]:
        _require_admin(request, cfg)
        kwargs: Dict[str, Any] = {
            "name": body.name,
            "email": body.email,
            "enabled": body.enabled,
            "priority": body.priority,
            "max_concurrency": body.max_concurrency,
        }
        if body.clear_cooldown:
            kwargs["cooldown_until"] = 0.0
            kwargs["status"] = "logged_in"
            kwargs["clear_error"] = True
        updated = acc_store.update_account(account_id, **kwargs)
        if updated is None:
            raise HTTPException(status_code=404, detail="Account not found")
        return {
            "account": _serialize_account(
                updated, dispatch.active_counts_by_account(), time.time()
            )
        }

    @app.delete("/admin/accounts/{account_id}")
    async def admin_delete_account(account_id: str, request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        try:
            deleted = acc_store.delete_account(account_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Account not found")
        return {"deleted": account_id}

    @app.post("/admin/accounts/check-all")
    async def admin_check_all_accounts(request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        accounts = [a for a in acc_store.list_accounts() if a.enabled]

        async def _probe_one(acc_id: str) -> Dict[str, Any]:
            try:
                updated, models, err = await dispatch.probe_and_update_account(acc_id)
                return {
                    "account": _serialize_account(
                        updated, dispatch.active_counts_by_account(), time.time()
                    ),
                    "models_count": len(models),
                    "error": err or None,
                }
            except Exception as exc:
                return {"account_id": acc_id, "models_count": 0, "error": str(exc)}

        results = list(await asyncio.gather(*(_probe_one(a.id) for a in accounts)))
        healthy = sum(
            1
            for r in results
            if isinstance(r.get("account"), dict) and r["account"].get("status") == "logged_in"
        )
        return {
            "checked": len(results),
            "healthy": healthy,
            "results": results,
        }

    @app.post("/admin/accounts/quota/refresh")
    async def admin_refresh_all_quotas(request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        results = await dispatch.refresh_all_quotas()
        now = time.time()
        active_counts = dispatch.active_counts_by_account()
        refreshed = sum(1 for _, _, err in results if not err)
        failed = sum(1 for _, _, err in results if err)
        return {
            "refreshed": refreshed,
            "failed": failed,
            "accounts": [
                _serialize_account(a, active_counts, now)
                for a in acc_store.list_accounts()
            ],
        }

    @app.post("/admin/accounts/{account_id}/quota")
    async def admin_refresh_account_quota(
        account_id: str, request: Request
    ) -> Dict[str, Any]:
        _require_admin(request, cfg)
        try:
            updated, quota, err = await dispatch.refresh_account_quota(account_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "account": _serialize_account(
                updated, dispatch.active_counts_by_account(), time.time()
            ),
            "quota": quota,
            "error": err or None,
        }

    @app.post("/admin/accounts/{account_id}/check")
    async def admin_check_account(account_id: str, request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        try:
            updated, models, err = await dispatch.probe_and_update_account(account_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "account": _serialize_account(
                updated, dispatch.active_counts_by_account(), time.time()
            ),
            "models_count": len(models),
            "error": err or None,
        }

    @app.post("/admin/accounts/{account_id}/login/start")
    async def admin_login_start(account_id: str, request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        acc = acc_store.get_account(account_id)
        if acc is None:
            raise HTTPException(status_code=404, detail="Account not found")
        try:
            session_id, auth_url, expires_at = await oauth_mgr.start_login(acc)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "account_id": account_id,
            "session_id": session_id,
            "auth_url": auth_url,
            "expires_at": expires_at,
        }

    @app.post("/admin/accounts/{account_id}/login/submit")
    async def admin_login_submit(
        account_id: str, body: SubmitLoginBody, request: Request
    ) -> Dict[str, Any]:
        _require_admin(request, cfg)
        acc = acc_store.get_account(account_id)
        if acc is None:
            raise HTTPException(status_code=404, detail="Account not found")
        try:
            sess_acc_id, output = await oauth_mgr.submit_code(
                body.session_id, body.code_or_url
            )
        except KeyError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if sess_acc_id != account_id:
            raise HTTPException(status_code=400, detail="Login session does not match account_id")

        updated, models, err = await dispatch.probe_and_update_account(account_id)
        return {
            "account": _serialize_account(
                updated, dispatch.active_counts_by_account(), time.time()
            ),
            "models_count": len(models),
            "login_output": output,
            "error": err or None,
        }

    @app.post("/admin/accounts/{account_id}/login/cancel")
    async def admin_login_cancel(
        account_id: str, body: CancelLoginBody, request: Request
    ) -> Dict[str, Any]:
        _require_admin(request, cfg)
        cancelled = await oauth_mgr.cancel_login(body.session_id)
        return {"account_id": account_id, "cancelled": cancelled}

    @app.put("/admin/routing")
    async def admin_update_routing(
        body: UpdateRoutingBody, request: Request
    ) -> Dict[str, Any]:
        _require_admin(request, cfg)
        try:
            strategy = acc_store.set_routing_strategy(body.routing_strategy)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"routing_strategy": strategy}

    @app.post("/admin/housekeeping")
    async def admin_housekeeping(request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        homes = [a.resolved_home() for a in acc_store.list_accounts()]
        expired_sessions = await asyncio.to_thread(dispatch.session_store.prune_expired)
        protected_cids = await asyncio.to_thread(
            dispatch.session_store.list_active_conversation_ids
        )
        report = await asyncio.to_thread(
            prune_agy_artifacts,
            homes,
            cfg.artifact_retention_minutes,
            cfg.workdir,
            protected_cids,
        )
        expired_checkpoints = await asyncio.to_thread(
            dispatch.checkpoint_store.prune_expired
        )
        out = asdict(report)
        out["pruned_checkpoints"] = expired_checkpoints
        out["pruned_sessions"] = expired_sessions
        out["protected_sessions"] = len(protected_cids)
        return out

    @app.get("/admin/context/stats")
    async def admin_context_stats(request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        return dispatch.context_stats_snapshot()

    @app.post("/admin/context/cache/clear")
    async def admin_clear_context_cache(request: Request) -> Dict[str, Any]:
        _require_admin(request, cfg)
        cleared = await asyncio.to_thread(dispatch.checkpoint_store.clear)
        cleared_sessions = 0
        if request.query_params.get("include_sessions") in ("1", "true", "yes"):
            cleared_sessions = await asyncio.to_thread(dispatch.session_store.clear)
        return {"cleared": cleared, "cleared_sessions": cleared_sessions}

    return app
