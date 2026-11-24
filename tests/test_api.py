"""FastAPI integration tests using a fake AgyRunner backend."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, List, Optional

from fastapi.testclient import TestClient

from agy_gateway.agy_client import (
    AgyChunk,
    AgyExecutionError,
    AgyModelInfo,
    UsageInfo,
)
from agy_gateway.app import create_app
from agy_gateway.config import Settings
from agy_gateway.dispatcher import AgyDispatcher, QueueTimeoutError
from agy_gateway.keys import KeyStore


class FakeRunner:
    """In-memory fake matching AgyRunner interface for deterministic HTTP tests."""

    def __init__(
        self,
        reply_chunks: Optional[List[str]] = None,
        error: Optional[str] = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.reply_chunks = reply_chunks or ["Hello ", "from ", "Gemini"]
        self.error = error
        self.delay_seconds = delay_seconds
        self.last_model: Optional[str] = None
        self.last_prompt: Optional[str] = None

    async def list_models(
        self, force_refresh: bool = False, env=None
    ) -> List[AgyModelInfo]:
        return [
            AgyModelInfo("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
            AgyModelInfo("gemini-4-hydrogen-high", "Gemini 4 Hydrogen (High)"),
        ]

    async def probe_account(self, env):
        return True, await self.list_models(), ""

    async def stream_prompt(
        self,
        prompt: str,
        model: str,
        effort: Optional[str] = None,
        timeout_seconds: float = 180.0,
        env=None,
    ) -> AsyncIterator[AgyChunk]:
        self.last_prompt = prompt
        self.last_model = model
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if self.error:
            raise AgyExecutionError(self.error)
        for text in self.reply_chunks:
            yield AgyChunk(text_delta=text, done=False)
        yield AgyChunk(
            done=True,
            usage=UsageInfo(input_tokens=50, output_tokens=12, total_tokens=62),
            conversation_id="fake-conv-1",
        )


def _make_client(tmp_path, admin_key: str = "admin-secret", runner: Optional[FakeRunner] = None):
    settings = Settings(
        workdir=tmp_path / "workdir",
        db_path=tmp_path / "gateway.db",
        accounts_dir=tmp_path / "accounts",
        admin_key=admin_key,
        max_concurrency=2,
    )
    store = KeyStore(settings.db_path)
    fake_runner = runner or FakeRunner()
    dispatcher = AgyDispatcher(settings=settings, runner=fake_runner)  # type: ignore[arg-type]
    app = create_app(settings=settings, key_store=store, dispatcher=dispatcher)
    return TestClient(app), store, fake_runner, dispatcher


def test_openai_models_and_chat_completions_non_stream_and_stream(tmp_path) -> None:
    client, store, fake_runner, _ = _make_client(tmp_path)
    _, raw_key = store.create_key(name="test-client", rpm_limit=20)
    headers = {"Authorization": f"Bearer {raw_key}"}

    # Unauthenticated request fails
    assert client.get("/v1/models").status_code == 401
    assert client.get("/health").status_code == 200

    # List models includes real models + aliases
    models_resp = client.get("/v1/models", headers=headers)
    assert models_resp.status_code == 200
    ids = {m["id"] for m in models_resp.json()["data"]}
    assert "gemini-3.8-flash-high" in ids
    assert "gemini-2.5-flash" in ids

    # Non-stream chat completion resolves alias
    resp = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Say hi"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello from Gemini"
    assert data["usage"]["total_tokens"] == 62
    assert fake_runner.last_model == "gemini-3.8-flash-high"

    # Streaming SSE chat completion with usage chunk
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "gemini-4-hydrogen-high",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Stream test"}],
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200
        lines = [ln for ln in stream_resp.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(ln[6:]) for ln in lines[:-1]]
    streamed_text = "".join(
        p["choices"][0]["delta"].get("content", "")
        for p in payloads
        if p.get("choices")
    )
    assert streamed_text == "Hello from Gemini"
    assert payloads[-1]["usage"]["prompt_tokens"] == 50

    # Accounting recorded for both requests
    rec = store.list_keys()[0]
    assert rec.total_requests == 2
    assert rec.total_input_tokens == 100
    assert rec.total_output_tokens == 24


def test_gemini_and_anthropic_endpoints(tmp_path) -> None:
    client, store, fake_runner, _ = _make_client(tmp_path)
    _, raw_key = store.create_key(name="multi-harness", rpm_limit=20)

    # Gemini via x-goog-api-key header
    g_models = client.get("/v1beta/models", headers={"x-goog-api-key": raw_key})
    assert g_models.status_code == 200
    assert any(m["name"] == "models/gemini-2.5-pro" for m in g_models.json()["models"])

    # Gemini generateContent via ?key= query param
    g_resp = client.post(
        f"/v1beta/models/gemini-2.5-pro:generateContent?key={raw_key}",
        json={"contents": [{"role": "user", "parts": [{"text": "Ping"}]}]},
    )
    assert g_resp.status_code == 200
    assert g_resp.json()["candidates"][0]["content"]["parts"][0]["text"] == "Hello from Gemini"
    assert fake_runner.last_model == "gemini-4-hydrogen-high"

    # Gemini SSE stream
    with client.stream(
        "POST",
        f"/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key={raw_key}",
        json={"contents": [{"role": "user", "parts": [{"text": "Stream"}]}]},
    ) as g_stream:
        assert g_stream.status_code == 200
        g_lines = [ln for ln in g_stream.iter_lines() if ln.startswith("data: ")]
    assert len(g_lines) == 4

    # Anthropic /v1/messages via x-api-key
    a_resp = client.post(
        "/v1/messages",
        headers={"x-api-key": raw_key},
        json={
            "model": "gemini-3.8-flash-high",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert a_resp.status_code == 200
    assert a_resp.json()["content"][0]["text"] == "Hello from Gemini"


def test_admin_endpoints_and_error_mapping(tmp_path) -> None:
    client_no_admin, _, _, _ = _make_client(tmp_path / "no_admin", admin_key="")
    assert client_no_admin.get("/admin/keys").status_code == 403

    client, store, _, dispatcher = _make_client(
        tmp_path / "err",
        admin_key="top-secret",
        runner=FakeRunner(error="agy crashed"),
    )
    assert client.get("/admin/keys").status_code == 401

    created = client.post(
        "/admin/keys",
        headers={"Authorization": "Bearer top-secret"},
        json={"name": "harness-b", "rpm_limit": 5},
    )
    assert created.status_code == 200
    api_key = created.json()["api_key"]
    key_id = created.json()["key"]["id"]

    # Upstream agy failure maps to 502
    bad_resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert bad_resp.status_code == 502

    # Queue timeout maps to 503
    async def fail_acquire() -> None:
        raise QueueTimeoutError("busy")

    dispatcher._acquire_slot = fail_acquire  # type: ignore[method-assign]
    busy_resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert busy_resp.status_code == 503

    # UI shell & root redirect
    root_resp = client.get("/", follow_redirects=False)
    assert root_resp.status_code == 302
    assert root_resp.headers["location"] == "/ui"
    ui_resp = client.get("/ui")
    assert ui_resp.status_code == 200
    assert "Antigravity API Gateway" in ui_resp.text

    # Admin accounts, routing strategy, and housekeeping endpoints
    accs = client.get("/admin/accounts", headers={"Authorization": "Bearer top-secret"})
    assert accs.status_code == 200
    assert accs.json()["routing_strategy"] == "least_busy"
    assert any(a["id"] == "default" for a in accs.json()["accounts"])

    new_acc = client.post(
        "/admin/accounts",
        headers={"Authorization": "Bearer top-secret"},
        json={"name": "Backup Account", "priority": 50, "max_concurrency": 4},
    )
    assert new_acc.status_code == 200
    new_acc_id = new_acc.json()["account"]["id"]

    checked = client.post(
        f"/admin/accounts/{new_acc_id}/check",
        headers={"Authorization": "Bearer top-secret"},
    )
    assert checked.status_code == 200
    assert checked.json()["account"]["status"] == "logged_in"

    routed = client.put(
        "/admin/routing",
        headers={"Authorization": "Bearer top-secret"},
        json={"routing_strategy": "round_robin"},
    )
    assert routed.status_code == 200
    assert routed.json()["routing_strategy"] == "round_robin"

    hk = client.post("/admin/housekeeping", headers={"Authorization": "Bearer top-secret"})
    assert hk.status_code == 200
    assert "removed_entries" in hk.json()

    # Built-in default account cannot be deleted, custom account can
    assert (
        client.delete(
            "/admin/accounts/default",
            headers={"Authorization": "Bearer top-secret"},
        ).status_code
        == 400
    )
    assert (
        client.delete(
            f"/admin/accounts/{new_acc_id}",
            headers={"Authorization": "Bearer top-secret"},
        ).status_code
        == 200
    )

    # Admin models list
    admin_models = client.get(
        "/admin/models",
        headers={"Authorization": "Bearer top-secret"},
    )
    assert admin_models.status_code == 200
    assert len(admin_models.json()["models"]) == 2

    # Edit name/rpm/models, disable & delete via admin API
    edited = client.patch(
        f"/admin/keys/{key_id}",
        headers={"Authorization": "Bearer top-secret"},
        json={
            "name": "harness-renamed",
            "rpm_limit": 60,
            "allowed_models": ["gemini-3.8-flash-high"],
            "enabled": False,
        },
    )
    assert edited.status_code == 200
    assert edited.json()["key"]["name"] == "harness-renamed"
    assert edited.json()["key"]["rpm_limit"] == 60
    assert edited.json()["key"]["enabled"] is False
    assert (
        client.delete(
            f"/admin/keys/{key_id}",
            headers={"Authorization": "Bearer top-secret"},
        ).status_code
        == 200
    )
    assert store.list_keys() == []


def test_sse_keepalive_emitted_while_waiting_for_slow_thinking(tmp_path) -> None:
    settings = Settings(
        workdir=tmp_path / "workdir",
        db_path=tmp_path / "gateway.db",
        sse_keepalive_seconds=0.05,
        max_concurrency=4,
    )
    store = KeyStore(settings.db_path)
    _, raw_key = store.create_key(name="slow-stream", rpm_limit=20)
    dispatcher = AgyDispatcher(settings=settings, runner=FakeRunner(delay_seconds=0.18))  # type: ignore[arg-type]
    app = create_app(settings=settings, key_store=store, dispatcher=dispatcher)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/messages",
        headers={"x-api-key": raw_key},
        json={
            "model": "gemini-4-hydrogen-xhigh",
            "stream": True,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Think carefully"}],
        },
    ) as resp:
        assert resp.status_code == 200
        raw_lines = list(resp.iter_lines())

    assert ": keepalive" in raw_lines
    assert any("Hello " in ln for ln in raw_lines)


def test_anthropic_sse_error_event_and_check_all_accounts(tmp_path) -> None:
    client, store, _, _ = _make_client(
        tmp_path,
        admin_key="adm-test",
        runner=FakeRunner(error="UNAVAILABLE (code 503): No capacity available"),
    )
    _, raw_key = store.create_key(name="claude-client", rpm_limit=20)

    # Anthropic SSE failure must emit valid Anthropic `event: error` instead of OpenAI `[DONE]`
    with client.stream(
        "POST",
        "/v1/messages",
        headers={"x-api-key": raw_key},
        json={
            "model": "gemini-4-hydrogen-xhigh",
            "stream": True,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as resp:
        assert resp.status_code == 200
        lines = list(resp.iter_lines())

    assert "event: error" in lines
    assert "data: [DONE]" not in lines
    assert any("No capacity available" in ln for ln in lines)

    # Batch check-all endpoint checks all enabled accounts concurrently
    check_all = client.post(
        "/admin/accounts/check-all",
        headers={"Authorization": "Bearer adm-test"},
    )
    assert check_all.status_code == 200
    body = check_all.json()
    assert body["checked"] == 1
    assert body["healthy"] == 1


def test_admin_quota_refresh_endpoints_and_stats_summary(tmp_path) -> None:
    class QuotaFakeRunner(FakeRunner):
        async def fetch_account_quota(self, env):
            return (
                True,
                {
                    "groups": [
                        {
                            "name": "Gemini Models",
                            "description": "Gemini",
                            "buckets": [
                                {
                                    "id": "gemini-5h",
                                    "name": "Five Hour Limit Remaining",
                                    "window": "5h",
                                    "remaining_fraction": 0.15,
                                    "remaining_percent": 15.0,
                                    "reset_time": "2026-11-19T15:00:00Z",
                                    "description": "",
                                }
                            ],
                        }
                    ],
                    "remaining_credits": 42,
                    "min_remaining_percent": 15.0,
                    "min_5h_percent": 15.0,
                    "min_weekly_percent": None,
                    "error": None,
                },
                "",
            )

    client, _, _, _ = _make_client(
        tmp_path,
        admin_key="adm-quota",
        runner=QuotaFakeRunner(),
    )
    ref_all = client.post(
        "/admin/accounts/quota/refresh",
        headers={"Authorization": "Bearer adm-quota"},
    )
    assert ref_all.status_code == 200
    assert ref_all.json()["refreshed"] == 1
    acc = ref_all.json()["accounts"][0]
    assert acc["quota"]["remaining_credits"] == 42
    assert acc["quota"]["min_remaining_percent"] == 15.0
    assert acc["quota_updated_at"] is not None

    ref_one = client.post(
        "/admin/accounts/default/quota",
        headers={"Authorization": "Bearer adm-quota"},
    )
    assert ref_one.status_code == 200
    assert ref_one.json()["quota"]["min_remaining_percent"] == 15.0

    stats = client.get("/admin/stats", headers={"Authorization": "Bearer adm-quota"}).json()
    assert stats["dispatcher"]["lowest_quota_percent"] == 15.0
    assert stats["dispatcher"]["accounts_low_quota"] == 1



