"""Unit tests for multi-account pool, failover routing, OAuth login flow, and housekeeping."""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from agy_gateway.accounts import AccountStore, OAuthLoginManager
from agy_gateway.agy_client import AgyChunk, AgyExecutionError, AgyModelInfo, UsageInfo
from agy_gateway.config import Settings
from agy_gateway.dispatcher import AgyDispatcher, ChatMessage, NormalizedRequest
from agy_gateway.housekeeping import prune_agy_artifacts


class MultiAccountFakeRunner:
    """Tracks which account HOME env handled each request and simulates per-account failures."""

    def __init__(self, fail_homes: Optional[Dict[str, str]] = None) -> None:
        self.fail_homes = fail_homes or {}
        self.calls: List[str] = []

    async def list_models(
        self, force_refresh: bool = False, env: Optional[Dict[str, str]] = None
    ) -> List[AgyModelInfo]:
        return [AgyModelInfo("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)")]

    async def probe_account(self, env: Dict[str, str]):
        home = env.get("HOME", "default")
        if home in self.fail_homes:
            return False, [], self.fail_homes[home]
        return True, [AgyModelInfo("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)")], ""

    async def stream_prompt(
        self,
        prompt: str,
        model: str,
        effort: Optional[str] = None,
        timeout_seconds: float = 180.0,
        env: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[AgyChunk]:
        home = (env or {}).get("HOME", "default")
        self.calls.append(home)
        if home in self.fail_homes:
            raise AgyExecutionError(self.fail_homes[home])
        yield AgyChunk(text_delta=f"ok:{Path(home).name}", done=False)
        yield AgyChunk(
            done=True,
            usage=UsageInfo(input_tokens=15, output_tokens=3, total_tokens=18),
            conversation_id="c-1",
        )


def test_account_store_crud_and_routing_failover(tmp_path) -> None:
    settings = Settings(
        workdir=tmp_path / "work",
        db_path=tmp_path / "gateway.db",
        accounts_dir=tmp_path / "accounts",
        max_concurrency=4,
    )
    store = AccountStore(settings.db_path, settings.accounts_dir)
    # Default account auto-created
    assert store.get_account("default") is not None

    acc_a = store.create_account(name="Account A", priority=10, max_concurrency=2)
    acc_b = store.create_account(name="Account B", priority=20, max_concurrency=2)
    # Mark both logged_in and disable default so test only routes between A & B
    store.update_account("default", enabled=False)
    store.update_account(acc_a.id, status="logged_in")
    store.update_account(acc_b.id, status="logged_in")

    # Round-robin alternates across A and B
    store.set_routing_strategy("round_robin")
    runner = MultiAccountFakeRunner()
    dispatcher = AgyDispatcher(settings=settings, runner=runner, account_store=store)  # type: ignore[arg-type]
    req = NormalizedRequest(
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="ping")],
    )

    t1, _, _ = asyncio.run(dispatcher.complete(req))
    t2, _, _ = asyncio.run(dispatcher.complete(req))
    assert {t1, t2} == {f"ok:{acc_a.id}", f"ok:{acc_b.id}"}

    # Failover when Account A hits 429 quota error -> automatically cools down A and succeeds on B
    store.set_routing_strategy("priority")
    fail_runner = MultiAccountFakeRunner(fail_homes={acc_a.home_dir: "HTTP 429 quota exceeded"})
    fail_dispatcher = AgyDispatcher(settings=settings, runner=fail_runner, account_store=store)  # type: ignore[arg-type]

    text, usage, _ = asyncio.run(fail_dispatcher.complete(req))
    assert text == f"ok:{acc_b.id}"
    assert usage.input_tokens == 15

    a_after = store.get_account(acc_a.id)
    assert a_after is not None
    assert a_after.status == "cooldown"
    assert a_after.cooldown_until > time.time()
    assert a_after.total_errors == 1

    # Key pinned to acc_b only routes to acc_b
    b_only_runner = MultiAccountFakeRunner()
    b_dispatcher = AgyDispatcher(settings=settings, runner=b_only_runner, account_store=store)  # type: ignore[arg-type]
    store.update_account(acc_a.id, status="logged_in", cooldown_until=0.0, clear_error=True)
    text_b, _, _ = asyncio.run(b_dispatcher.complete(req, allowed_accounts=[acc_b.id]))
    assert text_b == f"ok:{acc_b.id}"

    # Delete acc_b cleans up home dir
    assert Path(acc_b.home_dir).is_dir()
    assert store.delete_account(acc_b.id) is True
    assert not Path(acc_b.home_dir).exists()


def test_oauth_login_manager_roundtrip(tmp_path) -> None:
    fake_agy = tmp_path / "fake_agy.py"
    fake_agy.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "print('Open this link in a browser to log in with Google:', flush=True)\n"
        "print('  https://accounts.google.com/o/oauth2/auth?client_id=test&state=xyz', flush=True)\n"
        "code = sys.stdin.readline().strip()\n"
        "if code != 'GOOD_CODE':\n"
        "    print('invalid code', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "print('Login successful!')\n",
        encoding="utf-8",
    )
    fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IXUSR)

    store = AccountStore(tmp_path / "gateway.db", tmp_path / "accounts")
    acc = store.create_account(name="OAuth Test")
    mgr = OAuthLoginManager(agy_bin=str(fake_agy), workdir=tmp_path / "work")

    async def _run():
        sid, url, exp = await mgr.start_login(acc)
        assert url.startswith("https://accounts.google.com/o/oauth2/auth?")
        assert exp > time.time()
        acc_id, out = await mgr.submit_code(sid, "GOOD_CODE")
        assert acc_id == acc.id
        assert "Login successful" in out

    asyncio.run(_run())


def test_housekeeping_prunes_only_old_artifacts(tmp_path) -> None:
    home = tmp_path / "acc_home"
    cli = home / ".gemini" / "antigravity-cli"
    (cli / "conversations").mkdir(parents=True)
    (cli / "brain" / "old_conv").mkdir(parents=True)
    (cli / "log").mkdir(parents=True)

    old_db = cli / "conversations" / "old.db"
    old_db.write_bytes(b"x" * 2048)
    fresh_db = cli / "conversations" / "fresh.db"
    fresh_db.write_bytes(b"y" * 1024)
    old_brain_file = cli / "brain" / "old_conv" / "transcript.jsonl"
    old_brain_file.write_text("old", encoding="utf-8")
    settings_file = cli / "settings.json"
    settings_file.write_text("{}", encoding="utf-8")

    old_ts = time.time() - 7200
    os.utime(old_db, (old_ts, old_ts))
    os.utime(cli / "brain" / "old_conv", (old_ts, old_ts))
    os.utime(settings_file, (old_ts, old_ts))

    report = prune_agy_artifacts([home], retention_minutes=30)
    assert report.removed_entries == 2
    assert report.freed_bytes >= 2048
    assert not old_db.exists()
    assert not (cli / "brain" / "old_conv").exists()
    assert fresh_db.exists()
    assert settings_file.exists()


def test_per_account_concurrency_queues_instead_of_502_and_caches_models(tmp_path) -> None:
    settings = Settings(
        workdir=tmp_path / "work",
        db_path=tmp_path / "gateway.db",
        accounts_dir=tmp_path / "accounts",
        max_concurrency=4,
        queue_timeout_seconds=5.0,
    )
    store = AccountStore(settings.db_path, settings.accounts_dir)
    store.update_account("default", enabled=False)
    acc_a = store.create_account(name="SingleSlot", priority=10, max_concurrency=1)
    store.update_account(acc_a.id, status="logged_in")

    class SlowFakeRunner:
        def __init__(self) -> None:
            self.active = 0
            self.max_seen = 0
            self.model_calls = 0

        async def list_models(
            self,
            force_refresh: bool = False,
            env: Optional[Dict[str, str]] = None,
            cache_key: str = "default",
        ) -> List[AgyModelInfo]:
            self.model_calls += 1
            return [AgyModelInfo("gemini-4-neon-high", "Gemini 4 Neon (High)")]

        async def stream_prompt(
            self,
            prompt: str,
            model: str,
            effort: Optional[str] = None,
            timeout_seconds: float = 180.0,
            env: Optional[Dict[str, str]] = None,
        ) -> AsyncIterator[AgyChunk]:
            self.active += 1
            self.max_seen = max(self.max_seen, self.active)
            try:
                await asyncio.sleep(0.08)
                yield AgyChunk(text_delta="queued-ok", done=False)
                yield AgyChunk(
                    done=True,
                    usage=UsageInfo(input_tokens=5, output_tokens=2, total_tokens=7),
                    conversation_id="c-q",
                )
            finally:
                self.active -= 1

    runner = SlowFakeRunner()
    dispatcher = AgyDispatcher(settings=settings, runner=runner, account_store=store)  # type: ignore[arg-type]
    req = NormalizedRequest(
        model="gemini-4-neon-high",
        messages=[ChatMessage(role="user", content="hello")],
    )

    async def _run_parallel():
        return await asyncio.gather(
            dispatcher.complete(req, allowed_accounts=[acc_a.id]),
            dispatcher.complete(req, allowed_accounts=[acc_a.id]),
        )

    results = asyncio.run(_run_parallel())
    assert [r[0] for r in results] == ["queued-ok", "queued-ok"]
    assert runner.max_seen == 1

