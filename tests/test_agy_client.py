"""Unit tests for agy CLI output parsing and subprocess stream-json runner."""

from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

import pytest

from agy_gateway.agy_client import (
    AgyExecutionError,
    AgyRunner,
    parse_models_output,
    parse_quota_payload,
)


def test_parse_models_output_strips_spinner_and_ansi() -> None:
    raw = (
        "\r\x1b[36m⠋\x1b[0m Fetching available models..."
        "\r\x1b[2Kgemini-4-neon-high\tGemini 4 Neon (High)\n"
        "gemini-4-hydrogen-high   Gemini 4 Hydrogen (High)\n"
        "gemini-3.8-flash-high    Gemini 3.8 Flash (High)\n"
        "gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n"
    )
    models = parse_models_output(raw)
    assert [(m.id, m.display_name) for m in models] == [
        ("gemini-4-neon-high", "Gemini 4 Neon (High)"),
        ("gemini-4-hydrogen-high", "Gemini 4 Hydrogen (High)"),
        ("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
    ]


def _write_fake_agy(script_path: Path, body: str) -> Path:
    script_path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    return script_path


def test_runner_streams_stdin_ndjson_and_handles_error(tmp_path) -> None:
    fake_agy = _write_fake_agy(
        tmp_path / "fake_agy.py",
        """
import json, sys
line = sys.stdin.readline().strip()
msg = json.loads(line)
prompt = msg["message"]["content"]
if "FAIL" in prompt:
    print(json.dumps({"event": "result", "result": {"status": "ERROR", "error": "boom"}}))
    sys.exit(0)
print(json.dumps({"event": "init", "conversation_id": "conv-1"}))
print(json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Hello "}}))
print(json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "world", "usage": {"input_tokens": 11, "output_tokens": 2, "total_tokens": 13}}}))
print(json.dumps({"event": "result", "result": {"conversation_id": "conv-1", "status": "SUCCESS", "response": "Hello world", "usage": {"input_tokens": 11, "output_tokens": 2, "total_tokens": 13}}}))
""",
    )

    runner = AgyRunner(agy_bin=str(fake_agy), workdir=tmp_path / "work")
    text, usage, conv_id = asyncio.run(
        runner.complete_prompt(prompt="Hi", model="gemini-3.8-flash-high")
    )
    assert text == "Hello world"
    assert usage.input_tokens == 11
    assert usage.output_tokens == 2
    assert conv_id == "conv-1"

    with pytest.raises(AgyExecutionError, match="boom"):
        asyncio.run(runner.complete_prompt(prompt="FAIL", model="gemini-3.8-flash-high"))


def test_runner_caches_models_per_account_and_drains_large_stderr(tmp_path) -> None:
    counter_file = tmp_path / "calls.txt"
    fake_agy = _write_fake_agy(
        tmp_path / "fake_agy_cache.py",
        f"""
import json, sys
if len(sys.argv) > 1 and sys.argv[1] == "models":
    with open({str(counter_file)!r}, "a", encoding="utf-8") as f:
        f.write("1\\n")
    print("gemini-4-neon-high\\tGemini 4 Neon (High)")
    sys.exit(0)
sys.stdin.readline()
# Emit > 128 KiB to stderr (exceeds Linux 64 KiB pipe buffer) to verify no deadlock
sys.stderr.write(("W" * 1024 + "\\n") * 140)
sys.stderr.flush()
print(json.dumps({{"event": "result", "result": {{"conversation_id": "c-big", "status": "SUCCESS", "response": "ok-stderr"}}}}))
""",
    )
    runner = AgyRunner(agy_bin=str(fake_agy), workdir=tmp_path / "work", model_cache_ttl_seconds=60.0)
    custom_env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "acc_x")}

    async def _run():
        m1 = await runner.list_models(env=custom_env, cache_key="acc_x")
        m2 = await runner.list_models(env=custom_env, cache_key="acc_x")
        assert len(m1) == 1 and len(m2) == 1
        # Only invoked once thanks to per-account cache
        assert len(counter_file.read_text(encoding="utf-8").splitlines()) == 1
        text, _, _ = await runner.complete_prompt(
            prompt="hi", model="gemini-4-neon-high", timeout_seconds=10.0
        )
        assert text == "ok-stderr"

    asyncio.run(_run())


def test_parse_quota_payload_and_runner_fetch_account_quota(tmp_path) -> None:
    quota_json = """{
      "status": "SUCCESS",
      "command": {
        "name": "usage",
        "data": {
          "groups": [
            {
              "name": "Gemini Models",
              "description": "Gemini family",
              "buckets": [
                {"id": "gemini-weekly", "name": "Weekly Limit Remaining", "window": "weekly", "remaining_fraction": 0.5849, "reset_time": "2026-11-25T07:25:09Z"},
                {"id": "gemini-5h", "name": "Five Hour Limit Remaining", "window": "5h", "remaining_fraction": 0.8936, "reset_time": "2026-11-19T14:30:37Z"}
              ]
            },
            {
              "name": "Claude and GPT models",
              "buckets": [
                {"id": "3p-weekly", "name": "Weekly Limit Remaining", "window": "weekly", "remaining_fraction": 1.0, "reset_time": "2026-11-26T10:31:39Z"}
              ]
            }
          ]
        }
      }
    }"""
    credits_json = '{"status": "SUCCESS", "command": {"name": "credits", "data": {"remaining_credits": 250}}}'
    ok, data, err = parse_quota_payload(quota_json, credits_json)
    assert ok is True
    assert err == ""
    assert data["remaining_credits"] == 250
    assert data["min_remaining_percent"] == 58.5
    assert data["min_5h_percent"] == 89.4
    assert data["min_weekly_percent"] == 58.5
    assert len(data["groups"]) == 2

    fake_agy = _write_fake_agy(
        tmp_path / "fake_agy_quota.py",
        f"""
import sys
if "/quota" in sys.argv:
    print({quota_json!r})
elif "/credits" in sys.argv:
    print({credits_json!r})
""",
    )
    runner = AgyRunner(agy_bin=str(fake_agy), workdir=tmp_path / "work")
    q_ok, q_data, q_err = asyncio.run(runner.fetch_account_quota(env={"PATH": "/usr/bin:/bin"}))
    assert q_ok is True
    assert q_err == ""
    assert q_data["remaining_credits"] == 250
    assert q_data["min_remaining_percent"] == 58.5


