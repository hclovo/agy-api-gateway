"""Tests for multimodal image/video/audio/document support across OpenAI, Anthropic, and Gemini."""

from __future__ import annotations

import base64
import os
import struct
import time
import zlib
from pathlib import Path
from typing import AsyncIterator, List, Optional

import pytest
from fastapi.testclient import TestClient

from agy_gateway.agy_client import AgyChunk, AgyModelInfo, UsageInfo
from agy_gateway.app import create_app
from agy_gateway.config import Settings
from agy_gateway.dispatcher import AgyDispatcher
from agy_gateway.housekeeping import prune_agy_artifacts
from agy_gateway.keys import KeyStore


def _make_tiny_png() -> bytes:
    w, h = 4, 4
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _make_fake_mp4() -> bytes:
    # Valid ISO BMFF ftyp box header so MIME sniffing detects video/mp4
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 32


class MediaInspectingRunner:
    """Fake runner verifying that staged files exist during execution and recording prompt/cwd."""

    def __init__(self) -> None:
        self.last_prompt: str = ""
        self.last_cwd: Optional[Path] = None
        self.observed_files: List[str] = []

    async def list_models(
        self, force_refresh: bool = False, env=None, cache_key: str = "default"
    ) -> List[AgyModelInfo]:
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
        cwd: Optional[Path] = None,
    ) -> AsyncIterator[AgyChunk]:
        self.last_prompt = prompt
        self.last_cwd = cwd
        self.observed_files = []
        if cwd is not None and cwd.is_dir():
            for child in sorted(cwd.iterdir()):
                if child.is_file():
                    self.observed_files.append(f"{child.name}:{child.stat().st_size}")
        yield AgyChunk(text_delta="inspected-ok", done=False)
        yield AgyChunk(
            done=True,
            usage=UsageInfo(input_tokens=40, output_tokens=5, total_tokens=45),
            conversation_id="media-conv-1",
        )


def _make_media_client(tmp_path: Path, max_file_bytes: int = 5 * 1024 * 1024):
    settings = Settings(
        workdir=tmp_path / "workdir",
        db_path=tmp_path / "gateway.db",
        accounts_dir=tmp_path / "accounts",
        max_concurrency=2,
        max_media_file_bytes=max_file_bytes,
    )
    store = KeyStore(settings.db_path)
    _, raw_key = store.create_key(name="media-client", rpm_limit=60)
    runner = MediaInspectingRunner()
    dispatcher = AgyDispatcher(settings=settings, runner=runner)  # type: ignore[arg-type]
    app = create_app(settings=settings, key_store=store, dispatcher=dispatcher)
    return TestClient(app), raw_key, runner, settings


def test_openai_image_and_video_staging_and_cleanup(tmp_path: Path) -> None:
    client, raw_key, runner, settings = _make_media_client(tmp_path)
    png_bytes = _make_tiny_png()
    mp4_bytes = _make_fake_mp4()
    png_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    mp4_uri = "data:video/mp4;base64," + base64.b64encode(mp4_bytes).decode()

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={
            "model": "gemini-3.8-flash-high",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare the image and video"},
                        {"type": "image_url", "image_url": {"url": png_uri}},
                        {"type": "video_url", "video_url": {"url": mp4_uri}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "inspected-ok"

    # Both files were materialized inside an isolated media_* cwd and listed for read_file
    assert runner.last_cwd is not None
    assert runner.last_cwd.name.startswith("media_")
    assert runner.observed_files == [
        f"attachment_01.png:{len(png_bytes)}",
        f"attachment_02.mp4:{len(mp4_bytes)}",
    ]
    assert "read_file" in runner.last_prompt
    assert str(runner.last_cwd / "attachment_01.png") in runner.last_prompt
    assert str(runner.last_cwd / "attachment_02.mp4") in runner.last_prompt
    assert "__AGY_MEDIA_PATH_" not in runner.last_prompt

    # Staging directory must be completely cleaned up after request ends
    assert not runner.last_cwd.exists()
    assert list(settings.workdir.glob("media_*")) == []


def test_anthropic_image_document_and_tool_result_image(tmp_path: Path) -> None:
    client, raw_key, runner, _ = _make_media_client(tmp_path)
    png_b64 = base64.b64encode(_make_tiny_png()).decode()
    pdf_b64 = base64.b64encode(b"%PDF-1.7\nhello pdf\n").decode()

    resp = client.post(
        "/v1/messages",
        headers={"x-api-key": raw_key},
        json={
            "model": "gemini-3.8-flash-high",
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": png_b64,
                            },
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": [
                                {"type": "text", "text": "Screenshot captured:"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": png_b64,
                                    },
                                },
                            ],
                        },
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert len(runner.observed_files) == 3
    assert runner.observed_files[0].startswith("attachment_01.png:")
    assert runner.observed_files[1].startswith("attachment_02.pdf:")
    assert runner.observed_files[2].startswith("attachment_03.png:")
    assert runner.last_cwd is not None and not runner.last_cwd.exists()


def test_gemini_inline_video_and_image_parts(tmp_path: Path) -> None:
    client, raw_key, runner, _ = _make_media_client(tmp_path)
    mp4_b64 = base64.b64encode(_make_fake_mp4()).decode()

    resp = client.post(
        f"/v1beta/models/gemini-3.8-flash-high:generateContent?key={raw_key}",
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "video/mp4", "data": mp4_b64}},
                        {"text": "Describe this video"},
                    ],
                }
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["candidates"][0]["content"]["parts"][0]["text"] == "inspected-ok"
    assert len(runner.observed_files) == 1
    assert runner.observed_files[0].startswith("attachment_01.mp4:")
    assert runner.last_cwd is not None and not runner.last_cwd.exists()


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:7890/secret.png",
        "http://localhost:9090/configs",
        "http://mihomo:9090/proxies",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
    ],
)
def test_ssrf_and_invalid_media_urls_rejected_with_400(tmp_path: Path, bad_url: str) -> None:
    client, raw_key, _, settings = _make_media_client(tmp_path)
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={
            "model": "gemini-3.8-flash-high",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": bad_url}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 400
    assert list(settings.workdir.glob("media_*")) == []


def test_oversized_media_rejected_and_housekeeping_prunes_orphaned_media(tmp_path: Path) -> None:
    client, raw_key, _, settings = _make_media_client(tmp_path, max_file_bytes=64)
    big_png = _make_tiny_png() + (b"\x00" * 100)
    uri = "data:image/png;base64," + base64.b64encode(big_png).decode()

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={
            "model": "gemini-3.8-flash-high",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]}
            ],
        },
    )
    assert resp.status_code == 400
    assert "size limit" in resp.text
    assert list(settings.workdir.glob("media_*")) == []

    # Simulate orphaned media dir from a hard crash
    orphan = settings.workdir / "media_orphaned1234"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "attachment_01.mp4").write_bytes(b"12345678")
    old_ts = time.time() - 7200
    os.utime(orphan, (old_ts, old_ts))

    report = prune_agy_artifacts([], retention_minutes=30, workdir=settings.workdir)
    assert report.removed_entries == 1
    assert not orphan.exists()
