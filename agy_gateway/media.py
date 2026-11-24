"""Multimodal attachment extraction, SSRF-safe downloading, and isolated per-request staging."""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import re
import shutil
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class MediaError(ValueError):
    """Raised when an incoming image/video/audio/document attachment is invalid or blocked."""


_DATA_URL_RE = re.compile(
    r"^data:([a-zA-Z0-9.+_-]+/[a-zA-Z0-9.+_-]+)?(?:;[a-zA-Z0-9=._-]+)*;base64,(.+)$",
    re.DOTALL,
)

_MIME_TO_EXT: Dict[str, str] = {
    # Images
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/avif": ".avif",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
    # Videos
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/mpeg": ".mpeg",
    "video/ogg": ".ogv",
    "video/x-msvideo": ".avi",
    "video/3gpp": ".3gp",
    "video/x-flv": ".flv",
    "video/x-ms-wmv": ".wmv",
    # Audio
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    # Documents
    "application/pdf": ".pdf",
}

_EXT_TO_MIME: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".ogv": "video/ogg",
    ".avi": "video/x-msvideo",
    ".3gp": "video/3gpp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".pdf": "application/pdf",
}

_AUDIO_FORMAT_TO_MIME: Dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
}


@dataclass(frozen=True)
class MediaAttachment:
    """Normalized media attachment extracted from OpenAI, Anthropic, or Gemini requests."""

    index: int
    kind: str  # image | video | audio | document
    mime_type: str
    data: Optional[bytes] = None
    url: Optional[str] = None
    filename: Optional[str] = None


@dataclass(frozen=True)
class StagedMedia:
    """Materialized media file staged inside a per-request isolated temp directory."""

    index: int
    kind: str
    mime_type: str
    path: Path
    size_bytes: int


def media_placeholder(index: int) -> str:
    return f"__AGY_MEDIA_PATH_{index}__"


def normalize_mime_type(raw_mime: Optional[str], fallback_kind: str = "image") -> str:
    cleaned = (raw_mime or "").split(";", 1)[0].strip().lower()
    if cleaned == "image/jpg":
        return "image/jpeg"
    if cleaned in ("audio/wave", "audio/x-wav"):
        return "audio/wav"
    if cleaned == "audio/mp3":
        return "audio/mpeg"
    if cleaned in _MIME_TO_EXT:
        return cleaned
    if cleaned.startswith(("image/", "video/", "audio/")):
        return cleaned
    if cleaned == "application/pdf":
        return cleaned
    if fallback_kind == "video":
        return "video/mp4"
    if fallback_kind == "audio":
        return "audio/wav"
    if fallback_kind == "document":
        return "application/pdf"
    return "image/png"


def kind_from_mime(mime_type: str, fallback: str = "image") -> str:
    lower = (mime_type or "").lower()
    if lower.startswith("image/"):
        return "image"
    if lower.startswith("video/"):
        return "video"
    if lower.startswith("audio/"):
        return "audio"
    if lower == "application/pdf":
        return "document"
    return fallback


def sniff_mime_type(data: bytes) -> Optional[str]:
    """Best-effort magic-byte MIME detection for common image, video, audio, and PDF files."""
    if len(data) < 12:
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "video/x-msvideo"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand == b"qt  ":
            return "video/quicktime"
        if brand in (b"heic", b"heix", b"mif1"):
            return "image/heic"
        if brand == b"avif":
            return "image/avif"
        if brand == b"M4A ":
            return "audio/mp4"
        return "video/mp4"
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if data.startswith(b"ID3") or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    return None


def _guess_mime_from_name_or_url(name_or_url: Optional[str]) -> Optional[str]:
    if not name_or_url:
        return None
    parsed = urllib.parse.urlparse(name_or_url)
    path = parsed.path or name_or_url
    suffix = Path(path).suffix.lower()
    return _EXT_TO_MIME.get(suffix)


def _decode_base64_payload(b64_text: str) -> bytes:
    cleaned = "".join(b64_text.split())
    if not cleaned:
        raise MediaError("Empty base64 media payload")
    try:
        return base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError):
        # Fallback for unpadded base64 strings sent by some SDKs
        pad = (-len(cleaned)) % 4
        try:
            return base64.b64decode(cleaned + ("=" * pad))
        except Exception as exc:
            raise MediaError("Invalid base64 media payload") from exc


def parse_data_or_remote_url(
    raw_url: str,
    fallback_kind: str = "image",
    explicit_mime: Optional[str] = None,
) -> Tuple[str, str, Optional[bytes], Optional[str]]:
    """Parse either a `data:<mime>;base64,...` URI or an `http(s)://` URL."""
    url = raw_url.strip()
    match = _DATA_URL_RE.match(url)
    if match:
        data_mime = match.group(1) or explicit_mime
        raw_bytes = _decode_base64_payload(match.group(2))
        sniffed = sniff_mime_type(raw_bytes)
        mime = normalize_mime_type(data_mime or sniffed, fallback_kind=fallback_kind)
        return kind_from_mime(mime, fallback_kind), mime, raw_bytes, None

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
        guessed = explicit_mime or _guess_mime_from_name_or_url(url)
        mime = normalize_mime_type(guessed, fallback_kind=fallback_kind)
        return kind_from_mime(mime, fallback_kind), mime, None, url

    raise MediaError("Unsupported media URL (expected data:<mime>;base64,... or http(s):// URL)")


def _append_attachment(
    attachments: List[MediaAttachment],
    *,
    kind: str,
    mime_type: str,
    data: Optional[bytes] = None,
    url: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    idx = len(attachments) + 1
    attachments.append(
        MediaAttachment(
            index=idx,
            kind=kind,
            mime_type=mime_type,
            data=data,
            url=url,
            filename=filename,
        )
    )
    return f"[Attached {kind} #{idx} ({mime_type}): {media_placeholder(idx)}]"


def _extract_source_block(
    source: Dict[str, Any],
    fallback_kind: str,
    attachments: List[MediaAttachment],
) -> Optional[str]:
    """Parse Anthropic-style `{source: {type: base64|url|text, ...}}` blocks."""
    src_type = str(source.get("type") or "").lower()
    if src_type == "text":
        text = source.get("data")
        return str(text) if isinstance(text, str) else None
    if src_type == "base64":
        raw_b64 = source.get("data")
        if not isinstance(raw_b64, str):
            return None
        raw_bytes = _decode_base64_payload(raw_b64)
        raw_mime = source.get("media_type") or source.get("mime_type") or sniff_mime_type(raw_bytes)
        mime = normalize_mime_type(raw_mime, fallback_kind=fallback_kind)
        return _append_attachment(
            attachments,
            kind=kind_from_mime(mime, fallback_kind),
            mime_type=mime,
            data=raw_bytes,
        )
    if src_type == "url" and isinstance(source.get("url"), str):
        kind, mime, data, url = parse_data_or_remote_url(
            source["url"],
            fallback_kind=fallback_kind,
            explicit_mime=source.get("media_type") or source.get("mime_type"),
        )
        return _append_attachment(attachments, kind=kind, mime_type=mime, data=data, url=url)
    return None


def extract_media_block(
    item: Dict[str, Any],
    attachments: List[MediaAttachment],
) -> Optional[str]:
    """If `item` is a multimodal block from OpenAI, Anthropic, or Gemini, extract it and return placeholder text."""
    item_type = str(item.get("type") or "").lower()

    # OpenAI image_url
    if item_type == "image_url" or "image_url" in item:
        raw_img = item.get("image_url")
        raw_url = raw_img.get("url") if isinstance(raw_img, dict) else (raw_img if isinstance(raw_img, str) else None)
        if raw_url:
            kind, mime, data, url = parse_data_or_remote_url(raw_url, fallback_kind="image")
            return _append_attachment(attachments, kind=kind, mime_type=mime, data=data, url=url)

    # OpenAI / custom video_url / input_video
    if item_type in ("video_url", "input_video") or "video_url" in item:
        raw_vid = item.get("video_url") or item.get("input_video")
        raw_url = raw_vid.get("url") if isinstance(raw_vid, dict) else (raw_vid if isinstance(raw_vid, str) else None)
        if raw_url:
            kind, mime, data, url = parse_data_or_remote_url(raw_url, fallback_kind="video")
            return _append_attachment(attachments, kind=kind, mime_type=mime, data=data, url=url)

    # OpenAI input_audio
    if item_type == "input_audio" and isinstance(item.get("input_audio"), dict):
        aud = item["input_audio"]
        raw_b64 = aud.get("data")
        if isinstance(raw_b64, str) and raw_b64.strip():
            raw_bytes = _decode_base64_payload(raw_b64)
            fmt = str(aud.get("format") or "wav").lower()
            mime = _AUDIO_FORMAT_TO_MIME.get(fmt) or sniff_mime_type(raw_bytes) or "audio/wav"
            return _append_attachment(attachments, kind="audio", mime_type=mime, data=raw_bytes)

    # OpenAI file block ({type: "file", file: {file_data, filename}})
    if item_type == "file" and isinstance(item.get("file"), dict):
        fobj = item["file"]
        raw_data = fobj.get("file_data")
        fname = fobj.get("filename") if isinstance(fobj.get("filename"), str) else None
        if isinstance(raw_data, str) and raw_data.strip():
            if raw_data.strip().startswith(("data:", "http://", "https://")):
                kind, mime, data, url = parse_data_or_remote_url(
                    raw_data,
                    fallback_kind="document",
                    explicit_mime=_guess_mime_from_name_or_url(fname),
                )
                return _append_attachment(
                    attachments, kind=kind, mime_type=mime, data=data, url=url, filename=fname
                )
            raw_bytes = _decode_base64_payload(raw_data)
            mime = normalize_mime_type(
                _guess_mime_from_name_or_url(fname) or sniff_mime_type(raw_bytes),
                fallback_kind="document",
            )
            return _append_attachment(
                attachments,
                kind=kind_from_mime(mime, "document"),
                mime_type=mime,
                data=raw_bytes,
                filename=fname,
            )

    # Anthropic image / video / document blocks
    if item_type in ("image", "video", "document") and isinstance(item.get("source"), dict):
        return _extract_source_block(item["source"], fallback_kind=item_type, attachments=attachments)

    # Gemini inlineData / inline_data
    inline = item.get("inlineData") or item.get("inline_data")
    if isinstance(inline, dict) and isinstance(inline.get("data"), str):
        raw_bytes = _decode_base64_payload(inline["data"])
        raw_mime = inline.get("mimeType") or inline.get("mime_type") or sniff_mime_type(raw_bytes)
        mime = normalize_mime_type(raw_mime, fallback_kind="image")
        return _append_attachment(
            attachments,
            kind=kind_from_mime(mime, "image"),
            mime_type=mime,
            data=raw_bytes,
        )

    # Gemini fileData / file_data (supports http(s):// or data: URIs)
    file_data = item.get("fileData") or item.get("file_data")
    if isinstance(file_data, dict):
        uri = file_data.get("fileUri") or file_data.get("file_uri")
        if isinstance(uri, str) and uri.strip():
            raw_mime = file_data.get("mimeType") or file_data.get("mime_type")
            fallback_k = kind_from_mime(raw_mime or "", "video")
            kind, mime, data, url = parse_data_or_remote_url(
                uri, fallback_kind=fallback_k, explicit_mime=raw_mime
            )
            return _append_attachment(attachments, kind=kind, mime_type=mime, data=data, url=url)

    return None


def _validate_remote_url(url: str) -> None:
    """Reject non-HTTP(S) schemes and private/loopback/link-local hosts to prevent SSRF."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise MediaError(f"Disallowed media URL scheme: {parsed.scheme}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise MediaError("Media URL is missing a hostname")
    if host in ("localhost", "mihomo", "metadata.google.internal") or host.endswith(
        (".local", ".internal", ".localhost")
    ):
        raise MediaError(f"Blocked internal media host: {host}")

    # Direct IP literal check first
    try:
        ip_obj = ipaddress.ip_address(host)
        if not ip_obj.is_global:
            raise MediaError(f"Blocked non-public media IP: {host}")
        return
    except ValueError:
        pass

    # DNS resolution check (if resolvable locally, reject any non-public target IP)
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError:
        # When running behind an outbound proxy, DNS may only resolve through the proxy;
        # hostname blocklist above still protects internal service names.
        return

    for info in infos:
        addr = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not ip_obj.is_global:
            raise MediaError(f"Blocked media host resolving to non-public IP: {host}")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        _validate_remote_url(resolved)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _download_remote_media_sync(
    url: str,
    max_bytes: int,
    timeout_seconds: float,
) -> Tuple[bytes, Optional[str]]:
    _validate_remote_url(url)
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "agy-api-gateway/0.2"},
        method="GET",
    )
    try:
        with opener.open(req, timeout=timeout_seconds) as resp:
            content_type = resp.headers.get("Content-Type")
            content_len = resp.headers.get("Content-Length")
            if content_len and content_len.isdigit() and int(content_len) > max_bytes:
                raise MediaError(
                    f"Remote media exceeds size limit ({int(content_len)} > {max_bytes} bytes)"
                )
            buf = bytearray()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise MediaError(f"Remote media exceeds size limit ({max_bytes} bytes)")
            return bytes(buf), content_type
    except MediaError:
        raise
    except urllib.error.HTTPError as exc:
        raise MediaError(f"Failed to download media URL (HTTP {exc.code})") from exc
    except Exception as exc:
        raise MediaError(f"Failed to download media URL: {exc}") from exc


async def stage_media_attachments(
    attachments: Sequence[MediaAttachment],
    workdir: Path,
    *,
    max_files: int = 16,
    max_file_bytes: int = 25 * 1024 * 1024,
    max_total_bytes: int = 50 * 1024 * 1024,
    download_timeout_seconds: float = 30.0,
    allow_remote_urls: bool = True,
) -> Tuple[Path, List[StagedMedia]]:
    """Materialize attachments into an isolated per-request temp directory under workdir."""
    if len(attachments) > max_files:
        raise MediaError(f"Too many media attachments ({len(attachments)} > max {max_files})")

    workdir.mkdir(parents=True, exist_ok=True)
    staging_dir = (workdir / f"media_{uuid.uuid4().hex[:16]}").resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        staging_dir.chmod(0o700)
    except OSError:
        pass

    staged: List[StagedMedia] = []
    total_bytes = 0

    try:
        for att in attachments:
            raw_bytes: bytes
            effective_mime = att.mime_type

            if att.data is not None:
                raw_bytes = att.data
                if len(raw_bytes) > max_file_bytes:
                    raise MediaError(
                        f"Attachment #{att.index} exceeds file size limit ({len(raw_bytes)} > {max_file_bytes} bytes)"
                    )
            elif att.url:
                if not allow_remote_urls:
                    raise MediaError("Remote media URLs are disabled by gateway configuration")
                raw_bytes, resp_ct = await asyncio.to_thread(
                    _download_remote_media_sync,
                    att.url,
                    max_file_bytes,
                    download_timeout_seconds,
                )
                sniffed = sniff_mime_type(raw_bytes)
                effective_mime = normalize_mime_type(
                    sniffed or resp_ct or att.mime_type,
                    fallback_kind=att.kind,
                )
            else:
                raise MediaError(f"Attachment #{att.index} has neither data nor url")

            if not raw_bytes:
                raise MediaError(f"Attachment #{att.index} is empty")

            sniffed_final = sniff_mime_type(raw_bytes)
            if sniffed_final:
                effective_mime = normalize_mime_type(sniffed_final, fallback_kind=att.kind)
            else:
                effective_mime = normalize_mime_type(effective_mime, fallback_kind=att.kind)

            ext = _MIME_TO_EXT.get(effective_mime)
            if not ext:
                raise MediaError(f"Unsupported media MIME type: {effective_mime}")

            total_bytes += len(raw_bytes)
            if total_bytes > max_total_bytes:
                raise MediaError(
                    f"Total media payload exceeds limit ({total_bytes} > {max_total_bytes} bytes)"
                )

            out_path = staging_dir / f"attachment_{att.index:02d}{ext}"
            out_path.write_bytes(raw_bytes)
            staged.append(
                StagedMedia(
                    index=att.index,
                    kind=kind_from_mime(effective_mime, att.kind),
                    mime_type=effective_mime,
                    path=out_path,
                    size_bytes=len(raw_bytes),
                )
            )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return staging_dir, staged


def cleanup_staging_dir(staging_dir: Optional[Path]) -> None:
    if staging_dir is not None:
        shutil.rmtree(staging_dir, ignore_errors=True)
