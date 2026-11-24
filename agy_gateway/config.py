"""Configuration settings for agy-api-gateway."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_MODEL_ALIASES: Dict[str, str] = {
    "default": "gemini-3.8-flash-high",
    "gemini-flash": "gemini-3.8-flash-high",
    "gemini-pro": "gemini-4-hydrogen-high",
    "gemini-2.5-flash": "gemini-3.8-flash-high",
    "gemini-2.5-pro": "gemini-4-hydrogen-high",
    "gemini-3-flash": "gemini-3.8-flash-high",
    "gemini-3-pro": "gemini-4-hydrogen-high",
    "gemini-3.8-flash": "gemini-3.8-flash-high",
    "gemini-4-neon": "gemini-4-neon-high",
    "gemini-4-hydrogen": "gemini-4-hydrogen-high",
    "gemini-4-carbon": "gemini-4-carbon-high",
    "gemini-4-argon": "gemini-4-argon-high",
    "claude-opus-5-5": "claude-opus-5-5-high",
    "claude-sonnet-5-5": "claude-sonnet-5-5-high",
}


class Settings(BaseSettings):
    """Gateway runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="AGY_GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8787
    agy_bin: str = Field(
        default_factory=lambda: shutil.which("agy")
        or os.path.expanduser("~/.local/bin/agy")
    )
    workdir: Path = Path("./data/agy-workdir")
    db_path: Path = Path("./data/gateway.db")
    database_url: str = ""
    accounts_dir: Path = Path("./data/accounts")
    admin_key: str = ""
    max_concurrency: int = 12
    queue_timeout_seconds: float = 300.0
    request_timeout_seconds: float = 600.0
    sse_keepalive_seconds: float = 10.0
    max_prompt_bytes: int = 191_500
    compaction_mode: str = "deterministic"  # deterministic | hybrid | llm
    compaction_model: str = ""
    compaction_keep_recent_bytes: int = 96_000
    compaction_checkpoint_max_bytes: int = 24_000
    compaction_tool_output_max_bytes: int = 12_000
    compaction_cache_ttl_hours: int = 24
    overflow_max_retries: int = 2
    session_reuse_enabled: bool = True
    session_ttl_hours: int = 24
    max_media_files: int = 16
    max_media_file_bytes: int = 25 * 1024 * 1024
    max_media_total_bytes: int = 50 * 1024 * 1024
    media_download_timeout_seconds: float = 30.0
    allow_remote_media_urls: bool = True
    routing_strategy: str = "least_busy"
    account_cooldown_seconds: float = 90.0
    artifact_retention_minutes: int = 30
    housekeeping_interval_seconds: float = 300.0
    quota_poll_interval_seconds: float = 300.0
    default_model: str = "gemini-3.8-flash-high"
    default_effort: str = ""
    sandbox: bool = True
    disable_slash_commands: bool = True
    model_cache_ttl_seconds: float = 300.0
    model_aliases: Dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_MODEL_ALIASES)
    )

    def resolve_model(self, requested: str | None) -> str:
        """Resolve an external model alias to a target Antigravity model ID."""
        if not requested:
            return self.default_model
        cleaned = requested.strip()
        if cleaned.startswith("models/"):
            cleaned = cleaned[len("models/") :]
        return self.model_aliases.get(cleaned, cleaned)

    def ensure_dirs(self) -> None:
        """Ensure runtime directories for SQLite storage, accounts, and isolated agy workdir exist."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.accounts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached gateway settings instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
