"""Periodic disk housekeeping for stateless `agy` CLI per-request conversation artifacts."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class CleanupReport:
    """Summary of removed stale `agy` artifacts across account profiles."""

    removed_entries: int = 0
    freed_bytes: int = 0


def _entry_size(path: Path) -> int:
    try:
        if path.is_symlink() or path.is_file():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for child in path.rglob("*"):
                if child.is_file() and not child.is_symlink():
                    try:
                        total += child.stat().st_size
                    except OSError:
                        pass
            return total
    except OSError:
        return 0
    return 0


def prune_agy_artifacts(
    home_dirs: Iterable[Path],
    retention_minutes: int = 30,
    workdir: Optional[Path] = None,
    protected_conversation_ids: Optional[Iterable[str]] = None,
) -> CleanupReport:
    """Prune old `agy` conversation DBs, brain folders, lockfiles, logs, and orphaned media dirs.

    Credentials, settings.json, builtin skills/plugins, active reusable session conversations
    listed in `protected_conversation_ids`, and any files modified within `retention_minutes`
    are strictly preserved.
    """
    cutoff = time.time() - max(1, retention_minutes) * 60
    removed = 0
    freed = 0
    seen_dirs: set[Path] = set()
    protected_ids = {str(cid).strip() for cid in (protected_conversation_ids or ()) if str(cid).strip()}

    # Clean orphaned per-request media staging folders if any survived a hard crash
    if workdir is not None:
        wd = Path(workdir)
        if wd.is_dir():
            try:
                media_entries = list(wd.glob("media_*"))
            except OSError:
                media_entries = []
            for entry in media_entries:
                if entry.is_symlink() or not entry.is_dir():
                    continue
                try:
                    if entry.stat().st_mtime >= cutoff:
                        continue
                except OSError:
                    continue
                size = _entry_size(entry)
                try:
                    shutil.rmtree(entry, ignore_errors=False)
                    removed += 1
                    freed += size
                except OSError:
                    continue

    target_subdirs = ("conversations", "brain", "presence", "implicit", "crashes", "log")

    for home in home_dirs:
        cli_root = Path(home) / ".gemini" / "antigravity-cli"
        try:
            resolved = cli_root.resolve()
        except OSError:
            continue
        if resolved in seen_dirs or not resolved.is_dir():
            continue
        seen_dirs.add(resolved)

        for sub in target_subdirs:
            sub_dir = resolved / sub
            if not sub_dir.is_dir():
                continue
            entries: List[Path] = []
            try:
                entries = list(sub_dir.iterdir())
            except OSError:
                continue

            for entry in entries:
                if entry.is_symlink():
                    continue
                if protected_ids and sub in ("conversations", "brain", "presence"):
                    # Preserve <cid>.db, <cid>.db-wal, <cid>.db-shm, <cid>.lock, brain/<cid>
                    base_id = entry.name.split(".", 1)[0]
                    if base_id in protected_ids or entry.stem in protected_ids or entry.name in protected_ids:
                        continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime >= cutoff:
                    continue
                size = _entry_size(entry)
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=False)
                    else:
                        entry.unlink(missing_ok=True)
                    removed += 1
                    freed += size
                except OSError:
                    continue

    return CleanupReport(removed_entries=removed, freed_bytes=freed)
