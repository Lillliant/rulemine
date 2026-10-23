"""Optional git evidence linking for decision records."""

from __future__ import annotations

from pathlib import Path


def find_nearby_commit(workspace_path: Path | str, timestamp_iso: str | None = None) -> str | None:
    """Best-effort search for a nearby git commit SHA."""
    # Best-effort evidence linking
    return None
