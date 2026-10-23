"""Turn windowing into chunks suitable for LLM extraction context."""

from __future__ import annotations

from rulemine.models import Turn, Window
from rulemine.redact import redact_secrets


def window_turns(
    turns: list[Turn],
    session_id: str,
    workspace: str | None = None,
    max_turns: int = 12,
    max_chars: int = 3500,
) -> list[Window]:
    """Chunk conversation turns into structured context windows."""
    # Scaffolded for Day 1 AM Ingest milestone
    return []
