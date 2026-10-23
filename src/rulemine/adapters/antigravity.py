"""Adapter to discover and read Antigravity agent transcripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from rulemine.models import SessionRef, Turn


def discover_sessions(since: datetime | None = None) -> list[SessionRef]:
    """Discover Antigravity conversation sessions since a given datetime."""
    # Scaffolded for Day 1 AM Ingest milestone
    return []


def load_turns(session: SessionRef) -> list[Turn]:
    """Load and parse turns from session transcript.jsonl."""
    # Scaffolded for Day 1 AM Ingest milestone
    return []
