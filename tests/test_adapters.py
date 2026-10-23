"""Unit tests for Antigravity transcript adapter."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import pytest

from rulemine.adapters.antigravity import (
    discover_sessions,
    load_turns,
    parse_since,
    _parse_workspace_uris,
)
from rulemine.models import SessionRef, Turn


def test_parse_since():
    now = datetime.now(timezone.utc)

    d7 = parse_since("7d")
    assert d7 is not None
    assert abs((now - d7).total_seconds() - 7 * 86400) < 5

    h24 = parse_since("24h")
    assert h24 is not None
    assert abs((now - h24).total_seconds() - 24 * 3600) < 5

    m30 = parse_since("30m")
    assert m30 is not None
    assert abs((now - m30).total_seconds() - 30 * 60) < 5

    dt_str = parse_since("2026-10-22")
    assert dt_str is not None
    assert dt_str.year == 2026
    assert dt_str.month == 10
    assert dt_str.day == 22
    assert dt_str.tzinfo is not None

    with pytest.raises(ValueError):
        parse_since("invalid-date-format")


def test_parse_workspace_uris():
    # Single URI with percent encoding
    raw = '["file:///workspace/projects/RuleMine%20Project"]'
    assert _parse_workspace_uris(raw) == "/workspace/projects/RuleMine Project"

    # Multiple URIs (returns first)
    raw_multi = '["file:///path/to/first%20dir", "file:///path/to/second"]'
    assert _parse_workspace_uris(raw_multi) == "/path/to/first dir"

    # None or empty
    assert _parse_workspace_uris(None) is None
    assert _parse_workspace_uris("") is None
    assert _parse_workspace_uris("[]") is None


def test_discover_sessions_and_load_turns(tmp_path: Path):
    # Setup mock Antigravity directory structure
    root = tmp_path / "antigravity"
    brain_dir = root / "brain"
    db_path = root / "conversation_summaries.db"
    root.mkdir(parents=True)

    # Create mock SQLite database
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute(
            """
            CREATE TABLE conversation_summaries (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                workspace_uris TEXT,
                last_modified_time TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO conversation_summaries (conversation_id, title, workspace_uris, last_modified_time)
            VALUES (?, ?, ?, ?)
            """,
            (
                "session-1",
                "Fix Build Script",
                '["file:///workspace/dev/my%20project"]',
                "2026-10-22T10:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO conversation_summaries (conversation_id, title, workspace_uris, last_modified_time)
            VALUES (?, ?, ?, ?)
            """,
            (
                "session-2",
                "Old Session",
                '["file:///workspace/dev/other"]',
                "2026-01-01T00:00:00Z",
            ),
        )
    conn.close()

    # Create transcript files
    s1_log_dir = brain_dir / "session-1" / ".system_generated" / "logs"
    s1_log_dir.mkdir(parents=True)
    t1_file = s1_log_dir / "transcript.jsonl"
    t1_lines = [
        json.dumps(
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-10-22T10:01:00Z",
                "content": "Please switch to npm.",
            }
        ),
        json.dumps(
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-10-22T10:01:05Z",
                "content": "Running npm install...",
            }
        ),
        # Corrupted line to test error tolerance
        "{ bad json",
    ]
    t1_file.write_text("\n".join(t1_lines), encoding="utf-8")

    # Discover sessions
    sessions = discover_sessions(base_dir=root)
    assert len(sessions) == 1
    assert sessions[0].session_id == "session-1"
    assert sessions[0].title == "Fix Build Script"
    assert sessions[0].workspace == "/workspace/dev/my project"

    # Test load_turns
    turns = load_turns(sessions[0])
    assert len(turns) == 2
    assert turns[0].step_index == 0
    assert turns[0].source == "USER_EXPLICIT"
    assert turns[0].content == "Please switch to npm."
    assert turns[1].step_index == 1
    assert turns[1].source == "MODEL"


def test_discover_sessions_workspace_filter(tmp_path: Path):
    root = tmp_path / "antigravity"
    brain_dir = root / "brain"
    db_path = root / "conversation_summaries.db"
    root.mkdir(parents=True)

    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute(
            """
            CREATE TABLE conversation_summaries (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                workspace_uris TEXT,
                last_modified_time TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?)",
            ("s-a", "Session A", '["file:///workspace/dev/repo-a"]', "2026-10-22T10:00:00Z"),
        )
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?)",
            ("s-b", "Session B", '["file:///workspace/dev/repo-b"]', "2026-10-22T10:00:00Z"),
        )
    conn.close()

    for sid in ["s-a", "s-b"]:
        log_dir = brain_dir / sid / ".system_generated" / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "transcript.jsonl").write_text(
            json.dumps({"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "hi"})
        )

    # Filter for repo-a
    filtered = discover_sessions(workspace_filter="repo-a", base_dir=root)
    assert len(filtered) == 1
    assert filtered[0].session_id == "s-a"

    # Filter for non-existent workspace
    filtered_none = discover_sessions(workspace_filter="repo-nonexistent", base_dir=root)
    assert len(filtered_none) == 0
