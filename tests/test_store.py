"""Unit tests for SQLite persistence store."""

from pathlib import Path
from rulemine.store import (
    compute_transcript_hash,
    get_session,
    init_db,
    save_session,
    should_remine,
)


def test_init_db(tmp_path: Path):
    db_file = tmp_path / "cache" / "test.sqlite"
    conn = init_db(db_file)
    assert db_file.exists()

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cursor.fetchall()}
    assert "sessions" in tables
    assert "events" in tables
    assert "drafts" in tables


def test_compute_transcript_hash(tmp_path: Path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world", encoding="utf-8")

    h1 = compute_transcript_hash(test_file)
    assert len(h1) == 64  # sha256 hex string

    # Non-existent file
    assert compute_transcript_hash(tmp_path / "nonexistent.txt") == ""


def test_save_and_get_session(tmp_path: Path):
    conn = init_db(tmp_path / "store.sqlite")

    # Initial get
    assert get_session(conn, "sess-1") is None

    # Save session
    save_session(
        conn=conn,
        session_id="sess-1",
        transcript_path="/path/to/transcript.jsonl",
        content_hash="abc123hash",
        workspace="/path/to/ws",
        title="Sample Session",
    )

    stored = get_session(conn, "sess-1")
    assert stored is not None
    assert stored["session_id"] == "sess-1"
    assert stored["content_hash"] == "abc123hash"
    assert stored["workspace"] == "/path/to/ws"
    assert stored["title"] == "Sample Session"

    # Upsert with new hash
    save_session(
        conn=conn,
        session_id="sess-1",
        transcript_path="/path/to/transcript.jsonl",
        content_hash="def456newhash",
        workspace="/path/to/ws",
        title="Updated Session",
    )

    updated = get_session(conn, "sess-1")
    assert updated is not None
    assert updated["content_hash"] == "def456newhash"
    assert updated["title"] == "Updated Session"


def test_should_remine(tmp_path: Path):
    conn = init_db(tmp_path / "store.sqlite")

    # Unknown session should remine
    assert should_remine(conn, "sess-x", "hash-1") is True

    save_session(
        conn=conn,
        session_id="sess-x",
        transcript_path="/path/transcript.jsonl",
        content_hash="hash-1",
    )

    # Identical hash should NOT remine
    assert should_remine(conn, "sess-x", "hash-1") is False

    # Different hash should remine
    assert should_remine(conn, "sess-x", "hash-2") is True
