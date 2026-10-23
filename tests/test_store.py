"""Unit tests for SQLite persistence store."""

from pathlib import Path
from rulemine.models import DecisionDraft, ExtractedEvent, RuleDraft, SkillCandidate
from rulemine.store import (
    compute_transcript_hash,
    get_drafts,
    get_events,
    get_session,
    get_store_stats,
    init_db,
    save_drafts,
    save_events,
    save_session,
    should_remine,
    update_draft_status,
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
        model="gemma2:2b",
    )

    stored = get_session(conn, "sess-1")
    assert stored is not None
    assert stored["session_id"] == "sess-1"
    assert stored["content_hash"] == "abc123hash"
    assert stored["workspace"] == "/path/to/ws"
    assert stored["title"] == "Sample Session"
    assert stored["model"] == "gemma2:2b"

    # Upsert with new hash and model
    save_session(
        conn=conn,
        session_id="sess-1",
        transcript_path="/path/to/transcript.jsonl",
        content_hash="def456newhash",
        workspace="/path/to/ws",
        title="Updated Session",
        model="gemma3:4b",
    )

    updated = get_session(conn, "sess-1")
    assert updated is not None
    assert updated["content_hash"] == "def456newhash"
    assert updated["title"] == "Updated Session"
    assert updated["model"] == "gemma3:4b"


def test_should_remine(tmp_path: Path):
    conn = init_db(tmp_path / "store.sqlite")

    # Unknown session should remine
    assert should_remine(conn, "sess-x", "hash-1") is True

    save_session(
        conn=conn,
        session_id="sess-x",
        transcript_path="/path/transcript.jsonl",
        content_hash="hash-1",
        model="gemma2:2b",
    )

    # Identical hash and model should NOT remine
    assert should_remine(conn, "sess-x", "hash-1", model="gemma2:2b") is False

    # Different hash should remine
    assert should_remine(conn, "sess-x", "hash-2", model="gemma2:2b") is True

    # Different model should remine
    assert should_remine(conn, "sess-x", "hash-1", model="gemma3:4b") is True


def test_save_and_get_events(tmp_path: Path):
    db_file = tmp_path / "events_test.sqlite"

    ev1 = ExtractedEvent(
        type="correction",
        quote="Use npm instead of yarn",
        normalized="Use npm, not yarn in this repo",
        reason="Pinned CI workflow",
        confidence=0.9,
        session_id="sess-1",
        step_index=2,
        workspace="/workspace/app1",
    )
    ev2 = ExtractedEvent(
        type="decision",
        quote="Let's use PostgreSQL for persistence",
        normalized="Use PostgreSQL for primary database storage",
        reason="ACID transactions required",
        confidence=0.85,
        session_id="sess-1",
        step_index=5,
        workspace="/workspace/app1",
    )
    ev3 = ExtractedEvent(
        type="dead_end",
        quote="Docker compose in root failed due to port clash",
        normalized="Avoid root docker-compose for test harnesses",
        reason="Port conflicts in CI runner",
        confidence=0.8,
        session_id="sess-2",
        step_index=1,
        workspace="/workspace/app2",
    )

    # Save events
    saved_count = save_events([ev1, ev2, ev3], db_path=db_file)
    assert saved_count == 3

    # Retrieve all
    all_events = get_events(db_path=db_file)
    assert len(all_events) == 3

    # Filter by event_type
    corrections = get_events(event_type="correction", db_path=db_file)
    assert len(corrections) == 1
    assert corrections[0].normalized == "Use npm, not yarn in this repo"

    # Filter by session_id
    sess1_events = get_events(session_id="sess-1", db_path=db_file)
    assert len(sess1_events) == 2

    # Empty list handling
    assert save_events([], db_path=db_file) == 0


def test_get_store_stats(tmp_path: Path):
    db_file = tmp_path / "stats_test.sqlite"

    # Non-existent DB stats
    empty_stats = get_store_stats(tmp_path / "nonexistent.sqlite")
    assert empty_stats["session_count"] == 0
    assert empty_stats["event_count"] == 0

    # Populated DB
    conn = init_db(db_file)
    save_session(
        conn=conn,
        session_id="sess-stats-1",
        transcript_path="/path/t1.jsonl",
        content_hash="hash-t1",
        last_mined_at="2026-10-23T10:00:00Z",
    )

    ev_corr = ExtractedEvent(
        type="correction",
        quote="Use hyphens",
        normalized="Use plain hyphens",
        confidence=0.9,
        session_id="sess-stats-1",
    )
    ev_dec = ExtractedEvent(
        type="decision",
        quote="Use SQLite",
        normalized="Use SQLite",
        confidence=0.8,
        session_id="sess-stats-1",
    )
    save_events([ev_corr, ev_dec], conn=conn)

    stats = get_store_stats(conn=conn)
    assert stats["session_count"] == 1
    assert stats["event_count"] == 2
    assert stats["last_mined_at"] == "2026-10-23T10:00:00Z"
    assert stats["events_by_type"]["correction"] == 1
    assert stats["events_by_type"]["decision"] == 1
    assert stats["events_by_type"]["dead_end"] == 0
    assert stats["events_by_type"]["skill_need"] == 0


def test_save_get_and_update_drafts(tmp_path: Path):
    db_file = tmp_path / "drafts_test.sqlite"

    rule = RuleDraft(
        text="Use plain hyphens, never em dashes",
        support_count=2,
        workspaces=["/workspace/repo1"],
        source_event_ids=["ev1", "ev2"],
    )
    decision = DecisionDraft(
        text="Use SQLite for local caching",
        kind="decision",
        reason="Zero config, no daemon required",
        git_sha="abcdef1",
        workspace="/workspace/repo1",
        source_event_ids=["ev3"],
    )
    skill = SkillCandidate(
        name="log-task-to-obsidian",
        description="Automate task logging to central vault.",
        trigger="When a task milestone is completed.",
        support_count=3,
        workspaces=["/workspace/repo1", "/workspace/repo2"],
        source_event_ids=["ev4", "ev5", "ev6"],
    )

    # Save drafts
    count = save_drafts([rule, decision, skill], db_path=db_file)
    assert count == 3

    # Retrieve all drafts
    all_drafts = get_drafts(db_path=db_file)
    assert len(all_drafts) == 3

    # Filter by kind
    rule_drafts = get_drafts(kind="rule", db_path=db_file)
    assert len(rule_drafts) == 1
    assert rule_drafts[0]["text"] == "Use plain hyphens, never em dashes"
    assert rule_drafts[0]["status"] == "proposed"

    decision_drafts = get_drafts(kind="decision", db_path=db_file)
    assert len(decision_drafts) == 1
    assert decision_drafts[0]["text"] == "Use SQLite for local caching"

    skill_drafts = get_drafts(kind="skill", db_path=db_file)
    assert len(skill_drafts) == 1
    assert skill_drafts[0]["name"] == "log-task-to-obsidian"

    # Update status
    r_id = rule.draft_id
    success = update_draft_status(r_id, "approved", db_path=db_file)
    assert success is True

    approved_drafts = get_drafts(status="approved", db_path=db_file)
    assert len(approved_drafts) == 1
    assert approved_drafts[0]["draft_id"] == r_id

    # Re-saving existing approved draft preserves approved status
    save_drafts([rule], db_path=db_file)
    updated_draft = get_drafts(kind="rule", db_path=db_file)[0]
    assert updated_draft["status"] == "approved"
