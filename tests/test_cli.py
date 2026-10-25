"""Unit tests for RuleMine CLI commands (mine, status, etc.)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
from typer.testing import CliRunner

from rulemine.cli import app
from rulemine.models import ExtractedEvent
from rulemine.store import get_events, get_session, get_store_stats, init_db, save_events

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Mine local agent transcripts" in result.output


def test_cli_status_empty(tmp_path: Path):
    nonexistent = tmp_path / "absent.sqlite"
    result = runner.invoke(app, ["status", "--db", str(nonexistent)])
    assert result.exit_code == 0
    assert "No store found" in result.output


def test_cli_status_populated(tmp_path: Path):
    db_file = tmp_path / "populated.sqlite"
    conn = init_db(db_file)
    ev = ExtractedEvent(
        type="correction",
        quote="Use poetry",
        normalized="Use poetry in this repo",
        confidence=0.9,
        session_id="sess-cli-1",
    )
    save_events([ev], conn=conn)

    result = runner.invoke(app, ["status", "--db", str(db_file)])
    assert result.exit_code == 0
    assert "RuleMine Cache Status" in result.output
    assert "Total Extracted Events" in result.output


def test_cli_mine_dry_run():
    result = runner.invoke(
        app,
        [
            "mine",
            "--base-dir",
            "fixtures/antigravity",
            "--since",
            "7d",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry-Run Ingest" in result.output
    assert "Dry-run complete" in result.output


def test_cli_mine_live_mocked(tmp_path: Path):
    db_file = tmp_path / "live_mine.sqlite"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "events": [
                        {
                            "type": "correction",
                            "quote": "always use poetry instead of pipenv",
                            "normalized": "Use poetry instead of pipenv in this repository",
                            "confidence": 0.95,
                            "session_id": "session-fixture-01",
                            "workspace": "/workspace/projects/RuleMine",
                        },
                        {
                            "type": "decision",
                            "quote": "We should use SQLite rather than DuckDB",
                            "normalized": "Use SQLite for local cache store",
                            "reason": "Zero additional dependencies and built-in Python library support",
                            "confidence": 0.9,
                            "session_id": "session-fixture-01",
                            "workspace": "/workspace/projects/RuleMine",
                        },
                    ]
                }
            )
        }
    }

    with patch("httpx.Client.post", return_value=mock_response):
        # First live run: should mine the session
        result1 = runner.invoke(
            app,
            [
                "mine",
                "--base-dir",
                "fixtures/antigravity",
                "--since",
                "7d",
                "--db",
                str(db_file),
            ],
        )
        assert result1.exit_code == 0
        assert "Mining Summary" in result1.output
        assert "Newly Mined: 1 session(s)" in result1.output

        # Verify database contents
        events = get_events(db_path=db_file)
        assert len(events) == 2
        types = {e.type for e in events}
        assert "correction" in types
        assert "decision" in types

        # Check status command
        status_res = runner.invoke(app, ["status", "--db", str(db_file)])
        assert status_res.exit_code == 0
        assert "Total Extracted Events" in status_res.output

        # Second live run: should skip already mined session
        result2 = runner.invoke(
            app,
            [
                "mine",
                "--base-dir",
                "fixtures/antigravity",
                "--since",
                "7d",
                "--db",
                str(db_file),
            ],
        )
        assert result2.exit_code == 0
        assert "Cached / Skipped: 1 session(s)" in result2.output
        assert "Newly Mined: 0 session(s)" in result2.output


def test_cli_mine_timeout_error(tmp_path: Path):
    db_file = tmp_path / "timeout.sqlite"

    with patch("httpx.Client.post", side_effect=httpx.ReadTimeout("timed out")):
        result = runner.invoke(
            app,
            [
                "mine",
                "--base-dir",
                "fixtures/antigravity",
                "--since",
                "7d",
                "--db",
                str(db_file),
                "--timeout",
                "30",
            ],
        )
        assert result.exit_code == 1
        assert "Ollama request timed out after 30s" in result.output
        assert "rulemine mine --timeout" in result.output


def test_cli_mine_timeout_continue_on_error(tmp_path: Path):
    db_file = tmp_path / "continue_error.sqlite"

    with patch("httpx.Client.post", side_effect=httpx.ReadTimeout("timed out")):
        result = runner.invoke(
            app,
            [
                "mine",
                "--base-dir",
                "fixtures/antigravity",
                "--since",
                "7d",
                "--db",
                str(db_file),
                "--continue-on-error",
            ],
        )
        assert result.exit_code == 0
        assert "timed out after" in result.output
        assert "Mining Summary" in result.output


def test_cli_mine_quick_mode():
    result = runner.invoke(
        app,
        [
            "mine",
            "--base-dir",
            "fixtures/antigravity",
            "--since",
            "7d",
            "--dry-run",
            "--quick",
        ],
    )
    assert result.exit_code == 0
    assert "Discovered Session(s)" in result.output


def test_cli_promote_workspace_and_decision_support(tmp_path: Path):
    db_file = tmp_path / "filter_promo.sqlite"
    conn = init_db(db_file)
    ev1 = ExtractedEvent(
        type="decision",
        quote="Use SQLite",
        normalized="Use SQLite for local metadata",
        confidence=0.9,
        session_id="sess-ws-1",
        workspace="/workspace/repoA",
    )
    ev2 = ExtractedEvent(
        type="decision",
        quote="Use SQLite",
        normalized="Use SQLite for local metadata",
        confidence=0.95,
        session_id="sess-ws-2",
        workspace="/workspace/repoA",
    )
    ev3 = ExtractedEvent(
        type="decision",
        quote="Use Postgres",
        normalized="Use Postgres in cloud",
        confidence=0.9,
        session_id="sess-ws-3",
        workspace="/workspace/repoB",
    )
    save_events([ev1, ev2, ev3], conn=conn)

    # 1. Promote with workspace filter
    res = runner.invoke(app, ["promote", "--db", str(db_file), "--workspace", "/workspace/repoA"])
    assert res.exit_code == 0
    assert "Filtered to 2 event(s)" in res.output

    # 2. Review filter by kind
    review_res = runner.invoke(app, ["review", "--db", str(db_file), "--kind", "decision", "--all"])
    assert review_res.exit_code == 0
    assert "Approved all" in review_res.output
