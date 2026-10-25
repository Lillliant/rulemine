"""Unit tests for Cursor IDE transcript adapter."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import pytest
from typer.testing import CliRunner

from rulemine.adapters.cursor import (
    discover_cursor_sessions,
    load_turns,
)
from rulemine.cli import app

runner = CliRunner()


def test_discover_cursor_workspace_composer(tmp_path: Path):
    """Test discovering composer sessions from workspaceStorage."""
    cursor_dir = tmp_path / "Cursor" / "User"
    ws_dir = cursor_dir / "workspaceStorage" / "mock-workspace-hash-01"
    ws_dir.mkdir(parents=True)

    # 1. Write workspace.json
    (ws_dir / "workspace.json").write_text(
        json.dumps({"folder": "file:///workspace/dev/cursor-project"}),
        encoding="utf-8",
    )

    # 2. Write state.vscdb with ItemTable
    db_path = ws_dir / "state.vscdb"
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)")
        composer_payload = {
            "allComposers": [
                {
                    "composerId": "comp-42",
                    "name": "Add JWT Authentication",
                    "createdAt": 1729680000000,
                    "lastUpdatedAt": 1729680005000,
                    "conversation": [
                        {
                            "bubbleId": "b1",
                            "type": 1,  # User
                            "text": "Please use pyjwt rather than python-jose.",
                        },
                        {
                            "bubbleId": "b2",
                            "type": 2,  # AI
                            "text": "Refactoring authentication to use PyJWT...",
                        },
                    ],
                }
            ]
        }
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            ("composer.composerData", json.dumps(composer_payload)),
        )
    conn.close()

    cache_dir = tmp_path / "cache"
    sessions = discover_cursor_sessions(base_dir=cursor_dir, cache_dir=cache_dir)

    assert len(sessions) == 1
    session = sessions[0]
    assert session.session_id == "cursor-comp-42"
    assert session.title == "Add JWT Authentication"
    assert session.workspace == "/workspace/dev/cursor-project"

    turns = load_turns(session)
    assert len(turns) == 2
    assert turns[0].step_index == 0
    assert turns[0].source == "USER_EXPLICIT"
    assert turns[0].type == "USER_INPUT"
    assert turns[0].content == "Please use pyjwt rather than python-jose."

    assert turns[1].step_index == 1
    assert turns[1].source == "MODEL"
    assert turns[1].type == "PLANNER_RESPONSE"
    assert "PyJWT" in turns[1].content


def test_discover_cursor_global_storage(tmp_path: Path):
    """Test discovering composer sessions from globalStorage cursorDiskKV."""
    cursor_dir = tmp_path / "Cursor" / "User"
    global_dir = cursor_dir / "globalStorage"
    global_dir.mkdir(parents=True)

    db_path = global_dir / "state.vscdb"
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")

        composer_meta = {
            "name": "Fix Database Pool",
            "createdAt": 1729681000000,
            "conversation": [
                {"bubbleId": "b-user-1", "type": "user"},
                {"bubbleId": "b-ai-1", "type": "ai"},
            ],
        }

        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            ("composerData:comp-global-99", json.dumps(composer_meta)),
        )

        bubble_user = {"text": "Make sure max_overflow is at least 10."}
        bubble_ai = {"text": "Setting max_overflow=10 on SQLAlchemy pool."}

        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            ("bubbleId:comp-global-99:b-user-1", json.dumps(bubble_user)),
        )
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            ("bubbleId:comp-global-99:b-ai-1", json.dumps(bubble_ai)),
        )
    conn.close()

    cache_dir = tmp_path / "cache"
    sessions = discover_cursor_sessions(base_dir=cursor_dir, cache_dir=cache_dir)

    assert len(sessions) == 1
    session = sessions[0]
    assert session.session_id == "cursor-comp-global-99"
    assert session.title == "Fix Database Pool"

    turns = load_turns(session)
    assert len(turns) == 2
    assert turns[0].source == "USER_EXPLICIT"
    assert turns[0].content == "Make sure max_overflow is at least 10."
    assert turns[1].source == "MODEL"
    assert turns[1].content == "Setting max_overflow=10 on SQLAlchemy pool."


def test_cli_mine_cursor_dry_run(tmp_path: Path):
    """Test CLI rulemine mine --adapter cursor --dry-run."""
    cursor_dir = tmp_path / "Cursor" / "User"
    ws_dir = cursor_dir / "workspaceStorage" / "mock-ws"
    ws_dir.mkdir(parents=True)

    (ws_dir / "workspace.json").write_text(
        json.dumps({"folder": "file:///workspace/apps/billing"}),
        encoding="utf-8",
    )

    db_path = ws_dir / "state.vscdb"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (
                "composer.composerData",
                json.dumps(
                    {
                        "allComposers": [
                            {
                                "composerId": "comp-cli",
                                "name": "Billing Stripe Integration",
                                "createdAt": now_ms,
                                "conversation": [
                                    {"type": 1, "text": "Do not hardcode Stripe test secret keys!"},
                                    {"type": 2, "text": "Moving Stripe keys to environment variables."},
                                ],
                            }
                        ]
                    }
                ),
            ),
        )
    conn.close()

    res = runner.invoke(
        app,
        [
            "mine",
            "--adapter",
            "cursor",
            "--base-dir",
            str(cursor_dir),
            "--dry-run",
        ],
    )
    assert res.exit_code == 0
    assert "RuleMine Dry-Run Ingest: 1 Discovered Session(s)" in res.output
    assert "Billing Stripe" in res.output
    assert "Integration" in res.output
    assert "billing" in res.output
    assert "Dry-run complete" in res.output
