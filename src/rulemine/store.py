"""SQLite persistence layer for sessions, raw events, and drafts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
from typing import Any


DEFAULT_DB_PATH = Path("out/rulemine.sqlite")


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialize database tables for sessions, events, and drafts."""
    if str(db_path) != ":memory:":
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                transcript_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                workspace TEXT,
                title TEXT,
                last_mined_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                workspace TEXT,
                confidence REAL,
                data JSON NOT NULL,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drafts (
                draft_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                data JSON NOT NULL,
                status TEXT NOT NULL
            )
            """
        )

    return conn


def compute_transcript_hash(transcript_path: Path | str) -> str:
    """Compute sha256 content hash of transcript file."""
    p = Path(transcript_path)
    if not p.is_file():
        return ""

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_session(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    """Retrieve session record by session_id, or None if not found."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT session_id, transcript_path, content_hash, workspace, title, last_mined_at FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(row)


def save_session(
    conn: sqlite3.Connection,
    session_id: str,
    transcript_path: str,
    content_hash: str,
    workspace: str | None = None,
    title: str | None = None,
    last_mined_at: str | None = None,
) -> None:
    """Insert or update session cache record."""
    mined_at = last_mined_at or datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, transcript_path, content_hash, workspace, title, last_mined_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                transcript_path = excluded.transcript_path,
                content_hash = excluded.content_hash,
                workspace = excluded.workspace,
                title = excluded.title,
                last_mined_at = excluded.last_mined_at
            """,
            (session_id, transcript_path, content_hash, workspace, title, mined_at),
        )


def should_remine(conn: sqlite3.Connection, session_id: str, content_hash: str) -> bool:
    """Check if session needs mining (new session or changed content hash)."""
    stored = get_session(conn, session_id)
    if stored is None:
        return True
    return stored.get("content_hash") != content_hash
