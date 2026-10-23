"""SQLite persistence layer for sessions, raw events, and drafts."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def init_db(db_path: Path | str) -> sqlite3.Connection:
    """Initialize database tables for sessions, events, and drafts."""
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                transcript_path TEXT,
                content_hash TEXT,
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
                session_id TEXT,
                event_type TEXT,
                workspace TEXT,
                confidence REAL,
                data JSON,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drafts (
                draft_id TEXT PRIMARY KEY,
                kind TEXT,
                data JSON,
                status TEXT
            )
            """
        )
    return conn
