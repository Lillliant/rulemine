"""SQLite persistence layer for sessions, raw events, and drafts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from rulemine.models import DecisionDraft, ExtractedEvent, RuleDraft, SkillCandidate

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
                last_mined_at TEXT,
                model TEXT
            )
            """
        )
        # Migrate existing sessions table if model column is missing
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN model TEXT")
        except sqlite3.OperationalError:
            pass

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
            "CREATE INDEX IF NOT EXISTS idx_events_session_id ON events (session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_event_type ON events (event_type)"
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
        "SELECT session_id, transcript_path, content_hash, workspace, title, last_mined_at, model FROM sessions WHERE session_id = ?",
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
    model: str | None = None,
) -> None:
    """Insert or update session cache record."""
    mined_at = last_mined_at or datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, transcript_path, content_hash, workspace, title, last_mined_at, model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                transcript_path = excluded.transcript_path,
                content_hash = excluded.content_hash,
                workspace = excluded.workspace,
                title = excluded.title,
                last_mined_at = excluded.last_mined_at,
                model = excluded.model
            """,
            (session_id, transcript_path, content_hash, workspace, title, mined_at, model),
        )


def should_remine(
    conn: sqlite3.Connection,
    session_id: str,
    content_hash: str,
    model: str | None = None,
) -> bool:
    """Check if session needs mining (new session, changed content hash, or different model)."""
    stored = get_session(conn, session_id)
    if stored is None:
        return True
    if stored.get("content_hash") != content_hash:
        return True
    if model and stored.get("model") and stored.get("model") != model:
        return True
    return False


def save_events(
    events: list[ExtractedEvent],
    db_path: Path | str = DEFAULT_DB_PATH,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Persist extracted events into the events table."""
    if not events:
        return 0

    close_conn = False
    if conn is None:
        conn = init_db(db_path)
        close_conn = True

    count = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            for ev in events:
                conn.execute(
                    """
                    INSERT INTO events (event_id, session_id, event_type, workspace, confidence, data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        workspace = excluded.workspace,
                        confidence = excluded.confidence,
                        data = excluded.data,
                        created_at = excluded.created_at
                    """,
                    (
                        ev.event_id,
                        ev.session_id,
                        ev.type,
                        ev.workspace,
                        ev.confidence,
                        ev.model_dump_json(),
                        ev.timestamp or now_iso,
                    ),
                )
                count += 1
    finally:
        if close_conn:
            conn.close()
    return count


def get_events(
    event_type: str | None = None,
    session_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    conn: sqlite3.Connection | None = None,
) -> list[ExtractedEvent]:
    """Retrieve persisted events filtered by event_type or session_id."""
    close_conn = False
    if conn is None:
        if isinstance(db_path, (str, Path)) and str(db_path) != ":memory:" and not Path(db_path).exists():
            return []
        conn = init_db(db_path)
        close_conn = True

    try:
        query = "SELECT data FROM events WHERE 1=1"
        params: list[Any] = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at ASC"

        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        events: list[ExtractedEvent] = []
        for row in rows:
            raw_data = row["data"] if isinstance(row, sqlite3.Row) else row[0]
            events.append(ExtractedEvent.model_validate_json(raw_data))
        return events
    finally:
        if close_conn:
            conn.close()


def get_store_stats(
    db_path: Path | str = DEFAULT_DB_PATH,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Return counts of sessions, total events, events grouped by type, and last mined timestamp."""
    close_conn = False
    if conn is None:
        if isinstance(db_path, (str, Path)) and str(db_path) != ":memory:" and not Path(db_path).exists():
            return {
                "session_count": 0,
                "event_count": 0,
                "draft_count": 0,
                "events_by_type": {
                    "correction": 0,
                    "decision": 0,
                    "dead_end": 0,
                    "skill_need": 0,
                },
                "last_mined_at": None,
            }
        conn = init_db(db_path)
        close_conn = True

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(last_mined_at) FROM sessions")
        last_mined_row = cursor.fetchone()
        last_mined_at = last_mined_row[0] if last_mined_row else None

        cursor.execute("SELECT COUNT(*) FROM events")
        event_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM drafts")
        draft_count = cursor.fetchone()[0]

        cursor.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
        events_by_type: dict[str, int] = {
            "correction": 0,
            "decision": 0,
            "dead_end": 0,
            "skill_need": 0,
        }
        for row in cursor.fetchall():
            etype = row[0]
            cnt = row[1]
            events_by_type[etype] = cnt

        return {
            "session_count": session_count,
            "event_count": event_count,
            "draft_count": draft_count,
            "events_by_type": events_by_type,
            "last_mined_at": last_mined_at,
        }
    finally:
        if close_conn:
            conn.close()


def save_drafts(
    drafts: list[RuleDraft | DecisionDraft | SkillCandidate],
    db_path: Path | str = DEFAULT_DB_PATH,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Persist drafts into the drafts table."""
    if not drafts:
        return 0

    close_conn = False
    if conn is None:
        conn = init_db(db_path)
        close_conn = True

    count = 0
    try:
        with conn:
            cursor = conn.cursor()
            for d in drafts:
                d_id = d.draft_id
                kind = "rule" if isinstance(d, RuleDraft) else (d.kind if isinstance(d, DecisionDraft) else "skill")

                # Check existing status if already in DB to avoid clobbering user reviews
                cursor.execute("SELECT status FROM drafts WHERE draft_id = ?", (d_id,))
                row = cursor.fetchone()
                if row:
                    existing_status = row["status"] if isinstance(row, sqlite3.Row) else row[0]
                    status_to_use = existing_status
                else:
                    status_to_use = d.status

                dump_dict = d.model_dump()
                dump_dict["status"] = status_to_use
                json_data = json.dumps(dump_dict)

                conn.execute(
                    """
                    INSERT INTO drafts (draft_id, kind, data, status)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(draft_id) DO UPDATE SET
                        kind = excluded.kind,
                        data = excluded.data,
                        status = excluded.status
                    """,
                    (d_id, kind, json_data, status_to_use),
                )
                count += 1
    finally:
        if close_conn:
            conn.close()
    return count


def get_drafts(
    kind: str | None = None,
    status: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Retrieve drafts filtered by kind and/or status."""
    close_conn = False
    if conn is None:
        if isinstance(db_path, (str, Path)) and str(db_path) != ":memory:" and not Path(db_path).exists():
            return []
        conn = init_db(db_path)
        close_conn = True

    try:
        query = "SELECT draft_id, kind, data, status FROM drafts WHERE 1=1"
        params: list[Any] = []
        if kind:
            if kind in ("decisions", "decision"):
                query += " AND kind IN ('decision', 'dead_end')"
            elif kind in ("rules", "rule"):
                query += " AND kind = 'rule'"
            elif kind in ("skills", "skill"):
                query += " AND kind = 'skill'"
            else:
                query += " AND kind = ?"
                params.append(kind)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY draft_id ASC"

        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        drafts: list[dict[str, Any]] = []
        for row in rows:
            draft_id = row["draft_id"] if isinstance(row, sqlite3.Row) else row[0]
            row_kind = row["kind"] if isinstance(row, sqlite3.Row) else row[1]
            raw_data = row["data"] if isinstance(row, sqlite3.Row) else row[2]
            row_status = row["status"] if isinstance(row, sqlite3.Row) else row[3]
            data = json.loads(raw_data)
            data["draft_id"] = draft_id
            data["kind"] = row_kind
            data["status"] = row_status
            drafts.append(data)
        return drafts
    finally:
        if close_conn:
            conn.close()


def update_draft_status(
    draft_id: str,
    status: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    conn: sqlite3.Connection | None = None,
) -> bool:
    """Update review status of a draft."""
    close_conn = False
    if conn is None:
        if isinstance(db_path, (str, Path)) and str(db_path) != ":memory:" and not Path(db_path).exists():
            return False
        conn = init_db(db_path)
        close_conn = True

    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM drafts WHERE draft_id = ?", (draft_id,))
            row = cursor.fetchone()
            if not row:
                return False

            raw_data = row["data"] if isinstance(row, sqlite3.Row) else row[0]
            data = json.loads(raw_data)
            data["status"] = status
            json_data = json.dumps(data)

            cursor.execute(
                "UPDATE drafts SET status = ?, data = ? WHERE draft_id = ?",
                (status, json_data, draft_id),
            )
            return cursor.rowcount > 0
    finally:
        if close_conn:
            conn.close()
