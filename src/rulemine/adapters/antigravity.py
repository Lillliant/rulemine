"""Adapter to discover and read Antigravity agent transcripts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any
import urllib.parse

from rulemine.models import SessionRef, Turn


def parse_since(since: str | datetime | None) -> datetime | None:
    """Parse a since string or datetime into a UTC datetime.

    Supports formats:
      - '7d', '14d', '1d' (days ago)
      - '24h', '2h', '1h' (hours ago)
      - '30m', '15m' (minutes ago)
      - 'YYYY-MM-DD'
      - ISO-8601 strings
    """
    if since is None:
        return None

    if isinstance(since, datetime):
        if since.tzinfo is None:
            return since.replace(tzinfo=timezone.utc)
        return since.astimezone(timezone.utc)

    s = since.strip()
    now = datetime.now(timezone.utc)

    # Relative time matching (e.g. 7d, 24h, 30m)
    rel_match = re.match(r"^(\d+)\s*([dhm])$", s, re.IGNORECASE)
    if rel_match:
        qty = int(rel_match.group(1))
        unit = rel_match.group(2).lower()
        if unit == "d":
            return now - timedelta(days=qty)
        elif unit == "h":
            return now - timedelta(hours=qty)
        elif unit == "m":
            return now - timedelta(minutes=qty)

    # YYYY-MM-DD date matching
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            dt = datetime.strptime(s, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # ISO-8601 format
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError as e:
        raise ValueError(
            f"Unsupported date format for --since: '{since}'. "
            "Use e.g. '7d', '24h', '30m', or 'YYYY-MM-DD'."
        ) from e


def _parse_workspace_uris(uris_raw: str | None) -> str | None:
    """Parse workspace_uris string into an absolute filesystem path."""
    if not uris_raw:
        return None

    uris_raw = uris_raw.strip()
    if not uris_raw:
        return None

    uris: list[str] = []
    try:
        parsed = json.loads(uris_raw)
        if isinstance(parsed, list):
            uris = [u for u in parsed if isinstance(u, str)]
        elif isinstance(parsed, str):
            uris = [parsed]
    except Exception:
        uris = [uris_raw]

    for u in uris:
        cleaned = u.strip()
        if cleaned.startswith("file://"):
            cleaned = cleaned[7:]
        decoded = urllib.parse.unquote(cleaned)
        if decoded:
            return decoded

    return None


def _read_conversation_summaries(db_path: Path) -> dict[str, dict[str, Any]]:
    """Safely read conversation summaries from sqlite DB avoiding WAL locks."""
    summaries: dict[str, dict[str, Any]] = {}
    if not db_path.is_file():
        return summaries

    temp_dir = tempfile.mkdtemp()
    try:
        temp_db = Path(temp_dir) / "summaries_copy.db"
        shutil.copy2(db_path, temp_db)

        wal = db_path.with_name(db_path.name + "-wal")
        shm = db_path.with_name(db_path.name + "-shm")
        if wal.exists():
            shutil.copy2(wal, Path(temp_dir) / (temp_db.name + "-wal"))
        if shm.exists():
            shutil.copy2(shm, Path(temp_dir) / (temp_db.name + "-shm"))

        conn = sqlite3.connect(str(temp_db))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_summaries'"
            )
            if not cursor.fetchone():
                return summaries

            cursor.execute(
                "SELECT conversation_id, title, workspace_uris, last_modified_time FROM conversation_summaries"
            )
            for conv_id, title, uris_raw, last_mod in cursor.fetchall():
                ws = _parse_workspace_uris(uris_raw)
                summaries[str(conv_id)] = {
                    "title": title or None,
                    "workspace": ws,
                    "last_modified_time": last_mod,
                }
        finally:
            conn.close()
    except Exception:
        # Gracefully tolerate any read error
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return summaries


def discover_sessions(
    since: datetime | None = None,
    workspace_filter: str | None = None,
    base_dir: Path | None = None,
) -> list[SessionRef]:
    """Discover Antigravity conversation sessions since a given datetime."""
    root = base_dir or (Path.home() / ".gemini" / "antigravity")
    brain_dir = root / "brain"
    summaries_db = root / "conversation_summaries.db"

    db_meta = _read_conversation_summaries(summaries_db)
    sessions: list[SessionRef] = []

    if not brain_dir.is_dir():
        return sessions

    since_dt = parse_since(since)

    # Search for transcript.jsonl under brain/<id>/.system_generated/logs/transcript.jsonl
    for transcript_path in brain_dir.glob("*/.system_generated/logs/transcript.jsonl"):
        session_id = transcript_path.parent.parent.parent.name
        try:
            st = transcript_path.stat()
            mtime = st.st_mtime
            mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        except OSError:
            continue

        # Filter by since
        if since_dt is not None:
            if mtime_dt < since_dt:
                # Check if last turns in transcript are newer than since_dt
                has_recent_turn = False
                try:
                    with open(transcript_path, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 4096))
                        lines = f.read().decode("utf-8", errors="replace").splitlines()
                        for line in reversed(lines):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                ca = data.get("created_at")
                                if ca:
                                    dt = datetime.fromisoformat(ca.replace("Z", "+00:00"))
                                    if dt.tzinfo is None:
                                        dt = dt.replace(tzinfo=timezone.utc)
                                    if dt >= since_dt:
                                        has_recent_turn = True
                                        break
                            except Exception:
                                continue
                except Exception:
                    pass

                if not has_recent_turn:
                    continue

        meta = db_meta.get(session_id, {})
        title = meta.get("title")
        workspace = meta.get("workspace")

        # Filter by workspace if specified
        if workspace_filter:
            wf_clean = workspace_filter.strip().rstrip("/\\").lower()
            ws_clean = (workspace or "").strip().rstrip("/\\").lower()
            match = False
            if wf_clean and ws_clean:
                if wf_clean == ws_clean or wf_clean in ws_clean:
                    match = True
                else:
                    try:
                        if Path(workspace).resolve() == Path(workspace_filter).resolve():
                            match = True
                    except Exception:
                        pass
            if not match:
                continue

        sessions.append(
            SessionRef(
                session_id=session_id,
                transcript_path=str(transcript_path),
                workspace=workspace,
                title=title,
                mtime=mtime,
            )
        )

    # Sort descending by mtime
    sessions.sort(key=lambda s: s.mtime or 0.0, reverse=True)
    return sessions


def load_turns(session: SessionRef) -> list[Turn]:
    """Load and parse turns from session transcript.jsonl."""
    turns: list[Turn] = []
    path = Path(session.transcript_path)
    if not path.is_file():
        return turns

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        continue
                    turns.append(
                        Turn(
                            step_index=data.get("step_index"),
                            source=data.get("source"),
                            type=data.get("type"),
                            created_at=data.get("created_at"),
                            content=data.get("content", "") or "",
                        )
                    )
                except Exception:
                    continue
    except Exception:
        pass

    return turns
