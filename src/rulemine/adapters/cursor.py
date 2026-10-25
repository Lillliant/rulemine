"""Adapter to discover and read Cursor IDE composer and chat transcripts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any
import urllib.parse

from rulemine.adapters.antigravity import _parse_workspace_uris, parse_since
from rulemine.models import SessionRef, Turn

DEFAULT_CURSOR_CACHE_DIR = Path("out/cursor_transcripts")


def get_default_cursor_user_dir() -> Path:
    """Return platform-specific default Cursor User directory."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Cursor" / "User"
    elif sys.platform == "win32":
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "Cursor" / "User"
        return Path.home() / "AppData" / "Roaming" / "Cursor" / "User"
    else:
        # Linux and other Unixes
        config_dir = os.environ.get("XDG_CONFIG_HOME")
        if config_dir:
            return Path(config_dir) / "Cursor" / "User"
        return Path.home() / ".config" / "Cursor" / "User"


def _safe_sqlite_query(
    db_path: Path,
    query_fn: Any,
) -> Any:
    """Safely execute queries on a SQLite file, trying read-only mode first."""
    if not db_path.is_file():
        return None

    # Fast path: try read-only URI connection without copying
    try:
        conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        try:
            return query_fn(conn)
        finally:
            conn.close()
    except Exception:
        pass

    # Fallback path: copy to temp directory (for WAL locks or permission blocks)
    temp_dir = tempfile.mkdtemp()
    try:
        temp_db = Path(temp_dir) / db_path.name
        shutil.copy2(db_path, temp_db)

        wal = db_path.with_name(db_path.name + "-wal")
        shm = db_path.with_name(db_path.name + "-shm")
        if wal.exists():
            shutil.copy2(wal, Path(temp_dir) / (temp_db.name + "-wal"))
        if shm.exists():
            shutil.copy2(shm, Path(temp_dir) / (temp_db.name + "-shm"))

        conn = sqlite3.connect(str(temp_db))
        try:
            return query_fn(conn)
        finally:
            conn.close()
    except Exception:
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _parse_workspace_json(workspace_json_path: Path) -> str | None:
    """Extract workspace directory path from Cursor's workspace.json."""
    if not workspace_json_path.is_file():
        return None
    try:
        data = json.loads(workspace_json_path.read_text(encoding="utf-8", errors="replace"))
        folder = data.get("folder") or data.get("workspace")
        if folder:
            return _parse_workspace_uris(folder)
    except Exception:
        pass
    return None


def _export_turns_to_jsonl(turns: list[Turn], out_path: Path) -> None:
    """Write parsed turns to a cached jsonl file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "step_index": t.step_index,
                "source": t.source,
                "type": t.type,
                "created_at": t.created_at,
                "content": t.content,
            }
        )
        for t in turns
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_composer_conversation(
    conv_items: list[dict[str, Any]],
    bubble_lookup: dict[str, str] | None = None,
) -> list[Turn]:
    """Parse composer conversation items/bubbles into Turn models."""
    turns: list[Turn] = []
    step = 0
    lookup = bubble_lookup or {}

    for item in conv_items:
        if not isinstance(item, dict):
            continue

        raw_type = item.get("type")
        bubble_id = item.get("bubbleId", "")

        # Determine user vs assistant
        # Cursor uses 1 for user, 2 for assistant, or "user" / "ai" / "assistant"
        is_user = raw_type in (1, "user", "USER", "human")

        # Resolve text content
        text = item.get("text") or item.get("richText") or ""
        if not text and bubble_id and bubble_id in lookup:
            text = lookup[bubble_id]

        if not isinstance(text, str):
            text = str(text)

        text = text.strip()
        if not text:
            continue

        created_at = None
        ts = item.get("createdAt")
        if ts:
            try:
                # Could be epoch ms or iso string
                if isinstance(ts, (int, float)):
                    created_at = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat()
                elif isinstance(ts, str):
                    created_at = ts
            except Exception:
                pass

        turns.append(
            Turn(
                step_index=step,
                source="USER_EXPLICIT" if is_user else "MODEL",
                type="USER_INPUT" if is_user else "PLANNER_RESPONSE",
                created_at=created_at,
                content=text,
            )
        )
        step += 1

    return turns


def _parse_chatdata_tabs(tabs: list[dict[str, Any]]) -> list[tuple[str, str, float | None, list[Turn]]]:
    """Parse older Cursor aichat tabs into (tab_id, title, mtime, turns)."""
    results = []
    for tab in tabs:
        if not isinstance(tab, dict):
            continue
        tab_id = str(tab.get("id") or tab.get("tabId") or f"tab-{len(results)}")
        title = tab.get("chatTitle") or tab.get("title") or "Cursor Chat"
        bubbles = tab.get("bubbles", [])
        if not isinstance(bubbles, list) or not bubbles:
            continue

        turns: list[Turn] = []
        step = 0
        last_ts: float | None = None

        for b in bubbles:
            if not isinstance(b, dict):
                continue
            b_type = b.get("type")
            is_user = b_type in ("user", 1, "USER")
            text = (b.get("text") or b.get("rawText") or "").strip()
            if not text:
                continue

            ts = b.get("timestamp") or b.get("createdAt")
            created_at = None
            if ts:
                try:
                    if isinstance(ts, (int, float)):
                        last_ts = ts / 1000.0 if ts > 1e11 else float(ts)
                        created_at = datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()
                    elif isinstance(ts, str):
                        created_at = ts
                except Exception:
                    pass

            turns.append(
                Turn(
                    step_index=step,
                    source="USER_EXPLICIT" if is_user else "MODEL",
                    type="USER_INPUT" if is_user else "PLANNER_RESPONSE",
                    created_at=created_at,
                    content=text,
                )
            )
            step += 1

        if turns:
            results.append((tab_id, title, last_ts, turns))

    return results


def discover_cursor_sessions(
    since: datetime | None = None,
    workspace_filter: str | None = None,
    base_dir: Path | None = None,
    cache_dir: Path = DEFAULT_CURSOR_CACHE_DIR,
) -> list[SessionRef]:
    """Discover Cursor IDE conversation sessions since a given datetime."""
    user_dir = base_dir or get_default_cursor_user_dir()
    if not user_dir.is_dir():
        # If user passed a direct state.vscdb or custom workspace directory
        if user_dir.is_file() and user_dir.name == "state.vscdb":
            workspace_dirs = [user_dir.parent]
            global_db = None
        else:
            return []
    else:
        workspace_storage = user_dir / "workspaceStorage"
        workspace_dirs = []
        try:
            if workspace_storage.is_dir():
                workspace_dirs = list(workspace_storage.iterdir())
        except (PermissionError, OSError):
            workspace_dirs = []
        global_db = user_dir / "globalStorage" / "state.vscdb"

    since_dt = parse_since(since)
    sessions: list[SessionRef] = []
    composer_workspaces: dict[str, str] = {}

    # 1. Process workspaceStorage (workspace-specific composer & chat sessions)
    for ws_dir in workspace_dirs:
        try:
            if not ws_dir.is_dir():
                continue

            state_db = ws_dir / "state.vscdb"
            if not state_db.is_file():
                continue

            if since_dt is not None:
                if state_db.stat().st_mtime < since_dt.timestamp() - 3600:
                    continue
        except (PermissionError, OSError):
            continue

        workspace_path = _parse_workspace_json(ws_dir / "workspace.json")

        # Workspace filter check
        if workspace_filter and workspace_path:
            wf_clean = workspace_filter.strip().rstrip("/\\").lower()
            ws_clean = workspace_path.strip().rstrip("/\\").lower()
            if wf_clean != ws_clean and wf_clean not in ws_clean:
                continue

        def query_workspace_db(conn: sqlite3.Connection) -> dict[str, str]:
            res = {}
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ItemTable'")
            if not cursor.fetchone():
                return res
            cursor.execute(
                "SELECT key, value FROM ItemTable WHERE key IN ('workbench.panel.aichat.chatdata', 'composer.composerData')"
            )
            for k, v in cursor.fetchall():
                res[k] = v
            return res

        items = _safe_sqlite_query(state_db, query_workspace_db) or {}

        # Parse composer.composerData
        if "composer.composerData" in items:
            try:
                comp_data = json.loads(items["composer.composerData"])
                all_comps = comp_data.get("allComposers", [])
                for comp in all_comps:
                    comp_id = comp.get("composerId")
                    if not comp_id:
                        continue
                    if workspace_path:
                        composer_workspaces[comp_id] = workspace_path
                    title = comp.get("name") or comp.get("text") or "Cursor Composer"
                    conv = comp.get("conversation", [])
                    turns = _parse_composer_conversation(conv)
                    if not turns:
                        continue

                    # Determine mtime
                    ts = comp.get("lastUpdatedAt") or comp.get("createdAt")
                    mtime = (ts / 1000.0) if ts else state_db.stat().st_mtime
                    mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)

                    if since_dt and mtime_dt < since_dt:
                        continue

                    session_id = f"cursor-{comp_id}"
                    out_transcript = cache_dir / f"{session_id}.jsonl"
                    _export_turns_to_jsonl(turns, out_transcript)

                    sessions.append(
                        SessionRef(
                            session_id=session_id,
                            transcript_path=str(out_transcript),
                            workspace=workspace_path,
                            title=title,
                            mtime=mtime,
                        )
                    )
            except Exception:
                pass

        # Parse workbench.panel.aichat.chatdata
        if "workbench.panel.aichat.chatdata" in items:
            try:
                chat_data = json.loads(items["workbench.panel.aichat.chatdata"])
                tabs = chat_data.get("tabs", [])
                for tab_id, title, tab_mtime, turns in _parse_chatdata_tabs(tabs):
                    mtime = tab_mtime or state_db.stat().st_mtime
                    mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)

                    if since_dt and mtime_dt < since_dt:
                        continue

                    session_id = f"cursor-{tab_id}"
                    out_transcript = cache_dir / f"{session_id}.jsonl"
                    _export_turns_to_jsonl(turns, out_transcript)

                    sessions.append(
                        SessionRef(
                            session_id=session_id,
                            transcript_path=str(out_transcript),
                            workspace=workspace_path,
                            title=title,
                            mtime=mtime,
                        )
                    )
            except Exception:
                pass

    # 2. Process globalStorage/state.vscdb (global composerData and bubble entries)
    if global_db and global_db.is_file():
        # Fast mtime check
        skip_global = False
        if since_dt is not None:
            try:
                if global_db.stat().st_mtime < since_dt.timestamp() - 3600:
                    skip_global = True
            except OSError:
                skip_global = True

        if not skip_global:
            def query_global_db(conn: sqlite3.Connection) -> dict[str, str]:
                composers: dict[str, str] = {}
                cursor = conn.cursor()

                # Check cursorDiskKV table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cursorDiskKV'")
                if cursor.fetchone():
                    cursor.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'")
                    for k, v in cursor.fetchall():
                        composers[k.replace("composerData:", "")] = v

                # Fallback to ItemTable if cursorDiskKV wasn't used
                if not composers:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ItemTable'")
                    if cursor.fetchone():
                        cursor.execute("SELECT key, value FROM ItemTable WHERE key LIKE 'composerData:%'")
                        for k, v in cursor.fetchall():
                            composers[k.replace("composerData:", "")] = v

                return composers

            composers = _safe_sqlite_query(global_db, query_global_db) or {}
            for comp_id, raw_json in composers.items():
                session_id = f"cursor-{comp_id}"
                # Skip if already discovered from workspaceStorage
                if any(s.session_id == session_id for s in sessions):
                    continue

                workspace_path = composer_workspaces.get(comp_id)
                if workspace_filter and workspace_path:
                    wf_clean = workspace_filter.strip().rstrip("/\\").lower()
                    ws_clean = workspace_path.strip().rstrip("/\\").lower()
                    if wf_clean != ws_clean and wf_clean not in ws_clean:
                        continue

                try:
                    data = json.loads(raw_json)
                    title = data.get("name") or data.get("text") or "Cursor Composer"

                    ts = data.get("lastUpdatedAt") or data.get("createdAt")
                    mtime = (ts / 1000.0) if ts else global_db.stat().st_mtime
                    mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)

                    if since_dt and mtime_dt < since_dt:
                        continue

                    conv = data.get("conversation", [])
                    turns: list[Turn] = []

                    if conv:
                        # Parse bubble texts if stored separately as JSON blobs
                        bubble_lookup: dict[str, str] = {}
                        needed_bubble_ids = [
                            item.get("bubbleId")
                            for item in conv
                            if isinstance(item, dict) and not (item.get("text") or item.get("richText")) and item.get("bubbleId")
                        ]
                        if needed_bubble_ids:
                            def query_bubbles(conn: sqlite3.Connection) -> dict[str, str]:
                                b_res = {}
                                c = conn.cursor()
                                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cursorDiskKV'")
                                if not c.fetchone():
                                    return b_res
                                for bid in needed_bubble_ids:
                                    c.execute("SELECT value FROM cursorDiskKV WHERE key = ?", (f"bubbleId:{comp_id}:{bid}",))
                                    row = c.fetchone()
                                    if row:
                                        try:
                                            bdata = json.loads(row[0])
                                            b_res[bid] = bdata.get("text") or bdata.get("richText") or ""
                                        except Exception:
                                            b_res[bid] = row[0]
                                return b_res

                            bubble_lookup = _safe_sqlite_query(global_db, query_bubbles) or {}

                        turns = _parse_composer_conversation(conv, bubble_lookup=bubble_lookup)
                    else:
                        # Fallback: query bubbles from cursorDiskKV directly
                        def query_all_bubbles_for_comp(conn: sqlite3.Connection) -> list[dict[str, Any]]:
                            b_list = []
                            c = conn.cursor()
                            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cursorDiskKV'")
                            if not c.fetchone():
                                return b_list
                            c.execute("SELECT value FROM cursorDiskKV WHERE key LIKE 'bubbleId:' || ? || ':%'", (comp_id,))
                            for row in c.fetchall():
                                try:
                                    b = json.loads(row[0])
                                    text = (b.get("text") or b.get("rawText") or "").strip()
                                    if text:
                                        b_list.append(b)
                                except Exception:
                                    pass
                            return b_list

                        raw_bubbles = _safe_sqlite_query(global_db, query_all_bubbles_for_comp) or []
                        raw_bubbles.sort(key=lambda b: str(b.get("createdAt") or ""))
                        step = 0
                        for b in raw_bubbles:
                            b_type = b.get("type")
                            is_user = b_type in ("user", 1, "USER", "human")
                            text = (b.get("text") or b.get("rawText") or "").strip()
                            if not text:
                                continue
                            turns.append(
                                Turn(
                                    step_index=step,
                                    source="USER_EXPLICIT" if is_user else "MODEL",
                                    type="USER_INPUT" if is_user else "PLANNER_RESPONSE",
                                    created_at=b.get("createdAt"),
                                    content=text,
                                )
                            )
                            step += 1

                    if not turns:
                        continue

                    out_transcript = cache_dir / f"{session_id}.jsonl"
                    _export_turns_to_jsonl(turns, out_transcript)

                    sessions.append(
                        SessionRef(
                            session_id=session_id,
                            transcript_path=str(out_transcript),
                            workspace=workspace_path,
                            title=title,
                            mtime=mtime,
                        )
                    )
                except Exception:
                    continue

    sessions.sort(key=lambda s: s.mtime or 0.0, reverse=True)
    return sessions


def load_turns(session: SessionRef) -> list[Turn]:
    """Load turns from session's cached jsonl transcript file."""
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
