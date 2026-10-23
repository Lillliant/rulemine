"""Optional git evidence linking for decision records."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess


def find_nearby_commit(workspace_path: Path | str, timestamp_iso: str | None = None) -> str | None:
    """Best-effort search for a nearby git commit SHA."""
    if not workspace_path:
        return None

    ws = Path(workspace_path)
    if not ws.is_dir():
        return None

    try:
        # Check if ws is a git repo
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=ws,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if check.returncode != 0 or check.stdout.strip() != "true":
            return None

        if timestamp_iso:
            try:
                clean_ts = timestamp_iso.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean_ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                since = (dt - timedelta(hours=2)).isoformat()
                until = (dt + timedelta(hours=2)).isoformat()
                res = subprocess.run(
                    ["git", "log", f"--since={since}", f"--until={until}", "-n", "1", "--format=%h"],
                    cwd=ws,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3,
                )
                commit = res.stdout.strip()
                if commit:
                    return commit
                return None
            except Exception:
                return None

        # If no timestamp provided, get most recent commit
        res = subprocess.run(
            ["git", "log", "-n", "1", "--format=%h"],
            cwd=ws,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        commit = res.stdout.strip()
        return commit if commit else None

    except Exception:
        return None
