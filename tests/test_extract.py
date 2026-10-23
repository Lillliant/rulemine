"""Unit tests for extract module prompt construction and parsing logic."""

import json
from unittest.mock import MagicMock, patch
from rulemine.extract import (
    build_messages,
    extract_events_for_window,
    extract_events_from_text,
    load_fewshots,
    load_system_prompt,
    sanitize_hyphens,
)
from rulemine.models import ExtractedEvent, Window


def test_sanitize_hyphens():
    assert sanitize_hyphens("plain text") == "plain text"
    assert sanitize_hyphens("em\u2014dash") == "em--dash"
    assert sanitize_hyphens("en\u2013dash") == "en-dash"
    assert sanitize_hyphens("") == ""


def test_load_system_prompt_and_fewshots():
    prompt = load_system_prompt()
    assert "extract structured memory" in prompt.lower()
    assert "events" in prompt

    fewshots = load_fewshots()
    assert isinstance(fewshots, list)
    assert len(fewshots) >= 3
    # Check that expected fields exist in fewshots
    for ex in fewshots:
        assert "input_text" in ex
        assert "expected_events" in ex


def test_build_messages():
    # Without fewshots
    msgs_no_fewshots = build_messages(
        window_text="sample text",
        session_id="sess-01",
        workspace="/test/ws",
        include_fewshots=False,
    )
    assert len(msgs_no_fewshots) == 2
    assert msgs_no_fewshots[0]["role"] == "system"
    assert msgs_no_fewshots[1]["role"] == "user"
    assert "sample text" in msgs_no_fewshots[1]["content"]

    # With fewshots
    msgs_with_fewshots = build_messages(
        window_text="sample text",
        session_id="sess-01",
        workspace="/test/ws",
        include_fewshots=True,
    )
    assert len(msgs_with_fewshots) > 2
    assert msgs_with_fewshots[0]["role"] == "system"
    assert msgs_with_fewshots[-1]["role"] == "user"
    assert "sample text" in msgs_with_fewshots[-1]["content"]


def test_extract_events_sanitizes_hyphens_in_output():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "events": [
                        {
                            "type": "correction",
                            "quote": "Do not use em\u2014dashes or en\u2013dashes",
                            "normalized": "Use plain hyphens\u2014no em dashes",
                            "reason": "Style guide\u2014strictly enforced",
                            "confidence": 0.95,
                            "session_id": "sess-dash",
                        }
                    ]
                }
            )
        }
    }

    with patch("httpx.Client.post", return_value=mock_response):
        events = extract_events_from_text("input text", session_id="sess-dash", include_fewshots=False)
        assert len(events) == 1
        ev = events[0]
        assert "\u2014" not in ev.normalized
        assert "\u2013" not in ev.normalized
        assert "\u2014" not in ev.quote
        assert "\u2014" not in ev.reason
        assert "--" in ev.normalized


def test_extract_events_for_window():
    win = Window(
        session_id="win-session",
        workspace="/my/project",
        start_step=0,
        end_step=3,
        text="[USER step=0]\nUse npm instead of pnpm\n[MODEL step=1]\nUnderstood",
        turn_count=2,
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "events": [
                        {
                            "type": "correction",
                            "quote": "Use npm instead of pnpm",
                            "normalized": "Use npm, not pnpm in this repo",
                            "confidence": 0.9,
                            "session_id": "win-session",
                            "workspace": "/my/project",
                        }
                    ]
                }
            )
        }
    }

    with patch("httpx.Client.post", return_value=mock_response):
        events = extract_events_for_window(win, include_fewshots=False)
        assert len(events) == 1
        assert events[0].type == "correction"
        assert events[0].session_id == "win-session"
        assert events[0].workspace == "/my/project"
