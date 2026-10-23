"""Unit tests for JSON cleaning, schema validation, and retry repair."""

import json
from unittest.mock import MagicMock, patch
import pytest
from rulemine.extract import clean_json_text, extract_events_from_text
from rulemine.models import ExtractResponse, ExtractedEvent


def test_clean_json_text_markdown_fences():
    fenced = '```json\n{"events": []}\n```'
    assert clean_json_text(fenced) == '{"events": []}'


def test_clean_json_text_prefixed():
    prefixed = 'Here is your output:\n```\n{"events": []}\n```\nHope this helps!'
    assert clean_json_text(prefixed) == '{"events": []}'


def test_clean_json_text_raw_with_noise():
    noisy = 'Sure! {"events": []} is the answer.'
    assert clean_json_text(noisy) == '{"events": []}'


def test_models_validation():
    event_data = {
        "type": "correction",
        "quote": "Use npm instead of yarn",
        "normalized": "Use npm, not yarn",
        "confidence": 0.85,
        "session_id": "test-session",
    }
    event = ExtractedEvent.model_validate(event_data)
    assert event.type == "correction"
    assert event.confidence == 0.85
    assert len(event.event_id) == 16

    resp = ExtractResponse(events=[event])
    assert len(resp.events) == 1
    dumped = resp.model_dump_json()
    assert "Use npm, not yarn" in dumped


def test_extract_events_drops_low_confidence():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "events": [
                        {
                            "type": "correction",
                            "quote": "high confidence",
                            "normalized": "High conf rule",
                            "confidence": 0.9,
                            "session_id": "sess-1",
                        },
                        {
                            "type": "correction",
                            "quote": "low confidence",
                            "normalized": "Low conf rule",
                            "confidence": 0.4,
                            "session_id": "sess-1",
                        },
                    ]
                }
            )
        }
    }

    with patch("httpx.Client.post", return_value=mock_response):
        events = extract_events_from_text("some text", session_id="sess-1")
        assert len(events) == 1
        assert events[0].normalized == "High conf rule"


def test_extract_events_repair_retry_on_bad_json():
    bad_res = MagicMock()
    bad_res.status_code = 200
    bad_res.json.return_value = {"message": {"content": "This is not valid json {"}}

    good_res = MagicMock()
    good_res.status_code = 200
    good_res.json.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "events": [
                        {
                            "type": "decision",
                            "quote": "Use SQLite",
                            "normalized": "Use SQLite for cache",
                            "confidence": 0.9,
                            "session_id": "sess-repair",
                        }
                    ]
                }
            )
        }
    }

    with patch("httpx.Client.post", side_effect=[bad_res, good_res]) as mock_post:
        events = extract_events_from_text("sample input", session_id="sess-repair")
        assert mock_post.call_count == 2
        assert len(events) == 1
        assert events[0].normalized == "Use SQLite for cache"
