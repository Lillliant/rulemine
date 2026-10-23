"""Extraction pipeline calling local Ollama Gemma for structured event parsing."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import httpx
from pydantic import ValidationError

from rulemine.models import ExtractResponse, ExtractedEvent, Window

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "extract_system.md"
DEFAULT_FEWSHOT_PATH = PROMPTS_DIR / "extract_fewshot.json"


def sanitize_hyphens(text: str) -> str:
    """Replace em dashes and en dashes with plain hyphens or double hyphens."""
    if not text:
        return text
    # Replace em-dash with double hyphen
    # Replace en-dash with single hyphen
    return text.replace("\u2014", "--").replace("\u2013", "-")


def load_system_prompt() -> str:
    """Load system prompt from markdown file or fallback default."""
    if DEFAULT_SYSTEM_PROMPT_PATH.exists():
        return sanitize_hyphens(DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip())
    return (
        "You extract structured memory from coding-agent transcripts. Output JSON only. "
        "Types: correction | decision | dead_end | skill_need. "
        "If nothing clear, return {\"events\": []}."
    )


def load_fewshots() -> list[dict[str, Any]]:
    """Load few-shot examples if present."""
    if DEFAULT_FEWSHOT_PATH.exists():
        try:
            return json.loads(DEFAULT_FEWSHOT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def clean_json_text(text: str) -> str:
    """Strip markdown fences and extraneous wrapper text from model response."""
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    # Find outer bracket/brace if still noisy
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return text


def build_messages(
    window_text: str,
    session_id: str = "smoke-session",
    workspace: str | None = None,
    include_fewshots: bool = True,
) -> list[dict[str, str]]:
    """Construct prompt messages including system instructions, few-shots, and window content."""
    system_prompt = load_system_prompt()
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if include_fewshots:
        fewshots = load_fewshots()
        for shot in fewshots:
            input_text = shot.get("input_text", "")
            shot_sess = shot.get("session_id", "example-session")
            shot_ws = shot.get("workspace", "unknown")
            shot_events = shot.get("expected_events", [])
            shot_user = (
                f"Analyze the following conversation window from session '{shot_sess}' "
                f"(workspace: {shot_ws or 'unknown'}).\n\n"
                f"--- CONVERSATION START ---\n"
                f"{input_text}\n"
                f"--- CONVERSATION END ---\n\n"
                f"Extract all corrections, decisions, dead ends, or skill needs. "
                f"Remember: prefer {{\"events\": []}} over guessing."
            )
            shot_assistant = json.dumps({"events": shot_events})
            messages.append({"role": "user", "content": shot_user})
            messages.append({"role": "assistant", "content": shot_assistant})

    user_content = (
        f"Analyze the following conversation window from session '{session_id}' "
        f"(workspace: {workspace or 'unknown'}).\n\n"
        f"--- CONVERSATION START ---\n"
        f"{window_text}\n"
        f"--- CONVERSATION END ---\n\n"
        f"Extract all corrections, decisions, dead ends, or skill needs. "
        f"Remember: prefer {{\"events\": []}} over guessing."
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def extract_events_from_text(
    text: str,
    session_id: str = "adhoc-session",
    workspace: str | None = None,
    model: str = "gemma2:2b",
    host: str = "http://127.0.0.1:11434",
    timeout: float = 60.0,
    include_fewshots: bool = True,
) -> list[ExtractedEvent]:
    """Call Ollama chat API to extract structured events from formatted text."""
    schema = ExtractResponse.model_json_schema()
    messages = build_messages(
        window_text=text,
        session_id=session_id,
        workspace=workspace,
        include_fewshots=include_fewshots,
    )

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": {
            "temperature": 0.2,
        },
    }

    with httpx.Client(base_url=host, timeout=timeout) as client:
        response = client.post("/api/chat", json=payload)
        response.raise_for_status()
        res_json = response.json()
        raw_content = res_json.get("message", {}).get("content", "")

        # Parse and validate with Pydantic
        cleaned = clean_json_text(raw_content)
        parsed_response: ExtractResponse | None = None
        try:
            parsed_response = ExtractResponse.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError) as err:
            # One repair retry
            repair_messages = list(messages)
            repair_messages.append({"role": "assistant", "content": raw_content})
            repair_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your output failed validation ({err}). "
                        f"Return ONLY valid JSON strictly adhering to schema: {json.dumps(schema)}"
                    ),
                }
            )
            retry_payload = {
                "model": model,
                "messages": repair_messages,
                "stream": False,
                "format": schema,
                "options": {"temperature": 0.1},
            }
            retry_res = client.post("/api/chat", json=retry_payload)
            retry_res.raise_for_status()
            retry_content = clean_json_text(retry_res.json().get("message", {}).get("content", ""))
            parsed_response = ExtractResponse.model_validate_json(retry_content)

        # Drop any events with confidence < 0.6 per spec
        # Enforce plain hyphens and backfill session_id / workspace if missing
        valid_events: list[ExtractedEvent] = []
        for ev in parsed_response.events:
            if ev.confidence < 0.6:
                continue

            if not ev.session_id:
                ev.session_id = session_id
            if not ev.workspace and workspace:
                ev.workspace = workspace

            ev.normalized = sanitize_hyphens(ev.normalized)
            ev.quote = sanitize_hyphens(ev.quote)
            if ev.reason:
                ev.reason = sanitize_hyphens(ev.reason)
            if ev.trigger:
                ev.trigger = sanitize_hyphens(ev.trigger)
            if ev.steps_hint:
                ev.steps_hint = [sanitize_hyphens(step) for step in ev.steps_hint]

            valid_events.append(ev)

        return valid_events


def extract_events_for_window(
    window: Window,
    model: str = "gemma2:2b",
    host: str = "http://127.0.0.1:11434",
    timeout: float = 60.0,
    include_fewshots: bool = True,
) -> list[ExtractedEvent]:
    """Extract events from a Window model instance."""
    return extract_events_from_text(
        text=window.text,
        session_id=window.session_id,
        workspace=window.workspace,
        model=model,
        host=host,
        timeout=timeout,
        include_fewshots=include_fewshots,
    )


def smoke_test_ollama(
    model: str = "gemma2:2b",
    host: str = "http://127.0.0.1:11434",
) -> tuple[bool, str]:
    """Verify Ollama service is reachable and can return valid structured events."""
    sample_text = (
        "[USER step=0]\n"
        "Please use poetry instead of pipenv in this repository because our lockfiles require poetry.\n"
        "[MODEL step=1]\n"
        "Understood, switching to poetry."
    )
    try:
        events = extract_events_from_text(
            text=sample_text,
            session_id="smoke-test-session",
            workspace="/tmp/sample-workspace",
            model=model,
            host=host,
            timeout=30.0,
            include_fewshots=False,
        )
        summary = (
            f"Successfully connected to Ollama at {host}. "
            f"Model {model} returned {len(events)} event(s):\n"
            + json.dumps([e.model_dump() for e in events], indent=2)
        )
        return True, summary
    except Exception as exc:
        return False, str(exc)
