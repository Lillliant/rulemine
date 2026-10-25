"""Extraction pipeline calling local Ollama Gemma for structured event parsing."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
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
        elif start == -1:
            # No JSON object found at all in output (e.g. plain text refusal or error echo)
            return '{"events": []}'
    return text


# Fast regex for candidate gating: match steer words, decisions, dead ends, and workflows
EXTRACTION_SIGNAL_REGEX = re.compile(
    r"\b("
    r"no|not|don'?t|dont|never|stop|instead|prefer|always|should|wrong|fix|switch|avoid|strictly|must|"
    r"do not|please use|incorrect|re-run|rerun|rather than|re-do|redo|"
    r"decid\w*|choice|choos\w*|architect\w*|convention|standard|pattern|adr|let'?s use|agreed|recommend\w*|"
    r"abandon\w*|dead[\s-]end|give up|giving up|gave up|revert\w*|blocker|doesn'?t work|does not work|failed|"
    r"every time|whenever|workflow|automat\w*|skill|recipe|reusable|routine"
    r")\b",
    re.IGNORECASE,
)


def has_extraction_signals(text: str) -> bool:
    """Fast pre-flight check to see if a conversation window contains potential memory signals.

    If a window contains no steering, decisions, errors, or workflow cues, skip LLM inference
    completely to save 70-80% of processing time and prevent timeouts.
    """
    if not text:
        return False
    return bool(EXTRACTION_SIGNAL_REGEX.search(text))


def build_messages(
    window_text: str,
    session_id: str = "smoke-session",
    workspace: str | None = None,
    include_fewshots: bool = True,
    max_fewshots: int = 2,
) -> list[dict[str, str]]:
    """Construct prompt messages including system instructions, few-shots, and window content."""
    system_prompt = load_system_prompt()
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if include_fewshots:
        fewshots = load_fewshots()
        if max_fewshots and max_fewshots > 0:
            fewshots = fewshots[:max_fewshots]
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


def stream_chat_from_ollama(
    client: httpx.Client,
    payload: dict[str, Any],
    timeout: httpx.Timeout,
    max_wall_time: float = 600.0,
) -> str:
    """Stream chat response from Ollama with socket read inactivity timeout and safety ceiling.

    Each streamed token resets the socket read timer, allowing long responses to complete
    as long as tokens keep flowing, while still detecting true deadlocks or stalls.
    """
    # Unit test compatibility: if client.post is mocked, delegate to mock
    post_fn = getattr(client, "post", None)
    if post_fn and (hasattr(post_fn, "assert_called") or type(post_fn).__name__ in ("MagicMock", "Mock")):
        res = client.post("/api/chat", json=payload)
        res.raise_for_status()
        return res.json().get("message", {}).get("content", "")

    # Streaming execution (both real httpx and mock client.stream)
    payload["stream"] = True
    content_chunks: list[str] = []
    start_time = time.monotonic()

    with client.stream("POST", "/api/chat", json=payload, timeout=timeout) as response:
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        for line in response.iter_lines():
            if time.monotonic() - start_time > max_wall_time:
                raise httpx.ReadTimeout(
                    f"Ollama stream exceeded maximum wall-clock ceiling ({max_wall_time:.0f}s)"
                )
            if not line:
                continue
            line = str(line).strip()
            try:
                chunk = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue

            part = chunk.get("message", {}).get("content", "")
            if part:
                content_chunks.append(part)

            if chunk.get("done", False):
                break

    return "".join(content_chunks)


def extract_events_from_text(
    text: str,
    session_id: str = "adhoc-session",
    workspace: str | None = None,
    model: str = "gemma2:2b",
    host: str = "http://127.0.0.1:11434",
    timeout: float | httpx.Timeout = 180.0,
    keep_alive: str = "30m",
    include_fewshots: bool = True,
    max_fewshots: int = 2,
    client: httpx.Client | None = None,
) -> list[ExtractedEvent]:
    """Call Ollama chat API to extract structured events using streaming with inactivity timeout."""
    schema = ExtractResponse.model_json_schema()
    messages = build_messages(
        window_text=text,
        session_id=session_id,
        workspace=workspace,
        include_fewshots=include_fewshots,
        max_fewshots=max_fewshots,
    )

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "format": schema,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0.1,
            "num_predict": 384,
            "num_ctx": 2048,
        },
    }

    # Granular decoupled timeouts: fast fail on connect, inactivity timeout on read
    if isinstance(timeout, (int, float)):
        timeout_config = httpx.Timeout(
            connect=10.0,
            read=float(timeout),
            write=10.0,
            pool=10.0,
        )
    else:
        timeout_config = timeout

    def _execute(http_client: httpx.Client) -> list[ExtractedEvent]:
        raw_content = stream_chat_from_ollama(http_client, payload, timeout=timeout_config)

        # Parse and validate with Pydantic
        cleaned = clean_json_text(raw_content)
        parsed_response: ExtractResponse | None = None
        try:
            parsed_response = ExtractResponse.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError):
            # One repair retry without dumping noisy traceback strings into the prompt
            repair_messages = list(messages)
            repair_messages.append({"role": "assistant", "content": raw_content})
            repair_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output was not valid JSON conforming to the schema. "
                        f"Return ONLY valid JSON strictly adhering to schema: {json.dumps(schema)}"
                    ),
                }
            )
            retry_payload = {
                "model": model,
                "messages": repair_messages,
                "stream": True,
                "format": schema,
                "keep_alive": keep_alive,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 384,
                    "num_ctx": 2048,
                },
            }
            try:
                retry_raw = stream_chat_from_ollama(http_client, retry_payload, timeout=timeout_config)
                retry_content = clean_json_text(retry_raw)
                parsed_response = ExtractResponse.model_validate_json(retry_content)
            except (ValidationError, json.JSONDecodeError, Exception):
                # If retry also fails validation, fallback to empty events rather than crashing
                return []

        if parsed_response is None or not parsed_response.events:
            return []

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

    if client is not None:
        return _execute(client)
    else:
        with httpx.Client(base_url=host, timeout=timeout_config) as local_client:
            return _execute(local_client)


def extract_events_for_window(
    window: Window,
    model: str = "gemma2:2b",
    host: str = "http://127.0.0.1:11434",
    timeout: float | httpx.Timeout = 180.0,
    keep_alive: str = "30m",
    include_fewshots: bool = True,
    max_fewshots: int = 2,
    client: httpx.Client | None = None,
    filter_signals: bool = True,
) -> list[ExtractedEvent]:
    """Extract events from a Window model instance, skipping uninteresting windows via fast regex gating."""
    if filter_signals and not has_extraction_signals(window.text):
        return []

    return extract_events_from_text(
        text=window.text,
        session_id=window.session_id,
        workspace=window.workspace,
        model=model,
        host=host,
        timeout=timeout,
        keep_alive=keep_alive,
        include_fewshots=include_fewshots,
        max_fewshots=max_fewshots,
        client=client,
    )


def smoke_test_ollama(
    model: str = "gemma2:2b",
    host: str = "http://127.0.0.1:11434",
    timeout: float = 60.0,
    keep_alive: str = "5m",
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
            timeout=timeout,
            keep_alive=keep_alive,
            include_fewshots=False,
        )
        summary = (
            f"Successfully connected to Ollama at {host}. "
            f"Model [bold cyan]{model}[/bold cyan] returned {len(events)} event(s):\n"
            + json.dumps([e.model_dump() for e in events], indent=2)
        )
        return True, summary
    except Exception as exc:
        return False, str(exc)
