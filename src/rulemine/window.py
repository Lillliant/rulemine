"""Turn windowing into chunks suitable for LLM extraction context."""

from __future__ import annotations

from rulemine.models import Turn, Window
from rulemine.redact import redact_secrets


def _turn_label(turn: Turn) -> str:
    """Determine role label and header string for a turn."""
    source = (turn.source or "").upper()
    type_ = (turn.type or "").upper()

    if source == "USER_EXPLICIT" or type_ == "USER_INPUT":
        role = "USER"
    elif source == "MODEL" or type_ in ("PLANNER_RESPONSE", "GENERIC"):
        role = "MODEL"
    elif source == "SYSTEM" or type_ == "SYSTEM_MESSAGE":
        role = "SYSTEM"
    else:
        role = source or "TURN"

    if turn.step_index is not None:
        return f"[{role} step={turn.step_index}]"
    return f"[{role}]"


def _format_turn(turn: Turn) -> str:
    """Format a turn into labeled text."""
    header = _turn_label(turn)
    content = (turn.content or "").strip()
    if content:
        return f"{header}\n{content}"
    return header


def _is_user_text_turn(turn: Turn) -> bool:
    """Check if turn is a user input with non-empty text."""
    source = (turn.source or "").upper()
    type_ = (turn.type or "").upper()
    is_user = source == "USER_EXPLICIT" or type_ == "USER_INPUT"
    return is_user and bool(turn.content and turn.content.strip())


def window_turns(
    turns: list[Turn],
    session_id: str,
    workspace: str | None = None,
    max_turns: int = 12,
    max_chars: int = 3500,
) -> list[Window]:
    """Chunk conversation turns into structured context windows.

    Target 8-12 turns or max ~3500 characters.
    Prefer ending windows on user turns when possible.
    Only includes windows containing at least one user text turn.
    Applies secret redaction to all window text before returning.
    """
    if not turns:
        return []

    min_turns = min(8, max_turns)
    windows: list[Window] = []

    current_turns: list[Turn] = []
    current_formatted: list[str] = []
    current_char_count = 0

    def _flush_window() -> None:
        nonlocal current_turns, current_formatted, current_char_count
        if not current_turns:
            return

        # Must contain at least one user text turn
        if any(_is_user_text_turn(t) for t in current_turns):
            full_text = "\n\n".join(current_formatted)
            redacted_text, _ = redact_secrets(full_text)
            start_step = current_turns[0].step_index
            end_step = current_turns[-1].step_index
            windows.append(
                Window(
                    session_id=session_id,
                    workspace=workspace,
                    start_step=start_step,
                    end_step=end_step,
                    text=redacted_text,
                    turn_count=len(current_turns),
                )
            )

        current_turns = []
        current_formatted = []
        current_char_count = 0

    for turn in turns:
        formatted = _format_turn(turn)
        added_len = len(formatted) + (2 if current_formatted else 0)

        # Check if adding this turn exceeds max_chars or max_turns
        if current_turns and (
            len(current_turns) >= max_turns
            or (current_char_count + added_len > max_chars)
        ):
            _flush_window()

        current_turns.append(turn)
        current_formatted.append(formatted)
        current_char_count += added_len

        # Prefer ending window on user turn once minimum turns reached
        if len(current_turns) >= min_turns and _is_user_text_turn(turn):
            _flush_window()

    # Flush any remaining turns
    _flush_window()

    return windows
