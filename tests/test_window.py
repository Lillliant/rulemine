"""Unit tests for turn windowing and labeled context chunking."""

from rulemine.models import Turn
from rulemine.window import window_turns


def test_window_turns_basic():
    turns = [
        Turn(
            step_index=0,
            source="USER_EXPLICIT",
            type="USER_INPUT",
            content="Please use npm instead of pnpm.",
        ),
        Turn(
            step_index=1,
            source="MODEL",
            type="PLANNER_RESPONSE",
            content="Understood. Switching to npm.",
        ),
        Turn(
            step_index=2,
            source="MODEL",
            type="GENERIC",
            content="npm install completed.",
        ),
    ]

    windows = window_turns(turns, session_id="test-session", workspace="/dev/repo")
    assert len(windows) == 1

    w = windows[0]
    assert w.session_id == "test-session"
    assert w.workspace == "/dev/repo"
    assert w.start_step == 0
    assert w.end_step == 2
    assert w.turn_count == 3
    assert "[USER step=0]" in w.text
    assert "Please use npm instead of pnpm." in w.text
    assert "[MODEL step=1]" in w.text


def test_window_turns_skip_without_user_turn():
    # Only model and system turns
    turns = [
        Turn(step_index=0, source="SYSTEM", type="SYSTEM_MESSAGE", content="System setup."),
        Turn(step_index=1, source="MODEL", type="PLANNER_RESPONSE", content="Working on it..."),
    ]
    windows = window_turns(turns, session_id="test-session")
    assert len(windows) == 0


def test_window_turns_redacts_secrets():
    turns = [
        Turn(
            step_index=0,
            source="USER_EXPLICIT",
            type="USER_INPUT",
            content="Here is the key: sk-abcdef12345678901234567890 please run it.",
        )
    ]
    windows = window_turns(turns, session_id="test-session")
    assert len(windows) == 1
    assert "sk-abcdef" not in windows[0].text
    assert "[REDACTED_SECRET]" in windows[0].text


def test_window_turns_splits_on_max_turns():
    # 25 turns, with user turns at step 0 and step 14
    turns = []
    for i in range(25):
        if i in (0, 14):
            turns.append(
                Turn(step_index=i, source="USER_EXPLICIT", type="USER_INPUT", content=f"User request {i}")
            )
        else:
            turns.append(
                Turn(step_index=i, source="MODEL", type="GENERIC", content=f"Model output {i}")
            )

    windows = window_turns(turns, session_id="test-session", max_turns=12)
    assert len(windows) >= 2
    assert all(w.turn_count <= 12 for w in windows)
