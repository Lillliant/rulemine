"""Unit tests for secret redaction."""

from rulemine.redact import redact_secrets


def test_redact_secrets_removes_api_keys():
    text = "Here is my secret sk-abcdef12345678901234567890 and token ghp_123456789012345678901234567890123456"
    sanitized, count = redact_secrets(text)
    assert "sk-abcdef" not in sanitized
    assert "ghp_1234" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert count >= 2


def test_redact_clean_text():
    text = "Use npm, not pnpm in this repository."
    sanitized, count = redact_secrets(text)
    assert sanitized == text
    assert count == 0
