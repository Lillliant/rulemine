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


def test_redact_slack_token():
    text = "Slack webhook token: xoxb-123456789012-1234567890123-abcdefghijklmnopqrstuvwx"
    sanitized, count = redact_secrets(text)
    assert "xoxb-" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert count == 1


def test_redact_bearer_token():
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    sanitized, count = redact_secrets(text)
    assert "eyJhbGci" not in sanitized
    assert "Bearer [REDACTED_SECRET]" in sanitized
    assert count == 1


def test_redact_aws_key():
    text = "AWS config: AKIAIOSFODNN7EXAMPLE credentials"
    sanitized, count = redact_secrets(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert count == 1


def test_redact_private_key():
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA04mH...fakekeycontent...\n"
        "-----END RSA PRIVATE KEY-----"
    )
    sanitized, count = redact_secrets(text)
    assert "BEGIN RSA PRIVATE KEY" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert count == 1


def test_redact_dotenv_assignments():
    text = (
        "OPENAI_API_KEY=sk-testvalue123456789012345\n"
        "DATABASE_PASSWORD=super_secret_password_here\n"
        "PORT=8000\n"
    )
    sanitized, count = redact_secrets(text)
    assert "sk-testvalue" not in sanitized
    assert "super_secret_password_here" not in sanitized
    assert "PORT=8000" in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert count >= 2
