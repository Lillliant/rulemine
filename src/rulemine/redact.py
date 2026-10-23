"""Redact secrets and sensitive tokens from transcript text prior to model inference."""

from __future__ import annotations

import re

# Standalone token patterns replaced directly with [REDACTED_SECRET]
TOKEN_PATTERNS = [
    # Private key blocks
    re.compile(
        r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"
    ),
    # OpenAI API keys (standard and project)
    re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"),
    # GitHub tokens (classic, fine-grained, oauth, refresh)
    re.compile(r"\bgh[pousr]_[a-zA-Z0-9_]{36,}\b"),
    re.compile(r"\bgithub_pat_[a-zA-Z0-9_]{22,}\b"),
    # Slack tokens
    re.compile(r"\bxox[baprs]-[0-9a-zA-Z-]{10,80}\b"),
    # AWS Access Key ID
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]

# Bearer token pattern: Bearer <token>
BEARER_PATTERN = re.compile(
    r"\bBearer\s+(?!\[REDACTED_SECRET\])([a-zA-Z0-9_\-\.]{20,})\b",
    re.IGNORECASE,
)

# .env style assignments: KEY=value or KEY="value"
ENV_PATTERN = re.compile(
    r"(?m)^(\s*[A-Z0-9_]*(?:API[_-]?KEY|SECRET|PASSWORD|PASSWD|AUTH(?:[_-]?TOKEN)?|PRIVATE[_-]?KEY|ACCESS[_-]?TOKEN|CLIENT[_-]?SECRET)[A-Z0-9_]*\s*=\s*)(?!\[REDACTED_SECRET\])(?:'[^'\r\n]+'|\"[^\"\r\n]+\"|\S+)"
)

# Generic key-value pattern: api_key: "value" or secret="value"
KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b((?:api[_-]?key|secret(?:[_-]?key)?|password|auth[_-]?token)\s*[:=]\s*['\"]?)(?!\[REDACTED_SECRET\])([a-zA-Z0-9_\-\.]{8,})(['\"]?)"
)


def redact_secrets(text: str) -> tuple[str, int]:
    """Redact known secret patterns from text, returning sanitized text and match count.

    Replaces discovered secrets with [REDACTED_SECRET].
    """
    if not text:
        return "", 0

    findings_count = 0
    redacted = text

    # 1. Private keys and standalone tokens
    for pat in TOKEN_PATTERNS:
        matches = pat.findall(redacted)
        if matches:
            findings_count += len(matches)
            redacted = pat.sub("[REDACTED_SECRET]", redacted)

    # 2. Bearer tokens
    bearer_matches = BEARER_PATTERN.findall(redacted)
    if bearer_matches:
        findings_count += len(bearer_matches)
        redacted = BEARER_PATTERN.sub("Bearer [REDACTED_SECRET]", redacted)

    # 3. .env assignments
    env_matches = ENV_PATTERN.findall(redacted)
    if env_matches:
        findings_count += len(env_matches)
        redacted = ENV_PATTERN.sub(r"\g<1>[REDACTED_SECRET]", redacted)

    # 4. Key-value pairs
    kv_matches = KEY_VALUE_PATTERN.findall(redacted)
    if kv_matches:
        findings_count += len(kv_matches)
        redacted = KEY_VALUE_PATTERN.sub(r"\g<1>[REDACTED_SECRET]\g<3>", redacted)

    return redacted, findings_count
