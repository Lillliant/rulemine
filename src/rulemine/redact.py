"""Redact secrets and sensitive tokens from transcript text prior to model inference."""

from __future__ import annotations

import re

# Standard credential patterns
PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),  # OpenAI / API keys
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub personal tokens
    re.compile(r"xox[baprs]-[0-9a-zA-Z]{10,48}"),  # Slack tokens
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),  # Bearer tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key ID
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?"),
]


def redact_secrets(text: str) -> tuple[str, int]:
    """Redact known secret patterns from text, returning sanitized text and match count."""
    findings_count = 0
    redacted = text

    for pat in PATTERNS:
        matches = pat.findall(redacted)
        findings_count += len(matches)
        redacted = pat.sub("[REDACTED_SECRET]", redacted)

    return redacted, findings_count
