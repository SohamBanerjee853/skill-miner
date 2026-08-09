"""Secret/PII redaction.

Applied at PARSE time, so the on-disk cache is already clean and no other
module can accidentally forward raw transcript content to an API. llm.py
additionally runs redact() over every outbound payload as a second fence.
"""

from __future__ import annotations

import re

# Order matters: specific token formats first, then generic assignments,
# then broad patterns (emails, URLs with credentials).
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ANTHROPIC_KEY", re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}")),
    ("OPENAI_KEY", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|github_pat_[A-Za-z0-9_]{20,}")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("DATABRICKS_TOKEN", re.compile(r"\bdapi[0-9a-f]{30,}\b")),
    ("GOOGLE_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("PRIVATE_KEY_BLOCK", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S)),
    ("URL_CREDENTIALS", re.compile(r"(?<=://)[^/\s:@]{1,64}:[^/\s@]{1,256}@")),
    # Strong keys: always redact the value (false positives here are cheap,
    # leaks are not).
    ("KEY_ASSIGNMENT", re.compile(
        r"(?i)\b((?:api[_\-]?key|apikey|access[_\-]?key|secret[_\-]?key|secret|token|passwd|password|pwd|bearer)"
        r"[\"']?\s*[:=]\s*[\"']?)[^\s\"',;]{6,}")),
    # Weak keys ("auth:", "credentials:") often introduce prose ("Auth:
    # databricks-sdk env-var flow"); only redact secret-looking values —
    # containing a digit or 16+ chars.
    ("KEY_ASSIGNMENT", re.compile(
        r"(?i)\b((?:auth|credential[s]?)[\"']?\s*[:=]\s*[\"']?)"
        r"(?=[^\s\"',;]*\d|[^\s\"',;]{16,})[^\s\"',;]{6,}")),
    ("BEARER_HEADER", re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9_\-.=+/]{16,}")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("HEX_SECRET", re.compile(r"\b[0-9a-fA-F]{40,}\b")),
]

# Assignment-style patterns keep the left-hand side so redacted text stays readable.
_KEEP_PREFIX = {"KEY_ASSIGNMENT", "BEARER_HEADER"}


def redact(text: str) -> str:
    """Replace anything secret-shaped with a [REDACTED:<kind>] placeholder."""
    if not text:
        return text
    for kind, pattern in _PATTERNS:
        if kind in _KEEP_PREFIX:
            text = pattern.sub(lambda m, k=kind: m.group(1) + f"[REDACTED:{k}]", text)
        else:
            text = pattern.sub(f"[REDACTED:{kind}]", text)
    return text


def redact_obj(obj):
    """Recursively redact every string in a JSON-like structure."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    return obj
