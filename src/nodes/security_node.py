"""AgentCore Platform v1.0"""

# S-2/S-3 security gate logic for CMN-C1-113.
# PII scan (S-2) and credential/injection scan (S-3) extracted from src/agent.py.
# Used by the graph security hooks and can be invoked from tests directly.

import re
from typing import Optional

# Note: \b (word boundary) does not work adjacent to Unicode/CJK characters in Python.
# Patterns with \b match when digits are surrounded by spaces or ASCII word boundaries.
# For Japanese text, callers should ensure digit sequences are space-delimited when
# submitting to the S-2 gate, or upgrade patterns to use Unicode-aware boundaries.
_PII_PATTERNS = [
    re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{4}\b"),
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),  # My Number (個人番号) with optional separators
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    re.compile(r"0\d{1,4}[\-\s]?\d{1,4}[\-\s]?\d{4}"),
    re.compile(r"\b\d{3}-\d{4}\b"),
]

_CREDENTIAL_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"eyJ[a-zA-Z0-9._\-]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"(?i)Bearer\s+[a-zA-Z0-9._\-]{20,}"),
    re.compile(r"(?i)(password|passwd|secret|api_key|apikey)\s*[:=]\s*\S+"),
]

_INJECTION_PATTERNS = [
    re.compile(r"<\|"),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"IGNORE\s+PREVIOUS\s+INSTRUCTIONS", re.IGNORECASE),
    re.compile(r"</s>"),
    re.compile(r"###\s*Human:|###\s*Assistant:", re.IGNORECASE),
    re.compile(r"<\|im_start\|>|<\|im_end\|>"),
]


def check_pii(text: Optional[str], field_name: str) -> Optional[str]:
    """Return error message if PII detected, else None."""
    if not text:
        return None
    for pattern in _PII_PATTERNS:
        if pattern.search(text):
            return f"S-2 violation: PII pattern detected in '{field_name}'"
    return None


def check_output(answer_text: Optional[str]) -> Optional[str]:
    """Return error message if credential/injection detected, else None."""
    if not answer_text:
        return None
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(answer_text):
            return "S-3 violation: credential pattern detected in output — redacted"
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(answer_text):
            return "S-3 violation: prompt injection marker detected in output — redacted"
    return None
