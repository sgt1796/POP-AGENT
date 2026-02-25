from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


_DEFAULT_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bjina_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bpplx-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}\b", re.IGNORECASE),
]

_SENSITIVE_KEY_PARTS = (
    "key",
    "token",
    "secret",
    "password",
    "authorization",
    "api_key",
)

_USAGE_TOKEN_STAT_KEYS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "estimate_input_tokens",
    "estimate_output_tokens",
    "estimate_total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_token_count",
    "input_token_count",
    "output_token_count",
    "candidates_token_count",
    "prompt_eval_count",
    "eval_count",
}


def _compile_patterns(extra_patterns: Iterable[str] | None) -> List[re.Pattern[str]]:
    compiled = list(_DEFAULT_PATTERNS)
    for pattern in extra_patterns or []:
        text = str(pattern or "").strip()
        if not text:
            continue
        compiled.append(re.compile(text))
    return compiled


def _is_sensitive_key(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    if lowered in _USAGE_TOKEN_STAT_KEYS:
        return False
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _redact_string(value: str, patterns: List[re.Pattern[str]]) -> str:
    output = str(value)
    for pattern in patterns:
        output = pattern.sub("[REDACTED]", output)
    return output


def redact_data(value: Any, extra_patterns: Iterable[str] | None = None) -> Any:
    patterns = _compile_patterns(extra_patterns)
    return _redact_value(value, patterns)


def _redact_value(value: Any, patterns: List[re.Pattern[str]]) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, raw in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                out[key_text] = "[REDACTED]"
            else:
                out[key_text] = _redact_value(raw, patterns)
        return out

    if isinstance(value, list):
        return [_redact_value(item, patterns) for item in value]

    if isinstance(value, tuple):
        return [_redact_value(item, patterns) for item in value]

    if isinstance(value, str):
        return _redact_string(value, patterns)

    return value
