"""Usage tracking helpers for POP-agent.

This module keeps usage normalization and totals accumulation in one
place. It prefers POP's native implementation when available and falls
back to local best-effort logic otherwise.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

ANOMALY_THRESHOLD = 0.5

_CANONICAL_RECORD_KEYS = {
    "provider",
    "model",
    "source",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "provider_cost",
    "provider_cost_currency",
    "estimate_input_tokens",
    "estimate_output_tokens",
    "estimate_total_tokens",
    "estimate_method",
    "anomaly_flag",
    "anomaly_ratio",
    "anomaly_threshold",
    "latency_ms",
    "timestamp",
}

try:
    from POP.usage_tracking import (  # type: ignore
        build_usage_record as _pop_build_usage_record,
        accumulate_totals as _pop_accumulate_totals,
        init_usage_totals as _pop_init_usage_totals,
    )
except Exception:
    _pop_build_usage_record = None
    _pop_accumulate_totals = None
    _pop_init_usage_totals = None


def init_usage_totals() -> Dict[str, Any]:
    """Return a fresh totals dictionary."""
    if _pop_init_usage_totals is not None:
        try:
            totals = dict(_pop_init_usage_totals())
            return _merge_totals_defaults(totals)
        except Exception:
            pass
    return _merge_totals_defaults({})


def accumulate_totals(totals: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """Accumulate one usage record into `totals`."""
    target = _merge_totals_defaults(dict(totals or {}))
    payload = dict(record or {})
    if _pop_accumulate_totals is not None:
        try:
            merged = _pop_accumulate_totals(target, payload)
            return _merge_totals_defaults(dict(merged or {}))
        except Exception:
            pass

    target["calls"] = int(target.get("calls", 0)) + 1
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = payload.get(key)
        if isinstance(value, int):
            target[key] = int(target.get(key, 0)) + value
    source = str(payload.get("source", "")).strip().lower()
    if source == "provider":
        target["provider_calls"] = int(target.get("provider_calls", 0)) + 1
    elif source == "estimate":
        target["estimated_calls"] = int(target.get("estimated_calls", 0)) + 1
    elif source == "hybrid":
        target["hybrid_calls"] = int(target.get("hybrid_calls", 0)) + 1
    if bool(payload.get("anomaly_flag")):
        target["anomaly_calls"] = int(target.get("anomaly_calls", 0)) + 1
    provider_cost = payload.get("provider_cost")
    if isinstance(provider_cost, (int, float)):
        target["provider_cost_total"] = float(target.get("provider_cost_total", 0.0)) + float(provider_cost)
    return target


def ensure_usage_record(
    *,
    usage: Optional[Dict[str, Any]] = None,
    response: Any = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    reply_text: str = "",
    provider: str = "",
    model: str = "",
    tools: Any = None,
    response_format: Any = None,
    latency_ms: int = 0,
    timestamp: Optional[float] = None,
    anomaly_threshold: float = ANOMALY_THRESHOLD,
) -> Dict[str, Any]:
    """Return one normalized usage record.

    Preference order:
    1. Existing normalized record if already present in `usage`.
    2. POP's `build_usage_record` (provider-first).
    3. Local fallback estimate.
    """
    if timestamp is None:
        timestamp = time.time()

    current = dict(usage or {})
    if _is_normalized_record(current):
        record = _coerce_record_fields(current)
        if not record.get("provider"):
            record["provider"] = str(provider or "")
        if not record.get("model"):
            record["model"] = str(model or "")
        record["latency_ms"] = _safe_int(record.get("latency_ms")) or int(latency_ms)
        record["timestamp"] = _safe_float(record.get("timestamp")) or float(timestamp)
        return _fill_record_defaults(record, anomaly_threshold=anomaly_threshold)

    pop_response = response
    if pop_response is None and current:
        pop_response = {"usage": current}
    if _pop_build_usage_record is not None:
        try:
            built = _pop_build_usage_record(
                response=pop_response,
                messages=list(messages or []),
                reply_text=reply_text or "",
                provider=provider,
                model=model,
                tools=tools,
                response_format=response_format,
                latency_ms=int(latency_ms),
                timestamp=float(timestamp),
                anomaly_threshold=float(anomaly_threshold),
            )
            return _fill_record_defaults(_coerce_record_fields(dict(built or {})), anomaly_threshold=anomaly_threshold)
        except Exception:
            pass

    return _build_fallback_record(
        raw_usage=current,
        messages=list(messages or []),
        reply_text=reply_text or "",
        provider=provider,
        model=model,
        tools=tools,
        response_format=response_format,
        latency_ms=int(latency_ms),
        timestamp=float(timestamp),
        anomaly_threshold=float(anomaly_threshold),
    )


def _merge_totals_defaults(value: Dict[str, Any]) -> Dict[str, Any]:
    totals = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "provider_calls": 0,
        "estimated_calls": 0,
        "hybrid_calls": 0,
        "anomaly_calls": 0,
        "provider_cost_total": 0.0,
    }
    for key in totals:
        if key in value:
            totals[key] = value[key]
    return totals


def _is_normalized_record(record: Dict[str, Any]) -> bool:
    if not record:
        return False
    if "source" not in record:
        return False
    if not any(key in record for key in ("input_tokens", "total_tokens", "estimate_total_tokens")):
        return False
    return True


def _coerce_record_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(record or {})
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "estimate_input_tokens",
        "estimate_output_tokens",
        "estimate_total_tokens",
        "latency_ms",
    ):
        output[key] = _safe_int(output.get(key))
    output["provider_cost"] = _safe_float(output.get("provider_cost"))
    output["anomaly_ratio"] = _safe_float(output.get("anomaly_ratio"))
    output["anomaly_threshold"] = _safe_float(output.get("anomaly_threshold"))
    output["timestamp"] = _safe_float(output.get("timestamp"))
    output["anomaly_flag"] = bool(output.get("anomaly_flag"))
    output["source"] = str(output.get("source") or "").strip() or "none"
    output["provider"] = str(output.get("provider") or "")
    output["model"] = str(output.get("model") or "")
    if output.get("provider_cost_currency") is None:
        output["provider_cost_currency"] = None
    return output


def _fill_record_defaults(record: Dict[str, Any], *, anomaly_threshold: float) -> Dict[str, Any]:
    output = dict(record or {})
    for key in _CANONICAL_RECORD_KEYS:
        output.setdefault(key, None)
    if output.get("source") is None:
        output["source"] = "none"
    if output.get("anomaly_flag") is None:
        output["anomaly_flag"] = False
    if output.get("anomaly_threshold") is None:
        output["anomaly_threshold"] = float(anomaly_threshold)
    if output.get("latency_ms") is None:
        output["latency_ms"] = 0
    if output.get("timestamp") is None:
        output["timestamp"] = time.time()
    return output


def _safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return _jsonable(dumped)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def _count_tokens(text: str, model: str) -> Tuple[int, str]:
    payload = text or ""
    try:
        import tiktoken  # type: ignore

        try:
            encoder = tiktoken.encoding_for_model(model)
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(payload)), "tiktoken"
    except Exception:
        if not payload:
            return 0, "utf8_char4"
        return int(math.ceil(len(payload.encode("utf-8")) / 4.0)), "utf8_char4"


def _extract_provider_usage(raw_usage: Dict[str, Any]) -> Dict[str, Any]:
    usage = dict(raw_usage or {})
    prompt_details = dict(_first(usage, "prompt_tokens_details", "input_tokens_details") or {})
    completion_details = dict(_first(usage, "completion_tokens_details", "output_tokens_details") or {})

    input_tokens = _safe_int(
        _first(usage, "prompt_tokens", "input_tokens", "prompt_eval_count", "input_token_count")
    )
    output_tokens = _safe_int(
        _first(usage, "completion_tokens", "output_tokens", "eval_count", "output_token_count", "candidates_token_count")
    )
    total_tokens = _safe_int(_first(usage, "total_tokens", "total_token_count"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    cached_tokens = _safe_int(_first(usage, "cached_tokens", "cache_hit_tokens"))
    if cached_tokens is None:
        cached_tokens = _safe_int(_first(prompt_details, "cached_tokens", "cache_hit_tokens"))
    reasoning_tokens = _safe_int(_first(usage, "reasoning_tokens"))
    if reasoning_tokens is None:
        reasoning_tokens = _safe_int(_first(completion_details, "reasoning_tokens", "reasoning_output_tokens"))
    provider_cost = _safe_float(_first(usage, "cost", "total_cost", "usd_cost"))
    provider_cost_currency = _first(usage, "cost_currency", "currency")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "provider_cost": provider_cost,
        "provider_cost_currency": provider_cost_currency,
    }


def _build_fallback_record(
    *,
    raw_usage: Dict[str, Any],
    messages: List[Dict[str, Any]],
    reply_text: str,
    provider: str,
    model: str,
    tools: Any,
    response_format: Any,
    latency_ms: int,
    timestamp: float,
    anomaly_threshold: float,
) -> Dict[str, Any]:
    request_payload = {
        "messages": _jsonable(messages),
        "tools": _jsonable(tools),
        "response_format": _jsonable(response_format),
    }
    request_text = json.dumps(request_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    estimate_input_tokens, estimate_method = _count_tokens(request_text, model)
    estimate_output_tokens, _ = _count_tokens(reply_text or "", model)
    estimate_total_tokens = estimate_input_tokens + estimate_output_tokens

    provider_usage = _extract_provider_usage(raw_usage)
    p_input = provider_usage.get("input_tokens")
    p_output = provider_usage.get("output_tokens")
    p_total = provider_usage.get("total_tokens")
    e_input = estimate_input_tokens
    e_output = estimate_output_tokens
    e_total = estimate_total_tokens

    provider_has_any = any(
        value is not None
        for value in (
            p_input,
            p_output,
            p_total,
            provider_usage.get("cached_tokens"),
            provider_usage.get("reasoning_tokens"),
            provider_usage.get("provider_cost"),
        )
    )
    estimate_has_any = True
    provider_complete = p_input is not None and p_output is not None and p_total is not None
    if provider_complete:
        source = "provider"
    elif provider_has_any and estimate_has_any:
        source = "hybrid"
    elif provider_has_any:
        source = "provider"
    elif estimate_has_any:
        source = "estimate"
    else:
        source = "none"

    input_tokens = p_input if p_input is not None else (e_input if source in {"hybrid", "estimate"} else None)
    output_tokens = p_output if p_output is not None else (e_output if source in {"hybrid", "estimate"} else None)

    total_tokens = p_total
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if total_tokens is None and source in {"hybrid", "estimate"}:
        total_tokens = e_total

    provider_compare_total = p_total
    if provider_compare_total is None and p_input is not None and p_output is not None:
        provider_compare_total = p_input + p_output

    anomaly_ratio: Optional[float] = None
    anomaly_flag = False
    if provider_compare_total is not None:
        denominator = max(provider_compare_total, e_total, 1)
        anomaly_ratio = abs(provider_compare_total - e_total) / float(denominator)
        anomaly_flag = anomaly_ratio > anomaly_threshold

    return _fill_record_defaults(
        _coerce_record_fields(
            {
                "provider": str(provider or ""),
                "model": str(model or ""),
                "source": source,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": provider_usage.get("cached_tokens"),
                "reasoning_tokens": provider_usage.get("reasoning_tokens"),
                "provider_cost": provider_usage.get("provider_cost"),
                "provider_cost_currency": provider_usage.get("provider_cost_currency"),
                "estimate_input_tokens": e_input,
                "estimate_output_tokens": e_output,
                "estimate_total_tokens": e_total,
                "estimate_method": estimate_method,
                "anomaly_flag": anomaly_flag,
                "anomaly_ratio": anomaly_ratio,
                "anomaly_threshold": anomaly_threshold,
                "latency_ms": int(latency_ms),
                "timestamp": float(timestamp),
            }
        ),
        anomaly_threshold=anomaly_threshold,
    )


__all__ = ["ANOMALY_THRESHOLD", "init_usage_totals", "accumulate_totals", "ensure_usage_record"]
