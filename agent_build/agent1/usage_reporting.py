from typing import Any, Dict


_DELTA_KEYS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "provider_calls",
    "estimated_calls",
    "hybrid_calls",
    "anomaly_calls",
)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def usage_delta(before_summary: Dict[str, Any], after_summary: Dict[str, Any]) -> Dict[str, int]:
    before = dict(before_summary or {})
    after = dict(after_summary or {})
    delta: Dict[str, int] = {}
    for key in _DELTA_KEYS:
        delta[key] = _as_int(after.get(key)) - _as_int(before.get(key))
    return delta


def format_turn_usage_line(delta: Dict[str, Any], last_usage: Dict[str, Any] | None) -> str:
    values = dict(delta or {})
    calls = _as_int(values.get("calls"))
    if calls <= 0:
        return ""

    provider_calls = _as_int(values.get("provider_calls"))
    estimated_calls = _as_int(values.get("estimated_calls"))
    hybrid_calls = _as_int(values.get("hybrid_calls"))
    anomaly_calls = _as_int(values.get("anomaly_calls"))
    source = str((last_usage or {}).get("source") or "none").strip().lower()
    return (
        f"[usage] turn calls={calls} in={_as_int(values.get('input_tokens'))} "
        f"out={_as_int(values.get('output_tokens'))} total={_as_int(values.get('total_tokens'))} "
        f"source={source} mix(p/e/h)={provider_calls}/{estimated_calls}/{hybrid_calls} anomalies={anomaly_calls}"
    )


def format_cumulative_usage_fragment(summary: Dict[str, Any]) -> str:
    totals = dict(summary or {})
    return (
        f"usage(total={_as_int(totals.get('total_tokens'))},"
        f"in={_as_int(totals.get('input_tokens'))},"
        f"out={_as_int(totals.get('output_tokens'))},"
        f"calls={_as_int(totals.get('calls'))},"
        f"p/e/h={_as_int(totals.get('provider_calls'))}/"
        f"{_as_int(totals.get('estimated_calls'))}/{_as_int(totals.get('hybrid_calls'))},"
        f"anom={_as_int(totals.get('anomaly_calls'))})"
    )
