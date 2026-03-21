from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from eval.agent_step_summary import (
    PromptFunction,
    collect_distinct_tool_names,
    format_summary_model,
    resolve_step_summary_client,
)
from eval.core.artifacts import JsonArtifactWriter


_FACTOR_NAMES = [
    "correct",
    "runtime_error",
    "total_tokens",
    "latency_ms",
    "call_count",
    "distinct_tool_count",
]
_PLACEHOLDER_PREDICTIONS = {
    "",
    "(no assistant text returned)",
    "(assistant text unavailable)",
}
_TIMEOUT_RE = re.compile(r"(?:^|[\s_-])tim(?:e)?d?\s*out|timeout", re.IGNORECASE)
_FORMAT_LABEL_RE = re.compile(r"^\s*(?:final\s+answer|answer)\s*[:\-]", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CAUSE_SUMMARY_PROMPT = (
    "You summarize why an evaluation sample failed.\n"
    'Return strict JSON only in the form {"summary":"..."}.\n'
    "- One or two sentences.\n"
    "- Maximum 48 words.\n"
    "- Mention the main attempted actions and the concrete blockers or errors.\n"
    "- Prefer concrete tool names, commands, files, URLs, or queries when present.\n"
    "- Use only observable evidence from the trace, prediction, and errors.\n"
    "- Do not speculate about hidden reasoning.\n"
)


def analyze_run_artifacts(
    summary: Dict[str, Any],
    samples: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    events_by_sample: Dict[str, List[Dict[str, Any]]],
    *,
    summarize_failure_causes: bool = False,
    summary_provider: Optional[str] = None,
    summary_model: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    updated_summary = dict(summary or {})
    updated_samples: List[Dict[str, Any]] = []
    prompt_functions = _PromptFunctionCache() if summarize_failure_causes else None

    for sample_record in samples:
        sample = dict(sample_record)
        result = sample.get("result") if isinstance(sample.get("result"), dict) else {}
        result_dict = dict(result or {})
        sample["result"] = result_dict
        events = list(events_by_sample.get(str(sample.get("sample_id") or ""), []))
        failure_analysis = _build_failure_analysis(
            sample,
            events,
            manifest=manifest,
            summarize_failure_causes=summarize_failure_causes,
            summary_provider=summary_provider,
            summary_model=summary_model,
            prompt_functions=prompt_functions,
        )
        if failure_analysis:
            result_dict["failure_analysis"] = failure_analysis
        else:
            result_dict.pop("failure_analysis", None)
        updated_samples.append(sample)

    metrics = updated_summary.get("metrics") if isinstance(updated_summary.get("metrics"), dict) else {}
    metrics_dict = dict(metrics or {})
    metrics_dict["analysis"] = build_run_analysis(updated_samples, events_by_sample)
    updated_summary["metrics"] = metrics_dict
    return updated_summary, updated_samples, metrics_dict["analysis"]


def persist_run_analysis(
    run_dir: str,
    *,
    summary: Optional[Dict[str, Any]] = None,
    samples: Optional[Sequence[Dict[str, Any]]] = None,
    manifest: Optional[Dict[str, Any]] = None,
    events_by_sample: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    summarize_failure_causes: bool = False,
    summary_provider: Optional[str] = None,
    summary_model: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    run_path = Path(str(run_dir or "")).resolve()
    summary_payload = dict(summary or _load_json_file(run_path / "summary.json"))
    samples_payload = list(samples or _load_jsonl_file(run_path / "samples.jsonl"))
    manifest_payload = dict(manifest or _load_json_file_if_exists(run_path / "manifest.json"))
    grouped_events = events_by_sample or _group_events_by_sample(_load_jsonl_file_if_exists(run_path / "events.jsonl"))

    updated_summary, updated_samples, analysis = analyze_run_artifacts(
        summary_payload,
        samples_payload,
        manifest_payload,
        grouped_events,
        summarize_failure_causes=summarize_failure_causes,
        summary_provider=summary_provider,
        summary_model=summary_model,
    )

    samples_path = run_path / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in updated_samples:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    JsonArtifactWriter(str(run_path)).write_summary(updated_summary)
    return updated_summary, updated_samples, analysis


def build_run_analysis(
    samples: Sequence[Dict[str, Any]],
    events_by_sample: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    failure_scope_counter: Dict[str, Counter[str]] = {
        "runtime_error": Counter(),
        "incorrect": Counter(),
    }

    for sample in samples:
        result = sample.get("result") if isinstance(sample.get("result"), dict) else {}
        score_result = result.get("score_result") if isinstance(result.get("score_result"), dict) else {}
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        usage_delta = usage.get("delta") if isinstance(usage.get("delta"), dict) else {}
        failure_analysis = result.get("failure_analysis") if isinstance(result.get("failure_analysis"), dict) else {}
        sample_id = str(sample.get("sample_id") or "")
        events = list(events_by_sample.get(sample_id, []))

        rows.append(
            {
                "correct": 1.0 if bool(score_result.get("correct")) else 0.0,
                "runtime_error": 1.0 if str(result.get("status") or "").strip() == "error" else 0.0,
                "total_tokens": float(_to_int(usage_delta.get("total_tokens"))),
                "latency_ms": float(_to_float(result.get("latency_ms"))),
                "call_count": float(_to_int(usage_delta.get("calls"))),
                "distinct_tool_count": float(_resolve_distinct_tool_count(result, events)),
            }
        )

        scope = str(failure_analysis.get("scope") or "").strip()
        cause = str(failure_analysis.get("primary_cause") or "").strip()
        if scope in failure_scope_counter and cause:
            failure_scope_counter[scope][cause] += 1

    return {
        "factor_names": list(_FACTOR_NAMES),
        "correlations": _build_correlations(rows),
        "cohorts": _build_cohorts(rows),
        "failure_causes": {
            "total_non_correct": sum(sum(counter.values()) for counter in failure_scope_counter.values()),
            "runtime_error": _counter_rows(failure_scope_counter["runtime_error"]),
            "incorrect": _counter_rows(failure_scope_counter["incorrect"]),
        },
    }


def has_persisted_run_analysis(summary: Dict[str, Any], samples: Sequence[Dict[str, Any]]) -> bool:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    analysis = metrics.get("analysis") if isinstance(metrics, dict) else {}
    if not isinstance(analysis, dict) or not analysis.get("factor_names") or not analysis.get("cohorts"):
        return False
    for sample in samples:
        result = sample.get("result") if isinstance(sample.get("result"), dict) else {}
        score_result = result.get("score_result") if isinstance(result.get("score_result"), dict) else {}
        if bool(score_result.get("correct")):
            continue
        failure_analysis = result.get("failure_analysis")
        if not isinstance(failure_analysis, dict) or not failure_analysis.get("primary_cause"):
            return False
    return True


class _PromptFunctionCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def get(self, provider: str):
        key = str(provider or "").strip().lower() or "gemini"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if PromptFunction is None:
            raise RuntimeError("POP PromptFunction is unavailable.")
        prompt_fn = PromptFunction(
            sys_prompt=_CAUSE_SUMMARY_PROMPT,
            prompt="",
            client=key,
        )
        self._cache[key] = prompt_fn
        return prompt_fn


def _build_failure_analysis(
    sample: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    *,
    manifest: Dict[str, Any],
    summarize_failure_causes: bool,
    summary_provider: Optional[str],
    summary_model: Optional[str],
    prompt_functions: Optional[_PromptFunctionCache],
) -> Dict[str, Any]:
    result = sample.get("result") if isinstance(sample.get("result"), dict) else {}
    score_result = result.get("score_result") if isinstance(result.get("score_result"), dict) else {}
    if bool(score_result.get("correct")):
        return {}

    prediction = str(result.get("prediction") or "").strip()
    status = str(result.get("status") or "").strip()
    runtime_error = str(result.get("error") or "").strip()
    blocked_tools = _collect_blocked_tools(events)
    tool_errors = _collect_tool_errors(events)
    assistant_errors = _collect_assistant_errors(events)
    evidence: List[str] = []

    if runtime_error:
        _append_evidence(evidence, f"result.error={runtime_error}")
    for tool_name, reason in blocked_tools[:3]:
        _append_evidence(evidence, f"blocked {tool_name}: {reason}")
    for tool_name, error_text in tool_errors[:3]:
        _append_evidence(evidence, f"tool error {tool_name}: {error_text}")
    for stop_reason, error_text in assistant_errors[:3]:
        message = error_text or stop_reason
        if message:
            _append_evidence(evidence, f"assistant {stop_reason or 'error'}: {message}")
    if _is_placeholder_prediction(prediction):
        _append_evidence(evidence, f"prediction={prediction or '(empty)'}")

    if status == "error":
        analysis = _classify_runtime_failure(prediction, runtime_error, blocked_tools, tool_errors, assistant_errors, evidence)
    else:
        analysis = _classify_incorrect_failure(sample, prediction, blocked_tools, evidence)

    existing = _normalize_failure_analysis(result.get("failure_analysis"))
    _merge_existing_failure_enrichment(analysis, existing)

    if summarize_failure_causes:
        if not analysis.get("ai_summary") and not analysis.get("summary_error"):
            _enrich_failure_analysis(
                analysis,
                sample,
                events,
                manifest=manifest,
                summary_provider=summary_provider,
                summary_model=summary_model,
                prompt_functions=prompt_functions,
            )

    return analysis


def _classify_runtime_failure(
    prediction: str,
    runtime_error: str,
    blocked_tools: Sequence[Tuple[str, str]],
    tool_errors: Sequence[Tuple[str, str]],
    assistant_errors: Sequence[Tuple[str, str]],
    evidence: List[str],
) -> Dict[str, Any]:
    if runtime_error and _looks_like_timeout(runtime_error):
        return _failure_payload("runtime_error", "runner_timeout", "high", evidence)
    assistant_timeout = next(
        (
            error_text
            for _stop_reason, error_text in assistant_errors
            if _looks_like_timeout(error_text)
        ),
        "",
    )
    if assistant_timeout:
        return _failure_payload("runtime_error", "llm_timeout", "high", evidence)
    if blocked_tools:
        return _failure_payload("runtime_error", "tool_blocked", "high", evidence)
    if tool_errors:
        return _failure_payload("runtime_error", "tool_error", "high", evidence)
    if _is_placeholder_prediction(prediction):
        return _failure_payload("runtime_error", "no_final_answer_after_error", "medium", evidence)
    return _failure_payload("runtime_error", "unknown_runtime_error", "low", evidence)


def _classify_incorrect_failure(
    sample: Dict[str, Any],
    prediction: str,
    blocked_tools: Sequence[Tuple[str, str]],
    evidence: List[str],
) -> Dict[str, Any]:
    ground_truth = str(sample.get("ground_truth") or "").strip()
    if _is_placeholder_prediction(prediction):
        return _failure_payload("incorrect", "no_final_answer", "high", evidence)
    if _looks_like_output_format_violation(prediction, ground_truth):
        return _failure_payload("incorrect", "output_format_violation", "medium", evidence)
    if _looks_like_precision_miss(prediction, ground_truth):
        return _failure_payload("incorrect", "answer_precision_or_rounding", "medium", evidence)
    if blocked_tools:
        return _failure_payload("incorrect", "tool_blocked_then_incorrect", "medium", evidence)
    if prediction:
        return _failure_payload("incorrect", "wrong_answer", "low", evidence)
    return _failure_payload("incorrect", "unknown_incorrect", "low", evidence)


def _failure_payload(scope: str, primary_cause: str, confidence: str, evidence: Iterable[str]) -> Dict[str, Any]:
    return {
        "scope": scope,
        "primary_cause": primary_cause,
        "confidence": confidence,
        "evidence": [str(item).strip() for item in evidence if str(item).strip()][:5],
        "source": "deterministic",
        "generated_at": _utc_now_iso(),
    }


def _merge_existing_failure_enrichment(target: Dict[str, Any], existing: Dict[str, Any]) -> None:
    if not existing:
        return
    if str(existing.get("primary_cause") or "").strip() != str(target.get("primary_cause") or "").strip():
        return
    ai_summary = str(existing.get("ai_summary") or "").strip()
    if ai_summary:
        target["ai_summary"] = ai_summary
        summary_model = str(existing.get("summary_model") or "").strip()
        if summary_model:
            target["summary_model"] = summary_model
        generated_at = str(existing.get("generated_at") or "").strip()
        if generated_at:
            target["generated_at"] = generated_at
        target["source"] = "hybrid"
        return
    summary_error = str(existing.get("summary_error") or "").strip()
    if summary_error:
        target["summary_error"] = summary_error
        summary_model = str(existing.get("summary_model") or "").strip()
        if summary_model:
            target["summary_model"] = summary_model
        generated_at = str(existing.get("generated_at") or "").strip()
        if generated_at:
            target["generated_at"] = generated_at


def _enrich_failure_analysis(
    analysis: Dict[str, Any],
    sample: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    *,
    manifest: Dict[str, Any],
    summary_provider: Optional[str],
    summary_model: Optional[str],
    prompt_functions: Optional[_PromptFunctionCache],
) -> None:
    provider, model = resolve_step_summary_client(
        manifest=manifest,
        sample_record=sample,
        provider_override=summary_provider,
        model_override=summary_model,
    )
    analysis["summary_model"] = format_summary_model(provider, model)

    if PromptFunction is None:
        analysis["summary_error"] = "POP PromptFunction is unavailable."
        return

    trace_prompt = _build_failure_prompt(sample, analysis, events)
    if not trace_prompt:
        analysis["summary_error"] = "No usable failure trace was available for summarization."
        return

    cache = prompt_functions or _PromptFunctionCache()
    try:
        prompt_fn = cache.get(provider or "gemini")
        execute_kwargs: Dict[str, Any] = {"tools": [], "tool_choice": "auto"}
        if model:
            execute_kwargs["model"] = model
        raw_output = prompt_fn.execute(trace_prompt, **execute_kwargs)
        summary_text = _parse_failure_summary_output(raw_output)
    except Exception as exc:
        analysis["summary_error"] = _clean_error(exc)
        return

    if summary_text:
        analysis["ai_summary"] = summary_text
        analysis["source"] = "hybrid"
        analysis["generated_at"] = _utc_now_iso()


def _build_failure_prompt(sample: Dict[str, Any], analysis: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> str:
    lines = [
        f"Sample ID: {str(sample.get('sample_id') or '').strip()}",
        f"Primary cause: {str(analysis.get('primary_cause') or '').strip()}",
        f"Confidence: {str(analysis.get('confidence') or '').strip()}",
    ]
    result = sample.get("result") if isinstance(sample.get("result"), dict) else {}
    prediction = str(result.get("prediction") or "").strip()
    ground_truth = str(sample.get("ground_truth") or "").strip()
    if prediction:
        lines.extend(["Prediction:", _truncate(prediction, 500)])
    if ground_truth:
        lines.extend(["Ground truth:", _truncate(ground_truth, 500)])
    error_text = str(result.get("error") or "").strip()
    if error_text:
        lines.extend(["Runtime error:", _truncate(error_text, 320)])
    evidence = analysis.get("evidence") if isinstance(analysis.get("evidence"), list) else []
    if evidence:
        lines.append("Evidence:")
        for item in evidence[:5]:
            lines.append(f"- {item}")
    trace_lines = _failure_trace_lines(events)
    if trace_lines:
        lines.append("Trace:")
        lines.extend(trace_lines)
    return "\n".join(line for line in lines if line).strip()


def _failure_trace_lines(events: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    total_chars = 0
    for event_record in events:
        event = event_record.get("event") if isinstance(event_record.get("event"), dict) else {}
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "").strip()
        if not event_type:
            continue

        rendered = ""
        if event_type == "tool_execution_start":
            rendered = f"tool start {event.get('toolName')}: {_json_preview(event.get('args'), 120)}"
        elif event_type == "tool_execution_end":
            result = event.get("result") if isinstance(event.get("result"), dict) else {}
            details = result.get("details") if isinstance(result, dict) else {}
            if isinstance(details, dict):
                detail_bits = []
                if details.get("blocked"):
                    detail_bits.append(f"blocked={details.get('block_reason')}")
                if details.get("error"):
                    detail_bits.append(f"error={details.get('error')}")
                result_preview = _tool_result_preview(result)
                if result_preview and (detail_bits or bool(event.get("isError"))):
                    detail_bits.append(result_preview)
                rendered = f"tool end {event.get('toolName')}: " + (", ".join(detail_bits) or "ok")
        elif event_type in {"message_end", "turn_end"}:
            message = event.get("message") if isinstance(event.get("message"), dict) else {}
            if message.get("role") == "assistant":
                stop_reason = str(message.get("stopReason") or "").strip()
                error_text = str(message.get("errorMessage") or "").strip()
                text = _assistant_text_preview(message)
                detail_bits: List[str] = []
                if text:
                    detail_bits.append(f"text={text}")
                if stop_reason and stop_reason != "stop":
                    detail_bits.append(f"stop_reason={stop_reason}")
                if error_text:
                    detail_bits.append(f"error={error_text}")
                if detail_bits:
                    rendered = "assistant: " + " | ".join(detail_bits)
        if not rendered:
            continue
        if len(lines) >= 24:
            lines.append("[trace truncated]")
            break
        if total_chars + len(rendered) > 3600:
            lines.append("[trace truncated]")
            break
        lines.append(rendered)
        total_chars += len(rendered)
    return lines


def _parse_failure_summary_output(raw_output: Any) -> str:
    if isinstance(raw_output, dict):
        return str(raw_output.get("summary") or "").strip()
    text = str(raw_output or "").strip()
    if not text:
        return ""
    cleaned = _JSON_FENCE_RE.sub("", text).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return cleaned.splitlines()[0].strip()[:240]
    if isinstance(parsed, dict):
        return str(parsed.get("summary") or "").strip()
    return ""


def _build_correlations(rows: Sequence[Dict[str, float]]) -> List[Dict[str, Any]]:
    correlations: List[Dict[str, Any]] = []
    for index, x_name in enumerate(_FACTOR_NAMES):
        for y_name in _FACTOR_NAMES[index + 1 :]:
            values = [
                (float(row.get(x_name, 0.0)), float(row.get(y_name, 0.0)))
                for row in rows
                if x_name in row and y_name in row
            ]
            x_values = [item[0] for item in values]
            y_values = [item[1] for item in values]
            r_value = _pearson(x_values, y_values)
            correlations.append(
                {
                    "x": x_name,
                    "y": y_name,
                    "r": None if r_value is None else round(r_value, 4),
                    "n": len(values),
                }
            )
    return correlations


def _build_cohorts(rows: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, float]]] = {
        "correct": [],
        "incorrect": [],
        "runtime_error": [],
    }
    for row in rows:
        if row.get("runtime_error", 0.0) >= 1.0:
            grouped["runtime_error"].append(row)
        elif row.get("correct", 0.0) >= 1.0:
            grouped["correct"].append(row)
        else:
            grouped["incorrect"].append(row)

    output: Dict[str, Dict[str, Any]] = {}
    for key, items in grouped.items():
        output[key] = {
            "count": len(items),
            "avg_total_tokens": _stat(items, "total_tokens", statistics.mean),
            "median_total_tokens": _stat(items, "total_tokens", statistics.median),
            "avg_latency_ms": _stat(items, "latency_ms", statistics.mean),
            "median_latency_ms": _stat(items, "latency_ms", statistics.median),
            "avg_call_count": _stat(items, "call_count", statistics.mean),
            "median_call_count": _stat(items, "call_count", statistics.median),
            "avg_distinct_tool_count": _stat(items, "distinct_tool_count", statistics.mean),
            "median_distinct_tool_count": _stat(items, "distinct_tool_count", statistics.median),
        }
    return output


def _stat(rows: Sequence[Dict[str, float]], key: str, fn) -> float:
    values = [float(item.get(key, 0.0)) for item in rows]
    if not values:
        return 0.0
    try:
        return float(fn(values))
    except Exception:
        return 0.0


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_variance = sum((x - x_mean) ** 2 for x in x_values)
    y_variance = sum((y - y_mean) ** 2 for y in y_values)
    if x_variance <= 0 or y_variance <= 0:
        return None
    value = numerator / math.sqrt(x_variance * y_variance)
    if not math.isfinite(value):
        return None
    return value


def _counter_rows(counter: Counter[str]) -> List[Dict[str, Any]]:
    rows = [{"cause": cause, "count": count} for cause, count in counter.items()]
    rows.sort(key=lambda item: (-_to_int(item.get("count")), str(item.get("cause") or "")))
    return rows


def _collect_blocked_tools(events: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    blocked: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for event_record in events:
        event = event_record.get("event") if isinstance(event_record.get("event"), dict) else {}
        if str(event.get("type") or "").strip() != "tool_execution_end":
            continue
        details = (event.get("result") or {}).get("details") if isinstance(event.get("result"), dict) else {}
        if not isinstance(details, dict) or not details.get("blocked"):
            continue
        item = (str(event.get("toolName") or "unknown").strip() or "unknown", str(details.get("block_reason") or "blocked").strip() or "blocked")
        if item in seen:
            continue
        seen.add(item)
        blocked.append(item)
    return blocked


def _collect_tool_errors(events: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    tool_errors: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for event_record in events:
        event = event_record.get("event") if isinstance(event_record.get("event"), dict) else {}
        if str(event.get("type") or "").strip() != "tool_execution_end":
            continue
        details = (event.get("result") or {}).get("details") if isinstance(event.get("result"), dict) else {}
        if not isinstance(details, dict):
            continue
        error_text = str(details.get("error") or "").strip()
        if not error_text:
            continue
        item = (str(event.get("toolName") or "unknown").strip() or "unknown", error_text)
        if item in seen:
            continue
        seen.add(item)
        tool_errors.append(item)
    return tool_errors


def _collect_assistant_errors(events: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    errors: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for event_record in events:
        event = event_record.get("event") if isinstance(event_record.get("event"), dict) else {}
        if str(event.get("type") or "").strip() not in {"message_end", "turn_end"}:
            continue
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        if str(message.get("role") or "").strip() != "assistant":
            continue
        stop_reason = str(message.get("stopReason") or "").strip()
        error_text = str(message.get("errorMessage") or "").strip()
        if stop_reason == "stop" and not error_text:
            continue
        if not stop_reason and not error_text:
            continue
        item = (stop_reason, error_text)
        if item in seen:
            continue
        seen.add(item)
        errors.append(item)
    return errors


def _resolve_distinct_tool_count(result: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> int:
    agent_summary = result.get("agent_execution_summary") if isinstance(result.get("agent_execution_summary"), dict) else {}
    tool_names = agent_summary.get("tool_names") if isinstance(agent_summary.get("tool_names"), list) else []
    normalized = [str(item).strip() for item in tool_names if str(item).strip()]
    if normalized:
        return len(dict.fromkeys(normalized))
    return len(collect_distinct_tool_names(events))


def _looks_like_timeout(value: str) -> bool:
    return bool(_TIMEOUT_RE.search(str(value or "").strip()))


def _is_placeholder_prediction(prediction: str) -> bool:
    return str(prediction or "").strip().lower() in {item.lower() for item in _PLACEHOLDER_PREDICTIONS}


def _looks_like_output_format_violation(prediction: str, ground_truth: str) -> bool:
    text = str(prediction or "").strip()
    truth = str(ground_truth or "").strip()
    if not text:
        return False
    if "\n" in text:
        return True
    if _FORMAT_LABEL_RE.search(text):
        return True
    if truth and text != truth and truth in text:
        return True
    return False


def _looks_like_precision_miss(prediction: str, ground_truth: str) -> bool:
    pred_time = _parse_time_value(prediction)
    truth_time = _parse_time_value(ground_truth)
    if pred_time is not None and truth_time is not None:
        return abs(pred_time - truth_time) <= 0.01

    pred_number = _parse_number(prediction)
    truth_number = _parse_number(ground_truth)
    if pred_number is None or truth_number is None:
        return False
    delta = abs(pred_number - truth_number)
    tolerance = max(0.01, abs(truth_number) * 0.001)
    return delta <= tolerance


def _parse_time_value(value: str) -> Optional[float]:
    text = str(value or "").strip()
    if not text or " " in text:
        return None
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        return None
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    except Exception:
        return None


def _parse_number(value: str) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    cleaned = text.replace(",", "").replace("$", "").replace("%", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _normalize_failure_analysis(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    evidence = [str(item).strip() for item in value.get("evidence", []) if str(item).strip()] if isinstance(value.get("evidence"), list) else []
    output = {
        "scope": str(value.get("scope") or "").strip(),
        "primary_cause": str(value.get("primary_cause") or "").strip(),
        "confidence": str(value.get("confidence") or "").strip(),
        "evidence": evidence[:5],
        "source": str(value.get("source") or "").strip(),
        "generated_at": str(value.get("generated_at") or "").strip(),
    }
    ai_summary = str(value.get("ai_summary") or "").strip()
    if ai_summary:
        output["ai_summary"] = ai_summary
    summary_model = str(value.get("summary_model") or "").strip()
    if summary_model:
        output["summary_model"] = summary_model
    summary_error = str(value.get("summary_error") or "").strip()
    if summary_error:
        output["summary_error"] = summary_error
    return output


def _assistant_text_preview(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    chunks: List[str] = []
    for item in content:
        if not isinstance(item, dict) or str(item.get("type") or "") != "text":
            continue
        text = str(item.get("text") or "").strip()
        if text:
            chunks.append(text)
    return _truncate(" ".join(chunks), 160)


def _tool_result_preview(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    chunks: List[str] = []
    for item in content:
        if not isinstance(item, dict) or str(item.get("type") or "") != "text":
            continue
        text = str(item.get("text") or "").strip()
        if text:
            chunks.append(text)
    return _truncate(" ".join(chunks), 180)


def _json_preview(value: Any, limit: int) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except Exception:
        rendered = str(value)
    return _truncate(rendered, limit)


def _append_evidence(items: List[str], value: str) -> None:
    text = str(value or "").strip()
    if not text or text in items:
        return
    items.append(_truncate(text, 220))


def _truncate(text: str, limit: int) -> str:
    raw = str(text or "")
    return raw if len(raw) <= limit else raw[: limit - 3].rstrip() + "..."


def _clean_error(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


def _group_events_by_sample(events: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        sample_id = str(event.get("sample_id") or "").strip()
        if sample_id:
            grouped[sample_id].append(event)
    for sample_events in grouped.values():
        sample_events.sort(key=lambda item: _to_int(item.get("event_index")))
    return dict(grouped)


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_json_file_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _load_json_file(path)
    except Exception:
        return {}


def _load_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = str(line or "").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _load_jsonl_file_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return _load_jsonl_file(path)
    except Exception:
        return []


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _to_float(value: Any) -> float:
    try:
        value_float = float(value)
    except Exception:
        return 0.0
    return value_float if math.isfinite(value_float) else 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
