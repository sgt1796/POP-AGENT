from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from POP import PromptFunction
except Exception:  # pragma: no cover - exercised via failure-path tests
    PromptFunction = None  # type: ignore[assignment]


_MAX_TRACE_LINES = 180
_MAX_TRACE_CHARS = 16_000
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_NUMBERED_STEP_RE = re.compile(r"^\s*\d+[\.\):-]\s*")
_SAMPLE_PROMPT_PREFIX = (
    "You summarize agent execution traces into concise high-level steps for evaluation reports.\n"
    "Return strict JSON only in the form {\"steps\": [\"...\"]}.\n"
    "Rules:\n"
    "- 1 to 8 steps.\n"
    "- Each step must be short and concrete.\n"
    "- Do not mention internal reasoning.\n"
    "- Prefer externally observable actions such as searching, opening pages, reading files, or answering.\n"
    "- Do not include markdown, numbering, commentary, or extra keys.\n"
)


def collect_distinct_tool_names(events: Sequence[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for event_record in events:
        event = event_record.get("event") if isinstance(event_record.get("event"), dict) else {}
        if not isinstance(event, dict):
            continue
        names: List[str] = []
        tool_name = str(event.get("toolName") or "").strip()
        if tool_name:
            names.append(tool_name)
        if not names and str(event.get("type") or "") == "turn_end":
            tool_results = event.get("toolResults")
            if isinstance(tool_results, list):
                for item in tool_results:
                    if not isinstance(item, dict):
                        continue
                    result_name = str(item.get("toolName") or "").strip()
                    if result_name:
                        names.append(result_name)
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return ordered


def format_summary_model(provider: Optional[str], model: Optional[str]) -> str:
    provider_text = str(provider or "").strip()
    model_text = str(model or "").strip()
    if provider_text and model_text:
        return f"{provider_text} / {model_text}"
    return provider_text or model_text


def humanize_tool_name(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", raw) if part]
    acronym_map = {
        "api": "API",
        "csv": "CSV",
        "html": "HTML",
        "json": "JSON",
        "pdf": "PDF",
        "url": "URL",
    }
    rendered = []
    for part in parts:
        lowered = part.lower()
        rendered.append(acronym_map.get(lowered, part.capitalize()))
    return " ".join(rendered)


def normalize_agent_execution_summary(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    steps = _normalize_steps(value.get("steps"))
    tool_names = _normalize_string_list(value.get("tool_names"))
    output: Dict[str, Any] = {
        "steps": steps,
        "step_count": len(steps) if steps else _safe_int(value.get("step_count")),
        "tool_names": tool_names,
        "tool_count": len(tool_names) if tool_names else _safe_int(value.get("tool_count")),
        "summary_model": str(value.get("summary_model") or "").strip(),
        "generated_at": _normalize_timestamp_value(value.get("generated_at")),
        "source": str(value.get("source") or "").strip(),
    }
    error_text = str(value.get("error") or "").strip()
    if error_text:
        output["error"] = error_text
    return output


def resolve_result_timing(sample_record: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    result = sample_record.get("result") if isinstance(sample_record.get("result"), dict) else {}
    started_at = _normalize_timestamp_value(result.get("started_at")) if isinstance(result, dict) else None
    ended_at = _normalize_timestamp_value(result.get("ended_at")) if isinstance(result, dict) else None

    timestamps = _collect_event_timestamps(events)
    if timestamps:
        if started_at is None:
            started_at = timestamps[0][1]
        if ended_at is None:
            ended_at = timestamps[-1][1]

    latency_ms = _safe_float(result.get("latency_ms")) if isinstance(result, dict) else 0.0
    if latency_ms > 0:
        if started_at and not ended_at:
            ended_at = _shift_iso_timestamp(started_at, latency_ms / 1000.0)
        elif ended_at and not started_at:
            started_at = _shift_iso_timestamp(ended_at, -(latency_ms / 1000.0))

    return started_at, ended_at


def persist_agent_execution_summaries(
    run_dir: str,
    *,
    samples: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    events_by_sample: Dict[str, List[Dict[str, Any]]],
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    prompt_functions = _PromptFunctionCache()
    updated_samples: List[Dict[str, Any]] = []

    for sample_record in samples:
        sample = dict(sample_record)
        result = sample.get("result") if isinstance(sample.get("result"), dict) else {}
        result_dict = dict(result or {})
        sample["result"] = result_dict
        sample_id = str(sample.get("sample_id") or "").strip()
        events = list(events_by_sample.get(sample_id, []))
        result_dict["agent_execution_summary"] = generate_agent_execution_summary(
            sample,
            events,
            manifest=manifest,
            provider_override=provider_override,
            model_override=model_override,
            prompt_functions=prompt_functions,
        )
        updated_samples.append(sample)

    samples_path = Path(run_dir) / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in updated_samples:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return updated_samples


def generate_agent_execution_summary(
    sample_record: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    *,
    manifest: Dict[str, Any],
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    prompt_functions: Optional["_PromptFunctionCache"] = None,
) -> Dict[str, Any]:
    tool_names = collect_distinct_tool_names(events)
    provider, model = resolve_step_summary_client(
        manifest=manifest,
        sample_record=sample_record,
        provider_override=provider_override,
        model_override=model_override,
    )
    summary: Dict[str, Any] = {
        "steps": [],
        "step_count": 0,
        "tool_names": tool_names,
        "tool_count": len(tool_names),
        "summary_model": format_summary_model(provider, model),
        "generated_at": _utc_now_iso(),
        "source": "promptfunction",
    }

    if not events:
        summary["error"] = "No event trace was captured for this sample."
        return summary

    if PromptFunction is None:
        summary["error"] = "POP PromptFunction is unavailable."
        return summary

    trace_text = _build_trace_prompt(sample_record, events)
    if not trace_text:
        summary["error"] = "No usable execution trace was available for summarization."
        return summary

    cache = prompt_functions or _PromptFunctionCache()
    try:
        prompt_fn = cache.get(provider or "gemini")
        execute_kwargs: Dict[str, Any] = {"tools": [], "tool_choice": "auto"}
        if model:
            execute_kwargs["model"] = model
        raw_output = prompt_fn.execute(trace_text, **execute_kwargs)
        steps = _parse_steps_output(raw_output)
    except Exception as exc:
        summary["error"] = _clean_error(exc)
        return summary

    summary["steps"] = steps
    summary["step_count"] = len(steps)
    return summary


def resolve_step_summary_client(
    *,
    manifest: Dict[str, Any],
    sample_record: Dict[str, Any],
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    default_provider, default_model = _manifest_model_hint(manifest)
    if not default_provider or not default_model:
        sample_provider, sample_model = _sample_usage_model_hint(sample_record)
        default_provider = default_provider or sample_provider
        default_model = default_model or sample_model

    provider_text = str(provider_override or "").strip()
    model_text = str(model_override or "").strip()
    if provider_text:
        return provider_text, model_text or None

    provider = default_provider or "gemini"
    if model_text:
        return provider, model_text
    return provider, default_model or None


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
            sys_prompt=_SAMPLE_PROMPT_PREFIX,
            prompt="",
            client=key,
        )
        self._cache[key] = prompt_fn
        return prompt_fn


def _build_trace_prompt(sample_record: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    prompt_text = str(sample_record.get("prompt") or "").strip()
    if prompt_text:
        lines.append("Task prompt:")
        lines.append(_truncate(prompt_text, 1_200))
        lines.append("")
    lines.append("Execution trace:")

    total_chars = sum(len(line) for line in lines)
    event_lines = 0
    for event_record in events:
        line = _summarize_event(event_record)
        if not line:
            continue
        if event_lines >= _MAX_TRACE_LINES:
            lines.append("[trace truncated]")
            break
        if total_chars + len(line) > _MAX_TRACE_CHARS:
            lines.append("[trace truncated]")
            break
        lines.append(line)
        total_chars += len(line)
        event_lines += 1

    return "\n".join(lines).strip()


def _summarize_event(event_record: Dict[str, Any]) -> str:
    event = event_record.get("event") if isinstance(event_record.get("event"), dict) else {}
    if not isinstance(event, dict):
        return ""
    event_type = str(event.get("type") or "").strip()
    if not event_type:
        return ""

    if event_type == "turn_start":
        return "turn_start"

    if event_type in {"tool_execution_start", "tool_policy_blocked"}:
        tool_name = str(event.get("toolName") or "unknown").strip() or "unknown"
        args = _json_preview(event.get("args"), limit=260)
        return f"{event_type} tool={tool_name} args={args}"

    if event_type == "tool_execution_end":
        tool_name = str(event.get("toolName") or "unknown").strip() or "unknown"
        is_error = bool(event.get("isError"))
        preview = _tool_result_preview(event)
        suffix = f" preview={preview}" if preview else ""
        return f"tool_execution_end tool={tool_name} error={is_error}{suffix}"

    if event_type == "message_end":
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        role = str(message.get("role") or "").strip()
        if role == "user":
            return ""
        preview = _message_preview(message)
        if not preview:
            return ""
        return f"message_end role={role} preview={_truncate(preview, 320)}"

    if event_type == "turn_end":
        tool_results = event.get("toolResults")
        tool_count = len(tool_results) if isinstance(tool_results, list) else 0
        return f"turn_end tool_results={tool_count}"

    return ""


def _tool_result_preview(event: Dict[str, Any]) -> str:
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    if not isinstance(result, dict):
        return ""
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    if isinstance(details, dict):
        error_text = str(details.get("error") or "").strip()
        if error_text:
            return _truncate(error_text, 220)
    content = result.get("content")
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") != "text":
                continue
            text = str(item.get("text") or "").strip()
            if text:
                texts.append(text)
        if texts:
            return _truncate(" ".join(texts), 220)
    return ""


def _message_preview(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    chunks: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "text":
            text = str(item.get("text") or "").strip()
            if text:
                chunks.append(text)
            continue
        if item_type == "toolCall":
            tool_name = str(item.get("name") or "unknown").strip() or "unknown"
            args = _json_preview(item.get("arguments"), limit=180)
            chunks.append(f"toolCall {tool_name}: {args}")
    return "\n".join(chunks).strip()


def _parse_steps_output(raw_output: Any) -> List[str]:
    parsed: Any
    if isinstance(raw_output, dict):
        parsed = raw_output
    elif isinstance(raw_output, list):
        parsed = raw_output
    else:
        text = str(raw_output or "").strip()
        if not text:
            raise ValueError("PromptFunction returned an empty response.")
        cleaned = _JSON_FENCE_RE.sub("", text).strip()
        parsed = _parse_json_like_payload(cleaned)

    if isinstance(parsed, dict):
        steps = parsed.get("steps")
    else:
        steps = parsed

    normalized = _normalize_steps(steps)
    if not normalized:
        raise ValueError("PromptFunction returned no usable steps.")
    return normalized


def _parse_json_like_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start < 0 or end <= start:
            continue
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("PromptFunction did not return valid JSON.")


def _manifest_model_hint(manifest: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    config = manifest.get("config") if isinstance(manifest, dict) else {}
    if not isinstance(config, dict):
        return "", None
    executor_options = config.get("executor_options")
    if not isinstance(executor_options, dict):
        return "", None
    override = executor_options.get("model_override")
    if not isinstance(override, dict):
        return "", None
    provider = str(override.get("provider") or "").strip()
    model = str(override.get("id") or override.get("model") or "").strip() or None
    return provider, model


def _sample_usage_model_hint(sample_record: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    result = sample_record.get("result") if isinstance(sample_record.get("result"), dict) else {}
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    last = usage.get("last") if isinstance(usage.get("last"), dict) else {}
    provider = str(last.get("provider") or "").strip()
    model = str(last.get("model") or "").strip() or None
    return provider, model


def _collect_event_timestamps(events: Sequence[Dict[str, Any]]) -> List[Tuple[float, str]]:
    found: List[Tuple[float, str]] = []
    for event_record in events:
        _walk_timestamps(event_record, found)
    dedup: Dict[float, str] = {}
    for epoch, iso_value in found:
        dedup[epoch] = iso_value
    ordered = sorted(dedup.items(), key=lambda item: item[0])
    return [(epoch, iso_value) for epoch, iso_value in ordered]


def _walk_timestamps(value: Any, found: List[Tuple[float, str]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "timestamp":
                normalized = _normalize_timestamp_value(child, with_epoch=True)
                if normalized is not None:
                    found.append(normalized)
            _walk_timestamps(child, found)
        return
    if isinstance(value, list):
        for item in value:
            _walk_timestamps(item, found)


def _normalize_steps(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    normalized: List[str] = []
    for item in value:
        text = _NUMBERED_STEP_RE.sub("", str(item or "").strip())
        if text:
            normalized.append(text)
    return normalized


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            output.append(text)
    return output


def _normalize_timestamp_value(value: Any, *, with_epoch: bool = False) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
        iso_value = dt.isoformat()
        if with_epoch:
            return (dt.timestamp(), iso_value)
        return iso_value

    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        try:
            return _normalize_timestamp_value(float(text), with_epoch=with_epoch)
        except Exception:
            return None

    parsed = _parse_iso_datetime(text)
    if parsed is None:
        return None
    iso_value = parsed.isoformat()
    if with_epoch:
        return (parsed.timestamp(), iso_value)
    return iso_value


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _shift_iso_timestamp(value: str, seconds: float) -> Optional[str]:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    return (parsed + timedelta(seconds=float(seconds))).isoformat()


def _json_preview(value: Any, *, limit: int) -> str:
    if value in (None, "", {}, []):
        return "n/a"
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except Exception:
        text = str(value)
    return _truncate(text, limit)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _clean_error(exc: Exception) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
