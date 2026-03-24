from __future__ import annotations

import json
import re
from collections import defaultdict
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
_BULLETED_STEP_RE = re.compile(r"^\s*(?:[-*•]+)\s*")
_NON_ALPHA_STEP_RE = re.compile(r"^[\W\d_]+$")
_MEMORY_FILE_RE = re.compile(r"^(?:.+?)_sample_\d+_(.+)$")
_MEMORY_MESSAGE_RE = re.compile(r"^(user|assistant):\s*(.*)$", re.IGNORECASE | re.DOTALL)
_MEMORY_TOOL_START_RE = re.compile(r"^tool_execution_start\s+([^\s]+)(?:\s+cmd=(.+))?$", re.IGNORECASE)
_MEMORY_TOOL_END_RE = re.compile(r"^tool_execution_end\s+([^\s]+)\s+error=(true|false)(?:\s+cmd=(.+))?$", re.IGNORECASE)
_MEMORY_TOOL_ERROR_RE = re.compile(r"^tool_execution_error\s+([^\s:]+)(?::\s*(.+))?$", re.IGNORECASE)
_STEERING_PREFIXES = (
    "evaluation steering:",
    "use the memory context as soft background.",
    "you are in an evaluation environment.",
)
_SAMPLE_PROMPT_PREFIX = (
    "You summarize agent execution traces into concise high-level steps for evaluation reports.\n"
    'Return strict JSON only in the form {"steps": ["..."]}.\n'
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


def load_run_memory_by_sample(run_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    memory_dir = Path(str(run_dir or "")) / "_memory"
    if not memory_dir.is_dir():
        return {}

    loaded: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(memory_dir.glob("*.jsonl")):
        sample_key = _memory_file_sample_key(path)
        if not sample_key:
            continue
        records = _load_jsonl_dicts(path)
        if records:
            loaded[sample_key] = records
    return loaded


def resolve_sample_memory_entries(
    memory_by_sample: Dict[str, List[Dict[str, Any]]],
    sample_id: str,
) -> List[Dict[str, Any]]:
    if not memory_by_sample:
        return []
    raw_key = str(sample_id or "").strip()
    if raw_key and raw_key in memory_by_sample:
        return list(memory_by_sample.get(raw_key) or [])
    safe_key = _safe_trace_id(raw_key)
    if safe_key and safe_key in memory_by_sample:
        return list(memory_by_sample.get(safe_key) or [])
    return []


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
    memory_by_sample = load_run_memory_by_sample(run_dir)
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
            memory_entries=resolve_sample_memory_entries(memory_by_sample, sample_id),
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
    memory_entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    provider, model = resolve_step_summary_client(
        manifest=manifest,
        sample_record=sample_record,
        provider_override=provider_override,
        model_override=model_override,
    )

    trace_source = "event"
    actions = _build_memory_actions(memory_entries or [], events)
    if actions:
        trace_source = "memory"
    else:
        actions = _build_event_actions(events)

    tool_names = _merge_tool_names(collect_distinct_tool_names(events), _collect_tool_names_from_actions(actions))
    summary: Dict[str, Any] = {
        "steps": [],
        "step_count": 0,
        "tool_names": tool_names,
        "tool_count": len(tool_names),
        "summary_model": format_summary_model(provider, model),
        "generated_at": _utc_now_iso(),
        "source": f"{trace_source}+promptfunction",
    }

    trace_text = _build_trace_prompt_from_actions(sample_record, actions)
    fallback_steps = _build_fallback_steps_from_actions(actions)

    if not trace_text:
        summary["source"] = f"{trace_source}_fallback"
        if fallback_steps:
            summary["steps"] = fallback_steps
            summary["step_count"] = len(fallback_steps)
        else:
            summary["error"] = "No usable execution trace was available for summarization."
        return summary

    if PromptFunction is None:
        summary["source"] = f"{trace_source}_fallback"
        if fallback_steps:
            summary["steps"] = fallback_steps
            summary["step_count"] = len(fallback_steps)
        else:
            summary["error"] = "POP PromptFunction is unavailable."
        return summary

    cache = prompt_functions or _PromptFunctionCache()
    try:
        prompt_fn = cache.get(provider or "gemini")
        execute_kwargs: Dict[str, Any] = {"tools": [], "tool_choice": "auto"}
        if model:
            execute_kwargs["model"] = model
        raw_output = prompt_fn.execute(trace_text, **execute_kwargs)
        steps = _parse_steps_output(raw_output)
    except Exception:
        steps = []

    if steps:
        summary["steps"] = steps
        summary["step_count"] = len(steps)
        return summary

    summary["source"] = f"{trace_source}_fallback"
    if fallback_steps:
        summary["steps"] = fallback_steps
        summary["step_count"] = len(fallback_steps)
        return summary

    summary["error"] = "No usable execution trace was available for summarization."
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


def _build_trace_prompt_from_actions(sample_record: Dict[str, Any], actions: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    prompt_text = str(sample_record.get("prompt") or "").strip()
    if prompt_text:
        lines.append("Task prompt:")
        lines.append(_truncate(prompt_text, 1_200))
        lines.append("")
    lines.append("Condensed execution trace:")

    total_chars = sum(len(line) for line in lines)
    action_lines = 0
    for action in actions:
        line = _format_trace_action_line(action)
        if not line:
            continue
        if action_lines >= _MAX_TRACE_LINES:
            lines.append("[trace truncated]")
            break
        if total_chars + len(line) > _MAX_TRACE_CHARS:
            lines.append("[trace truncated]")
            break
        lines.append(line)
        total_chars += len(line)
        action_lines += 1

    if action_lines == 0:
        return ""
    return "\n".join(lines).strip()


def _build_memory_actions(
    memory_entries: Sequence[Dict[str, Any]],
    events: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not memory_entries:
        return []

    start_events, end_events = _build_tool_event_queues(events)
    start_positions: Dict[str, int] = defaultdict(int)
    end_positions: Dict[str, int] = defaultdict(int)
    open_actions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    actions: List[Dict[str, Any]] = []

    for record in memory_entries:
        if not isinstance(record, dict):
            continue
        memory_type = str(record.get("memory_type") or "").strip().lower()
        text = str(record.get("text") or "").strip()
        if not text:
            continue

        if memory_type == "tool_call":
            parsed_tool = _parse_memory_tool_line(text)
            if not parsed_tool:
                continue
            tool_name = str(parsed_tool.get("tool_name") or "").strip()
            if not tool_name:
                continue
            if str(parsed_tool.get("phase")) == "start":
                start_event = _consume_tool_event(start_events, start_positions, tool_name)
                action = {
                    "kind": "tool",
                    "tool_name": tool_name,
                    "args": _event_args(start_event),
                    "command": str(parsed_tool.get("command") or "").strip(),
                    "result_preview": "",
                    "error": "",
                }
                actions.append(action)
                open_actions[tool_name].append(action)
                continue

            end_event = _consume_tool_event(end_events, end_positions, tool_name)
            action = open_actions[tool_name].pop(0) if open_actions.get(tool_name) else None
            if action is None:
                action = {
                    "kind": "tool",
                    "tool_name": tool_name,
                    "args": _event_args(end_event),
                    "command": str(parsed_tool.get("command") or "").strip(),
                    "result_preview": "",
                    "error": "",
                }
                actions.append(action)
            result_preview = _tool_result_preview(_event_payload(end_event))
            if result_preview:
                action["result_preview"] = result_preview
            if bool(parsed_tool.get("is_error")):
                action["error"] = _event_error_text(_event_payload(end_event)) or "tool execution failed"
            continue

        if memory_type == "error":
            tool_name, error_text = _parse_memory_error_line(text)
            if not tool_name or not error_text:
                continue
            action = _find_latest_tool_action(actions, tool_name)
            if action is None:
                action = {
                    "kind": "tool",
                    "tool_name": tool_name,
                    "args": None,
                    "command": "",
                    "result_preview": "",
                    "error": "",
                }
                actions.append(action)
            action["error"] = error_text
            continue

        if memory_type == "message":
            role, message_text = _parse_memory_message_line(text)
            if role != "assistant" or not _is_useful_assistant_memory_message(message_text):
                continue
            actions.append({"kind": "assistant", "text": message_text})

    return actions


def _build_event_actions(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    open_by_call_id: Dict[str, Dict[str, Any]] = {}
    open_by_tool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for event_record in events:
        event = _event_payload(event_record)
        event_type = str(event.get("type") or "").strip()
        if event_type == "tool_execution_start":
            tool_name = str(event.get("toolName") or "unknown").strip() or "unknown"
            action = {
                "kind": "tool",
                "tool_name": tool_name,
                "args": event.get("args") if isinstance(event.get("args"), dict) else None,
                "command": "",
                "result_preview": "",
                "error": "",
            }
            call_id = str(event.get("toolCallId") or "").strip()
            actions.append(action)
            if call_id:
                open_by_call_id[call_id] = action
            open_by_tool[tool_name].append(action)
            continue

        if event_type == "tool_execution_end":
            tool_name = str(event.get("toolName") or "unknown").strip() or "unknown"
            call_id = str(event.get("toolCallId") or "").strip()
            action = open_by_call_id.pop(call_id, None) if call_id else None
            if action is None and open_by_tool.get(tool_name):
                action = open_by_tool[tool_name].pop(0)
            elif action is not None and open_by_tool.get(tool_name):
                try:
                    open_by_tool[tool_name].remove(action)
                except ValueError:
                    pass
            if action is None:
                action = {
                    "kind": "tool",
                    "tool_name": tool_name,
                    "args": None,
                    "command": "",
                    "result_preview": "",
                    "error": "",
                }
                actions.append(action)
            if not action.get("args") and isinstance(event.get("args"), dict):
                action["args"] = event.get("args")
            result_preview = _tool_result_preview(event)
            if result_preview:
                action["result_preview"] = result_preview
            error_text = _event_error_text(event)
            if error_text:
                action["error"] = error_text
            continue

        if event_type in {"tool_execution_error", "tool_policy_blocked"}:
            tool_name = str(event.get("toolName") or "unknown").strip() or "unknown"
            action = {
                "kind": "tool",
                "tool_name": tool_name,
                "args": event.get("args") if isinstance(event.get("args"), dict) else None,
                "command": "",
                "result_preview": "",
                "error": _event_error_text(event) or "tool execution failed",
            }
            actions.append(action)
            continue

        if event_type != "message_end":
            continue
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "assistant":
            continue
        assistant_text = _assistant_visible_text(message)
        if not _is_usable_assistant_answer(assistant_text):
            continue
        actions.append({"kind": "assistant", "text": assistant_text})

    return actions


def _build_fallback_steps_from_actions(actions: Sequence[Dict[str, Any]]) -> List[str]:
    raw_steps: List[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("kind") or "") == "tool":
            step = _fallback_tool_step(action)
        elif str(action.get("kind") or "") == "assistant":
            step = _fallback_assistant_step(str(action.get("text") or ""))
        else:
            step = ""
        if step:
            raw_steps.append(step)
    normalized = _normalize_steps(raw_steps)
    return normalized[:8]


def _format_trace_action_line(action: Dict[str, Any]) -> str:
    kind = str(action.get("kind") or "").strip()
    if kind == "tool":
        tool_name = str(action.get("tool_name") or "").strip() or "unknown"
        target = _tool_target_phrase(tool_name, action.get("args"), command=action.get("command"))
        result_preview = str(action.get("result_preview") or "").strip()
        error_text = str(action.get("error") or "").strip()
        bits = [f"tool={tool_name}"]
        if target:
            bits.append(f"action={target}")
        if error_text:
            bits.append(f"error={_truncate(error_text, 180)}")
        elif result_preview:
            bits.append(f"result={_truncate(result_preview, 180)}")
        return "- " + " | ".join(bits)
    if kind == "assistant":
        text = str(action.get("text") or "").strip()
        if not text:
            return ""
        return f"- assistant_answer text={_truncate(text, 180)}"
    return ""


def _fallback_tool_step(action: Dict[str, Any]) -> str:
    tool_name = str(action.get("tool_name") or "").strip() or "unknown"
    phrase = _tool_target_phrase(tool_name, action.get("args"), command=action.get("command"))
    error_text = str(action.get("error") or "").strip()
    if error_text:
        if phrase:
            return _sentence_case(f"try to {phrase}, but hit {error_text}")
        return _sentence_case(f"try {humanize_tool_name(tool_name) or tool_name}, but hit {error_text}")
    if phrase:
        return _sentence_case(phrase)
    return _sentence_case(f"use {humanize_tool_name(tool_name) or tool_name}")


def _fallback_assistant_step(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    return _sentence_case(f"return the final answer {_truncate(cleaned, 80)}")


def _tool_target_phrase(tool_name: str, args: Any, *, command: str = "") -> str:
    raw_name = str(tool_name or "").strip()
    args_dict = args if isinstance(args, dict) else {}
    command_text = str(command or "").strip()
    target_text = _args_target_preview(args_dict)
    lowered = raw_name.lower()

    if lowered == "file_read":
        path_value = _first_non_empty(args_dict, ("workspace_path", "path", "local_path"))
        query_value = _first_non_empty(args_dict, ("query",))
        line_start = _safe_int(args_dict.get("line_start"))
        line_count = _safe_int(args_dict.get("line_count"))
        file_label = _basename(path_value) or "a local file"
        if query_value:
            return f'search {file_label} for "{_truncate(query_value, 80)}"'
        if line_start > 0 and line_count > 0:
            return f"read {file_label} at lines {line_start}-{line_start + max(line_count - 1, 0)}"
        return f"read {file_label}"

    if lowered == "calculator":
        expression = _first_non_empty(args_dict, ("expression",))
        if expression:
            return f"calculate {_truncate(expression, 120)}"
        return "calculate the requested value"

    if lowered in {"perplexity_search", "search_engine", "openalex_works"}:
        if target_text:
            return f"search for {_truncate(target_text, 120)}"
        return f"search with {humanize_tool_name(raw_name) or raw_name}"

    if lowered in {"jina_web_snapshot", "web_snapshot", "web_browser"}:
        if target_text:
            return f"open {_truncate(target_text, 120)}"
        return f"open the target page with {humanize_tool_name(raw_name) or raw_name}"

    if lowered == "download_url_to_file":
        if target_text:
            return f"download {_truncate(target_text, 120)}"
        return "download the target file"

    if lowered == "bash_exec":
        if command_text:
            return f"inspect the local workspace with {_truncate(command_text, 120)}"
        if target_text:
            return f"inspect the local workspace using {_truncate(target_text, 120)}"
        return "inspect the local workspace"

    if target_text:
        return f"use {humanize_tool_name(raw_name) or raw_name} on {_truncate(target_text, 120)}"
    return f"use {humanize_tool_name(raw_name) or raw_name}"


def _args_target_preview(args: Dict[str, Any]) -> str:
    for key in (
        "query",
        "q",
        "open",
        "url",
        "uri",
        "source_url",
        "final_url",
        "path",
        "workspace_path",
        "local_path",
        "expression",
        "doi",
        "title",
        "id",
    ):
        value = _first_non_empty(args, (key,))
        if value:
            return value
    if args:
        return _truncate(_json_preview(args, limit=180), 180)
    return ""


def _first_non_empty(mapping: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _collect_tool_names_from_actions(actions: Sequence[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("kind") or "") != "tool":
            continue
        tool_name = str(action.get("tool_name") or "").strip()
        if not tool_name or tool_name in seen:
            continue
        seen.add(tool_name)
        ordered.append(tool_name)
    return ordered


def _merge_tool_names(primary: Sequence[str], secondary: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    merged: List[str] = []
    for collection in (primary, secondary):
        for item in collection:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def _build_tool_event_queues(
    events: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    start_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    end_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event_record in events:
        event = _event_payload(event_record)
        event_type = str(event.get("type") or "").strip()
        tool_name = str(event.get("toolName") or "").strip()
        if not tool_name:
            continue
        if event_type == "tool_execution_start":
            start_events[tool_name].append(event_record)
        elif event_type == "tool_execution_end":
            end_events[tool_name].append(event_record)
    return start_events, end_events


def _consume_tool_event(
    event_queues: Dict[str, List[Dict[str, Any]]],
    positions: Dict[str, int],
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    queue = event_queues.get(str(tool_name or "").strip())
    if not queue:
        return None
    index = positions[str(tool_name or "").strip()]
    if index >= len(queue):
        return None
    positions[str(tool_name or "").strip()] = index + 1
    return queue[index]


def _parse_memory_tool_line(text: str) -> Optional[Dict[str, Any]]:
    start_match = _MEMORY_TOOL_START_RE.match(str(text or "").strip())
    if start_match:
        return {
            "phase": "start",
            "tool_name": str(start_match.group(1) or "").strip(),
            "command": str(start_match.group(2) or "").strip(),
        }

    end_match = _MEMORY_TOOL_END_RE.match(str(text or "").strip())
    if end_match:
        return {
            "phase": "end",
            "tool_name": str(end_match.group(1) or "").strip(),
            "is_error": str(end_match.group(2) or "").strip().lower() == "true",
            "command": str(end_match.group(3) or "").strip(),
        }
    return None


def _parse_memory_error_line(text: str) -> Tuple[str, str]:
    match = _MEMORY_TOOL_ERROR_RE.match(str(text or "").strip())
    if not match:
        return "", ""
    tool_name = str(match.group(1) or "").strip()
    error_text = str(match.group(2) or "").strip()
    return tool_name, error_text


def _parse_memory_message_line(text: str) -> Tuple[str, str]:
    match = _MEMORY_MESSAGE_RE.match(str(text or "").strip())
    if not match:
        return "", ""
    role = str(match.group(1) or "").strip().lower()
    message_text = str(match.group(2) or "").strip()
    return role, message_text


def _is_useful_assistant_memory_message(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return _is_usable_assistant_answer(candidate)


def _is_usable_assistant_answer(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    lowered = candidate.lower()
    return not any(lowered.startswith(prefix) for prefix in _STEERING_PREFIXES)


def _assistant_visible_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    chunks: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip() != "text":
            continue
        text = str(item.get("text") or "").strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def _find_latest_tool_action(actions: Sequence[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
    lookup = str(tool_name or "").strip()
    for action in reversed(actions):
        if not isinstance(action, dict):
            continue
        if str(action.get("kind") or "") != "tool":
            continue
        if str(action.get("tool_name") or "").strip() == lookup:
            return action
    return None


def _event_payload(event_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(event_record, dict):
        return {}
    event = event_record.get("event")
    return event if isinstance(event, dict) else {}


def _event_args(event_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    event = _event_payload(event_record)
    args = event.get("args")
    return args if isinstance(args, dict) else None


def _event_error_text(event: Dict[str, Any]) -> str:
    if not isinstance(event, dict):
        return ""
    for key in ("error", "message", "reason"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    result = event.get("result")
    if isinstance(result, dict):
        details = result.get("details")
        if isinstance(details, dict):
            error_text = str(details.get("error") or "").strip()
            if error_text:
                return error_text
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
        result_text = str(details.get("result_text") or "").strip()
        if result_text:
            return _truncate(result_text, 220)
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
            joined = " ".join(texts)
            if joined.startswith("{") and joined.endswith("}"):
                try:
                    parsed = json.loads(joined)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    preview = _first_non_empty(
                        parsed,
                        ("content_preview", "workspace_path", "path", "saved_landing_page_path", "final_url", "error"),
                    )
                    if preview:
                        return _truncate(preview, 220)
            return _truncate(joined, 220)
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
    parsed: Any = None
    cleaned_text = ""
    if isinstance(raw_output, dict):
        parsed = raw_output
    elif isinstance(raw_output, list):
        parsed = raw_output
    else:
        text = str(raw_output or "").strip()
        if not text:
            raise ValueError("PromptFunction returned an empty response.")
        cleaned_text = _JSON_FENCE_RE.sub("", text).strip()
        try:
            parsed = _parse_json_like_payload(cleaned_text)
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        steps = parsed.get("steps")
    else:
        steps = parsed

    normalized = _normalize_steps(steps)
    if normalized:
        return normalized

    fallback_steps = _extract_steps_from_text(cleaned_text)
    if fallback_steps:
        return fallback_steps
    raise ValueError("PromptFunction returned no usable steps.")


def _extract_steps_from_text(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    candidates: List[str] = []
    for line in raw.splitlines():
        stripped = str(line or "").strip()
        if not stripped:
            continue
        normalized = _NUMBERED_STEP_RE.sub("", stripped)
        normalized = _BULLETED_STEP_RE.sub("", normalized).strip()
        if (
            normalized
            and (
                normalized != stripped
                or _NUMBERED_STEP_RE.match(stripped) is not None
                or _BULLETED_STEP_RE.match(stripped) is not None
            )
        ):
            candidates.append(normalized)

    if not candidates and raw.count("\n") >= 1:
        multi_line = [str(line or "").strip() for line in raw.splitlines() if str(line or "").strip()]
        if 1 < len(multi_line) <= 8:
            candidates = multi_line

    return _normalize_steps(candidates)


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
    seen: set[str] = set()
    for item in value:
        text = _NUMBERED_STEP_RE.sub("", str(item or "").strip())
        text = re.sub(r"\s+", " ", text).strip()
        if not _is_usable_step_text(text):
            continue
        dedupe_key = text.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(text)
    return normalized


def _is_usable_step_text(text: str) -> bool:
    candidate = str(text or "").strip()
    if len(candidate) < 4:
        return False
    if _NON_ALPHA_STEP_RE.fullmatch(candidate):
        return False
    return any(char.isalpha() for char in candidate)


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


def _memory_file_sample_key(path: Path) -> str:
    stem = str(path.stem or "").strip()
    if not stem:
        return ""
    match = _MEMORY_FILE_RE.match(stem)
    if match:
        return str(match.group(1) or "").strip()
    return stem


def _load_jsonl_dicts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
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
    except Exception:
        return []
    return rows


def _safe_trace_id(value: str) -> str:
    text = str(value or "sample")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _basename(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part for part in re.split(r"[\\/]+", text) if part]
    return parts[-1] if parts else text


def _sentence_case(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]
