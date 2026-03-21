"""Generate styled HTML reports for POP-Agent eval runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Tuple

from eval.agent_step_summary import (
    collect_distinct_tool_names,
    humanize_tool_name,
    normalize_agent_execution_summary,
    persist_agent_execution_summaries,
    resolve_result_timing,
)
from eval.run_analysis import (
    analyze_run_artifacts,
    build_run_analysis,
    has_persisted_run_analysis,
    persist_run_analysis,
)


BASE_STYLE = """
  <style>
    :root { --bg:#f4eee3; --panel:rgba(255,251,245,.88); --ink:#1f2937; --muted:#5b6675; --line:rgba(31,41,55,.10); --shadow:0 18px 48px rgba(43,56,73,.12); --teal:#1c7c7d; --mint:#2d8c68; --blue:#356ea9; --gold:#c48824; --coral:#c65b39; --slate:#6a7280; --sans:"Aptos","Segoe UI Variable","Segoe UI",sans-serif; --mono:"Cascadia Code","Consolas",monospace; }
    * { box-sizing:border-box; } body { margin:0; color:var(--ink); font-family:var(--sans); background:radial-gradient(circle at top left, rgba(255,246,225,.95), transparent 36%), radial-gradient(circle at top right, rgba(213,235,231,.85), transparent 30%), linear-gradient(180deg, #f4eee2 0%, #f7f3ea 44%, #f2ecdf 100%); } h1,h2,h3,p { margin:0; } a { color:var(--blue); text-decoration:none; } a:hover { text-decoration:underline; }
    .page { width:min(1480px, calc(100% - 32px)); margin:24px auto 40px; } .panel,.table-shell,.subpanel { border:1px solid var(--line); border-radius:24px; background:var(--panel); box-shadow:var(--shadow); backdrop-filter:blur(16px); } .panel { padding:24px; margin-top:20px; } .subpanel { padding:18px; }
    .hero { padding:28px; } .hero-grid { display:grid; grid-template-columns:minmax(0,1fr) 240px; gap:24px; align-items:start; } .eyebrow { margin-bottom:10px; font-size:.82rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--teal); } .hero-title { font-size:clamp(2rem, 4vw, 3.2rem); line-height:1.0; letter-spacing:-.04em; margin-bottom:14px; word-break:break-word; } .hero-copy { color:var(--muted); line-height:1.6; margin-bottom:18px; max-width:78ch; }
    .nav-row { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:18px; } .nav-link { display:inline-flex; align-items:center; min-height:36px; padding:0 14px; border-radius:999px; border:1px solid rgba(31,41,55,.08); background:rgba(255,255,255,.72); font-weight:700; }
    .chip-grid { display:flex; flex-wrap:wrap; gap:12px; } .chip { min-width:140px; padding:12px 14px; border-radius:16px; border:1px solid rgba(31,41,55,.08); background:rgba(255,255,255,.68); } .chip-label { display:block; color:var(--muted); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:6px; } .chip-value { display:block; font-size:.95rem; line-height:1.35; word-break:break-word; }
    .score { width:220px; height:220px; border-radius:50%; padding:15px; } .score-inner { width:100%; height:100%; border-radius:50%; display:flex; align-items:center; justify-content:center; text-align:center; padding:20px; background:rgba(255,251,245,.96); border:1px solid rgba(31,41,55,.06); } .score-kicker { display:block; color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; } .score-value { display:block; font-size:2.5rem; line-height:1; letter-spacing:-.05em; margin-bottom:10px; } .score-note { display:block; color:var(--muted); font-size:.92rem; line-height:1.4; }
    .section-head { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:16px; } .section-copy { color:var(--muted); line-height:1.6; max-width:78ch; } .metric-grid,.chart-grid,.subpanel-grid,.columns { display:grid; gap:16px; } .metric-grid { grid-template-columns:repeat(3, minmax(0,1fr)); } .chart-grid { grid-template-columns:1.4fr 1fr 1fr; } .subpanel-grid,.columns { grid-template-columns:repeat(2, minmax(0,1fr)); }
    .metric { position:relative; padding:18px; border-radius:20px; border:1px solid rgba(31,41,55,.08); background:rgba(255,255,255,.72); overflow:hidden; } .metric::before { content:""; position:absolute; inset:0 auto auto 0; width:100%; height:4px; background:var(--tone); } .metric--teal { --tone:var(--teal); } .metric--mint { --tone:var(--mint); } .metric--blue { --tone:var(--blue); } .metric--gold { --tone:var(--gold); } .metric--coral { --tone:var(--coral); } .metric--slate { --tone:var(--slate); } .metric--danger { --tone:var(--coral); } .metric--warning { --tone:var(--gold); } .metric--success { --tone:var(--mint); } .metric-label { display:block; color:var(--muted); font-size:.8rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:12px; } .metric-value { display:block; font-size:clamp(1.4rem, 2.6vw, 2.1rem); line-height:1.05; letter-spacing:-.04em; margin-bottom:10px; } .metric-meta { color:var(--muted); font-size:.94rem; line-height:1.5; }
    .chart { border:1px solid rgba(31,41,55,.08); border-radius:20px; padding:18px; background:rgba(255,255,255,.72); } .chart-copy { color:var(--muted); font-size:.92rem; line-height:1.5; margin:6px 0 14px; } .chart-frame { height:300px; } .chart-empty { display:none; color:var(--muted); font-size:.92rem; margin-top:12px; }
    .table-shell { overflow:auto; background:rgba(255,255,255,.80); } table { width:100%; min-width:720px; border-collapse:collapse; } thead th { position:sticky; top:0; z-index:1; background:#f5efe7; color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; text-align:left; padding:14px 16px; border-bottom:1px solid var(--line); } tbody td { padding:14px 16px; border-top:1px solid rgba(31,41,55,.07); vertical-align:top; } tbody tr:hover { background:rgba(28,124,125,.05); }
    .error-row { background:rgba(198,91,57,.08); } .correct-row { background:rgba(45,140,104,.06); } .incorrect-row { background:rgba(196,136,36,.08); } .pill,.level { display:inline-flex; align-items:center; justify-content:center; min-height:30px; padding:0 12px; border-radius:999px; font-size:.82rem; font-weight:700; white-space:nowrap; } .level { background:rgba(28,124,125,.12); color:var(--teal); } .level--soft { background:rgba(31,41,55,.06); color:var(--ink); } .pill--success { background:rgba(45,140,104,.14); color:var(--mint); } .pill--warning { background:rgba(196,136,36,.14); color:#9a670d; } .pill--danger { background:rgba(198,91,57,.14); color:var(--coral); } .pill--info { background:rgba(53,110,169,.14); color:var(--blue); } .pill--neutral { background:rgba(31,41,55,.08); color:var(--slate); }
    .accuracy-cell { min-width:180px; } .accuracy-cell strong { display:inline-block; margin-top:8px; } .meter { width:100%; height:10px; border-radius:999px; background:rgba(31,41,55,.08); overflow:hidden; } .meter span { display:block; height:100%; background:linear-gradient(90deg, var(--teal), #34a0a4); } .corr-cell { text-align:center; transition:background-color .18s ease, color .18s ease; } .corr-cell--diag { box-shadow:inset 0 0 0 1px rgba(45,140,104,.12); }
    .mono { font-family:var(--mono); font-size:.9rem; } .sample-id,.model-cell { word-break:break-word; } .sample-link { font-weight:700; } .subtle,.empty { color:var(--muted); } .error-text { display:inline-block; max-width:360px; line-height:1.5; }
    .kv-table { width:100%; border-collapse:collapse; } .kv-table th,.kv-table td { padding:12px 14px; text-align:left; vertical-align:top; border-top:1px solid rgba(31,41,55,.07); } .kv-table tbody tr:first-child th,.kv-table tbody tr:first-child td { border-top:none; } .kv-table th { width:220px; color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.06em; }
    pre { margin:0; padding:16px; border-radius:18px; border:1px solid rgba(31,41,55,.08); background:rgba(31,41,55,.04); overflow:auto; white-space:pre-wrap; word-break:break-word; line-height:1.55; font-family:var(--mono); font-size:.88rem; }
    .plain-list { margin:0; padding-left:22px; } .plain-list li { margin-top:8px; line-height:1.6; } .plain-list li:first-child { margin-top:0; }
    .timeline { display:grid; gap:14px; } .timeline-item { padding:16px 18px; border-radius:20px; border:1px solid rgba(31,41,55,.08); background:rgba(255,255,255,.72); } .timeline-head { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-bottom:8px; } .timeline-meta { color:var(--muted); font-size:.92rem; line-height:1.5; margin-bottom:8px; } .timeline-copy { line-height:1.6; white-space:pre-wrap; }
    .raw-toggle { margin-top:16px; border:1px solid rgba(31,41,55,.08); border-radius:20px; background:rgba(255,255,255,.62); padding:14px 16px; } .raw-toggle summary { cursor:pointer; font-weight:700; }
    .charts-unavailable .chart-empty { display:block; } .charts-unavailable canvas { display:none !important; }
    @media (max-width:1100px) { .hero-grid { grid-template-columns:1fr; } .metric-grid,.chart-grid,.subpanel-grid,.columns { grid-template-columns:1fr 1fr; } .chart-grid .chart:first-child { grid-column:1 / -1; } } @media (max-width:760px) { .page { width:min(100% - 16px, 1480px); margin:16px auto 24px; } .hero,.panel { padding:20px; } .metric-grid,.chart-grid,.subpanel-grid,.columns { grid-template-columns:1fr; } .score { width:190px; height:190px; } .section-head { flex-direction:column; align-items:flex-start; } }
  </style>
"""


@dataclass
class RunArtifacts:
    summary: Dict[str, Any]
    samples: List[Dict[str, Any]]
    manifest: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    events_by_sample: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


@dataclass
class SampleView:
    index: int
    sample_id: str
    slug: str
    filename: str
    href: str
    level: str
    status: str
    correct: bool
    outcome_label: str
    outcome_tone: str
    status_tone: str
    score_reason: str
    score: float
    latency_ms: float
    total_tokens: int
    call_count: int
    distinct_tool_count: int
    provider_model: str
    warning_count: int
    error: str
    failure_analysis: Dict[str, Any]
    note_text: str
    prompt: str
    ground_truth: str
    prediction: str
    trace_ref: str
    metadata: Dict[str, Any]
    annotator_metadata: Dict[str, Any]
    usage_before: Dict[str, Any]
    usage_after: Dict[str, Any]
    usage_delta: Dict[str, Any]
    usage_last: Dict[str, Any]
    warnings: List[str]
    attachments: List[Dict[str, Any]]
    error_record: Optional[Dict[str, Any]]
    events: List[Dict[str, Any]]
    agent_started_at: Optional[str]
    agent_ended_at: Optional[str]
    agent_tool_names: List[str]
    agent_execution_summary: Dict[str, Any]
    normalized_prediction: Any
    normalized_ground_truth: Any
    raw_sample: Dict[str, Any]


def _read_run_data(run_dir_or_zip: str) -> RunArtifacts:
    source = str(run_dir_or_zip or "").strip()
    if source.lower().endswith(".zip"):
        return _read_run_data_from_zip(source)
    return _read_run_data_from_dir(source)


def _prepare_run_artifacts(
    run_dir_or_zip: str,
    *,
    summarize_agent_steps: bool = False,
    summary_provider: Optional[str] = None,
    summary_model: Optional[str] = None,
    summarize_failure_causes: bool = False,
) -> RunArtifacts:
    source = str(run_dir_or_zip or "").strip()
    if source.lower().endswith(".zip"):
        if summarize_agent_steps:
            raise ValueError(
                "--summarize-agent-steps is not supported for zip inputs because report generation needs to rewrite samples.jsonl."
            )
        if summarize_failure_causes:
            raise ValueError(
                "--summarize-failure-causes is not supported for zip inputs because report generation needs to rewrite samples.jsonl."
            )
        artifacts = _read_run_data_from_zip(source)
        if not has_persisted_run_analysis(artifacts.summary, artifacts.samples):
            artifacts.summary, artifacts.samples, _analysis = analyze_run_artifacts(
                artifacts.summary,
                artifacts.samples,
                artifacts.manifest,
                artifacts.events_by_sample,
            )
        return artifacts

    artifacts = _read_run_data_from_dir(source)
    if summarize_agent_steps:
        artifacts.samples = persist_agent_execution_summaries(
            source,
            samples=artifacts.samples,
            manifest=artifacts.manifest,
            events_by_sample=artifacts.events_by_sample,
            provider_override=summary_provider,
            model_override=summary_model,
        )
    if summarize_failure_causes or not has_persisted_run_analysis(artifacts.summary, artifacts.samples):
        artifacts.summary, artifacts.samples, _analysis = persist_run_analysis(
            source,
            summary=artifacts.summary,
            samples=artifacts.samples,
            manifest=artifacts.manifest,
            events_by_sample=artifacts.events_by_sample,
            summarize_failure_causes=summarize_failure_causes,
            summary_provider=summary_provider,
            summary_model=summary_model,
        )
    return artifacts


def _read_run_data_from_dir(run_dir: str) -> RunArtifacts:
    summary_path = os.path.join(run_dir, "summary.json")
    samples_path = os.path.join(run_dir, "samples.jsonl")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"samples.jsonl not found in {run_dir}")
    return RunArtifacts(
        summary=_load_json_file(summary_path),
        samples=_load_jsonl_file(samples_path),
        manifest=_load_json_file_if_exists(os.path.join(run_dir, "manifest.json")),
        errors=_load_jsonl_file_if_exists(os.path.join(run_dir, "errors.jsonl")),
        events_by_sample=_group_events_by_sample(_load_jsonl_file_if_exists(os.path.join(run_dir, "events.jsonl"))),
    )


def _read_run_data_from_zip(zip_path: str) -> RunArtifacts:
    with zipfile.ZipFile(zip_path, "r") as zf:
        summary_name = _find_zip_member(zf, "summary.json")
        samples_name = _find_zip_member(zf, "samples.jsonl")
        if summary_name is None:
            raise FileNotFoundError(f"summary.json not found in {zip_path}")
        if samples_name is None:
            raise FileNotFoundError(f"samples.jsonl not found in {zip_path}")
        return RunArtifacts(
            summary=json.loads(zf.read(summary_name).decode("utf-8")),
            samples=_load_jsonl_zip(zf, samples_name),
            manifest=_load_json_zip_if_exists(zf, _find_zip_member(zf, "manifest.json")),
            errors=_load_jsonl_zip(zf, _find_zip_member(zf, "errors.jsonl")),
            events_by_sample=_group_events_by_sample(_load_jsonl_zip(zf, _find_zip_member(zf, "events.jsonl"))),
        )


def _find_zip_member(zf: zipfile.ZipFile, filename: str) -> Optional[str]:
    exact = None
    matches: List[str] = []
    for name in zf.namelist():
        normalized = name.rstrip("/")
        if normalized == filename:
            exact = normalized
            break
        if normalized.endswith("/" + filename):
            matches.append(normalized)
    if exact is not None:
        return exact
    if matches:
        matches.sort(key=len)
        return matches[0]
    return None


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_json_file_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        return _load_json_file(path)
    except Exception:
        return {}


def _load_json_zip_if_exists(zf: zipfile.ZipFile, member: Optional[str]) -> Dict[str, Any]:
    if not member:
        return {}
    try:
        return json.loads(zf.read(member).decode("utf-8"))
    except Exception:
        return {}


def _load_jsonl_file(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_jsonl_line(line)
            if parsed is not None:
                rows.append(parsed)
    return rows


def _load_jsonl_file_if_exists(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        return _load_jsonl_file(path)
    except Exception:
        return []


def _load_jsonl_zip(zf: zipfile.ZipFile, member: Optional[str]) -> List[Dict[str, Any]]:
    if not member:
        return []
    rows: List[Dict[str, Any]] = []
    with zf.open(member) as handle:
        for raw_line in handle:
            try:
                line = raw_line.decode("utf-8")
            except Exception:
                continue
            parsed = _parse_jsonl_line(line)
            if parsed is not None:
                rows.append(parsed)
    return rows


def _parse_jsonl_line(line: str) -> Optional[Dict[str, Any]]:
    text = str(line or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _group_events_by_sample(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        sample_id = str(event.get("sample_id") or "").strip()
        if not sample_id:
            continue
        grouped.setdefault(sample_id, []).append(event)
    for sample_events in grouped.values():
        sample_events.sort(key=lambda item: _to_int(item.get("event_index")))
    return grouped


def _build_sample_views(artifacts: RunArtifacts, sample_dir_name: str) -> List[SampleView]:
    errors_by_sample = {
        str(item.get("sample_id") or "").strip(): item
        for item in artifacts.errors
        if str(item.get("sample_id") or "").strip()
    }
    model_hint = _manifest_model_hint(artifacts.manifest)
    sample_views: List[SampleView] = []
    for index, sample_record in enumerate(artifacts.samples, 1):
        result = sample_record.get("result") if isinstance(sample_record.get("result"), dict) else {}
        score_result = result.get("score_result") if isinstance(result.get("score_result"), dict) else {}
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        metadata = sample_record.get("metadata") if isinstance(sample_record.get("metadata"), dict) else {}
        annotator = metadata.get("Annotator Metadata") if isinstance(metadata.get("Annotator Metadata"), dict) else {}
        usage_delta = usage.get("delta") if isinstance(usage.get("delta"), dict) else {}
        usage_last = usage.get("last") if isinstance(usage.get("last"), dict) else {}
        warnings = [str(item).strip() for item in usage.get("warnings", []) if str(item).strip()] if isinstance(usage.get("warnings"), list) else []
        attachments = [item for item in usage.get("attachments", []) if isinstance(item, dict)] if isinstance(usage.get("attachments"), list) else []
        sample_events = list(artifacts.events_by_sample.get(str(sample_record.get("sample_id") or f"sample-{index}"), []))
        agent_execution_summary = normalize_agent_execution_summary(result.get("agent_execution_summary"))
        agent_tool_names = list(agent_execution_summary.get("tool_names") or collect_distinct_tool_names(sample_events))
        failure_analysis = result.get("failure_analysis") if isinstance(result.get("failure_analysis"), dict) else {}
        agent_started_at, agent_ended_at = resolve_result_timing(sample_record, sample_events)
        sample_id = str(sample_record.get("sample_id") or f"sample-{index}")
        status = str(result.get("status", "unknown"))
        correct = bool(score_result.get("correct", False))
        if status == "error":
            outcome_label, outcome_tone = "Failed", "danger"
        elif correct:
            outcome_label, outcome_tone = "Correct", "success"
        else:
            outcome_label, outcome_tone = "Incorrect", "warning"
        status_tone = "danger" if status == "error" else ("info" if status == "ok" else "neutral")
        provider = str(usage_last.get("provider") or model_hint[0]).strip()
        model = str(usage_last.get("model") or model_hint[1]).strip()
        provider_model = " / ".join(part for part in [provider, model] if part)
        error_text = _clean_text(result.get("error"))
        cause_text = _humanize_failure_cause(failure_analysis)
        note_text = error_text or cause_text or (f"{len(warnings)} warning(s)" if warnings else "")
        slug = _slugify(sample_id)
        sample_views.append(
            SampleView(
                index=index,
                sample_id=sample_id,
                slug=slug,
                filename=f"{index:04d}_{slug}.html",
                href=f"{sample_dir_name}/{index:04d}_{slug}.html",
                level=str(metadata.get("Level", metadata.get("level", "unknown"))),
                status=status,
                correct=correct,
                outcome_label=outcome_label,
                outcome_tone=outcome_tone,
                status_tone=status_tone,
                score_reason=str(score_result.get("reason") or "n/a"),
                score=_to_float(score_result.get("score")),
                latency_ms=_to_float(result.get("latency_ms")),
                total_tokens=_to_int(usage_delta.get("total_tokens")),
                call_count=_to_int(usage_delta.get("calls")),
                distinct_tool_count=len(agent_tool_names),
                provider_model=provider_model,
                warning_count=len(warnings),
                error=error_text,
                failure_analysis=failure_analysis,
                note_text=note_text,
                prompt=str(sample_record.get("prompt") or ""),
                ground_truth=str(sample_record.get("ground_truth") or ""),
                prediction=str(result.get("prediction") or ""),
                trace_ref=str(result.get("trace_ref") or ""),
                metadata=metadata,
                annotator_metadata=annotator,
                usage_before=usage.get("before") if isinstance(usage.get("before"), dict) else {},
                usage_after=usage.get("after") if isinstance(usage.get("after"), dict) else {},
                usage_delta=usage_delta,
                usage_last=usage_last,
                warnings=warnings,
                attachments=attachments,
                error_record=errors_by_sample.get(sample_id),
                events=sample_events,
                agent_started_at=agent_started_at,
                agent_ended_at=agent_ended_at,
                agent_tool_names=agent_tool_names,
                agent_execution_summary=agent_execution_summary,
                normalized_prediction=score_result.get("normalized_prediction"),
                normalized_ground_truth=score_result.get("normalized_ground_truth"),
                raw_sample=sample_record,
            )
        )
    return sample_views


def _manifest_model_hint(manifest: Dict[str, Any]) -> Tuple[str, str]:
    config = manifest.get("config") if isinstance(manifest, dict) else {}
    if not isinstance(config, dict):
        return ("", "")
    executor_options = config.get("executor_options")
    if not isinstance(executor_options, dict):
        return ("", "")
    override = executor_options.get("model_override")
    if not isinstance(override, dict):
        return ("", "")
    return (str(override.get("provider") or ""), str(override.get("id") or override.get("model") or ""))


def _aggregate_metrics(samples: Iterable[SampleView]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    overall = {"total": 0, "correct": 0, "error_count": 0, "latencies": [], "tokens": [], "calls": [], "tools": []}
    by_level: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        overall["total"] += 1
        if sample.correct:
            overall["correct"] += 1
        if sample.status == "error":
            overall["error_count"] += 1
        overall["latencies"].append(sample.latency_ms)
        overall["tokens"].append(sample.total_tokens)
        overall["calls"].append(sample.call_count)
        overall["tools"].append(sample.distinct_tool_count)
        bucket = by_level.setdefault(
            sample.level,
            {"total": 0, "correct": 0, "error_count": 0, "latencies": [], "tokens": [], "calls": [], "tools": []},
        )
        bucket["total"] += 1
        if sample.correct:
            bucket["correct"] += 1
        if sample.status == "error":
            bucket["error_count"] += 1
        bucket["latencies"].append(sample.latency_ms)
        bucket["tokens"].append(sample.total_tokens)
        bucket["calls"].append(sample.call_count)
        bucket["tools"].append(sample.distinct_tool_count)

    def derive(metrics: Dict[str, Any]) -> None:
        total = metrics["total"]
        metrics["accuracy"] = metrics["correct"] / total if total else 0.0
        metrics["avg_latency_ms"] = float(statistics.mean(metrics["latencies"])) if metrics["latencies"] else 0.0
        metrics["median_latency_ms"] = float(statistics.median(metrics["latencies"])) if metrics["latencies"] else 0.0
        metrics["max_latency_ms"] = float(max(metrics["latencies"])) if metrics["latencies"] else 0.0
        metrics["avg_tokens"] = float(statistics.mean(metrics["tokens"])) if metrics["tokens"] else 0.0
        metrics["median_tokens"] = float(statistics.median(metrics["tokens"])) if metrics["tokens"] else 0.0
        metrics["max_tokens"] = float(max(metrics["tokens"])) if metrics["tokens"] else 0.0
        metrics["avg_calls"] = float(statistics.mean(metrics["calls"])) if metrics["calls"] else 0.0
        metrics["avg_tools"] = float(statistics.mean(metrics["tools"])) if metrics["tools"] else 0.0

    derive(overall)
    for item in by_level.values():
        derive(item)
    return overall, by_level


def _escape(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _display(value: Any, fallback: str = "n/a") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _fmt_int(value: Any, fallback: str = "n/a") -> str:
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return fallback


def _fmt_float(value: Any, digits: int = 1, suffix: str = "", fallback: str = "n/a") -> str:
    try:
        rendered = f"{float(value):,.{digits}f}"
    except Exception:
        return fallback
    return f"{rendered}{suffix}"


def _fmt_pct(value: Any, digits: int = 1, fallback: str = "n/a") -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return fallback


def _fmt_duration(seconds: Any) -> str:
    try:
        total_seconds = max(0, int(round(float(seconds or 0))))
    except Exception:
        return _display(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _fmt_ts(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "n/a"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        zone = dt.strftime("%Z")
        return dt.strftime("%b %d, %Y %H:%M:%S") + (f" {zone}" if zone else "")
    except Exception:
        return text


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _slugify(value: str) -> str:
    raw = str(value or "sample").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_")
    return (safe or "sample")[:96]


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _card(label: str, value: str, meta: str, tone: str) -> str:
    meta_html = f'<p class="metric-meta">{_escape(meta)}</p>' if meta else ""
    return f'<article class="metric metric--{tone}"><span class="metric-label">{_escape(label)}</span><strong class="metric-value">{_escape(value)}</strong>{meta_html}</article>'


def _chip(label: str, value: str) -> str:
    return f'<div class="chip"><span class="chip-label">{_escape(label)}</span><strong class="chip-value">{_escape(value)}</strong></div>'


def _pill(label: str, tone: str) -> str:
    return f'<span class="pill pill--{tone}">{_escape(label)}</span>'


def _json_block(value: Any) -> str:
    if value in ({}, [], None, ""):
        return '<p class="empty">n/a</p>'
    return f"<pre>{_escape(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))}</pre>"


def _text_block(value: Any) -> str:
    text = _display(value)
    if text == "n/a":
        return '<p class="empty">n/a</p>'
    return f"<pre>{_escape(text)}</pre>"


def _multiline_value(value: Any) -> str:
    text = _display(value)
    if text == "n/a":
        return '<p class="empty">n/a</p>'
    return f'<div style="white-space:pre-wrap; line-height:1.6">{_escape(text)}</div>'


def _step_list_block(value: Any) -> str:
    steps = _coerce_steps(value)
    if steps:
        return '<ol class="plain-list">' + "".join(f"<li>{_escape(step)}</li>" for step in steps) + "</ol>"
    text = _display(value)
    if text == "n/a":
        return '<p class="empty">n/a</p>'
    return f"<pre>{_escape(text)}</pre>"


def _kv_table(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return '<p class="empty">n/a</p>'
    body = "".join(f"<tr><th>{_escape(label)}</th><td>{value}</td></tr>" for label, value in rows)
    return f'<table class="kv-table"><tbody>{body}</tbody></table>'


def _string_list(items: List[str]) -> str:
    if not items:
        return '<p class="empty">n/a</p>'
    return '<ul class="plain-list">' + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _coerce_steps(value: Any) -> List[str]:
    if isinstance(value, list):
        output = [str(item).strip() for item in value if str(item).strip()]
        return output
    text = str(value or "").strip()
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and all(_looks_numbered_step(line) for line in lines):
        return [_strip_step_number(line) for line in lines]
    return []


def _looks_numbered_step(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if len(text) > 1 and text[0].isdigit() and "." in text[:4]:
        return True
    if len(text) > 1 and text[0].isdigit() and ")" in text[:4]:
        return True
    return False


def _strip_step_number(line: str) -> str:
    text = str(line or "").strip()
    for marker in (".", ")"):
        position = text.find(marker)
        if 0 < position <= 3 and text[:position].isdigit():
            return text[position + 1 :].strip()
    return text


def _fmt_sample_time_window(started_at: Optional[str], ended_at: Optional[str], latency_ms: float) -> str:
    if started_at and ended_at:
        start_dt = _parse_iso_dt(started_at)
        end_dt = _parse_iso_dt(ended_at)
        if start_dt is not None and end_dt is not None:
            duration_s = max((end_dt - start_dt).total_seconds(), 0.0)
            return f"{_fmt_ts(started_at)} -> {_fmt_ts(ended_at)} ({_fmt_duration(duration_s)})"
        return f"{_fmt_ts(started_at)} -> {_fmt_ts(ended_at)}"
    if started_at:
        suffix = f" ({_fmt_duration(latency_ms / 1000.0)})" if latency_ms > 0 else ""
        return f"Started {_fmt_ts(started_at)}{suffix}"
    if ended_at:
        suffix = f" ({_fmt_duration(latency_ms / 1000.0)})" if latency_ms > 0 else ""
        return f"Ended {_fmt_ts(ended_at)}{suffix}"
    return "n/a"


def _parse_iso_dt(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _format_tool_name_list(tool_names: List[str]) -> str:
    if not tool_names:
        return "n/a"
    lines = [f"{index}. {humanize_tool_name(name) or name}" for index, name in enumerate(tool_names, 1)]
    return "\n".join(lines)


def _humanize_failure_cause(value: Dict[str, Any]) -> str:
    cause = str((value or {}).get("primary_cause") or "").strip()
    if not cause:
        return ""
    return " ".join(part.capitalize() for part in cause.split("_"))


def _failure_confidence_label(value: Dict[str, Any]) -> str:
    confidence = str((value or {}).get("confidence") or "").strip()
    return confidence.capitalize() if confidence else "n/a"


def _failure_tone(value: Dict[str, Any]) -> str:
    scope = str((value or {}).get("scope") or "").strip()
    if scope == "runtime_error":
        return "danger"
    if scope == "incorrect":
        return "warning"
    return "neutral"


def _render_agent_steps(sample: SampleView) -> str:
    summary = sample.agent_execution_summary
    steps = summary.get("steps") if isinstance(summary.get("steps"), list) else []
    if steps:
        return _step_list_block(steps)
    error_text = str(summary.get("error") or "").strip()
    if error_text:
        return f'<p class="subtle">Agent step summarization failed: {_escape(error_text)}</p>'
    return '<p class="subtle">No agent step summary was generated. Rebuild the report with <span class="mono">--summarize-agent-steps</span> to add it.</p>'


def _attachments_table(attachments: List[Dict[str, Any]]) -> str:
    if not attachments:
        return '<p class="empty">n/a</p>'
    rows = []
    for item in attachments:
        rows.append(
            "<tr>"
            f"<td>{_escape(_display(item.get('name')))}</td>"
            f"<td class=\"mono\">{_escape(_display(item.get('workspace_path') or item.get('local_path')))}</td>"
            f"<td>{_escape(_display(item.get('source_uri') or item.get('dataset_path')))}</td>"
            "</tr>"
        )
    return '<div class="table-shell"><table><thead><tr><th>Name</th><th>Workspace Path</th><th>Source</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"


def _metadata_table(metadata: Dict[str, Any]) -> str:
    rows: List[Tuple[str, str]] = []
    for key in sorted(metadata.keys()):
        if key == "Annotator Metadata":
            continue
        value = metadata[key]
        rows.append((str(key), _json_block(value) if isinstance(value, (dict, list)) else _escape(_display(value))))
    return _kv_table(rows)


def _usage_panels(sample: SampleView) -> str:
    panels = []
    for title, value in [
        ("Delta", sample.usage_delta),
        ("Last Call", sample.usage_last),
        ("Before", sample.usage_before),
        ("After", sample.usage_after),
    ]:
        panels.append(f'<article class="subpanel"><h3>{_escape(title)}</h3>{_json_block(value)}</article>')
    return '<div class="subpanel-grid">' + "".join(panels) + "</div>"


def _render_failure_analysis(sample: SampleView) -> str:
    failure = sample.failure_analysis if isinstance(sample.failure_analysis, dict) else {}
    if not failure:
        return '<p class="empty">n/a</p>'
    rows = [
        ("Scope", _pill(_display(failure.get("scope")).replace("_", " ").upper(), _failure_tone(failure))),
        ("Primary Cause", _escape(_humanize_failure_cause(failure) or "n/a")),
        ("Confidence", _escape(_failure_confidence_label(failure))),
        ("Source", _escape(_display(failure.get("source")))),
        ("Evidence", _string_list([str(item).strip() for item in failure.get("evidence", []) if str(item).strip()] if isinstance(failure.get("evidence"), list) else [])),
        ("AI Summary", _multiline_value(failure.get("ai_summary"))),
        ("Summary Model", _escape(_display(failure.get("summary_model")))),
        ("Summary Error", _escape(_display(failure.get("summary_error")))),
    ]
    return _kv_table(rows)


def _event_summary(event_record: Dict[str, Any]) -> Tuple[str, str, str]:
    event = event_record.get("event") if isinstance(event_record.get("event"), dict) else {}
    event_type = str(event.get("type") or "unknown")
    if event_type.startswith("tool_execution"):
        tool_name = _display(event.get("toolName"))
        args = event.get("args")
        preview = _truncate(json.dumps(args, sort_keys=True, ensure_ascii=True), 180) if isinstance(args, dict) else "n/a"
        if event_type.endswith("end"):
            result = event.get("result")
            if isinstance(result, dict):
                details = result.get("details")
                if isinstance(details, dict) and details.get("error"):
                    preview = _display(details.get("error"))
        return (event_type.replace("_", " ").title(), f"Tool: {tool_name}", preview)

    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    if event_type.startswith("message"):
        return (
            event_type.replace("_", " ").title(),
            f"Role: {_display(message.get('role'))}",
            _truncate(_message_preview(message) or "n/a", 220),
        )

    if event_type == "turn_end":
        tool_results = event.get("toolResults")
        if isinstance(tool_results, list):
            return ("Turn End", "Aggregated tool results", f"{len(tool_results)} tool result(s)")

    return (event_type.replace("_", " ").title(), "n/a", "n/a")


def _message_preview(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    chunks: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "text":
            text = str(item.get("text") or "").strip()
            if text:
                chunks.append(text)
        elif item_type == "toolCall":
            name = _display(item.get("name"))
            arguments = item.get("arguments")
            preview = json.dumps(arguments, sort_keys=True, ensure_ascii=True) if isinstance(arguments, dict) else "n/a"
            chunks.append(f"toolCall {name}: {_truncate(preview, 140)}")
    return "\n".join(chunks).strip()


def _timeline(sample: SampleView) -> str:
    if not sample.events:
        return '<p class="empty">No event trace was captured for this sample.</p>'
    items = []
    for event_record in sample.events:
        title, meta, preview = _event_summary(event_record)
        items.append(
            '<article class="timeline-item">'
            '<div class="timeline-head">'
            f"{_pill(str((event_record.get('event') or {}).get('type') or 'unknown').upper(), 'neutral')}"
            f"<strong>{_escape(title)}</strong>"
            f'<span class="subtle mono">#{_escape(str(_to_int(event_record.get("event_index"))))}</span>'
            "</div>"
            f'<p class="timeline-meta">{_escape(meta)}</p>'
            f'<p class="timeline-copy">{_escape(preview)}</p>'
            "</article>"
        )
    raw = json.dumps(sample.events, indent=2, sort_keys=True, ensure_ascii=True)
    return '<div class="timeline">' + "".join(items) + f'</div><details class="raw-toggle"><summary>Show raw event JSON</summary><pre>{_escape(raw)}</pre></details>'


def _correlation_matrix_table(run_analysis: Dict[str, Any]) -> str:
    factor_names = [str(item).strip() for item in run_analysis.get("factor_names", []) if str(item).strip()]
    correlations = run_analysis.get("correlations") if isinstance(run_analysis.get("correlations"), list) else []
    lookup: Dict[Tuple[str, str], Any] = {}
    for item in correlations:
        if not isinstance(item, dict):
            continue
        x_name = str(item.get("x") or "").strip()
        y_name = str(item.get("y") or "").strip()
        if not x_name or not y_name:
            continue
        lookup[(x_name, y_name)] = item.get("r")
        lookup[(y_name, x_name)] = item.get("r")

    if not factor_names:
        return '<p class="empty">n/a</p>'

    header = "<tr><th>Factor</th>" + "".join(f"<th>{_escape(name.replace('_', ' ').title())}</th>" for name in factor_names) + "</tr>"
    rows = []
    for x_name in factor_names:
        cells = [f"<td><strong>{_escape(x_name.replace('_', ' ').title())}</strong></td>"]
        for y_name in factor_names:
            if x_name == y_name:
                cells.append(
                    '<td class="mono corr-cell corr-cell--diag" '
                    f'style="{_correlation_cell_style(1.0, diagonal=True)}" '
                    'title="Self-correlation">1.0000</td>'
                )
                continue
            r_value = lookup.get((x_name, y_name))
            title = "Correlation unavailable" if r_value is None else f"r={_fmt_float(r_value, digits=4)}"
            cells.append(
                '<td class="mono corr-cell" '
                f'style="{_correlation_cell_style(r_value)}" '
                f'title="{_escape(title)}">'
                + (_escape(_fmt_float(r_value, digits=4)) if r_value is not None else "n/a")
                + "</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table-shell"><table><thead>' + header + "</thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _correlation_cell_style(r_value: Any, *, diagonal: bool = False) -> str:
    if diagonal:
        return "background:hsla(120, 56%, 74%, 0.30); color:#173b2d; font-weight:700;"
    if r_value is None:
        return "background:rgba(31,41,55,.04); color:#5b6675;"
    try:
        clamped = max(-1.0, min(1.0, float(r_value)))
    except Exception:
        return "background:rgba(31,41,55,.04); color:#5b6675;"
    hue = ((clamped + 1.0) / 2.0) * 120.0
    alpha = 0.10 + (abs(clamped) * 0.24)
    text_color = "#1f2937" if abs(clamped) < 0.58 else ("#173b2d" if clamped > 0 else "#5b2215")
    font_weight = "700" if abs(clamped) >= 0.75 else "600"
    return f"background:hsla({hue:.1f}, 62%, 74%, {alpha:.3f}); color:{text_color}; font-weight:{font_weight};"


def _cohort_table(run_analysis: Dict[str, Any]) -> str:
    cohorts = run_analysis.get("cohorts") if isinstance(run_analysis.get("cohorts"), dict) else {}
    if not cohorts:
        return '<p class="empty">n/a</p>'
    rows = []
    for key in ("correct", "incorrect", "runtime_error"):
        item = cohorts.get(key) if isinstance(cohorts.get(key), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{_escape(key.replace('_', ' ').title())}</td>"
            f"<td>{_fmt_int(item.get('count'))}</td>"
            f"<td>{_fmt_float(item.get('avg_total_tokens'))}</td>"
            f"<td>{_fmt_float(item.get('median_total_tokens'))}</td>"
            f"<td>{_fmt_float(item.get('avg_latency_ms'), suffix=' ms')}</td>"
            f"<td>{_fmt_float(item.get('median_latency_ms'), suffix=' ms')}</td>"
            f"<td>{_fmt_float(item.get('avg_call_count'))}</td>"
            f"<td>{_fmt_float(item.get('avg_distinct_tool_count'))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-shell"><table><thead><tr><th>Cohort</th><th>Count</th><th>Avg Tokens</th><th>Median Tokens</th>'
        "<th>Avg Latency</th><th>Median Latency</th><th>Avg Calls</th><th>Avg Distinct Tools</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _failure_causes_table(run_analysis: Dict[str, Any]) -> str:
    failure_causes = run_analysis.get("failure_causes") if isinstance(run_analysis.get("failure_causes"), dict) else {}
    rows = []
    for scope in ("runtime_error", "incorrect"):
        for item in failure_causes.get(scope, []) if isinstance(failure_causes.get(scope), list) else []:
            if not isinstance(item, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{_escape(scope.replace('_', ' ').title())}</td>"
                f"<td>{_escape(_humanize_failure_cause({'primary_cause': item.get('cause')}) or 'n/a')}</td>"
                f"<td>{_fmt_int(item.get('count'))}</td>"
                "</tr>"
            )
    if not rows:
        return '<p class="empty">n/a</p>'
    return (
        '<div class="table-shell"><table><thead><tr><th>Scope</th><th>Primary Cause</th><th>Count</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _render_doc(title: str, body: str, *, include_chart_js: bool = False, script: str = "") -> str:
    chart_head = '  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n' if include_chart_js else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape(title)}</title>
{BASE_STYLE}{chart_head}</head>
<body>
{body}
{script}
</body>
</html>"""


def _chart_script(
    samples: List[SampleView],
    levels: List[str],
    by_level: Dict[str, Dict[str, Any]],
    *,
    include_level_breakdown: bool,
) -> str:
    def scatter_points(x_attr: str, y_attr: str, *, kind: str) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        for sample in samples:
            sample_kind = "runtime_error" if sample.status == "error" else ("correct" if sample.correct else "incorrect")
            if sample_kind != kind:
                continue
            points.append(
                {
                    "x": getattr(sample, x_attr),
                    "y": getattr(sample, y_attr),
                    "sample_id": sample.sample_id,
                }
            )
        return points

    accuracy = [round(by_level[level]["accuracy"] * 100.0, 2) for level in levels]
    latency = [round(by_level[level]["avg_latency_ms"], 2) for level in levels]
    tokens = [round(by_level[level]["avg_tokens"], 2) for level in levels]
    correct = [by_level[level]["correct"] for level in levels]
    errors = [by_level[level]["error_count"] for level in levels]
    incorrect = [max(by_level[level]["total"] - by_level[level]["correct"] - by_level[level]["error_count"], 0) for level in levels]
    token_latency = {
        "correct": scatter_points("total_tokens", "latency_ms", kind="correct"),
        "incorrect": scatter_points("total_tokens", "latency_ms", kind="incorrect"),
        "runtime_error": scatter_points("total_tokens", "latency_ms", kind="runtime_error"),
    }
    calls_tokens = {
        "correct": scatter_points("call_count", "total_tokens", kind="correct"),
        "incorrect": scatter_points("call_count", "total_tokens", kind="incorrect"),
        "runtime_error": scatter_points("call_count", "total_tokens", kind="runtime_error"),
    }
    template = Template(
        """
<script>
  (function() {
    if (typeof Chart === "undefined") { document.documentElement.classList.add("charts-unavailable"); return; }
    const css = getComputedStyle(document.documentElement);
    const colors = { ink: css.getPropertyValue("--ink").trim() || "#1f2937", muted: css.getPropertyValue("--muted").trim() || "#5b6675", line: css.getPropertyValue("--line").trim() || "rgba(31,41,55,.10)", teal: css.getPropertyValue("--teal").trim() || "#1c7c7d", mint: css.getPropertyValue("--mint").trim() || "#2d8c68", blue: css.getPropertyValue("--blue").trim() || "#356ea9", gold: css.getPropertyValue("--gold").trim() || "#c48824", coral: css.getPropertyValue("--coral").trim() || "#c65b39", sans: css.getPropertyValue("--sans").trim() || "sans-serif" };
    const labels = ${labels}, accuracy = ${accuracy}, latency = ${latency}, tokens = ${tokens}, correct = ${correct}, incorrect = ${incorrect}, errors = ${errors};
    const tokenLatency = ${token_latency}, callsTokens = ${calls_tokens};
    const intFmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }), compactFmt = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
    Chart.defaults.color = colors.muted; Chart.defaults.font.family = colors.sans;
    const plugins = { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 10, color: colors.ink } }, tooltip: { backgroundColor: "rgba(28,33,39,.92)", titleColor: "#fff", bodyColor: "#fff", padding: 12 } };
    const makeScatter = (id, datasets, xLabel, yLabel) => {
      const node = document.getElementById(id);
      if (!node) { return; }
      new Chart(node.getContext("2d"), {
        type: "scatter",
        data: {
          datasets: datasets,
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            ...plugins,
            tooltip: {
              ...plugins.tooltip,
              callbacks: {
                label: ctx => {
                  const point = ctx.raw || {};
                  return (ctx.dataset.label || "") + ": " + (point.sample_id || "") + " (" + compactFmt.format(point.x || 0) + ", " + compactFmt.format(point.y || 0) + ")";
                },
              },
            },
          },
          scales: {
            x: { title: { display: true, text: xLabel }, ticks: { color: colors.muted, callback: v => compactFmt.format(v) }, grid: { color: colors.line } },
            y: { title: { display: true, text: yLabel }, ticks: { color: colors.muted, callback: v => compactFmt.format(v) }, grid: { color: colors.line } },
          },
        },
      });
    };
    makeScatter("tokenLatencyChart", [
      { label: "Correct", data: tokenLatency.correct, backgroundColor: "rgba(45,140,104,.70)", borderColor: colors.mint, pointRadius: 5 },
      { label: "Incorrect", data: tokenLatency.incorrect, backgroundColor: "rgba(196,136,36,.75)", borderColor: colors.gold, pointRadius: 5 },
      { label: "Runtime Error", data: tokenLatency.runtime_error, backgroundColor: "rgba(198,91,57,.75)", borderColor: colors.coral, pointRadius: 6 },
    ], "Total Tokens", "Latency (ms)");
    makeScatter("callsTokensChart", [
      { label: "Correct", data: callsTokens.correct, backgroundColor: "rgba(45,140,104,.70)", borderColor: colors.mint, pointRadius: 5 },
      { label: "Incorrect", data: callsTokens.incorrect, backgroundColor: "rgba(196,136,36,.75)", borderColor: colors.gold, pointRadius: 5 },
      { label: "Runtime Error", data: callsTokens.runtime_error, backgroundColor: "rgba(198,91,57,.75)", borderColor: colors.coral, pointRadius: 6 },
    ], "Call Count", "Total Tokens");
    if (${include_level_breakdown}) {
      new Chart(document.getElementById("levelOutcomeChart").getContext("2d"), { type: "bar", data: { labels, datasets: [{ label: "Correct", data: correct, backgroundColor: "rgba(45,140,104,.14)", borderColor: colors.mint, borderWidth: 1, stack: "count", borderRadius: 10, maxBarThickness: 42 }, { label: "Incorrect", data: incorrect, backgroundColor: "rgba(196,136,36,.14)", borderColor: colors.gold, borderWidth: 1, stack: "count", borderRadius: 10, maxBarThickness: 42 }, { label: "Errors", data: errors, backgroundColor: "rgba(198,91,57,.14)", borderColor: colors.coral, borderWidth: 1, stack: "count", borderRadius: 10, maxBarThickness: 42 }, { label: "Accuracy", type: "line", data: accuracy, yAxisID: "accuracy", borderColor: colors.teal, backgroundColor: colors.teal, borderWidth: 3, tension: .34, pointRadius: 4, pointHoverRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins, scales: { x: { stacked: true, grid: { display: false }, ticks: { color: colors.muted } }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0, color: colors.muted, callback: v => intFmt.format(v) }, grid: { color: colors.line } }, accuracy: { position: "right", beginAtZero: true, max: 100, grid: { drawOnChartArea: false }, ticks: { color: colors.muted, callback: v => v + "%" } } } } });
      new Chart(document.getElementById("levelLatencyChart").getContext("2d"), { type: "line", data: { labels, datasets: [{ label: "Avg Latency (ms)", data: latency, fill: true, borderColor: colors.blue, backgroundColor: "rgba(53,110,169,.14)", borderWidth: 3, tension: .35, pointRadius: 4, pointHoverRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins, scales: { x: { grid: { display: false }, ticks: { color: colors.muted } }, y: { beginAtZero: true, ticks: { color: colors.muted, callback: v => compactFmt.format(v) + " ms" }, grid: { color: colors.line } } } } });
      new Chart(document.getElementById("levelTokenChart").getContext("2d"), { type: "line", data: { labels, datasets: [{ label: "Avg Tokens", data: tokens, fill: true, borderColor: colors.gold, backgroundColor: "rgba(196,136,36,.14)", borderWidth: 3, tension: .35, pointRadius: 4, pointHoverRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins, scales: { x: { grid: { display: false }, ticks: { color: colors.muted } }, y: { beginAtZero: true, ticks: { color: colors.muted, callback: v => compactFmt.format(v) }, grid: { color: colors.line } } } } });
    }
  })();
</script>
"""
    )
    return template.substitute(
        labels=json.dumps(levels),
        accuracy=json.dumps(accuracy),
        latency=json.dumps(latency),
        tokens=json.dumps(tokens),
        correct=json.dumps(correct),
        incorrect=json.dumps(incorrect),
        errors=json.dumps(errors),
        token_latency=json.dumps(token_latency),
        calls_tokens=json.dumps(calls_tokens),
        include_level_breakdown=json.dumps(bool(include_level_breakdown)),
    )


def _render_summary_html(
    summary: Dict[str, Any],
    manifest: Dict[str, Any],
    overall: Dict[str, Any],
    by_level: Dict[str, Dict[str, Any]],
    run_analysis: Dict[str, Any],
    samples: List[SampleView],
    title: str,
) -> str:
    levels = [value for value in by_level.keys() if str(value or "").strip() and str(value).strip().lower() not in {"unknown", "n/a", "none"}]
    try:
        levels.sort(key=lambda value: (int(value) if str(value).isdigit() else float("inf"), value))
    except Exception:
        levels.sort()
    include_level_breakdown = bool(levels)

    level_rows = []
    for level in levels:
        metrics = by_level[level]
        incorrect = max(metrics["total"] - metrics["correct"] - metrics["error_count"], 0)
        pct = metrics["accuracy"] * 100.0
        level_rows.append(
            "<tr>"
            f'<td><span class="level">Level {_escape(level)}</span></td>'
            f"<td>{_fmt_int(metrics['total'])}</td>"
            f"<td>{_fmt_int(metrics['correct'])}</td>"
            f"<td>{_fmt_int(incorrect)}</td>"
            f"<td>{_fmt_int(metrics['error_count'])}</td>"
            f'<td class="accuracy-cell"><div class="meter"><span style="width:{pct:.1f}%"></span></div><strong>{_fmt_pct(metrics["accuracy"])}</strong></td>'
            f"<td>{_fmt_float(metrics['avg_latency_ms'], suffix=' ms')}</td>"
            f"<td>{_fmt_float(metrics['avg_tokens'])}</td>"
            "</tr>"
        )

    sample_rows = []
    for sample in samples:
        row_class = "error-row" if sample.status == "error" else ("correct-row" if sample.correct else "incorrect-row")
        note_html = f'<span class="error-text" title="{_escape(sample.note_text)}">{_escape(_truncate(sample.note_text))}</span>' if sample.note_text else '<span class="subtle">n/a</span>'
        cause_html = _escape(_humanize_failure_cause(sample.failure_analysis) or "n/a")
        sample_rows.append(
            f'<tr class="{row_class}">'
            f"<td>{_fmt_int(sample.index)}</td>"
            f'<td class="mono sample-id"><a class="sample-link" href="{_escape(sample.href)}">{_escape(sample.sample_id)}</a></td>'
            f'<td><span class="level level--soft">L{_escape(sample.level)}</span></td>'
            f"<td>{_pill(sample.status.upper(), sample.status_tone)}</td>"
            f"<td>{_pill(sample.outcome_label, sample.outcome_tone)}</td>"
            f'<td class="mono">{_escape(_display(sample.score_reason))}</td>'
            f'<td class="mono">{_fmt_int(sample.call_count)}</td>'
            f'<td class="mono model-cell">{_escape(_display(sample.provider_model))}</td>'
            f'<td class="mono">{_fmt_float(sample.latency_ms, suffix=" ms")}</td>'
            f'<td class="mono">{_fmt_int(sample.total_tokens)}</td>'
            f"<td>{cause_html}</td>"
            f"<td>{note_html}</td>"
            "</tr>"
        )

    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    model_hint = _manifest_model_hint(manifest)
    chips = "".join(
        [
            _chip("Benchmark", _display(summary.get("benchmark"))),
            _chip("Split", _display(summary.get("split"))),
            _chip("Run ID", _display(summary.get("run_id"))),
            _chip("Executor", _display(config.get("executor"))),
            _chip("Model", _display(" / ".join(part for part in model_hint if part))),
            _chip("Started", _fmt_ts(summary.get("started_at"))),
            _chip("Ended", _fmt_ts(summary.get("ended_at"))),
            _chip("Git SHA", _display(manifest.get("git_sha"))),
        ]
    )
    completed = max(overall["total"] - overall["error_count"], 0)
    incorrect = max(completed - overall["correct"], 0)
    cards = "".join(
        [
            _card("Total Samples", _fmt_int(overall["total"]), f"{_fmt_int(completed)} completed without runtime error", "teal"),
            _card("Correct Answers", _fmt_int(overall["correct"]), f"{_fmt_int(incorrect)} completed but incorrect", "mint"),
            _card("Runtime Errors", _fmt_int(overall["error_count"]), f"{_fmt_pct(overall['error_count'] / overall['total']) if overall['total'] else '0.0%'} of all samples", "coral"),
            _card("Average Latency", _fmt_float(overall["avg_latency_ms"], suffix=" ms"), f"Median {_fmt_float(overall['median_latency_ms'])} / Max {_fmt_float(overall['max_latency_ms'])}", "blue"),
            _card("Average Tokens", _fmt_float(overall["avg_tokens"]), f"Median {_fmt_float(overall['median_tokens'])} / Max {_fmt_float(overall['max_tokens'])}", "gold"),
            _card("Average Calls", _fmt_float(overall["avg_calls"]), f"Avg distinct tools {_fmt_float(overall['avg_tools'])} per sample", "slate"),
        ]
    )
    accuracy_pct = overall["accuracy"] * 100.0
    split_text = _display(summary.get("split"))
    cohorts = run_analysis.get("cohorts") if isinstance(run_analysis.get("cohorts"), dict) else {}
    failure_causes = run_analysis.get("failure_causes") if isinstance(run_analysis.get("failure_causes"), dict) else {}
    failure_cards = "".join(
        [
            _card("Non-Correct Samples", _fmt_int(failure_causes.get("total_non_correct")), "All incorrect or runtime-error samples", "gold"),
            _card("Incorrect", _fmt_int((cohorts.get("incorrect") or {}).get("count")), "Completed runs with wrong final answers", "warning"),
            _card("Runtime Errors", _fmt_int((cohorts.get("runtime_error") or {}).get("count")), "Runs that ended with status=error", "danger"),
        ]
    )
    body = f"""
  <div class="page">
    <section class="panel hero">
      <div class="hero-grid">
        <div>
          <p class="eyebrow">POP-Agent Eval Report</p>
          <h1 class="hero-title">{_escape(title)}</h1>
          <p class="hero-copy">{_escape(f"{split_text.title()} split / {_fmt_int(overall['total'])} samples / {_fmt_duration(summary.get('duration_s', 0))} elapsed.")}</p>
          <div class="chip-grid">{chips}</div>
        </div>
        <div class="score" style="background:conic-gradient(var(--teal) {accuracy_pct:.1f}%, rgba(28,124,125,.14) 0);"><div class="score-inner"><div><span class="score-kicker">Accuracy</span><strong class="score-value">{_escape(_fmt_pct(overall["accuracy"]))}</strong><span class="score-note">{_escape(f"{_fmt_int(overall['correct'])} of {_fmt_int(overall['total'])} samples answered correctly")}</span></div></div></div>
      </div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Run Snapshot</h2><p class="section-copy">Core metrics, runtime cost, and stability signals for the selected run.</p></div></div>
      <div class="metric-grid">{cards}</div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Correlation Overview</h2><p class="section-copy">These are correlations, not causation. The matrix and scatter plots show how correctness, runtime errors, tokens, latency, calls, and tool breadth move together.</p></div></div>
      <article class="subpanel"><h3>Correlation Matrix</h3>{_correlation_matrix_table(run_analysis)}</article>
      <div class="chart-grid" style="grid-template-columns:repeat(2, minmax(0,1fr)); margin-top:16px">
        <article class="chart"><h3>Total Tokens vs Latency</h3><p class="chart-copy">Each point is a sample, colored by outcome.</p><div class="chart-frame"><canvas id="tokenLatencyChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
        <article class="chart"><h3>Calls vs Total Tokens</h3><p class="chart-copy">Shows how tool-call volume tracks overall token usage.</p><div class="chart-frame"><canvas id="callsTokensChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
      </div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Outcome Cohorts</h2><p class="section-copy">Averages and medians for correct, incorrect, and runtime-error samples.</p></div></div>
      {_cohort_table(run_analysis)}
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Failure Causes</h2><p class="section-copy">Primary cause counts for all non-correct samples, combining runtime failures and completed-but-wrong answers.</p></div></div>
      <div class="metric-grid">{failure_cards}</div>
      <div style="margin-top:18px">{_failure_causes_table(run_analysis)}</div>
    </section>
    {(
        f'''<section class="panel">
      <div class="section-head"><div><h2>Level Breakdown</h2><p class="section-copy">Secondary benchmark-specific view retained when level metadata is available.</p></div></div>
      <div class="chart-grid">
        <article class="chart"><h3>Outcome Mix</h3><p class="chart-copy">Correct, incorrect, and runtime-error counts with accuracy overlaid.</p><div class="chart-frame"><canvas id="levelOutcomeChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
        <article class="chart"><h3>Latency Profile</h3><p class="chart-copy">Average latency in milliseconds per level.</p><div class="chart-frame"><canvas id="levelLatencyChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
        <article class="chart"><h3>Token Profile</h3><p class="chart-copy">Average token use per level.</p><div class="chart-frame"><canvas id="levelTokenChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
      </div>
      <div class="table-shell" style="margin-top:18px"><table><thead><tr><th>Level</th><th>Total</th><th>Correct</th><th>Incorrect</th><th>Errors</th><th>Accuracy</th><th>Avg Latency</th><th>Avg Tokens</th></tr></thead><tbody>{"".join(level_rows)}</tbody></table></div>
    </section>'''
        if include_level_breakdown
        else ""
    )}
    <section class="panel">
      <div class="section-head"><div><h2>Sample Results</h2><p class="section-copy">{_escape(f"{_fmt_int(len(samples))} samples listed. Click a sample ID to open the detailed page with prompt, score details, usage, metadata, and the raw event trace toggle.")}</p></div></div>
      <div class="table-shell" style="margin-top:18px"><table><thead><tr><th>#</th><th>Sample ID</th><th>Level</th><th>Status</th><th>Outcome</th><th>Score Reason</th><th>Calls</th><th>Provider / Model</th><th>Latency</th><th>Total Tokens</th><th>Failure Cause</th><th>Warnings / Error</th></tr></thead><tbody>{"".join(sample_rows)}</tbody></table></div>
      <p class="subtle" style="margin-top:16px">Generated from <span class="mono">summary.json</span>, <span class="mono">samples.jsonl</span>, and optional manifest, events, and error artifacts.</p>
    </section>
  </div>
"""
    return _render_doc(
        title,
        body,
        include_chart_js=True,
        script=_chart_script(samples, levels, by_level, include_level_breakdown=include_level_breakdown),
    )


def _render_sample_html(summary: Dict[str, Any], manifest: Dict[str, Any], sample: SampleView, previous_sample: Optional[SampleView], next_sample: Optional[SampleView], back_href: str) -> str:
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    nav = [f'<a class="nav-link" href="{_escape(back_href)}">Back to summary</a>']
    if previous_sample is not None:
        nav.append(f'<a class="nav-link" href="{_escape(previous_sample.filename)}">Previous sample</a>')
    if next_sample is not None:
        nav.append(f'<a class="nav-link" href="{_escape(next_sample.filename)}">Next sample</a>')
    chips = "".join(
        [
            _chip("Sample ID", sample.sample_id),
            _chip("Benchmark", _display(summary.get("benchmark"))),
            _chip("Split", _display(summary.get("split"))),
            _chip("Run ID", _display(summary.get("run_id"))),
            _chip("Started", _fmt_ts(summary.get("started_at"))),
            _chip("Ended", _fmt_ts(summary.get("ended_at"))),
        ]
    )
    snapshot_cards = "".join(
        [
            _card("Outcome", sample.outcome_label, f"Status {_display(sample.status)}", sample.outcome_tone),
            _card("Score Reason", _display(sample.score_reason), f"Score {_fmt_float(sample.score, digits=2)}", "teal"),
            _card("Latency", _fmt_float(sample.latency_ms, suffix=" ms"), f"Trace {_display(sample.trace_ref)}", "blue"),
            _card("Tokens", _fmt_int(sample.total_tokens), f"Calls {_fmt_int(sample.call_count)}", "gold"),
            _card("Provider / Model", _display(sample.provider_model), f"Level {_display(sample.level)}", "slate"),
            _card(
                "Failure Cause",
                _display(_humanize_failure_cause(sample.failure_analysis), fallback="n/a"),
                _display(sample.error or "No runtime error"),
                "coral",
            ),
        ]
    )
    score_rows = _kv_table(
        [
            ("Correct", _pill("YES", "success") if sample.correct else _pill("NO", "warning")),
            ("Status", _pill(sample.status.upper(), sample.status_tone)),
            ("Reason", _escape(_display(sample.score_reason))),
            ("Score", _escape(_fmt_float(sample.score, digits=2))),
            ("Normalized Prediction", _escape(_display(sample.normalized_prediction))),
            ("Normalized Ground Truth", _escape(_display(sample.normalized_ground_truth))),
        ]
    )
    run_rows = _kv_table(
        [
            ("Benchmark", _escape(_display(summary.get("benchmark")))),
            ("Split", _escape(_display(summary.get("split")))),
            ("Run ID", f'<span class="mono">{_escape(_display(summary.get("run_id")))}</span>'),
            ("Executor", _escape(_display(config.get("executor")))),
            ("Started", _escape(_fmt_ts(summary.get("started_at")))),
            ("Ended", _escape(_fmt_ts(summary.get("ended_at")))),
            ("Trace Ref", f'<span class="mono">{_escape(_display(sample.trace_ref))}</span>'),
        ]
    )
    annotator_rows = _kv_table(
        [
            ("Number of steps", _escape(_display(sample.annotator_metadata.get("Number of steps")))),
            ("How long did this take?", _escape(_display(sample.annotator_metadata.get("How long did this take?")))),
            ("Number of tools", _escape(_display(sample.annotator_metadata.get("Number of tools")))),
            ("Tools", _multiline_value(sample.annotator_metadata.get("Tools"))),
        ]
    )
    agent_step_count = len(sample.agent_execution_summary.get("steps", [])) if isinstance(sample.agent_execution_summary.get("steps"), list) else 0
    agent_tool_count = str(len(sample.agent_tool_names)) if (sample.agent_tool_names or sample.events) else "n/a"
    agent_rows = _kv_table(
        [
            ("Number of steps", _escape(str(agent_step_count) if sample.agent_execution_summary else "n/a")),
            ("How long did this take?", _multiline_value(_fmt_sample_time_window(sample.agent_started_at, sample.agent_ended_at, sample.latency_ms))),
            ("Number of tools", _escape(agent_tool_count)),
            ("Tools", _multiline_value(_format_tool_name_list(sample.agent_tool_names))),
        ]
    )
    error_payload = sample.error_record or ({"sample_id": sample.sample_id, "status": sample.status, "error": sample.error, "trace_ref": sample.trace_ref} if sample.error else {})
    body = f"""
  <div class="page">
    <section class="panel hero">
      <div class="nav-row">{"".join(nav)}</div>
      <div class="hero-grid">
        <div>
          <p class="eyebrow">POP-Agent Sample Report</p>
          <h1 class="hero-title">{_escape(sample.sample_id)}</h1>
          <p class="hero-copy">{_escape(f"Sample {sample.index} for run {_display(summary.get('run_id'))}. Detailed view with prompt, scoring, metadata, usage, and curated event timeline.")}</p>
          <div class="chip-grid">{chips}</div>
        </div>
        <div style="display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end">{_pill(sample.status.upper(), sample.status_tone)}{_pill(sample.outcome_label, sample.outcome_tone)}</div>
      </div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Outcome Snapshot</h2><p class="section-copy">Top-level result, scoring status, resource usage, and model details for this sample run.</p></div></div>
      <div class="metric-grid">{snapshot_cards}</div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Inputs And Outputs</h2><p class="section-copy">The exact evaluation prompt, the expected answer, and the model output.</p></div></div>
      <div class="subpanel-grid">
        <article class="subpanel"><h3>Prompt</h3>{_text_block(sample.prompt)}</article>
        <article class="subpanel"><h3>Ground Truth</h3>{_text_block(sample.ground_truth)}</article>
        <article class="subpanel"><h3>Prediction</h3>{_text_block(sample.prediction)}</article>
        <article class="subpanel"><h3>Score Details</h3>{score_rows}</article>
      </div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Run Context</h2><p class="section-copy">Run-level metadata, benchmark metadata, and side-by-side annotator versus agent execution guidance for this sample.</p></div></div>
      <div class="columns">
        <article class="subpanel"><h3>Run Metadata</h3>{run_rows}</article>
        <article class="subpanel"><h3>Sample Metadata</h3>{_metadata_table(sample.metadata)}</article>
        <article class="subpanel"><h3>Annotator Summary</h3>{annotator_rows}</article>
        <article class="subpanel"><h3>Agent Summary</h3>{agent_rows}</article>
        <article class="subpanel"><h3>Annotator Steps</h3>{_step_list_block(sample.annotator_metadata.get("Steps"))}</article>
        <article class="subpanel"><h3>Agent Steps</h3>{_render_agent_steps(sample)}</article>
      </div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Usage And Attachments</h2><p class="section-copy">Per-sample usage snapshots, warnings, and staged attachments captured during execution.</p></div></div>
      <div class="subpanel-grid">
        <article class="subpanel"><h3>Warnings</h3>{_string_list(sample.warnings)}</article>
        <article class="subpanel"><h3>Attachments</h3>{_attachments_table(sample.attachments)}</article>
      </div>
      <div style="margin-top:16px">{_usage_panels(sample)}</div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Failure Analysis</h2><p class="section-copy">Deterministic cause classification for non-correct samples, with optional PromptFunction enrichment when available.</p></div></div>
      {_render_failure_analysis(sample)}
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Error Record</h2><p class="section-copy">The matched record from <span class="mono">errors.jsonl</span> when available, otherwise the runtime error captured on the sample payload.</p></div></div>
      {_json_block(error_payload)}
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Event Timeline</h2><p class="section-copy">Curated event sequence followed by the full raw event stream behind a toggle.</p></div></div>
      {_timeline(sample)}
      <details class="raw-toggle"><summary>Show raw sample payload</summary>{_json_block(sample.raw_sample)}</details>
    </section>
  </div>
"""
    return _render_doc(f"Sample Report - {sample.sample_id}", body)


def _write_report_bundle(
    run_dir: str,
    output: str,
    *,
    summarize_agent_steps: bool = False,
    summary_provider: Optional[str] = None,
    summary_model: Optional[str] = None,
    summarize_failure_causes: bool = False,
) -> str:
    artifacts = _prepare_run_artifacts(
        run_dir,
        summarize_agent_steps=summarize_agent_steps,
        summary_provider=summary_provider,
        summary_model=summary_model,
        summarize_failure_causes=summarize_failure_causes,
    )
    out_path = os.path.abspath(output)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    sample_dir_name = os.path.splitext(os.path.basename(out_path))[0] + "_samples"
    sample_dir_path = os.path.join(out_dir, sample_dir_name)
    if os.path.isdir(sample_dir_path):
        shutil.rmtree(sample_dir_path)
    os.makedirs(sample_dir_path, exist_ok=True)

    sample_views = _build_sample_views(artifacts, sample_dir_name)
    overall, by_level = _aggregate_metrics(sample_views)
    metrics = artifacts.summary.get("metrics") if isinstance(artifacts.summary.get("metrics"), dict) else {}
    run_analysis = metrics.get("analysis") if isinstance(metrics.get("analysis"), dict) else build_run_analysis(artifacts.samples, artifacts.events_by_sample)
    benchmark = _display(artifacts.summary.get("benchmark"))
    run_id = _display(artifacts.summary.get("run_id"))
    summary_html = _render_summary_html(
        artifacts.summary,
        artifacts.manifest,
        overall,
        by_level,
        run_analysis,
        sample_views,
        f"Evaluation Report - {benchmark} ({run_id})",
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(summary_html)

    back_href = os.path.relpath(out_path, sample_dir_path).replace("\\", "/")
    for index, sample in enumerate(sample_views):
        previous_sample = sample_views[index - 1] if index > 0 else None
        next_sample = sample_views[index + 1] if index + 1 < len(sample_views) else None
        sample_html = _render_sample_html(artifacts.summary, artifacts.manifest, sample, previous_sample, next_sample, back_href)
        with open(os.path.join(sample_dir_path, sample.filename), "w", encoding="utf-8") as handle:
            handle.write(sample_html)

    return str(Path(out_path).resolve())


def generate_html_report(
    run_dir: str,
    output: str,
    *,
    summarize_agent_steps: bool = False,
    summary_provider: Optional[str] = None,
    summary_model: Optional[str] = None,
    summarize_failure_causes: bool = False,
) -> str:
    return _write_report_bundle(
        run_dir,
        output,
        summarize_agent_steps=summarize_agent_steps,
        summary_provider=summary_provider,
        summary_model=summary_model,
        summarize_failure_causes=summarize_failure_causes,
    )


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report from POP-Agent evaluation run")
    parser.add_argument("--run-dir", required=True, help="Run directory or zip file containing run results")
    parser.add_argument("--output", required=False, help="Path to output HTML report file")
    parser.add_argument("--summarize-agent-steps", action="store_true", help="Generate and persist PromptFunction-based agent step summaries before building the report.")
    parser.add_argument("--summary-provider", required=False, help="PromptFunction provider override shared by eval summaries.")
    parser.add_argument("--summary-model", required=False, help="PromptFunction model override shared by eval summaries.")
    parser.add_argument("--summarize-failure-causes", action="store_true", help="Generate and persist PromptFunction-based summaries for ambiguous non-correct samples.")
    args = parser.parse_args(argv)
    output_path = args.output or (os.path.splitext(os.path.basename(args.run_dir))[0] + "_report.html")
    out = generate_html_report(
        args.run_dir,
        output_path,
        summarize_agent_steps=bool(args.summarize_agent_steps),
        summary_provider=args.summary_provider,
        summary_model=args.summary_model,
        summarize_failure_causes=bool(args.summarize_failure_causes),
    )
    print(f"HTML report generated at: {out}")


if __name__ == "__main__":
    main()
