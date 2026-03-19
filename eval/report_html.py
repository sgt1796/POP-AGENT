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
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    .accuracy-cell { min-width:180px; } .accuracy-cell strong { display:inline-block; margin-top:8px; } .meter { width:100%; height:10px; border-radius:999px; background:rgba(31,41,55,.08); overflow:hidden; } .meter span { display:block; height:100%; background:linear-gradient(90deg, var(--teal), #34a0a4); }
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
    provider_model: str
    warning_count: int
    error: str
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
    normalized_prediction: Any
    normalized_ground_truth: Any
    raw_sample: Dict[str, Any]


def _read_run_data(run_dir_or_zip: str) -> RunArtifacts:
    source = str(run_dir_or_zip or "").strip()
    if source.lower().endswith(".zip"):
        return _read_run_data_from_zip(source)
    return _read_run_data_from_dir(source)


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
        note_text = error_text or (f"{len(warnings)} warning(s)" if warnings else "")
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
                provider_model=provider_model,
                warning_count=len(warnings),
                error=error_text,
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
                events=list(artifacts.events_by_sample.get(sample_id, [])),
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
    overall = {"total": 0, "correct": 0, "error_count": 0, "latencies": [], "tokens": [], "calls": []}
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
        bucket = by_level.setdefault(
            sample.level,
            {"total": 0, "correct": 0, "error_count": 0, "latencies": [], "tokens": [], "calls": []},
        )
        bucket["total"] += 1
        if sample.correct:
            bucket["correct"] += 1
        if sample.status == "error":
            bucket["error_count"] += 1
        bucket["latencies"].append(sample.latency_ms)
        bucket["tokens"].append(sample.total_tokens)
        bucket["calls"].append(sample.call_count)

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


def _kv_table(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return '<p class="empty">n/a</p>'
    body = "".join(f"<tr><th>{_escape(label)}</th><td>{value}</td></tr>" for label, value in rows)
    return f'<table class="kv-table"><tbody>{body}</tbody></table>'


def _string_list(items: List[str]) -> str:
    if not items:
        return '<p class="empty">n/a</p>'
    return '<ul class="plain-list">' + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


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


def _chart_script(levels: List[str], by_level: Dict[str, Dict[str, Any]]) -> str:
    accuracy = [round(by_level[level]["accuracy"] * 100.0, 2) for level in levels]
    latency = [round(by_level[level]["avg_latency_ms"], 2) for level in levels]
    tokens = [round(by_level[level]["avg_tokens"], 2) for level in levels]
    correct = [by_level[level]["correct"] for level in levels]
    errors = [by_level[level]["error_count"] for level in levels]
    incorrect = [max(by_level[level]["total"] - by_level[level]["correct"] - by_level[level]["error_count"], 0) for level in levels]
    template = Template(
        """
<script>
  (function() {
    if (typeof Chart === "undefined") { document.documentElement.classList.add("charts-unavailable"); return; }
    const css = getComputedStyle(document.documentElement);
    const colors = { ink: css.getPropertyValue("--ink").trim() || "#1f2937", muted: css.getPropertyValue("--muted").trim() || "#5b6675", line: css.getPropertyValue("--line").trim() || "rgba(31,41,55,.10)", teal: css.getPropertyValue("--teal").trim() || "#1c7c7d", mint: css.getPropertyValue("--mint").trim() || "#2d8c68", blue: css.getPropertyValue("--blue").trim() || "#356ea9", gold: css.getPropertyValue("--gold").trim() || "#c48824", coral: css.getPropertyValue("--coral").trim() || "#c65b39", sans: css.getPropertyValue("--sans").trim() || "sans-serif" };
    const labels = ${labels}, accuracy = ${accuracy}, latency = ${latency}, tokens = ${tokens}, correct = ${correct}, incorrect = ${incorrect}, errors = ${errors};
    const intFmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }), compactFmt = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
    Chart.defaults.color = colors.muted; Chart.defaults.font.family = colors.sans;
    const plugins = { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 10, color: colors.ink } }, tooltip: { backgroundColor: "rgba(28,33,39,.92)", titleColor: "#fff", bodyColor: "#fff", padding: 12 } };
    new Chart(document.getElementById("accuracyChart").getContext("2d"), { type: "bar", data: { labels, datasets: [{ label: "Correct", data: correct, backgroundColor: "rgba(45,140,104,.14)", borderColor: colors.mint, borderWidth: 1, stack: "count", borderRadius: 10, maxBarThickness: 42 }, { label: "Incorrect", data: incorrect, backgroundColor: "rgba(196,136,36,.14)", borderColor: colors.gold, borderWidth: 1, stack: "count", borderRadius: 10, maxBarThickness: 42 }, { label: "Errors", data: errors, backgroundColor: "rgba(198,91,57,.14)", borderColor: colors.coral, borderWidth: 1, stack: "count", borderRadius: 10, maxBarThickness: 42 }, { label: "Accuracy", type: "line", data: accuracy, yAxisID: "accuracy", borderColor: colors.teal, backgroundColor: colors.teal, borderWidth: 3, tension: .34, pointRadius: 4, pointHoverRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins, scales: { x: { stacked: true, grid: { display: false }, ticks: { color: colors.muted } }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0, color: colors.muted, callback: v => intFmt.format(v) }, grid: { color: colors.line } }, accuracy: { position: "right", beginAtZero: true, max: 100, grid: { drawOnChartArea: false }, ticks: { color: colors.muted, callback: v => v + "%" } } } } });
    new Chart(document.getElementById("latencyChart").getContext("2d"), { type: "line", data: { labels, datasets: [{ label: "Avg Latency (ms)", data: latency, fill: true, borderColor: colors.blue, backgroundColor: "rgba(53,110,169,.14)", borderWidth: 3, tension: .35, pointRadius: 4, pointHoverRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins, scales: { x: { grid: { display: false }, ticks: { color: colors.muted } }, y: { beginAtZero: true, ticks: { color: colors.muted, callback: v => compactFmt.format(v) + " ms" }, grid: { color: colors.line } } } } });
    new Chart(document.getElementById("tokenChart").getContext("2d"), { type: "line", data: { labels, datasets: [{ label: "Avg Tokens", data: tokens, fill: true, borderColor: colors.gold, backgroundColor: "rgba(196,136,36,.14)", borderWidth: 3, tension: .35, pointRadius: 4, pointHoverRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins, scales: { x: { grid: { display: false }, ticks: { color: colors.muted } }, y: { beginAtZero: true, ticks: { color: colors.muted, callback: v => compactFmt.format(v) }, grid: { color: colors.line } } } } });
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
    )


def _render_summary_html(summary: Dict[str, Any], manifest: Dict[str, Any], overall: Dict[str, Any], by_level: Dict[str, Dict[str, Any]], samples: List[SampleView], title: str) -> str:
    levels = list(by_level.keys())
    try:
        levels.sort(key=lambda value: (int(value) if str(value).isdigit() else float("inf"), value))
    except Exception:
        levels.sort()

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
    warning_samples = sum(1 for sample in samples if sample.warning_count > 0)
    completed = max(overall["total"] - overall["error_count"], 0)
    incorrect = max(completed - overall["correct"], 0)
    cards = "".join(
        [
            _card("Total Samples", _fmt_int(overall["total"]), f"{_fmt_int(completed)} completed without runtime error", "teal"),
            _card("Correct Answers", _fmt_int(overall["correct"]), f"{_fmt_int(incorrect)} completed but incorrect", "mint"),
            _card("Runtime Errors", _fmt_int(overall["error_count"]), f"{_fmt_pct(overall['error_count'] / overall['total']) if overall['total'] else '0.0%'} of all samples", "coral"),
            _card("Average Latency", _fmt_float(overall["avg_latency_ms"], suffix=" ms"), f"Median {_fmt_float(overall['median_latency_ms'])} / Max {_fmt_float(overall['max_latency_ms'])}", "blue"),
            _card("Average Tokens", _fmt_float(overall["avg_tokens"]), f"Median {_fmt_float(overall['median_tokens'])} / Max {_fmt_float(overall['max_tokens'])}", "gold"),
            _card("Warnings Present", _fmt_int(warning_samples), f"Avg calls {_fmt_float(overall['avg_calls'])} per sample", "slate"),
        ]
    )
    accuracy_pct = overall["accuracy"] * 100.0
    split_text = _display(summary.get("split"))
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
      <div class="section-head"><div><h2>Performance By Level</h2><p class="section-copy">Difficulty-level breakdown for outcomes, latency, and token usage.</p></div></div>
      <div class="chart-grid">
        <article class="chart"><h3>Outcome Mix</h3><p class="chart-copy">Correct, incorrect, and runtime-error counts with accuracy overlaid.</p><div class="chart-frame"><canvas id="accuracyChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
        <article class="chart"><h3>Latency Profile</h3><p class="chart-copy">Average latency in milliseconds per level.</p><div class="chart-frame"><canvas id="latencyChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
        <article class="chart"><h3>Token Profile</h3><p class="chart-copy">Average token use per level.</p><div class="chart-frame"><canvas id="tokenChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
      </div>
      <div class="table-shell" style="margin-top:18px"><table><thead><tr><th>Level</th><th>Total</th><th>Correct</th><th>Incorrect</th><th>Errors</th><th>Accuracy</th><th>Avg Latency</th><th>Avg Tokens</th></tr></thead><tbody>{"".join(level_rows)}</tbody></table></div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Sample Results</h2><p class="section-copy">{_escape(f"{_fmt_int(len(samples))} samples listed. Click a sample ID to open the detailed page with prompt, score details, usage, metadata, and the raw event trace toggle.")}</p></div></div>
      <div class="table-shell" style="margin-top:18px"><table><thead><tr><th>#</th><th>Sample ID</th><th>Level</th><th>Status</th><th>Outcome</th><th>Score Reason</th><th>Calls</th><th>Provider / Model</th><th>Latency</th><th>Total Tokens</th><th>Warnings / Error</th></tr></thead><tbody>{"".join(sample_rows)}</tbody></table></div>
      <p class="subtle" style="margin-top:16px">Generated from <span class="mono">summary.json</span>, <span class="mono">samples.jsonl</span>, and optional manifest, events, and error artifacts.</p>
    </section>
  </div>
"""
    return _render_doc(title, body, include_chart_js=True, script=_chart_script(levels, by_level))


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
            _card("Warnings", _fmt_int(sample.warning_count), _display(sample.error or "No runtime error"), "coral"),
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
            ("Tools", _escape(_display(sample.annotator_metadata.get("Tools")))),
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
      <div class="section-head"><div><h2>Run Context</h2><p class="section-copy">Run-level metadata, benchmark metadata, and GAIA annotator guidance carried with the sample.</p></div></div>
      <div class="columns">
        <article class="subpanel"><h3>Run Metadata</h3>{run_rows}</article>
        <article class="subpanel"><h3>Sample Metadata</h3>{_metadata_table(sample.metadata)}</article>
        <article class="subpanel"><h3>Annotator Summary</h3>{annotator_rows}</article>
        <article class="subpanel"><h3>Annotator Steps</h3>{_text_block(sample.annotator_metadata.get("Steps"))}</article>
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


def _write_report_bundle(run_dir: str, output: str) -> str:
    artifacts = _read_run_data(run_dir)
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
    benchmark = _display(artifacts.summary.get("benchmark"))
    run_id = _display(artifacts.summary.get("run_id"))
    summary_html = _render_summary_html(
        artifacts.summary,
        artifacts.manifest,
        overall,
        by_level,
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

    return out_path


def generate_html_report(run_dir: str, output: str) -> str:
    return _write_report_bundle(run_dir, output)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report from POP-Agent evaluation run")
    parser.add_argument("--run-dir", required=True, help="Run directory or zip file containing run results")
    parser.add_argument("--output", required=False, help="Path to output HTML report file")
    args = parser.parse_args(argv)
    output_path = args.output or (os.path.splitext(os.path.basename(args.run_dir))[0] + "_report.html")
    out = generate_html_report(args.run_dir, output_path)
    print(f"HTML report generated at: {out}")


if __name__ == "__main__":
    main()
