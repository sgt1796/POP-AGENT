"""Generate styled HTML reports for POP-Agent eval runs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import zipfile
from dataclasses import dataclass
from datetime import datetime
from html import escape
from string import Template
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class SampleSummary:
    sample_id: str
    level: str
    status: str
    correct: bool
    latency_ms: float
    total_tokens: int
    error: str


def _read_run_data(run_dir_or_zip: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if run_dir_or_zip.lower().endswith(".zip"):
        with zipfile.ZipFile(run_dir_or_zip, "r") as zf:
            prefix = None
            for name in zf.namelist():
                parts = name.split("/")
                if len(parts) > 1 and parts[0]:
                    prefix = parts[0]
                    break
            if prefix is None:
                raise FileNotFoundError(f"Could not determine run directory inside zip: {run_dir_or_zip}")
            summary_data = json.loads(zf.read(f"{prefix}/summary.json"))
            samples = []
            with zf.open(f"{prefix}/samples.jsonl") as f:
                for line in f:
                    try:
                        samples.append(json.loads(line))
                    except Exception:
                        continue
            return summary_data, samples

    summary_path = os.path.join(run_dir_or_zip, "summary.json")
    samples_path = os.path.join(run_dir_or_zip, "samples.jsonl")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json not found in {run_dir_or_zip}")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"samples.jsonl not found in {run_dir_or_zip}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    samples = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except Exception:
                continue
    return summary_data, samples


def _extract_sample_summary(sample_record: Dict[str, Any]) -> SampleSummary:
    result = sample_record.get("result", {})
    metadata = sample_record.get("metadata", {})
    usage = result.get("usage", {})
    delta = usage.get("delta", {}) if isinstance(usage, dict) else {}
    raw_error = result.get("error")
    return SampleSummary(
        sample_id=str(sample_record.get("sample_id", "")),
        level=str(metadata.get("Level", metadata.get("level", "unknown"))),
        status=str(result.get("status", "unknown")),
        correct=bool(result.get("score_result", {}).get("correct", False)),
        latency_ms=float(result.get("latency_ms", 0.0) or 0.0),
        total_tokens=int(delta.get("total_tokens", 0) or 0),
        error="" if raw_error in (None, "") else str(raw_error).strip(),
    )


def _aggregate_metrics(samples: Iterable[SampleSummary]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    overall = {"total": 0, "correct": 0, "error_count": 0, "latencies": [], "tokens": []}
    by_level: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        overall["total"] += 1
        if sample.correct:
            overall["correct"] += 1
        if sample.status == "error":
            overall["error_count"] += 1
        overall["latencies"].append(sample.latency_ms)
        overall["tokens"].append(sample.total_tokens)

        metrics = by_level.setdefault(
            sample.level,
            {"total": 0, "correct": 0, "error_count": 0, "latencies": [], "tokens": []},
        )
        metrics["total"] += 1
        if sample.correct:
            metrics["correct"] += 1
        if sample.status == "error":
            metrics["error_count"] += 1
        metrics["latencies"].append(sample.latency_ms)
        metrics["tokens"].append(sample.total_tokens)

    def derive(metrics: Dict[str, Any]) -> None:
        total = metrics["total"]
        metrics["accuracy"] = metrics["correct"] / total if total else 0.0
        latencies = metrics["latencies"]
        tokens = metrics["tokens"]
        metrics["avg_latency_ms"] = float(statistics.mean(latencies)) if latencies else 0.0
        metrics["median_latency_ms"] = float(statistics.median(latencies)) if latencies else 0.0
        metrics["max_latency_ms"] = float(max(latencies)) if latencies else 0.0
        metrics["avg_tokens"] = float(statistics.mean(tokens)) if tokens else 0.0
        metrics["median_tokens"] = float(statistics.median(tokens)) if tokens else 0.0
        metrics["max_tokens"] = float(max(tokens)) if tokens else 0.0

    derive(overall)
    for metrics in by_level.values():
        derive(metrics)
    return overall, by_level


def _escape(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value or "")


def _fmt_float(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value or "")


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return str(value or "")


def _fmt_duration(seconds: Any) -> str:
    try:
        total_seconds = max(0, int(round(float(seconds or 0))))
    except Exception:
        return str(seconds or "")
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
        return ""
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


def _card(label: str, value: str, meta: str, tone: str) -> str:
    meta_html = f'<p class="metric-meta">{_escape(meta)}</p>' if meta else ""
    return (
        f'<article class="metric metric--{tone}">'
        f'<span class="metric-label">{_escape(label)}</span>'
        f'<strong class="metric-value">{_escape(value)}</strong>'
        f"{meta_html}</article>"
    )


def _chip(label: str, value: str) -> str:
    return (
        '<div class="chip">'
        f'<span class="chip-label">{_escape(label)}</span>'
        f'<strong class="chip-value">{_escape(value)}</strong>'
        "</div>"
    )


def _pill(label: str, tone: str) -> str:
    return f'<span class="pill pill--{tone}">{_escape(label)}</span>'


def _render_html(
    summary: Dict[str, Any],
    overall: Dict[str, Any],
    by_level: Dict[str, Dict[str, Any]],
    sample_summaries: List[SampleSummary],
    title: str,
) -> str:
    levels = list(by_level.keys())
    try:
        levels.sort(key=lambda x: (int(x) if str(x).isdigit() else float("inf"), x))
    except Exception:
        levels.sort()

    accuracy_data = [round(by_level[level]["accuracy"] * 100.0, 2) for level in levels]
    latency_data = [round(by_level[level]["avg_latency_ms"], 2) for level in levels]
    token_data = [round(by_level[level]["avg_tokens"], 2) for level in levels]
    correct_data = [by_level[level]["correct"] for level in levels]
    error_data = [by_level[level]["error_count"] for level in levels]
    incorrect_data = [
        max(by_level[level]["total"] - by_level[level]["correct"] - by_level[level]["error_count"], 0)
        for level in levels
    ]

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
            '<td class="accuracy-cell"><div class="meter">'
            f'<span style="width:{pct:.1f}%"></span></div><strong>{_fmt_pct(metrics["accuracy"])}</strong></td>'
            f"<td>{_fmt_float(metrics['avg_latency_ms'])} ms</td>"
            f"<td>{_fmt_float(metrics['avg_tokens'])}</td>"
            "</tr>"
        )

    sample_rows = []
    for index, sample in enumerate(sample_summaries, 1):
        row_class = "error-row" if sample.status == "error" else ("correct-row" if sample.correct else "incorrect-row")
        status_tone = "danger" if sample.status == "error" else ("info" if sample.status == "ok" else "neutral")
        if sample.status == "error":
            outcome_label, outcome_tone = "Failed", "danger"
        elif sample.correct:
            outcome_label, outcome_tone = "Correct", "success"
        else:
            outcome_label, outcome_tone = "Incorrect", "warning"
        error_html = '<span class="subtle">-</span>'
        if sample.error:
            error_html = (
                f'<span class="error-text" title="{_escape(sample.error)}">'
                f"{_escape(_truncate(sample.error))}</span>"
            )
        sample_rows.append(
            f'<tr class="{row_class}">'
            f"<td>{_fmt_int(index)}</td>"
            f'<td class="mono sample-id">{_escape(sample.sample_id)}</td>'
            f'<td><span class="level level--soft">L{_escape(sample.level)}</span></td>'
            f"<td>{_pill(sample.status.upper(), status_tone)}</td>"
            f"<td>{_pill(outcome_label, outcome_tone)}</td>"
            f'<td class="mono">{_fmt_float(sample.latency_ms)} ms</td>'
            f'<td class="mono">{_fmt_int(sample.total_tokens)}</td>'
            f"<td>{error_html}</td>"
            "</tr>"
        )

    benchmark = str(summary.get("benchmark") or "unknown")
    split = str(summary.get("split") or "n/a")
    started = _fmt_ts(summary.get("started_at"))
    ended = _fmt_ts(summary.get("ended_at"))
    duration = _fmt_duration(summary.get("duration_s", 0))
    accuracy_pct = overall["accuracy"] * 100.0
    completed = max(overall["total"] - overall["error_count"], 0)
    incorrect = max(completed - overall["correct"], 0)

    cards = "".join(
        [
            _card("Total Samples", _fmt_int(overall["total"]), f"{_fmt_int(completed)} completed without runtime error", "teal"),
            _card("Correct Answers", _fmt_int(overall["correct"]), f"{_fmt_int(incorrect)} completed but incorrect", "mint"),
            _card("Runtime Errors", _fmt_int(overall["error_count"]), f"{_fmt_pct(overall['error_count'] / overall['total']) if overall['total'] else '0.0%'} of all samples", "coral"),
            _card("Average Latency", f"{_fmt_float(overall['avg_latency_ms'])} ms", f"Median {_fmt_float(overall['median_latency_ms'])} / Max {_fmt_float(overall['max_latency_ms'])}", "blue"),
            _card("Average Tokens", _fmt_float(overall["avg_tokens"]), f"Median {_fmt_float(overall['median_tokens'])} / Max {_fmt_float(overall['max_tokens'])}", "gold"),
            _card("Run Duration", duration, f"{started} to {ended}" if started and ended else "", "slate"),
        ]
    )
    chips = "".join(
        [
            _chip("Benchmark", benchmark),
            _chip("Split", split),
            _chip("Run ID", str(summary.get("run_id") or "n/a")),
            _chip("Started", started or "n/a"),
            _chip("Ended", ended or "n/a"),
        ]
    )

    template = Template(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${page_title}</title>
  <style>
    :root { --bg:#f4eee3; --panel:rgba(255,251,245,.86); --ink:#1f2937; --muted:#5b6675; --line:rgba(31,41,55,.10); --shadow:0 18px 48px rgba(43,56,73,.12); --teal:#1c7c7d; --mint:#2d8c68; --blue:#356ea9; --gold:#c48824; --coral:#c65b39; --slate:#6a7280; --sans:"Aptos","Segoe UI Variable","Segoe UI",sans-serif; --mono:"Cascadia Code","Consolas",monospace; }
    * { box-sizing:border-box; } body { margin:0; color:var(--ink); font-family:var(--sans); background:radial-gradient(circle at top left, rgba(255,246,225,.95), transparent 36%), radial-gradient(circle at top right, rgba(213,235,231,.85), transparent 30%), linear-gradient(180deg, #f4eee2 0%, #f7f3ea 44%, #f2ecdf 100%); } h1,h2,h3,p { margin:0; }
    .page { width:min(1400px, calc(100% - 32px)); margin:24px auto 40px; } .panel,.table-shell { border:1px solid var(--line); border-radius:24px; background:var(--panel); box-shadow:var(--shadow); backdrop-filter:blur(16px); } .panel { padding:24px; margin-top:20px; }
    .hero { display:grid; grid-template-columns:minmax(0,1fr) 240px; gap:24px; padding:28px; } .eyebrow { margin-bottom:10px; font-size:.82rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--teal); } .hero-title { font-size:clamp(2rem, 4vw, 3.2rem); line-height:.98; letter-spacing:-.04em; margin-bottom:14px; max-width:12ch; } .hero-copy { color:var(--muted); line-height:1.6; margin-bottom:18px; max-width:70ch; }
    .chip-grid { display:flex; flex-wrap:wrap; gap:12px; } .chip { min-width:140px; padding:12px 14px; border-radius:16px; border:1px solid rgba(31,41,55,.08); background:rgba(255,255,255,.68); } .chip-label { display:block; color:var(--muted); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:6px; } .chip-value { display:block; font-size:.95rem; line-height:1.35; word-break:break-word; }
    .score { --progress:${score_progress}; width:220px; height:220px; border-radius:50%; padding:15px; background:conic-gradient(var(--teal) calc(var(--progress) * 1%), rgba(28,124,125,.14) 0); } .score-inner { width:100%; height:100%; border-radius:50%; display:flex; align-items:center; justify-content:center; text-align:center; padding:20px; background:rgba(255,251,245,.96); border:1px solid rgba(31,41,55,.06); } .score-kicker { display:block; color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; } .score-value { display:block; font-size:2.5rem; line-height:1; letter-spacing:-.05em; margin-bottom:10px; } .score-note { display:block; color:var(--muted); font-size:.92rem; line-height:1.4; }
    .section-head { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:16px; } .section-copy { color:var(--muted); line-height:1.6; max-width:70ch; } .metric-grid, .chart-grid { display:grid; gap:16px; } .metric-grid { grid-template-columns:repeat(3, minmax(0,1fr)); } .chart-grid { grid-template-columns:1.4fr 1fr 1fr; }
    .metric { position:relative; padding:18px; border-radius:20px; border:1px solid rgba(31,41,55,.08); background:rgba(255,255,255,.72); overflow:hidden; } .metric::before { content:""; position:absolute; inset:0 auto auto 0; width:100%; height:4px; background:var(--tone); } .metric--teal { --tone:var(--teal); } .metric--mint { --tone:var(--mint); } .metric--blue { --tone:var(--blue); } .metric--gold { --tone:var(--gold); } .metric--coral { --tone:var(--coral); } .metric--slate { --tone:var(--slate); } .metric-label { display:block; color:var(--muted); font-size:.8rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:12px; } .metric-value { display:block; font-size:clamp(1.5rem, 2.6vw, 2.1rem); line-height:1.05; letter-spacing:-.04em; margin-bottom:10px; } .metric-meta { color:var(--muted); font-size:.94rem; line-height:1.5; }
    .chart { border:1px solid rgba(31,41,55,.08); border-radius:20px; padding:18px; background:rgba(255,255,255,.72); } .chart-copy { color:var(--muted); font-size:.92rem; line-height:1.5; margin:6px 0 14px; } .chart-frame { height:300px; } .chart-empty { display:none; color:var(--muted); font-size:.92rem; margin-top:12px; }
    .table-shell { overflow:auto; margin-top:18px; background:rgba(255,255,255,.80); } table { width:100%; min-width:900px; border-collapse:collapse; } thead th { position:sticky; top:0; z-index:1; background:#f5efe7; color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; text-align:left; padding:14px 16px; border-bottom:1px solid var(--line); } tbody td { padding:14px 16px; border-top:1px solid rgba(31,41,55,.07); vertical-align:top; } tbody tr:hover { background:rgba(28,124,125,.05); }
    .error-row { background:rgba(198,91,57,.08); } .correct-row { background:rgba(45,140,104,.06); } .incorrect-row { background:rgba(196,136,36,.08); } .level,.pill { display:inline-flex; align-items:center; justify-content:center; min-height:30px; padding:0 12px; border-radius:999px; font-size:.82rem; font-weight:700; white-space:nowrap; } .level { background:rgba(28,124,125,.12); color:var(--teal); } .level--soft { background:rgba(31,41,55,.06); color:var(--ink); } .pill--success { background:rgba(45,140,104,.14); color:var(--mint); } .pill--warning { background:rgba(196,136,36,.14); color:#9a670d; } .pill--danger { background:rgba(198,91,57,.14); color:var(--coral); } .pill--info { background:rgba(53,110,169,.14); color:var(--blue); } .pill--neutral { background:rgba(31,41,55,.08); color:var(--slate); }
    .accuracy-cell { min-width:180px; } .accuracy-cell strong { display:inline-block; margin-top:8px; } .meter { width:100%; height:10px; border-radius:999px; background:rgba(31,41,55,.08); overflow:hidden; } .meter span { display:block; height:100%; background:linear-gradient(90deg, var(--teal), #34a0a4); } .mono { font-family:var(--mono); font-size:.9rem; } .sample-id { max-width:280px; word-break:break-all; } .subtle { color:var(--muted); } .error-text { display:inline-block; max-width:420px; line-height:1.5; } .footer { margin-top:16px; color:var(--muted); font-size:.92rem; }
    .charts-unavailable .chart-empty { display:block; } .charts-unavailable canvas { display:none !important; }
    @media (max-width:1100px) { .hero { grid-template-columns:1fr; } .metric-grid,.chart-grid { grid-template-columns:1fr 1fr; } .chart-grid .chart:first-child { grid-column:1 / -1; } } @media (max-width:720px) { .page { width:min(100% - 16px, 1400px); margin:16px auto 24px; } .hero,.panel { padding:20px; } .hero-title { max-width:none; } .score { width:190px; height:190px; } .metric-grid,.chart-grid { grid-template-columns:1fr; } .section-head { flex-direction:column; align-items:flex-start; } }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <div class="page">
    <section class="panel hero">
      <div>
        <p class="eyebrow">POP-Agent Eval Report</p>
        <h1 class="hero-title">${hero_title}</h1>
        <p class="hero-copy">${hero_copy}</p>
        <div class="chip-grid">${chips}</div>
      </div>
      <div class="score"><div class="score-inner"><div><span class="score-kicker">Accuracy</span><strong class="score-value">${score_value}</strong><span class="score-note">${score_note}</span></div></div></div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Run Snapshot</h2><p class="section-copy">Core metrics, runtime cost, and stability signals for the selected run.</p></div></div>
      <div class="metric-grid">${cards}</div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Performance By Level</h2><p class="section-copy">Difficulty-level breakdown for outcomes, latency, and token usage.</p></div></div>
      <div class="chart-grid">
        <article class="chart"><h3>Outcome Mix</h3><p class="chart-copy">Correct, incorrect, and runtime-error counts with accuracy overlaid.</p><div class="chart-frame"><canvas id="accuracyChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
        <article class="chart"><h3>Latency Profile</h3><p class="chart-copy">Average latency in milliseconds per level.</p><div class="chart-frame"><canvas id="latencyChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
        <article class="chart"><h3>Token Profile</h3><p class="chart-copy">Average token use per level.</p><div class="chart-frame"><canvas id="tokenChart"></canvas></div><p class="chart-empty">Charts are unavailable because Chart.js did not load.</p></article>
      </div>
      <div class="table-shell"><table><thead><tr><th>Level</th><th>Total</th><th>Correct</th><th>Incorrect</th><th>Errors</th><th>Accuracy</th><th>Avg Latency</th><th>Avg Tokens</th></tr></thead><tbody>${level_rows}</tbody></table></div>
    </section>
    <section class="panel">
      <div class="section-head"><div><h2>Sample Results</h2><p class="section-copy">${sample_copy}</p></div></div>
      <div class="table-shell"><table><thead><tr><th>#</th><th>Sample ID</th><th>Level</th><th>Status</th><th>Outcome</th><th>Latency</th><th>Total Tokens</th><th>Error</th></tr></thead><tbody>${sample_rows}</tbody></table></div>
      <p class="footer">Generated from <span class="mono">summary.json</span> and <span class="mono">samples.jsonl</span> in the selected run directory.</p>
    </section>
  </div>
  <script>
    (function() {
      if (typeof Chart === "undefined") { document.documentElement.classList.add("charts-unavailable"); return; }
      const css = getComputedStyle(document.documentElement);
      const colors = { ink: css.getPropertyValue("--ink").trim() || "#1f2937", muted: css.getPropertyValue("--muted").trim() || "#5b6675", line: css.getPropertyValue("--line").trim() || "rgba(31,41,55,.10)", teal: css.getPropertyValue("--teal").trim() || "#1c7c7d", mint: css.getPropertyValue("--mint").trim() || "#2d8c68", blue: css.getPropertyValue("--blue").trim() || "#356ea9", gold: css.getPropertyValue("--gold").trim() || "#c48824", coral: css.getPropertyValue("--coral").trim() || "#c65b39", sans: css.getPropertyValue("--sans").trim() || "sans-serif" };
      const labels = ${level_labels_json}, accuracy = ${accuracy_json}, latency = ${latency_json}, tokens = ${token_json}, correct = ${correct_json}, incorrect = ${incorrect_json}, errors = ${error_json};
      const intFmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }), compactFmt = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
      Chart.defaults.color = colors.muted; Chart.defaults.font.family = colors.sans;
      const plugins = { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 10, color: colors.ink } }, tooltip: { backgroundColor: "rgba(28,33,39,.92)", titleColor: "#fff", bodyColor: "#fff", padding: 12 } };
      new Chart(document.getElementById("accuracyChart").getContext("2d"), { type: "bar", data: { labels, datasets: [{ label: "Correct", data: correct, backgroundColor: "rgba(45,140,104,.14)", borderColor: colors.mint, borderWidth: 1, stack: "count", borderRadius: 10, maxBarThickness: 42 }, { label: "Incorrect", data: incorrect, backgroundColor: "rgba(196,136,36,.14)", borderColor: colors.gold, borderWidth: 1, stack: "count", borderRadius: 10, maxBarThickness: 42 }, { label: "Errors", data: errors, backgroundColor: "rgba(198,91,57,.14)", borderColor: colors.coral, borderWidth: 1, stack: "count", borderRadius: 10, maxBarThickness: 42 }, { label: "Accuracy", type: "line", data: accuracy, yAxisID: "accuracy", borderColor: colors.teal, backgroundColor: colors.teal, borderWidth: 3, tension: .34, pointRadius: 4, pointHoverRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins, scales: { x: { stacked: true, grid: { display: false }, ticks: { color: colors.muted } }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0, color: colors.muted, callback: v => intFmt.format(v) }, grid: { color: colors.line } }, accuracy: { position: "right", beginAtZero: true, max: 100, grid: { drawOnChartArea: false }, ticks: { color: colors.muted, callback: v => v + "%" } } } } });
      new Chart(document.getElementById("latencyChart").getContext("2d"), { type: "line", data: { labels, datasets: [{ label: "Avg Latency (ms)", data: latency, fill: true, borderColor: colors.blue, backgroundColor: "rgba(53,110,169,.14)", borderWidth: 3, tension: .35, pointRadius: 4, pointHoverRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins, scales: { x: { grid: { display: false }, ticks: { color: colors.muted } }, y: { beginAtZero: true, ticks: { color: colors.muted, callback: v => compactFmt.format(v) + " ms" }, grid: { color: colors.line } } } } });
      new Chart(document.getElementById("tokenChart").getContext("2d"), { type: "line", data: { labels, datasets: [{ label: "Avg Tokens", data: tokens, fill: true, borderColor: colors.gold, backgroundColor: "rgba(196,136,36,.14)", borderWidth: 3, tension: .35, pointRadius: 4, pointHoverRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins, scales: { x: { grid: { display: false }, ticks: { color: colors.muted } }, y: { beginAtZero: true, ticks: { color: colors.muted, callback: v => compactFmt.format(v) }, grid: { color: colors.line } } } } });
    })();
  </script>
</body>
</html>"""
    )

    return template.substitute(
        page_title=_escape(title),
        hero_title=_escape(title),
        hero_copy=_escape(f"{split.title() if split not in {'', 'n/a'} else split} split / {_fmt_int(overall['total'])} samples / {duration} elapsed."),
        chips=chips,
        score_progress=f"{accuracy_pct:.1f}",
        score_value=_escape(_fmt_pct(overall["accuracy"])),
        score_note=_escape(f"{_fmt_int(overall['correct'])} of {_fmt_int(overall['total'])} samples answered correctly"),
        cards=cards,
        level_rows="".join(level_rows),
        sample_copy=_escape(f"{_fmt_int(len(sample_summaries))} samples listed. Error text is truncated in-table, with the full value shown on hover."),
        sample_rows="".join(sample_rows),
        level_labels_json=json.dumps(levels),
        accuracy_json=json.dumps(accuracy_data),
        latency_json=json.dumps(latency_data),
        token_json=json.dumps(token_data),
        correct_json=json.dumps(correct_data),
        incorrect_json=json.dumps(incorrect_data),
        error_json=json.dumps(error_data),
    )


def generate_html_report(run_dir: str, output: str) -> str:
    summary, raw_samples = _read_run_data(run_dir)
    sample_summaries = [_extract_sample_summary(record) for record in raw_samples]
    overall, by_level = _aggregate_metrics(sample_summaries)
    benchmark = str(summary.get("benchmark") or "unknown")
    run_id = str(summary.get("run_id") or "n/a")
    html = _render_html(summary, overall, by_level, sample_summaries, f"Evaluation Report - {benchmark} ({run_id})")

    out_path = os.path.abspath(output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


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
