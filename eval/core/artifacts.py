from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from .redaction import redact_data


class JsonArtifactWriter:
    def __init__(self, run_dir: str, *, redact_patterns: Iterable[str] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._redact_patterns = list(redact_patterns or [])

        self.manifest_path = self.run_dir / "manifest.json"
        self.samples_path = self.run_dir / "samples.jsonl"
        self.events_path = self.run_dir / "events.jsonl"
        self.errors_path = self.run_dir / "errors.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.summary_md_path = self.run_dir / "summary.md"

        for path in (self.samples_path, self.events_path, self.errors_path):
            path.touch(exist_ok=True)

    @property
    def artifact_paths(self) -> Dict[str, str]:
        return {
            "manifest": str(self.manifest_path),
            "samples": str(self.samples_path),
            "events": str(self.events_path),
            "errors": str(self.errors_path),
            "summary": str(self.summary_path),
            "summary_md": str(self.summary_md_path),
        }

    def write_manifest(self, payload: Dict[str, Any]) -> str:
        sanitized = self._sanitize(payload)
        self.manifest_path.write_text(json.dumps(sanitized, indent=2, sort_keys=True), encoding="utf-8")
        return str(self.manifest_path)

    def write_sample(self, payload: Dict[str, Any]) -> None:
        sanitized = self._sanitize(payload)
        self._append_jsonl(self.samples_path, sanitized)

    def write_event(self, payload: Dict[str, Any]) -> None:
        sanitized = self._sanitize(payload)
        self._append_jsonl(self.events_path, sanitized)

    def write_error(self, payload: Dict[str, Any]) -> None:
        sanitized = self._sanitize(payload)
        self._append_jsonl(self.errors_path, sanitized)

    def write_summary(self, payload: Dict[str, Any]) -> str:
        sanitized = self._sanitize(payload)
        self.summary_path.write_text(json.dumps(sanitized, indent=2, sort_keys=True), encoding="utf-8")
        self.summary_md_path.write_text(self._to_markdown(sanitized), encoding="utf-8")
        return str(self.summary_path)

    def _sanitize(self, payload: Any) -> Any:
        json_ready = self._to_json_ready(payload)
        return redact_data(json_ready, self._redact_patterns)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _to_json_ready(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._to_json_ready(asdict(value))
        if isinstance(value, dict):
            return {str(k): self._to_json_ready(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_json_ready(item) for item in value]
        if isinstance(value, tuple):
            return [self._to_json_ready(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _to_markdown(self, summary: Dict[str, Any]) -> str:
        lines = [
            "# Evaluation Summary",
            "",
            f"- Run ID: `{summary.get('run_id', '')}`",
            f"- Benchmark: `{summary.get('benchmark', '')}`",
            f"- Split: `{summary.get('split', '')}`",
            f"- Total: `{summary.get('total', 0)}`",
            f"- Correct: `{summary.get('correct', 0)}`",
            f"- Errors: `{summary.get('error_count', 0)}`",
            f"- Accuracy: `{summary.get('accuracy', 0.0):.4f}`",
            "",
            "## Metrics",
            "",
        ]

        metrics = summary.get("metrics", {})
        if not isinstance(metrics, dict) or not metrics:
            lines.append("(no additional metrics)")
        else:
            analysis = metrics.get("analysis") if isinstance(metrics.get("analysis"), dict) else {}
            for key in sorted(metrics.keys()):
                if key == "analysis":
                    continue
                lines.append(f"- `{key}`: `{metrics[key]}`")
            if analysis:
                lines.extend(
                    [
                        "",
                        "## Analysis",
                        "",
                        f"- Factors: `{', '.join(str(item) for item in analysis.get('factor_names', []))}`",
                    ]
                )
                correlations = [item for item in analysis.get("correlations", []) if isinstance(item, dict) and item.get("r") is not None]
                correlations.sort(key=lambda item: abs(float(item.get("r", 0.0))), reverse=True)
                if correlations:
                    lines.append("- Top correlations:")
                    for item in correlations[:5]:
                        lines.append(
                            f"  - `{item.get('x')}` vs `{item.get('y')}`: `r={float(item.get('r', 0.0)):.4f}` over `{item.get('n', 0)}` samples"
                        )
                cohorts = analysis.get("cohorts") if isinstance(analysis.get("cohorts"), dict) else {}
                if cohorts:
                    lines.append("- Cohorts:")
                    for cohort_name in ("correct", "incorrect", "runtime_error"):
                        cohort = cohorts.get(cohort_name)
                        if not isinstance(cohort, dict):
                            continue
                        lines.append(
                            "  - "
                            f"`{cohort_name}` count=`{cohort.get('count', 0)}` avg_tokens=`{self._fmt_markdown_number(cohort.get('avg_total_tokens'))}` "
                            f"avg_latency_ms=`{self._fmt_markdown_number(cohort.get('avg_latency_ms'))}` avg_calls=`{self._fmt_markdown_number(cohort.get('avg_call_count'))}`"
                        )
                failure_causes = analysis.get("failure_causes") if isinstance(analysis.get("failure_causes"), dict) else {}
                if failure_causes:
                    lines.append(f"- Non-correct samples: `{failure_causes.get('total_non_correct', 0)}`")
                    for scope in ("runtime_error", "incorrect"):
                        rows = [item for item in failure_causes.get(scope, []) if isinstance(item, dict)]
                        if not rows:
                            continue
                        rendered = ", ".join(f"{item.get('cause')}={item.get('count')}" for item in rows)
                        lines.append(f"- {scope}: `{rendered}`")

        return "\n".join(lines) + "\n"

    def _fmt_markdown_number(self, value: Any) -> str:
        try:
            return f"{float(value):.1f}"
        except Exception:
            return str(value)
