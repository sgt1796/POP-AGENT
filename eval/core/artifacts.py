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
            for key in sorted(metrics.keys()):
                lines.append(f"- `{key}`: `{metrics[key]}`")

        return "\n".join(lines) + "\n"
