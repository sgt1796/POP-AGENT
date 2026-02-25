from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from eval.core.contracts import EvalConfig
from eval.core.runner import run_evaluation, summarize_run


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = _build_run_config(args)
        progress_callback = None if args.quiet else _build_progress_printer()
        summary = run_evaluation(config, progress_callback=progress_callback)
        print(f"Run directory: {summary.run_dir}")
        print(f"Accuracy: {summary.accuracy:.4f} ({summary.correct}/{summary.total})")
        print(f"Errors: {summary.error_count}")
        return 0

    if args.command == "summarize":
        payload = summarize_run(args.run_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark evaluation runner")
    subparsers = parser.add_subparsers(dest="command")

    default_config = Path(__file__).resolve().parent / "configs" / "gaia_validation.yaml"

    run_parser = subparsers.add_parser("run", help="Run an evaluation")
    run_parser.add_argument("--config", default=str(default_config), help="Config file path (yaml or json)")
    run_parser.add_argument("--benchmark", help="Benchmark name")
    run_parser.add_argument("--split", help="Dataset split")
    run_parser.add_argument("--limit", type=int, help="Max samples")
    run_parser.add_argument("--seed", type=int, help="Random seed")
    run_parser.add_argument("--timeout-s", type=float, help="Per-sample timeout (seconds)")
    run_parser.add_argument("--output-root", help="Output root directory")
    run_parser.add_argument("--run-id", help="Run ID override")
    run_parser.add_argument("--executor", help="Executor kind (agent1 or echo)")
    run_parser.add_argument(
        "--benchmark-option",
        action="append",
        default=[],
        help="Benchmark option as key=value (repeatable)",
    )
    run_parser.add_argument(
        "--executor-option",
        action="append",
        default=[],
        help="Executor option as key=value (repeatable)",
    )
    run_parser.add_argument(
        "--redact-pattern",
        action="append",
        default=[],
        help="Additional redaction regex pattern (repeatable)",
    )
    run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first sample error instead of continue-on-error",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live progress output and print only final summary.",
    )

    summarize_parser = subparsers.add_parser("summarize", help="Rewrite and print run summary")
    summarize_parser.add_argument("--run-dir", required=True, help="Run directory or summary.json path")

    return parser


def _build_run_config(args: argparse.Namespace) -> EvalConfig:
    config_file = str(args.config or "").strip()
    base = _load_config_file(config_file)

    merged: Dict[str, Any] = {
        "benchmark": "gaia",
        "split": "validation",
        "continue_on_error": True,
        "executor": "agent1",
    }
    merged.update(base)

    if args.benchmark:
        merged["benchmark"] = args.benchmark
    if args.split:
        merged["split"] = args.split
    if args.limit is not None:
        merged["limit"] = int(args.limit)
    if args.seed is not None:
        merged["seed"] = int(args.seed)
    if args.timeout_s is not None:
        merged["timeout_s"] = float(args.timeout_s)
    if args.output_root:
        merged["output_root"] = args.output_root
    if args.run_id:
        merged["run_id"] = args.run_id
    if args.executor:
        merged["executor"] = args.executor

    merged["benchmark_options"] = _merge_nested_dict(
        merged.get("benchmark_options", {}),
        _parse_key_values(args.benchmark_option),
    )
    merged["executor_options"] = _merge_nested_dict(
        merged.get("executor_options", {}),
        _parse_key_values(args.executor_option),
    )

    redact_patterns = list(merged.get("redact_patterns", []))
    redact_patterns.extend(args.redact_pattern)
    merged["redact_patterns"] = [pattern for pattern in redact_patterns if str(pattern).strip()]

    if args.fail_fast:
        merged["continue_on_error"] = False

    return EvalConfig(**merged)


def _build_progress_printer():
    def _printer(event: Dict[str, Any]) -> None:
        event_type = str(event.get("type", "")).strip().lower()
        if event_type == "run_start":
            print(
                "[eval] run started "
                f"benchmark={event.get('benchmark')} split={event.get('split')} "
                f"run_id={event.get('run_id')}"
            )
            print(f"[eval] run_dir={event.get('run_dir')}")
            return

        if event_type == "loading_samples":
            print(
                "[eval] loading samples "
                f"benchmark={event.get('benchmark')} split={event.get('split')}"
            )
            return

        if event_type == "samples_loaded":
            print(f"[eval] loaded samples count={event.get('count')}")
            return

        if event_type == "sample_start":
            print(
                "[eval] sample start "
                f"{event.get('sample_number')}/{event.get('sample_total')} "
                f"id={event.get('sample_id')}"
            )
            return

        if event_type == "sample_end":
            status = str(event.get("status", "unknown"))
            correct = event.get("correct")
            latency_ms = event.get("latency_ms")
            suffix = ""
            if status == "error":
                suffix = f" error={event.get('error')}"
            elif correct is not None:
                suffix = f" correct={correct}"
            print(
                "[eval] sample end "
                f"{event.get('sample_number')}/{event.get('sample_total')} "
                f"id={event.get('sample_id')} status={status} latency_ms={latency_ms}{suffix}"
            )
            return

        if event_type == "run_complete":
            print(
                "[eval] run complete "
                f"accuracy={event.get('accuracy')} "
                f"correct={event.get('correct')}/{event.get('total')} "
                f"errors={event.get('error_count')} "
                f"duration_s={event.get('duration_s')}"
            )
            return

    return _printer


def _merge_nested_dict(base: Any, override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    out.update(override)
    return out


def _parse_key_values(items: Iterable[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Expected key=value, got: {item}")
        key, raw_value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid key in option: {item}")
        parsed[key] = _coerce_value(raw_value.strip())
    return parsed


def _coerce_value(value: str) -> Any:
    lowered = str(value).strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in lowered:
            return float(lowered)
        return int(lowered)
    except Exception:
        pass

    if (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]")):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _load_config_file(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not path or not cfg_path.exists():
        return {}

    text = cfg_path.read_text(encoding="utf-8")
    suffix = cfg_path.suffix.lower()

    if suffix == ".json":
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config JSON must be an object: {path}")
        return loaded

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required to read YAML configs") from exc
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config YAML must be a mapping: {path}")
        return loaded

    raise ValueError(f"Unsupported config format: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
