from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from eval.benchmarks.registry import get_adapter
from eval.core.artifacts import JsonArtifactWriter
from eval.core.contracts import (
    AgentExecutor,
    BenchmarkAdapter,
    BenchmarkSample,
    EvalConfig,
    RunSummary,
    SampleResult,
    ScoreResult,
)
from eval.executors.agent1_runtime_executor import Agent1RuntimeExecutor
from eval.executors.echo_executor import EchoExecutor


def run_evaluation(
    config: EvalConfig,
    *,
    adapter: Optional[BenchmarkAdapter] = None,
    executor: Optional[AgentExecutor] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> RunSummary:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_evaluation_async(
                config,
                adapter=adapter,
                executor=executor,
                progress_callback=progress_callback,
            )
        )
    raise RuntimeError("run_evaluation cannot run inside an active event loop; use run_evaluation_async instead")


async def run_evaluation_async(
    config: EvalConfig,
    *,
    adapter: Optional[BenchmarkAdapter] = None,
    executor: Optional[AgentExecutor] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> RunSummary:
    cfg = _coerce_config(config)
    started_ts = time.time()
    started_at = _utc_now_iso()
    run_id = str(cfg.run_id or uuid.uuid4().hex[:12])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(cfg.output_root, f"{timestamp}_{run_id}")
    run_dir_abs = str(Path(run_dir).resolve())

    _emit_progress(
        progress_callback,
        {
            "type": "run_start",
            "run_id": run_id,
            "benchmark": cfg.benchmark,
            "split": cfg.split,
            "run_dir": run_dir_abs,
        },
    )

    writer = JsonArtifactWriter(run_dir, redact_patterns=cfg.redact_patterns)
    benchmark_adapter = adapter or get_adapter(cfg.benchmark)
    _emit_progress(
        progress_callback,
        {
            "type": "loading_samples",
            "benchmark": cfg.benchmark,
            "split": cfg.split,
        },
    )
    benchmark_samples = benchmark_adapter.load_samples(
        split=cfg.split,
        limit=cfg.limit,
        seed=cfg.seed,
        options=dict(cfg.benchmark_options),
    )
    sample_count = len(benchmark_samples)
    _emit_progress(
        progress_callback,
        {
            "type": "samples_loaded",
            "count": sample_count,
            "benchmark": cfg.benchmark,
            "split": cfg.split,
        },
    )

    resolved_executor = executor or _build_executor(cfg.executor)

    manifest_payload = {
        "run_id": run_id,
        "benchmark": cfg.benchmark,
        "split": cfg.split,
        "started_at": started_at,
        "ended_at": None,
        "duration_s": None,
        "sample_count": sample_count,
        "git_sha": _git_sha(),
        "environment": _environment_snapshot(),
        "config": _to_jsonable(cfg),
        "artifacts": writer.artifact_paths,
    }
    writer.write_manifest(manifest_payload)

    sample_results: list[SampleResult] = []

    for index, base_sample in enumerate(benchmark_samples):
        prompt = benchmark_adapter.build_prompt(base_sample)
        eval_prefix = "You are in evaluation enviornment. Please strictly follow the instructions, and return only the final answer without any extra information.\n\n"
        sample = BenchmarkSample(
            sample_id=base_sample.sample_id,
            prompt=eval_prefix + prompt,
            ground_truth=base_sample.ground_truth,
            metadata=dict(base_sample.metadata or {}),
            assets=dict(base_sample.assets or {}),
        )
        sample_number = index + 1
        _emit_progress(
            progress_callback,
            {
                "type": "sample_start",
                "sample_index": index,
                "sample_number": sample_number,
                "sample_total": sample_count,
                "sample_id": sample.sample_id,
            },
        )

        try:
            execution = await resolved_executor.run_sample(
                sample,
                timeout_s=cfg.timeout_s,
                sample_index=index,
                run_id=run_id,
                run_dir=run_dir,
                executor_options=dict(cfg.executor_options),
            )
        except Exception as exc:
            execution = None
            execution_error = str(exc)
        else:
            execution_error = None

        if execution is None:
            score_result = ScoreResult(
                correct=False,
                score=0.0,
                reason="executor_exception",
                normalized_prediction="",
                normalized_ground_truth=sample.ground_truth,
            )
            sample_result = SampleResult(
                sample_id=sample.sample_id,
                status="error",
                prediction="",
                score_result=score_result,
                usage={},
                latency_ms=0.0,
                error=execution_error,
                trace_ref=f"events.jsonl#sample_id={sample.sample_id}",
                metadata=dict(sample.metadata or {}),
            )
            sample_results.append(sample_result)
            _write_sample_artifacts(writer, sample, sample_result)
            _emit_progress(
                progress_callback,
                {
                    "type": "sample_end",
                    "sample_index": index,
                    "sample_number": sample_number,
                    "sample_total": sample_count,
                    "sample_id": sample.sample_id,
                    "status": sample_result.status,
                    "correct": sample_result.score_result.correct,
                    "latency_ms": sample_result.latency_ms,
                    "error": sample_result.error,
                },
            )
            if not cfg.continue_on_error:
                raise RuntimeError(f"Executor failed on sample {sample.sample_id}: {execution_error}")
            continue

        status = execution.status
        prediction = execution.prediction
        sample_error = execution.error

        if status == "ok":
            try:
                score_result = benchmark_adapter.score(prediction, sample.ground_truth, sample)
            except Exception as score_exc:
                status = "error"
                sample_error = f"score error: {score_exc}"
                score_result = ScoreResult(
                    correct=False,
                    score=0.0,
                    reason="scoring_exception",
                    normalized_prediction=prediction,
                    normalized_ground_truth=sample.ground_truth,
                )
        else:
            score_result = ScoreResult(
                correct=False,
                score=0.0,
                reason="execution_error",
                normalized_prediction=prediction,
                normalized_ground_truth=sample.ground_truth,
            )

        sample_result = SampleResult(
            sample_id=sample.sample_id,
            status=status,
            prediction=prediction,
            score_result=score_result,
            usage=dict(execution.usage or {}),
            latency_ms=float(execution.latency_ms or 0.0),
            error=sample_error,
            trace_ref=execution.trace_ref,
            metadata=dict(sample.metadata or {}),
        )
        sample_results.append(sample_result)

        _write_sample_artifacts(writer, sample, sample_result)
        for event_index, event in enumerate(execution.events):
            writer.write_event(
                {
                    "sample_id": sample.sample_id,
                    "event_index": event_index,
                    "event": event,
                }
            )

        _emit_progress(
            progress_callback,
            {
                "type": "sample_end",
                "sample_index": index,
                "sample_number": sample_number,
                "sample_total": sample_count,
                "sample_id": sample.sample_id,
                "status": sample_result.status,
                "correct": sample_result.score_result.correct,
                "latency_ms": sample_result.latency_ms,
                "error": sample_result.error,
            },
        )

        if status == "error" and not cfg.continue_on_error:
            raise RuntimeError(f"Sample {sample.sample_id} failed: {sample_error}")

    ended_ts = time.time()
    ended_at = _utc_now_iso()
    duration_s = max(0.0, ended_ts - started_ts)

    adapter_metrics = benchmark_adapter.aggregate(sample_results)
    totals = _derive_totals(sample_results, adapter_metrics)

    summary_payload = {
        "run_id": run_id,
        "benchmark": cfg.benchmark,
        "split": cfg.split,
        "total": totals["total"],
        "correct": totals["correct"],
        "error_count": totals["error_count"],
        "accuracy": totals["accuracy"],
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "run_dir": run_dir_abs,
        "metrics": adapter_metrics,
        "artifacts": writer.artifact_paths,
    }
    writer.write_summary(summary_payload)

    manifest_payload["ended_at"] = ended_at
    manifest_payload["duration_s"] = duration_s
    manifest_payload["totals"] = totals
    writer.write_manifest(manifest_payload)

    _emit_progress(
        progress_callback,
        {
            "type": "run_complete",
            "run_id": run_id,
            "total": totals["total"],
            "correct": totals["correct"],
            "error_count": totals["error_count"],
            "accuracy": totals["accuracy"],
            "duration_s": duration_s,
            "run_dir": run_dir_abs,
        },
    )

    return RunSummary(
        run_id=run_id,
        benchmark=cfg.benchmark,
        split=cfg.split,
        total=totals["total"],
        correct=totals["correct"],
        error_count=totals["error_count"],
        accuracy=totals["accuracy"],
        run_dir=run_dir_abs,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        metrics=adapter_metrics,
        artifact_paths=writer.artifact_paths,
    )


def summarize_run(run_dir: str) -> Dict[str, Any]:
    summary_json_path = _resolve_summary_json_path(run_dir)

    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    writer = JsonArtifactWriter(str(summary_json_path.parent))
    writer.write_summary(payload)
    return payload


def _resolve_summary_json_path(run_dir: str) -> Path:
    raw_input = str(run_dir or "").strip()
    if not raw_input:
        raise FileNotFoundError("summary.json path is empty")

    raw_path = Path(raw_input)
    candidates: list[Path] = []

    def _add_candidate(path: Path) -> None:
        summary_path = path if path.name == "summary.json" else path / "summary.json"
        if summary_path not in candidates:
            candidates.append(summary_path)

    _add_candidate(raw_path)

    if not raw_path.is_absolute():
        lowered_parts = [part.lower() for part in raw_path.parts]
        if lowered_parts and lowered_parts[0] == "runs":
            _add_candidate(Path("eval") / raw_path)
        if len(raw_path.parts) == 1 and raw_path.name != "summary.json":
            _add_candidate(Path("eval") / "runs" / raw_path)

    for path in candidates:
        if path.exists():
            return path

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"summary.json not found for --run-dir={run_dir!r}. "
        f"Checked: {checked}. "
        "If running from repo root, use --run-dir eval/runs/<timestamp_runid>."
    )


def _coerce_config(config: EvalConfig) -> EvalConfig:
    if isinstance(config, EvalConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = EvalConfig(**dict(config))
    else:
        raise TypeError("config must be EvalConfig or dict")

    if cfg.concurrency <= 0:
        cfg.concurrency = 1
    if cfg.concurrency != 1:
        # v1 remains deterministic and single-threaded by design.
        cfg.concurrency = 1
    return cfg


def _emit_progress(callback: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(dict(event))
    except Exception:
        return


def _build_executor(name: str) -> AgentExecutor:
    key = str(name or "agent1").strip().lower()
    if key == "agent1":
        return Agent1RuntimeExecutor()
    if key == "echo":
        return EchoExecutor()
    raise ValueError(f"Unknown executor: {name}. Supported executors: ['agent1', 'echo']")


def _derive_totals(sample_results: list[SampleResult], metrics: Dict[str, Any]) -> Dict[str, Any]:
    total = int(metrics.get("total", len(sample_results)))
    correct = int(metrics.get("correct", sum(1 for item in sample_results if item.score_result.correct)))
    error_count = int(metrics.get("error_count", sum(1 for item in sample_results if item.status == "error")))
    accuracy = float(metrics.get("accuracy", (float(correct) / float(total)) if total else 0.0))
    return {
        "total": total,
        "correct": correct,
        "error_count": error_count,
        "accuracy": accuracy,
    }


def _write_sample_artifacts(writer: JsonArtifactWriter, sample: BenchmarkSample, sample_result: SampleResult) -> None:
    sample_payload = {
        "sample_id": sample.sample_id,
        "prompt": sample.prompt,
        "ground_truth": sample.ground_truth,
        "metadata": sample.metadata,
        "assets": sample.assets,
        "result": _to_jsonable(sample_result),
    }
    writer.write_sample(sample_payload)

    if sample_result.status == "error":
        writer.write_error(
            {
                "sample_id": sample.sample_id,
                "error": sample_result.error,
                "status": sample_result.status,
                "trace_ref": sample_result.trace_ref,
            }
        )


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _environment_snapshot() -> Dict[str, Any]:
    prefixes = (
        "POP_AGENT_",
        "PYTHON",
        "VIRTUAL_ENV",
        "CONDA",
        "OPENAI",
        "GEMINI",
        "HUGGINGFACE",
        "HF_",
        "JINAAI",
        "PERPLEXITY",
        "DEEPSEEK",
        "ALIYUN",
    )
    snapshot: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefixes):
            snapshot[key] = value
    return snapshot


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
