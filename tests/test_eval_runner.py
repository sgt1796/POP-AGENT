import json
from pathlib import Path

from eval.core.contracts import BenchmarkSample, EvalConfig, ExecutionResult, SampleResult, ScoreResult
from eval.core.runner import run_evaluation


class _FakeAdapter:
    name = "fake"

    def load_samples(self, *, split, limit, seed, options):
        del split, seed, options
        samples = [
            BenchmarkSample(sample_id="s1", prompt="p1", ground_truth="a1", metadata={"Level": "1"}),
            BenchmarkSample(sample_id="s2", prompt="p2", ground_truth="a2", metadata={"Level": "2"}),
        ]
        if limit is not None:
            return samples[: int(limit)]
        return samples

    def build_prompt(self, sample: BenchmarkSample) -> str:
        return sample.prompt

    def score(self, prediction: str, ground_truth: str, sample: BenchmarkSample) -> ScoreResult:
        del sample
        correct = prediction == ground_truth
        return ScoreResult(
            correct=correct,
            score=1.0 if correct else 0.0,
            reason="exact_match" if correct else "mismatch",
            normalized_prediction=prediction,
            normalized_ground_truth=ground_truth,
        )

    def aggregate(self, sample_results):
        total = len(sample_results)
        correct = sum(1 for item in sample_results if item.score_result.correct)
        errors = sum(1 for item in sample_results if item.status == "error")
        return {
            "total": total,
            "correct": correct,
            "error_count": errors,
            "accuracy": (correct / total) if total else 0.0,
        }


class _FakeExecutor:
    async def run_sample(self, sample, *, timeout_s, sample_index, run_id, run_dir, executor_options):
        del timeout_s, sample_index, run_id, run_dir, executor_options
        if sample.sample_id == "s2":
            return ExecutionResult(
                status="error",
                prediction="",
                usage={"delta": {"calls": 1}},
                latency_ms=1.0,
                error="boom",
                events=[{"type": "error"}],
                trace_ref="events.jsonl#s2",
            )

        return ExecutionResult(
            status="ok",
            prediction="a1",
            usage={"delta": {"calls": 1}},
            latency_ms=1.0,
            error=None,
            events=[{"type": "done"}],
            trace_ref="events.jsonl#s1",
        )


def test_runner_continue_on_error_and_artifacts(tmp_path: Path):
    cfg = EvalConfig(
        benchmark="fake",
        split="validation",
        output_root=str(tmp_path),
        continue_on_error=True,
        executor="echo",
    )

    summary = run_evaluation(cfg, adapter=_FakeAdapter(), executor=_FakeExecutor())

    assert summary.total == 2
    assert summary.error_count == 1
    assert summary.correct == 1
    assert summary.accuracy == 0.5

    run_dir = Path(summary.run_dir)
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "samples.jsonl").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "errors.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == summary.run_id
    assert "config" in manifest

    sample_lines = [line for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(sample_lines) == 2

    error_lines = [line for line in (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(error_lines) == 1


def test_runner_emits_progress_events(tmp_path: Path):
    cfg = EvalConfig(
        benchmark="fake",
        split="validation",
        output_root=str(tmp_path),
        continue_on_error=True,
        executor="echo",
    )
    events = []

    summary = run_evaluation(
        cfg,
        adapter=_FakeAdapter(),
        executor=_FakeExecutor(),
        progress_callback=events.append,
    )

    assert summary.total == 2
    event_types = [str(event.get("type")) for event in events]
    assert event_types[0] == "run_start"
    assert "loading_samples" in event_types
    assert "samples_loaded" in event_types
    assert event_types.count("sample_start") == 2
    assert event_types.count("sample_end") == 2
    assert event_types[-1] == "run_complete"
