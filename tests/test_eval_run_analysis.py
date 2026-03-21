import json
from pathlib import Path

import pytest

from eval import run_analysis


def _sample(
    sample_id: str,
    *,
    status: str = "ok",
    correct: bool = False,
    prediction: str = "",
    ground_truth: str = "",
    error: str | None = None,
    total_tokens: int = 100,
    calls: int = 2,
    latency_ms: float = 1000.0,
    failure_analysis: dict | None = None,
) -> dict:
    result = {
        "sample_id": sample_id,
        "status": status,
        "prediction": prediction,
        "score_result": {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "reason": "exact_match" if correct else "mismatch",
            "normalized_prediction": prediction,
            "normalized_ground_truth": ground_truth,
        },
        "usage": {
            "delta": {
                "calls": calls,
                "total_tokens": total_tokens,
            }
        },
        "latency_ms": latency_ms,
        "error": error,
    }
    if failure_analysis is not None:
        result["failure_analysis"] = failure_analysis
    return {
        "sample_id": sample_id,
        "prompt": f"prompt {sample_id}",
        "ground_truth": ground_truth,
        "metadata": {"Level": "1"},
        "assets": {},
        "result": result,
    }


def _events(sample_id: str, payloads: list[dict]) -> list[dict]:
    return [{"sample_id": sample_id, "event_index": index, "event": payload} for index, payload in enumerate(payloads)]


def test_analyze_run_artifacts_classifies_failures_and_builds_metrics():
    summary = {"metrics": {}, "benchmark": "fake", "run_id": "run123"}
    samples = [
        _sample("correct", status="ok", correct=True, prediction="ok", ground_truth="ok", total_tokens=50, calls=1),
        _sample("runtime", status="error", correct=False, prediction="", ground_truth="a", error="timeout after 180s", total_tokens=120, calls=4, latency_ms=180000),
        _sample("format", status="ok", correct=False, prediction="2017 Komo Mai Drive 900000", ground_truth="900000", total_tokens=90, calls=2),
        _sample("precision", status="ok", correct=False, prediction="1:41.61", ground_truth="1:41.614", total_tokens=95, calls=2),
        _sample("missing", status="ok", correct=False, prediction="(no assistant text returned)", ground_truth="answer", total_tokens=80, calls=2),
    ]
    events_by_sample = {
        "runtime": _events(
            "runtime",
            [
                {
                    "type": "tool_execution_end",
                    "toolName": "bash_exec",
                    "result": {"details": {"blocked": True, "block_reason": "command_not_allowed"}},
                },
                {
                    "type": "message_end",
                    "message": {"role": "assistant", "stopReason": "aborted", "errorMessage": "LLM request timed out after 120s"},
                },
            ],
        ),
        "format": _events("format", [{"type": "tool_execution_end", "toolName": "perplexity_search", "result": {"details": {"ok": True}}}]),
        "precision": _events("precision", [{"type": "tool_execution_end", "toolName": "perplexity_search", "result": {"details": {"ok": True}}}]),
        "missing": _events("missing", []),
    }

    updated_summary, updated_samples, analysis = run_analysis.analyze_run_artifacts(summary, samples, {}, events_by_sample)

    by_id = {item["sample_id"]: item for item in updated_samples}
    assert by_id["runtime"]["result"]["failure_analysis"]["primary_cause"] == "runner_timeout"
    assert by_id["format"]["result"]["failure_analysis"]["primary_cause"] == "output_format_violation"
    assert by_id["precision"]["result"]["failure_analysis"]["primary_cause"] == "answer_precision_or_rounding"
    assert by_id["missing"]["result"]["failure_analysis"]["primary_cause"] == "no_final_answer"
    assert "failure_analysis" not in by_id["correct"]["result"]

    assert analysis["factor_names"] == [
        "correct",
        "runtime_error",
        "total_tokens",
        "latency_ms",
        "call_count",
        "distinct_tool_count",
    ]
    correlations = {(item["x"], item["y"]): item for item in analysis["correlations"]}
    assert correlations[("correct", "runtime_error")]["r"] is not None
    assert analysis["cohorts"]["runtime_error"]["count"] == 1
    assert analysis["failure_causes"]["total_non_correct"] == 4
    assert updated_summary["metrics"]["analysis"]["failure_causes"]["incorrect"][0]["count"] >= 1


def test_analyze_run_artifacts_omits_zero_variance_correlation():
    summary = {"metrics": {}}
    samples = [
        _sample("s1", status="ok", correct=True, prediction="a", ground_truth="a", total_tokens=10, calls=2),
        _sample("s2", status="ok", correct=False, prediction="b", ground_truth="a", total_tokens=20, calls=2),
    ]

    _updated_summary, _updated_samples, analysis = run_analysis.analyze_run_artifacts(summary, samples, {}, {})

    correlations = {(item["x"], item["y"]): item for item in analysis["correlations"]}
    assert correlations[("correct", "call_count")]["r"] is None


def test_analyze_run_artifacts_uses_promptfunction_for_all_non_correct_when_enabled(monkeypatch: pytest.MonkeyPatch):
    class _FakePromptFunction:
        init_clients: list[str] = []
        execute_calls: list[dict[str, object]] = []

        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt
            _FakePromptFunction.init_clients.append(str(client))

        def execute(self, *_args, **kwargs):
            _FakePromptFunction.execute_calls.append(kwargs)
            return json.dumps({"summary": "The search path did not converge on the right answer."})

    monkeypatch.setattr(run_analysis, "PromptFunction", _FakePromptFunction)

    summary = {"metrics": {}}
    manifest = {"config": {"executor_options": {"model_override": {"provider": "openai", "id": "gpt-5-mini"}}}}
    samples = [
        _sample("wrong", status="ok", correct=False, prediction="Number of citations", ground_truth="Number of common ancestors", total_tokens=50, calls=2),
        _sample("missing", status="ok", correct=False, prediction="(no assistant text returned)", ground_truth="answer", total_tokens=40, calls=1),
    ]

    _updated_summary, updated_samples, _analysis = run_analysis.analyze_run_artifacts(
        summary,
        samples,
        manifest,
        {},
        summarize_failure_causes=True,
    )

    by_id = {item["sample_id"]: item for item in updated_samples}
    assert by_id["wrong"]["result"]["failure_analysis"]["source"] == "hybrid"
    assert "ai_summary" in by_id["wrong"]["result"]["failure_analysis"]
    assert by_id["missing"]["result"]["failure_analysis"]["source"] == "hybrid"
    assert _FakePromptFunction.init_clients == ["openai"]
    assert len(_FakePromptFunction.execute_calls) == 2
    assert _FakePromptFunction.execute_calls[0]["model"] == "gpt-5-mini"


def test_analyze_run_artifacts_enriches_high_confidence_runtime_timeout(monkeypatch: pytest.MonkeyPatch):
    prompts: list[str] = []

    class _FakePromptFunction:
        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt, client

        def execute(self, prompt_text: str, **_kwargs):
            prompts.append(prompt_text)
            return json.dumps(
                {
                    "summary": "The agent tried downloading the source and shell-based PDF extraction, but download_url_to_file and blocked bash_exec commands kept it from resolving the timeout."
                }
            )

    monkeypatch.setattr(run_analysis, "PromptFunction", _FakePromptFunction)

    sample = _sample(
        "runtime-timeout",
        status="error",
        correct=False,
        prediction="",
        ground_truth="answer",
        error="timeout after 180s",
    )
    events_by_sample = {
        "runtime-timeout": _events(
            "runtime-timeout",
            [
                {
                    "type": "tool_execution_start",
                    "toolName": "download_url_to_file",
                    "args": {"url": "https://example.test/paper.pdf", "output_path": "paper.pdf"},
                },
                {
                    "type": "tool_execution_end",
                    "toolName": "download_url_to_file",
                    "result": {
                        "content": [{"type": "text", "text": "download_url_to_file error: http_error 404"}],
                        "details": {"error": "http_error"},
                    },
                    "isError": True,
                },
                {
                    "type": "tool_execution_start",
                    "toolName": "bash_exec",
                    "args": {"cmd": "python extract.py"},
                },
                {
                    "type": "tool_execution_end",
                    "toolName": "bash_exec",
                    "result": {
                        "content": [{"type": "text", "text": "bash_exec blocked: command_not_allowed"}],
                        "details": {"blocked": True, "block_reason": "command_not_allowed"},
                    },
                    "isError": True,
                },
            ],
        )
    }

    _updated_summary, updated_samples, _analysis = run_analysis.analyze_run_artifacts(
        {"metrics": {}},
        [sample],
        {},
        events_by_sample,
        summarize_failure_causes=True,
    )

    failure_analysis = updated_samples[0]["result"]["failure_analysis"]
    assert failure_analysis["primary_cause"] == "runner_timeout"
    assert failure_analysis["source"] == "hybrid"
    assert "downloading the source" in failure_analysis["ai_summary"]
    assert prompts
    assert "tool start download_url_to_file" in prompts[0]
    assert "tool end bash_exec: blocked=command_not_allowed" in prompts[0]


def test_analyze_run_artifacts_reuses_persisted_ai_summary_without_model_call(monkeypatch: pytest.MonkeyPatch):
    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("PromptFunction should not be used when persisted failure analysis already exists")

    monkeypatch.setattr(run_analysis, "PromptFunction", _fail_prompt)

    samples = [
        _sample(
            "wrong",
            status="ok",
            correct=False,
            prediction="Number of citations",
            ground_truth="Number of common ancestors",
            failure_analysis={
                "scope": "incorrect",
                "primary_cause": "wrong_answer",
                "confidence": "low",
                "evidence": ["prediction=Number of citations"],
                "source": "hybrid",
                "ai_summary": "The answer stayed too generic and missed the requested concept.",
                "summary_model": "openai / gpt-5-mini",
                "generated_at": "2026-03-19T22:00:00+00:00",
            },
        )
    ]

    _updated_summary, updated_samples, _analysis = run_analysis.analyze_run_artifacts(
        {"metrics": {}},
        samples,
        {},
        {},
        summarize_failure_causes=True,
    )

    failure_analysis = updated_samples[0]["result"]["failure_analysis"]
    assert failure_analysis["source"] == "hybrid"
    assert failure_analysis["ai_summary"] == "The answer stayed too generic and missed the requested concept."
    assert failure_analysis["summary_model"] == "openai / gpt-5-mini"


def test_analyze_run_artifacts_records_prompt_failure(monkeypatch: pytest.MonkeyPatch):
    class _FailingPromptFunction:
        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt, client

        def execute(self, *_args, **_kwargs):
            raise RuntimeError("prompt failure")

    monkeypatch.setattr(run_analysis, "PromptFunction", _FailingPromptFunction)

    _updated_summary, updated_samples, _analysis = run_analysis.analyze_run_artifacts(
        {"metrics": {}},
        [_sample("wrong", status="ok", correct=False, prediction="Number of citations", ground_truth="Number of common ancestors")],
        {},
        {},
        summarize_failure_causes=True,
    )

    failure_analysis = updated_samples[0]["result"]["failure_analysis"]
    assert failure_analysis["source"] == "deterministic"
    assert failure_analysis["summary_error"] == "prompt failure"


def test_persist_run_analysis_rewrites_summary_and_samples(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"metrics": {}, "run_id": "run123"}), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(
        json.dumps(_sample("wrong", status="ok", correct=False, prediction="Number of citations", ground_truth="Number of common ancestors"), ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    summary, samples, analysis = run_analysis.persist_run_analysis(str(run_dir))

    assert summary["metrics"]["analysis"]["failure_causes"]["total_non_correct"] == 1
    persisted_samples = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert persisted_samples[0]["result"]["failure_analysis"]["primary_cause"] == "wrong_answer"
    assert (run_dir / "summary.md").exists()
    assert analysis["cohorts"]["incorrect"]["count"] == 1


def test_analyze_run_artifacts_failure_prompt_omits_normal_stop_reason(monkeypatch: pytest.MonkeyPatch):
    prompts: list[str] = []

    class _FakePromptFunction:
        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt, client

        def execute(self, prompt_text: str, **_kwargs):
            prompts.append(prompt_text)
            return json.dumps({"summary": "The agent read the file but used the wrong value."})

    monkeypatch.setattr(run_analysis, "PromptFunction", _FakePromptFunction)

    sample = _sample(
        "wrong-stop",
        status="ok",
        correct=False,
        prediction="2.06",
        ground_truth="1.456",
    )
    events_by_sample = {
        "wrong-stop": _events(
            "wrong-stop",
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "2.06"}],
                        "stopReason": "stop",
                    },
                }
            ],
        )
    }

    _updated_summary, updated_samples, _analysis = run_analysis.analyze_run_artifacts(
        {"metrics": {}},
        [sample],
        {},
        events_by_sample,
        summarize_failure_causes=True,
    )

    assert prompts
    assert "Ground truth:" in prompts[0]
    assert "assistant: text=2.06" in prompts[0]
    assert "stop_reason=stop" not in prompts[0]
    failure_analysis = updated_samples[0]["result"]["failure_analysis"]
    assert failure_analysis["ai_summary"] == "The agent read the file but used the wrong value."
