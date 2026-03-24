import json
import zipfile
from pathlib import Path

import pytest

from eval import agent_step_summary
from eval.report_html import generate_html_report


def _build_sample(
    *,
    sample_id: str,
    prompt: str,
    ground_truth: str,
    prediction: str,
    status: str,
    correct: bool,
    score_reason: str,
    error: str | None,
    started_at: str | None = "2026-03-18T20:01:00+00:00",
    ended_at: str | None = "2026-03-18T20:01:04+00:00",
    agent_execution_summary: dict | None = None,
) -> dict:
    result = {
        "sample_id": sample_id,
        "status": status,
        "prediction": prediction,
        "score_result": {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "reason": score_reason,
            "normalized_prediction": prediction.lower(),
            "normalized_ground_truth": ground_truth.lower(),
        },
        "usage": {
            "before": {"calls": 0, "total_tokens": 0},
            "after": {"calls": 2, "total_tokens": 300},
            "delta": {"calls": 2, "input_tokens": 120, "output_tokens": 15, "total_tokens": 300},
            "last": {"provider": "gemini", "model": "gemini-3-pro-preview", "latency_ms": 1234},
            "warnings": ["warning text"] if status == "ok" else [],
            "attachments": [{"name": "evidence.txt", "workspace_path": "attachments/evidence.txt", "source_uri": "hf://dataset/evidence.txt"}],
        },
        "latency_ms": 4567.0,
        "error": error,
        "trace_ref": f"events.jsonl#sample_id={sample_id}",
    }
    if started_at is not None:
        result["started_at"] = started_at
    if ended_at is not None:
        result["ended_at"] = ended_at
    if agent_execution_summary is not None:
        result["agent_execution_summary"] = agent_execution_summary

    return {
        "sample_id": sample_id,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "metadata": {
            "Level": "2",
            "Annotator Metadata": {
                "Steps": "1. Search\n2. Compare\n3. Answer",
                "Number of steps": "3",
                "How long did this take?": "2 minutes",
                "Tools": "browser, search",
                "Number of tools": "2",
            },
        },
        "assets": {},
        "result": result,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + ("\n" if rows else "")
    path.write_text(text, encoding="utf-8")


def _safe_id(value: str) -> str:
    text = str(value or "sample")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _write_memory_trace(run_dir: Path, *, sample_index: int, sample_id: str, rows: list[dict]) -> None:
    memory_dir = run_dir / "_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / f"run123_sample_{sample_index:06d}_{_safe_id(sample_id)}.jsonl"
    _write_jsonl(path, rows)


def _sample_ok_events(*, nested: bool) -> list[dict]:
    if not nested:
        return [
            {"sample_id": "sample-ok", "event_index": 0, "event": {"type": "agent_start", "timestamp": "2026-03-18T20:01:00+00:00"}},
            {
                "sample_id": "sample-ok",
                "event_index": 1,
                "event": {
                    "type": "tool_execution_start",
                    "toolName": "search_engine",
                    "args": {"query": "Prompt A"},
                    "timestamp": "2026-03-18T20:01:01+00:00",
                },
            },
            {
                "sample_id": "sample-ok",
                "event_index": 2,
                "event": {
                    "type": "tool_execution_end",
                    "toolName": "search_engine",
                    "result": {"details": {"ok": True}},
                    "timestamp": "2026-03-18T20:01:02+00:00",
                },
            },
            {
                "sample_id": "sample-ok",
                "event_index": 3,
                "event": {
                    "type": "tool_execution_start",
                    "toolName": "web_browser",
                    "args": {"open": "https://example.test"},
                    "timestamp": "2026-03-18T20:01:03+00:00",
                },
            },
            {
                "sample_id": "sample-ok",
                "event_index": 4,
                "event": {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Answer A"}],
                        "timestamp": "2026-03-18T20:01:04+00:00",
                    },
                },
            },
        ]

    return [
        {"sample_id": "sample-ok", "event_index": 0, "event": {"type": "agent_start", "timestamp": "2026-03-18T20:01:00+00:00"}},
        {"sample_id": "sample-ok", "event_index": 1, "event": {"type": "turn_start", "timestamp": "2026-03-18T20:01:00+00:00"}},
        {
            "sample_id": "sample-ok",
            "event_index": 2,
            "event": {
                "type": "message_start",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "timestamp": "2026-03-18T20:01:00+00:00",
                },
            },
        },
        {
            "sample_id": "sample-ok",
            "event_index": 3,
            "event": {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "text_start",
                    "contentIndex": 0,
                    "partial": {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                },
                "message": {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            },
        },
        {
            "sample_id": "sample-ok",
            "event_index": 4,
            "event": {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "name": "search_engine", "arguments": {"query": "Prompt A"}}],
                    "timestamp": "2026-03-18T20:01:01+00:00",
                },
            },
        },
        {
            "sample_id": "sample-ok",
            "event_index": 5,
            "event": {
                "type": "tool_execution_start",
                "toolCallId": "call-1",
                "toolName": "search_engine",
                "args": {"query": "Prompt A"},
                "timestamp": "2026-03-18T20:01:01+00:00",
            },
        },
        {
            "sample_id": "sample-ok",
            "event_index": 6,
            "event": {
                "type": "tool_execution_end",
                "toolCallId": "call-1",
                "toolName": "search_engine",
                "result": {"details": {"ok": True}},
                "timestamp": "2026-03-18T20:01:02+00:00",
            },
        },
        {
            "sample_id": "sample-ok",
            "event_index": 7,
            "event": {
                "type": "turn_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "name": "search_engine", "arguments": {"query": "Prompt A"}}],
                    "timestamp": "2026-03-18T20:01:02+00:00",
                },
                "toolResults": [{"toolName": "search_engine"}],
            },
        },
        {"sample_id": "sample-ok", "event_index": 8, "event": {"type": "turn_start", "timestamp": "2026-03-18T20:01:03+00:00"}},
        {
            "sample_id": "sample-ok",
            "event_index": 9,
            "event": {
                "type": "message_start",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "timestamp": "2026-03-18T20:01:03+00:00",
                },
            },
        },
        {
            "sample_id": "sample-ok",
            "event_index": 10,
            "event": {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "text_start",
                    "contentIndex": 0,
                    "partial": {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                },
                "message": {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            },
        },
        {
            "sample_id": "sample-ok",
            "event_index": 11,
            "event": {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "text_delta",
                    "contentIndex": 0,
                    "delta": "Answer A",
                    "partial": {"role": "assistant", "content": [{"type": "text", "text": "Answer A"}]},
                },
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Answer A"}]},
            },
        },
        {
            "sample_id": "sample-ok",
            "event_index": 12,
            "event": {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Answer A"}],
                    "timestamp": "2026-03-18T20:01:04+00:00",
                },
            },
        },
        {
            "sample_id": "sample-ok",
            "event_index": 13,
            "event": {
                "type": "turn_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Answer A"}],
                    "timestamp": "2026-03-18T20:01:04+00:00",
                },
            },
        },
        {"sample_id": "sample-ok", "event_index": 14, "event": {"type": "agent_end", "timestamp": "2026-03-18T20:01:04+00:00"}},
    ]


def _create_run_dir(
    tmp_path: Path,
    *,
    include_optional: bool = True,
    persisted_summary: bool = False,
    include_timing: bool = True,
    include_memory: bool = False,
    nested_events: bool = False,
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    samples = [
        _build_sample(
            sample_id="sample-ok",
            prompt="Prompt A",
            ground_truth="Answer A",
            prediction="Answer A",
            status="ok",
            correct=True,
            score_reason="exact_match",
            error=None,
            started_at="2026-03-18T20:01:00+00:00" if include_timing else None,
            ended_at="2026-03-18T20:01:04+00:00" if include_timing else None,
            agent_execution_summary={
                "steps": ["Search for the source", "Compare the retrieved evidence", "Return the answer"],
                "step_count": 3,
                "tool_names": ["search_engine", "web_browser"],
                "tool_count": 2,
                "summary_model": "gemini / gemini-3-pro-preview",
                "generated_at": "2026-03-18T20:10:00+00:00",
                "source": "promptfunction",
            }
            if persisted_summary
            else None,
        ),
        _build_sample(
            sample_id="sample-error",
            prompt="Prompt B",
            ground_truth="Answer B",
            prediction="",
            status="error",
            correct=False,
            score_reason="execution_error",
            error="timeout after 180s",
            started_at="2026-03-18T20:02:00+00:00" if include_timing else None,
            ended_at="2026-03-18T20:02:05+00:00" if include_timing else None,
        ),
    ]
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run123",
                "benchmark": "gaia",
                "split": "validation",
                "total": 2,
                "correct": 1,
                "error_count": 1,
                "accuracy": 0.5,
                "started_at": "2026-03-18T20:00:00+00:00",
                "ended_at": "2026-03-18T20:10:00+00:00",
                "duration_s": 600.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_jsonl(run_dir / "samples.jsonl", samples)
    if include_optional:
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "git_sha": "abc123",
                    "config": {
                        "executor": "agent1",
                        "executor_options": {
                            "model_override": {"provider": "gemini", "id": "gemini-3-pro-preview"}
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _write_jsonl(
            run_dir / "errors.jsonl",
            [
                {
                    "sample_id": "sample-error",
                    "status": "error",
                    "error": "timeout after 180s",
                    "trace_ref": "events.jsonl#sample_id=sample-error",
                }
            ],
        )
        _write_jsonl(
            run_dir / "events.jsonl",
            _sample_ok_events(nested=nested_events)
            + [
                {
                    "sample_id": "sample-error",
                    "event_index": 0,
                    "event": {
                        "type": "tool_execution_end",
                        "toolName": "web_snapshot",
                        "result": {"details": {"error": "network unavailable"}},
                        "timestamp": "2026-03-18T20:02:05+00:00",
                    },
                },
            ],
        )
        if include_memory:
            _write_memory_trace(
                run_dir,
                sample_index=0,
                sample_id="sample-ok",
                rows=[
                    {
                        "text": "user: Prompt A",
                        "session_id": "session-a",
                        "memory_type": "message",
                        "timestamp": 1773883260.0,
                    },
                    {
                        "text": "tool_execution_start search_engine",
                        "session_id": "session-a",
                        "memory_type": "tool_call",
                        "timestamp": 1773883261.0,
                    },
                    {
                        "text": "tool_execution_end search_engine error=False",
                        "session_id": "session-a",
                        "memory_type": "tool_call",
                        "timestamp": 1773883262.0,
                    },
                    {
                        "text": "tool_execution_start web_browser",
                        "session_id": "session-a",
                        "memory_type": "tool_call",
                        "timestamp": 1773883263.0,
                    },
                    {
                        "text": "assistant: Answer A",
                        "session_id": "session-a",
                        "memory_type": "message",
                        "timestamp": 1773883264.0,
                    },
                ],
            )
    return run_dir


def test_generate_html_report_builds_bundle(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=True)
    output_path = tmp_path / "report.html"

    out = generate_html_report(str(run_dir), str(output_path))

    assert out == str(output_path.resolve())
    assert output_path.exists()
    sample_dir = tmp_path / "report_samples"
    assert sample_dir.exists()

    summary_html = output_path.read_text(encoding="utf-8")
    assert "report_samples/0001_sample-ok.html" in summary_html
    assert "Score Reason" in summary_html
    assert "Provider / Model" in summary_html
    assert "Correlation Overview" in summary_html
    assert "corr-cell corr-cell--diag" in summary_html
    assert "background:hsla(" in summary_html
    assert "Outcome Cohorts" in summary_html
    assert "Failure Causes" in summary_html
    assert "Level Breakdown" in summary_html

    ok_html = (sample_dir / "0001_sample-ok.html").read_text(encoding="utf-8")
    assert "Prompt A" in ok_html
    assert "Answer A" in ok_html
    assert "exact_match" in ok_html
    assert "Annotator Summary" in ok_html
    assert "Agent Summary" in ok_html
    assert "Annotator Steps" in ok_html
    assert "Agent Steps" in ok_html
    assert "Search for the source" in ok_html
    assert "Web Browser" in ok_html
    assert "Mar 18, 2026 20:01:00" in ok_html
    assert "warning text" in ok_html
    assert "Show raw event JSON" in ok_html
    assert "Failure Analysis" in ok_html

    error_html = (sample_dir / "0002_sample-error.html").read_text(encoding="utf-8")
    assert "timeout after 180s" in error_html
    assert "network unavailable" in error_html
    assert "Failure Analysis" in error_html


def test_generate_html_report_tolerates_missing_optional_artifacts(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path, include_optional=False)
    output_path = tmp_path / "basic.html"

    out = generate_html_report(str(run_dir), str(output_path))

    assert out == str(output_path.resolve())
    assert output_path.exists()
    assert (tmp_path / "basic_samples").exists()


def test_generate_html_report_accepts_zip_input(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path / "src", include_optional=True, persisted_summary=True)
    zip_path = tmp_path / "run.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_path in run_dir.iterdir():
            zf.write(file_path, arcname=f"archived-run/{file_path.name}")

    output_path = tmp_path / "zip_report.html"
    out = generate_html_report(str(zip_path), str(output_path))

    assert out == str(output_path.resolve())
    assert output_path.exists()
    assert (tmp_path / "zip_report_samples").exists()
    assert "Correlation Overview" in output_path.read_text(encoding="utf-8")


def test_generate_html_report_summarizes_agent_steps_and_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _FakePromptFunction:
        init_clients: list[str] = []
        execute_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt
            _FakePromptFunction.init_clients.append(str(client))

        def execute(self, *args, **kwargs):
            _FakePromptFunction.execute_calls.append((args, kwargs))
            return json.dumps({"steps": ["Search the web", "Open the relevant page", "Answer the question"]})

    monkeypatch.setattr(agent_step_summary, "PromptFunction", _FakePromptFunction)

    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=False)
    output_path = tmp_path / "report.html"
    out = generate_html_report(str(run_dir), str(output_path), summarize_agent_steps=True)

    assert out == str(output_path.resolve())
    assert _FakePromptFunction.init_clients == ["gemini"]
    assert _FakePromptFunction.execute_calls
    assert _FakePromptFunction.execute_calls[0][1]["model"] == "gemini-3-pro-preview"

    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    persisted = samples[0]["result"]["agent_execution_summary"]
    assert persisted["steps"] == ["Search the web", "Open the relevant page", "Answer the question"]
    assert persisted["step_count"] == 3
    assert persisted["tool_names"] == ["search_engine", "web_browser"]
    assert persisted["tool_count"] == 2
    assert persisted["source"] == "event+promptfunction"
    assert "generated_at" in persisted

    html = (tmp_path / "report_samples" / "0001_sample-ok.html").read_text(encoding="utf-8")
    assert "Search the web" in html
    assert "Open the relevant page" in html


def test_generate_html_report_summarizes_agent_steps_from_memory_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _FakePromptFunction:
        execute_calls: list[str] = []

        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt, client

        def execute(self, trace_text: str, **_kwargs):
            _FakePromptFunction.execute_calls.append(trace_text)
            return json.dumps({"steps": ["Search for the target", "Open the page", "Answer"]})

    monkeypatch.setattr(agent_step_summary, "PromptFunction", _FakePromptFunction)

    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=False, include_memory=True)
    output_path = tmp_path / "memory.html"
    generate_html_report(str(run_dir), str(output_path), summarize_agent_steps=True)

    assert _FakePromptFunction.execute_calls
    trace_text = _FakePromptFunction.execute_calls[0]
    assert "Condensed execution trace:" in trace_text
    assert "tool=search_engine" in trace_text
    assert "tool=web_browser" in trace_text
    assert "turn_end tool_results" not in trace_text

    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    persisted = samples[0]["result"]["agent_execution_summary"]
    assert persisted["source"] == "memory+promptfunction"


def test_generate_html_report_summary_overrides_provider_and_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _FakePromptFunction:
        init_clients: list[str] = []
        execute_kwargs: list[dict[str, object]] = []

        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt
            _FakePromptFunction.init_clients.append(str(client))

        def execute(self, *_args, **kwargs):
            _FakePromptFunction.execute_kwargs.append(kwargs)
            return {"steps": ["Use override model"]}

    monkeypatch.setattr(agent_step_summary, "PromptFunction", _FakePromptFunction)

    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=False)
    output_path = tmp_path / "override.html"
    generate_html_report(
        str(run_dir),
        str(output_path),
        summarize_agent_steps=True,
        summary_provider="openai",
        summary_model="gpt-5-mini",
    )

    assert _FakePromptFunction.init_clients == ["openai"]
    assert _FakePromptFunction.execute_kwargs[0]["model"] == "gpt-5-mini"


def test_generate_html_report_reuses_persisted_summary_without_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("PromptFunction should not be used when reusing persisted summaries")

    monkeypatch.setattr(agent_step_summary, "PromptFunction", _fail_prompt)

    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=True)
    output_path = tmp_path / "report.html"
    generate_html_report(str(run_dir), str(output_path))

    html = (tmp_path / "report_samples" / "0001_sample-ok.html").read_text(encoding="utf-8")
    assert "Search for the source" in html


def test_generate_html_report_falls_back_to_event_timestamps_for_old_runs(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path, include_optional=True, include_timing=False)
    output_path = tmp_path / "fallback.html"
    generate_html_report(str(run_dir), str(output_path))

    html = (tmp_path / "fallback_samples" / "0001_sample-ok.html").read_text(encoding="utf-8")
    assert "Mar 18, 2026 20:01:00" in html
    assert "Mar 18, 2026 20:01:04" in html
    assert "Search Engine" in html


def test_generate_html_report_marks_summary_failures_without_aborting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _FailingPromptFunction:
        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt, client

        def execute(self, *_args, **_kwargs):
            raise RuntimeError("prompt failure")

    monkeypatch.setattr(agent_step_summary, "PromptFunction", _FailingPromptFunction)

    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=False)
    output_path = tmp_path / "report.html"
    generate_html_report(str(run_dir), str(output_path), summarize_agent_steps=True)

    html = (tmp_path / "report_samples" / "0001_sample-ok.html").read_text(encoding="utf-8")
    assert "Search for Prompt A" in html
    assert "Agent step summarization failed" not in html

    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    persisted = samples[0]["result"]["agent_execution_summary"]
    assert persisted["source"] == "event_fallback"
    assert persisted["steps"] == [
        "Search for Prompt A",
        "Open https://example.test",
        "Return the final answer Answer A",
    ]


def test_generate_html_report_hides_level_breakdown_without_level_metadata(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=False)
    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for sample in samples:
        if isinstance(sample.get("metadata"), dict):
            sample["metadata"].pop("Level", None)
    _write_jsonl(run_dir / "samples.jsonl", samples)

    output_path = tmp_path / "report.html"
    generate_html_report(str(run_dir), str(output_path))

    html = output_path.read_text(encoding="utf-8")
    assert "Correlation Overview" in html
    assert "Level Breakdown" not in html


def test_generate_html_report_rejects_zip_input_when_summarizing_failure_causes(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path / "src", include_optional=True)
    zip_path = tmp_path / "run.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_path in run_dir.iterdir():
            zf.write(file_path, arcname=f"archived-run/{file_path.name}")

    with pytest.raises(ValueError, match="--summarize-failure-causes"):
        generate_html_report(str(zip_path), str(tmp_path / "zip_report.html"), summarize_failure_causes=True)


def test_generate_html_report_rejects_zip_input_when_summarizing(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path / "src", include_optional=True)
    zip_path = tmp_path / "run.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_path in run_dir.iterdir():
            zf.write(file_path, arcname=f"archived-run/{file_path.name}")

    with pytest.raises(ValueError, match="--summarize-agent-steps"):
        generate_html_report(str(zip_path), str(tmp_path / "zip_report.html"), summarize_agent_steps=True)


def test_generate_html_report_accepts_numbered_text_steps_from_promptfunction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _FakePromptFunction:
        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt, client

        def execute(self, *_args, **_kwargs):
            return "1. Search the web\n2. Open the relevant page\n3. Answer the question"

    monkeypatch.setattr(agent_step_summary, "PromptFunction", _FakePromptFunction)

    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=False)
    output_path = tmp_path / "report.html"
    generate_html_report(str(run_dir), str(output_path), summarize_agent_steps=True)

    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    persisted = samples[0]["result"]["agent_execution_summary"]
    assert persisted["steps"] == ["Search the web", "Open the relevant page", "Answer the question"]


def test_generate_html_report_falls_back_to_event_steps_when_prompt_output_is_unusable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _FakePromptFunction:
        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt, client

        def execute(self, *_args, **_kwargs):
            return {"steps": [46]}

    monkeypatch.setattr(agent_step_summary, "PromptFunction", _FakePromptFunction)

    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=False)
    output_path = tmp_path / "report.html"
    generate_html_report(str(run_dir), str(output_path), summarize_agent_steps=True)

    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    persisted = samples[0]["result"]["agent_execution_summary"]
    assert persisted["source"] == "event_fallback"
    assert persisted["steps"] == [
        "Search for Prompt A",
        "Open https://example.test",
        "Return the final answer Answer A",
    ]
    assert "error" not in persisted


def test_generate_html_report_falls_back_to_memory_steps_when_prompt_output_is_unusable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _FakePromptFunction:
        def __init__(self, sys_prompt: str = "", prompt: str = "", client: str | None = None):
            del sys_prompt, prompt, client

        def execute(self, *_args, **_kwargs):
            return {"steps": [46]}

    monkeypatch.setattr(agent_step_summary, "PromptFunction", _FakePromptFunction)

    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=False, include_memory=True)
    output_path = tmp_path / "memory_fallback.html"
    generate_html_report(str(run_dir), str(output_path), summarize_agent_steps=True)

    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    persisted = samples[0]["result"]["agent_execution_summary"]
    assert persisted["source"] == "memory_fallback"
    assert persisted["steps"] == [
        "Search for Prompt A",
        "Open https://example.test",
        "Return the final answer Answer A",
    ]


def test_generate_html_report_renders_nested_timeline_boxes_with_per_box_json(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path, include_optional=True, persisted_summary=True, nested_events=True)
    output_path = tmp_path / "nested.html"
    generate_html_report(str(run_dir), str(output_path))

    html = (tmp_path / "nested_samples" / "0001_sample-ok.html").read_text(encoding="utf-8")
    assert "Agent Run" in html
    assert "Turn" in html
    assert "Message · Assistant" in html
    assert "Tool · Search Engine" in html
    assert "Show JSON" in html
    assert "Show raw event JSON" in html
    assert "Event · Message Update" not in html
