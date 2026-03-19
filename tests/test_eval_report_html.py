import json
import zipfile
from pathlib import Path

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
) -> dict:
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
        "result": {
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
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + ("\n" if rows else "")
    path.write_text(text, encoding="utf-8")


def _create_run_dir(tmp_path: Path, *, include_optional: bool = True) -> Path:
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
            [
                {"sample_id": "sample-ok", "event_index": 0, "event": {"type": "agent_start"}},
                {
                    "sample_id": "sample-ok",
                    "event_index": 1,
                    "event": {
                        "type": "message_end",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "Answer A"}]},
                    },
                },
                {
                    "sample_id": "sample-error",
                    "event_index": 0,
                    "event": {
                        "type": "tool_execution_end",
                        "toolName": "web_snapshot",
                        "result": {"details": {"error": "network unavailable"}},
                    },
                },
            ],
        )
    return run_dir


def test_generate_html_report_builds_bundle(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path, include_optional=True)
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

    ok_html = (sample_dir / "0001_sample-ok.html").read_text(encoding="utf-8")
    assert "Prompt A" in ok_html
    assert "Answer A" in ok_html
    assert "exact_match" in ok_html
    assert "1. Search" in ok_html
    assert "warning text" in ok_html
    assert "Show raw event JSON" in ok_html

    error_html = (sample_dir / "0002_sample-error.html").read_text(encoding="utf-8")
    assert "timeout after 180s" in error_html
    assert "network unavailable" in error_html


def test_generate_html_report_tolerates_missing_optional_artifacts(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path, include_optional=False)
    output_path = tmp_path / "basic.html"

    out = generate_html_report(str(run_dir), str(output_path))

    assert out == str(output_path.resolve())
    assert output_path.exists()
    assert (tmp_path / "basic_samples").exists()


def test_generate_html_report_accepts_zip_input(tmp_path: Path):
    run_dir = _create_run_dir(tmp_path / "src", include_optional=True)
    zip_path = tmp_path / "run.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_path in run_dir.iterdir():
            zf.write(file_path, arcname=f"archived-run/{file_path.name}")

    output_path = tmp_path / "zip_report.html"
    out = generate_html_report(str(zip_path), str(output_path))

    assert out == str(output_path.resolve())
    assert output_path.exists()
    assert (tmp_path / "zip_report_samples").exists()
