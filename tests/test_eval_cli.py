import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from eval import cli


def test_cli_run_and_summarize_smoke(monkeypatch, tmp_path: Path, capsys):
    frame = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "question": "echo me",
                "final_answer": "echo me",
                "Level": "1",
            }
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "benchmark": "gaia",
                "split": "validation",
                "limit": 1,
                "output_root": str(tmp_path / "runs"),
                "executor": "echo",
                "continue_on_error": True,
                "benchmark_options": {"parquet_path": "dummy"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run", "--config", str(config_path)])
    assert exit_code == 0
    capsys.readouterr()

    run_dirs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    report_path = run_dirs[0] / "report.html"
    sample_pages_dir = run_dirs[0] / "report_samples"
    assert report_path.exists()
    assert sample_pages_dir.exists()
    assert any(sample_pages_dir.iterdir())

    report_path.unlink()
    shutil.rmtree(sample_pages_dir)

    summarize_code = cli.main(["summarize", "--run-dir", str(run_dirs[0])])
    assert summarize_code == 0
    summarize_output = capsys.readouterr().out
    payload = json.loads(summarize_output)
    assert payload["run_id"]
    assert report_path.exists()
    assert sample_pages_dir.exists()
    assert any(sample_pages_dir.iterdir())
    assert (run_dirs[0] / "summary.json").exists()
    assert (run_dirs[0] / "summary.md").exists()


def test_cli_quiet_suppresses_progress_lines(monkeypatch, tmp_path: Path, capsys):
    frame = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "question": "echo me",
                "final_answer": "echo me",
            }
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "benchmark": "gaia",
                "split": "validation",
                "limit": 1,
                "output_root": str(tmp_path / "runs"),
                "executor": "echo",
                "continue_on_error": True,
                "benchmark_options": {"parquet_path": "dummy"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run", "--quiet", "--config", str(config_path)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "[eval]" not in output


def test_cli_run_no_report_skips_default_html_generation(monkeypatch, tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "question": "echo me",
                "final_answer": "echo me",
                "Level": "1",
            }
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "benchmark": "gaia",
                "split": "validation",
                "limit": 1,
                "output_root": str(tmp_path / "runs"),
                "executor": "echo",
                "continue_on_error": True,
                "benchmark_options": {"parquet_path": "dummy"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run", "--quiet", "--no-report", "--config", str(config_path)])
    assert exit_code == 0

    run_dirs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "report.html").exists()
    assert not (run_dirs[0] / "report_samples").exists()


def test_cli_summarize_accepts_runs_prefixed_path(monkeypatch, tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "question": "echo me",
                "final_answer": "echo me",
            }
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "benchmark": "gaia",
                "split": "validation",
                "limit": 1,
                "output_root": str(tmp_path / "eval" / "runs"),
                "executor": "echo",
                "continue_on_error": True,
                "benchmark_options": {"parquet_path": "dummy"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run", "--quiet", "--config", str(config_path)])
    assert exit_code == 0

    run_dirs = [p for p in (tmp_path / "eval" / "runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1

    monkeypatch.chdir(tmp_path)
    summarize_code = cli.main(["summarize", "--run-dir", f"runs/{run_dirs[0].name}"])
    assert summarize_code == 0
    assert (run_dirs[0] / "report.html").exists()
    assert (run_dirs[0] / "report_samples").exists()


def test_cli_run_without_seed_preserves_null_seed_in_manifest(monkeypatch, tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "question": "echo me",
                "final_answer": "echo me",
            }
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "benchmark": "gaia",
                "split": "validation",
                "limit": 1,
                "output_root": str(tmp_path / "runs"),
                "executor": "echo",
                "continue_on_error": True,
                "benchmark_options": {"parquet_path": "dummy", "shuffle": True},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run", "--quiet", "--config", str(config_path)])
    assert exit_code == 0

    run_dirs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1

    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["seed"] is None


def test_cli_report_writes_custom_bundle(monkeypatch, tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "question": "echo me",
                "final_answer": "echo me",
                "Level": "1",
            }
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "benchmark": "gaia",
                "split": "validation",
                "limit": 1,
                "output_root": str(tmp_path / "runs"),
                "executor": "echo",
                "continue_on_error": True,
                "benchmark_options": {"parquet_path": "dummy"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run", "--quiet", "--no-report", "--config", str(config_path)])
    assert exit_code == 0

    run_dirs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1

    output_path = tmp_path / "custom_report.html"
    report_code = cli.main(["report", "--run-dir", str(run_dirs[0]), "--output", str(output_path)])
    assert report_code == 0
    assert output_path.exists()
    sample_pages_dir = tmp_path / "custom_report_samples"
    assert sample_pages_dir.exists()
    assert any(sample_pages_dir.iterdir())


def test_cli_run_forwards_step_summary_flags(monkeypatch, tmp_path: Path, capsys):
    config_path = tmp_path / "cfg.json"
    config_path.write_text("{}", encoding="utf-8")
    captured = {}

    def _fake_run_evaluation(config, progress_callback=None):
        del config, progress_callback
        return SimpleNamespace(run_dir=str(tmp_path / "runs" / "run1"), accuracy=1.0, correct=1, total=1, error_count=0)

    def _fake_generate_default_run_report(run_dir, **kwargs):
        captured["run_dir"] = run_dir
        captured.update(kwargs)
        return str(tmp_path / "runs" / "run1" / "report.html")

    monkeypatch.setattr(cli, "run_evaluation", _fake_run_evaluation)
    monkeypatch.setattr(cli, "_generate_default_run_report", _fake_generate_default_run_report)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--summarize-agent-steps",
            "--step-summary-provider",
            "openai",
            "--step-summary-model",
            "gpt-5-mini",
        ]
    )

    assert exit_code == 0
    assert captured["summarize_agent_steps"] is True
    assert captured["step_summary_provider"] == "openai"
    assert captured["step_summary_model"] == "gpt-5-mini"
    assert captured["run_dir"].endswith("run1")
    capsys.readouterr()


def test_cli_summarize_forwards_step_summary_flags(monkeypatch, capsys):
    captured = {}

    def _fake_summarize_run(run_dir, **kwargs):
        captured["run_dir"] = run_dir
        captured.update(kwargs)
        return {"run_id": "run123"}

    monkeypatch.setattr(cli, "summarize_run", _fake_summarize_run)

    exit_code = cli.main(
        [
            "summarize",
            "--run-dir",
            "eval/runs/run123",
            "--summarize-agent-steps",
            "--step-summary-provider",
            "gemini",
            "--step-summary-model",
            "gemini-3-pro-preview",
        ]
    )

    assert exit_code == 0
    assert captured["run_dir"] == "eval/runs/run123"
    assert captured["summarize_agent_steps"] is True
    assert captured["step_summary_provider"] == "gemini"
    assert captured["step_summary_model"] == "gemini-3-pro-preview"
    printed = json.loads(capsys.readouterr().out)
    assert printed["run_id"] == "run123"


def test_cli_report_forwards_step_summary_flags(monkeypatch, tmp_path: Path, capsys):
    captured = {}
    output_path = tmp_path / "custom_report.html"

    def _fake_generate_html_report(run_dir, output, **kwargs):
        captured["run_dir"] = run_dir
        captured["output"] = output
        captured.update(kwargs)
        return str(output_path)

    monkeypatch.setattr(cli, "generate_html_report", _fake_generate_html_report)

    exit_code = cli.main(
        [
            "report",
            "--run-dir",
            "eval/runs/run123",
            "--output",
            str(output_path),
            "--summarize-agent-steps",
            "--step-summary-provider",
            "openai",
            "--step-summary-model",
            "gpt-5-mini",
        ]
    )

    assert exit_code == 0
    assert captured["run_dir"] == "eval/runs/run123"
    assert captured["output"] == str(output_path)
    assert captured["summarize_agent_steps"] is True
    assert captured["step_summary_provider"] == "openai"
    assert captured["step_summary_model"] == "gpt-5-mini"
    assert "custom_report.html" in capsys.readouterr().out
