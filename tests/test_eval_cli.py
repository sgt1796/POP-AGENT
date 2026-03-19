import json
import shutil
from pathlib import Path

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
