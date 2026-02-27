import json
from pathlib import Path

import pandas as pd

from eval import cli


def test_cli_run_and_summarize_smoke(monkeypatch, tmp_path: Path):
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

    run_dirs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1

    summarize_code = cli.main(["summarize", "--run-dir", str(run_dirs[0])])
    assert summarize_code == 0
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
