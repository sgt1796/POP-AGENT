from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_build.agent1 import scheduled_runner


def test_main_list_prints_tasks(monkeypatch, capsys):
    manager = SimpleNamespace(
        list_scheduled_tasks=lambda include_history=False: [
            {
                "id": "task-1",
                "task_name": "digest",
                "schedule_type": "cron",
                "enabled": True,
                "next_run_at_utc": "2030-01-01T00:00:00Z",
            }
        ]
    )
    monkeypatch.setattr(scheduled_runner, "_build_manager_agent", lambda: manager)

    exit_code = scheduled_runner.main(["--list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "task-1 name=digest schedule=cron enabled=True" in captured.out


def test_execute_task_runs_runtime_lifecycle(monkeypatch):
    calls = []
    async def _fake_execute_scheduled_task(task, **kwargs):
        assert task == {"id": "abc123", "prompt": "run prompt"}
        assert kwargs["enable_event_logger"] is False
        assert kwargs["overrides"].bash_prompt_approval is False
        assert callable(kwargs["on_warning"])
        calls.append("execute")
        return {"status": "success", "summary": "assistant-reply"}

    monkeypatch.setattr(scheduled_runner, "execute_scheduled_task", _fake_execute_scheduled_task)

    result = asyncio.run(scheduled_runner._execute_task({"id": "abc123", "prompt": "run prompt"}))

    assert result["status"] == "success"
    assert result["summary"] == "assistant-reply"
    assert calls == ["execute"]


def test_execute_task_shutdown_runs_on_turn_failure(monkeypatch):
    async def _fake_execute_scheduled_task(task, **kwargs):
        del task, kwargs
        return {"status": "error", "summary": "boom", "error": "boom"}

    monkeypatch.setattr(scheduled_runner, "execute_scheduled_task", _fake_execute_scheduled_task)

    result = asyncio.run(scheduled_runner._execute_task({"id": "abc123", "prompt": "run prompt"}))

    assert result["status"] == "error"
    assert "boom" in result["summary"]


def test_run_due_once_passes_max_parallel_and_returns_non_zero_on_errors(monkeypatch, capsys):
    manager = SimpleNamespace()
    seen = []
    monkeypatch.setattr(scheduled_runner, "_build_manager_agent", lambda: manager)

    async def _fake_run_due_scheduled_tasks(agent, *, max_parallel=3, on_warning=None):
        assert agent is manager
        assert callable(on_warning)
        seen.append(max_parallel)
        return {
            "due_count": 2,
            "success_count": 1,
            "error_count": 1,
            "max_parallel": max_parallel,
            "tasks": [
                {"id": "task-1", "status": "success", "summary": "ok"},
                {"id": "task-2", "status": "error", "summary": "failed"},
            ],
        }

    monkeypatch.setattr(scheduled_runner, "run_due_scheduled_tasks", _fake_run_due_scheduled_tasks)

    exit_code = asyncio.run(scheduled_runner._run_due_once(max_parallel=3))
    captured = capsys.readouterr()

    assert seen == [3]
    assert exit_code == 1
    assert "due=2 success=1 error=1" in captured.out


def test_main_defaults_max_parallel_to_three(monkeypatch):
    seen = []

    async def _fake_run_due_once(max_parallel: int):
        seen.append(max_parallel)
        return 0

    monkeypatch.setattr(scheduled_runner, "_run_due_once", _fake_run_due_once)

    exit_code = scheduled_runner.main([])

    assert exit_code == 0
    assert seen == [3]


def test_get_daemon_status_reports_stale_pid_file(tmp_path, monkeypatch):
    pid_file = tmp_path / "scheduled_runner.pid"
    pid_file.write_text("4321\n", encoding="utf-8")

    monkeypatch.setattr(scheduled_runner, "_process_is_running", lambda pid: False)

    status = scheduled_runner.get_daemon_status(
        project_root=str(tmp_path),
        pid_file=str(pid_file),
    )

    assert status["running"] is False
    assert status["stale_pid_file"] is True
    assert status["pid"] == 4321


def test_get_daemon_status_prefers_invoked_python_path(tmp_path, monkeypatch):
    pid_file = tmp_path / "scheduled_runner.pid"
    pid_file.write_text("4321\n", encoding="utf-8")
    invoked_python = str(tmp_path / ".venv" / "bin" / "python")

    monkeypatch.setattr(scheduled_runner, "_process_is_running", lambda pid: True)
    monkeypatch.setattr(scheduled_runner, "_process_command_path", lambda pid: invoked_python)
    monkeypatch.setattr(scheduled_runner, "_process_executable", lambda pid: "/usr/bin/python3")

    status = scheduled_runner.get_daemon_status(
        project_root=str(tmp_path),
        pid_file=str(pid_file),
    )

    assert status["running"] is True
    assert status["python_executable"] == invoked_python


def test_main_daemon_status_prints_status(monkeypatch, capsys):
    monkeypatch.setattr(
        scheduled_runner,
        "get_daemon_status",
        lambda **kwargs: {
            "ok": True,
            "running": True,
            "pid": 2468,
            "pid_file": "/tmp/scheduled_runner.pid",
            "log_file": "/tmp/scheduled_runner.log",
            "stale_pid_file": False,
        },
    )

    exit_code = scheduled_runner.main(["--daemon-status"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "daemon running pid=2468" in captured.out


def test_main_daemon_start_invokes_start_helper(monkeypatch, capsys):
    monkeypatch.setattr(
        scheduled_runner,
        "start_daemon",
        lambda **kwargs: {
            "ok": True,
            "started": True,
            "already_running": False,
            "running": True,
            "pid": 1357,
            "pid_file": "/tmp/scheduled_runner.pid",
            "log_file": "/tmp/scheduled_runner.log",
            "stale_pid_file": False,
        },
    )

    exit_code = scheduled_runner.main(["--daemon-start"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "daemon running pid=1357" in captured.out


def test_start_daemon_uses_current_python_executable(monkeypatch, tmp_path):
    desired_python = str(tmp_path / "venv" / "bin" / "python")
    launched = {"started": False}
    commands = []

    def _fake_get_daemon_status(**kwargs):
        pid_file = str(kwargs["pid_file"])
        log_file = str(kwargs["log_file"])
        status = {
            "ok": True,
            "pid_file": pid_file,
            "log_file": log_file,
            "stale_pid_file": False,
            "project_root": str(tmp_path),
        }
        if launched["started"]:
            return {
                **status,
                "running": True,
                "pid": 777,
                "python_executable": scheduled_runner.os.path.realpath(desired_python),
            }
        return {
            **status,
            "running": False,
            "pid": None,
            "python_executable": None,
        }

    class _FakePopen:
        def __init__(self, command, **kwargs):
            del kwargs
            commands.append(list(command))
            launched["started"] = True
            self.pid = 777

        def poll(self):
            return None

    monkeypatch.setattr(scheduled_runner.sys, "executable", desired_python)
    monkeypatch.setattr(scheduled_runner, "get_daemon_status", _fake_get_daemon_status)
    monkeypatch.setattr(scheduled_runner.subprocess, "Popen", _FakePopen)

    info = scheduled_runner.start_daemon(project_root=str(tmp_path))

    assert info["ok"] is True
    assert info["started"] is True
    assert commands[0][0] == scheduled_runner.os.path.realpath(desired_python)
    assert info["python_executable"] == scheduled_runner.os.path.realpath(desired_python)


def test_start_daemon_preserves_virtualenv_python_path(monkeypatch, tmp_path):
    desired_python = str(tmp_path / ".venv" / "bin" / "python")
    dereferenced_python = str(tmp_path / "usr" / "bin" / "python3")
    launched = {"started": False}
    commands = []
    real_realpath = scheduled_runner.os.path.realpath

    def _fake_realpath(value):
        if value == desired_python:
            return dereferenced_python
        return real_realpath(value)

    def _fake_get_daemon_status(**kwargs):
        pid_file = str(kwargs["pid_file"])
        log_file = str(kwargs["log_file"])
        status = {
            "ok": True,
            "pid_file": pid_file,
            "log_file": log_file,
            "stale_pid_file": False,
            "project_root": str(tmp_path),
        }
        if launched["started"]:
            return {
                **status,
                "running": True,
                "pid": 778,
                "python_executable": desired_python,
            }
        return {
            **status,
            "running": False,
            "pid": None,
            "python_executable": None,
        }

    class _FakePopen:
        def __init__(self, command, **kwargs):
            del kwargs
            commands.append(list(command))
            launched["started"] = True
            self.pid = 778

        def poll(self):
            return None

    monkeypatch.setattr(scheduled_runner.os.path, "realpath", _fake_realpath)
    monkeypatch.setattr(scheduled_runner.sys, "executable", desired_python)
    monkeypatch.setattr(scheduled_runner, "get_daemon_status", _fake_get_daemon_status)
    monkeypatch.setattr(scheduled_runner.subprocess, "Popen", _FakePopen)

    info = scheduled_runner.start_daemon(project_root=str(tmp_path))

    assert info["ok"] is True
    assert info["started"] is True
    assert commands[0][0] == desired_python
    assert info["python_executable"] == desired_python


def test_start_daemon_restarts_running_daemon_when_python_executable_differs(monkeypatch, tmp_path):
    desired_python = str(tmp_path / "venv" / "bin" / "python")
    running_python = str(tmp_path / "old-env" / "bin" / "python")
    state = {"stopped": False, "started": False}
    commands = []
    stop_calls = []

    def _fake_get_daemon_status(**kwargs):
        pid_file = str(kwargs["pid_file"])
        log_file = str(kwargs["log_file"])
        base = {
            "ok": True,
            "pid_file": pid_file,
            "log_file": log_file,
            "stale_pid_file": False,
            "project_root": str(tmp_path),
        }
        if not state["stopped"]:
            return {
                **base,
                "running": True,
                "pid": 111,
                "python_executable": scheduled_runner.os.path.realpath(running_python),
            }
        if not state["started"]:
            return {
                **base,
                "running": False,
                "pid": None,
                "python_executable": None,
            }
        return {
            **base,
            "running": True,
            "pid": 222,
            "python_executable": scheduled_runner.os.path.realpath(desired_python),
        }

    def _fake_stop_daemon(**kwargs):
        stop_calls.append(dict(kwargs))
        state["stopped"] = True
        return {
            "ok": True,
            "stopped": True,
            "already_stopped": False,
            "running": False,
            "pid": 111,
            "pid_file": str(kwargs["pid_file"]),
            "log_file": str(kwargs["log_file"]),
            "stale_pid_file": False,
            "python_executable": scheduled_runner.os.path.realpath(running_python),
        }

    class _FakePopen:
        def __init__(self, command, **kwargs):
            del kwargs
            commands.append(list(command))
            state["started"] = True
            self.pid = 222

        def poll(self):
            return None

    monkeypatch.setattr(scheduled_runner, "get_daemon_status", _fake_get_daemon_status)
    monkeypatch.setattr(scheduled_runner, "stop_daemon", _fake_stop_daemon)
    monkeypatch.setattr(scheduled_runner.subprocess, "Popen", _FakePopen)

    info = scheduled_runner.start_daemon(
        project_root=str(tmp_path),
        python_executable=desired_python,
    )

    assert info["ok"] is True
    assert info["started"] is True
    assert stop_calls == [
        {
            "project_root": str(tmp_path),
            "pid_file": str(tmp_path / "agent" / "mem" / "scheduled_runner.pid"),
            "log_file": str(tmp_path / "agent" / "mem" / "scheduled_runner.log"),
        }
    ]
    assert commands[0][0] == scheduled_runner.os.path.realpath(desired_python)


def test_main_daemon_stop_invokes_stop_helper(monkeypatch, capsys):
    monkeypatch.setattr(
        scheduled_runner,
        "stop_daemon",
        lambda **kwargs: {
            "ok": True,
            "stopped": True,
            "already_stopped": False,
            "running": False,
            "pid": 8642,
            "pid_file": "/tmp/scheduled_runner.pid",
            "log_file": "/tmp/scheduled_runner.log",
            "stale_pid_file": False,
        },
    )

    exit_code = scheduled_runner.main(["--daemon-stop"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "daemon stopped pid_file=/tmp/scheduled_runner.pid" in captured.out


def test_run_daemon_loop_exits_when_idle(tmp_path, monkeypatch, capsys):
    calls = []

    async def _fake_run_due_report(_max_parallel):
        calls.append("run_due_report")
        return {
            "due_count": 1,
            "success_count": 1,
            "error_count": 0,
            "max_parallel": 1,
            "tasks": [{"id": "task-1", "status": "success", "summary": "done"}],
            "enabled_count": 0,
        }

    monkeypatch.setattr(scheduled_runner, "_run_due_report", _fake_run_due_report)

    pid_file = tmp_path / "scheduled_runner.pid"
    log_file = tmp_path / "scheduled_runner.log"
    exit_code = asyncio.run(
        scheduled_runner._run_daemon_loop(
            max_parallel=1,
            poll_interval_s=1.0,
            project_root=str(tmp_path),
            pid_file=str(pid_file),
            log_file=str(log_file),
            exit_when_idle=True,
        )
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == ["run_due_report"]
    assert pid_file.exists() is False
    assert "daemon idle" in captured.out
