from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
import re
from typing import Any, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from agent import Agent
    from agent_build.agent1.runtime import RuntimeOverrides, execute_scheduled_task, run_due_scheduled_tasks
else:
    from agent import Agent
    from .runtime import RuntimeOverrides, execute_scheduled_task, run_due_scheduled_tasks


DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_MAX_PARALLEL = 3
DEFAULT_DAEMON_PID_PATH = os.path.join("agent", "mem", "scheduled_runner.pid")
DEFAULT_DAEMON_LOG_PATH = os.path.join("agent", "mem", "scheduled_runner.log")
_DAEMON_START_TIMEOUT_S = 3.0
_WINDOWS_DRIVE_PATH_PATTERN = re.compile(r"^(?P<drive>[A-Za-z]):[\\/]*(?P<rest>.*)$")
_WSL_DRIVE_PATH_PATTERN = re.compile(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$")


def _print_warning(text: str) -> None:
    print(str(text), file=sys.stderr)


async def _noop_stream(_model: Dict[str, Any], _context: Dict[str, Any], _options: Dict[str, Any]) -> Any:
    raise RuntimeError("stream_fn should not be called by scheduled_runner")


async def _execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
    return await execute_scheduled_task(
        task,
        enable_event_logger=False,
        overrides=RuntimeOverrides(
            bash_prompt_approval=False,
        ),
        on_warning=_print_warning,
    )


def _project_root(value: Optional[str] = None) -> str:
    return os.path.realpath(_normalize_platform_path(value or os.getcwd()))


def _normalize_platform_path(value: Optional[str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if os.name == "nt":
        wsl_match = _WSL_DRIVE_PATH_PATTERN.match(text.replace("\\", "/"))
        if wsl_match is not None:
            drive = str(wsl_match.group("drive") or "").upper()
            rest = str(wsl_match.group("rest") or "").replace("/", "\\").lstrip("\\")
            return f"{drive}:\\" if not rest else f"{drive}:\\{rest}"
        return text
    windows_match = _WINDOWS_DRIVE_PATH_PATTERN.match(text)
    if windows_match is None:
        return text
    drive = str(windows_match.group("drive") or "").lower()
    rest = str(windows_match.group("rest") or "").replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}" if not rest else f"/mnt/{drive}/{rest}"


def _resolve_project_path(project_root: str, value: Optional[str], default_relpath: str) -> str:
    text = _normalize_platform_path(value)
    candidate = text or os.path.join(project_root, default_relpath)
    if not os.path.isabs(candidate):
        candidate = os.path.join(project_root, candidate)
    return os.path.realpath(candidate)


def _normalize_executable_path(value: Optional[str]) -> str:
    candidate = _normalize_platform_path(value)
    if not candidate:
        return ""
    if candidate and not os.path.isabs(candidate):
        resolved = shutil.which(candidate)
        if resolved:
            candidate = resolved
    return os.path.abspath(candidate) if candidate else ""


def _virtualenv_python_path(env_root: Optional[str]) -> str:
    root = _normalize_platform_path(env_root)
    if not root:
        return ""
    candidates = [
        os.path.join(root, "bin", "python"),
        os.path.join(root, "bin", "python3"),
        os.path.join(root, "Scripts", "python.exe"),
        os.path.join(root, "Scripts", "python"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return ""


def _python_environment_identity(value: Optional[str]) -> str:
    candidate = _normalize_executable_path(value)
    if not candidate:
        return ""
    parent = os.path.dirname(candidate)
    if os.path.basename(parent) in {"bin", "Scripts"}:
        env_root = os.path.dirname(parent)
        if os.path.isfile(os.path.join(env_root, "pyvenv.cfg")):
            return f"venv:{os.path.realpath(env_root)}"
    return f"python:{os.path.realpath(candidate)}"


def _resolve_python_executable(value: Optional[str] = None, *, project_root: Optional[str] = None) -> str:
    explicit = _normalize_executable_path(value)
    if explicit:
        return explicit
    current_python = _normalize_executable_path(getattr(sys, "executable", ""))
    if current_python:
        return current_python
    active_env_python = _virtualenv_python_path(os.environ.get("VIRTUAL_ENV"))
    if active_env_python:
        return active_env_python
    root = _project_root(project_root)
    project_env_python = _virtualenv_python_path(os.path.join(root, ".venv"))
    if project_env_python:
        return project_env_python
    return current_python


def _read_pid_file(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    try:
        pid = int(raw)
    except Exception:
        return None
    return pid if pid > 0 else None


def _process_is_running(pid: int) -> bool:
    if int(pid or 0) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    except Exception:
        return False
    return True


def _process_executable(pid: int) -> Optional[str]:
    if int(pid or 0) <= 0:
        return None
    if os.name == "nt":
        return None
    try:
        resolved = os.readlink(f"/proc/{int(pid)}/exe")
    except Exception:
        return None
    text = str(resolved or "").strip()
    return os.path.realpath(text) if text else None


def _process_command_path(pid: int) -> Optional[str]:
    if int(pid or 0) <= 0:
        return None
    if os.name == "nt":
        return None
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as handle:
            raw = handle.read().split(b"\0", 1)[0]
    except Exception:
        return None
    text = raw.decode("utf-8", errors="ignore").strip()
    normalized = _normalize_executable_path(text)
    return normalized or None


def _remove_pid_file(path: str, *, expected_pid: Optional[int] = None) -> None:
    if expected_pid is not None:
        current_pid = _read_pid_file(path)
        if current_pid not in {None, int(expected_pid)}:
            return
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def _claim_pid_file(path: str, *, pid: int) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{int(pid)}\n")


def get_daemon_status(
    *,
    project_root: Optional[str] = None,
    pid_file: Optional[str] = None,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    root = _project_root(project_root)
    pid_path = _resolve_project_path(root, pid_file, DEFAULT_DAEMON_PID_PATH)
    log_path = _resolve_project_path(root, log_file, DEFAULT_DAEMON_LOG_PATH)
    pid = _read_pid_file(pid_path)
    pid_file_exists = os.path.exists(pid_path)
    running = bool(pid and _process_is_running(pid))
    stale_pid_file = bool(pid_file_exists and not running)
    python_executable = None
    if running:
        python_executable = _process_command_path(int(pid or 0)) or _process_executable(int(pid or 0))
    return {
        "ok": True,
        "running": running,
        "pid": pid,
        "pid_file": pid_path,
        "log_file": log_path,
        "stale_pid_file": stale_pid_file,
        "python_executable": python_executable,
        "project_root": root,
    }


def start_daemon(
    *,
    project_root: Optional[str] = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    pid_file: Optional[str] = None,
    log_file: Optional[str] = None,
    exit_when_idle: bool = False,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    root = _project_root(project_root)
    pid_path = _resolve_project_path(root, pid_file, DEFAULT_DAEMON_PID_PATH)
    log_path = _resolve_project_path(root, log_file, DEFAULT_DAEMON_LOG_PATH)
    resolved_python = _resolve_python_executable(python_executable, project_root=root)
    desired_python_identity = _python_environment_identity(resolved_python)
    status = get_daemon_status(project_root=root, pid_file=pid_path, log_file=log_path)
    running_python_raw = str(status.get("python_executable") or "").strip()
    running_python = _normalize_executable_path(running_python_raw) if running_python_raw else ""
    running_python_identity = _python_environment_identity(running_python) if running_python else ""
    if bool(status.get("running")):
        if running_python_identity and desired_python_identity and running_python_identity != desired_python_identity:
            stopped = stop_daemon(project_root=root, pid_file=pid_path, log_file=log_path)
            if not bool(stopped.get("ok")):
                return {
                    "ok": False,
                    "started": False,
                    "already_running": False,
                    "pid": status.get("pid"),
                    "pid_file": pid_path,
                    "log_file": log_path,
                    "project_root": root,
                    "python_executable": resolved_python,
                    "running_python_executable": running_python,
                    "error": f"daemon_python_mismatch_stop_failed:{stopped.get('error') or 'unknown_error'}",
                }
            status = get_daemon_status(project_root=root, pid_file=pid_path, log_file=log_path)
        else:
            return {
                "ok": True,
                "started": False,
                "already_running": True,
                "python_executable": running_python or resolved_python,
                **status,
            }
    if bool(status.get("stale_pid_file")):
        _remove_pid_file(pid_path)

    poll_value = poll_interval_s
    try:
        poll_value = float(poll_interval_s)
    except Exception:
        poll_value = DEFAULT_POLL_INTERVAL_S
    poll_value = poll_value if poll_value > 0 else DEFAULT_POLL_INTERVAL_S
    parallel = max(1, int(max_parallel or DEFAULT_MAX_PARALLEL))

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    command = [
        resolved_python,
        "-u",
        "-m",
        "agent_build.agent1.scheduled_runner",
        "--daemon-run",
        "--max-parallel",
        str(parallel),
        "--poll-interval",
        str(poll_value),
        "--pid-file",
        pid_path,
        "--log-file",
        log_path,
    ]
    if exit_when_idle:
        command.append("--exit-when-idle")

    creationflags = 0
    popen_kwargs: Dict[str, Any] = {
        "cwd": root,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
        "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
    }
    if os.name == "nt":
        detached_process = int(getattr(subprocess, "DETACHED_PROCESS", 0))
        new_group = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        creationflags = detached_process | new_group
        if creationflags:
            popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    with open(log_path, "a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=handle, **popen_kwargs)

    deadline = time.time() + _DAEMON_START_TIMEOUT_S
    while time.time() < deadline:
        status = get_daemon_status(project_root=root, pid_file=pid_path, log_file=log_path)
        if bool(status.get("running")):
            return {
                "ok": True,
                "started": True,
                "already_running": False,
                "spawn_pid": int(process.pid or 0),
                "python_executable": resolved_python,
                **status,
            }
        if process.poll() is not None:
            break
        time.sleep(0.05)

    return {
        "ok": False,
        "started": False,
        "already_running": False,
        "pid": _read_pid_file(pid_path) or int(process.pid or 0),
        "pid_file": pid_path,
        "log_file": log_path,
        "project_root": root,
        "python_executable": resolved_python,
        "error": "daemon_failed_to_start",
    }


def stop_daemon(
    *,
    project_root: Optional[str] = None,
    pid_file: Optional[str] = None,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    status = get_daemon_status(project_root=project_root, pid_file=pid_file, log_file=log_file)
    pid = int(status.get("pid") or 0)
    pid_path = str(status.get("pid_file") or "")
    if not bool(status.get("running")):
        if bool(status.get("stale_pid_file")):
            _remove_pid_file(pid_path)
        return {"ok": True, "stopped": False, "already_stopped": True, **status}

    try:
        if os.name == "nt":
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if completed.returncode != 0:
                return {
                    "ok": False,
                    "stopped": False,
                    "already_stopped": False,
                    **status,
                    "error": f"taskkill_failed:{completed.returncode}",
                }
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception as exc:
        return {
            "ok": False,
            "stopped": False,
            "already_stopped": False,
            **status,
            "error": str(exc),
        }

    deadline = time.time() + _DAEMON_START_TIMEOUT_S
    while time.time() < deadline:
        if not _process_is_running(pid):
            _remove_pid_file(pid_path)
            return {
                "ok": True,
                "stopped": True,
                "already_stopped": False,
                **status,
            }
        time.sleep(0.05)

    return {
        "ok": False,
        "stopped": False,
        "already_stopped": False,
        **status,
        "error": "daemon_did_not_exit",
    }


def _format_daemon_status(status: Dict[str, Any]) -> str:
    pid_file = str(status.get("pid_file") or "")
    log_file = str(status.get("log_file") or "")
    pid = status.get("pid")
    if bool(status.get("running")):
        return f"[scheduled_runner] daemon running pid={pid} pid_file={pid_file} log_file={log_file}"
    if bool(status.get("stale_pid_file")):
        return f"[scheduled_runner] daemon stopped stale_pid_file={pid_file} log_file={log_file}"
    return f"[scheduled_runner] daemon stopped pid_file={pid_file} log_file={log_file}"


def _print_task_list(tasks: List[Dict[str, Any]]) -> None:
    if not tasks:
        print("[scheduled_runner] no scheduled tasks")
        return
    for task in tasks:
        task_id = str(task.get("id") or "")
        name = str(task.get("task_name") or "")
        schedule_type = str(task.get("schedule_type") or "")
        next_run = str(task.get("next_run_at_utc") or "")
        enabled = bool(task.get("enabled"))
        print(
            f"{task_id} name={name} schedule={schedule_type} enabled={enabled} next_run_at_utc={next_run}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run due scheduled agent tasks or manage the persistent runner.")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help=f"Maximum concurrent due tasks (default: {DEFAULT_MAX_PARALLEL})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help=f"Daemon poll interval in seconds (default: {DEFAULT_POLL_INTERVAL_S:g})",
    )
    parser.add_argument("--list", action="store_true", help="List configured scheduled tasks and exit")
    parser.add_argument("--include-history", action="store_true", help="Include task history with --list")
    parser.add_argument("--daemon-start", action="store_true", help="Start the detached persistent scheduler daemon")
    parser.add_argument("--daemon-stop", action="store_true", help="Stop the detached persistent scheduler daemon")
    parser.add_argument("--daemon-status", action="store_true", help="Show detached scheduler daemon status")
    parser.add_argument("--daemon-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exit-when-idle", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pid-file", type=str, default="", help="Override the daemon PID file path")
    parser.add_argument("--log-file", type=str, default="", help="Override the daemon log file path")
    return parser


def _build_manager_agent() -> Agent:
    return Agent(
        {
            "stream_fn": _noop_stream,
            "project_root": _project_root(),
        }
    )


async def _run_due_report(max_parallel: int) -> Dict[str, Any]:
    manager = _build_manager_agent()
    return await run_due_scheduled_tasks(manager, max_parallel=max_parallel, on_warning=_print_warning)


def _print_due_report(report: Dict[str, Any], *, print_empty: bool) -> int:
    due_count = int(report.get("due_count", 0) or 0)
    success_count = int(report.get("success_count", 0) or 0)
    error_count = int(report.get("error_count", 0) or 0)
    if due_count <= 0:
        if print_empty:
            print("[scheduled_runner] no due tasks")
        return 0

    print(
        "[scheduled_runner] completed "
        f"due={due_count} success={success_count} error={error_count} "
        f"max_parallel={int(report.get('max_parallel', 0) or 0)}"
    )
    for outcome in list(report.get("tasks", []) or []):
        task_id = str(outcome.get("id") or "")
        status = str(outcome.get("status") or "")
        summary = str(outcome.get("summary") or "").replace("\n", " ").strip()
        print(f"- id={task_id} status={status} summary={summary}")
    return 0 if error_count == 0 else 1


async def _run_due_once(max_parallel: int) -> int:
    try:
        report = await _run_due_report(max_parallel)
    except Exception as exc:
        _print_warning(f"[scheduled_runner] failed: {exc}")
        return 1
    return _print_due_report(report, print_empty=True)


def _install_stop_signal_handlers(stop_event: asyncio.Event) -> List[Tuple[int, Any]]:
    previous: List[Tuple[int, Any]] = []

    def _handle_stop(_signum: int, _frame: Any) -> None:
        if not stop_event.is_set():
            stop_event.set()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous.append((sig, signal.getsignal(sig)))
            signal.signal(sig, _handle_stop)
        except Exception:
            continue
    return previous


def _restore_stop_signal_handlers(previous: List[Tuple[int, Any]]) -> None:
    for sig, handler in previous:
        with contextlib.suppress(Exception):
            signal.signal(sig, handler)


async def _run_daemon_loop(
    *,
    max_parallel: int,
    poll_interval_s: float,
    project_root: Optional[str] = None,
    pid_file: Optional[str] = None,
    log_file: Optional[str] = None,
    exit_when_idle: bool = False,
) -> int:
    root = _project_root(project_root)
    pid_path = _resolve_project_path(root, pid_file, DEFAULT_DAEMON_PID_PATH)
    log_path = _resolve_project_path(root, log_file, DEFAULT_DAEMON_LOG_PATH)
    current_pid = os.getpid()
    status = get_daemon_status(project_root=root, pid_file=pid_path, log_file=log_path)
    if bool(status.get("running")) and int(status.get("pid") or 0) != current_pid:
        _print_warning(
            f"[scheduled_runner] daemon already running pid={status.get('pid')} pid_file={pid_path}"
        )
        return 1
    if bool(status.get("stale_pid_file")):
        _remove_pid_file(pid_path)
    try:
        _claim_pid_file(pid_path, pid=current_pid)
    except FileExistsError:
        status = get_daemon_status(project_root=root, pid_file=pid_path, log_file=log_path)
        if bool(status.get("running")) and int(status.get("pid") or 0) != current_pid:
            _print_warning(
                f"[scheduled_runner] daemon already running pid={status.get('pid')} pid_file={pid_path}"
            )
            return 1
        _remove_pid_file(pid_path)
        _claim_pid_file(pid_path, pid=current_pid)

    stop_event: asyncio.Event = asyncio.Event()
    previous_handlers = _install_stop_signal_handlers(stop_event)

    def _cleanup() -> None:
        _restore_stop_signal_handlers(previous_handlers)
        _remove_pid_file(pid_path, expected_pid=current_pid)

    atexit.register(_cleanup)
    try:
        print(
            "[scheduled_runner] daemon started "
            f"pid={current_pid} poll_interval={poll_interval_s:g}s max_parallel={max_parallel} "
            f"pid_file={pid_path} log_file={log_path} python_executable={_resolve_python_executable()}"
        )
        while not stop_event.is_set():
            try:
                report = await _run_due_report(max_parallel)
                _print_due_report(report, print_empty=False)
                enabled_count = report.get("enabled_count")
                if exit_when_idle and enabled_count is not None and int(enabled_count or 0) <= 0:
                    print(f"[scheduled_runner] daemon idle pid={current_pid} enabled=0")
                    stop_event.set()
            except Exception as exc:
                _print_warning(f"[scheduled_runner] daemon iteration failed: {exc}")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_s)
            except asyncio.TimeoutError:
                continue
        print(f"[scheduled_runner] daemon stopping pid={current_pid}")
        return 0
    finally:
        _cleanup()
        with contextlib.suppress(Exception):
            atexit.unregister(_cleanup)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    max_parallel = max(1, int(args.max_parallel or DEFAULT_MAX_PARALLEL))
    poll_interval_s = float(args.poll_interval or DEFAULT_POLL_INTERVAL_S)
    pid_file = str(args.pid_file or "").strip() or None
    log_file = str(args.log_file or "").strip() or None

    if args.daemon_status:
        status = get_daemon_status(pid_file=pid_file, log_file=log_file)
        print(_format_daemon_status(status))
        return 0

    if args.daemon_start:
        started = start_daemon(
            max_parallel=max_parallel,
            poll_interval_s=poll_interval_s,
            pid_file=pid_file,
            log_file=log_file,
            exit_when_idle=args.exit_when_idle,
        )
        if bool(started.get("ok")):
            print(_format_daemon_status(started))
            return 0
        error = str(started.get("error") or "unknown_error")
        _print_warning(f"[scheduled_runner] failed to start daemon: {error}")
        return 1

    if args.daemon_stop:
        stopped = stop_daemon(pid_file=pid_file, log_file=log_file)
        print(_format_daemon_status(stopped))
        return 0 if bool(stopped.get("ok")) else 1

    if args.daemon_run:
        return asyncio.run(
            _run_daemon_loop(
                max_parallel=max_parallel,
                poll_interval_s=poll_interval_s,
                pid_file=pid_file,
                log_file=log_file,
                exit_when_idle=args.exit_when_idle,
            )
        )

    if args.list:
        manager = _build_manager_agent()
        tasks = manager.list_scheduled_tasks(include_history=bool(args.include_history))
        _print_task_list(tasks)
        return 0

    return asyncio.run(_run_due_once(max_parallel=max_parallel))


if __name__ == "__main__":
    raise SystemExit(main())
