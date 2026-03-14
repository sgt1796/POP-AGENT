from __future__ import annotations

import asyncio
import copy
import json
import ntpath
import os
import posixpath
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

DEFAULT_SCHEDULED_JOBS_PATH = os.path.join("agent", "mem", "scheduled_jobs.json")
DEFAULT_RUNNER_LOCK_PATH = os.path.join("agent", "mem", "scheduled_jobs.lock")
SCHEMA_VERSION = 1
DEFAULT_MAX_HISTORY = 20
_TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_WINDOWS_DRIVE_PATH_PATTERN = re.compile(r"^(?P<drive>[A-Za-z]):[\\/]*(?P<rest>.*)$")
_WSL_DRIVE_PATH_PATTERN = re.compile(r"^/mnt/(?P<drive>[A-Za-z])(?:/(?P<rest>.*))?$")


TaskExecutor = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]] | Dict[str, Any] | Any]


class ScheduledTaskError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(str(message or "scheduled task error"))
        self.code = str(code or "scheduled_task_error")
        self.message = str(message or "scheduled task error")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_datetime(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ScheduledTaskError("invalid_datetime", "run_at is required")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception as exc:
        raise ScheduledTaskError("invalid_datetime", f"invalid ISO datetime: {text}") from exc


def _parse_utc_iso(value: str) -> datetime:
    parsed = _parse_iso_datetime(value)
    if parsed.tzinfo is None:
        raise ScheduledTaskError("invalid_datetime", f"timestamp must include timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _common_path(a: str, b: str) -> str:
    try:
        return os.path.commonpath([a, b])
    except ValueError:
        return ""


def _path_style(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    if _WINDOWS_DRIVE_PATH_PATTERN.match(text):
        return "nt"
    if _WSL_DRIVE_PATH_PATTERN.match(text) or text.startswith("/"):
        return "posix"
    return None


def _path_module(platform: Optional[str] = None) -> Any:
    target_platform = str(platform or os.name or "").strip().lower()
    return ntpath if target_platform == "nt" else posixpath


def _normalize_platform_path(value: Any, *, platform: Optional[str] = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    target_platform = str(platform or os.name or "").strip().lower()
    if target_platform == "nt":
        wsl_match = _WSL_DRIVE_PATH_PATTERN.match(text.replace("\\", "/"))
        if wsl_match is not None:
            drive = str(wsl_match.group("drive") or "").upper()
            rest = str(wsl_match.group("rest") or "").replace("/", "\\").lstrip("\\")
            return f"{drive}:\\" if not rest else f"{drive}:\\{rest}"
        return text.replace("/", "\\")
    windows_match = _WINDOWS_DRIVE_PATH_PATTERN.match(text)
    if windows_match is None:
        return text.replace("\\", "/")
    drive = str(windows_match.group("drive") or "").lower()
    rest = str(windows_match.group("rest") or "").replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}" if not rest else f"/mnt/{drive}/{rest}"


def _realpath_compat(value: Any, *, platform: Optional[str] = None) -> str:
    normalized = _normalize_platform_path(value, platform=platform)
    if not normalized:
        return ""
    target_platform = str(platform or os.name or "").strip().lower()
    if target_platform == str(os.name or "").strip().lower():
        return os.path.realpath(normalized)
    return _path_module(target_platform).normpath(normalized)


def _resolve_project_path(
    project_root: str,
    value: Optional[str],
    default_relpath: str,
    *,
    platform: Optional[str] = None,
) -> str:
    pathmod = _path_module(platform)
    text = _normalize_platform_path(value, platform=platform)
    default_text = _normalize_platform_path(default_relpath, platform=platform)
    candidate = text or pathmod.join(project_root, default_text)
    if not pathmod.isabs(candidate):
        candidate = pathmod.join(project_root, candidate)
    return _realpath_compat(candidate, platform=platform)


def _path_within_root(path: str, root: str, *, platform: Optional[str] = None) -> bool:
    pathmod = _path_module(platform)
    normalized_path = pathmod.normcase(_realpath_compat(path, platform=platform))
    normalized_root = pathmod.normcase(_realpath_compat(root, platform=platform))
    try:
        return pathmod.commonpath([normalized_path, normalized_root]) == normalized_root
    except ValueError:
        return False


def _default_timezone_name() -> str:
    local_tz = datetime.now().astimezone().tzinfo
    if isinstance(local_tz, ZoneInfo):
        return local_tz.key
    key = getattr(local_tz, "key", None)
    if isinstance(key, str) and key:
        try:
            ZoneInfo(key)
            return key
        except Exception:
            pass
    if isinstance(local_tz, timezone):
        offset = local_tz.utcoffset(None)
        if offset == timedelta(0):
            return "UTC"
    candidate = str(local_tz or "").strip()
    if candidate:
        try:
            ZoneInfo(candidate)
            return candidate
        except Exception:
            pass
    return "UTC"


def _load_timezone(name: Optional[str]) -> Tuple[str, ZoneInfo]:
    zone_name = str(name or "").strip() or _default_timezone_name()
    try:
        tz = ZoneInfo(zone_name)
    except Exception as exc:
        raise ScheduledTaskError("invalid_timezone", f"unknown timezone: {zone_name}") from exc
    return zone_name, tz


def _normalize_prompt(prompt: Any) -> str:
    text = str(prompt or "").strip()
    if not text:
        raise ScheduledTaskError("invalid_prompt", "prompt is required")
    return text


def _normalize_task_name(task_name: Optional[str], *, task_id: str) -> str:
    text = str(task_name or "").strip()
    if not text:
        return f"task-{task_id[:8]}"
    if not _TASK_NAME_PATTERN.fullmatch(text):
        raise ScheduledTaskError(
            "invalid_task_name",
            "task_name must match [A-Za-z0-9_-] and be 1-64 chars",
        )
    return text


def _compute_next_cron_utc(cron_expr: str, tz: ZoneInfo, *, now_utc: datetime) -> datetime:
    try:
        from croniter import croniter
    except Exception as exc:
        raise ScheduledTaskError("missing_dependency", "croniter is required for cron schedules") from exc

    expr = str(cron_expr or "").strip()
    if not expr:
        raise ScheduledTaskError("invalid_cron", "cron expression is required")
    try:
        base_local = now_utc.astimezone(tz)
        iterator = croniter(expr, base_local)
        next_local = iterator.get_next(datetime)
    except Exception as exc:
        raise ScheduledTaskError("invalid_cron", f"invalid cron expression: {expr}") from exc
    if not isinstance(next_local, datetime):
        raise ScheduledTaskError("invalid_cron", f"invalid cron expression: {expr}")
    if next_local.tzinfo is None:
        next_local = next_local.replace(tzinfo=tz)
    return next_local.astimezone(timezone.utc)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _history_entry(outcome: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "started_at_utc": str(outcome.get("started_at_utc") or ""),
        "finished_at_utc": str(outcome.get("finished_at_utc") or ""),
        "status": str(outcome.get("status") or "error"),
        "summary": str(outcome.get("summary") or ""),
    }


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class ScheduledTaskStore:
    def __init__(
        self,
        *,
        project_root: str,
        jobs_path: Optional[str] = None,
        runner_lock_path: Optional[str] = None,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        root_input = project_root or os.getcwd()
        platform = _path_style(root_input)
        root = _realpath_compat(root_input, platform=platform)
        jobs = _resolve_project_path(root, jobs_path, DEFAULT_SCHEDULED_JOBS_PATH, platform=platform)
        lock = _resolve_project_path(root, runner_lock_path, DEFAULT_RUNNER_LOCK_PATH, platform=platform)

        if not _path_within_root(jobs, root, platform=platform):
            raise ValueError("jobs_path must be within project_root")
        if not _path_within_root(lock, root, platform=platform):
            raise ValueError("runner_lock_path must be within project_root")

        self.project_root = root
        self.jobs_path = jobs
        self.runner_lock_path = lock
        self.max_history = max(1, int(max_history or DEFAULT_MAX_HISTORY))

    def schedule_task(
        self,
        prompt: str,
        *,
        run_at: Optional[str] = None,
        cron: Optional[str] = None,
        timezone_name: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt_text = _normalize_prompt(prompt)
        has_run_at = bool(str(run_at or "").strip())
        has_cron = bool(str(cron or "").strip())
        if has_run_at == has_cron:
            raise ScheduledTaskError("invalid_schedule", "exactly one of run_at or cron must be provided")

        tz_name, tz = _load_timezone(timezone_name)
        now_utc = _utcnow()
        task_id = uuid.uuid4().hex
        normalized_name = _normalize_task_name(task_name, task_id=task_id)

        schedule_type = "one_time"
        run_at_iso: Optional[str] = None
        cron_expr: Optional[str] = None
        if has_run_at:
            schedule_type = "one_time"
            run_at_dt = _parse_iso_datetime(str(run_at or ""))
            if run_at_dt.tzinfo is None:
                run_at_dt = run_at_dt.replace(tzinfo=tz)
            next_run = run_at_dt.astimezone(timezone.utc)
            run_at_iso = run_at_dt.isoformat()
        else:
            schedule_type = "cron"
            cron_expr = str(cron or "").strip()
            next_run = _compute_next_cron_utc(cron_expr, tz, now_utc=now_utc)

        task = {
            "id": task_id,
            "task_name": normalized_name,
            "prompt": prompt_text,
            "schedule_type": schedule_type,
            "run_at": run_at_iso,
            "cron": cron_expr,
            "timezone": tz_name,
            "enabled": True,
            "next_run_at_utc": _to_iso_utc(next_run),
            "created_at_utc": _to_iso_utc(now_utc),
            "updated_at_utc": _to_iso_utc(now_utc),
            "last_run_at_utc": None,
            "run_count": 0,
            "history": [],
        }

        payload = self._load_payload()
        payload["tasks"].append(task)
        self._write_payload(payload)
        return self._public_task(task, include_history=True)

    def list_tasks(self, *, include_history: bool = False) -> List[Dict[str, Any]]:
        payload = self._load_payload()
        tasks = [self._public_task(item, include_history=include_history) for item in payload.get("tasks", [])]
        tasks.sort(key=lambda item: (str(item.get("next_run_at_utc") or ""), str(item.get("id") or "")))
        return tasks

    def remove_task(self, task_id: str) -> bool:
        target = str(task_id or "").strip()
        if not target:
            return False
        payload = self._load_payload()
        tasks = list(payload.get("tasks", []))
        filtered = [task for task in tasks if str(task.get("id", "")).strip() != target]
        if len(filtered) == len(tasks):
            return False
        payload["tasks"] = filtered
        self._write_payload(payload)
        return True

    def mark_task_due_now(self, task_id: str) -> bool:
        target = str(task_id or "").strip()
        if not target:
            return False
        payload = self._load_payload()
        now_utc = _utcnow()
        found = False
        for task in payload.get("tasks", []):
            if str(task.get("id", "")).strip() != target:
                continue
            task["enabled"] = True
            task["next_run_at_utc"] = _to_iso_utc(now_utc)
            task["updated_at_utc"] = _to_iso_utc(now_utc)
            found = True
            break
        if not found:
            return False
        self._write_payload(payload)
        return True

    async def run_due_tasks(
        self,
        executor: TaskExecutor,
        *,
        max_parallel: int = 3,
    ) -> Dict[str, Any]:
        if not callable(executor):
            raise ValueError("executor must be callable")
        parallel = max(1, int(max_parallel or 1))

        with self._runner_lock():
            payload = self._load_payload()
            now_utc = _utcnow()
            due_tasks = self._find_due_tasks(payload.get("tasks", []), now_utc=now_utc)
            if not due_tasks:
                schedule_state = self._schedule_state(payload.get("tasks", []))
                return {
                    "ok": True,
                    "due_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "max_parallel": parallel,
                    "tasks": [],
                    **schedule_state,
                }

            semaphore = asyncio.Semaphore(parallel)

            async def _run_one(task: Dict[str, Any]) -> Dict[str, Any]:
                async with semaphore:
                    started = _utcnow()
                    status = "success"
                    summary = ""
                    result_payload: Dict[str, Any] = {}
                    try:
                        maybe_result = executor(copy.deepcopy(task))
                        if asyncio.iscoroutine(maybe_result) or hasattr(maybe_result, "__await__"):
                            resolved = await maybe_result  # type: ignore[misc]
                        else:
                            resolved = maybe_result
                        if isinstance(resolved, dict):
                            result_payload = dict(resolved)
                        elif resolved is not None:
                            result_payload = {"result": str(resolved)}
                        status_text = str(result_payload.get("status") or "").strip().lower()
                        if status_text == "error":
                            status = "error"
                        summary = str(
                            result_payload.get("summary")
                            or result_payload.get("reply")
                            or result_payload.get("error")
                            or ""
                        ).strip()
                    except Exception as exc:
                        status = "error"
                        summary = str(exc)
                        result_payload = {"error": str(exc)}
                    finished = _utcnow()
                    if not summary:
                        summary = "completed" if status == "success" else "failed"
                    return {
                        "id": str(task.get("id") or ""),
                        "status": status,
                        "summary": summary,
                        "result": result_payload,
                        "started_at_utc": _to_iso_utc(started),
                        "finished_at_utc": _to_iso_utc(finished),
                    }

            outcomes = await asyncio.gather(*[asyncio.create_task(_run_one(task)) for task in due_tasks])
            # Executors can mutate the jobs file (for example by creating follow-up tasks).
            # Reload the latest payload so we do not clobber those writes with the stale pre-run snapshot.
            payload = self._load_payload()
            task_map = {str(item.get("id") or ""): item for item in payload.get("tasks", [])}
            for outcome in outcomes:
                task = task_map.get(str(outcome.get("id") or ""))
                if task is None:
                    continue
                self._apply_task_outcome(task, outcome)
            self._write_payload(payload)
            schedule_state = self._schedule_state(payload.get("tasks", []))

        success_count = sum(1 for item in outcomes if str(item.get("status") or "") == "success")
        error_count = max(0, len(outcomes) - success_count)
        return {
            "ok": error_count == 0,
            "due_count": len(outcomes),
            "success_count": success_count,
            "error_count": error_count,
            "max_parallel": parallel,
            "tasks": outcomes,
            **schedule_state,
        }

    def _find_due_tasks(self, tasks: List[Dict[str, Any]], *, now_utc: datetime) -> List[Dict[str, Any]]:
        due: List[Dict[str, Any]] = []
        for task in tasks:
            if not _coerce_bool(task.get("enabled"), True):
                continue
            next_value = str(task.get("next_run_at_utc") or "").strip()
            if not next_value:
                continue
            try:
                next_dt = _parse_utc_iso(next_value)
            except ScheduledTaskError:
                continue
            if next_dt <= now_utc:
                due.append(copy.deepcopy(task))
        due.sort(key=lambda item: str(item.get("next_run_at_utc") or ""))
        return due

    def _schedule_state(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        task_list = [item for item in tasks if isinstance(item, dict)]
        enabled_tasks = [item for item in task_list if _coerce_bool(item.get("enabled"), True)]
        next_runs = sorted(
            str(item.get("next_run_at_utc") or "").strip()
            for item in enabled_tasks
            if str(item.get("next_run_at_utc") or "").strip()
        )
        return {
            "task_count": len(task_list),
            "enabled_count": len(enabled_tasks),
            "next_run_at_utc": next_runs[0] if next_runs else None,
        }

    def _apply_task_outcome(self, task: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        finished_utc = _parse_utc_iso(str(outcome.get("finished_at_utc") or _to_iso_utc(_utcnow())))
        task["run_count"] = _coerce_int(task.get("run_count"), 0) + 1
        task["last_run_at_utc"] = _to_iso_utc(finished_utc)
        task["updated_at_utc"] = _to_iso_utc(finished_utc)

        history = list(task.get("history") or [])
        history.append(_history_entry(outcome))
        task["history"] = history[-self.max_history :]

        schedule_type = str(task.get("schedule_type") or "").strip()
        if schedule_type == "one_time":
            task["enabled"] = False
            task["next_run_at_utc"] = None
            return

        if schedule_type != "cron":
            task["enabled"] = False
            task["next_run_at_utc"] = None
            return

        try:
            tz_name, tz = _load_timezone(str(task.get("timezone") or "").strip())
            cron_expr = str(task.get("cron") or "").strip()
            next_run = _compute_next_cron_utc(cron_expr, tz, now_utc=finished_utc)
        except ScheduledTaskError as exc:
            task["enabled"] = False
            task["next_run_at_utc"] = None
            history = list(task.get("history") or [])
            history.append(
                {
                    "started_at_utc": _to_iso_utc(finished_utc),
                    "finished_at_utc": _to_iso_utc(finished_utc),
                    "status": "error",
                    "summary": f"cron_disabled: {exc.message}",
                }
            )
            task["history"] = history[-self.max_history :]
            return

        task["timezone"] = tz_name
        task["enabled"] = True
        task["next_run_at_utc"] = _to_iso_utc(next_run)

    @contextmanager
    def _runner_lock(self):
        parent = os.path.dirname(self.runner_lock_path) or "."
        os.makedirs(parent, exist_ok=True)
        fd: Optional[int] = None
        try:
            fd = os.open(self.runner_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = None
                payload = {
                    "pid": os.getpid(),
                    "acquired_at_utc": _to_iso_utc(_utcnow()),
                }
                json.dump(payload, handle, ensure_ascii=True)
                handle.write("\n")
        except FileExistsError as exc:
            raise ScheduledTaskError(
                "runner_locked",
                f"runner lock already exists: {self.runner_lock_path}",
            ) from exc
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass

        try:
            yield
        finally:
            try:
                os.unlink(self.runner_lock_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _load_payload(self) -> Dict[str, Any]:
        path = self.jobs_path
        if not os.path.exists(path):
            return {"schema_version": SCHEMA_VERSION, "tasks": []}

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            raise ScheduledTaskError(
                "invalid_jobs_file",
                f"unable to parse scheduled jobs file '{path}': {exc}",
            ) from exc

        if not isinstance(payload, dict):
            raise ScheduledTaskError("invalid_jobs_file", "scheduled jobs file must contain a JSON object")

        schema_version = _coerce_int(payload.get("schema_version"), SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            raise ScheduledTaskError(
                "unsupported_schema",
                f"unsupported scheduled jobs schema_version: {schema_version}",
            )

        raw_tasks = payload.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raise ScheduledTaskError("invalid_jobs_file", "scheduled jobs file tasks field must be an array")

        tasks: List[Dict[str, Any]] = []
        for raw_task in raw_tasks:
            if not isinstance(raw_task, dict):
                raise ScheduledTaskError("invalid_jobs_file", "scheduled task entries must be objects")
            task = dict(raw_task)
            task_id = str(task.get("id") or "").strip()
            if not task_id:
                raise ScheduledTaskError("invalid_jobs_file", "scheduled task missing id")
            task["id"] = task_id
            task["task_name"] = _normalize_task_name(task.get("task_name"), task_id=task_id)
            task["prompt"] = _normalize_prompt(task.get("prompt"))
            schedule_type = str(task.get("schedule_type") or "").strip()
            if schedule_type not in {"one_time", "cron"}:
                raise ScheduledTaskError("invalid_jobs_file", f"invalid schedule_type for task {task_id}")
            task["schedule_type"] = schedule_type
            tz_name, _tz = _load_timezone(task.get("timezone"))
            task["timezone"] = tz_name
            if schedule_type == "cron":
                cron_text = str(task.get("cron") or "").strip()
                if not cron_text:
                    raise ScheduledTaskError("invalid_jobs_file", f"cron task {task_id} missing cron expression")
                task["cron"] = cron_text
                task["run_at"] = None
            else:
                task["cron"] = None
                run_at_text = task.get("run_at")
                task["run_at"] = str(run_at_text).strip() if run_at_text is not None else None
            next_value = task.get("next_run_at_utc")
            task["next_run_at_utc"] = str(next_value).strip() if next_value is not None else None
            task["enabled"] = _coerce_bool(task.get("enabled"), True)
            task["created_at_utc"] = str(task.get("created_at_utc") or "").strip() or _to_iso_utc(_utcnow())
            task["updated_at_utc"] = str(task.get("updated_at_utc") or "").strip() or task["created_at_utc"]
            task["last_run_at_utc"] = (
                str(task.get("last_run_at_utc") or "").strip() if task.get("last_run_at_utc") else None
            )
            task["run_count"] = max(0, _coerce_int(task.get("run_count"), 0))
            history = task.get("history", [])
            if not isinstance(history, list):
                history = []
            task["history"] = [
                {
                    "started_at_utc": str(item.get("started_at_utc") or "") if isinstance(item, dict) else "",
                    "finished_at_utc": str(item.get("finished_at_utc") or "") if isinstance(item, dict) else "",
                    "status": str(item.get("status") or "") if isinstance(item, dict) else "",
                    "summary": str(item.get("summary") or "") if isinstance(item, dict) else "",
                }
                for item in history
            ][-self.max_history :]
            tasks.append(task)

        return {"schema_version": SCHEMA_VERSION, "tasks": tasks}

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        parent = os.path.dirname(self.jobs_path) or "."
        os.makedirs(parent, exist_ok=True)

        serializable = {
            "schema_version": SCHEMA_VERSION,
            "tasks": payload.get("tasks", []),
        }
        fd, temp_path = tempfile.mkstemp(prefix=".scheduled_jobs_", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(serializable, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_path, self.jobs_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            raise

    def _public_task(self, task: Dict[str, Any], *, include_history: bool) -> Dict[str, Any]:
        result = {
            "id": str(task.get("id") or ""),
            "task_name": str(task.get("task_name") or ""),
            "prompt": str(task.get("prompt") or ""),
            "schedule_type": str(task.get("schedule_type") or ""),
            "run_at": task.get("run_at"),
            "cron": task.get("cron"),
            "timezone": str(task.get("timezone") or ""),
            "enabled": _coerce_bool(task.get("enabled"), True),
            "next_run_at_utc": task.get("next_run_at_utc"),
            "created_at_utc": str(task.get("created_at_utc") or ""),
            "updated_at_utc": str(task.get("updated_at_utc") or ""),
            "last_run_at_utc": task.get("last_run_at_utc"),
            "run_count": _coerce_int(task.get("run_count"), 0),
        }
        if include_history:
            result["history"] = copy.deepcopy(list(task.get("history") or []))
        return result


__all__ = [
    "ScheduledTaskError",
    "ScheduledTaskStore",
    "TaskExecutor",
    "DEFAULT_MAX_HISTORY",
    "DEFAULT_RUNNER_LOCK_PATH",
    "DEFAULT_SCHEDULED_JOBS_PATH",
    "SCHEMA_VERSION",
]
