from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, Optional

from agent import Agent
from agent.agent_types import AgentTool, AgentToolResult, TextContent


_TRUE_WORDS = {"1", "true", "yes", "y", "on"}
_FALSE_WORDS = {"0", "false", "no", "n", "off"}
RunDueTasksNowFn = Callable[[], Any]


def _to_text(value: Any) -> str:
    return str(value or "").strip()


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = _to_text(value).lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return default


class TaskSchedulerTool(AgentTool):
    name = "task_scheduler"
    description = (
        "Manage scheduled prompt tasks. Actions: create, list, remove, run_now. "
        "run_now only marks a task due; execution happens in the external scheduled runner."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "remove", "run_now"],
                "description": "Scheduler action to execute.",
            },
            "task_id": {
                "type": "string",
                "description": "Task id used by remove/run_now.",
            },
            "task_name": {
                "type": "string",
                "description": "Optional human-friendly task name for create.",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt executed by the scheduled runner for action=create.",
            },
            "run_at": {
                "type": "string",
                "description": "ISO datetime for one-time schedule (action=create).",
            },
            "cron": {
                "type": "string",
                "description": "Cron expression for recurring schedule (action=create).",
            },
            "timezone": {
                "type": "string",
                "description": "IANA timezone, e.g. America/New_York (action=create).",
            },
            "include_history": {
                "type": "boolean",
                "description": "Include task execution history in list output.",
            },
        },
        "required": ["action"],
    }
    label = "Task Scheduler"

    def __init__(self, agent: Agent, run_due_tasks_now_fn: Optional[RunDueTasksNowFn] = None) -> None:
        self.agent = agent
        self._run_due_tasks_now_fn = run_due_tasks_now_fn

    async def _run_due_tasks_now(self) -> Dict[str, Any]:
        if self._run_due_tasks_now_fn is None:
            return {}
        maybe_result = self._run_due_tasks_now_fn()
        if inspect.isawaitable(maybe_result):
            resolved = await maybe_result
        else:
            resolved = maybe_result
        if isinstance(resolved, dict):
            return dict(resolved)
        if resolved is None:
            return {}
        return {"result": resolved}

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={"ok": True, **details},
        )

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={"ok": False, **details},
        )

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update

        action = _to_text(params.get("action")).lower()
        if not action:
            return self._error(
                "task_scheduler error: missing action",
                {"error": "missing_action"},
            )

        try:
            if action == "create":
                task = self.agent.schedule_task(
                    _to_text(params.get("prompt")),
                    run_at=_to_text(params.get("run_at")) or None,
                    cron=_to_text(params.get("cron")) or None,
                    timezone=_to_text(params.get("timezone")) or None,
                    task_name=_to_text(params.get("task_name")) or None,
                )
                text = (
                    "task_scheduler create: ok\n"
                    f"id={task.get('id')}\n"
                    f"task_name={task.get('task_name')}\n"
                    f"schedule_type={task.get('schedule_type')}\n"
                    f"next_run_at_utc={task.get('next_run_at_utc')}"
                )
                return self._ok(text, {"action": action, "task": task})

            if action == "list":
                include_history = _to_bool(params.get("include_history"), default=False)
                tasks = self.agent.list_scheduled_tasks(include_history=include_history)
                payload = {
                    "count": len(tasks),
                    "tasks": tasks,
                }
                return self._ok(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    {"action": action, **payload},
                )

            if action == "remove":
                task_id = _to_text(params.get("task_id"))
                if not task_id:
                    return self._error(
                        "task_scheduler remove error: missing task_id",
                        {"action": action, "error": "missing_task_id"},
                    )
                removed = bool(self.agent.remove_scheduled_task(task_id))
                status_text = "removed" if removed else "not_found"
                return self._ok(
                    f"task_scheduler remove: {status_text} ({task_id})",
                    {"action": action, "task_id": task_id, "removed": removed},
                )

            if action == "run_now":
                task_id = _to_text(params.get("task_id"))
                if not task_id:
                    return self._error(
                        "task_scheduler run_now error: missing task_id",
                        {"action": action, "error": "missing_task_id"},
                    )
                marked = bool(self.agent.run_scheduled_task_now(task_id))
                run_report: Optional[Dict[str, Any]] = None
                immediate_run_error = ""
                if marked and self._run_due_tasks_now_fn is not None:
                    try:
                        run_report = await self._run_due_tasks_now()
                    except Exception as exc:
                        immediate_run_error = str(exc)

                status_text = "marked_due" if marked else "not_found"
                if marked and isinstance(run_report, dict):
                    due_count = int(run_report.get("due_count", 0) or 0)
                    success_count = int(run_report.get("success_count", 0) or 0)
                    error_count = int(run_report.get("error_count", 0) or 0)
                    if due_count > 0:
                        status_text = f"executed due={due_count} success={success_count} error={error_count}"
                if marked and immediate_run_error:
                    status_text = f"marked_due immediate_run_error={immediate_run_error}"
                return self._ok(
                    f"task_scheduler run_now: {status_text} ({task_id})",
                    {
                        "action": action,
                        "task_id": task_id,
                        "marked_due": marked,
                        "run_report": run_report,
                        "immediate_run_error": immediate_run_error or None,
                    },
                )

            return self._error(
                f"task_scheduler error: unsupported action '{action}'",
                {"action": action, "error": "unsupported_action"},
            )
        except Exception as exc:
            return self._error(
                f"task_scheduler {action} error: {exc}",
                {"action": action, "error": str(exc)},
            )


__all__ = ["TaskSchedulerTool"]
