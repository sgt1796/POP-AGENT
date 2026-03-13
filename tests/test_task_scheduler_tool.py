from __future__ import annotations

import asyncio
from typing import Any, Dict

from agent.tools.task_scheduler_tool import TaskSchedulerTool


class _FakeAgent:
    def __init__(self) -> None:
        self.created: list[Dict[str, Any]] = []
        self.removed: list[str] = []
        self.marked: list[str] = []

    def schedule_task(self, prompt: str, *, run_at=None, cron=None, timezone=None, task_name=None):
        record = {
            "id": "task-1",
            "task_name": task_name or "task-1",
            "prompt": prompt,
            "schedule_type": "one_time" if run_at else "cron",
            "run_at": run_at,
            "cron": cron,
            "timezone": timezone or "UTC",
            "enabled": True,
            "next_run_at_utc": "2030-01-01T00:00:00Z",
        }
        self.created.append(record)
        return record

    def list_scheduled_tasks(self, *, include_history: bool = False):
        task = {
            "id": "task-1",
            "task_name": "task-1",
            "prompt": "do work",
            "schedule_type": "cron",
            "cron": "* * * * *",
            "timezone": "UTC",
            "enabled": True,
            "next_run_at_utc": "2030-01-01T00:00:00Z",
        }
        if include_history:
            task["history"] = [{"status": "success", "summary": "ok"}]
        return [task]

    def remove_scheduled_task(self, task_id: str) -> bool:
        self.removed.append(task_id)
        return task_id == "task-1"

    def run_scheduled_task_now(self, task_id: str) -> bool:
        self.marked.append(task_id)
        return task_id == "task-1"


def _run(tool: TaskSchedulerTool, params: Dict[str, Any]):
    return asyncio.run(tool.execute("tc1", params))


def test_task_scheduler_create_action():
    tool = TaskSchedulerTool(agent=_FakeAgent())
    result = _run(
        tool,
        {
            "action": "create",
            "prompt": "do something later",
            "run_at": "2030-01-01T10:00:00",
            "timezone": "UTC",
            "task_name": "later_job",
        },
    )

    assert result.details["ok"] is True
    assert result.details["action"] == "create"
    assert result.details["task"]["task_name"] == "later_job"


def test_task_scheduler_list_action_with_history():
    tool = TaskSchedulerTool(agent=_FakeAgent())
    result = _run(tool, {"action": "list", "include_history": True})

    assert result.details["ok"] is True
    assert result.details["action"] == "list"
    assert result.details["count"] == 1
    assert "history" in result.details["tasks"][0]


def test_task_scheduler_remove_and_run_now_actions():
    fake_agent = _FakeAgent()
    tool = TaskSchedulerTool(agent=fake_agent)

    remove_ok = _run(tool, {"action": "remove", "task_id": "task-1"})
    remove_missing = _run(tool, {"action": "remove", "task_id": "missing"})
    run_now_ok = _run(tool, {"action": "run_now", "task_id": "task-1"})

    assert remove_ok.details["removed"] is True
    assert remove_missing.details["removed"] is False
    assert run_now_ok.details["marked_due"] is True
    assert fake_agent.removed == ["task-1", "missing"]
    assert fake_agent.marked == ["task-1"]


def test_task_scheduler_run_now_can_execute_due_tasks_immediately():
    fake_agent = _FakeAgent()
    calls = []

    async def _run_due_now():
        calls.append("run_due_now")
        return {
            "due_count": 1,
            "success_count": 1,
            "error_count": 0,
            "tasks": [{"id": "task-1", "status": "success", "summary": "done"}],
        }

    tool = TaskSchedulerTool(agent=fake_agent, run_due_tasks_now_fn=_run_due_now)
    result = _run(tool, {"action": "run_now", "task_id": "task-1"})

    assert result.details["ok"] is True
    assert result.details["marked_due"] is True
    assert result.details["run_report"]["due_count"] == 1
    assert "executed due=1 success=1 error=0" in result.content[0].text
    assert fake_agent.marked == ["task-1"]
    assert calls == ["run_due_now"]


def test_task_scheduler_rejects_unsupported_action():
    tool = TaskSchedulerTool(agent=_FakeAgent())
    result = _run(tool, {"action": "unknown"})

    assert result.details["ok"] is False
    assert result.details["error"] == "unsupported_action"
