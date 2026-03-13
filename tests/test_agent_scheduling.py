from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from agent import Agent


async def _unused_stream_fn(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]):
    del model, context, options
    raise RuntimeError("unused")


def _make_agent(tmp_path: Path) -> Agent:
    return Agent(
        {
            "stream_fn": _unused_stream_fn,
            "project_root": str(tmp_path),
        }
    )


def test_agent_schedule_api_create_list_remove_run_now(tmp_path: Path):
    agent = _make_agent(tmp_path)

    created = agent.schedule_task(
        "Prepare the weekly digest",
        run_at="2030-01-01T09:30:00",
        timezone="UTC",
        task_name="weekly_digest",
    )
    assert created["task_name"] == "weekly_digest"
    assert created["schedule_type"] == "one_time"

    listed = agent.list_scheduled_tasks()
    assert len(listed) == 1
    assert listed[0]["id"] == created["id"]

    assert agent.run_scheduled_task_now(created["id"]) is True
    assert agent.run_scheduled_task_now("missing") is False

    assert agent.remove_scheduled_task(created["id"]) is True
    assert agent.remove_scheduled_task(created["id"]) is False


def test_agent_run_due_tasks_executes_due_items(tmp_path: Path):
    agent = _make_agent(tmp_path)
    created = agent.schedule_task(
        "Run recurring check",
        cron="* * * * *",
        timezone="UTC",
        task_name="recurring_check",
    )
    assert agent.run_scheduled_task_now(created["id"]) is True

    async def _executor(task: Dict[str, Any]) -> Dict[str, Any]:
        assert task["id"] == created["id"]
        return {"status": "success", "summary": "done"}

    report = asyncio.run(agent.run_due_tasks(_executor, max_parallel=3))
    assert report["due_count"] == 1
    assert report["success_count"] == 1
    assert report["error_count"] == 0

    listed = agent.list_scheduled_tasks(include_history=True)
    assert len(listed) == 1
    assert listed[0]["run_count"] == 1
    assert len(listed[0]["history"]) == 1
