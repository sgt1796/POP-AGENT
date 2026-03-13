from __future__ import annotations

import asyncio

import pytest

from agent.scheduler import ScheduledTaskError, ScheduledTaskStore


def test_schedule_list_remove_and_run_now(tmp_path):
    store = ScheduledTaskStore(project_root=str(tmp_path))

    created = store.schedule_task(
        "Send summary",
        run_at="2030-01-01T10:00:00",
        timezone_name="UTC",
        task_name="summary_job",
    )
    assert created["schedule_type"] == "one_time"
    assert created["task_name"] == "summary_job"

    listed = store.list_tasks()
    assert len(listed) == 1
    assert listed[0]["id"] == created["id"]

    assert store.mark_task_due_now(created["id"]) is True
    assert store.mark_task_due_now("missing") is False

    assert store.remove_task(created["id"]) is True
    assert store.remove_task(created["id"]) is False


def test_schedule_requires_exactly_one_of_run_at_or_cron(tmp_path):
    store = ScheduledTaskStore(project_root=str(tmp_path))

    with pytest.raises(ScheduledTaskError):
        store.schedule_task("x", timezone_name="UTC")

    with pytest.raises(ScheduledTaskError):
        store.schedule_task("x", run_at="2030-01-01T00:00:00Z", cron="* * * * *", timezone_name="UTC")


def test_run_due_tasks_runs_due_cron_once_and_advances(tmp_path):
    store = ScheduledTaskStore(project_root=str(tmp_path))
    created = store.schedule_task("Do cron work", cron="* * * * *", timezone_name="UTC", task_name="cron_job")
    assert store.mark_task_due_now(created["id"]) is True

    async def _executor(task):
        assert task["id"] == created["id"]
        return {"status": "success", "summary": "ok"}

    report = asyncio.run(store.run_due_tasks(_executor, max_parallel=3))
    assert report["due_count"] == 1
    assert report["success_count"] == 1
    assert report["error_count"] == 0

    tasks = store.list_tasks(include_history=True)
    assert len(tasks) == 1
    task = tasks[0]
    assert task["id"] == created["id"]
    assert task["enabled"] is True
    assert task["run_count"] == 1
    assert isinstance(task.get("next_run_at_utc"), str) and task["next_run_at_utc"]
    assert len(task.get("history", [])) == 1


def test_run_due_tasks_history_is_trimmed_to_20(tmp_path):
    store = ScheduledTaskStore(project_root=str(tmp_path))
    created = store.schedule_task("Keep looping", cron="* * * * *", timezone_name="UTC", task_name="trim_job")

    async def _executor(_task):
        return {"status": "success", "summary": "ok"}

    for _ in range(25):
        assert store.mark_task_due_now(created["id"]) is True
        report = asyncio.run(store.run_due_tasks(_executor, max_parallel=3))
        assert report["due_count"] == 1

    task = store.list_tasks(include_history=True)[0]
    assert task["run_count"] == 25
    assert len(task.get("history", [])) == 20


def test_run_due_tasks_preserves_tasks_created_during_execution(tmp_path):
    store = ScheduledTaskStore(project_root=str(tmp_path))
    created = store.schedule_task(
        "Create follow-up work",
        run_at="2030-01-01T00:00:00",
        timezone_name="UTC",
        task_name="parent_job",
    )
    assert store.mark_task_due_now(created["id"]) is True

    async def _executor(_task):
        nested = store.schedule_task(
            "Nested work",
            run_at="2030-01-01T00:05:00",
            timezone_name="UTC",
            task_name="child_job",
        )
        return {"status": "success", "summary": f"created {nested['id']}"}

    report = asyncio.run(store.run_due_tasks(_executor, max_parallel=1))
    assert report["due_count"] == 1
    assert report["success_count"] == 1

    tasks = store.list_tasks(include_history=True)
    assert len(tasks) == 2

    parent = next(task for task in tasks if task["id"] == created["id"])
    child = next(task for task in tasks if task["task_name"] == "child_job")

    assert parent["enabled"] is False
    assert parent["run_count"] == 1
    assert len(parent.get("history", [])) == 1
    assert child["enabled"] is True
    assert child["run_count"] == 0
    assert child["id"] != parent["id"]


def test_load_payload_raises_for_invalid_json(tmp_path):
    store = ScheduledTaskStore(project_root=str(tmp_path))
    path = tmp_path / "agent" / "mem" / "scheduled_jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ScheduledTaskError):
        store.list_tasks()


def test_runner_lock_blocks_concurrent_runs(tmp_path):
    store = ScheduledTaskStore(project_root=str(tmp_path))
    created = store.schedule_task("Run once", run_at="2030-01-01T00:00:00", timezone_name="UTC", task_name="lock_job")
    assert store.mark_task_due_now(created["id"]) is True

    async def _executor(_task):
        return {"status": "success", "summary": "ok"}

    with store._runner_lock():
        with pytest.raises(ScheduledTaskError) as exc:
            asyncio.run(store.run_due_tasks(_executor, max_parallel=1))
        assert exc.value.code == "runner_locked"
