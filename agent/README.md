# Pop Agent (Core Package)

`agent/` contains the core Python implementation of the POP agent loop.
It is framework-light and designed to be embedded into other applications.

If you want runnable app-level behavior (sessions, presets, entrypoints), use `agent_build/agent1/` in the repository root.

---

## Table of Contents
- [Purpose](#purpose)
- [Key capabilities](#key-capabilities)
- [Minimal usage example](#minimal-usage-example)
- [Core APIs](#core-apis)
- [Scheduler model](#scheduler-model)
- [Usage tracking](#usage-tracking)
- [Module map](#module-map)
- [Security note: `bash_exec` tool](#security-note-bash_exec-tool)

---

## Purpose
This package provides:
- Agent state management (messages, tools, model config, runtime flags).
- Event-driven execution (`agent_start`, message events, tool events, completion).
- Tool-calling orchestration.
- Steering/follow-up queues for guided multi-step interactions.
- Optional scheduler primitives for one-time and cron task execution.

It intentionally does **not** bundle a specific LLM provider; pass your own `stream_fn` (for example via `pop-python`).

---

## Key capabilities

- **Event streaming API** for UI/CLI consumers.
- **Pluggable transport** through user-supplied `stream_fn`.
- **Tool system** via `AgentTool` subclasses.
- **Scheduler integration** with persistent JSON store and lock-based execution safety.
- **Usage accounting** with normalized token metrics and summary APIs.

---

## Minimal usage example

```python
import asyncio
from pop_agent import Agent


async def dummy_stream_fn(model, context, options):
    user_messages = [m for m in context["messages"] if m["role"] == "user"]
    reply = user_messages[-1]["content"][0]["text"] if user_messages else "Hello"

    assistant_msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": reply}],
        "timestamp": options.get("timestamp", 0),
        "stopReason": "stop",
        "api": model.get("api"),
        "provider": model.get("provider"),
        "model": model.get("id"),
        "usage": {},
    }

    class SimpleStream:
        async def __aiter__(self):
            yield {"type": "done", "reason": "stop", "message": assistant_msg}

        async def result(self):
            return assistant_msg

    return SimpleStream()


async def main():
    agent = Agent({
        "initial_state": {"system_prompt": "You are helpful."},
        "stream_fn": dummy_stream_fn,
    })

    agent.subscribe(lambda event: print(event["type"]))
    await agent.prompt("Hello, world!")
    await agent.wait_for_idle()

asyncio.run(main())
```

---

## Core APIs

Typical `Agent` entrypoints:
- Prompt/turn lifecycle:
  - `prompt(...)`
  - `continue(...)`
  - `abort(...)`
  - `wait_for_idle(...)`
- Scheduler APIs:
  - `schedule_task(prompt, *, run_at=None, cron=None, timezone=None, task_name=None)`
  - `list_scheduled_tasks(*, include_history=False)`
  - `remove_scheduled_task(task_id)`
  - `run_scheduled_task_now(task_id)`
  - `run_due_tasks(executor, *, max_parallel=3)`

---

## Scheduler model

Persistence:
- `agent/mem/scheduled_jobs.json`
- Lock file: `agent/mem/scheduled_jobs.lock`

Behavior summary:
- Supports one-time (`run_at`) and recurring cron (`cron`) schedules.
- Handles timezone-aware scheduling.
- Uses bounded parallel execution in `run_due_tasks`.
- Stores compact per-task execution history.
- Recurring jobs use catch-up-once semantics after downtime.

LLM-facing scheduler tool:
- Tool name: `task_scheduler`
- Actions: `create`, `list`, `remove`, `run_now`

---

## Usage tracking

In-memory APIs:
- `get_last_usage()`
- `get_usage_history(limit=None)`
- `get_usage_summary()`
- `reset_usage_tracking()`

Notes:
- Tracking is recorded per assistant completion.
- History is bounded; totals are cumulative within process lifetime.
- `agent.reset()` does not clear usage metrics; `reset_usage_tracking()` does.

---

## Module map

- `agent/agent.py` – High-level `Agent` class and public APIs.
- `agent/agent_loop.py` – Main turn orchestration.
- `agent/event_stream.py` – Async event stream utilities.
- `agent/agent_types.py` – Shared types/dataclasses.
- `agent/scheduler.py` – Scheduler storage, validation, due-task logic.
- `agent/memory.py` – Memory storage/retrieval helpers.
- `agent/proxy.py` – Optional proxy transport helpers.
- `agent/tools/` – Tool implementations (scheduler, file, search, AgentMail, etc.).

---

## Security note: `bash_exec` tool

The secure `bash_exec` tool (wired by runtime code such as `agent_build/agent0.py`) is designed for constrained execution:
- Allowlisted commands.
- Path-scoped reads/writes.
- Risk-based approvals.
- Timeout and output limits.
- No shell operator chaining.

Use runtime environment variables to further restrict allowed/writable roots and behavior.
