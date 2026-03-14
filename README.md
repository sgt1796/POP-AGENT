# POP-agent
Agent built w/ pop-python package

## Use Examples

### Docker (Python 3.13)
```bash
docker build -t pop-agent:py313 .
docker run --rm -it --privileged --env-file .env pop-agent:py313
```

### Run TUI directly
```bash
python run_tui.py
```

### Run agent module
```bash
python -m agent_build.agent1.agent1
```

## Recent Changes (2026-03-14)
- Replaced SMTP relay delivery with a new `agentmail_send` tool:
  - Sends outbound email through the official AgentMail Python SDK.
  - Supports `subject`, required `text_body`, optional `html_body`, and optional workspace attachments.
  - Always sends to the configured owner recipient from env; recipients are not tool-call configurable.
- Added AgentMail runtime configuration via env:
  - Required: `AGENTMAIL_API_KEY`, `POP_AGENT_AGENTMAIL_INBOX_ID`, `POP_AGENT_AGENTMAIL_TO_EMAIL`
- Scheduled prompts can now call `agentmail_send` to mail work reports from a Raspberry Pi runner or any host running the agent1 runtime.

## Recent Changes (2026-03-12)
- Added a persistent scheduler subsystem in core `agent`:
  - New `Agent` APIs:
    - `schedule_task(prompt, *, run_at=None, cron=None, timezone=None, task_name=None)`
    - `list_scheduled_tasks(*, include_history=False)`
    - `remove_scheduled_task(task_id)`
    - `run_scheduled_task_now(task_id)` (marks task due now)
    - `run_due_tasks(executor, *, max_parallel=3)`
  - Persistence file: `agent/mem/scheduled_jobs.json`
  - Runner lock file: `agent/mem/scheduled_jobs.lock`
  - Supports one-time ISO schedules and recurring cron schedules with per-task timezone.
  - Misfire policy is catch-up-once for recurring tasks.
  - Keeps last 20 execution history entries per task.
- Added scheduler tool for the agent runtime:
  - New tool: `task_scheduler`
  - Actions: `create`, `list`, `remove`, `run_now`
  - `run_now` marks a task due; it does not execute inline.
- Added one-shot external scheduled runner:
  - Script/module: `agent_build/agent1/scheduled_runner.py`
  - Runs all due tasks once, bounded concurrency (`--max-parallel`, default `3`), then exits.
  - Uses agent1 runtime lifecycle per task:
    `create_runtime_session -> switch_session -> run_user_turn -> shutdown_runtime_session`
- Added dependencies for scheduler support:
  - `croniter`
  - `tzdata`

### Run scheduled tasks once
```bash
python -m agent_build.agent1.scheduled_runner
```

### Start the persistent scheduler daemon
```bash
python -m agent_build.agent1.scheduled_runner --daemon-start
```

### Show daemon status
```bash
python -m agent_build.agent1.scheduled_runner --daemon-status
```

### Stop the persistent scheduler daemon
```bash
python -m agent_build.agent1.scheduled_runner --daemon-stop
```

### List scheduled tasks
```bash
python -m agent_build.agent1.scheduled_runner --list
```

Note:
- The TUI prefers a detached scheduler daemon by default, so scheduled tasks can continue after you close the app.
- The daemon PID is stored in `agent/mem/scheduled_runner.pid` and logs are written to `agent/mem/scheduled_runner.log`.
- Set `POP_AGENT_SCHEDULER_PERSISTENT=0` to disable daemon startup and fall back to the in-process scheduler worker.

## AgentMail tool

Configure AgentMail in `.env` or in your shell environment. If you run with Docker, the existing `--env-file .env` example will load these automatically.

### 1. Add AgentMail settings

Example `.env`:

```bash
AGENTMAIL_API_KEY=your_agentmail_api_key
POP_AGENT_AGENTMAIL_INBOX_ID=your_agentmail_inbox_id
POP_AGENT_AGENTMAIL_TO_EMAIL=you@example.com
```

Required:
- `AGENTMAIL_API_KEY`
- `POP_AGENT_AGENTMAIL_INBOX_ID`
- `POP_AGENT_AGENTMAIL_TO_EMAIL`

Rules:
- `AGENTMAIL_API_KEY` is the SDK credential described in the AgentMail onboarding docs.
- `POP_AGENT_AGENTMAIL_INBOX_ID` should be an existing AgentMail inbox id. The included `test.py` helper can create one and reuse it via env.
- `POP_AGENT_AGENTMAIL_TO_EMAIL` remains the fixed owner recipient address for scheduled reports.

### 2. Restart the agent runtime

After changing `.env`, restart the process that runs the agent or scheduled runner so it picks up the new environment.

### 3. Use `agentmail_send`

Tool payload shape:

```json
{
  "subject": "Nightly work report",
  "text_body": "Completed the sync job and generated fresh metrics.",
  "html_body": "<p>Completed the sync job and generated fresh metrics.</p>",
  "attachment_paths": ["agent/mem/nightly-report.txt"]
}
```

Notes:
- `agentmail_send` always delivers to `POP_AGENT_AGENTMAIL_TO_EMAIL`.
- `attachment_paths` must resolve inside the workspace.
- Scheduled tasks can email you by including a final `agentmail_send` step in the scheduled prompt.

### 4. Scheduled report example

Example scheduled prompt:

```text
Check the project status, summarize completed work, and then call agentmail_send with:
- subject: "Raspberry Pi work report"
- text_body: a concise plain-text summary
- html_body: an optional HTML version of the same report
```

If you want to attach a generated file, save it inside the workspace first and then pass its path in `attachment_paths`.

## Scheduler Deep Dive (implementation + call flow)

This section explains exactly how scheduling is wired in this repo.

### Core components
- `agent/scheduler.py`
  - `ScheduledTaskStore`: persistence, validation, due-task selection, lock handling, and post-run task advancement.
  - `ScheduledTaskError`: typed runtime errors (`invalid_schedule`, `invalid_timezone`, `invalid_cron`, `runner_locked`, etc.).
- `agent/agent.py`
  - Thin public API wrapper over `ScheduledTaskStore`:
    - `schedule_task(...)`
    - `list_scheduled_tasks(...)`
    - `remove_scheduled_task(...)`
    - `run_scheduled_task_now(...)`
    - `run_due_tasks(...)`
- `agent/tools/task_scheduler_tool.py`
  - LLM-facing tool (`task_scheduler`) that calls the public `Agent` scheduler APIs.
- `agent_build/agent1/scheduled_runner.py`
  - External runner with both one-shot and detached daemon modes.
  - One-shot mode executes all currently due tasks, then exits.
  - Daemon mode polls for due tasks, writes a PID file, and keeps running after the TUI exits.

### Persistence model
- File path: `agent/mem/scheduled_jobs.json`
- Lock path: `agent/mem/scheduled_jobs.lock`
- Schema:
```json
{
  "schema_version": 1,
  "tasks": [
    {
      "id": "uuid-hex",
      "task_name": "human_readable_name",
      "prompt": "prompt executed at run time",
      "schedule_type": "one_time | cron",
      "run_at": "ISO datetime or null",
      "cron": "cron expr or null",
      "timezone": "IANA zone",
      "enabled": true,
      "next_run_at_utc": "next UTC timestamp",
      "created_at_utc": "UTC timestamp",
      "updated_at_utc": "UTC timestamp",
      "last_run_at_utc": "UTC timestamp or null",
      "run_count": 0,
      "history": []
    }
  ]
}
```

### Create/list/remove/run_now behavior
- `schedule_task(...)`
  - Requires exactly one of `run_at` or `cron`.
  - `run_at` parsing:
    - timezone-aware ISO is accepted directly.
    - naive ISO is interpreted in provided `timezone` (or local timezone default).
  - `cron` parsing:
    - computed with `croniter` in task timezone.
  - Stores normalized `next_run_at_utc`.
- `list_scheduled_tasks(...)`
  - Returns normalized task records sorted by next run time.
  - `include_history=True` includes latest run summaries.
- `remove_scheduled_task(task_id)`
  - Deletes by id.
- `run_scheduled_task_now(task_id)`
  - Marks `next_run_at_utc` to now and keeps task enabled.
  - Does not execute the task inline.

### Due-task execution behavior
- Entry point: `Agent.run_due_tasks(executor, max_parallel=3)`
- Execution steps:
  1. Acquire lock file (`scheduled_jobs.lock`) to prevent overlapping runners.
  2. Load tasks and select due tasks (`enabled && next_run_at_utc <= now`).
  3. Execute due tasks with bounded asyncio concurrency (`max_parallel`).
  4. Persist run outcome into `history` (keeps last 20 entries).
  5. Advance schedule:
     - `one_time`: disable task and clear next run.
     - `cron`: compute next run from completion time.
- Misfire model:
  - catch-up-once: if many cron intervals were missed while runner was offline, one due execution is run, then schedule advances to the next future tick.

### Runner call flow
- `python -m agent_build.agent1.scheduled_runner`
  1. Build a lightweight manager `Agent`.
  2. Call `Agent.run_due_tasks(...)`.
  3. For each due task, executor `_execute_task(...)` does:
     - `create_runtime_session(enable_event_logger=False)`
     - `switch_session("scheduled:<task_id>")`
     - `run_user_turn(session, task.prompt)`
     - `shutdown_runtime_session(session)`
  4. Print summary and exit.

### Runner CLI
- Run due tasks once:
```bash
python -m agent_build.agent1.scheduled_runner
```
- Limit concurrency:
```bash
python -m agent_build.agent1.scheduled_runner --max-parallel 3
```
- Start daemon mode:
```bash
python -m agent_build.agent1.scheduled_runner --daemon-start
```
- Show daemon status:
```bash
python -m agent_build.agent1.scheduled_runner --daemon-status
```
- Stop daemon mode:
```bash
python -m agent_build.agent1.scheduled_runner --daemon-stop
```
- List registered tasks:
```bash
python -m agent_build.agent1.scheduled_runner --list
```

### How the LLM calls scheduler
- Tool name: `task_scheduler`
- Actions: `create`, `list`, `remove`, `run_now`
- Example tool payloads:
```json
{"action":"create","task_name":"daily_digest","prompt":"Summarize new issues","cron":"0 9 * * *","timezone":"America/New_York"}
```
```json
{"action":"run_now","task_id":"<task_id>"}
```
```json
{"action":"list","include_history":true}
```

### Direct Python API usage
```python
from agent import Agent

agent = Agent({"stream_fn": my_stream_fn})
task = agent.schedule_task(
    "Check inbox and summarize priority emails",
    cron="*/30 * * * *",
    timezone="America/New_York",
    task_name="inbox_half_hour",
)

agent.run_scheduled_task_now(task["id"])  # mark due now
```

## Recent Changes (2026-03-05)
- Added new research-paper tool: `openalex_works`.
- Updated `agent_build.agent1` runtime defaults to include `openalex_works`.
- `openalex_works` supports:
  - `action="search"` for OpenAlex works search with query, pagination, sort, and filters.
  - `action="fetch_openalex_record"` for fetching a single work by OpenAlex ID/URL or DOI.
- Tool output now returns normalized paper metadata (title, authors, year/date, citations, DOI, OA links), with optional abstract reconstruction via `include_abstract=true`.
- OpenAlex setup:
  - No extra SDK dependency required (uses direct OpenAlex HTTP API).
  - Optional polite pool email: `OPENALEX_EMAIL=...`
  - Optional API key: `OPENALEX_API_KEY=...`

## Recent Changes (2026-02-26)
- Updated `agent_build/agent1` runtime and eval behavior:
  - `RuntimeOverrides.enable_memory` now correctly enables/disables memory wiring.
  - `RuntimeOverrides.long_memory_base_path` now controls the disk memory location.
  - Eval executor now supports `enable_event_logger` (default: `true`).
- Added tool-call stream logging visibility:
  - Logs `toolcall_start` / `toolcall_end` (and `toolcall_delta` in fuller modes) from assistant stream updates.
  - Applies to both CLI event logger and TUI activity formatting.
- Added regression tests for memory override wiring and tool-call stream logging.

## Recent Changes (2026-02-26)
- Added auto-session memory enhancements in `agent_build/agent1`:
  - Default session propagation for memory retrieval and `memory_search`.
  - Resilient auto-title rename with rollback on failure.
  - Debug session/memory logging (CLI + TUI when log level is `debug`).
  - TUI session switch modal (Ctrl+N) with runtime-level session switching.
- Added tests for default-session retrieval and `memory_search` defaults.

## Recent Changes (2026-02-25)
- Added a new search tools package at `agent/tools/search/`.
- Added search tools:
  - `jina_web_snapshot`
  - `perplexity_search`
  - `perplexity_web_snapshot` (stub)
- Renamed the legacy `websnapshot` tool name to `jina_web_snapshot`.
- Updated `agent_build.agent1` runtime defaults to include all three search tools above.
- Perplexity setup:
  - Install SDK: `pip install perplexity`
  - Set API key: `PERPLEXITY_API_KEY=...`
