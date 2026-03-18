# POP-Agent

POP-Agent is a Python agent runtime built on top of `pop-python`. It provides a practical local-first setup for interactive agent sessions, tool calling, scheduled automations, and report delivery via AgentMail.

This repository includes:
- A reusable core agent package in `agent/`.
- A runnable runtime in `agent_build/agent1/`.
- Evaluation utilities in `eval/`.

---

## Table of Contents
- [What this project is for](#what-this-project-is-for)
- [Quick start](#quick-start)
- [Core workflows](#core-workflows)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Feature highlights](#feature-highlights)
- [Where to go next](#where-to-go-next)

---

## What this project is for
Use POP-Agent when you want an agent that can:
- Run interactive local sessions (TUI or module execution).
- Call tools for filesystem, web/search, research, and integrations.
- Schedule one-time or recurring tasks with persistent state.
- Send scheduled summaries/reports through AgentMail.

If you are looking for core agent-loop implementation details, see `agent/README.md`.

---

## Quick start

### 1) Run with Docker (Python 3.13)
```bash
docker build -t pop-agent:py313 .
docker run --rm -it --privileged --env-file .env pop-agent:py313
```

### 2) Run the TUI directly
```bash
python run_tui.py
```

### 3) Run the default runtime module
```bash
python -m agent_build.agent1.agent1
```

---

## Core workflows

### Interactive usage
- Start the TUI with `python run_tui.py`.
- Chat with the runtime and invoke enabled tools naturally.

### Schedule tasks
```bash
python -m agent_build.agent1.scheduled_runner --list
python -m agent_build.agent1.scheduled_runner --daemon-start
python -m agent_build.agent1.scheduled_runner --daemon-status
python -m agent_build.agent1.scheduled_runner --daemon-stop
```

Notes:
- One-shot mode runs currently due jobs and exits.
- Daemon mode keeps polling so schedules continue after the TUI closes.
- Scheduler state is stored in `agent/mem/scheduled_jobs.json`.

### Send email reports with AgentMail
1. Configure environment variables:
```bash
AGENTMAIL_API_KEY=your_agentmail_api_key
POP_AGENT_AGENTMAIL_INBOX_ID=your_agentmail_inbox_id
POP_AGENT_AGENTMAIL_TO_EMAIL=you@example.com
```
2. Restart your runtime.
3. Use `agentmail_send` from prompts/scheduled tasks.

`agentmail_send` always sends to `POP_AGENT_AGENTMAIL_TO_EMAIL` and supports:
- `subject`
- required `text_body`
- optional `html_body`
- optional workspace `attachment_paths`

---

## Project structure

```text
.
├── agent/                  # Core agent package (loop, state, tools, scheduler, memory)
├── agent_build/agent1/     # Ready-to-run runtime + scheduled runner entrypoints
├── eval/                   # Evaluation framework, configs, benchmarks, executors
├── tests/                  # Test suite
├── run_tui.py              # TUI entrypoint
└── README.md
```

---

## Configuration

Common runtime environment variables:

- `AGENTMAIL_API_KEY`
- `POP_AGENT_AGENTMAIL_INBOX_ID`
- `POP_AGENT_AGENTMAIL_TO_EMAIL`
- `OPENALEX_EMAIL` (optional)
- `OPENALEX_API_KEY` (optional)
- `POP_AGENT_SCHEDULER_PERSISTENT` (`1`/`0`, default daemon-friendly behavior)

Scheduler artifacts:
- Jobs: `agent/mem/scheduled_jobs.json`
- Lock: `agent/mem/scheduled_jobs.lock`
- Daemon PID: `agent/mem/scheduled_runner.pid`
- Daemon log: `agent/mem/scheduled_runner.log`

---

## Feature highlights

- Persistent scheduler with one-time + cron jobs, timezone support, and run history.
- `task_scheduler` tool (`create`, `list`, `remove`, `run_now`).
- AgentMail integration via `agentmail_send`.
- OpenAlex paper search/fetch support.
- Search integrations including Jina and Perplexity tools.
- Usage tracking APIs for token/accounting visibility.

---

## Where to go next

- Core package details: `agent/README.md`
- Runtime code: `agent_build/agent1/`
- Eval workflows: `eval/README.md`
