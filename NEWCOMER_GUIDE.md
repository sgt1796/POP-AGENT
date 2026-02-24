# Newcomer Guide

This repository is a Python "agent runtime" project with two related tracks:

1. **Core reusable library** under `agent/` (the actual agent framework).
2. **Assembled app/runtime** under `agent_build/agent1/` (CLI/TUI experience with curated tools and policies).

## Quick mental model

- Start with `agent/` if you want to understand the framework.
- Start with `agent_build/agent1/` if you want to run and modify the end-user experience.
- Read `tests/` to see expected behaviors and guardrails.

## Directory map

- `agent/`
  - Core `Agent` class and evented loop.
  - Tool interfaces + built-in tools.
  - Dynamic tool generation subsystem (`toolsmaker/`).
- `agent_build/agent1/`
  - Modularized runtime for the "agent1" app.
  - Runtime wiring, environment parsing, prompting policy, TUI, approvals, and usage reporting.
- `tests/`
  - Coverage for runtime behavior, approvals, prompting, usage tracking, tool shims, and toolsmaker flow.
- `quickstart.py`
  - Gmail OAuth helper used to generate credentials for Gmail integration demos.

## Important concepts to learn first

1. **Agent state + loop**
   - `agent/agent.py` is the high-level API users interact with.
   - `agent/agent_loop.py` executes turns, streams model output, handles tool calls, and emits events.

2. **Event-driven architecture**
   - Consumers subscribe to events (logger, memory ingestion, approvals, TUI updates).
   - This keeps core loop logic separate from UI/side effects.

3. **Tool model**
   - Tools conform to the `AgentTool` protocol in `agent/agent_types.py`.
   - Runtime tools are assembled in `agent_build/agent1/runtime.py`.

4. **Dynamic tools (toolsmaker)**
   - `agent/toolsmaker/` supports a human-gated build/validate/approve/activate lifecycle for generated tools.
   - This is where policy + safety controls are centralized.

## How the agent1 runtime is composed

`agent_build/agent1/runtime.py` is the composition root:

- Creates `Agent` and model/timeout.
- Builds memory components and subscribers.
- Builds runtime tools (web snapshot, memory search, toolsmaker, bash, gmail fetch, pdf merge; optional demo tools).
- Applies system prompt + execution profile.
- Registers subscribers for logging, memory ingestion, and approval handling.
- Provides per-turn execution helpers and interactive loop behavior.

`agent_build/agent1/README.md` provides module-level context and env flags.

## What to run first

- CLI entrypoint:
  - `python agent_build/agent1/agent1.py`
- TUI entrypoint:
  - `python -m agent_build.agent1.tui`

If Gmail tools are relevant:
- run `python quickstart.py` to bootstrap Gmail credentials.

## Pointers for what to learn next

1. **Read tests as executable documentation**
   - Start with:
     - `tests/test_agent1_runtime.py`
     - `tests/test_agent1_prompting.py`
     - `tests/test_agent1_approvals.py`
     - `tests/test_chatroom_tool_creation_flow.py`

2. **Trace one request end-to-end**
   - Start at `agent_build/agent1/runtime.py::run_user_turn`
   - Follow into `agent/agent.py` and `agent/agent_loop.py`
   - Observe emitted events and subscriber reactions.

3. **Understand policy boundaries before extending tools**
   - Read:
     - `agent/tools/bash_exec_tool.py`
     - `agent/toolsmaker/policy.py`
     - `agent/toolsmaker/validator.py`

4. **Then customize prompts + UX**
   - Prompt strategy: `agent_build/agent1/prompting.py`
   - TUI UX: `agent_build/agent1/tui.py` and `agent_build/agent1/tui_runtime.py`

## Practical newcomer checklist

- [ ] Run the runtime once in CLI mode.
- [ ] Run tests for runtime + approvals.
- [ ] Inspect one tool call in logs/events.
- [ ] Make one prompt change and observe behavior.
- [ ] Add or modify one tool, then validate policy constraints.

