# Session Handoff (2026-02-22)

This section is a context dump for continuing work in a new session.

## 0. Change note (2026-02-22, execution-first upgrade)

This runtime was upgraded to improve task completion reliability while keeping existing safety defaults.

Highlights:

* Added `prompting.py` to build a stronger execution-first system prompt with profile-aware behavior (`balanced`, `aggressive`, `conservative`).
* Added non-interactive `ToolsmakerAutoContinueSubscriber` for non-manual mode so create results can auto-advance to approve + activate.
* Updated runtime tool assembly so demo tools (`slow`, `fast`) are disabled by default and enabled only with `POP_AGENT_INCLUDE_DEMO_TOOLS=true`.
* Added first-class static workflow tools:
  - `gmail_fetch` (search Gmail + download attachments)
  - `pdf_merge` (merge PDFs for print workflows)
* Moved `MemorySearchTool` and `ToolsmakerTool` implementations to `agent/tools/` and kept `agent_build/agent1/tools.py` as a compatibility shim.
* Added new env controls:
  - `POP_AGENT_EXECUTION_PROFILE`
  - `POP_AGENT_TOOLSMAKER_AUTO_CONTINUE`
  - `POP_AGENT_INCLUDE_DEMO_TOOLS`
* Added focused tests:
  - `tests/test_agent1_prompting.py`
  - `tests/test_agent1_approvals.py`
  - `tests/test_agent1_runtime.py`


## 0.1 Change note (2026-02-24, context management upgrade)

Highlights:

* Added session-aware short-term memory via `ConversationMemory` so each chat session has isolated in-session retrieval.
* Extended disk memory records with `session_id` metadata so long-term retrieval is scoped to the active session for cross-session persistence.
* Added runtime chat session switching (`/session <name>`) with configurable initial session from `POP_AGENT_SESSION_ID`.
* Added `ContextCompressor` with configurable thresholds (`POP_AGENT_CONTEXT_TRIGGER_CHARS`, `POP_AGENT_CONTEXT_TARGET_CHARS`) to summarize old history before context overflows.
* Compression summaries are persisted in long-term memory as `compression_summary` records so critical prior context survives pruning.

## 0.2 Change note (2026-02-25, auto-session upgrade)

Highlights:

* Each run now starts in a unique auto-generated session id (no more `default`).
* After the first completed turn, the runtime asynchronously generates a short session title via `POP.PromptFunction` and renames session memory.
* Auto-title is canceled on manual `/session` changes and on shutdown.

## 1. What was requested

User request:

* Copy `agent0` into a new project folder `agent_build/agent1/agent1.py`.
* Decompose `agent0` into modular scripts.
* Keep all features.

## 2. What was implemented

A new modularized agent was created under `agent_build/agent1/` with `agent1.py` as entrypoint.

Created files:

* `agent_build/agent1/__init__.py`
* `agent_build/agent1/agent1.py`
* `agent_build/agent1/runtime.py`
* `agent_build/agent1/constants.py`
* `agent_build/agent1/env_utils.py`
* `agent_build/agent1/message_utils.py`
* `agent_build/agent1/event_logger.py`
* `agent_build/agent1/memory.py`
* `agent_build/agent1/tools.py`
* `agent_build/agent1/approvals.py`

## 3. Module map

* `agent1.py`
  Script launcher. Supports direct script execution and package import execution.
* `runtime.py`
  Main async chat loop; builds agent, configures model/tools, handles subscriptions, memory retrieval, prompt execution, and shutdown.
* `constants.py`
  Log levels, prompt markers, allowed capabilities, bash command allowlists.
* `env_utils.py`
  All env parsing helpers and CSV formatting helper.
* `message_utils.py`
  Message text extraction and formatting helpers used across runtime/logger/memory.
* `event_logger.py`
  Event logger factory with `quiet/simple/full/debug` behaviors.
* `memory.py`
  In-memory and disk-backed memory stores, retrieval logic, ingestion worker, and memory subscriber.
* `prompting.py`
  System-prompt builder with execution profiles and runtime-policy hints.
* `tools.py`
  Compatibility re-exports for `MemorySearchTool` and `ToolsmakerTool` (implementations now in `agent/tools/agent1_tools.py`).
* `approvals.py`
  Interactive approval prompts plus non-interactive auto-continue for `toolsmaker`, and approval prompts for `bash_exec`.

## 4. Feature parity checklist (agent0 -> agent1)

Preserved:

* Same model selection (`gemini-3-flash-preview`) and timeout.
* Upgraded memory architecture:
  `ConversationMemory` + session-scoped `DiskMemory` + retrieval injection into augmented prompt.
* Expanded memory tool behavior:
  `memory_search` with `query`, `top_k`, `scope`, and optional `session_id`.
* Same `toolsmaker` behavior:
  lifecycle actions, capability validation, intent guardrails, create->approve->activate workflow support.
* Same `bash_exec` configuration path:
  allowlists, roots, timeout/output caps, approval prompt behavior.
* Same event logging behavior.
* Same interactive terminal loop and shutdown behavior.

Enhancements:

* Stronger execution-first system prompt generation in `prompting.py`.
* Optional non-interactive toolsmaker auto-continue when manual prompts are disabled.
* Demo tools (`slow`, `fast`) are now opt-in via `POP_AGENT_INCLUDE_DEMO_TOOLS`.
* First-class Gmail + PDF workflow support via `gmail_fetch` and `pdf_merge`.

## 5. Runtime/environment controls (important)

Toolsmaker:

* `POP_AGENT_TOOLSMAKER_ALLOWED_CAPS`
  Default: `fs_read,fs_write,http`
* `POP_AGENT_TOOLSMAKER_PROMPT_APPROVAL`
  Default: `true`
* `POP_AGENT_TOOLSMAKER_AUTO_ACTIVATE`
  Default: `true`
* `POP_AGENT_TOOLSMAKER_AUTO_CONTINUE`
  Default: `true` (effective when manual approval prompts are disabled)

Bash exec:

* `POP_AGENT_BASH_ALLOWED_ROOTS`
  Default: current workspace root
* `POP_AGENT_BASH_WRITABLE_ROOTS`
  Default: current workspace root
* `POP_AGENT_BASH_TIMEOUT_S`
  Default: `15.0`
* `POP_AGENT_BASH_MAX_OUTPUT_CHARS`
  Default: `20000`
* `POP_AGENT_BASH_PROMPT_APPROVAL`
  Default: `true`

General:

* `POP_AGENT_LOG_LEVEL`
  Default: `quiet` (`simple`, `full`, `debug` also supported; legacy `messages`/`stream` aliases are accepted)
* `POP_AGENT_MEMORY_TOP_K`
  Default: `3`
* `POP_AGENT_SESSION_ID`
  Note: no longer controls the initial session id; use `/session <name>` to switch manually.
* `POP_AGENT_SESSION_ID`
  Default: `default`
* `POP_AGENT_CONTEXT_TRIGGER_CHARS`
  Default: `20000`
* `POP_AGENT_CONTEXT_TARGET_CHARS`
  Default: `12000`
* `POP_AGENT_EXECUTION_PROFILE`
  Default: `balanced` (`balanced`, `aggressive`, `conservative`)
* `POP_AGENT_INCLUDE_DEMO_TOOLS`
  Default: `false` (enables `slow` and `fast` when set to true)

## 5.5. Token Usage Tracking

Agent1 now surfaces usage from core `Agent` tracking.

Highlights:

* Usage is tracked per assistant turn in memory (session scoped; no disk persistence).
* Runtime CLI prints one compact `[usage]` line after each completed user turn when calls increase.
* TUI status line shows cumulative token usage and source mix.
* TUI activity log appends one per-turn usage line after each run.
* UI stays token-focused (`in`, `out`, `total`, source mix, anomaly count). Cost fields are not shown.

## 6. Commands for next session

Run modular agent:

```bash
python agent_build/agent1/agent1.py
```

Run Textual TUI frontend:

```bash
python test_ui.py
python -m agent_build.agent1.tui
```

TUI notes:

* `textual` must be installed in the active Python environment.
* In TUI mode, `bash_exec` and `toolsmaker` manual approvals are presented as in-app modals.
* Closing an approval modal (including `Esc`) defaults to reject/deny.
* Press `Ctrl+S` to open runtime settings (model, timeout, execution profile, memory top-k, activity level).

Basic integrity checks:

```bash
python -m compileall agent_build/agent1
python -c "import agent_build.agent1.agent1 as a; print('agent1 import ok')"
```

## 7. Validation that was run

Executed and passed:

* `python -m compileall agent_build/agent1`
* `python -c "import agent_build.agent1.agent1 as a; print('agent1 import ok')"`

## 8. Repository state note

The repository had many pre-existing modified files in the working tree.
Only new files under `agent_build/agent1/` and this README handoff section were added for this task.
