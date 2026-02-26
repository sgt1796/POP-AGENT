# POP-agent
Agent built w/ pop-python package

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
