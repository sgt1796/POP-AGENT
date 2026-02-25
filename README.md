# POP-agent
Agent built w/ pop-python package

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
