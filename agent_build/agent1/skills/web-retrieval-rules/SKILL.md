---
name: web-retrieval-rules
description: Rules for web search, snapshots, and source verification.
kind: tool_rules
priority: 70
tools:
  - jina_web_snapshot
  - perplexity_search
  - perplexity_web_snapshot
  - openalex_works
  - download_url_to_file
triggers:
  - search
  - source
scope: system
---
For web retrieval, prefer narrow queries, exact identifiers, domain filters, and direct source URLs before broad retries.
Do not answer from search-result snippets alone when you can still open the cited page or local document and verify the exact field there.
If search results drift to irrelevant sites, tighten the query or pivot to a known URL, identifier, or local artifact instead of repeating the same broad search.
If `jina_web_snapshot` fails with a 4xx or proxy access error on a known URL, try the original URL directly or save it locally as `.html` before broadening search.
If a retrieval tool returns concrete recovery hints such as `final_url`, `pdf_link_candidates`, `content_preview`, or `saved_landing_page_path`, follow those exact leads before broad search.
When saving fetched artifacts locally, prefer descriptive paths under `downloads/` so the follow-up `file_read` path is predictable.
For numeric, count, or comparison questions, verify the exact value list or field on the strongest candidate source before answering from a broad summary.
For chained or filtered questions, verify the eligible entity set and each stated constraint before selecting the final answer.
