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
