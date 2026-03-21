---
name: web-verification
description: Workflow for verifying facts from official or exact web sources.
kind: workflow
priority: 90
tools:
  - jina_web_snapshot
  - perplexity_search
  - perplexity_web_snapshot
  - file_read
triggers:
  - website
  - source
  - verify
  - official site
  - url
scope: turn
---
Start from the exact URL or the narrowest possible search query, then verify the requested field from the cited page.
If exact source retrieval fails with a concrete transport or server error and the snippet already states the requested field explicitly, use that snippet as fallback evidence rather than stalling on the same page.
Once you have a trustworthy local or cited source for the target fact, stop broad source rediscovery and answer from that evidence.
