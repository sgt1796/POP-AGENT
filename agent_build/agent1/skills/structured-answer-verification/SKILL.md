---
name: structured-answer-verification
description: Workflow for chained lookups, multi-constraint filtering, and comparison answers.
kind: workflow
priority: 92
tools:
  - calculator
  - jina_web_snapshot
  - perplexity_search
  - file_read
triggers:
  - compare
  - comparison
  - difference
  - how many
  - count
  - maximum
  - minimum
  - heaviest
  - lightest
  - furthest
  - farthest
  - nearest
  - closest
  - shared
  - possible
scope: turn
---
Decompose the task into the eligible set, any intermediate entity links, and the final requested field before searching.
For chained lookups, verify each hop explicitly from source page to selected entity to requested field instead of jumping from an early clue to the final answer.
For max/min, comparison, count, or difference tasks, write down a compact table of every eligible candidate with verified value, unit, and source before using calculator or answering.
Discard any candidate that fails even one stated filter, boundary, membership rule, or counting convention.
Do not use memory hits or weak search snippets as final evidence for current numeric facts when an exact page, local artifact, or structured record can still be checked.
