---
name: local-document-evidence
description: Workflow for extracting answers from local files, attachments, and downloaded documents.
kind: workflow
priority: 100
tools:
  - file_read
  - bash_exec
  - download_url_to_file
triggers:
  - attachment
  - attached
  - local file
  - document
  - pdf
  - csv
  - xlsx
scope: turn
---
Treat staged attachments, downloaded files, and workspace-local documents as primary evidence before remote fetches.
If the target artifact already exists locally, read it first and answer from the explicit nearby passage or field instead of rediscovering the same source on the web.
When a local document contains the target phrase, inspect the surrounding passage and answer from the explicit attribution there rather than from unrelated names elsewhere in the file.
