---
name: file-io-rules
description: Rules for reading local artifacts and writing workspace text files.
kind: tool_rules
priority: 80
tools:
  - file_read
  - file_write
  - download_url_to_file
triggers:
  - file
  - attachment
scope: system
---
Prefer `file_read` for attachments, downloaded documents, and structured files before falling back to shell file reads.
For large local text or PDF artifacts, use bounded reads and inspect the nearby passage instead of broad shell search against binary files.
Use `file_write` for text creation or replacement, and do not fetch a remote copy of a file that already exists locally unless the local artifact clearly failed or is incomplete.
