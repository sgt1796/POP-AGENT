---
name: paper-pdf-retrieval
description: Workflow for locating research papers and downloading the best open-access PDF.
kind: workflow
priority: 95
tools:
  - openalex_works
  - download_url_to_file
  - file_read
triggers:
  - paper
  - doi
  - openalex
  - preprint
  - journal
  - abstract
scope: turn
---
For research papers, use `openalex_works` to locate the work record and prefer `best_oa_pdf_url` when available.
Use `download_url_to_file` to save the PDF locally, then switch to `file_read` on the local artifact instead of continuing web rediscovery.
If the download returns HTML instead of a PDF, treat it as a landing page and use the returned hints such as `final_url`, `title`, and `pdf_link_candidates` to recover the real document.
