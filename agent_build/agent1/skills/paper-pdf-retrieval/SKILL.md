---
name: paper-pdf-retrieval
description: Workflow for locating DOI-linked scholarly documents and recovering a readable PDF or preview.
kind: workflow
priority: 95
tools:
  - openalex_works
  - download_url_to_file
  - file_read
triggers:
  - paper
  - doi
  - book
  - chapter
  - monograph
  - openalex
  - preprint
  - journal
  - abstract
scope: turn
---
For DOI-linked scholarly documents, including books and chapters, use `openalex_works` to locate the record and prefer the exact OpenAlex OA PDF or landing URL before broad web search.
Use `download_url_to_file` to save the PDF locally, then switch to `file_read` on the local artifact instead of continuing web rediscovery.
If the download returns HTML instead of a PDF, treat it as a landing page and use the returned `final_url`, landing page title, `content_preview`, and `pdf_link_candidates` to recover the real document or a readable preview before searching elsewhere.
