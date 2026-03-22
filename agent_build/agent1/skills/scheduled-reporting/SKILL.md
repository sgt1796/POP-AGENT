---
name: scheduled-reporting
description: Workflow for creating scheduled tasks and emailing reports later.
kind: workflow
priority: 85
tools:
  - task_scheduler
  - agentmail_send
triggers:
  - schedule
  - later
  - tomorrow
  - every
  - cron
  - email me
scope: turn
---
When the user wants work to happen later, create or manage the scheduled task instead of trying to simulate delayed execution in the current turn.
If the requested outcome includes notifying the configured owner, use `agentmail_send` from the scheduled workflow rather than only returning the result in chat.
Be explicit about the scheduled action, timing, and any report artifact paths or message content that the later run should produce.
