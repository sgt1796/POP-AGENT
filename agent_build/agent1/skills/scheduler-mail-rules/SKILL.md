---
name: scheduler-mail-rules
description: Rules for delayed execution and owner email delivery.
kind: tool_rules
priority: 60
tools:
  - task_scheduler
  - agentmail_send
triggers:
  - schedule
  - email
scope: system
---
Use `task_scheduler` when the user asks to run work later or on a recurring cadence with `run_at` or cron-like recurrence.
`task_scheduler` `run_now` marks the task as due now; execution still happens through the active scheduled runner.
Use `agentmail_send` when the user asks to email the configured owner a report, summary, or attachment.
