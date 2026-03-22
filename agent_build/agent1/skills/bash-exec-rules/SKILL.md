---
name: bash-exec-rules
description: Hard rules for using bash_exec safely inside the configured allowlists.
kind: tool_rules
priority: 100
tools:
  - bash_exec
triggers:
  - bash_exec
  - shell
scope: system
---
Use `bash_exec` only for allowed shell inspection or edits inside policy.
It runs one program without a shell, so do not rely on pipes, redirection, `&&`, `||`, heredocs, or shell builtins.
Treat `command_not_allowed`, `blocked_shell_operator`, `command_not_available_on_host`, `approval_required_or_denied`, and `path_outside_*` as hard constraints and switch tools instead of retrying shell syntax variants.
