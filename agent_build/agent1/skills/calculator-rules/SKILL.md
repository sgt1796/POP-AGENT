---
name: calculator-rules
description: Rules for using the restricted calculator expression tool.
kind: tool_rules
priority: 90
tools:
  - calculator
triggers:
  - calculator
  - count
scope: system
---
Use `calculator` for arithmetic, unit conversions, explicit list/count logic, checksum work, and small brute-force enumeration before reaching for shell execution.
Call allowed functions directly such as `sin`, `cos`, `acos`, `asin`, `atan2`, `radians`, `sqrt`, `max`, `min`, `sum`, `len`, and `enumerate`.
Calculator accepts one restricted expression; use `bindings` for long tables or lists instead of multiline Python, imports, lambdas, or attribute access like `math.sin`.
If a calculator call fails because a name or function is unsupported, rewrite it as a direct allowed call or move the data into `bindings`, then retry once instead of guessing.
