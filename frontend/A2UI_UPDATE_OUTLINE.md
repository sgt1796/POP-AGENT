# A2UI Update Outline

This outline describes the next update for the POP Agent web UI's assistant-to-UI path.

## Goal

Turn high-value assistant responses into reliable structured UI without breaking the normal markdown chat experience.

## Current State

- Chat rendering is powered by CopilotKit in `frontend/copilotkit-app`.
- Live assistant streaming reaches the chat through the AG-UI bridge in `agent_build/agent1/web/agui.py`.
- Side-panel and generated cards are driven by session snapshots from `agent_build/agent1/web/session_service.py`.
- Assistant-generated UI currently supports only:
  - `PlanChecklist`
  - `ResultTable`
- Generated UI is inferred heuristically from markdown in `agent_build/agent1/web/ui_extraction.py`.
- `ui_spec` is singular, so one response can produce at most one generated card.

## What Was Just Fixed

1. The AG-UI bridge now emits a valid text lifecycle:
   - `TEXT_MESSAGE_START`
   - `TEXT_MESSAGE_CONTENT`
   - `TEXT_MESSAGE_END`
   - `RUN_FINISHED`
2. Tool events now emit a valid tool lifecycle:
   - `TOOL_CALL_START`
   - `TOOL_CALL_ARGS`
   - `TOOL_CALL_END`
   - `TOOL_CALL_RESULT`
3. Markdown summaries are no longer misclassified as checklists just because they contain bullets.

## Update Scope

### Phase 1: Stabilize The Existing A2UI Contract

- Keep chat-first behavior as the default.
- Only promote responses into UI when the structure is clear and safe.
- Preserve plain markdown answers when extraction confidence is low.
- Keep AG-UI event emission strict and test-covered.

### Phase 2: Expand Generated UI Components

Add new assistant-generated card types with strong schemas.

Recommended order:

1. `StatGrid`
   - Best for market summaries, KPIs, scorecards, and dashboards.
2. `KeyValueList`
   - Best for compact summaries, report metadata, and risk summaries.
3. `Callout`
   - Best for warnings, blockers, and headline conclusions.
4. `SourceList`
   - Best for citations and link-heavy research responses.
5. `Timeline`
   - Best for event sequences, execution history, and incident summaries.

### Phase 3: Support Multiple Generated Cards Per Response

- Change `ui_spec` to `ui_specs`.
- Allow one answer to produce mixed layouts such as:
  - `StatGrid` for headline metrics
  - `Callout` for market driver
  - `SourceList` for citations
- Keep ordering deterministic.

### Phase 4: Move Away From Markdown Heuristics

Replace reverse-parsing of markdown with an explicit A2UI payload contract.

Recommended direction:

- The model emits a validated JSON block for UI.
- The backend validates it with Pydantic.
- The frontend renders only known schema types.
- The markdown answer remains available as a fallback.

Example target shape:

```json
{
  "type": "StatGrid",
  "props": {
    "title": "US Market",
    "items": [
      { "label": "S&P 500", "value": "6,477.16", "delta": "-1.74%" },
      { "label": "Nasdaq", "value": "21,408.08", "delta": "-2.38%" }
    ]
  }
}
```

### Phase 5: Add Stronger Validation And Observability

- Add backend unit tests for extraction and schema validation.
- Add bridge tests for AG-UI event order and completion.
- Add frontend render tests for each generated component.
- Log when extraction succeeds, fails, or falls back to markdown.
- Track malformed UI payloads separately from normal assistant errors.

## File-Level Work Plan

### Backend

- `agent_build/agent1/web/schemas.py`
  - Add new UI spec models.
- `agent_build/agent1/web/ui_extraction.py`
  - Add or remove heuristic extractors depending on the phase.
- `agent_build/agent1/web/session_service.py`
  - Upgrade snapshot shape if multi-card support is added.
- `agent_build/agent1/web/agui.py`
  - Keep AG-UI event emission compliant and regression-tested.

### Frontend

- `frontend/copilotkit-app/lib/types.ts`
  - Mirror backend schema types.
- `frontend/copilotkit-app/components/ui-cards.tsx`
  - Implement new card renderers.
- `frontend/copilotkit-app/components/pop-ui-renderer.tsx`
  - Route new card types.
- `frontend/copilotkit-app/components/chat-shell.tsx`
  - Support multi-card layout if `ui_specs` is introduced.

### Tests

- `tests/test_agent1_web_ui.py`
  - Coverage for extraction and non-extraction cases.
- `tests/test_agent1_web_agui.py`
  - Coverage for AG-UI stream correctness.

## Acceptance Criteria

- Normal assistant prose still renders cleanly in chat.
- Structured outputs render as cards without raw markdown leakage.
- AG-UI streams always produce a complete lifecycle.
- At least one new generated card type is added and tested.
- The system can clearly fall back to chat markdown when UI extraction is uncertain.

## Recommended Next Milestone

Implement `StatGrid` first, then migrate from singular `ui_spec` to `ui_specs`, then introduce an explicit JSON A2UI payload contract.
