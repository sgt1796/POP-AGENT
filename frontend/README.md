# Frontend README

This folder contains the web UI documentation for the POP Agent CopilotKit app and its assistant-to-UI integration.

## What This Frontend Is

The web UI is a Next.js app in `frontend/copilotkit-app` that sits on top of the existing Python runtime. It has two separate but connected paths:

1. Chat path
   - CopilotKit renders the user/assistant conversation.
   - The frontend proxies chat requests to the Python backend's AG-UI endpoint.
2. Session snapshot path
   - The frontend subscribes to a server-sent event stream for session state.
   - That stream powers the status header, approval cards, tool activity, recent events, session list, scheduler state, and assistant-generated UI cards.

This split is intentional:

- AG-UI powers conversational streaming.
- Session snapshots power operational and structured UI state.

## Key Folders

### `frontend/copilotkit-app`

The actual Next.js application.

Important files:

- `app/page.tsx`
  - Entry page. Renders `ChatShell`.
- `app/layout.tsx`
  - Global layout and CopilotKit stylesheet import.
- `app/globals.css`
  - App-wide styling for the dashboard and card layout.
- `app/api/copilotkit/route.ts`
  - Next.js route that creates a Copilot runtime and forwards chat traffic to the Python AG-UI endpoint.
- `app/api/pop-agent/sessions/[sessionId]/stream/route.ts`
  - Proxy for the session snapshot SSE stream.
- `app/api/pop-agent/sessions/[sessionId]/approval/route.ts`
  - Proxy for approval actions.
- `components/chat-shell.tsx`
  - Main page composition. Builds the workspace columns, tracks generated UI entries by assistant transcript message ID, and wires chat/events/sidebar cards into the dock system.
- `components/copilot-chat-panel.tsx`
  - Thin CopilotKit wrapper around `CopilotChat`.
- `components/generated-ui-chat.tsx`
  - Generated UI docking model. Matches extracted cards to assistant messages, derives dynamic drop zones from rendered panels, and manages drag/drop state.
- `components/generated-ui-assistant-message.tsx`
  - Custom CopilotKit assistant message wrapper that attaches the inline generated UI card or moved-note to the matching assistant message.
- `components/pop-ui-renderer.tsx`
  - Generated assistant UI renderer.
- `components/ui-cards.tsx`
  - Shared card components for generated UI and runtime UI.
- `lib/use-session-stream.ts`
  - Client hook for session snapshot SSE and approval actions.
- `lib/types.ts`
  - Frontend type contract for snapshot data and supported UI specs.
- `lib/backend-base-url.ts`
  - Backend URL resolution from environment variables.

## Related Backend Files

The frontend depends on these Python files:

- `agent_build/agent1/web/app.py`
  - FastAPI web adapter. Exposes health, session APIs, SSE session stream, and AG-UI chat endpoint.
- `agent_build/agent1/web/agui.py`
  - Bridge between internal runtime events and AG-UI streaming events expected by CopilotKit.
- `agent_build/agent1/web/session_service.py`
  - Session lifecycle, snapshot construction, activity log, approvals, tool progress, and scheduler state.
- `agent_build/agent1/web/ui_extraction.py`
  - Heuristic markdown-to-UI extraction for assistant-generated cards.
- `agent_build/agent1/web/schemas.py`
  - Shared backend schema for snapshot payloads and UI spec types.
- `run_web_ui.py`
  - Convenience launcher that starts both the Python backend and the Next frontend.

## Architecture

### 1. Chat Request Flow

1. The user sends a message in the browser.
2. `CopilotChat` posts to `/api/copilotkit`.
3. `app/api/copilotkit/route.ts` creates a `HttpAgent` that points to the Python endpoint:
   - `/api/agui/default?session_id=...`
4. The Python AG-UI endpoint runs the turn through the runtime.
5. `agent_build/agent1/web/agui.py` converts runtime events into AG-UI events.
6. CopilotKit renders the assistant response in the chat surface.

### 2. Session Snapshot Flow

1. `useSessionStream()` opens an `EventSource` to:
   - `/api/pop-agent/sessions/{sessionId}/stream`
2. The Next proxy forwards that to the Python backend.
3. `session_service.py` publishes serialized `SessionSnapshot` payloads whenever the session changes.
4. `ChatShell` consumes the latest snapshot and updates:
   - header status
   - generated assistant UI entries
   - recent events
   - approvals
   - tool progress
   - sessions
   - scheduler state

## Generated Assistant UI

The assistant can currently generate one structured card per response through `ui_spec`.
The frontend preserves those cards across the conversation as per-message generated UI entries, so older assistant replies can still keep their own card after later replies arrive.

Supported generated card types:

- `StatGrid`
- `PlanChecklist`
- `ResultTable`

These are distinct from runtime-managed cards such as:

- `ApprovalCard`
- `ToolProgressList`
- `SessionSwitcher`
- `SchedulerStatus`

### How Generated UI Works Today

1. The assistant finishes a normal text reply.
2. `session_service.py` stores that reply in `last_message`, appends it to `transcript`, and extracts one `ui_spec` from the same reply text.
3. `ChatShell` records that `ui_spec` as a generated UI entry keyed by the assistant transcript message ID.
4. `generated-ui-assistant-message.tsx` attaches that entry back to the matching CopilotKit assistant message.
5. If the card stays inline, it renders directly under that assistant message.
6. If the user drags it out, `generated-ui-chat.tsx` docks it into a workspace insertion point and replaces the inline card with a single muted "Generated UI moved to ..." note.
7. Drop targets are derived from the currently rendered panels in the main column and sidebar, so cards can move above, below, or between existing boxes without hardcoding fixed slots.

### Important Limitation

`ui_spec` is singular. The assistant can only generate one card per answer today.
The frontend can keep multiple generated cards across the conversation, but still only one new structured card can be extracted from each assistant reply.

If the system needs richer output, the next step is to move to `ui_specs: UIComponent[]`.

## Recent Fixes In This Update

### AG-UI Streaming Fix

The original AG-UI bridge was not reliably emitting a complete assistant message lifecycle for CopilotKit. The bridge now emits:

- `TEXT_MESSAGE_START`
- `TEXT_MESSAGE_CONTENT`
- `TEXT_MESSAGE_END`
- `RUN_FINISHED`

It also emits a proper tool lifecycle:

- `TOOL_CALL_START`
- `TOOL_CALL_ARGS`
- `TOOL_CALL_END`
- `TOOL_CALL_RESULT`

This matters because CopilotKit expects complete event boundaries to finalize chat messages.

### UI Extraction Safety Fix

The original markdown extractor was too permissive and converted ordinary summary bullet lists into `PlanChecklist` cards. That caused raw markdown-like content to appear in structured UI.

The extractor is now stricter:

- plain report bullets stay in chat markdown
- actual plans/checklists still promote to `PlanChecklist`
- markdown tables still promote to `ResultTable`

### Generated UI Workspace Fix

The original generated UI behavior only supported a few hardcoded dock positions and could mis-bind extracted cards to the wrong assistant messages.

The current implementation now:

- tracks generated UI entries by assistant transcript message ID
- matches cards back onto CopilotKit messages by message identity and normalized content
- keeps one moved-note per message instead of repeating the same note across later assistant replies
- derives drop targets from the rendered workspace panels instead of baking positions into the component
- lets the user move cards above, below, or between the currently visible dashboard boxes

### SSR Safety Fix

The custom assistant message integration was split so the CopilotKit message renderer stays in a client-only component.

This avoids pulling browser-only A2UI code into the server render path and prevents `CustomEvent is not defined` during Next.js SSR.

## Running The UI

### Preferred

From the repository root:

```bash
python run_web_ui.py
```

Default ports:

- backend: `8000`
- frontend: `3001`

### Manual

Backend:

```bash
python -m uvicorn agent_build.agent1.web.app:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend/copilotkit-app
npm run dev -- --port 3001
```

## Environment Variables

- `POP_AGENT_API_BASE_URL`
  - Backend base URL used by the Next app.
- `NEXT_PUBLIC_POP_AGENT_API_BASE_URL`
  - Public backend base URL fallback for the browser-side app.
- `POP_AGENT_WEB_PORT`
  - Python backend port.
- `POP_AGENT_FRONTEND_PORT`
  - Next.js frontend port.
- `POP_AGENT_WEB_ALLOW_ORIGINS`
  - Extra allowed CORS origins for the backend adapter.

## Testing

The most relevant regression tests for this update are:

```bash
pytest -s tests/test_agent1_web_agui.py tests/test_agent1_web_ui.py
```

What they cover:

- AG-UI event translation from runtime events
- final text message lifecycle emission
- no duplicate fallback text after live deltas
- markdown table extraction
- checklist extraction
- non-checklist summary bullets staying as plain markdown

## How To Add A New Generated UI Component

If you want the assistant to render a new card such as `StatGrid` or `SourceList`, the work spans both backend and frontend.

### Backend Steps

1. Add the new schema in `agent_build/agent1/web/schemas.py`.
2. Include it in the `UIComponent` union.
3. Decide how the assistant creates it:
   - short-term: extend `ui_extraction.py`
   - better long-term: emit explicit structured JSON
4. Update `session_service.py` if snapshot shape changes.

### Frontend Steps

1. Add the mirrored type in `frontend/copilotkit-app/lib/types.ts`.
2. Add the React renderer in `frontend/copilotkit-app/components/ui-cards.tsx` or a new component file.
3. Add the route in `frontend/copilotkit-app/components/pop-ui-renderer.tsx`.
4. Update layout logic if multiple generated cards are introduced.

### Testing Steps

1. Add backend extraction tests.
2. Add render tests if frontend test coverage is introduced.
3. Manually verify:
   - normal markdown fallback
   - correct card rendering
   - no raw markdown leaking into cards

## Recommended Next Improvements

1. Replace `ui_spec` with `ui_specs` so one assistant reply can intentionally emit multiple cards.
2. Move from markdown heuristics to an explicit validated A2UI payload.
3. Persist dock placement per generated UI entry if layout state should survive reloads.
4. Add citations as a `SourceList` component rather than keeping them only in prose.
5. Add frontend test coverage for the generated UI docking and message-assignment logic.

## Troubleshooting

### The session stream is open, but no chat answer appears

Check the AG-UI stream path first. The browser chat depends on `/api/agui/default`, not the session snapshot feed.

Expected assistant event lifecycle:

- `RUN_STARTED`
- `TEXT_MESSAGE_START`
- one or more `TEXT_MESSAGE_CONTENT`
- `TEXT_MESSAGE_END`
- `RUN_FINISHED`

If the backend was restarted before the frontend, refresh the page and retry.

### The assistant answer shows as a strange checklist

That usually means the markdown extractor promoted the answer into `PlanChecklist`. Tighten the extraction rules or force a plain markdown answer.

### The answer should be structured, but it stays plain markdown

That is the intended fallback when extraction confidence is low. The current system prefers preserving the chat answer over generating a misleading card.

If a reply should have generated a card but did not, first check whether `ui_extraction.py` recognizes the markdown pattern. The frontend only renders generated UI when the snapshot includes `ui_spec`.

### Generated UI repeats or attaches to the wrong message

That usually means the browser is still running stale dev state from before a frontend change.

Try this in order:

- restart `next dev`
- create a new session from the `Sessions` card
- retry with a fresh assistant reply that should extract to `StatGrid`, `ResultTable`, or `PlanChecklist`

### Browser console shows `favicon.ico 404`

That is not relevant to chat or A2UI behavior.

### Browser console shows React DevTools or Lit dev warnings

Those are normal development warnings and not part of the AG-UI or A2UI contract.
