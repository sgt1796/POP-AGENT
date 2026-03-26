from __future__ import annotations

import asyncio
import os
import uuid

import uvicorn
from ag_ui.core import RunAgentInput
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .agui import AGUIEventBridge, extract_user_message
from .schemas import ApprovalDecisionRequest, TurnRequest
from .session_service import WebRuntimeService, _normalize_session_id, _sse_payload


service = WebRuntimeService()


def _allowed_origins() -> list[str]:
    defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]
    extra = [
        item.strip()
        for item in str(os.getenv("POP_AGENT_WEB_ALLOW_ORIGINS", "") or "").split(",")
        if item.strip()
    ]
    seen: set[str] = set()
    values: list[str] = []
    for origin in defaults + extra:
        if origin in seen:
            continue
        seen.add(origin)
        values.append(origin)
    return values


app = FastAPI(title="POP Agent Web Adapter", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    await service.initialize()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await service.shutdown()


@app.get("/api/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "name": "POP Agent Web Adapter",
        "ok": True,
        "health": "/api/health",
        "sessions": "/api/web/sessions",
        "agui": "/api/agui/default",
    }


@app.get("/api/web/sessions")
async def list_sessions() -> JSONResponse:
    return JSONResponse(
        {
            "sessions": service.list_known_sessions(),
            "scheduler_status": service.build_scheduler_status().model_dump(exclude_none=True),
        }
    )


@app.get("/api/web/sessions/{session_id}")
async def get_session_snapshot(session_id: str) -> JSONResponse:
    snapshot = await service.get_snapshot(session_id)
    return JSONResponse(snapshot.model_dump(exclude_none=True))


@app.post("/api/web/sessions/{session_id}/turns")
async def run_turn(session_id: str, payload: TurnRequest) -> JSONResponse:
    session = await service.get_or_create_session(session_id)
    try:
        snapshot = await session.run_turn(payload.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(snapshot.model_dump(exclude_none=True))


@app.post("/api/web/sessions/{session_id}/approval")
async def resolve_approval(session_id: str, payload: ApprovalDecisionRequest) -> JSONResponse:
    try:
        approved = await service.resolve_approval(session_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    snapshot = await service.get_snapshot(session_id)
    return JSONResponse(
        {
            "approved": approved,
            "snapshot": snapshot.model_dump(exclude_none=True),
        }
    )


@app.post("/api/web/sessions/{session_id}/shutdown")
async def shutdown_session(session_id: str) -> dict[str, bool]:
    await service.shutdown_session(session_id)
    return {"ok": True}


@app.get("/api/web/sessions/{session_id}/stream")
async def stream_session_snapshot(session_id: str, request: Request) -> StreamingResponse:
    session = await service.get_or_create_session(session_id)
    queue = session.add_stream_subscriber()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield _sse_payload(payload, event="snapshot")
        finally:
            session.remove_stream_subscriber(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/agui/default")
async def agui_default(input_data: RunAgentInput, request: Request) -> StreamingResponse:
    query_session_id = request.query_params.get("session_id")
    session_id = _normalize_session_id(query_session_id or getattr(input_data, "thread_id", None))
    user_message = extract_user_message(input_data)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message was found in the AG-UI payload.")

    managed_session = await service.get_or_create_session(session_id)
    bridge = AGUIEventBridge(
        managed_session,
        thread_id=session_id,
        run_id=str(getattr(input_data, "run_id", None) or uuid.uuid4().hex),
        accept=request.headers.get("accept"),
    )
    return StreamingResponse(
        bridge.stream_turn(user_message),
        media_type=bridge.encoder.get_content_type(),
    )


def main() -> None:
    port = int(os.getenv("POP_AGENT_WEB_PORT", os.getenv("PORT", "8000")))
    uvicorn.run(
        "agent_build.agent1.web.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
