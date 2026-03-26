from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder

from .session_service import ManagedRuntimeSession


def extract_user_message(input_data: RunAgentInput) -> str:
    messages = list(getattr(input_data, "messages", []) or [])
    for message in reversed(messages):
        if getattr(message, "role", "") != "user":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or "").strip())
                elif hasattr(item, "text"):
                    text = str(getattr(item, "text", "") or "").strip()
                    if text:
                        parts.append(text)
            joined = "\n".join([part for part in parts if part])
            if joined.strip():
                return joined.strip()
    return ""


class AGUIEventBridge:
    def __init__(self, managed_session: ManagedRuntimeSession, *, thread_id: str, run_id: str, accept: str | None) -> None:
        self.managed_session = managed_session
        self.thread_id = thread_id
        self.run_id = run_id
        self.encoder = EventEncoder(accept=accept)
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.message_id = uuid.uuid4().hex
        self._sent_text_chunk = False
        self._text_message_open = False
        self._open_tool_calls: set[str] = set()

    def on_agent_event(self, event: dict[str, Any]) -> None:
        etype = str(event.get("type", "")).strip()
        if etype == "message_update":
            assistant_event = event.get("assistantMessageEvent") or {}
            event_type = str(assistant_event.get("type", "")).strip()
            if event_type == "text_delta":
                delta = str(assistant_event.get("delta") or "")
                if delta:
                    self._sent_text_chunk = True
                    self._ensure_text_message_started()
                    self._emit(
                        TextMessageContentEvent(
                            type=EventType.TEXT_MESSAGE_CONTENT,
                            message_id=self.message_id,
                            delta=delta,
                        )
                    )
                return

        if etype == "tool_execution_start":
            tool_call_id = str(event.get("toolCallId") or uuid.uuid4().hex)
            tool_name = str(event.get("toolName") or "unknown")
            self._ensure_tool_call_started(tool_call_id, tool_name=tool_name)
            self._emit(
                ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=tool_call_id,
                    delta=json.dumps(event.get("args") or {}, default=str),
                )
            )
            return

        if etype == "tool_execution_end":
            tool_call_id = str(event.get("toolCallId") or uuid.uuid4().hex)
            tool_name = str(event.get("toolName") or "unknown")
            self._ensure_tool_call_started(tool_call_id, tool_name=tool_name)
            self._close_tool_call(tool_call_id)
            result = event.get("result")
            detail = getattr(result, "details", None)
            if detail is None and isinstance(result, dict):
                detail = result.get("details")
            self._emit(
                ToolCallResultEvent(
                    type=EventType.TOOL_CALL_RESULT,
                    message_id=uuid.uuid4().hex,
                    tool_call_id=tool_call_id,
                    content=json.dumps(detail or {}, default=str),
                    role="tool",
                )
            )
            return

    def _emit(self, event: Any) -> None:
        self.queue.put_nowait(self._encode(event))

    def _encode(self, event: Any) -> str:
        return self.encoder.encode(event)

    def _ensure_text_message_started(self) -> None:
        if self._text_message_open:
            return
        self._emit(
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=self.message_id,
                role="assistant",
            )
        )
        self._text_message_open = True

    def _close_text_message(self) -> None:
        if not self._text_message_open:
            return
        self._emit(
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=self.message_id,
            )
        )
        self._text_message_open = False

    def _ensure_tool_call_started(self, tool_call_id: str, *, tool_name: str) -> None:
        if tool_call_id in self._open_tool_calls:
            return
        self._emit(
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tool_call_id,
                tool_call_name=tool_name,
                parent_message_id=self.message_id,
            )
        )
        self._open_tool_calls.add(tool_call_id)

    def _close_tool_call(self, tool_call_id: str) -> None:
        if tool_call_id not in self._open_tool_calls:
            return
        self._emit(
            ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=tool_call_id,
            )
        )
        self._open_tool_calls.discard(tool_call_id)

    async def stream_turn(self, user_message: str):
        unsubscribe = self.managed_session.runtime.agent.subscribe(self.on_agent_event)
        try:
            yield self.encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=self.thread_id,
                    run_id=self.run_id,
                )
            )

            turn_task = asyncio.create_task(self.managed_session.run_turn(user_message))

            while True:
                if turn_task.done() and self.queue.empty():
                    break
                try:
                    chunk = await asyncio.wait_for(self.queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if chunk is None:
                    break
                yield chunk

            snapshot = await turn_task
            if snapshot.message and not self._sent_text_chunk:
                if not self._text_message_open:
                    self._text_message_open = True
                    yield self._encode(
                        TextMessageStartEvent(
                            type=EventType.TEXT_MESSAGE_START,
                            message_id=self.message_id,
                            role="assistant",
                        )
                    )
                yield self._encode(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=self.message_id,
                        delta=snapshot.message,
                    )
                )
            if self._text_message_open:
                self._text_message_open = False
                yield self._encode(
                    TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=self.message_id,
                    )
                )
            while self._open_tool_calls:
                tool_call_id = next(iter(self._open_tool_calls))
                self._open_tool_calls.discard(tool_call_id)
                yield self._encode(
                    ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=tool_call_id,
                    )
                )
            yield self._encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=self.thread_id,
                    run_id=self.run_id,
                    result=snapshot.message or None,
                )
            )
        except Exception as exc:
            yield self._encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=str(exc),
                )
            )
        finally:
            unsubscribe()
            self.queue.put_nowait(None)
