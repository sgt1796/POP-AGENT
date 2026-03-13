import asyncio
from collections import deque
from typing import Any, Deque, Dict, Generic, Optional, TypeVar

from .message_utils import extract_bash_exec_command


T = TypeVar("T")


class AsyncDecisionQueue(Generic[T]):
    """Small FIFO queue with async `get` and immediate `put` support."""

    def __init__(self) -> None:
        self._items: Deque[T] = deque()
        self._waiters: Deque[asyncio.Future[T]] = deque()

    def put(self, item: T) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if waiter.done():
                continue
            waiter.set_result(item)
            return
        self._items.append(item)

    async def get(self) -> T:
        if self._items:
            return self._items.popleft()
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[T] = loop.create_future()
        self._waiters.append(waiter)
        return await waiter

    def __len__(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        return len(self._items) == 0

    def drain(self) -> list[T]:
        """Remove and return all currently queued items."""
        if not self._items:
            return []
        items = list(self._items)
        self._items.clear()
        return items


def format_activity_event(event: Dict[str, Any], level: str = "full") -> Optional[str]:
    etype = str(event.get("type", "")).strip()

    if etype == "message_update":
        assistant_event = event.get("assistantMessageEvent") or {}
        assistant_event_type = str(assistant_event.get("type", "")).strip()
        if assistant_event_type in {"toolcall_start", "toolcall_delta", "toolcall_end"}:
            if level == "simple" and assistant_event_type == "toolcall_delta":
                return None
            call_id = ""
            tool_name = "unknown"
            args = None
            partial = assistant_event.get("partial")
            if isinstance(partial, dict):
                content = partial.get("content")
                if isinstance(content, list):
                    for item in reversed(content):
                        if not isinstance(item, dict):
                            continue
                        if str(item.get("type", "")).strip() != "toolCall":
                            continue
                        call_id = str(item.get("id", "")).strip()
                        tool_name = str(item.get("name", "")).strip() or "unknown"
                        args = item.get("arguments")
                        break
            suffix = f" id={call_id}" if call_id else ""
            if level == "simple" or args in (None, ""):
                return f"[tool:call] {assistant_event_type} {tool_name}{suffix}"
            return f"[tool:call] {assistant_event_type} {tool_name}{suffix} args={args}"

    if etype == "tool_execution_start":
        tool_name = str(event.get("toolName", "")).strip() or "unknown"
        if level == "simple":
            if tool_name == "bash_exec":
                command = extract_bash_exec_command(event)
                preview = " ".join(command.split()[:6]) if command else ""
                if preview:
                    return f"[tool:start] bash_exec cmd={preview}"
            return f"[tool:start] {tool_name}"
        return f"[tool:start] {tool_name} args={event.get('args')}"

    if etype == "tool_execution_end":
        tool_name = str(event.get("toolName", "")).strip() or "unknown"
        is_error = bool(event.get("isError"))
        command = extract_bash_exec_command(event) if tool_name == "bash_exec" else ""
        if level == "simple" and command:
            preview = " ".join(command.split()[:6])
            return f"[tool:end] {tool_name} error={is_error} cmd={preview}"
        if command:
            return f"[tool:end] {tool_name} error={is_error} cmd={command}"
        return f"[tool:end] {tool_name} error={is_error}"

    if etype == "tool_policy_blocked":
        return f"[tool:blocked] {event.get('toolName')} error={event.get('error')}"

    if etype == "message_update" and level in {"full", "debug"}:
        assistant_event = event.get("assistantMessageEvent") or {}
        if assistant_event.get("type") == "text_delta":
            delta = str(assistant_event.get("delta", "") or "")
            if delta:
                return f"[stream] {delta}"

    if etype in {"tool_execution_result", "tool_execution_error", "memory_context", "memory_lookup"} and level in {
        "full",
        "debug",
    }:
        return f"[context] {event}"

    if etype == "message_end":
        message = event.get("message")
        if getattr(message, "role", None) == "assistant" and getattr(message, "error_message", None):
            return f"[assistant:error] {message.error_message}"

    return None
