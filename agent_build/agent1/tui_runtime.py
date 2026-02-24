import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Dict, Generic, Optional, Set, Tuple, TypeVar

from agent import Agent

from .approvals import DEFAULT_TOOL_REJECT_REASON, read_toolsmaker_create_details
from .message_utils import extract_bash_exec_command


@dataclass(frozen=True)
class ToolsmakerDecision:
    approve: bool
    activate: bool = False
    reason: str = DEFAULT_TOOL_REJECT_REASON


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


    if etype in {"tool_execution_result", "tool_execution_error", "memory_context", "memory_lookup"} and level in {"full", "debug"}:
        return f"[context] {event}"

    if etype == "message_end":
        message = event.get("message")
        if getattr(message, "role", None) == "assistant" and getattr(message, "error_message", None):
            return f"[assistant:error] {message.error_message}"

    return None


DecisionResolver = Callable[[Dict[str, Any]], Awaitable[ToolsmakerDecision]]
ActivitySink = Optional[Callable[[str], None]]


class AsyncToolsmakerApprovalSubscriber:
    """Asynchronous manual approval flow for toolsmaker create->approval_required events."""

    def __init__(
        self,
        *,
        agent: Agent,
        resolve_decision: DecisionResolver,
        on_activity: ActivitySink = None,
    ) -> None:
        self.agent = agent
        self._resolve_decision = resolve_decision
        self._on_activity = on_activity
        self._handled: Set[Tuple[str, int]] = set()

    def _activity(self, text: str) -> None:
        if self._on_activity is not None:
            self._on_activity(text)

    def on_event(self, event: Dict[str, Any]) -> None:
        details = read_toolsmaker_create_details(event)
        if details is None:
            return

        name = str(details.get("name", "")).strip()
        try:
            version = int(details.get("version", 0) or 0)
        except Exception:
            version = 0
        if not name or version <= 0:
            return

        key = (name, version)
        if key in self._handled:
            return
        self._handled.add(key)

        self._activity(f"[toolsmaker] approval requested tool={name} version={version}")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._handle_decision(details, name, version))
        except RuntimeError as exc:
            self._activity(f"[toolsmaker] approval dispatch warning for {name} v{version}: {exc}")

    async def _handle_decision(self, details: Dict[str, Any], name: str, version: int) -> None:
        try:
            decision = await self._resolve_decision(details)
        except Exception as exc:
            self._activity(f"[toolsmaker] decision warning for {name} v{version}: {exc}")
            decision = ToolsmakerDecision(approve=False, activate=False, reason="decision_error")

        reason = str(decision.reason or "").strip() or DEFAULT_TOOL_REJECT_REASON

        if not decision.approve:
            try:
                rejected = self.agent.reject_dynamic_tool(name=name, version=version, reason=reason)
                self._activity(f"[toolsmaker] rejected tool={name} version={version} status={rejected.status}")
            except Exception as exc:
                self._activity(f"[toolsmaker] reject warning for {name} v{version}: {exc}")
            return

        try:
            approved = self.agent.approve_dynamic_tool(name=name, version=version)
            self._activity(f"[toolsmaker] approved tool={name} version={version} status={approved.status}")
        except Exception as exc:
            self._activity(f"[toolsmaker] approve warning for {name} v{version}: {exc}")
            return

        if not decision.activate:
            self._activity(f"[toolsmaker] activation skipped tool={name} version={version}")
            return

        try:
            activated = self.agent.activate_tool_version(name=name, version=version)
            self._activity(f"[toolsmaker] activated tool={activated.name} version={version}")
        except Exception as exc:
            self._activity(f"[toolsmaker] activation warning for {name} v{version}: {exc}")
