"""High level Agent class.

This module exposes the :class:`Agent` class which coordinates
conversation state, tool definitions and the asynchronous agent loop.
It provides a user friendly API for sending prompts, continuing
conversations, queueing steering or follow‑up messages and
subscribing to lifecycle events.  Internally it delegates all heavy
lifting to the functions in :mod:`pop_agent.agent_loop`.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import os
import time
from collections import deque
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .agent_loop import AgentLoopConfig, agent_loop, agent_loop_continue
from .event_stream import EventStream
from .agent_types import (
    AgentContext,
    AgentEvent,
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolResult,
    ThinkingContent,
    ThinkingLevel,
    ToolCallContent,
    TextContent,
    ImageContent,
)
from .scheduler import (
    ScheduledTaskStore,
    DEFAULT_SCHEDULED_JOBS_PATH,
    DEFAULT_RUNNER_LOCK_PATH,
)
from .usage_tracking import accumulate_totals, ensure_usage_record, init_usage_totals

# Attempt to import POP to obtain a default model and stream function
try:
    import POP  # type: ignore
except ImportError:
    POP = None


def _default_convert_to_llm(messages: List[AgentMessage]) -> Awaitable[List[dict]]:
    """Default conversion from AgentMessage objects to LLM message dictionaries.

    Only messages with roles ``user``, ``assistant`` or ``toolResult``
    are forwarded to the LLM. Other message types (e.g. custom
    notifications) are discarded.

    The payload sent back to the LLM is intentionally minimal. Rich
    metadata such as usage accounting, timestamps, and tool ``details``
    remain available in the in-memory transcript for UI/reporting, but
    they are not echoed back into the model context because they can
    dramatically inflate token usage without helping the next turn.
    """
    def _sanitize_content_item(item: Any) -> Optional[Dict[str, Any]]:
        def _extract_tool_call_signature(payload: Any) -> Optional[Dict[str, Any]]:
            if isinstance(payload, ToolCallContent):
                extra = payload.extra_content
                if isinstance(extra, dict):
                    google = extra.get("google")
                    if isinstance(google, dict):
                        signature = str(google.get("thought_signature") or "").strip()
                        if signature:
                            return {"google": {"thought_signature": signature}}
                return None
            if isinstance(payload, dict):
                extra = payload.get("extra_content")
                if extra is None:
                    extra = payload.get("extraContent")
                if isinstance(extra, dict):
                    google = extra.get("google")
                    if isinstance(google, dict):
                        signature = str(google.get("thought_signature") or google.get("thoughtSignature") or "").strip()
                        if signature:
                            return {"google": {"thought_signature": signature}}
                signature = str(payload.get("thought_signature") or payload.get("thoughtSignature") or "").strip()
                if signature:
                    return {"google": {"thought_signature": signature}}
            return None

        if isinstance(item, TextContent):
            return {"type": "text", "text": str(item.text or "")}
        if isinstance(item, ToolCallContent):
            arguments = item.arguments if isinstance(item.arguments, dict) else {}
            sanitized = {
                "type": "toolCall",
                "id": str(item.id or ""),
                "name": str(item.name or ""),
                "arguments": dict(arguments),
            }
            signature = _extract_tool_call_signature(item)
            if signature is not None:
                sanitized["extra_content"] = signature
            return sanitized
        if isinstance(item, ImageContent):
            return {
                "type": "image",
                "data": item.data,
                "mime_type": str(item.mime_type or "image/png"),
            }
        if isinstance(item, ThinkingContent):
            return None
        if dataclasses.is_dataclass(item):
            try:
                return _sanitize_content_item(dataclasses.asdict(item))
            except Exception:
                return None
        if isinstance(item, dict):
            item_type = str(item.get("type") or "").strip()
            if item_type == "text":
                return {"type": "text", "text": str(item.get("text") or "")}
            if item_type == "toolCall":
                arguments = item.get("arguments")
                sanitized = {
                    "type": "toolCall",
                    "id": str(item.get("id") or ""),
                    "name": str(item.get("name") or ""),
                    "arguments": dict(arguments) if isinstance(arguments, dict) else {},
                }
                signature = _extract_tool_call_signature(item)
                if signature is not None:
                    sanitized["extra_content"] = signature
                return sanitized
            if item_type == "image":
                return {
                    "type": "image",
                    "data": item.get("data", b""),
                    "mime_type": str(item.get("mime_type") or "image/png"),
                }
            if item_type == "thinking":
                return None
            return None
        text = str(item or "")
        if not text:
            return None
        return {"type": "text", "text": text}

    llm_msgs: List[dict] = []
    for m in messages:
        if m.role in {"user", "assistant", "toolResult"}:
            try:
                content: List[Dict[str, Any]] = []
                for item in m.content:
                    sanitized = _sanitize_content_item(item)
                    if sanitized is not None:
                        content.append(sanitized)
                payload: Dict[str, Any] = {
                    "role": m.role,
                    "content": content,
                }
                if m.role == "toolResult":
                    if m.tool_call_id is not None:
                        payload["toolCallId"] = m.tool_call_id
                    if m.tool_name is not None:
                        payload["toolName"] = m.tool_name
                llm_msgs.append(payload)
            except Exception:
                # Fallback: best effort conversion
                fallback_content: List[Dict[str, Any]] = []
                for item in m.content:
                    sanitized = _sanitize_content_item(item)
                    if sanitized is not None:
                        fallback_content.append(sanitized)
                llm_msgs.append({
                    "role": m.role,
                    "content": fallback_content,
                })
    async def _return():
        return llm_msgs
    return _return()


class Agent:
    """Stateful conversation manager.

    An :class:`Agent` instance encapsulates a conversation with a
    Large Language Model.  It tracks the system prompt, chosen
    model, previous messages and tool definitions.  The agent
    exposes methods to send prompts, continue a previous turn
    (useful after handling tool calls), queue steering messages to
    interrupt long running operations and queue follow‑up messages
    that are delivered only once the agent finishes processing all
    tool calls.  Callbacks can be registered via :meth:`subscribe`
    to observe fine grained events emitted during each turn.

    Parameters
    ----------
    opts : dict, optional
        Configuration dictionary.  All keys are optional; unknown keys
        are ignored.  Recognised keys include:

        ``initial_state`` : dict
            Override the default initial state of the agent.  See
            :class:`AgentState` for the fields.
        ``convert_to_llm`` : callable
            Custom function used to convert the internal message list
            into the LLM format.  If omitted the default
            implementation forwards user, assistant and tool result
            messages only.
        ``transform_context`` : callable
            Optional function that can prune or augment the context
            prior to each LLM call.  See :class:`AgentLoopConfig`.
        ``steering_mode`` : {"all", "one-at-a-time"}
            Determines whether all queued steering messages are sent
            at once or one per turn.
        ``follow_up_mode`` : {"all", "one-at-a-time"}
            Determines whether all queued follow‑up messages are sent
            at once or one per turn.
        ``stream_fn`` : callable
            LLM transport function.  Defaults to
            ``POP.stream.stream`` if available.
        ``session_id`` : str
            Session identifier forwarded to the LLM provider.
        ``get_api_key`` : callable
            Function returning an API key for each LLM call.
        ``thinking_budgets`` : dict
            Token budgets for reasoning levels.
        ``max_retry_delay_ms`` : int
            Maximum backoff delay in milliseconds to honour.
        ``request_timeout_s`` : float
            Overall timeout in seconds for each LLM request.
    """

    def __init__(self, opts: Optional[Dict[str, Any]] = None) -> None:
        opts = opts or {}
        # Initialise state with sensible defaults
        # Model: use POP.get_model to check if POP is available and the default model can be retrieved; if not, use a placeholder
 
        try:
            POP.get_client("gemini", "gemini-2.5-flash-preview")  # type: ignore
            default_model = {"provider": "gemini", "id": "gemini-2.5-flash-preview", "api": None}  # type: ignore
        except Exception as e:
            print(f"[ initializing ] POP exception: {e}.")
            default_model = {"provider": "unknown", "id": "unknown", "api": None}
        
        initial = AgentState(
            system_prompt="",
            model=default_model,
            thinking_level="off",
            tools=[],
            messages=[],
            is_streaming=False,
            stream_message=None,
            pending_tool_calls=set(),
            error=None,
        )
        # Override state from options if provided
        for key, value in opts.get("initial_state", {}).items():
            if hasattr(initial, key):
                setattr(initial, key, value)
        self._state: AgentState = initial
        # Event listeners
        self._listeners: set[Callable[[AgentEvent], None]] = set()
        project_root = opts.get("project_root", os.getcwd())
        scheduler_jobs_path = opts.get("scheduled_jobs_path", os.path.join(project_root, DEFAULT_SCHEDULED_JOBS_PATH))
        scheduler_lock_path = opts.get("scheduled_jobs_lock_path", os.path.join(project_root, DEFAULT_RUNNER_LOCK_PATH))
        self._scheduled_task_store = ScheduledTaskStore(
            project_root=project_root,
            jobs_path=scheduler_jobs_path,
            runner_lock_path=scheduler_lock_path,
        )
        # Conversion and context transform functions
        self._convert_to_llm = opts.get("convert_to_llm", _default_convert_to_llm)
        self._transform_context = opts.get("transform_context")
        # Steering and follow‑up queues and modes
        self._steering_queue: List[AgentMessage] = []
        self._follow_up_queue: List[AgentMessage] = []
        self._steering_mode: str = opts.get("steering_mode", "one-at-a-time")
        self._follow_up_mode: str = opts.get("follow_up_mode", "one-at-a-time")
        # Stream function and session configuration
        self.stream_fn = opts["stream_fn"] if "stream_fn" in opts else POP.stream.stream
        self._session_id: Optional[str] = opts.get("session_id")
        self.get_api_key: Optional[Callable[[str], Awaitable[Optional[str]]]] = opts.get("get_api_key")
        self._thinking_budgets: Optional[Dict[str, Any]] = opts.get("thinking_budgets")
        self._max_retry_delay_ms: Optional[int] = opts.get("max_retry_delay_ms")
        self._request_timeout_s: Optional[float] = opts.get("request_timeout_s", 120.0)
        # Usage tracking state (session-memory only)
        self._last_usage: Optional[Dict[str, Any]] = None
        self._usage_history: "deque[Dict[str, Any]]" = deque(maxlen=200)
        self._usage_totals: Dict[str, Any] = init_usage_totals()
        # Internal synchronization primitives
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # agent starts idle
        self._abort_event: Optional[asyncio.Event] = None

    # ---------------------------------------------------------------------
    # Properties and accessors

    @property
    def state(self) -> AgentState:
        """Return the current agent state.

        The returned object should be considered read only; mutating
        it directly may corrupt the agent.  Use the setter methods
        provided to modify the state.
        """
        return self._state

    @property
    def session_id(self) -> Optional[str]:
        """Get the current LLM provider session identifier."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        """Set the session identifier used when calling the LLM."""
        self._session_id = value

    @property
    def thinking_budgets(self) -> Optional[Dict[str, Any]]:
        return self._thinking_budgets

    @thinking_budgets.setter
    def thinking_budgets(self, value: Optional[Dict[str, Any]]) -> None:
        self._thinking_budgets = value

    @property
    def max_retry_delay_ms(self) -> Optional[int]:
        return self._max_retry_delay_ms

    @max_retry_delay_ms.setter
    def max_retry_delay_ms(self, value: Optional[int]) -> None:
        self._max_retry_delay_ms = value

    @property
    def request_timeout_s(self) -> Optional[float]:
        return self._request_timeout_s

    @request_timeout_s.setter
    def request_timeout_s(self, value: Optional[float]) -> None:
        self._request_timeout_s = value

    def get_last_usage(self) -> Optional[Dict[str, Any]]:
        if self._last_usage is None:
            return None
        return copy.deepcopy(self._last_usage)

    def get_usage_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        history = list(self._usage_history)
        if limit is not None:
            try:
                n = int(limit)
            except Exception:
                n = len(history)
            if n <= 0:
                return []
            history = history[-n:]
        return [copy.deepcopy(item) for item in history]

    def get_usage_summary(self) -> Dict[str, Any]:
        return copy.deepcopy(self._usage_totals)

    def reset_usage_tracking(self) -> None:
        self._last_usage = None
        self._usage_history.clear()
        self._usage_totals = init_usage_totals()

    # ---------------------------------------------------------------------
    # Event subscription

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Register a listener callback for agent events.

        The callback is invoked synchronously on the event loop
        thread whenever an event is emitted.  The return value is a
        function that, when called, removes the listener.
        """
        self._listeners.add(fn)
        def _unsubscribe() -> None:
            self._listeners.discard(fn)
        return _unsubscribe

    # ---------------------------------------------------------------------
    # State mutators

    def set_system_prompt(self, prompt: str) -> None:
        self._state.system_prompt = prompt

    def set_model(self, model: Dict[str, Any]) -> None:
        self._state.model = model

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._state.thinking_level = level  # type: ignore

    def set_timeout(self, timeout_s: float) -> None:
        self._request_timeout_s = timeout_s

    def set_steering_mode(self, mode: str) -> None:
        if mode not in {"all", "one-at-a-time"}:
            raise ValueError("steering_mode must be 'all' or 'one-at-a-time'")
        self._steering_mode = mode

    def get_steering_mode(self) -> str:
        return self._steering_mode

    def set_follow_up_mode(self, mode: str) -> None:
        if mode not in {"all", "one-at-a-time"}:
            raise ValueError("follow_up_mode must be 'all' or 'one-at-a-time'")
        self._follow_up_mode = mode

    def get_follow_up_mode(self) -> str:
        return self._follow_up_mode

    def set_tools(self, tools: Sequence[AgentTool]) -> None:
        self._state.tools = list(tools)

    def add_tool(self, tool: AgentTool) -> None:
        name = str(getattr(tool, "name", "")).strip()
        tools = [existing for existing in self._state.tools if str(getattr(existing, "name", "")).strip() != name]
        tools.append(tool)
        self._state.tools = tools

    def remove_tool(self, name: str) -> bool:
        target = str(name or "").strip()
        tools = [tool for tool in self._state.tools if str(getattr(tool, "name", "")).strip() != target]
        removed = len(tools) != len(self._state.tools)
        self._state.tools = tools
        return removed

    def list_tools(self) -> List[str]:
        names: List[str] = []
        seen = set()
        for tool in self._state.tools:
            name = str(getattr(tool, "name", "")).strip()
            if not name or name in seen:
                continue
            names.append(name)
            seen.add(name)
        return names

    def schedule_task(
        self,
        prompt: str,
        *,
        run_at: Optional[str] = None,
        cron: Optional[str] = None,
        timezone: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._scheduled_task_store.schedule_task(
            prompt,
            run_at=run_at,
            cron=cron,
            timezone_name=timezone,
            task_name=task_name,
        )

    def list_scheduled_tasks(self, *, include_history: bool = False) -> List[Dict[str, Any]]:
        return self._scheduled_task_store.list_tasks(include_history=include_history)

    def remove_scheduled_task(self, task_id: str) -> bool:
        return self._scheduled_task_store.remove_task(task_id)

    def run_scheduled_task_now(self, task_id: str) -> bool:
        return self._scheduled_task_store.mark_task_due_now(task_id)

    async def run_due_tasks(
        self,
        executor: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]] | Dict[str, Any] | Any],
        *,
        max_parallel: int = 3,
    ) -> Dict[str, Any]:
        return await self._scheduled_task_store.run_due_tasks(executor, max_parallel=max_parallel)

    def replace_messages(self, messages: Sequence[AgentMessage]) -> None:
        self._state.messages = list(messages)

    def append_message(self, message: AgentMessage) -> None:
        self._state.messages.append(message)

    # ---------------------------------------------------------------------
    # Queue management

    def steer(self, message: AgentMessage) -> None:
        """Queue a steering message to be delivered mid turn."""
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a follow up message to be delivered after completion."""
        self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue = []

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue = []

    def clear_all_queues(self) -> None:
        self._steering_queue = []
        self._follow_up_queue = []

    def clear_messages(self) -> None:
        self._state.messages = []

    # ---------------------------------------------------------------------
    # Control operations

    def abort(self) -> None:
        """Abort the currently running LLM call and any pending tools."""
        if self._abort_event is not None:
            self._abort_event.set()

    async def wait_for_idle(self) -> None:
        """Block until the agent has finished processing the current turn."""
        await self._idle_event.wait()

    def reset(self) -> None:
        """Reset the conversation state and internal queues."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self._steering_queue = []
        self._follow_up_queue = []

    # ---------------------------------------------------------------------
    # High level API

    async def prompt(
        self,
        input: Union[str, AgentMessage, Sequence[AgentMessage]],
        images: Optional[Sequence[ImageContent]] = None,
    ) -> None:
        """Send a prompt to the agent.

        The input may be a plain string (optionally accompanied by a
        list of images), a single :class:`AgentMessage` or an
        iterable of :class:`AgentMessage` objects.  When a string is
        provided it is wrapped into a user message with the current
        timestamp.
        """
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, or wait for completion."
            )
        # Determine prompt messages
        msgs: List[AgentMessage] = []
        if isinstance(input, AgentMessage):
            msgs = [input]
        elif isinstance(input, str):
            contents: List[Any] = [TextContent(type="text", text=input)]
            if images:
                contents.extend(images)
            msgs = [
                AgentMessage(
                    role="user",
                    content=contents,  # type: ignore
                    timestamp=time.time(),
                )
            ]
        else:
            # Iterable of AgentMessage
            try:
                msgs = [m for m in input]  # type: ignore
            except Exception:
                raise TypeError("prompt input must be a string, AgentMessage or iterable of AgentMessage")
        await self._run_loop(msgs)

    async def continue_run(self) -> None:
        """Continue the agent loop from the last non assistant message."""
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing. Wait for completion before continuing."
            )
        if not self._state.messages:
            raise RuntimeError("No messages to continue from")
        if self._state.messages[-1].role == "assistant":
            raise RuntimeError("Cannot continue from message role: assistant")
        await self._run_loop(None)

    # Alias 'continue' for backwards compatibility (cannot use reserved keyword)
    async def continue_(self) -> None:
        await self.continue_run()

    # ---------------------------------------------------------------------
    # Internal execution

    async def _run_loop(self, messages: Optional[Sequence[AgentMessage]]) -> None:
        """Execute a single turn of the agent loop.

        This method builds the context and configuration objects,
        schedules the low level loop, consumes events and updates the
        internal state accordingly.  It also handles exceptions from
        the loop and ensures that the agent returns to an idle
        state.
        """
        # Prepare the abort event and idle flag
        self._idle_event.clear()
        self._abort_event = asyncio.Event()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None
        # Build a copy of the current context
        tools_snapshot = self._snapshot_tools()
        self._state.tools = list(tools_snapshot)
        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(tools_snapshot),
        )
        # Assemble configuration for the loop
        loop_config = AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=self._get_steering_messages,
            get_follow_up_messages=self._get_follow_up_messages,
            get_tools=self._snapshot_tools,
            reasoning=None if self._state.thinking_level == "off" else self._state.thinking_level,
            session_id=self._session_id,
            thinking_budgets=self._thinking_budgets,
            max_retry_delay_ms=self._max_retry_delay_ms,
            request_timeout_s=self._request_timeout_s,
            api_key=None,
            other_options={},
        )
        # Determine which loop entry point to call
        try:
            if messages:
                stream = agent_loop(messages, context, loop_config, self._abort_event, self.stream_fn)
            else:
                stream = agent_loop_continue(context, loop_config, self._abort_event, self.stream_fn)
            # Consume events
            async for event in stream:
                etype = event.get("type")
                # Update internal state
                if etype == "message_start":
                    self._state.stream_message = event.get("message")
                elif etype == "message_update":
                    self._state.stream_message = event.get("message")
                elif etype == "message_end":
                    self._state.stream_message = None
                    msg = event.get("message")
                    if isinstance(msg, AgentMessage):
                        self._state.messages.append(msg)
                        if msg.role == "assistant" and isinstance(msg.usage, dict):
                            self._record_usage(msg.usage)
                elif etype == "tool_execution_start":
                    call_id = event.get("toolCallId")
                    if call_id:
                        self._state.pending_tool_calls.add(str(call_id))
                elif etype == "tool_execution_end":
                    call_id = event.get("toolCallId")
                    if call_id:
                        self._state.pending_tool_calls.discard(str(call_id))
                elif etype == "turn_end":
                    msg = event.get("message")
                    if isinstance(msg, AgentMessage) and msg.role == "assistant" and msg.error_message:
                        self._state.error = msg.error_message
                elif etype == "agent_end":
                    self._state.is_streaming = False
                    self._state.stream_message = None
                # Emit to subscribers
                self._emit(event)
            # Handle any leftover partial message
            # (not required: partial messages are converted to full messages in loop)
        except Exception as exc:
            # Create an assistant error message
            model = self._state.model
            error_msg = AgentMessage(
                role="assistant",
                content=[TextContent(type="text", text="")],
                timestamp=time.time(),
                api=model.get("api") if isinstance(model, dict) else None,
                provider=model.get("provider") if isinstance(model, dict) else None,
                model=model.get("id") if isinstance(model, dict) else None,
                usage=ensure_usage_record(
                    usage={},
                    messages=[],
                    reply_text="",
                    provider=str(model.get("provider") if isinstance(model, dict) else ""),
                    model=str(model.get("id") if isinstance(model, dict) else ""),
                    latency_ms=0,
                    timestamp=time.time(),
                ),
                stop_reason="aborted" if self._abort_event and self._abort_event.is_set() else "error",
                error_message=str(exc),
            )
            self._state.messages.append(error_msg)
            if isinstance(error_msg.usage, dict):
                self._record_usage(error_msg.usage)
            self._state.error = str(exc)
            self._emit({"type": "agent_end", "messages": [error_msg]})
        finally:
            # Reset streaming flags and clear the abort signal
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._abort_event = None
            self._idle_event.set()

    # ---------------------------------------------------------------------
    # Internal helper methods

    def _emit(self, event: AgentEvent) -> None:
        """Dispatch an event to all registered listeners."""
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                # Swallow exceptions from listeners to avoid breaking the agent
                pass

    def _snapshot_tools(self) -> List[AgentTool]:
        return list(self._state.tools)

    def _record_usage(self, usage: Dict[str, Any]) -> None:
        record = copy.deepcopy(dict(usage or {}))
        self._last_usage = record
        self._usage_history.append(record)
        self._usage_totals = accumulate_totals(self._usage_totals, record)

    async def _get_steering_messages(self) -> List[AgentMessage]:
        """Retrieve queued steering messages according to the configured mode."""
        if not self._steering_queue:
            return []
        if self._steering_mode == "one-at-a-time":
            msg = self._steering_queue.pop(0)
            return [msg]
        else:
            msgs = list(self._steering_queue)
            self._steering_queue = []
            return msgs

    async def _get_follow_up_messages(self) -> List[AgentMessage]:
        """Retrieve queued follow up messages according to the configured mode."""
        if not self._follow_up_queue:
            return []
        if self._follow_up_mode == "one-at-a-time":
            msg = self._follow_up_queue.pop(0)
            return [msg]
        else:
            msgs = list(self._follow_up_queue)
            self._follow_up_queue = []
            return msgs
