from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

from ..env_utils import parse_bool_env
from ..message_utils import extract_bash_exec_command
from ..runtime import (
    RuntimeOverrides,
    build_runtime_overrides_from_session,
    create_runtime_session,
    run_due_scheduled_tasks,
    run_user_turn,
    shutdown_runtime_session,
    switch_session,
)
from ..scheduled_runner import get_daemon_status, start_daemon
from ..tui_runtime import format_activity_event
from .schemas import (
    ActivityEvent,
    ApprovalAction,
    ApprovalCardProps,
    ApprovalCardSpec,
    ApprovalDecisionRequest,
    SchedulerStatusProps,
    SchedulerStatusSpec,
    SchedulerTaskSummary,
    SessionOption,
    SessionSnapshot,
    SessionState,
    SessionSwitcherProps,
    SessionSwitcherSpec,
    ToolProgressItem,
    ToolProgressListProps,
    ToolProgressListSpec,
    TranscriptMessage,
)
from .ui_extraction import extract_structured_ui


DEFAULT_HISTORY_PATH = os.path.join("agent", "mem", "history.jsonl")


def _now() -> float:
    return time.time()


def _normalize_session_id(value: str | None) -> str:
    text = str(value or "").strip()
    return text or "default"


def _truncate(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_args_preview(args: Any) -> str | None:
    if args in (None, "", {}):
        return None
    if isinstance(args, dict):
        preview = ", ".join(f"{key}={json.dumps(value, default=str)}" for key, value in sorted(args.items()))
    else:
        preview = str(args)
    preview = " ".join(preview.split())
    return _truncate(preview, limit=180) if preview else None


def _tool_status_from_event(event: dict[str, Any]) -> str:
    etype = str(event.get("type", "")).strip()
    if etype == "tool_policy_blocked":
        return "blocked"
    if etype == "tool_execution_end":
        return "error" if bool(event.get("isError")) else "success"
    return "running"


def _activity_status_from_event(event: dict[str, Any]) -> str:
    etype = str(event.get("type", "")).strip()
    if etype == "tool_policy_blocked":
        return "blocked"
    if etype == "tool_execution_end":
        return "error" if bool(event.get("isError")) else "success"
    if etype in {"tool_execution_start", "message_update"}:
        return "running"
    if etype == "message_end":
        message = event.get("message")
        if getattr(message, "error_message", None):
            return "error"
    return "info"


def _sse_payload(data: str, *, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


@dataclass
class PendingApproval:
    id: str
    payload: dict[str, Any]
    future: asyncio.Future[bool]
    created_at: float = field(default_factory=_now)


class ManagedRuntimeSession:
    def __init__(self, service: "WebRuntimeService", session_id: str, runtime_session: Any) -> None:
        self.service = service
        self.session_id = session_id
        self.runtime = runtime_session
        self.turn_lock = asyncio.Lock()
        self.turn_active = False
        self.last_error: str | None = None
        self.last_message: str = ""
        self.last_ui_spec: Any | None = None
        self.updated_at = _now()
        self.revision = 0
        self.transcript: list[TranscriptMessage] = []
        self.activity_events: Deque[ActivityEvent] = deque(maxlen=40)
        self.pending_tools: dict[str, ToolProgressItem] = {}
        self.recent_tools: Deque[ToolProgressItem] = deque(maxlen=10)
        self.current_approval: PendingApproval | None = None
        self.stream_subscribers: set[asyncio.Queue[str]] = set()
        self.unsubscribe_events = self.runtime.agent.subscribe(self._on_agent_event)

    @property
    def status(self) -> str:
        if self.current_approval is not None:
            return "awaiting_approval"
        if self.turn_active:
            return "running"
        if self.last_error:
            return "error"
        return "idle"

    def _touch(self) -> None:
        self.updated_at = _now()
        self.revision += 1
        self.service._publish_snapshot(self.session_id)

    def add_stream_subscriber(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
        self.stream_subscribers.add(queue)
        queue.put_nowait(self.build_snapshot().model_dump_json(exclude_none=True))
        return queue

    def remove_stream_subscriber(self, queue: asyncio.Queue[str]) -> None:
        self.stream_subscribers.discard(queue)

    def _push_stream_update(self, payload: str) -> None:
        for queue in list(self.stream_subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)

    def _record_activity(self, event_type: str, label: str, *, status: str = "info", detail: str | None = None) -> None:
        self.activity_events.append(
            ActivityEvent(
                id=uuid.uuid4().hex,
                type=event_type,
                label=_truncate(label, limit=300),
                status=status,
                detail=_truncate(detail, limit=300) if detail else None,
            )
        )
        self._touch()

    def _current_tool_items(self) -> list[ToolProgressItem]:
        running = sorted(self.pending_tools.values(), key=lambda item: item.updated_at, reverse=True)
        completed = sorted(self.recent_tools, key=lambda item: item.updated_at, reverse=True)
        seen = {item.id for item in running}
        merged = list(running)
        for item in completed:
            if item.id in seen:
                continue
            merged.append(item)
            seen.add(item.id)
        return merged[:8]

    def _build_approval_spec(self) -> ApprovalCardSpec | None:
        current = self.current_approval
        if current is None:
            return None
        payload = current.payload
        return ApprovalCardSpec(
            props=ApprovalCardProps(
                approval_id=current.id,
                title="bash_exec approval required",
                command=str(payload.get("command") or "").strip() or "(empty)",
                cwd=str(payload.get("cwd") or "").strip() or self.runtime.workspace_root or ".",
                risk=str(payload.get("risk") or "unknown").strip() or "unknown",
                justification=str(payload.get("justification") or "").strip() or None,
                actions=[
                    ApprovalAction(label="Approve", value="approve", variant="primary"),
                    ApprovalAction(label="Reject", value="reject", variant="danger"),
                ],
            )
        )

    def _build_tool_progress_spec(self) -> ToolProgressListSpec | None:
        items = self._current_tool_items()
        if not items:
            return None
        return ToolProgressListSpec(props=ToolProgressListProps(items=items))

    def build_snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session=SessionState(
                id=self.session_id,
                active_session_id=str(self.runtime.active_session_id or self.session_id),
                status=self.status,
                turn_active=self.turn_active,
                error=self.last_error,
            ),
            message=self.last_message,
            ui_spec=self.last_ui_spec,
            approval=self._build_approval_spec(),
            tool_progress=self._build_tool_progress_spec(),
            session_switcher=self.service.build_session_switcher(self.session_id),
            scheduler_status=self.service.build_scheduler_status(),
            events=list(self.activity_events),
            transcript=list(self.transcript),
            updated_at=self.updated_at,
            revision=self.revision,
        )

    def _upsert_recent_tool(self, item: ToolProgressItem) -> None:
        remaining = [existing for existing in self.recent_tools if existing.id != item.id]
        remaining.insert(0, item)
        self.recent_tools = deque(remaining[: self.recent_tools.maxlen], maxlen=self.recent_tools.maxlen)

    def _on_agent_event(self, event: dict[str, Any]) -> None:
        etype = str(event.get("type", "")).strip()
        if etype in {"tool_execution_start", "tool_execution_end", "tool_policy_blocked"}:
            tool_call_id = str(event.get("toolCallId") or uuid.uuid4().hex)
            tool_name = str(event.get("toolName") or "unknown").strip() or "unknown"
            item = ToolProgressItem(
                id=tool_call_id,
                tool_name=tool_name,
                status=_tool_status_from_event(event),
                args_preview=_format_args_preview(event.get("args")),
                command=extract_bash_exec_command(event) or None,
                detail=_truncate(str(event.get("error") or ""), limit=200) or None,
                updated_at=_now(),
            )
            if item.status == "running":
                self.pending_tools[item.id] = item
            else:
                self.pending_tools.pop(item.id, None)
                self._upsert_recent_tool(item)

        activity_text = format_activity_event(event, level="full")
        if activity_text:
            self.activity_events.append(
                ActivityEvent(
                    id=uuid.uuid4().hex,
                    type=etype or "runtime_event",
                    label=activity_text,
                    status=_activity_status_from_event(event),
                )
            )
            self._touch()

        if etype == "message_end":
            message = event.get("message")
            if getattr(message, "role", "") == "assistant":
                error_message = getattr(message, "error_message", None)
                self.last_error = str(error_message).strip() if error_message else None
                self._touch()

    def on_warning(self, text: str) -> None:
        self._record_activity("warning", text, status="info")

    async def request_bash_approval(self, request: dict[str, Any]) -> bool:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self.current_approval = PendingApproval(id=uuid.uuid4().hex, payload=dict(request), future=future)
        self._record_activity(
            "approval_requested",
            f"[approval] bash_exec requested risk={request.get('risk')} cmd={request.get('command')}",
            status="blocked",
        )
        try:
            decision = await future
            return bool(decision)
        finally:
            self.current_approval = None
            self._touch()

    async def resolve_approval(self, payload: ApprovalDecisionRequest) -> bool:
        current = self.current_approval
        if current is None:
            raise KeyError("no approval is currently pending")
        if not current.future.done():
            current.future.set_result(bool(payload.approved))
        self._record_activity(
            "approval_resolved",
            f"[approval] bash_exec {'approved' if payload.approved else 'rejected'}",
            status="success" if payload.approved else "blocked",
        )
        return bool(payload.approved)

    async def run_turn(self, message: str) -> SessionSnapshot:
        raw_message = str(message or "").strip()
        if not raw_message:
            raise ValueError("message must not be empty")

        async with self.turn_lock:
            if self.turn_active:
                raise RuntimeError("a turn is already running for this session")

            self.turn_active = True
            self.last_error = None
            self.last_message = ""
            self.last_ui_spec = None
            self.transcript.append(
                TranscriptMessage(
                    id=uuid.uuid4().hex,
                    role="user",
                    content=raw_message,
                )
            )
            self._record_activity("turn_started", f"[turn] user: {raw_message}", status="running")

            try:
                reply = await run_user_turn(self.runtime, raw_message, on_warning=self.on_warning)
                self.last_message = reply
                self.last_ui_spec = extract_structured_ui(reply)
                self.transcript.append(
                    TranscriptMessage(
                        id=uuid.uuid4().hex,
                        role="assistant",
                        content=reply,
                    )
                )
                return self.build_snapshot()
            except Exception as exc:
                self.last_error = str(exc)
                self.last_message = f"(error) {exc}"
                self.transcript.append(
                    TranscriptMessage(
                        id=uuid.uuid4().hex,
                        role="assistant",
                        content=self.last_message,
                    )
                )
                self._record_activity("turn_error", self.last_message, status="error")
                raise
            finally:
                self.turn_active = False
                self._touch()

    async def shutdown(self) -> None:
        self.unsubscribe_events()
        await shutdown_runtime_session(self.runtime)


class WebRuntimeService:
    def __init__(self) -> None:
        self.sessions: dict[str, ManagedRuntimeSession] = {}
        self.registry_lock = asyncio.Lock()
        self.scheduler_lock = asyncio.Lock()
        self.known_sessions: list[str] = []
        self.scheduler_persistent_enabled = parse_bool_env("POP_AGENT_SCHEDULER_PERSISTENT", True)
        self.scheduler_poll_interval_s = float(os.getenv("POP_AGENT_SCHEDULER_POLL_INTERVAL_S", "1.0") or "1.0")
        self.scheduler_max_parallel = int(os.getenv("POP_AGENT_SCHEDULER_MAX_PARALLEL", "1") or "1")
        self.scheduler_info: dict[str, Any] = {}
        self.scheduler_last_report: dict[str, Any] | None = None
        self._load_known_sessions_from_history()

    def _remember_session(self, session_id: str) -> None:
        normalized = _normalize_session_id(session_id)
        if normalized.startswith("scheduled:"):
            return
        if normalized in self.known_sessions:
            self.known_sessions = [sid for sid in self.known_sessions if sid != normalized]
        self.known_sessions.insert(0, normalized)

    def _load_known_sessions_from_history(self) -> None:
        path = os.path.realpath(DEFAULT_HISTORY_PATH)
        if not os.path.exists(path):
            return
        discovered: list[str] = []
        with contextlib.suppress(Exception):
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    text = raw_line.strip()
                    if not text or not text.startswith("{"):
                        continue
                    payload = json.loads(text)
                    session_id = _normalize_session_id(payload.get("session_id"))
                    if session_id.startswith("scheduled:") or session_id in discovered:
                        continue
                    discovered.append(session_id)
        self.known_sessions = discovered + [sid for sid in self.known_sessions if sid not in discovered]

    async def initialize(self) -> None:
        if self.scheduler_persistent_enabled:
            with contextlib.suppress(Exception):
                await self.ensure_scheduler_daemon()
        else:
            self.scheduler_info = {
                "ok": True,
                "running": False,
                "persistent_enabled": False,
            }

    async def get_or_create_session(self, session_id: str) -> ManagedRuntimeSession:
        normalized = _normalize_session_id(session_id)
        existing = self.sessions.get(normalized)
        if existing is not None:
            return existing

        async with self.registry_lock:
            existing = self.sessions.get(normalized)
            if existing is not None:
                return existing

            managed_ref: dict[str, ManagedRuntimeSession] = {}

            async def _approval(request: dict[str, Any]) -> bool:
                managed = managed_ref.get("session")
                if managed is None:
                    return False
                return await managed.request_bash_approval(request)

            async def _run_due() -> dict[str, Any]:
                return await self.run_due_tasks_now(normalized)

            runtime_session = create_runtime_session(
                enable_event_logger=False,
                bash_approval_fn=_approval,
                run_scheduled_tasks_now_fn=_run_due,
                ensure_scheduler_daemon_fn=self.ensure_scheduler_daemon,
                overrides=RuntimeOverrides(enable_auto_title=False),
            )
            switch_session(runtime_session, normalized)
            managed = ManagedRuntimeSession(self, normalized, runtime_session)
            managed_ref["session"] = managed
            self.sessions[normalized] = managed
            self._remember_session(normalized)
            self._publish_snapshot(normalized)
            return managed

    def build_session_switcher(self, current_session_id: str) -> SessionSwitcherSpec:
        options = [
            SessionOption(id=session_id, label=session_id, active=session_id == current_session_id)
            for session_id in self.known_sessions
        ]
        if not any(option.id == current_session_id for option in options):
            options.insert(0, SessionOption(id=current_session_id, label=current_session_id, active=True))
        return SessionSwitcherSpec(
            props=SessionSwitcherProps(
                current_session_id=current_session_id,
                sessions=options[:16],
            )
        )

    def build_scheduler_status(self) -> SchedulerStatusSpec:
        if not self.scheduler_persistent_enabled:
            return SchedulerStatusSpec(
                props=SchedulerStatusProps(
                    mode="disabled",
                    state="disabled",
                    message="Persistent scheduler is disabled.",
                    details={"persistent_enabled": False},
                )
            )

        info = dict(self.scheduler_info)
        running = bool(info.get("running") or info.get("ok") and info.get("already_running"))
        state = "running" if running else "warning"
        message = "Persistent scheduler daemon is running." if running else "Persistent scheduler daemon is unavailable."
        recent_tasks = []
        if isinstance(self.scheduler_last_report, dict):
            for task in list(self.scheduler_last_report.get("tasks", []) or [])[:5]:
                recent_tasks.append(
                    SchedulerTaskSummary(
                        id=str(task.get("id") or ""),
                        status=str(task.get("status") or ""),
                        summary=_truncate(task.get("summary") or "", limit=120),
                    )
                )

        return SchedulerStatusSpec(
            props=SchedulerStatusProps(
                mode="persistent",
                state=state,
                message=message,
                due_count=int((self.scheduler_last_report or {}).get("due_count", 0) or 0),
                success_count=int((self.scheduler_last_report or {}).get("success_count", 0) or 0),
                error_count=int((self.scheduler_last_report or {}).get("error_count", 0) or 0),
                recent_tasks=recent_tasks,
                details=info,
            )
        )

    def list_known_sessions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for session_id in self.known_sessions:
            live_session = self.sessions.get(session_id)
            rows.append(
                {
                    "id": session_id,
                    "live": live_session is not None,
                    "status": live_session.status if live_session is not None else "idle",
                    "updated_at": live_session.updated_at if live_session is not None else None,
                }
            )
        return rows

    async def get_snapshot(self, session_id: str) -> SessionSnapshot:
        session = await self.get_or_create_session(session_id)
        return session.build_snapshot()

    def _publish_snapshot(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        snapshot_json = session.build_snapshot().model_dump_json(exclude_none=True)
        session._push_stream_update(snapshot_json)

    async def run_due_tasks_now(self, session_id: str) -> dict[str, Any]:
        session = await self.get_or_create_session(session_id)
        report = await run_due_scheduled_tasks(
            session.runtime.agent,
            max_parallel=self.scheduler_max_parallel,
            bash_approval_fn=session.request_bash_approval if session.runtime.bash_prompt_approval else None,
            overrides=build_runtime_overrides_from_session(session.runtime),
            on_warning=session.on_warning,
        )
        self.scheduler_last_report = report
        self._publish_all()
        return report

    async def ensure_scheduler_daemon(self) -> dict[str, Any]:
        if not self.scheduler_persistent_enabled:
            self.scheduler_info = {
                "ok": True,
                "running": False,
                "persistent_enabled": False,
            }
            return dict(self.scheduler_info)

        async with self.scheduler_lock:
            info = await asyncio.to_thread(
                start_daemon,
                max_parallel=self.scheduler_max_parallel,
                poll_interval_s=self.scheduler_poll_interval_s,
                exit_when_idle=True,
                python_executable=sys.executable,
            )
            status = await asyncio.to_thread(get_daemon_status)
            self.scheduler_info = {**status, **info, "persistent_enabled": True}
            self._publish_all()
            return dict(self.scheduler_info)

    async def resolve_approval(self, session_id: str, payload: ApprovalDecisionRequest) -> bool:
        session = await self.get_or_create_session(session_id)
        return await session.resolve_approval(payload)

    def _publish_all(self) -> None:
        for session_id in list(self.sessions):
            self._publish_snapshot(session_id)

    async def shutdown_session(self, session_id: str) -> None:
        normalized = _normalize_session_id(session_id)
        session = self.sessions.pop(normalized, None)
        if session is None:
            return
        await session.shutdown()

    async def shutdown(self) -> None:
        for session_id in list(self.sessions):
            with contextlib.suppress(Exception):
                await self.shutdown_session(session_id)
