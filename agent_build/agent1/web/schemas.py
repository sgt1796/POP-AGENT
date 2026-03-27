from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class TranscriptMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: float = Field(default_factory=time.time)


class ActivityEvent(BaseModel):
    id: str
    type: str
    label: str
    status: Literal["info", "running", "success", "error", "blocked"] = "info"
    detail: str | None = None
    timestamp: float = Field(default_factory=time.time)


class ApprovalAction(BaseModel):
    label: str
    value: Literal["approve", "reject"]
    variant: Literal["primary", "danger", "ghost"] = "primary"


class ApprovalCardProps(BaseModel):
    approval_id: str
    title: str
    command: str
    cwd: str
    risk: str
    justification: str | None = None
    actions: list[ApprovalAction] = Field(default_factory=list)


class ApprovalCardSpec(BaseModel):
    type: Literal["ApprovalCard"] = "ApprovalCard"
    props: ApprovalCardProps


class ToolProgressItem(BaseModel):
    id: str
    tool_name: str
    status: Literal["running", "success", "error", "blocked"]
    args_preview: str | None = None
    command: str | None = None
    detail: str | None = None
    updated_at: float = Field(default_factory=time.time)


class ToolProgressListProps(BaseModel):
    title: str = "Tool Activity"
    items: list[ToolProgressItem] = Field(default_factory=list)


class ToolProgressListSpec(BaseModel):
    type: Literal["ToolProgressList"] = "ToolProgressList"
    props: ToolProgressListProps


class ResultTableProps(BaseModel):
    title: str
    columns: list[str]
    rows: list[list[str]]


class ResultTableSpec(BaseModel):
    type: Literal["ResultTable"] = "ResultTable"
    props: ResultTableProps


class StatGridItem(BaseModel):
    label: str
    value: str
    delta: str | None = None
    tone: Literal["positive", "negative", "neutral", "warning"] = "neutral"


class StatGridProps(BaseModel):
    title: str
    columns: int = Field(default=4, ge=1, le=4)
    items: list[StatGridItem]


class StatGridSpec(BaseModel):
    type: Literal["StatGrid"] = "StatGrid"
    props: StatGridProps


class PlanChecklistItem(BaseModel):
    label: str
    status: Literal["pending", "in_progress", "done", "blocked"] = "pending"


class PlanChecklistProps(BaseModel):
    title: str
    items: list[PlanChecklistItem]


class PlanChecklistSpec(BaseModel):
    type: Literal["PlanChecklist"] = "PlanChecklist"
    props: PlanChecklistProps


class SessionOption(BaseModel):
    id: str
    label: str
    active: bool = False


class SessionSwitcherProps(BaseModel):
    title: str = "Sessions"
    current_session_id: str
    sessions: list[SessionOption] = Field(default_factory=list)


class SessionSwitcherSpec(BaseModel):
    type: Literal["SessionSwitcher"] = "SessionSwitcher"
    props: SessionSwitcherProps


class SchedulerTaskSummary(BaseModel):
    id: str
    status: str
    summary: str


class SchedulerStatusProps(BaseModel):
    title: str = "Scheduler"
    mode: Literal["persistent", "manual", "disabled"] = "manual"
    state: Literal["running", "idle", "warning", "disabled"] = "idle"
    message: str
    due_count: int | None = None
    success_count: int | None = None
    error_count: int | None = None
    recent_tasks: list[SchedulerTaskSummary] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SchedulerStatusSpec(BaseModel):
    type: Literal["SchedulerStatus"] = "SchedulerStatus"
    props: SchedulerStatusProps


UIComponent = (
    ApprovalCardSpec
    | ToolProgressListSpec
    | ResultTableSpec
    | StatGridSpec
    | PlanChecklistSpec
    | SessionSwitcherSpec
    | SchedulerStatusSpec
)


class SessionState(BaseModel):
    id: str
    active_session_id: str
    status: Literal["idle", "running", "awaiting_approval", "error"] = "idle"
    turn_active: bool = False
    error: str | None = None


class SessionSnapshot(BaseModel):
    session: SessionState
    message: str = ""
    ui_spec: UIComponent | None = None
    approval: ApprovalCardSpec | None = None
    tool_progress: ToolProgressListSpec | None = None
    session_switcher: SessionSwitcherSpec | None = None
    scheduler_status: SchedulerStatusSpec | None = None
    events: list[ActivityEvent] = Field(default_factory=list)
    transcript: list[TranscriptMessage] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)
    revision: int = 0


class TurnRequest(BaseModel):
    message: str


class ApprovalDecisionRequest(BaseModel):
    approved: bool
