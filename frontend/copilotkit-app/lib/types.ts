export type UIType =
  | "ApprovalCard"
  | "ToolProgressList"
  | "ResultTable"
  | "StatGrid"
  | "PlanChecklist"
  | "SessionSwitcher"
  | "SchedulerStatus";

export type ActivityStatus = "info" | "running" | "success" | "error" | "blocked";

export type ActivityEvent = {
  id: string;
  type: string;
  label: string;
  status: ActivityStatus;
  detail?: string | null;
  timestamp: number;
};

export type TranscriptMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
};

export type ApprovalCardSpec = {
  type: "ApprovalCard";
  props: {
    approval_id: string;
    title: string;
    command: string;
    cwd: string;
    risk: string;
    justification?: string | null;
    actions: Array<{
      label: string;
      value: "approve" | "reject";
      variant: "primary" | "danger" | "ghost";
    }>;
  };
};

export type ToolProgressListSpec = {
  type: "ToolProgressList";
  props: {
    title: string;
    items: Array<{
      id: string;
      tool_name: string;
      status: "running" | "success" | "error" | "blocked";
      args_preview?: string | null;
      command?: string | null;
      detail?: string | null;
      updated_at: number;
    }>;
  };
};

export type ResultTableSpec = {
  type: "ResultTable";
  props: {
    title: string;
    columns: string[];
    rows: string[][];
  };
};

export type StatGridSpec = {
  type: "StatGrid";
  props: {
    title: string;
    columns: number;
    items: Array<{
      label: string;
      value: string;
      delta?: string | null;
      tone: "positive" | "negative" | "neutral" | "warning";
    }>;
  };
};

export type PlanChecklistSpec = {
  type: "PlanChecklist";
  props: {
    title: string;
    items: Array<{
      label: string;
      status: "pending" | "in_progress" | "done" | "blocked";
    }>;
  };
};

export type SessionSwitcherSpec = {
  type: "SessionSwitcher";
  props: {
    title: string;
    current_session_id: string;
    sessions: Array<{
      id: string;
      label: string;
      active: boolean;
    }>;
  };
};

export type SchedulerStatusSpec = {
  type: "SchedulerStatus";
  props: {
    title: string;
    mode: "persistent" | "manual" | "disabled";
    state: "running" | "idle" | "warning" | "disabled";
    message: string;
    due_count?: number | null;
    success_count?: number | null;
    error_count?: number | null;
    recent_tasks: Array<{
      id: string;
      status: string;
      summary: string;
    }>;
    details: Record<string, unknown>;
  };
};

export type UISpec =
  | ApprovalCardSpec
  | ToolProgressListSpec
  | ResultTableSpec
  | StatGridSpec
  | PlanChecklistSpec
  | SessionSwitcherSpec
  | SchedulerStatusSpec;

export type SessionSnapshot = {
  session: {
    id: string;
    active_session_id: string;
    status: "idle" | "running" | "awaiting_approval" | "error";
    turn_active: boolean;
    error?: string | null;
  };
  message: string;
  ui_spec?: UISpec | null;
  approval?: ApprovalCardSpec | null;
  tool_progress?: ToolProgressListSpec | null;
  session_switcher?: SessionSwitcherSpec | null;
  scheduler_status?: SchedulerStatusSpec | null;
  events: ActivityEvent[];
  transcript: TranscriptMessage[];
  updated_at: number;
  revision: number;
};
