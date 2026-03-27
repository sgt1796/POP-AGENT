"use client";

import { Fragment, useMemo, useState, type CSSProperties } from "react";

import type {
  ActivityEvent,
  ApprovalCardSpec,
  PlanChecklistSpec,
  ResultTableSpec,
  SchedulerStatusSpec,
  SessionSwitcherSpec,
  StatGridSpec,
  ToolProgressListSpec,
} from "@/lib/types";

function statusClass(status: string) {
  return `status-pill status-${status}`;
}

function cardClassName(className?: string) {
  return className ? `card ${className}` : "card";
}

function statToneClass(tone: StatGridSpec["props"]["items"][number]["tone"]) {
  const status =
    tone === "positive"
      ? "success"
      : tone === "negative"
        ? "danger"
        : tone === "warning"
          ? "warning"
          : "info";
  return statusClass(status);
}

type EventFeedLimit = 5 | 10 | 50 | "all";

const EVENT_FEED_LIMITS: EventFeedLimit[] = [5, 10, 50, "all"];

function InlineMarkdownText({ text }: { text: string }) {
  const parts = text.split(/(\*\*.+?\*\*)/g).filter(Boolean);
  if (parts.length <= 1) {
    return <>{text}</>;
  }
  return (
    <>
      {parts.map((part, index) => {
        const match = /^\*\*(.+?)\*\*$/.exec(part);
        if (!match) {
          return <Fragment key={`${part}-${index}`}>{part}</Fragment>;
        }
        return <strong key={`${match[1]}-${index}`}>{match[1]}</strong>;
      })}
    </>
  );
}

function formatEventTimestamp(timestamp: number) {
  if (!timestamp) {
    return "";
  }
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

export function ApprovalCard({
  spec,
  onDecision,
}: {
  spec: ApprovalCardSpec;
  onDecision: (approved: boolean) => Promise<void>;
}) {
  return (
    <section className="card">
      <div className="panel-head">
        <h3>{spec.props.title}</h3>
        <span className={statusClass("blocked")}>{spec.props.risk}</span>
      </div>
      <pre className="code-block">{spec.props.command}</pre>
      <p className="meta-line">cwd: {spec.props.cwd}</p>
      {spec.props.justification ? <p className="meta-line">why: {spec.props.justification}</p> : null}
      <div className="button-row">
        <button className="button button-primary" onClick={() => onDecision(true)}>
          Approve
        </button>
        <button className="button button-danger" onClick={() => onDecision(false)}>
          Reject
        </button>
      </div>
    </section>
  );
}

export function ToolProgressList({ spec }: { spec: ToolProgressListSpec }) {
  return (
    <section className="card">
      <div className="panel-head">
        <h3>{spec.props.title}</h3>
        <span className="meta-line">{spec.props.items.length} items</span>
      </div>
      <ul className="tool-list">
        {spec.props.items.map((item) => (
          <li key={item.id}>
            <div className="tool-row">
              <span className={statusClass(item.status)}>{item.status}</span>
              <strong>{item.tool_name}</strong>
            </div>
            {item.command ? <div className="meta-line">{item.command}</div> : null}
            {!item.command && item.args_preview ? <div className="meta-line">{item.args_preview}</div> : null}
            {item.detail ? <div className="meta-line">{item.detail}</div> : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

export function ResultTable({ spec, className }: { spec: ResultTableSpec; className?: string }) {
  return (
    <section className={cardClassName(className)}>
      <div className="panel-head">
        <h3>{spec.props.title}</h3>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {spec.props.columns.map((column) => (
                <th key={column}>
                  <InlineMarkdownText text={column} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {spec.props.rows.map((row, rowIndex) => (
              <tr key={`${spec.props.title}-${rowIndex}`}>
                {row.map((cell, columnIndex) => (
                  <td key={`${rowIndex}-${columnIndex}`}>
                    <InlineMarkdownText text={cell} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function StatGrid({ spec, className }: { spec: StatGridSpec; className?: string }) {
  const columnCount = Math.max(1, Math.min(spec.props.columns || 1, 4, spec.props.items.length || 1));
  const gridStyle = {
    "--stat-grid-columns": columnCount,
  } as CSSProperties & { "--stat-grid-columns": number };

  return (
    <section className={cardClassName(className)}>
      <div className="panel-head">
        <h3>{spec.props.title}</h3>
        <span className="meta-line">{spec.props.items.length} metrics</span>
      </div>
      <div className="stat-grid" style={gridStyle}>
        {spec.props.items.map((item, index) => (
          <article className="stat-tile" key={`${item.label}-${index}`}>
            <p className="stat-label">{item.label}</p>
            <strong className="stat-value">{item.value}</strong>
            {item.delta ? <span className={statToneClass(item.tone)}>{item.delta}</span> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

export function PlanChecklist({ spec, className }: { spec: PlanChecklistSpec; className?: string }) {
  return (
    <section className={cardClassName(className)}>
      <div className="panel-head">
        <h3>{spec.props.title}</h3>
      </div>
      <ul className="plan-list">
        {spec.props.items.map((item, index) => (
          <li key={`${item.label}-${index}`}>
            <span className={statusClass(item.status)}>{item.status.replace("_", " ")}</span>
            <span>{item.label}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function SessionSwitcher({
  spec,
  onSelect,
  onCreate,
}: {
  spec: SessionSwitcherSpec;
  onSelect: (sessionId: string) => void;
  onCreate: () => void;
}) {
  return (
    <section className="card">
      <div className="panel-head">
        <h3>{spec.props.title}</h3>
        <button className="button button-ghost" onClick={onCreate}>
          New Session
        </button>
      </div>
      <div className="session-grid">
        {spec.props.sessions.map((session) => (
          <button
            key={session.id}
            className={`session-chip${session.active ? " active" : ""}`}
            onClick={() => onSelect(session.id)}
          >
            {session.label}
          </button>
        ))}
      </div>
    </section>
  );
}

export function SchedulerStatus({ spec }: { spec: SchedulerStatusSpec }) {
  return (
    <section className="card">
      <div className="panel-head">
        <h3>{spec.props.title}</h3>
        <span className={statusClass(spec.props.state)}>{spec.props.mode}</span>
      </div>
      <p className="meta-line">{spec.props.message}</p>
      <div className="metric-row">
        <div>
          <strong>{spec.props.due_count ?? 0}</strong>
          <span>due</span>
        </div>
        <div>
          <strong>{spec.props.success_count ?? 0}</strong>
          <span>success</span>
        </div>
        <div>
          <strong>{spec.props.error_count ?? 0}</strong>
          <span>errors</span>
        </div>
      </div>
      {spec.props.recent_tasks.length > 0 ? (
        <ul className="tool-list compact">
          {spec.props.recent_tasks.map((task) => (
            <li key={task.id}>
              <div className="tool-row">
                <span className={statusClass(task.status)}>{task.status}</span>
                <strong>{task.id}</strong>
              </div>
              <div className="meta-line">{task.summary}</div>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

export function EventFeed({ events }: { events: ActivityEvent[] }) {
  const [limit, setLimit] = useState<EventFeedLimit>(10);
  const orderedEvents = useMemo(() => [...events].reverse(), [events]);
  const visibleEvents = useMemo(() => {
    if (limit === "all") {
      return orderedEvents;
    }
    return orderedEvents.slice(0, limit);
  }, [limit, orderedEvents]);

  return (
    <section className="card">
      <div className="panel-head">
        <div>
          <h3>Recent Events</h3>
          <p className="meta-line">
            Showing {visibleEvents.length} of {orderedEvents.length}
          </p>
        </div>
        <div className="event-feed-controls">
          {EVENT_FEED_LIMITS.map((candidateLimit) => (
            <button
              key={`event-limit-${candidateLimit}`}
              type="button"
              className={`event-feed-button${limit === candidateLimit ? " active" : ""}`}
              onClick={() => setLimit(candidateLimit)}
            >
              {candidateLimit === "all" ? "All" : candidateLimit}
            </button>
          ))}
        </div>
      </div>
      <ul className="event-feed">
        {visibleEvents.length === 0 ? <li className="meta-line">No runtime events yet.</li> : null}
        {visibleEvents.map((event) => (
          <li key={event.id}>
            <details className="event-entry">
              <summary className="event-summary">
                <div className="event-summary-main">
                  <span className={statusClass(event.status)}>{event.type}</span>
                  <span className="event-summary-label">{event.label}</span>
                </div>
                <span className="meta-line">{formatEventTimestamp(event.timestamp)}</span>
              </summary>
              <div className="event-detail">
                <p>{event.label}</p>
                {event.detail ? <p className="meta-line">{event.detail}</p> : null}
              </div>
            </details>
          </li>
        ))}
      </ul>
    </section>
  );
}
