"use client";

import type {
  ActivityEvent,
  ApprovalCardSpec,
  PlanChecklistSpec,
  ResultTableSpec,
  SchedulerStatusSpec,
  SessionSwitcherSpec,
  ToolProgressListSpec,
} from "@/lib/types";

function statusClass(status: string) {
  return `status-pill status-${status}`;
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

export function ResultTable({ spec }: { spec: ResultTableSpec }) {
  return (
    <section className="card">
      <div className="panel-head">
        <h3>{spec.props.title}</h3>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {spec.props.columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {spec.props.rows.map((row, rowIndex) => (
              <tr key={`${spec.props.title}-${rowIndex}`}>
                {row.map((cell, columnIndex) => (
                  <td key={`${rowIndex}-${columnIndex}`}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function PlanChecklist({ spec }: { spec: PlanChecklistSpec }) {
  return (
    <section className="card">
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
  return (
    <section className="card">
      <div className="panel-head">
        <h3>Recent Events</h3>
      </div>
      <ul className="event-feed">
        {events.length === 0 ? <li className="meta-line">No runtime events yet.</li> : null}
        {events.map((event) => (
          <li key={event.id}>
            <span className={statusClass(event.status)}>{event.type}</span>
            <span>{event.label}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
