"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";

import { PopUiRenderer } from "@/components/pop-ui-renderer";
import {
  ApprovalCard,
  EventFeed,
  SchedulerStatus,
  SessionSwitcher,
  ToolProgressList,
} from "@/components/ui-cards";
import { useSessionStream } from "@/lib/use-session-stream";

const CopilotChatPanel = dynamic(
  () => import("@/components/copilot-chat-panel").then((module) => module.CopilotChatPanel),
  {
    ssr: false,
    loading: () => (
      <div className="chat-loading">
        <strong>Loading CopilotKit…</strong>
        <span>The chat surface is client-only because CopilotKit depends on browser APIs.</span>
      </div>
    ),
  }
);

function createSessionId() {
  return `web-${new Date().toISOString().replace(/[:.]/g, "-")}`;
}

export function ChatShell() {
  const [selectedSessionId, setSelectedSessionId] = useState("web-default");
  const { snapshot, connectionState, approve } = useSessionStream(selectedSessionId);

  useEffect(() => {
    const stored = window.localStorage.getItem("pop-agent-session-id");
    if (stored) {
      setSelectedSessionId(stored);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem("pop-agent-session-id", selectedSessionId);
  }, [selectedSessionId]);

  const runtimeUrl = useMemo(
    () => `/api/copilotkit?session_id=${encodeURIComponent(selectedSessionId)}`,
    [selectedSessionId]
  );

  return (
    <main className="page-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">POP Agent + CopilotKit</p>
          <h1>CopilotKit chat on top of the existing Python runtime.</h1>
        </div>
        <div className="hero-status">
          <span className={`status-pill status-${snapshot?.session.status || "idle"}`}>
            {snapshot?.session.status || "idle"}
          </span>
          <span className="meta-line">stream: {connectionState}</span>
        </div>
      </section>

      <section className="content-grid">
        <div className="main-column">
          <PopUiRenderer spec={snapshot?.ui_spec} />

          <section className="card chat-card">
            <div className="panel-head">
              <div>
                <h3>Agent Chat</h3>
                <p className="meta-line">Session: {selectedSessionId}</p>
              </div>
            </div>
            <div className="chat-frame">
              <CopilotChatPanel runtimeUrl={runtimeUrl} sessionId={selectedSessionId} />
            </div>
          </section>

          <EventFeed events={snapshot?.events || []} />
        </div>

        <aside className="side-column">
          {snapshot?.session_switcher ? (
            <SessionSwitcher
              spec={snapshot.session_switcher}
              onSelect={(sessionId) => setSelectedSessionId(sessionId)}
              onCreate={() => setSelectedSessionId(createSessionId())}
            />
          ) : null}

          {snapshot?.scheduler_status ? <SchedulerStatus spec={snapshot.scheduler_status} /> : null}

          {snapshot?.approval ? <ApprovalCard spec={snapshot.approval} onDecision={approve} /> : null}

          {snapshot?.tool_progress ? <ToolProgressList spec={snapshot.tool_progress} /> : null}
        </aside>
      </section>
    </main>
  );
}
