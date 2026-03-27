"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";

import {
  buildWorkspaceZoneMetadata,
  GeneratedUiDockProvider,
  GeneratedUiWorkspaceColumn,
  type GeneratedUiEntry,
  type WorkspaceColumnSpec,
} from "@/components/generated-ui-chat";
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
  const [uiEntriesBySession, setUiEntriesBySession] = useState<
    Record<string, GeneratedUiEntry[]>
  >({});
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
  const uiEntries = useMemo(
    () => uiEntriesBySession[selectedSessionId] || [],
    [uiEntriesBySession, selectedSessionId]
  );
  const latestAssistantTranscriptMessage = useMemo(
    () =>
      [...(snapshot?.transcript || [])]
        .reverse()
        .find((message) => message.role === "assistant") || null,
    [snapshot?.transcript]
  );

  useEffect(() => {
    const uiSpec = snapshot?.ui_spec;
    const assistantMessage = latestAssistantTranscriptMessage;
    if (!uiSpec || !assistantMessage) {
      return;
    }
    const entryId = assistantMessage.id;
    const sourceText = assistantMessage.content;
    const serializedSpec = JSON.stringify(uiSpec);

    setUiEntriesBySession((current) => {
      const sessionEntries = current[selectedSessionId] || [];
      const entryIndex = sessionEntries.findIndex((entry) => entry.id === entryId);
      if (entryIndex >= 0) {
        const existingEntry = sessionEntries[entryIndex];
        if (
          existingEntry.sourceText === sourceText &&
          JSON.stringify(existingEntry.spec) === serializedSpec
        ) {
          return current;
        }
        const nextEntries = [...sessionEntries];
        nextEntries[entryIndex] = {
          id: entryId,
          sourceText,
          spec: uiSpec,
        };
        return {
          ...current,
          [selectedSessionId]: nextEntries,
        };
      }

      const nextEntry: GeneratedUiEntry = {
        id: entryId,
        sourceText,
        spec: uiSpec,
      };
      return {
        ...current,
        [selectedSessionId]: [...sessionEntries, nextEntry],
      };
    });
  }, [latestAssistantTranscriptMessage, selectedSessionId, snapshot?.ui_spec]);

  useEffect(() => {
    setUiEntriesBySession((current) => {
      const sessionEntries = current[selectedSessionId] || [];
      const validAssistantIds = new Set(
        (snapshot?.transcript || [])
          .filter((message) => message.role === "assistant")
          .map((message) => message.id)
      );
      const nextEntries = sessionEntries.filter((entry) => validAssistantIds.has(entry.id));
      if (nextEntries.length === sessionEntries.length) {
        return current;
      }
      return {
        ...current,
        [selectedSessionId]: nextEntries,
      };
    });
  }, [selectedSessionId, snapshot?.transcript]);

  const workspaceColumns: WorkspaceColumnSpec[] = [
    {
      id: "main",
      title: "main workspace",
      panels: [
        {
          id: "chat",
          title: "Agent Chat",
          content: (
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
          ),
        },
        {
          id: "events",
          title: "Recent Events",
          content: <EventFeed events={snapshot?.events || []} />,
        },
      ],
    },
    {
      id: "side",
      title: "sidebar",
      panels: [
        ...(snapshot?.session_switcher
          ? [
              {
                id: "sessions",
                title: "Sessions",
                content: (
                  <SessionSwitcher
                    spec={snapshot.session_switcher}
                    onSelect={(sessionId) => setSelectedSessionId(sessionId)}
                    onCreate={() => setSelectedSessionId(createSessionId())}
                  />
                ),
              },
            ]
          : []),
        ...(snapshot?.scheduler_status
          ? [
              {
                id: "scheduler",
                title: "Scheduler",
                content: <SchedulerStatus spec={snapshot.scheduler_status} />,
              },
            ]
          : []),
        ...(snapshot?.approval
          ? [
              {
                id: "approval",
                title: "Approval",
                content: <ApprovalCard spec={snapshot.approval} onDecision={approve} />,
              },
            ]
          : []),
        ...(snapshot?.tool_progress
          ? [
              {
                id: "tool-progress",
                title: "Tool Activity",
                content: <ToolProgressList spec={snapshot.tool_progress} />,
              },
            ]
          : []),
      ],
    },
  ];

  const { zoneLabels, zoneOrder } = useMemo(
    () => buildWorkspaceZoneMetadata(workspaceColumns),
    [workspaceColumns]
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

      <GeneratedUiDockProvider
        key={selectedSessionId}
        uiEntries={uiEntries}
        zoneLabels={zoneLabels}
        zoneOrder={zoneOrder}
      >
        <section className="content-grid">
          <div className="main-column">
            <GeneratedUiWorkspaceColumn {...workspaceColumns[0]} />
          </div>

          <aside className="side-column">
            <GeneratedUiWorkspaceColumn {...workspaceColumns[1]} />
          </aside>
        </section>
      </GeneratedUiDockProvider>
    </main>
  );
}
