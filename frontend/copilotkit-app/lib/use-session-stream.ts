"use client";

import { useEffect, useMemo, useState } from "react";

import type { SessionSnapshot } from "@/lib/types";

function buildStreamUrl(sessionId: string) {
  return `/api/pop-agent/sessions/${encodeURIComponent(sessionId)}/stream`;
}

function buildApprovalUrl(sessionId: string) {
  return `/api/pop-agent/sessions/${encodeURIComponent(sessionId)}/approval`;
}

export function useSessionStream(sessionId: string) {
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [connectionState, setConnectionState] = useState<"connecting" | "open" | "closed">("connecting");

  useEffect(() => {
    setConnectionState("connecting");
    const source = new EventSource(buildStreamUrl(sessionId));

    const handleSnapshot = (event: MessageEvent<string>) => {
      try {
        setSnapshot(JSON.parse(event.data) as SessionSnapshot);
        setConnectionState("open");
      } catch (error) {
        console.error("Unable to parse session snapshot", error);
      }
    };

    source.addEventListener("snapshot", handleSnapshot as EventListener);
    source.onerror = () => {
      setConnectionState("closed");
    };

    return () => {
      source.removeEventListener("snapshot", handleSnapshot as EventListener);
      source.close();
    };
  }, [sessionId]);

  const approve = useMemo(
    () => async (approved: boolean) => {
      await fetch(buildApprovalUrl(sessionId), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ approved }),
      });
    },
    [sessionId]
  );

  return {
    snapshot,
    connectionState,
    approve,
  };
}
