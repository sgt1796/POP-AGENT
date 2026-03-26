"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";

export function CopilotChatPanel({ runtimeUrl, sessionId }: { runtimeUrl: string; sessionId: string }) {
  return (
    <CopilotKit key={sessionId} runtimeUrl={runtimeUrl} agent="default" showDevConsole={false}>
      <CopilotChat />
    </CopilotKit>
  );
}
