"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";

import { GeneratedUiAssistantMessage } from "@/components/generated-ui-assistant-message";

export function CopilotChatPanel({ runtimeUrl, sessionId }: { runtimeUrl: string; sessionId: string }) {
  return (
    <div className="chat-panel-shell">
      <CopilotKit key={sessionId} runtimeUrl={runtimeUrl} agent="default" showDevConsole={false}>
        <CopilotChat className="pop-copilot-chat" AssistantMessage={GeneratedUiAssistantMessage} />
      </CopilotKit>
    </div>
  );
}
