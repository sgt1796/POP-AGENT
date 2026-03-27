"use client";

import { AssistantMessage as CopilotAssistantMessage } from "@copilotkit/react-ui";
import type { AssistantMessageProps } from "@copilotkit/react-ui";
import { GeneratedUiMessageAttachment } from "@/components/generated-ui-chat";

export function GeneratedUiAssistantMessage(props: AssistantMessageProps) {
  return (
    <div className="generated-ui-message">
      <CopilotAssistantMessage {...props} />
      <GeneratedUiMessageAttachment messageId={props.message?.id} messages={props.messages} />
    </div>
  );
}
