import { HttpAgent } from "@ag-ui/client";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";
import { getBackendBaseUrl } from "@/lib/backend-base-url";

async function handle(request: NextRequest) {
  const sessionId = request.nextUrl.searchParams.get("session_id") || "web-default";
  const agent = new HttpAgent({
    url: `${getBackendBaseUrl()}/api/agui/default?session_id=${encodeURIComponent(sessionId)}`,
  });
  const agents = {
    // The app currently installs a direct @ag-ui/client version that lags the one bundled by
    // CopilotKit. Runtime behavior is already working, but the constructor types diverge.
    default: agent as never,
  };

  const runtime = new CopilotRuntime({
    agents,
  });

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });

  return handleRequest(request);
}

export async function GET(request: NextRequest) {
  return handle(request);
}

export async function POST(request: NextRequest) {
  return handle(request);
}
