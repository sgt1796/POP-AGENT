import { NextRequest } from "next/server";

import { getBackendBaseUrl } from "@/lib/backend-base-url";

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ sessionId: string }> }
) {
  const { sessionId } = await context.params;
  let upstream: Response;
  try {
    upstream = await fetch(`${getBackendBaseUrl()}/api/web/sessions/${encodeURIComponent(sessionId)}/stream`, {
      headers: {
        Accept: "text/event-stream",
        "Cache-Control": "no-cache",
      },
      cache: "no-store",
    });
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: "backend_unreachable",
        message: `Could not reach POP agent backend at ${getBackendBaseUrl()}.`,
        detail: error instanceof Error ? error.message : String(error),
      }),
      {
        status: 502,
        headers: {
          "Content-Type": "application/json; charset=utf-8",
        },
      }
    );
  }

  if (!upstream.ok || !upstream.body) {
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "text/plain; charset=utf-8",
      },
    });
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
