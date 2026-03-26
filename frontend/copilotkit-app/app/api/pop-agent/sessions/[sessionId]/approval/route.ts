import { NextRequest } from "next/server";

import { getBackendBaseUrl } from "@/lib/backend-base-url";

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ sessionId: string }> }
) {
  const { sessionId } = await context.params;
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(`${getBackendBaseUrl()}/api/web/sessions/${encodeURIComponent(sessionId)}/approval`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body,
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

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || "application/json; charset=utf-8",
    },
  });
}
