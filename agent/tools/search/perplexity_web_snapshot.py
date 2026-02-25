from __future__ import annotations

from typing import Any, Dict, Optional

from ...agent_types import AgentTool, AgentToolResult, TextContent


class PerplexityWebSnapshotTool(AgentTool):
    name = "perplexity_web_snapshot"
    description = "Stub for a future Perplexity web snapshot endpoint."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Target URL to snapshot."},
            "return_format": {"type": "string", "description": "Desired return format (reserved)."},
            "timeout": {"type": "number", "description": "Timeout in seconds (reserved)."},
            "json_response": {"type": "boolean", "description": "Request JSON response (reserved)."},
        },
    }
    label = "Perplexity Web Snapshot"

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        url = str(params.get("url") or "").strip()
        return AgentToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        "perplexity_web_snapshot is currently a stub and is not implemented yet. "
                        "Use jina_web_snapshot or perplexity_search for now."
                    ),
                )
            ],
            details={
                "ok": False,
                "implemented": False,
                "provider": "perplexity",
                "tool": "perplexity_web_snapshot",
                "planned_capability": "web_snapshot",
                "url": url,
                "input": dict(params),
            },
        )


__all__ = ["PerplexityWebSnapshotTool"]
