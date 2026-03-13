from typing import Any, Dict, Optional

from agent.agent_types import AgentTool, AgentToolResult, TextContent


class MemorySearchTool(AgentTool):
    name = "memory_search"
    description = "Semantic search over stored chat memory."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results"},
            "scope": {
                "type": "string",
                "description": "Memory scope: short, long, or both",
                "enum": ["short", "long", "both"],
            },
            "session_id": {"type": "string", "description": "Optional chat session id"},
        },
        "required": ["query"],
    }
    label = "Memory Search"

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        query = str(params.get("query", "")).strip()
        if not query:
            return AgentToolResult(
                content=[TextContent(type="text", text="memory_search error: missing query")],
                details={"error": "missing query"},
            )
        try:
            top_k = int(params.get("top_k", 3) or 3)
        except Exception:
            top_k = 3
        top_k = max(1, top_k)
        scope = str(params.get("scope", "both")).strip().lower()
        if scope not in {"short", "long", "both"}:
            scope = "both"
        try:
            raw_session = params.get("session_id")
            session_id = str(raw_session).strip() if raw_session is not None else ""
            if not session_id:
                default_session = getattr(self.retriever, "default_session_id", "default")
                session_id = str(default_session or "default").strip() or "default"
            hits = self.retriever.retrieve(query=query, top_k=top_k, scope=scope, session_id=session_id)
        except Exception as exc:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"memory_search error: {exc}")],
                details={"error": str(exc)},
            )
        if not hits:
            text = "No matching memories found."
        else:
            text = "Memory search results:\n" + "\n".join(f"{i + 1}. {h}" for i, h in enumerate(hits))
        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details={"query": query, "top_k": top_k, "scope": scope, "session_id": session_id, "count": len(hits)},
        )
