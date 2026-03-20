from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from agent import Agent
from agent.agent_types import AgentMessage, TextContent


class _DoneStream:
    def __init__(self, message: Dict[str, Any]) -> None:
        self._message = message

    async def __aiter__(self):
        yield {"type": "done", "message": self._message}

    async def result(self):
        return self._message


def test_default_convert_to_llm_excludes_tool_details_from_context(tmp_path: Path):
    captured: Dict[str, Any] = {}

    async def _stream_fn(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]):
        del model, options
        captured["messages"] = context.get("messages")
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "timestamp": 0.0,
            "stopReason": "stop",
        }
        return _DoneStream(message)

    agent = Agent({"stream_fn": _stream_fn, "project_root": str(tmp_path)})
    agent.state.messages.append(
        AgentMessage(
            role="toolResult",
            content=[TextContent(type="text", text="tool summary")],
            timestamp=1.0,
            tool_call_id="tool-1",
            tool_name="perplexity_search",
            details={
                "ok": True,
                "results": [{"title": "Example", "snippet": "x" * 5000}],
                "usage": {"input_tokens": 99999},
            },
            is_error=False,
        )
    )

    asyncio.run(agent.prompt("hello"))

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "toolResult"
    assert messages[0]["content"] == [{"type": "text", "text": "tool summary", "text_signature": None}]
    assert messages[0]["toolCallId"] == "tool-1"
    assert messages[0]["toolName"] == "perplexity_search"
    assert "details" not in messages[0]
    assert "timestamp" not in messages[0]

