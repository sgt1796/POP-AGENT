from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from agent import Agent
from agent.agent_types import AgentMessage, TextContent, ToolCallContent


class _DoneStream:
    def __init__(self, message: Dict[str, Any]) -> None:
        self._message = message

    async def __aiter__(self):
        yield {"type": "done", "message": self._message}

    async def result(self):
        return self._message


def test_default_convert_to_llm_sanitizes_replayed_content(tmp_path: Path):
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
            role="user",
            content=[TextContent(type="text", text="existing user text", text_signature="sig-user")],
            timestamp=0.5,
        )
    )
    agent.state.messages.append(
        AgentMessage(
            role="assistant",
            content=[
                ToolCallContent(
                    type="toolCall",
                    id="call-1",
                    name="jina_web_snapshot",
                    arguments={"web_url": "https://example.com"},
                    partial_json='{"web_url":"https://example.com"}',
                    extra_content={"google": {"thought_signature": "x" * 1024}},
                )
            ],
            timestamp=0.75,
        )
    )
    agent.state.messages.append(
        AgentMessage(
            role="toolResult",
            content=[TextContent(type="text", text="tool summary", text_signature="sig-tool")],
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
    assert messages[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "existing user text"}],
    }
    assert messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "toolCall",
                "id": "call-1",
                "name": "jina_web_snapshot",
                "arguments": {"web_url": "https://example.com"},
                "extra_content": {"google": {"thought_signature": "x" * 1024}},
            }
        ],
    }
    assert messages[2]["role"] == "toolResult"
    assert messages[2]["content"] == [{"type": "text", "text": "tool summary"}]
    assert messages[2]["toolCallId"] == "tool-1"
    assert messages[2]["toolName"] == "perplexity_search"
    assert "details" not in messages[2]
    assert "timestamp" not in messages[2]
