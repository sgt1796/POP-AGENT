import json
from types import SimpleNamespace

import pytest

from agent_build.agent1.web.agui import AGUIEventBridge


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeAgent:
    def __init__(self) -> None:
        self._subscribers = []

    def subscribe(self, callback):
        self._subscribers.append(callback)

        def _unsubscribe():
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return _unsubscribe

    def emit(self, event):
        for callback in list(self._subscribers):
            callback(event)


class _FakeManagedSession:
    def __init__(self, agent: _FakeAgent, *, snapshot_message: str, emit_live_text: str | None = None) -> None:
        self.runtime = SimpleNamespace(agent=agent)
        self._snapshot_message = snapshot_message
        self._emit_live_text = emit_live_text

    async def run_turn(self, _user_message: str):
        if self._emit_live_text:
            self.runtime.agent.emit(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {
                        "type": "text_delta",
                        "delta": self._emit_live_text,
                    },
                }
            )
        return SimpleNamespace(message=self._snapshot_message)


def _decode_sse_payload(payload: str) -> dict:
    for line in payload.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise AssertionError(f"missing data line in payload: {payload!r}")


def test_agui_bridge_translates_text_and_tool_events_into_agui_models() -> None:
    agent = _FakeAgent()
    managed_session = SimpleNamespace(runtime=SimpleNamespace(agent=agent))
    bridge = AGUIEventBridge(managed_session, thread_id="thread-1", run_id="run-1", accept=None)

    bridge.on_agent_event(
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "hello"},
        }
    )
    bridge.on_agent_event(
        {
            "type": "tool_execution_start",
            "toolCallId": "tool-1",
            "toolName": "web_search",
            "args": {"query": "us market"},
        }
    )
    bridge.on_agent_event(
        {
            "type": "tool_execution_end",
            "toolCallId": "tool-1",
            "toolName": "web_search",
            "result": {"details": {"ok": True}},
        }
    )

    payloads = [_decode_sse_payload(bridge.queue.get_nowait()) for _ in range(bridge.queue.qsize())]

    assert [payload["type"] for payload in payloads] == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
    ]
    assert payloads[1]["delta"] == "hello"
    assert payloads[2]["toolCallId"] == "tool-1"
    assert payloads[3]["delta"] == '{"query": "us market"}'
    assert payloads[5]["content"] == '{"ok": true}'


@pytest.mark.anyio
async def test_agui_bridge_stream_turn_emits_final_text_message_lifecycle() -> None:
    agent = _FakeAgent()
    managed_session = _FakeManagedSession(agent, snapshot_message="final reply")
    bridge = AGUIEventBridge(managed_session, thread_id="thread-1", run_id="run-1", accept=None)

    payloads = [_decode_sse_payload(chunk) async for chunk in bridge.stream_turn("hello")]

    assert [payload["type"] for payload in payloads] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert payloads[2]["delta"] == "final reply"
    assert payloads[4]["result"] == "final reply"


@pytest.mark.anyio
async def test_agui_bridge_does_not_duplicate_snapshot_text_after_live_deltas() -> None:
    agent = _FakeAgent()
    managed_session = _FakeManagedSession(agent, snapshot_message="final reply", emit_live_text="live delta")
    bridge = AGUIEventBridge(managed_session, thread_id="thread-1", run_id="run-1", accept=None)

    payloads = [_decode_sse_payload(chunk) async for chunk in bridge.stream_turn("hello")]

    assert [payload["type"] for payload in payloads] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert payloads[2]["delta"] == "live delta"
    assert payloads[4]["result"] == "final reply"
