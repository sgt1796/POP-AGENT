import asyncio
from types import SimpleNamespace

from agent_build.agent1.tui_runtime import (
    AsyncDecisionQueue,
    AsyncToolsmakerApprovalSubscriber,
    ToolsmakerDecision,
    format_activity_event,
)


class _FakeActivatedTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAgent:
    def __init__(self) -> None:
        self.approve_calls = []
        self.activate_calls = []
        self.reject_calls = []

    def approve_dynamic_tool(self, name: str, version: int):
        self.approve_calls.append((name, version))
        return SimpleNamespace(status="approved")

    def activate_tool_version(self, name: str, version: int):
        self.activate_calls.append((name, version))
        return _FakeActivatedTool(name=name)

    def reject_dynamic_tool(self, name: str, version: int, reason: str):
        self.reject_calls.append((name, version, reason))
        return SimpleNamespace(status="rejected")


def _event(
    *,
    etype: str = "tool_execution_end",
    tool_name: str = "toolsmaker",
    action: str = "create",
    status: str = "approval_required",
    ok: bool = True,
    name: str = "gmail_fetcher",
    version: int = 1,
):
    details = {
        "ok": ok,
        "action": action,
        "status": status,
        "name": name,
        "version": version,
    }
    return {
        "type": etype,
        "toolName": tool_name,
        "result": SimpleNamespace(details=details),
    }


def test_async_decision_queue_fifo_order():
    queue: AsyncDecisionQueue[int] = AsyncDecisionQueue()

    async def _run() -> list[int]:
        queue.put(1)
        queue.put(2)
        return [await queue.get(), await queue.get()]

    result = asyncio.run(_run())
    assert result == [1, 2]


def test_async_decision_queue_waiter_gets_later_put():
    queue: AsyncDecisionQueue[str] = AsyncDecisionQueue()

    async def _run() -> str:
        task = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        queue.put("ready")
        return await task

    assert asyncio.run(_run()) == "ready"


def test_async_decision_queue_drain_returns_and_clears_items():
    queue: AsyncDecisionQueue[int] = AsyncDecisionQueue()
    queue.put(10)
    queue.put(20)

    drained = queue.drain()

    assert drained == [10, 20]
    assert queue.empty() is True


def test_format_activity_event_covers_tool_and_stream_events():
    start_text = format_activity_event({"type": "tool_execution_start", "toolName": "bash_exec", "args": {"cmd": "ls"}})
    end_text = format_activity_event(
        {"type": "tool_execution_end", "toolName": "bash_exec", "args": {"cmd": "ls"}, "isError": False}
    )
    stream_text = format_activity_event(
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "hello"}}
    )
    err_text = format_activity_event(
        {
            "type": "message_end",
            "message": SimpleNamespace(role="assistant", error_message="boom"),
        }
    )

    assert start_text == "[tool:start] bash_exec args={'cmd': 'ls'}"
    assert end_text == "[tool:end] bash_exec error=False cmd=ls"
    assert stream_text == "[stream] hello"
    assert err_text == "[assistant:error] boom"




def test_format_activity_event_simple_mode_uses_compact_bash_preview():
    start_text = format_activity_event(
        {"type": "tool_execution_start", "toolName": "bash_exec", "args": {"cmd": "python app.py --mode demo --verbose"}},
        level="simple",
    )
    stream_text = format_activity_event(
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "hello"}},
        level="simple",
    )

    assert start_text == "[tool:start] bash_exec cmd=python app.py --mode demo --verbose"
    assert stream_text is None


def test_async_toolsmaker_subscriber_approves_and_activates_once():
    agent = _FakeAgent()

    async def _resolve(_details):
        return ToolsmakerDecision(approve=True, activate=True, reason="")

    subscriber = AsyncToolsmakerApprovalSubscriber(agent=agent, resolve_decision=_resolve)
    event = _event(name="pdf_merger", version=3)

    async def _run() -> None:
        subscriber.on_event(event)
        subscriber.on_event(event)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert agent.approve_calls == [("pdf_merger", 3)]
    assert agent.activate_calls == [("pdf_merger", 3)]
    assert agent.reject_calls == []


def test_async_toolsmaker_subscriber_rejects_with_default_reason():
    agent = _FakeAgent()

    async def _resolve(_details):
        return ToolsmakerDecision(approve=False, activate=False, reason="")

    subscriber = AsyncToolsmakerApprovalSubscriber(agent=agent, resolve_decision=_resolve)

    async def _run() -> None:
        subscriber.on_event(_event(name="doc_tool", version=2))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert agent.approve_calls == []
    assert agent.activate_calls == []
    assert agent.reject_calls == [("doc_tool", 2, "rejected_by_reviewer")]


def test_async_toolsmaker_subscriber_handles_decision_exception_with_reject():
    agent = _FakeAgent()
    activities = []

    async def _resolve(_details):
        raise RuntimeError("decision failed")

    subscriber = AsyncToolsmakerApprovalSubscriber(
        agent=agent,
        resolve_decision=_resolve,
        on_activity=activities.append,
    )

    async def _run() -> None:
        subscriber.on_event(_event(name="unstable_tool", version=5))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert agent.approve_calls == []
    assert agent.activate_calls == []
    assert agent.reject_calls == [("unstable_tool", 5, "decision_error")]
    assert any("decision warning" in line for line in activities)
