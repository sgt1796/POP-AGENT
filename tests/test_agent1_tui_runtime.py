import asyncio
from types import SimpleNamespace

from agent_build.agent1.tui_runtime import AsyncDecisionQueue, format_activity_event


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


def test_format_activity_event_includes_stream_toolcall_events():
    event = {
        "type": "message_update",
        "assistantMessageEvent": {
            "type": "toolcall_start",
            "partial": {
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call-9",
                        "name": "memory_search",
                        "arguments": {"query": "alpha"},
                    }
                ]
            },
        },
    }

    assert format_activity_event(event, level="simple") == "[tool:call] toolcall_start memory_search id=call-9"
