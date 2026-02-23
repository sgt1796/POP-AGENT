from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from agent import Agent


class _DoneStream:
    def __init__(self, message: Dict[str, Any]) -> None:
        self._message = message

    async def __aiter__(self):
        yield {"type": "done", "message": self._message}

    async def result(self):
        return self._message


class _ErrorStream:
    def __init__(self, message: Dict[str, Any]) -> None:
        self._message = message

    async def __aiter__(self):
        yield {"type": "error", "error": self._message}

    async def result(self):
        return self._message


def _make_agent(tmp_path: Path, stream_fn):
    return Agent(
        {
            "stream_fn": stream_fn,
            "toolsmaker_dir": str(tmp_path / "toolsmaker"),
            "toolsmaker_audit_path": str(tmp_path / "toolsmaker" / "audit.jsonl"),
            "project_root": str(tmp_path),
        }
    )


def test_agent_tracks_usage_on_success_and_defensive_copies(tmp_path: Path):
    async def _stream_fn(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]):
        del model, context, options
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "timestamp": 0.0,
            "stopReason": "stop",
            "usage": {
                "provider": "openai",
                "model": "gpt-5-mini",
                "source": "provider",
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
                "anomaly_flag": False,
                "latency_ms": 10,
                "timestamp": 1.0,
            },
        }
        return _DoneStream(message)

    agent = _make_agent(tmp_path, _stream_fn)
    asyncio.run(agent.prompt("hello"))

    summary = agent.get_usage_summary()
    assert summary["calls"] == 1
    assert summary["total_tokens"] == 5
    assert summary["provider_calls"] == 1

    history = agent.get_usage_history()
    assert len(history) == 1
    assert history[0]["source"] == "provider"

    last = agent.get_last_usage()
    assert isinstance(last, dict)
    assert last["total_tokens"] == 5

    summary["calls"] = 999
    assert agent.get_usage_summary()["calls"] == 1
    history[0]["total_tokens"] = 999
    assert agent.get_usage_history()[0]["total_tokens"] == 5


def test_agent_tracks_usage_on_error_with_usage_payload(tmp_path: Path):
    async def _stream_fn(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]):
        del model, context, options
        error_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "timestamp": 0.0,
            "stopReason": "error",
            "errorMessage": "boom",
            "usage": {
                "provider": "openai",
                "model": "gpt-5-mini",
                "source": "provider",
                "input_tokens": 4,
                "output_tokens": 0,
                "total_tokens": 4,
                "anomaly_flag": False,
                "latency_ms": 20,
                "timestamp": 2.0,
            },
        }
        return _ErrorStream(error_message)

    agent = _make_agent(tmp_path, _stream_fn)
    asyncio.run(agent.prompt("trigger error"))

    summary = agent.get_usage_summary()
    assert summary["calls"] == 1
    assert summary["total_tokens"] == 4
    assert summary["provider_calls"] == 1
    assert str(agent.state.error) == "boom"


def test_agent_normalizes_usage_when_stream_omits_usage(tmp_path: Path):
    async def _stream_fn(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]):
        del model, context, options
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "no usage provided"}],
            "timestamp": 0.0,
            "stopReason": "stop",
        }
        return _DoneStream(message)

    agent = _make_agent(tmp_path, _stream_fn)
    asyncio.run(agent.prompt("hello"))

    last = agent.get_last_usage()
    assert isinstance(last, dict)
    assert "source" in last
    assert "estimate_total_tokens" in last
    assert isinstance(agent.state.messages[-1].usage, dict)
    assert "source" in dict(agent.state.messages[-1].usage or {})
    assert agent.get_usage_summary()["calls"] == 1


def test_agent_usage_history_is_bounded_and_reset_is_explicit(tmp_path: Path):
    async def _stream_fn(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]):
        del model, context, options
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "timestamp": 0.0,
            "stopReason": "stop",
            "usage": {
                "provider": "openai",
                "model": "gpt-5-mini",
                "source": "provider",
                "input_tokens": 1,
                "output_tokens": 0,
                "total_tokens": 1,
                "anomaly_flag": False,
                "latency_ms": 1,
                "timestamp": 3.0,
            },
        }
        return _DoneStream(message)

    agent = _make_agent(tmp_path, _stream_fn)

    async def _run_many() -> None:
        for i in range(205):
            await agent.prompt(f"msg {i}")

    asyncio.run(_run_many())

    assert len(agent.get_usage_history()) == 200
    totals_before_reset = agent.get_usage_summary()
    assert totals_before_reset["calls"] == 205
    assert totals_before_reset["total_tokens"] == 205

    agent.reset()
    assert agent.get_usage_summary()["calls"] == 205

    agent.reset_usage_tracking()
    assert agent.get_last_usage() is None
    assert agent.get_usage_history() == []
    assert agent.get_usage_summary()["calls"] == 0
