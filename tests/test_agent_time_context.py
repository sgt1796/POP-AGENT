from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from agent import Agent
from agent.time_utils import build_timestamped_system_prompt, build_turn_timestamp_block


class _DoneStream:
    def __init__(self, message: Dict[str, Any]) -> None:
        self._message = message

    async def __aiter__(self):
        yield {"type": "done", "message": self._message}

    async def result(self):
        return self._message


def _make_agent(tmp_path: Path, stream_fn):
    return Agent(
        {
            "stream_fn": stream_fn,
            "project_root": str(tmp_path),
        }
    )


def test_time_helpers_format_explicit_utc_timestamp():
    now = datetime(2026, 3, 12, 18, 45, 30, tzinfo=timezone.utc)

    system_prompt = build_timestamped_system_prompt("Base prompt.", now=now)
    turn_block = build_turn_timestamp_block(now=now)

    assert system_prompt.startswith("Base prompt.\n\nRuntime Time:")
    assert "Current UTC timestamp: 2026-03-12T18:45:30Z." in system_prompt
    assert "|Current timestamp|:" in turn_block
    assert "UTC: 2026-03-12T18:45:30Z" in turn_block


def test_agent_core_injects_runtime_timestamp_into_system_prompt(tmp_path: Path):
    captured: Dict[str, Any] = {}

    async def _stream_fn(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]):
        del model, options
        captured["system_prompt"] = context.get("system_prompt")
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "timestamp": 0.0,
            "stopReason": "stop",
        }
        return _DoneStream(message)

    agent = _make_agent(tmp_path, _stream_fn)
    agent.set_system_prompt("Base system prompt.")

    asyncio.run(agent.prompt("hello"))

    system_prompt = str(captured.get("system_prompt") or "")
    assert system_prompt.startswith("Base system prompt.")
    assert "Runtime Time:" in system_prompt
    assert "Current local timestamp:" in system_prompt
    assert "Current UTC timestamp:" in system_prompt
    assert "instead of inferring time from files" in system_prompt
