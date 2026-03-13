from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def _resolve_now(now: Optional[datetime] = None) -> datetime:
    value = now if now is not None else datetime.now().astimezone()
    if value.tzinfo is None:
        return value.astimezone()
    return value.astimezone()


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def build_runtime_timestamp_block(now: Optional[datetime] = None) -> str:
    local_now = _resolve_now(now)
    utc_now = local_now.astimezone(timezone.utc)
    tz_name = local_now.tzname() or "local"
    return "\n".join(
        [
            "Runtime Time:",
            f"Current local timestamp: {_format_timestamp(local_now)} ({tz_name}).",
            f"Current UTC timestamp: {_format_timestamp(utc_now)}.",
            "Use these timestamps for time-sensitive tasks instead of inferring time from files or stale context.",
        ]
    )


def build_turn_timestamp_block(now: Optional[datetime] = None) -> str:
    local_now = _resolve_now(now)
    utc_now = local_now.astimezone(timezone.utc)
    tz_name = local_now.tzname() or "local"
    return "\n".join(
        [
            "|Current timestamp|:",
            f"Local: {_format_timestamp(local_now)} ({tz_name})",
            f"UTC: {_format_timestamp(utc_now)}",
        ]
    )


def build_timestamped_system_prompt(system_prompt: str, now: Optional[datetime] = None) -> str:
    base = str(system_prompt or "").strip()
    timestamp_block = build_runtime_timestamp_block(now=now)
    if not base:
        return timestamp_block
    return f"{base}\n\n{timestamp_block}"
