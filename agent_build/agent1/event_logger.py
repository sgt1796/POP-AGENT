from typing import Any, Dict

from .constants import LOG_LEVELS
from .message_utils import extract_bash_exec_command, format_message_line


def resolve_log_level(value: str) -> int:
    if not value:
        return LOG_LEVELS["quiet"]
    key = str(value).strip().lower()
    if key.isdigit():
        return int(key)
    aliases = {
        "messages": "simple",
        "stream": "full",
    }
    key = aliases.get(key, key)
    return LOG_LEVELS.get(key, LOG_LEVELS["quiet"])


def make_event_logger(level: str = "quiet"):
    """Create an event logger function for agent events.
    Levels:
    - quiet: no logging
    - simple: log tool calls and bash command executions
    - full: log simple output plus event context
    - debug: log all events
    """
    level_value = resolve_log_level(level)

    def log(event: Dict[str, Any]) -> None:
        etype = event.get("type")
        if level_value <= LOG_LEVELS["quiet"]:
            return

        if etype == "tool_execution_start" and level_value >= LOG_LEVELS["simple"]:
            tool_name = str(event.get("toolName", "")).strip() or "unknown"
            if tool_name == "bash_exec":
                command = extract_bash_exec_command(event)
                if command:
                    preview = " ".join(command.split()[:6])
                    print(f"[tool:start] bash_exec cmd={preview}")
                else:
                    print("[tool:start] bash_exec")
            else:
                print(f"[tool:start] {tool_name}")
            return
        if etype == "tool_execution_end" and level_value >= LOG_LEVELS["simple"]:
            tool_name = str(event.get("toolName", "")).strip() or "unknown"
            is_error = bool(event.get("isError"))
            if tool_name == "bash_exec":
                command = extract_bash_exec_command(event)
                if command:
                    preview = " ".join(command.split()[:6])
                    print(f"[tool:end] bash_exec error={is_error} cmd={preview}")
                else:
                    print(f"[tool:end] bash_exec error={is_error}")
            else:
                print(f"[tool:end] {tool_name} error={is_error}")
            return
        if etype == "message_end" and level_value >= LOG_LEVELS["full"]:
            message = event.get("message")
            if message:
                print(format_message_line(message))
            return
        if etype == "message_update" and level_value >= LOG_LEVELS["full"]:
            assistant_event = event.get("assistantMessageEvent") or {}
            if assistant_event.get("type") == "text_delta":
                delta = assistant_event.get("delta")
                if delta:
                    print(f"[stream] {delta}")
            return
        if level_value >= LOG_LEVELS["full"]:
            if etype in {"tool_execution_result", "tool_execution_error", "memory_context", "memory_lookup"}:
                print(f"[context] {event}")
                return
        if level_value >= LOG_LEVELS["debug"]:
            print(f"[debug] {event}")

    return log
