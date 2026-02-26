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
    """Create a user-visible event logger function for agent events.
    Levels:
    - quiet: no user-visible logging
    - simple: log tool calls and bash command executions
    - full: log simple output plus event context
    - debug: log all events
    """
    level_value = resolve_log_level(level)

    def _read_toolcall_preview(assistant_event: Dict[str, Any]) -> tuple[str, str, Any]:
        partial = assistant_event.get("partial")
        if not isinstance(partial, dict):
            return "", "unknown", None
        content = partial.get("content")
        if not isinstance(content, list):
            return "", "unknown", None
        for item in reversed(content):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip() != "toolCall":
                continue
            call_id = str(item.get("id", "")).strip()
            tool_name = str(item.get("name", "")).strip() or "unknown"
            args = item.get("arguments")
            return call_id, tool_name, args
        return "", "unknown", None

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
            assistant_event_type = str(assistant_event.get("type", "")).strip()
            if assistant_event_type in {"toolcall_start", "toolcall_delta", "toolcall_end"}:
                call_id, tool_name, args = _read_toolcall_preview(assistant_event)
                suffix = f" id={call_id}" if call_id else ""
                if args not in (None, ""):
                    print(f"[tool:call] {assistant_event_type} {tool_name}{suffix} args={args}")
                else:
                    print(f"[tool:call] {assistant_event_type} {tool_name}{suffix}")
                return
            if assistant_event.get("type") == "text_delta":
                delta = assistant_event.get("delta")
                if delta:
                    print(f"[stream] {delta}")
            return
        if etype == "message_update" and level_value >= LOG_LEVELS["simple"]:
            assistant_event = event.get("assistantMessageEvent") or {}
            assistant_event_type = str(assistant_event.get("type", "")).strip()
            if assistant_event_type in {"toolcall_start", "toolcall_end"}:
                call_id, tool_name, _ = _read_toolcall_preview(assistant_event)
                suffix = f" id={call_id}" if call_id else ""
                print(f"[tool:call] {assistant_event_type} {tool_name}{suffix}")
                return
        if level_value >= LOG_LEVELS["full"]:
            if etype in {"tool_execution_result", "tool_execution_error", "memory_context", "memory_lookup"}:
                print(f"[context] {event}")
                return
        if level_value >= LOG_LEVELS["debug"]:
            print(f"[debug] {event}")

    return log
