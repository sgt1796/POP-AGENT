from __future__ import annotations

from typing import Any, Dict, Optional

from POP.utils import web_snapshot as websnapshot

from ...agent_types import AgentTool, AgentToolResult, TextContent


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in {"1", "true", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


class JinaWebSnapshotTool(AgentTool):
    name = "jina_web_snapshot"
    description = "Fetch a text snapshot of a webpage using POP.utils.web_snapshot."
    parameters = {
        "type": "object",
        "properties": {
            "web_url": {"type": "string", "description": "URL to snapshot"},
            "url": {"type": "string", "description": "Alias for web_url"},
            "use_api_key": {"type": "boolean"},
            "return_format": {"type": "string"},
            "timeout": {"type": "number"},
            "target_selector": {"type": "array", "items": {"type": "string"}},
            "wait_for_selector": {"type": "array", "items": {"type": "string"}},
            "exclude_selector": {"type": "array", "items": {"type": "string"}},
            "remove_image": {"type": "boolean"},
            "links_at_end": {"type": "boolean"},
            "images_at_end": {"type": "boolean"},
            "json_response": {"type": "boolean"},
            "image_caption": {
                "type": "boolean",
                "description": (
                    "Caption images in the snapshot using AI. Note: this may consume additional tokens and time."
                ),
            },
            "cookie": {"type": "string"},
        },
        "required": ["web_url"],
    }
    label = "Jina Web Snapshot"

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": False, **details})

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": True, **details})

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        web_url = str(params.get("web_url") or params.get("url") or "").strip()
        if not web_url:
            return self._error(
                "jina_web_snapshot error: missing web_url",
                {"error": "missing web_url"},
            )

        kwargs = {
            "use_api_key": _to_bool(params.get("use_api_key"), True),
            "return_format": str(params.get("return_format") or "default"),
            "timeout": _to_int(params.get("timeout"), 0),
            "target_selector": params.get("target_selector") or None,
            "wait_for_selector": params.get("wait_for_selector") or None,
            "exclude_selector": params.get("exclude_selector") or None,
            "remove_image": _to_bool(params.get("remove_image"), False),
            "links_at_end": _to_bool(params.get("links_at_end"), False),
            "images_at_end": _to_bool(params.get("images_at_end"), False),
            "json_response": _to_bool(params.get("json_response"), False),
            "image_caption": _to_bool(params.get("image_caption"), False),
            "cookie": params.get("cookie"),
        }
        try:
            snapshot = websnapshot.get_text_snapshot(web_url=web_url, **kwargs)
        except Exception as exc:
            return self._error(
                f"jina_web_snapshot error: {exc}",
                {"error": str(exc), "url": web_url},
            )
        return self._ok(str(snapshot), {"url": web_url})


# Backward-compatible class name.
WebSnapshotTool = JinaWebSnapshotTool


__all__ = ["JinaWebSnapshotTool", "WebSnapshotTool"]
