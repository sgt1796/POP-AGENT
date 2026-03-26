from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional, Sequence

from ..agent_types import AgentTool, AgentToolResult, TextContent
from .path_roots import normalize_allowed_roots, path_in_roots


_ACTIONS = {"write", "append", "replace"}


class FileWriteError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(str(message))
        self.code = str(code or "write_error")
        self.message = str(message or "file write failed")


def _resolve_workspace_path(path_value: str, workspace_root: str, allowed_roots: Sequence[str]) -> str:
    candidate = str(path_value or "").strip()
    if not candidate:
        raise FileWriteError("path_not_found", "path is required")
    absolute = os.path.realpath(candidate if os.path.isabs(candidate) else os.path.join(workspace_root, candidate))
    if not path_in_roots(absolute, allowed_roots):
        raise FileWriteError("path_outside_workspace", "path must be inside workspace or configured allowed roots")
    return absolute


def _workspace_relative_path(path: str, workspace_root: str) -> str:
    try:
        rel = os.path.relpath(path, workspace_root)
    except Exception:
        rel = path
    return str(rel).replace("\\", "/")


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _atomic_write_text(path: str, content: str) -> None:
    parent = os.path.dirname(path) or "."
    fd, temp_path = tempfile.mkstemp(prefix=".file_write_", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def write(
    path: str,
    *,
    workspace_root: Optional[str] = None,
    allowed_roots: Optional[Sequence[str]] = None,
    action: str = "write",
    content: Optional[Any] = None,
    find: Optional[Any] = None,
    replace_with: Optional[Any] = "",
    count: Optional[Any] = None,
    overwrite: bool = True,
    create_dirs: bool = False,
) -> Dict[str, Any]:
    root, roots = normalize_allowed_roots(workspace_root, allowed_roots)
    normalized_action = str(action or "write").strip().lower()
    if normalized_action not in _ACTIONS:
        raise FileWriteError("invalid_action", f"action must be one of: {', '.join(sorted(_ACTIONS))}")

    absolute = _resolve_workspace_path(path, root, roots)
    workspace_path = _workspace_relative_path(absolute, root)
    exists_before = os.path.exists(absolute)
    if exists_before and not os.path.isfile(absolute):
        raise FileWriteError("path_not_file", f"path is not a file: {path}")

    parent = os.path.dirname(absolute) or root
    if not os.path.isdir(parent):
        if not create_dirs:
            raise FileWriteError("parent_missing", f"parent directory does not exist: {parent}")
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception as exc:
            raise FileWriteError("parent_create_failed", f"unable to create parent directory: {exc}") from exc

    if normalized_action in {"write", "append"}:
        if content is None:
            raise FileWriteError("missing_content", "content is required for write/append")
        text = str(content)

        if normalized_action == "write":
            if exists_before and not overwrite:
                raise FileWriteError("path_exists", "file already exists and overwrite=false")
            try:
                _atomic_write_text(absolute, text)
            except Exception as exc:
                raise FileWriteError("write_error", f"unable to write file: {exc}") from exc

            return {
                "ok": True,
                "action": normalized_action,
                "path": absolute,
                "workspace_path": workspace_path,
                "created": not exists_before,
                "overwrote_existing": bool(exists_before),
                "char_count": len(text),
                "byte_size": os.path.getsize(absolute),
            }

        try:
            with open(absolute, "a", encoding="utf-8") as handle:
                handle.write(text)
        except Exception as exc:
            raise FileWriteError("write_error", f"unable to append file: {exc}") from exc
        return {
            "ok": True,
            "action": normalized_action,
            "path": absolute,
            "workspace_path": workspace_path,
            "created": not exists_before,
            "appended_char_count": len(text),
            "byte_size": os.path.getsize(absolute),
        }

    if not exists_before:
        raise FileWriteError("path_not_found", f"file not found: {path}")
    if find is None:
        raise FileWriteError("missing_find", "find is required for replace")
    find_text = str(find)
    if not find_text:
        raise FileWriteError("missing_find", "find must be non-empty for replace")
    replace_text = "" if replace_with is None else str(replace_with)

    count_limit: Optional[int] = None
    if count not in {None, ""}:
        try:
            parsed = int(count)
        except Exception as exc:
            raise FileWriteError("invalid_count", "count must be an integer > 0") from exc
        if parsed <= 0:
            raise FileWriteError("invalid_count", "count must be > 0")
        count_limit = parsed

    try:
        with open(absolute, "r", encoding="utf-8") as handle:
            original_text = handle.read()
    except UnicodeDecodeError as exc:
        raise FileWriteError("decode_error", f"unable to decode UTF-8 text: {exc}") from exc
    except Exception as exc:
        raise FileWriteError("read_error", f"unable to read file: {exc}") from exc

    occurrences = original_text.count(find_text)
    if count_limit is None:
        replaced_count = occurrences
        updated_text = original_text.replace(find_text, replace_text)
    else:
        replaced_count = min(occurrences, count_limit)
        updated_text = original_text.replace(find_text, replace_text, count_limit)

    changed = updated_text != original_text
    if changed:
        try:
            _atomic_write_text(absolute, updated_text)
        except Exception as exc:
            raise FileWriteError("write_error", f"unable to write replace result: {exc}") from exc

    return {
        "ok": True,
        "action": normalized_action,
        "path": absolute,
        "workspace_path": workspace_path,
        "find": find_text,
        "replace_with": replace_text,
        "count_limit": count_limit,
        "replacements": replaced_count,
        "changed": changed,
        "original_char_count": len(original_text),
        "final_char_count": len(updated_text),
        "byte_size": os.path.getsize(absolute),
    }


class FileWriteTool(AgentTool):
    name = "file_write"
    description = (
        "Write, append, and find/replace UTF-8 text files in the workspace or allowed roots. "
        "Prefer downloads/<name> for scratch scripts or generated artifacts unless editing a project file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "File path relative to the workspace root or an absolute path inside allowed roots. "
                    "Prefer downloads/<descriptive_name> for scratch scripts or generated files unless the task "
                    "names a project file."
                ),
            },
            "action": {
                "type": "string",
                "enum": ["write", "append", "replace"],
                "description": "Operation type. Defaults to write.",
            },
            "content": {"type": "string", "description": "Text content for write/append actions."},
            "find": {"type": "string", "description": "Literal text to find for replace action."},
            "replace_with": {"type": "string", "description": "Replacement text for replace action."},
            "count": {"type": "integer", "description": "Optional max replacements for replace action (>0)."},
            "overwrite": {
                "type": "boolean",
                "description": "For write action, overwrite existing file (default true).",
            },
            "create_dirs": {
                "type": "boolean",
                "description": "Create missing parent directories before writing (default false).",
            },
        },
        "required": ["path"],
    }
    label = "File Write"

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        max_output_chars: int = 20_000,
        allowed_roots: Optional[Sequence[str]] = None,
    ) -> None:
        self.workspace_root, self.allowed_roots = normalize_allowed_roots(workspace_root, allowed_roots)
        self.max_output_chars = max(512, int(max_output_chars or 20_000))

    @staticmethod
    def _error(path_value: str, action_value: str, code: str, message: str) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"file_write error: {message}")],
            details={
                "ok": False,
                "error": str(code or "write_error"),
                "message": str(message or "file write failed"),
                "path": str(path_value or ""),
                "action": str(action_value or "write"),
            },
        )

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        path_value = str(params.get("path", "") or "").strip()
        action_value = str(params.get("action", "write") or "write").strip().lower()
        if not path_value:
            return self._error(path_value, action_value, "path_not_found", "path is required")

        try:
            payload = write(
                path_value,
                workspace_root=self.workspace_root,
                allowed_roots=self.allowed_roots,
                action=action_value,
                content=params.get("content"),
                find=params.get("find"),
                replace_with=params.get("replace_with"),
                count=params.get("count"),
                overwrite=_to_bool(params.get("overwrite"), True),
                create_dirs=_to_bool(params.get("create_dirs"), False),
            )
        except FileWriteError as exc:
            return self._error(path_value, action_value, exc.code, exc.message)
        except Exception as exc:
            return self._error(path_value, action_value, "write_error", str(exc))

        details: Dict[str, Any] = dict(payload)
        rendered = json.dumps(payload, ensure_ascii=False)
        if len(rendered) > self.max_output_chars:
            rendered = rendered[: self.max_output_chars]
            details["output_truncated"] = True
            details["max_output_chars"] = self.max_output_chars

        return AgentToolResult(
            content=[TextContent(type="text", text=rendered)],
            details=details,
        )
