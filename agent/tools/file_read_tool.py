from __future__ import annotations

import base64
import csv
import json
import mimetypes
import os
from io import StringIO
from typing import Any, Dict, Optional

from ..agent_types import AgentTool, AgentToolResult, TextContent

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - exercised through runtime error path
    PdfReader = None


_TEXT_SUFFIXES = {".txt", ".md"}
_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".svg",
    ".ico",
    ".heic",
    ".heif",
    ".avif",
}


class FileReadError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(str(message))
        self.code = str(code or "parse_error")
        self.message = str(message or "unknown file read error")


def _path_in_workspace(path: str, workspace_root: str) -> bool:
    try:
        return os.path.commonpath([workspace_root, path]) == workspace_root
    except ValueError:
        return False


def _resolve_workspace_path(path_value: str, workspace_root: str) -> str:
    candidate = str(path_value or "").strip()
    if not candidate:
        raise FileReadError("path_not_found", "path is required")
    absolute = os.path.realpath(candidate if os.path.isabs(candidate) else os.path.join(workspace_root, candidate))
    if not _path_in_workspace(absolute, workspace_root):
        raise FileReadError("path_outside_workspace", "path must be inside workspace")
    if not os.path.exists(absolute):
        raise FileReadError("path_not_found", f"file not found: {path_value}")
    if not os.path.isfile(absolute):
        raise FileReadError("path_not_file", f"path is not a file: {path_value}")
    return absolute


def _workspace_relative_path(path: str, workspace_root: str) -> str:
    try:
        rel = os.path.relpath(path, workspace_root)
    except Exception:
        rel = path
    return str(rel).replace("\\", "/")


def _read_utf8_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except UnicodeDecodeError as exc:
        raise FileReadError("parse_error", f"unable to decode UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise FileReadError("parse_error", f"unable to read file: {exc}") from exc


def _detect_image_kind(path: str, suffix: str) -> bool:
    if suffix in _IMAGE_SUFFIXES:
        return True
    mime_type, _ = mimetypes.guess_type(path)
    return str(mime_type or "").strip().lower().startswith("image/")


def read(
    path: str,
    *,
    workspace_root: Optional[str] = None,
    max_chars: int = 200_000,
    max_bytes: int = 10_000_000,
) -> Dict[str, Any]:
    root = os.path.realpath(str(workspace_root or os.getcwd()))
    try:
        max_chars_value = int(max_chars)
    except Exception as exc:
        raise FileReadError("parse_error", "max_chars must be an integer") from exc
    if max_chars_value <= 0:
        raise FileReadError("parse_error", "max_chars must be > 0")

    try:
        max_bytes_value = int(max_bytes)
    except Exception as exc:
        raise FileReadError("parse_error", "max_bytes must be an integer") from exc
    if max_bytes_value <= 0:
        raise FileReadError("parse_error", "max_bytes must be > 0")

    absolute = _resolve_workspace_path(path, root)
    size = os.path.getsize(absolute)
    if size > max_bytes_value:
        raise FileReadError("file_too_large", f"file size {size} exceeds max_bytes={max_bytes_value}")

    suffix = os.path.splitext(absolute)[1].lower()
    workspace_path = _workspace_relative_path(absolute, root)

    if suffix in _TEXT_SUFFIXES:
        text = _read_utf8_text(absolute)
        original_chars = len(text)
        truncated = original_chars > max_chars_value
        return {
            "ok": True,
            "path": absolute,
            "workspace_path": workspace_path,
            "suffix": suffix,
            "kind": "text",
            "content": text[:max_chars_value] if truncated else text,
            "metadata": {"encoding": "utf-8", "char_count": original_chars, "byte_size": size},
            "truncated": truncated,
        }

    if suffix == ".json":
        text = _read_utf8_text(absolute)
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise FileReadError("parse_error", f"invalid json: {exc}") from exc
        return {
            "ok": True,
            "path": absolute,
            "workspace_path": workspace_path,
            "suffix": suffix,
            "kind": "json",
            "content": parsed,
            "metadata": {"encoding": "utf-8", "byte_size": size, "value_type": type(parsed).__name__},
            "truncated": False,
        }

    if suffix == ".csv":
        text = _read_utf8_text(absolute)
        try:
            reader = csv.DictReader(StringIO(text))
            headers = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        except Exception as exc:
            raise FileReadError("parse_error", f"invalid csv: {exc}") from exc
        return {
            "ok": True,
            "path": absolute,
            "workspace_path": workspace_path,
            "suffix": suffix,
            "kind": "csv",
            "content": rows,
            "metadata": {"encoding": "utf-8", "headers": headers, "row_count": len(rows), "byte_size": size},
            "truncated": False,
        }

    if suffix == ".pdf":
        if PdfReader is None:
            raise FileReadError("missing_dependency", "pypdf is required to read pdf files")
        try:
            with open(absolute, "rb") as handle:
                reader = PdfReader(handle)
                pages = list(reader.pages)
                parts = []
                non_empty_pages = 0
                for page in pages:
                    extracted = str(page.extract_text() or "")
                    if extracted.strip():
                        non_empty_pages += 1
                    parts.append(extracted)
        except FileReadError:
            raise
        except Exception as exc:
            raise FileReadError("parse_error", f"unable to parse pdf: {exc}") from exc

        text = "\n\n".join(parts)
        original_chars = len(text)
        truncated = original_chars > max_chars_value
        return {
            "ok": True,
            "path": absolute,
            "workspace_path": workspace_path,
            "suffix": suffix,
            "kind": "pdf",
            "content": text[:max_chars_value] if truncated else text,
            "metadata": {
                "page_count": len(pages),
                "non_empty_pages": non_empty_pages,
                "extracted_char_count": original_chars,
                "byte_size": size,
            },
            "truncated": truncated,
        }

    if _detect_image_kind(absolute, suffix):
        try:
            with open(absolute, "rb") as handle:
                payload = handle.read()
        except OSError as exc:
            raise FileReadError("parse_error", f"unable to read image: {exc}") from exc
        mime_type, _ = mimetypes.guess_type(absolute)
        encoded = base64.b64encode(payload).decode("ascii")
        return {
            "ok": True,
            "path": absolute,
            "workspace_path": workspace_path,
            "suffix": suffix,
            "kind": "image",
            "content": encoded,
            "metadata": {
                "mime_type": str(mime_type or "application/octet-stream"),
                "byte_size": len(payload),
                "base64_chars": len(encoded),
            },
            "truncated": False,
        }

    raise FileReadError("unsupported_suffix", f"unsupported file suffix: {suffix or '(none)'}")


class FileReadTool(AgentTool):
    name = "file_read"
    description = (
        "Read and parse workspace files by suffix. Supports txt, md, json, csv, pdf, and image-to-base64."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace root or absolute path."},
            "max_chars": {"type": "integer", "description": "Optional max returned chars for text/pdf content."},
        },
        "required": ["path"],
    }
    label = "File Read"

    def __init__(self, workspace_root: Optional[str] = None, max_output_chars: int = 20_000) -> None:
        self.workspace_root = os.path.realpath(str(workspace_root or os.getcwd()))
        self.max_output_chars = max(512, int(max_output_chars or 20_000))

    @staticmethod
    def _error(path_value: str, code: str, message: str) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"file_read error: {message}")],
            details={
                "ok": False,
                "error": str(code or "parse_error"),
                "message": str(message or "file read failed"),
                "path": str(path_value or ""),
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
        if not path_value:
            return self._error(path_value, "path_not_found", "path is required")

        max_chars = params.get("max_chars", 200_000)
        try:
            payload = read(path_value, workspace_root=self.workspace_root, max_chars=int(max_chars))
        except FileReadError as exc:
            return self._error(path_value, exc.code, exc.message)
        except Exception as exc:
            return self._error(path_value, "parse_error", str(exc))

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
