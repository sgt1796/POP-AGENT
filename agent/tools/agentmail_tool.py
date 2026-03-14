from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..agent_types import AgentTool, AgentToolResult, TextContent
from .path_roots import normalize_allowed_roots, path_in_roots

try:
    from agentmail import AgentMail
except ImportError:  # pragma: no cover - exercised via tool result, not import failure
    AgentMail = None  # type: ignore[assignment]


_REQUIRED_ENV_VARS = (
    "AGENTMAIL_API_KEY",
    "POP_AGENT_AGENTMAIL_INBOX_ID",
    "POP_AGENT_AGENTMAIL_TO_EMAIL",
)


def _resolve_workspace_path(path_value: str, workspace_root: str, allowed_roots: Sequence[str]) -> str:
    candidate = str(path_value).strip()
    if not candidate:
        raise ValueError("path is required")
    resolved = os.path.realpath(candidate if os.path.isabs(candidate) else os.path.join(workspace_root, candidate))
    if not path_in_roots(resolved, allowed_roots):
        raise ValueError("path_outside_workspace")
    return resolved


def _to_text(value: Any) -> str:
    return str(value or "").strip()


@dataclass
class _AgentMailConfig:
    api_key: str
    inbox_id: str
    to_email: str


class _AgentMailToolError(RuntimeError):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(str(message or code))
        self.code = str(code or "agentmail_error")
        self.details = dict(details or {})


class AgentMailSendTool(AgentTool):
    name = "agentmail_send"
    description = (
        "Send an email to the configured owner with AgentMail. "
        "Supports a plain-text body, optional HTML body, and optional attachments inside the workspace or allowed roots."
    )
    parameters = {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Email subject line.",
            },
            "text_body": {
                "type": "string",
                "description": "Plain-text email body.",
            },
            "html_body": {
                "type": "string",
                "description": "Optional HTML email body.",
            },
            "attachment_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional attachment paths. Each path must resolve inside the workspace or allowed roots.",
            },
        },
        "required": ["subject", "text_body"],
    }
    label = "AgentMail Send"

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Sequence[str]] = None,
    ) -> None:
        self.workspace_root, self.allowed_roots = normalize_allowed_roots(workspace_root, allowed_roots)

    @staticmethod
    def _error(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": False, **details})

    @staticmethod
    def _ok(text: str, details: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(type="text", text=text)], details={"ok": True, **details})

    def _load_config(self) -> _AgentMailConfig:
        missing = [name for name in _REQUIRED_ENV_VARS if not _to_text(os.getenv(name))]
        if missing:
            raise _AgentMailToolError(
                "missing_configuration",
                "missing required AgentMail environment variables",
                {"missing_env_vars": missing},
            )
        if AgentMail is None:
            raise _AgentMailToolError(
                "dependency_missing",
                "agentmail package is not installed",
                {"dependency": "agentmail"},
            )

        return _AgentMailConfig(
            api_key=_to_text(os.getenv("AGENTMAIL_API_KEY")),
            inbox_id=_to_text(os.getenv("POP_AGENT_AGENTMAIL_INBOX_ID")),
            to_email=_to_text(os.getenv("POP_AGENT_AGENTMAIL_TO_EMAIL")),
        )

    def _resolve_attachment_paths(self, params: Dict[str, Any]) -> List[str]:
        raw_paths = params.get("attachment_paths", [])
        if raw_paths is None or raw_paths == "":
            return []
        if not isinstance(raw_paths, list):
            raise _AgentMailToolError(
                "invalid_attachment_paths",
                "attachment_paths must be an array of strings",
            )

        resolved_paths: List[str] = []
        for item in raw_paths:
            raw = _to_text(item)
            if not raw:
                raise _AgentMailToolError(
                    "invalid_attachment_paths",
                    "attachment_paths may not contain empty values",
                )
            try:
                resolved = _resolve_workspace_path(raw, self.workspace_root, self.allowed_roots)
            except ValueError as exc:
                raise _AgentMailToolError(
                    "path_outside_workspace",
                    "attachment path must be inside the workspace or configured allowed roots",
                    {
                        "attachment_path": raw,
                        "workspace_root": self.workspace_root,
                        "allowed_roots": self.allowed_roots,
                    },
                ) from exc
            if not os.path.isfile(resolved):
                raise _AgentMailToolError(
                    "attachment_not_found",
                    "attachment file not found",
                    {"attachment_path": resolved},
                )
            resolved_paths.append(resolved)
        return resolved_paths

    def _build_attachments(self, attachment_paths: List[str]) -> List[Dict[str, str]]:
        attachments: List[Dict[str, str]] = []
        for path in attachment_paths:
            mime_type, encoding = mimetypes.guess_type(path)
            if not mime_type or encoding:
                mime_type = "application/octet-stream"
            with open(path, "rb") as handle:
                payload = handle.read()
            attachments.append(
                {
                    "content": base64.b64encode(payload).decode("ascii"),
                    "filename": os.path.basename(path),
                    "content_type": mime_type,
                }
            )
        return attachments

    def _send_with_agentmail(
        self,
        *,
        config: _AgentMailConfig,
        subject: str,
        text_body: str,
        html_body: str,
        attachments: List[Dict[str, str]],
    ) -> Any:
        client = AgentMail(api_key=config.api_key)
        send_kwargs: Dict[str, Any] = {
            "to": config.to_email,
            "subject": subject,
            "text": text_body,
        }
        if html_body:
            send_kwargs["html"] = html_body
        if attachments:
            send_kwargs["attachments"] = attachments
        return client.inboxes.messages.send(config.inbox_id, **send_kwargs)

    def _send_impl(self, params: Dict[str, Any]) -> AgentToolResult:
        subject = _to_text(params.get("subject"))
        text_body = str(params.get("text_body") or "")
        html_body = str(params.get("html_body") or "")

        if not subject:
            return self._error(
                "agentmail_send error: missing subject",
                {"error": "missing_subject"},
            )
        if not text_body.strip():
            return self._error(
                "agentmail_send error: missing text_body",
                {"error": "missing_text_body"},
            )

        try:
            config = self._load_config()
            attachment_paths = self._resolve_attachment_paths(params)
            attachments = self._build_attachments(attachment_paths)
            sent_message = self._send_with_agentmail(
                config=config,
                subject=subject,
                text_body=text_body,
                html_body=html_body,
                attachments=attachments,
            )
        except _AgentMailToolError as exc:
            return self._error(
                f"agentmail_send error: {exc}",
                {"error": exc.code, **exc.details},
            )
        except Exception as exc:
            return self._error(
                f"agentmail_send transport error: {exc}",
                {"error": "agentmail_transport_error"},
            )

        message_id = _to_text(getattr(sent_message, "message_id", ""))
        return self._ok(
            f"agentmail_send: sent email to {config.to_email} subject={subject} attachments={len(attachment_paths)}",
            {
                "recipient": config.to_email,
                "subject": subject,
                "attachment_count": len(attachment_paths),
                "attachment_paths": attachment_paths,
                "message_id": message_id,
                "inbox_id": config.inbox_id,
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
        return self._send_impl(params)


__all__ = ["AgentMailSendTool"]
