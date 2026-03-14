from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from agent.tools import AgentMailSendTool


AGENTMAIL_ENV_VARS = [
    "AGENTMAIL_API_KEY",
    "POP_AGENT_AGENTMAIL_INBOX_ID",
    "POP_AGENT_AGENTMAIL_TO_EMAIL",
]


class _FakeMessagesAPI:
    def __init__(self, owner: "_FakeAgentMail") -> None:
        self.owner = owner

    def send(self, inbox_id: str, **kwargs: Any) -> Any:
        call = {"inbox_id": inbox_id, **kwargs}
        self.owner.send_calls.append(call)
        if self.owner.raise_on_send is not None:
            raise self.owner.raise_on_send
        return self.owner.sent_message


class _FakeAgentMail:
    instances = []
    raise_on_send = None
    sent_message = SimpleNamespace(message_id="<msg-123@agentmail.test>")

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.send_calls = []
        self.inboxes = SimpleNamespace(messages=_FakeMessagesAPI(self))
        type(self).instances.append(self)


def _run(tool: AgentMailSendTool, params: Dict[str, Any]):
    return asyncio.run(tool.execute("tc1", params))


def _clear_agentmail_env(monkeypatch) -> None:
    for name in AGENTMAIL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _set_required_env(monkeypatch) -> None:
    _clear_agentmail_env(monkeypatch)
    monkeypatch.setenv("AGENTMAIL_API_KEY", "am_test_123")
    monkeypatch.setenv("POP_AGENT_AGENTMAIL_INBOX_ID", "inb_test_123")
    monkeypatch.setenv("POP_AGENT_AGENTMAIL_TO_EMAIL", "owner@example.com")


def test_agentmail_send_returns_error_when_env_missing(tmp_path: Path, monkeypatch):
    _clear_agentmail_env(monkeypatch)
    monkeypatch.setattr("agent.tools.agentmail_tool.AgentMail", _FakeAgentMail)

    tool = AgentMailSendTool(workspace_root=str(tmp_path))
    result = _run(tool, {"subject": "Status", "text_body": "All good"})

    assert result.details["ok"] is False
    assert result.details["error"] == "missing_configuration"
    assert "AGENTMAIL_API_KEY" in result.details["missing_env_vars"]
    assert "POP_AGENT_AGENTMAIL_TO_EMAIL" in result.details["missing_env_vars"]


def test_agentmail_send_returns_error_when_dependency_missing(tmp_path: Path, monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setattr("agent.tools.agentmail_tool.AgentMail", None)

    tool = AgentMailSendTool(workspace_root=str(tmp_path))
    result = _run(tool, {"subject": "Status", "text_body": "All good"})

    assert result.details["ok"] is False
    assert result.details["error"] == "dependency_missing"
    assert result.details["dependency"] == "agentmail"


def test_agentmail_send_uses_sdk_client_and_returns_message_id(tmp_path: Path, monkeypatch):
    _set_required_env(monkeypatch)
    _FakeAgentMail.instances = []
    _FakeAgentMail.raise_on_send = None
    monkeypatch.setattr("agent.tools.agentmail_tool.AgentMail", _FakeAgentMail)

    tool = AgentMailSendTool(workspace_root=str(tmp_path))
    result = _run(tool, {"subject": "Nightly report", "text_body": "Finished cleanly."})

    assert result.details["ok"] is True
    assert result.details["recipient"] == "owner@example.com"
    assert result.details["message_id"] == "<msg-123@agentmail.test>"
    assert result.details["inbox_id"] == "inb_test_123"
    assert result.details["attachment_count"] == 0
    assert len(_FakeAgentMail.instances) == 1
    client = _FakeAgentMail.instances[0]
    assert client.api_key == "am_test_123"
    assert client.send_calls == [
        {
            "inbox_id": "inb_test_123",
            "to": "owner@example.com",
            "subject": "Nightly report",
            "text": "Finished cleanly.",
        }
    ]


def test_agentmail_send_builds_html_and_base64_attachments(tmp_path: Path, monkeypatch):
    _set_required_env(monkeypatch)
    attachment_path = tmp_path / "report.pdf"
    attachment_path.write_bytes(b"%PDF-1.4 fake pdf")
    _FakeAgentMail.instances = []
    _FakeAgentMail.raise_on_send = None
    monkeypatch.setattr("agent.tools.agentmail_tool.AgentMail", _FakeAgentMail)

    tool = AgentMailSendTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "subject": "Work report",
            "text_body": "Plain summary",
            "html_body": "<p><strong>HTML summary</strong></p>",
            "attachment_paths": ["report.pdf"],
        },
    )

    assert result.details["ok"] is True
    assert result.details["attachment_count"] == 1
    assert result.details["attachment_paths"] == [str(attachment_path.resolve())]
    call = _FakeAgentMail.instances[0].send_calls[0]
    assert call["html"] == "<p><strong>HTML summary</strong></p>"
    assert len(call["attachments"]) == 1
    attachment = call["attachments"][0]
    assert attachment["filename"] == "report.pdf"
    assert attachment["content_type"] == "application/pdf"
    assert base64.b64decode(attachment["content"]) == b"%PDF-1.4 fake pdf"


def test_agentmail_send_allows_attachment_in_allowed_roots(tmp_path: Path, monkeypatch):
    _set_required_env(monkeypatch)
    external = tmp_path / "external"
    external.mkdir()
    attachment_path = external / "report.txt"
    attachment_path.write_text("summary", encoding="utf-8")
    _FakeAgentMail.instances = []
    _FakeAgentMail.raise_on_send = None
    monkeypatch.setattr("agent.tools.agentmail_tool.AgentMail", _FakeAgentMail)

    tool = AgentMailSendTool(workspace_root=str(tmp_path), allowed_roots=[str(external)])
    result = _run(
        tool,
        {
            "subject": "Work report",
            "text_body": "Plain summary",
            "attachment_paths": [str(attachment_path)],
        },
    )

    assert result.details["ok"] is True
    assert result.details["attachment_paths"] == [str(attachment_path.resolve())]


def test_agentmail_send_rejects_attachment_outside_workspace(tmp_path: Path, monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setattr("agent.tools.agentmail_tool.AgentMail", _FakeAgentMail)

    tool = AgentMailSendTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "subject": "Work report",
            "text_body": "Body",
            "attachment_paths": ["/tmp/outside.txt"],
        },
    )

    assert result.details["ok"] is False
    assert result.details["error"] == "path_outside_workspace"


def test_agentmail_send_rejects_missing_attachment_file(tmp_path: Path, monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setattr("agent.tools.agentmail_tool.AgentMail", _FakeAgentMail)

    tool = AgentMailSendTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "subject": "Work report",
            "text_body": "Body",
            "attachment_paths": ["missing.txt"],
        },
    )

    assert result.details["ok"] is False
    assert result.details["error"] == "attachment_not_found"


def test_agentmail_send_returns_transport_error_on_sdk_failure(tmp_path: Path, monkeypatch):
    _set_required_env(monkeypatch)
    _FakeAgentMail.instances = []
    _FakeAgentMail.raise_on_send = RuntimeError("boom")
    monkeypatch.setattr("agent.tools.agentmail_tool.AgentMail", _FakeAgentMail)

    tool = AgentMailSendTool(workspace_root=str(tmp_path))
    result = _run(tool, {"subject": "Status", "text_body": "Body"})

    assert result.details["ok"] is False
    assert result.details["error"] == "agentmail_transport_error"
