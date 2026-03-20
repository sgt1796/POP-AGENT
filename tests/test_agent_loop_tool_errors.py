from agent.agent_loop import _tool_result_indicates_error
from agent.agent_types import AgentToolResult, TextContent


def _result(details):
    return AgentToolResult(content=[TextContent(type="text", text="x")], details=details)


def test_tool_result_indicates_error_for_ok_false():
    assert _tool_result_indicates_error(_result({"ok": False})) is True


def test_tool_result_indicates_error_for_blocked_tool():
    assert _tool_result_indicates_error(_result({"blocked": True, "block_reason": "command_not_allowed"})) is True


def test_tool_result_indicates_error_for_error_detail():
    assert _tool_result_indicates_error(_result({"error": "dependency_missing"})) is True


def test_tool_result_indicates_error_ignores_success_details():
    assert _tool_result_indicates_error(_result({"ok": True, "count": 2})) is False
