from agent_build.agent1.event_logger import make_event_logger, resolve_log_level


def test_resolve_log_level_accepts_legacy_aliases():
    assert resolve_log_level("messages") == resolve_log_level("simple")
    assert resolve_log_level("stream") == resolve_log_level("full")


def test_quiet_level_suppresses_bash_exec_output(capsys):
    logger = make_event_logger("quiet")
    logger({"type": "tool_execution_end", "toolName": "bash_exec", "args": {"cmd": "ls"}, "isError": False})
    captured = capsys.readouterr()
    assert captured.out == ""


def test_simple_level_logs_tool_and_bash_preview(capsys):
    logger = make_event_logger("simple")
    logger({"type": "tool_execution_start", "toolName": "bash_exec", "args": {"cmd": "python script.py --flag --value 1"}})
    logger({"type": "tool_execution_end", "toolName": "websnapshot", "isError": False})
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured[0] == "[tool:start] bash_exec cmd=python script.py --flag --value 1"
    assert captured[1] == "[tool:end] websnapshot error=False"
