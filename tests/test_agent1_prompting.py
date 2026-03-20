from agent_build.agent1.prompting import build_system_prompt, resolve_execution_profile


def _build_prompt(**overrides):
    params = {
        "bash_read_csv": "cat, ls",
        "bash_write_csv": "touch",
        "bash_git_csv": "status",
        "bash_prompt_approval": True,
        "execution_profile": "balanced",
        "workspace_root": "/tmp/workspace",
    }
    params.update(overrides)
    return build_system_prompt(**params)


def test_prompt_includes_execution_first_sections_and_allowlists():
    prompt = _build_prompt()

    assert "Mission:" in prompt
    assert "Execute user requests end-to-end whenever feasible" in prompt
    assert "Tool Policy:" in prompt
    assert "A fresh current timestamp is injected at runtime" in prompt
    assert "Use calculator for arithmetic, unit conversions, checksum logic" in prompt
    assert "prefer narrow queries with domain filters" in prompt
    assert "bash_exec runs one program without a shell" in prompt
    assert "Prefer file_read for downloaded local documents and text-like files" in prompt
    assert "Allowed bash_exec read commands: cat, ls." in prompt
    assert "Allowed bash_exec write commands: touch." in prompt
    assert "Allowed bash_exec git subcommands: status." in prompt
    assert "Never call bash_exec with commands or subcommands outside allowlists." in prompt
    assert "Use file_write for creating files, writing text, and replacing words in text files." in prompt
    assert "Use task_scheduler when the user asks to run work later or on a recurring cadence" in prompt
    assert "task_scheduler run_now marks the task as due now" in prompt
    assert "Use agentmail_send when the user asks to email the configured owner" in prompt
    assert "Failure Recovery:" in prompt
    assert "Treat command_not_allowed, blocked_shell_operator, command_not_available_on_host" in prompt
    assert "After a hard bash_exec block, switch tools instead of retrying shell syntax variants." in prompt
    assert "If search results drift to irrelevant sites" in prompt
    assert "Do not use search tools as calculators or ask them to execute code for you." in prompt
    assert "Completion Criteria:" in prompt
    assert "When the user asks for only the final answer" in prompt

def test_prompt_includes_missing_capability_fallback_guidance():
    prompt = _build_prompt()
    assert "Missing Capability Flow:" in prompt
    assert "report the specific limitation and ask for one focused user input" in prompt


def test_prompt_fallbacks_to_balanced_profile_and_denied_write_hint():
    prompt = _build_prompt(execution_profile="not-a-real-profile", bash_prompt_approval=False)

    assert resolve_execution_profile("not-a-real-profile") == "balanced"
    assert "Execution profile: balanced." in prompt
    assert "medium/high-risk write commands are denied" in prompt
