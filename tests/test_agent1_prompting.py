from agent_build.agent1.prompting import (
    SYSTEM_PROMPT_CHAR_BUDGET,
    build_system_prompt,
    resolve_execution_profile,
)


def _build_prompt(**overrides):
    params = {
        "bash_read_csv": "cat, ls",
        "bash_write_csv": "touch",
        "bash_git_csv": "status",
        "bash_prompt_approval": True,
        "enabled_tool_names": ["bash_exec", "calculator", "file_read"],
        "tool_rule_text": "[calculator-rules]\nUse calculator for arithmetic first.",
        "execution_profile": "balanced",
        "workspace_root": "/tmp/workspace",
    }
    params.update(overrides)
    return build_system_prompt(**params)


def test_prompt_includes_core_sections_allowlists_and_tool_rules():
    prompt = _build_prompt()

    assert "Mission:" in prompt
    assert "Execute user requests end-to-end whenever feasible" in prompt
    assert "Execution profile: balanced." in prompt
    assert "Tool Policy:" in prompt
    assert "Prefer enabled tools and direct evidence before explaining limitations or speculating." in prompt
    assert "A fresh current timestamp is injected at runtime" in prompt
    assert "Enabled tools in this session: bash_exec, calculator, file_read." in prompt
    assert "Allowed bash_exec read commands: cat, ls." in prompt
    assert "Allowed bash_exec write commands: touch." in prompt
    assert "Allowed bash_exec git subcommands: status." in prompt
    assert "Enabled Tool Rules:" in prompt
    assert "[calculator-rules]" in prompt
    assert "Missing Capability Flow:" in prompt
    assert "Failure Recovery:" in prompt
    assert "use those concrete leads before reformulating the task as a generic search" in prompt
    assert "if a PDF fetch resolves to HTML or a verification/interstitial page" in prompt
    assert "final_url, pdf_link_candidates, or content_preview" in prompt
    assert "rewrite the expression with direct allowed calls or bindings" in prompt
    assert "Completion Criteria:" in prompt
    assert "eligible candidates from evidence before computing" in prompt
    assert "spend one targeted verification step" in prompt
    assert "no echoed template, labels, or extra units" in prompt
    assert "placeholder, copied template, or generic filler token" in prompt


def test_prompt_excludes_workflow_playbooks_and_stays_under_budget():
    prompt = _build_prompt(tool_rule_text="")

    assert len(prompt) < SYSTEM_PROMPT_CHAR_BUDGET
    assert "best_oa_pdf_url" not in prompt
    assert "search-result snippets alone" not in prompt
    assert "task_scheduler run_now marks the task as due now" not in prompt
    assert "local scientific text files (.pdb, .cif, .mmcif) as primary evidence" not in prompt
    assert "set/count logic" not in prompt


def test_prompt_fallbacks_to_balanced_profile_and_denied_write_hint():
    prompt = _build_prompt(execution_profile="not-a-real-profile", bash_prompt_approval=False)

    assert resolve_execution_profile("not-a-real-profile") == "balanced"
    assert "Execution profile: balanced." in prompt
    assert "medium/high-risk write commands are denied" in prompt
