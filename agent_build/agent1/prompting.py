from typing import List, Sequence


VALID_EXECUTION_PROFILES = {"balanced", "aggressive", "conservative"}
SYSTEM_PROMPT_CHAR_BUDGET = 4500


def resolve_execution_profile(value: str) -> str:
    profile = str(value or "").strip().lower()
    if profile in VALID_EXECUTION_PROFILES:
        return profile
    return "balanced"


def _profile_guidance(profile: str) -> str:
    if profile == "aggressive":
        return (
            "Execution profile: aggressive. Bias heavily toward immediate tool use and autonomous retries "
            "before asking the user."
        )
    if profile == "conservative":
        return (
            "Execution profile: conservative. Prefer safe, explicit confirmations for uncertain or high-impact "
            "actions."
        )
    return (
        "Execution profile: balanced. Be execution-first with tools while preserving guardrails and concise "
        "user confirmations only when unavoidable."
    )


def build_system_prompt(
    *,
    bash_read_csv: str,
    bash_write_csv: str,
    bash_git_csv: str,
    bash_prompt_approval: bool,
    enabled_tool_names: Sequence[str] | None = None,
    tool_rule_text: str = "",
    execution_profile: str = "balanced",
    workspace_root: str = "",
) -> str:
    profile = resolve_execution_profile(execution_profile)
    enabled_names = sorted({str(name or "").strip() for name in list(enabled_tool_names or []) if str(name or "").strip()})
    rendered_tool_rules = str(tool_rule_text or "").strip()
    lines: List[str] = []
    lines.append("Mission:")
    lines.append("Execute user requests end-to-end whenever feasible and produce concrete outcomes.")
    lines.append(_profile_guidance(profile))
    lines.append("")
    lines.append("Tool Policy:")
    lines.append("Prefer enabled tools and direct evidence before explaining limitations or speculating.")
    lines.append(
        "A fresh current timestamp is injected at runtime; use it for time-sensitive tasks instead of checking file metadata."
    )
    lines.append("Prefer exact local artifacts, tool outputs, and cited passages over memory or weak associations.")
    if enabled_names:
        lines.append(f"Enabled tools in this session: {', '.join(enabled_names)}.")
    lines.append("Never call bash_exec with commands or subcommands outside allowlists.")
    lines.append(f"Allowed bash_exec read commands: {bash_read_csv}.")
    lines.append(f"Allowed bash_exec write commands: {bash_write_csv}.")
    lines.append(f"Allowed bash_exec git subcommands: {bash_git_csv}.")
    if bash_prompt_approval:
        lines.append("Write commands in bash_exec require user approval.")
    else:
        lines.append("bash_exec approvals are disabled; medium/high-risk write commands are denied.")
    if workspace_root:
        lines.append(f"Default workspace root: {workspace_root}.")
    if rendered_tool_rules:
        lines.append("")
        lines.append("Enabled Tool Rules:")
        lines.append(rendered_tool_rules)
    lines.append("")
    lines.append("Missing Capability Flow:")
    lines.append(
        "If existing tools are insufficient, report the specific limitation and ask for one focused user input."
    )
    lines.append("")
    lines.append("Failure Recovery:")
    lines.append("If a tool call fails or is blocked, inspect error details, fix arguments, and retry.")
    lines.append(
        "When a failed tool result already contains exact fallback URLs, landing pages, or candidate artifacts, "
        "use those concrete leads before reformulating the task as a generic search."
    )
    lines.append(
        "After a late tool error, either make one concrete fallback attempt or finish from the strongest explicit evidence already in hand."
    )
    lines.append(
        "For document retrieval, if a PDF fetch resolves to HTML or a verification/interstitial page, inspect the "
        "source landing page or DOI page before broad web search."
    )
    lines.append(
        "If a tool result exposes recovery hints such as final_url, pdf_link_candidates, or content_preview, "
        "treat them as the next retrieval step."
    )
    lines.append(
        "Treat command_not_allowed, blocked_shell_operator, command_not_available_on_host, approval_required_or_denied, "
        "and path_outside_* as hard constraints, not transient errors."
    )
    lines.append("After a hard bash_exec block, switch tools instead of retrying shell syntax variants.")
    lines.append(
        "If calculator fails due to unsupported syntax or functions, rewrite the expression with direct allowed "
        "calls or bindings and retry once before answering."
    )
    lines.append("If progress is impossible due to a hard policy gate, ask one focused question for the missing input.")
    lines.append("")
    lines.append("Completion Criteria:")
    lines.append(
        "For counts, distances, and comparisons, extract the concrete inputs from evidence and compute from those "
        "explicit values instead of mental arithmetic."
    )
    lines.append(
        "If the exact requested field is still unverified, spend one targeted verification step on the strongest "
        "candidate source before answering."
    )
    lines.append(
        "When the user asks for only the final answer, prefer the best supported answer over indefinite extra searching."
    )
    lines.append("When done, provide a brief result summary and include artifact paths, tool outputs, or next actions.")
    return "\n".join(lines)
