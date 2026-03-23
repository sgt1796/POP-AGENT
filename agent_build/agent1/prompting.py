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
        "Execution profile: balanced. Be execution-first with tools while keeping guardrails."
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
        "A fresh current timestamp is injected at runtime; use it for time-sensitive tasks."
    )
    lines.append("Prefer exact local artifacts, tool outputs, and cited passages over memory.")
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
        "If existing tools are insufficient, report the limitation and ask for one focused user input."
    )
    lines.append("")
    lines.append("Failure Recovery:")
    lines.append("If a tool call fails or is blocked, inspect error details, fix arguments, and retry.")
    lines.append(
        "When a failed tool result contains concrete fallback URLs, landing pages, or artifacts, "
        "use those concrete leads before reformulating the task as a generic search."
    )
    lines.append(
        "After a late tool error, make one fallback attempt or finish from the strongest evidence in hand."
    )
    lines.append(
        "For document retrieval, if a PDF fetch resolves to HTML or a verification/interstitial page, inspect the "
        "landing or DOI page before broad search."
    )
    lines.append(
        "If jina_web_snapshot fails with a 4xx or proxy access error on a known page URL, try the original URL "
        "directly or save it as local .html with download_url_to_file, then inspect it with file_read."
    )
    lines.append(
        "If a relevant document is already local, use file_read query/context on the exact phrase or heading "
        "instead of shell grep."
    )
    lines.append(
        "If broad search results turn noisy or drift off-domain, pivot back to the exact domain or section heading "
        "instead of repeating broad searches."
    )
    lines.append(
        "If an exact-source fetch fails and later results only surface tangential names, generic topic summaries, "
        "or unverified numbers, do not promote that drift into the final answer. Stay anchored to the exact DOI, "
        "title, quote, or domain."
    )
    lines.append(
        "If a tool result exposes recovery hints such as final_url, pdf_link_candidates, content_preview, or "
        "saved_landing_page_path, treat them as the next retrieval step."
    )
    lines.append(
        "Treat command_not_allowed, blocked_shell_operator, command_not_available_on_host, approval_required_or_denied, "
        "and path_outside_* as hard constraints."
    )
    lines.append("After a hard bash_exec block, switch tools instead of retrying shell syntax variants.")
    lines.append(
        "If calculator fails, rewrite the expression with direct allowed calls or bindings and retry once."
    )
    lines.append("If progress is impossible due to a hard policy gate, ask one focused question.")
    lines.append("")
    lines.append("Completion Criteria:")
    lines.append(
        "For counts, distances, comparisons, and max/min selection over a bounded set, extract the concrete inputs "
        "and eligible candidates from evidence before computing."
    )
    lines.append(
        "For chained lookups, verify each hop explicitly: source list or page, selected entity, then the requested "
        "field."
    )
    lines.append(
        "For multi-constraint selection tasks, confirm the chosen candidate satisfies every stated filter, boundary, "
        "membership rule, and unit requirement before answering."
    )
    lines.append(
        "For compare/difference/max/min tasks, build a compact evidence table of eligible entities with verified "
        "values and units before using calculator or answering."
    )
    lines.append(
        "For quote-in-document tasks, answer from the nearby passage containing the quoted phrase or cited section, "
        "not from broad background about the same topic."
    )
    lines.append(
        "Before using calculator for a count or difference, identify the exact source-backed operands and labels you "
        "are combining; if you cannot name them, you are not ready to compute."
    )
    lines.append(
        "Honor the exact requested precision, unit conversion, and rounding rule; do not round to fewer decimals "
        "than required."
    )
    lines.append(
        "Translate requested precision into output decimals; nearest 0.001 of the reported unit requires three "
        "decimals."
    )
    lines.append(
        "If the exact requested field is still unverified, spend one targeted verification step on the strongest "
        "candidate source before answering."
    )
    lines.append(
        "When only the final answer is requested, prefer the best supported answer over repeated searching, and "
        "return only the answer field with no echoed template, labels, or extra units unless requested."
    )
    lines.append(
        "If the candidate answer is still a placeholder, copied template, or generic filler token, treat it as "
        "unverified and spend the targeted verification step instead."
    )
    lines.append("When done, provide a brief result summary with artifacts or next actions.")
    return "\n".join(lines)
