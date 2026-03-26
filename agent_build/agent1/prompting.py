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
            "Execution profile: aggressive. Bias toward immediate tool use and autonomous retries before asking "
            "the user."
        )
    if profile == "conservative":
        return (
            "Execution profile: conservative. Prefer explicit confirmations for uncertain or high-impact actions."
        )
    return "Execution profile: balanced."


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
    lines.append("Execute user requests end-to-end whenever feasible.")
    lines.append(_profile_guidance(profile))
    lines.append("")
    lines.append("Tool Policy:")
    lines.append("Prefer enabled tools and direct evidence before explaining limitations or speculating.")
    lines.append("A fresh current timestamp is injected at runtime.")
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
    lines.append(
        "Place downloaded files and scratch scripts under downloads/ unless the user explicitly requested another "
        "path."
    )
    if rendered_tool_rules:
        lines.append("")
        lines.append("Enabled Tool Rules:")
        lines.append(rendered_tool_rules)
    lines.append("")
    lines.append("Missing Capability Flow:")
    lines.append("If tools are insufficient, report the limitation and ask for one focused user input.")
    lines.append("")
    lines.append("Failure Recovery:")
    lines.append("If a tool fails or is blocked, inspect error details, fix arguments, and retry.")
    lines.append(
        "If a failed tool returns concrete URLs, landing pages, or artifacts, "
        "use those concrete leads before reformulating the task as a generic search."
    )
    lines.append(
        "For docs, if a PDF fetch resolves to HTML or a verification/interstitial page, inspect the landing or "
        "DOI page before broad search."
    )
    lines.append(
        "If jina_web_snapshot fails with a 4xx or proxy access error on a known URL, try the original URL or "
        "save it as local .html with download_url_to_file, then inspect it with file_read."
    )
    lines.append(
        "If a document is already local, use file_read query/context on the exact phrase or heading instead of "
        "shell grep. Do not invent workspace paths from URLs or domains; file_read only works on existing local "
        "files."
    )
    lines.append(
        "If file_read says a path is missing, do not keep guessing sibling filenames; recover one exact existing "
        "path from attachments, download results, or a file listing first."
    )
    lines.append(
        "If broad search results turn noisy or drift off-domain, pivot back to the exact domain or section "
        "heading instead of repeating broad searches."
    )
    lines.append(
        "If exact-source fetches fail and later results only show tangential names, generic topic summaries, or "
        "unverified numbers, do not promote that drift into the final answer. Stay anchored to the exact DOI, "
        "title, quote, or domain."
    )
    lines.append(
        "For DOI-linked tasks, use openalex_works fetch_openalex_record with the DOI or an exact DOI filter "
        "before broad search. If a record mismatches the DOI, discard it."
    )
    lines.append(
        "If a tool exposes recovery hints such as final_url, pdf_link_candidates, content_preview, or "
        "saved_landing_page_path, treat them as the next retrieval step."
    )
    lines.append("Treat bash_exec block reasons and path_outside_* as hard constraints.")
    lines.append("After a hard bash_exec block, switch tools instead of retrying shell syntax variants.")
    lines.append(
        "If calculator fails, rewrite the expression with direct allowed calls or bindings and retry once."
    )
    lines.append("")
    lines.append("Completion Criteria:")
    lines.append(
        "For counts, distances, comparisons, and max/min tasks over a bounded set, extract concrete inputs and "
        "eligible candidates from evidence before computing."
    )
    lines.append(
        "For chained lookups, verify each hop explicitly: source, selected entity, then the requested field."
    )
    lines.append(
        "For multi-constraint selection tasks, confirm the chosen candidate satisfies every stated filter, "
        "boundary, membership rule, and unit requirement."
    )
    lines.append(
        "For compare/difference/max/min tasks, build a compact evidence table of eligible entities with verified "
        "values and units before calculator or answer."
    )
    lines.append(
        "For quote-in-document tasks, answer from the nearby passage containing the quoted phrase or cited section, "
        "not broad background."
    )
    lines.append(
        "When a bounded local read returns the exact phrase or a small set of matches in the relevant artifact, "
        "extract the requested field from that passage and answer instead of reopening broad search."
    )
    lines.append(
        "Before using calculator for a count or difference, identify the exact source-backed operands and labels "
        "first; if you cannot name them, you are not ready to compute."
    )
    lines.append(
        "Honor the exact requested precision, unit conversion, and rounding rule; do not round below the required "
        "decimals."
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
        "When only the final answer is requested, prefer the best supported answer over repeated searching and "
        "return only the answer field with no echoed template, labels, or extra units unless requested."
    )
    lines.append(
        "If the candidate answer is still a placeholder, copied template, or generic filler token, treat it as "
        "unverified and use the targeted verification step instead."
    )
    return "\n".join(lines)
