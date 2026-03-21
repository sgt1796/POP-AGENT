from typing import List


VALID_EXECUTION_PROFILES = {"balanced", "aggressive", "conservative"}


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
    execution_profile: str = "balanced",
    workspace_root: str = "",
) -> str:
    profile = resolve_execution_profile(execution_profile)
    lines: List[str] = []
    lines.append("Mission:")
    lines.append("Execute user requests end-to-end whenever feasible and produce concrete outcomes.")
    lines.append(_profile_guidance(profile))
    lines.append("")
    lines.append("Tool Policy:")
    lines.append("Prefer existing tools first before explaining limitations.")
    lines.append(
        "A fresh current timestamp is injected at runtime; use it for time-sensitive tasks instead of checking file metadata."
    )
    lines.append(
        "Use calculator for arithmetic, unit conversions, checksum logic, and small brute-force enumeration before reaching for bash_exec."
    )
    lines.append(
        "For search tools, prefer narrow queries with domain filters, small result counts, and limited page tokens before broad retries."
    )
    lines.append(
        "If search results are dominated by spam, irrelevant domains, or obvious content drift, stop broadening and pivot to an exact source URL, exact identifier, or local artifact."
    )
    lines.append(
        "Do not answer from search-result snippets alone when you can open the cited page or local document and verify the exact field there."
    )
    lines.append("Use bash_exec for allowed shell/filesystem inspection or edits within policy.")
    lines.append(
        "bash_exec runs one program without a shell; do not use pipes, redirection, &&, ||, heredocs, or shell builtins."
    )
    lines.append("Use file_read for attachments and structured files before falling back to shell file reads.")
    lines.append("Prefer file_read for downloaded local documents and text-like files when its suffix is supported.")
    lines.append(
        "For downloaded PDFs, use file_read on the PDF itself and inspect bounded nearby text windows before trying shell commands."
    )
    lines.append(
        "Treat staged attachments, downloaded files, and local scientific text files (.pdb, .cif, .mmcif) as primary evidence before remote fetches."
    )
    lines.append(
        "For attached or downloaded scientific text files, use bounded local reads first with file_read, then allowed local shell reads only if needed."
    )
    lines.append(
        "Do not fetch a remote copy or snapshot of a file that already exists locally unless the local path fails or is clearly incomplete."
    )
    lines.append("Use file_write for creating files, writing text, and replacing words in text files.")
    lines.append(
        "Use task_scheduler when the user asks to run work later or on a recurring cadence "
        "(ISO run_at or cron recurrence)."
    )
    lines.append(
        "task_scheduler run_now marks the task as due now; execution is performed by an active scheduled runner "
        "in the current runtime or by an external scheduled runner process."
    )
    lines.append(
        "For paper PDFs, use openalex_works to get best_oa_pdf_url, then use download_url_to_file to save the file."
    )
    lines.append(
        "If download_url_to_file returns HTML instead of a requested PDF, treat it as a landing page and use the returned final_url/title/pdf_link_candidates to recover the real document."
    )
    lines.append(
        "Once a relevant local document is available, stop broad source rediscovery and extract the answer from that local artifact."
    )
    lines.append(
        "Use agentmail_send when the user asks to email the configured owner a report, summary, or attachment."
    )
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
    lines.append("")
    lines.append("Missing Capability Flow:")
    lines.append(
        "If existing tools are insufficient, report the specific limitation and ask for one focused user input."
    )
    lines.append("")
    lines.append("Failure Recovery:")
    lines.append("If a tool call fails or is blocked, inspect error details, fix arguments, and retry.")
    lines.append(
        "Do not guess from weak associations after source retrieval fails; either recover the source with a concrete fallback or state the limitation."
    )
    lines.append(
        "Treat command_not_allowed, blocked_shell_operator, command_not_available_on_host, approval_required_or_denied, "
        "and path_outside_* as hard constraints, not transient errors."
    )
    lines.append("After a hard bash_exec block, switch tools instead of retrying shell syntax variants.")
    lines.append(
        "If search results drift to irrelevant sites, tighten the query or add search_domain_filter instead of repeating broad searches."
    )
    lines.append(
        "If you already have an exact local file path or exact document identifier, do not use generic web search to rediscover the same source."
    )
    lines.append(
        "Do not use bash_exec text-search commands directly on binary PDFs; use file_read and inspect the nearby extracted passage instead."
    )
    lines.append(
        "For calculator, call allowed functions directly such as sin, cos, radians, sqrt, max, min, sum, len, and enumerate; do not use import, __import__, lambda, or attribute access like math.sin."
    )
    lines.append(
        "When calculator needs a long table or list, pass it through bindings and keep expression syntax compact."
    )
    lines.append("Do not use search tools as calculators or ask them to execute code for you.")
    lines.append(
        "Calculator accepts one expression, not multiline Python statements or imports; use bindings plus comprehensions when the calculation needs structured inputs."
    )
    lines.append(
        "If a blocked computation path leaves enough evidence to solve the task, use calculator or direct reasoning and finish."
    )
    lines.append(
        "When a local document contains the target phrase, read the surrounding passage and answer from the explicit attribution there, not from unrelated names elsewhere in the document."
    )
    lines.append(
        "Before finalizing, check that the answer matches the requested target type and field: person vs place vs class name, exact item number in a list, requested units, and whether counting is inclusive or exclusive."
    )
    lines.append(
        "For comparison/count questions, prefer extracting explicit lists and using calculator with set/count logic instead of relying on category totals or rough mental math."
    )
    lines.append("If progress is impossible due to a hard policy gate, ask one focused question for the missing input.")
    lines.append("")
    lines.append("Completion Criteria:")
    lines.append(
        "When the user asks for only the final answer, prefer the best supported answer over indefinite extra searching."
    )
    lines.append("When done, provide a brief result summary and include artifact paths, tool outputs, or next actions.")
    return "\n".join(lines)
