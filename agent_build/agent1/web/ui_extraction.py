from __future__ import annotations

import re

from .schemas import PlanChecklistItem, PlanChecklistProps, PlanChecklistSpec, ResultTableProps, ResultTableSpec


_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)(.+)$")
_CHECKBOX_RE = re.compile(r"^\s*(?:[-*+]\s+)?\[(?P<mark>[ xX~!])\]\s*(?P<label>.+)$")
_STATUS_PREFIX_RE = re.compile(
    r"^(?P<status>done|completed|pending|blocked|in progress|in_progress|working)\s*[:\-]\s*(?P<label>.+)$",
    re.IGNORECASE,
)
_CHECKLIST_TITLE_RE = re.compile(
    r"\b(plan|checklist|todo|to-do|next steps?|action items?|tasks?)\b",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_INLINE_PATTERNS = (
    (re.compile(r"`([^`]+)`"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"__([^_]+)__"), r"\1"),
    (re.compile(r"~~([^~]+)~~"), r"\1"),
    (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"\1"),
    (re.compile(r"(?<!_)_([^_]+)_(?!_)"), r"\1"),
)
_TASK_START_VERBS = {
    "add",
    "adjust",
    "audit",
    "build",
    "check",
    "clean",
    "confirm",
    "create",
    "debug",
    "deploy",
    "design",
    "document",
    "draft",
    "enable",
    "ensure",
    "fix",
    "gather",
    "implement",
    "improve",
    "inspect",
    "install",
    "investigate",
    "migrate",
    "monitor",
    "move",
    "open",
    "prepare",
    "refactor",
    "remove",
    "rename",
    "replace",
    "research",
    "review",
    "run",
    "ship",
    "summarize",
    "test",
    "update",
    "validate",
    "verify",
    "write",
}


def extract_structured_ui(text: str) -> ResultTableSpec | PlanChecklistSpec | None:
    value = str(text or "").strip()
    if not value:
        return None
    return _extract_result_table(value) or _extract_plan_checklist(value)


def _extract_result_table(text: str) -> ResultTableSpec | None:
    lines = text.splitlines()
    for index in range(len(lines) - 1):
        header_line = lines[index].strip()
        separator_line = lines[index + 1].strip()
        if "|" not in header_line:
            continue
        if not _TABLE_SEPARATOR_RE.match(separator_line):
            continue
        row_lines: list[str] = []
        cursor = index + 2
        while cursor < len(lines):
            candidate = lines[cursor].strip()
            if "|" not in candidate or not candidate:
                break
            row_lines.append(candidate)
            cursor += 1
        columns = _split_markdown_row(header_line)
        rows = [_split_markdown_row(row) for row in row_lines]
        if not columns or not rows:
            continue
        normalized_rows = []
        for row in rows:
            if len(row) < len(columns):
                row = row + [""] * (len(columns) - len(row))
            normalized_rows.append(row[: len(columns)])
        title = _find_nearest_title(lines, index, fallback="Results")
        return ResultTableSpec(
            props=ResultTableProps(
                title=title,
                columns=columns,
                rows=normalized_rows,
            )
        )
    return None


def _extract_plan_checklist(text: str) -> PlanChecklistSpec | None:
    lines = text.splitlines()
    items: list[PlanChecklistItem] = []
    list_start: int | None = None
    explicit_statuses = False

    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        checkbox_match = _CHECKBOX_RE.match(stripped)
        if checkbox_match:
            explicit_statuses = True
            if list_start is None:
                list_start = index
            items.append(
                PlanChecklistItem(
                    label=_clean_label(checkbox_match.group("label")),
                    status=_checkbox_status(checkbox_match.group("mark")),
                )
            )
            continue
        list_match = _LIST_LINE_RE.match(stripped)
        if list_match is None:
            continue
        status, label = _parse_status_prefix(list_match.group(1))
        if list_start is None:
            list_start = index
        explicit_statuses = explicit_statuses or status != "pending"
        items.append(PlanChecklistItem(label=label, status=status))

    if len(items) < 2:
        return None

    title = _find_nearest_title(lines, list_start or 0, fallback="Execution Plan")
    title_has_checklist_signal = _looks_like_checklist_title(title)
    if not explicit_statuses and not (
        title_has_checklist_signal or _looks_like_task_list([item.label for item in items])
    ):
        return None

    return PlanChecklistSpec(
        props=PlanChecklistProps(
            title=title,
            items=items,
        )
    )


def _split_markdown_row(line: str) -> list[str]:
    trimmed = line.strip().strip("|")
    if not trimmed:
        return []
    return [cell.strip() for cell in trimmed.split("|")]


def _find_nearest_title(lines: list[str], start_index: int, *, fallback: str) -> str:
    cursor = max(0, start_index - 1)
    while cursor >= 0:
        candidate = _normalize_markdown_text(lines[cursor]).strip(":")
        if candidate and "|" not in candidate and len(candidate) <= 120:
            return candidate
        cursor -= 1
    return fallback


def _checkbox_status(mark: str) -> str:
    normalized = (mark or "").strip().lower()
    if normalized == "x":
        return "done"
    if normalized in {"~", "!"}:
        return "in_progress"
    return "pending"


def _parse_status_prefix(value: str) -> tuple[str, str]:
    match = _STATUS_PREFIX_RE.match(value.strip())
    if match is None:
        return "pending", _clean_label(value)
    raw_status = str(match.group("status") or "").strip().lower().replace(" ", "_")
    normalized = {
        "done": "done",
        "completed": "done",
        "pending": "pending",
        "blocked": "blocked",
        "working": "in_progress",
        "in_progress": "in_progress",
    }.get(raw_status, "pending")
    return normalized, _clean_label(match.group("label"))


def _clean_label(value: str) -> str:
    normalized = _normalize_markdown_text(value).strip().strip("-")
    return " ".join(normalized.split())


def _normalize_markdown_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"^\s*>\s?", "", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    for pattern, replacement in _MARKDOWN_INLINE_PATTERNS:
        text = pattern.sub(replacement, text)
    return " ".join(text.split())


def _looks_like_checklist_title(value: str) -> bool:
    return bool(_CHECKLIST_TITLE_RE.search(_normalize_markdown_text(value)))


def _looks_like_task_list(labels: list[str]) -> bool:
    if len(labels) < 2:
        return False
    task_like_count = sum(1 for label in labels if _looks_like_task_label(label))
    required = max(2, (len(labels) + 1) // 2)
    return task_like_count >= required


def _looks_like_task_label(value: str) -> bool:
    label = _clean_label(value)
    if not label:
        return False
    if len(label) > 80:
        return False
    if label.endswith((".", "!", "?")):
        return False
    if ":" in label:
        return False
    words = label.split()
    if len(words) > 10:
        return False
    first = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", words[0].lower())
    return first in _TASK_START_VERBS
