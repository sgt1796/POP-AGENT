from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


_VALID_KINDS = {"tool_rules", "workflow"}
_VALID_SCOPES = {"system", "turn"}
_LOCAL_DOC_SUFFIXES = (".pdf", ".csv", ".xlsx", ".pdb", ".cif", ".mmcif")
_DOMAIN_LIKE_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b")
_LOCAL_PATH_RE = re.compile(r"(?i)(?:\b(?:[A-Z]:[\\/]|\.{1,2}[\\/]|~[\\/])\S+)")
_ATTACHMENT_RE = re.compile(r"\b(?:attachment|attached|upload|uploaded|workspace|local file|spreadsheet|document)\b")
_RESEARCH_RE = re.compile(r"\b(?:paper|doi|openalex|preprint|journal|abstract|citation|arxiv)\b")
_SCHEDULE_RE = re.compile(r"\b(?:schedule|later|tomorrow|every|cron|recurring|remind)\b")
_STRUCTURED_VERIFICATION_RE = re.compile(
    r"\b(?:compare|comparison|difference|how many|count|maximum|minimum|heaviest|lightest|"
    r"furthest|farthest|nearest|closest|shared|possible)\b"
)
_EVAL_GUIDANCE_HEADER = "Evaluation execution guidance:"
_EVAL_ATTACHMENT_HEADER = "Required attachment files are preloaded in the workspace at:"


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    kind: str
    priority: int
    tools: tuple[str, ...]
    triggers: tuple[str, ...]
    scope: str
    body: str


def _parse_scalar(value: str) -> object:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    return text


def _parse_frontmatter(frontmatter_text: str, source_path: str) -> dict:
    parsed: dict = {}
    lines = frontmatter_text.splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in raw_line:
            raise ValueError(f"{source_path}: invalid frontmatter line: {raw_line!r}")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{source_path}: frontmatter key cannot be empty")
        value_text = raw_value.strip()
        if value_text:
            parsed[key] = _parse_scalar(value_text)
            continue

        items: List[object] = []
        while index < len(lines):
            next_line = lines[index]
            next_stripped = next_line.strip()
            if not next_stripped:
                index += 1
                continue
            if next_line.lstrip().startswith("- "):
                items.append(_parse_scalar(next_line.lstrip()[2:].strip()))
                index += 1
                continue
            if next_line.startswith((" ", "\t")):
                raise ValueError(f"{source_path}: unsupported nested frontmatter value for {key!r}")
            break
        parsed[key] = items
    return parsed


def _split_frontmatter(text: str, source_path: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{source_path}: missing YAML frontmatter")
    closing_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        raise ValueError(f"{source_path}: unterminated YAML frontmatter")
    frontmatter = _parse_frontmatter("\n".join(lines[1:closing_index]), source_path)
    body = "\n".join(lines[closing_index + 1 :]).strip()
    if not body:
        raise ValueError(f"{source_path}: skill body cannot be empty")
    return frontmatter, body


def _normalize_string_list(value: object, field_name: str, source_path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"{source_path}: {field_name} must be a string list")
    normalized = []
    for item in values:
        text = str(item or "").strip().lower()
        if not text:
            raise ValueError(f"{source_path}: {field_name} items must be non-empty")
        normalized.append(text)
    return tuple(normalized)


def _parse_skill_file(path: str) -> SkillSpec:
    with open(path, "r", encoding="utf-8") as handle:
        raw_text = handle.read()
    frontmatter, body = _split_frontmatter(raw_text, path)

    name = str(frontmatter.get("name") or "").strip()
    description = str(frontmatter.get("description") or "").strip()
    kind = str(frontmatter.get("kind") or "").strip().lower()
    scope = str(frontmatter.get("scope") or "").strip().lower()
    priority_raw = frontmatter.get("priority")

    if not name:
        raise ValueError(f"{path}: name is required")
    if not description:
        raise ValueError(f"{path}: description is required")
    if kind not in _VALID_KINDS:
        raise ValueError(f"{path}: kind must be one of {sorted(_VALID_KINDS)}")
    if scope not in _VALID_SCOPES:
        raise ValueError(f"{path}: scope must be one of {sorted(_VALID_SCOPES)}")
    if kind == "tool_rules" and scope != "system":
        raise ValueError(f"{path}: tool_rules skills must use scope=system")
    if kind == "workflow" and scope != "turn":
        raise ValueError(f"{path}: workflow skills must use scope=turn")
    if not isinstance(priority_raw, int):
        raise ValueError(f"{path}: priority must be an integer")

    tools = _normalize_string_list(frontmatter.get("tools", []), "tools", path)
    triggers = _normalize_string_list(frontmatter.get("triggers", []), "triggers", path)
    if not tools:
        raise ValueError(f"{path}: tools must not be empty")

    return SkillSpec(
        name=name,
        description=description,
        kind=kind,
        priority=int(priority_raw),
        tools=tools,
        triggers=triggers,
        scope=scope,
        body=body,
    )


def _sorted_skills(skills: Iterable[SkillSpec]) -> list[SkillSpec]:
    return sorted(skills, key=lambda skill: (-skill.priority, skill.name))


def _skill_is_enabled(skill: SkillSpec, enabled_tool_names: Sequence[str]) -> bool:
    enabled = {str(name or "").strip().lower() for name in enabled_tool_names if str(name or "").strip()}
    return bool(enabled.intersection(skill.tools))


def _contains_any(text: str, items: Sequence[str]) -> bool:
    return any(item in text for item in items)


def _matches_domain_like_text(text: str) -> bool:
    for match in _DOMAIN_LIKE_RE.findall(text):
        if match.endswith((".pdf", ".csv", ".xlsx", ".json", ".txt", ".md", ".py", ".yaml", ".yml", ".pdb", ".cif", ".mmcif")):
            continue
        return True
    return False


def _matches_structural_cues(skill: SkillSpec, message: str) -> bool:
    lowered = message.lower()
    if skill.name == "local-document-evidence":
        return (
            _contains_any(lowered, _LOCAL_DOC_SUFFIXES)
            or _LOCAL_PATH_RE.search(message) is not None
            or _ATTACHMENT_RE.search(lowered) is not None
        )
    if skill.name == "web-verification":
        return (
            "http://" in lowered
            or "https://" in lowered
            or _matches_domain_like_text(lowered)
            or any(term in lowered for term in ("website", "source", "verify"))
        )
    if skill.name == "paper-pdf-retrieval":
        return _RESEARCH_RE.search(lowered) is not None or (".pdf" in lowered and _RESEARCH_RE.search(lowered) is not None)
    if skill.name == "structured-answer-verification":
        return _STRUCTURED_VERIFICATION_RE.search(lowered) is not None
    if skill.name == "scheduled-reporting":
        return _SCHEDULE_RE.search(lowered) is not None or "email me" in lowered
    return False


def _strip_eval_meta_guidance(message: str) -> str:
    text = str(message or "").strip()
    if not text or _EVAL_GUIDANCE_HEADER not in text:
        return text

    before, after = text.split(_EVAL_GUIDANCE_HEADER, 1)
    retained_sections = [before.strip()]
    if _EVAL_ATTACHMENT_HEADER in after:
        _, attachment_section = after.split(_EVAL_ATTACHMENT_HEADER, 1)
        retained_sections.append(f"{_EVAL_ATTACHMENT_HEADER}{attachment_section}".strip())
    return "\n\n".join(section for section in retained_sections if section).strip()


def load_skills(skill_root: str) -> list[SkillSpec]:
    root = os.path.realpath(skill_root)
    if not os.path.isdir(root):
        return []

    seen_names = set()
    loaded: List[SkillSpec] = []
    for entry in sorted(os.scandir(root), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        skill_path = os.path.join(entry.path, "SKILL.md")
        if not os.path.isfile(skill_path):
            continue
        skill = _parse_skill_file(skill_path)
        if skill.name in seen_names:
            raise ValueError(f"duplicate skill name: {skill.name}")
        seen_names.add(skill.name)
        loaded.append(skill)
    return _sorted_skills(loaded)


def select_system_skills(enabled_tool_names: Sequence[str], skills: Sequence[SkillSpec]) -> list[SkillSpec]:
    selected = [
        skill
        for skill in skills
        if skill.kind == "tool_rules" and skill.scope == "system" and _skill_is_enabled(skill, enabled_tool_names)
    ]
    return _sorted_skills(selected)


def select_turn_skills(user_message: str, enabled_tool_names: Sequence[str], skills: Sequence[SkillSpec]) -> list[SkillSpec]:
    selection_text = _strip_eval_meta_guidance(user_message)
    lowered = selection_text.lower()
    if not lowered:
        return []

    selected = []
    for skill in skills:
        if skill.kind != "workflow" or skill.scope != "turn":
            continue
        if not _skill_is_enabled(skill, enabled_tool_names):
            continue
        if _contains_any(lowered, skill.triggers) or _matches_structural_cues(skill, selection_text):
            selected.append(skill)
    return _sorted_skills(selected)[:2]


def render_skill_text(skills: Sequence[SkillSpec]) -> str:
    blocks = []
    for skill in skills:
        body = str(skill.body or "").strip()
        if not body:
            continue
        blocks.append(f"[{skill.name}]\n{body}")
    return "\n\n".join(blocks).strip()


__all__ = [
    "SkillSpec",
    "load_skills",
    "render_skill_text",
    "select_system_skills",
    "select_turn_skills",
]
