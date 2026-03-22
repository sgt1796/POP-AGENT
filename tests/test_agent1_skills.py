from pathlib import Path

import pytest

from agent_build.agent1.skill_registry import (
    SkillSpec,
    load_skills,
    render_skill_text,
    select_system_skills,
    select_turn_skills,
)


def test_load_skills_parses_repo_skill_files():
    skill_root = Path(__file__).resolve().parents[1] / "agent_build" / "agent1" / "skills"

    skills = load_skills(str(skill_root))

    assert {skill.name for skill in skills} == {
        "bash-exec-rules",
        "calculator-rules",
        "file-io-rules",
        "web-retrieval-rules",
        "scheduler-mail-rules",
        "local-document-evidence",
        "web-verification",
        "paper-pdf-retrieval",
        "structured-answer-verification",
        "scheduled-reporting",
    }


def test_load_skills_rejects_duplicate_names(tmp_path):
    body = """---
name: duplicate-skill
description: Demo.
kind: tool_rules
priority: 1
tools:
  - bash_exec
triggers:
  - bash
scope: system
---
Skill body.
"""
    (tmp_path / "one").mkdir()
    (tmp_path / "one" / "SKILL.md").write_text(body, encoding="utf-8")
    (tmp_path / "two").mkdir()
    (tmp_path / "two" / "SKILL.md").write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate skill name"):
        load_skills(str(tmp_path))


def test_load_skills_rejects_invalid_frontmatter(tmp_path):
    broken = """---
name: broken
description: Broken.
kind: invalid
priority: 1
tools:
  - bash_exec
triggers:
  - bash
scope: system
---
Skill body.
"""
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text(broken, encoding="utf-8")

    with pytest.raises(ValueError, match="kind must be one of"):
        load_skills(str(tmp_path))


def test_select_system_skills_ignores_disabled_tools():
    skills = [
        SkillSpec(
            name="bash-exec-rules",
            description="bash rules",
            kind="tool_rules",
            priority=10,
            tools=("bash_exec",),
            triggers=("bash",),
            scope="system",
            body="Use bash carefully.",
        ),
        SkillSpec(
            name="calculator-rules",
            description="calc rules",
            kind="tool_rules",
            priority=20,
            tools=("calculator",),
            triggers=("calc",),
            scope="system",
            body="Use calculator first.",
        ),
    ]

    selected = select_system_skills(["calculator"], skills)

    assert [skill.name for skill in selected] == ["calculator-rules"]


def test_select_turn_skills_matches_structural_cues_and_caps_results():
    skills = [
        SkillSpec(
            name="local-document-evidence",
            description="doc workflow",
            kind="workflow",
            priority=100,
            tools=("file_read",),
            triggers=("document",),
            scope="turn",
            body="Read local files first.",
        ),
        SkillSpec(
            name="paper-pdf-retrieval",
            description="paper workflow",
            kind="workflow",
            priority=95,
            tools=("openalex_works",),
            triggers=("paper", "doi"),
            scope="turn",
            body="Use OpenAlex first.",
        ),
        SkillSpec(
            name="web-verification",
            description="web workflow",
            kind="workflow",
            priority=90,
            tools=("jina_web_snapshot",),
            triggers=("verify",),
            scope="turn",
            body="Verify from sources.",
        ),
    ]

    selected = select_turn_skills(
        "Verify this DOI from https://example.org and check the attached report.pdf",
        ["file_read", "openalex_works", "jina_web_snapshot"],
        skills,
    )

    assert [skill.name for skill in selected] == [
        "local-document-evidence",
        "paper-pdf-retrieval",
    ]


def test_select_turn_skills_ignores_eval_meta_guidance_tool_hints():
    skills = [
        SkillSpec(
            name="local-document-evidence",
            description="doc workflow",
            kind="workflow",
            priority=100,
            tools=("file_read",),
            triggers=("document",),
            scope="turn",
            body="Read local files first.",
        ),
        SkillSpec(
            name="paper-pdf-retrieval",
            description="paper workflow",
            kind="workflow",
            priority=95,
            tools=("openalex_works",),
            triggers=("paper", "doi"),
            scope="turn",
            body="Use OpenAlex first.",
        ),
    ]

    prompt = (
        "In the Scikit-Learn July 2017 changelog, what other predictor base command received a bug fix?\n\n"
        "Evaluation execution guidance:\n"
        "- For scholarly or document tasks, prefer openalex_works and exact local files before perplexity_search or web snapshots.\n"
        "- Do not answer from search-result snippets alone if you can open the cited page or a local artifact and verify the exact field.\n"
    )

    selected = select_turn_skills(prompt, ["file_read", "openalex_works"], skills)

    assert selected == []


def test_select_turn_skills_keeps_attachment_section_after_stripping_eval_guidance():
    skills = [
        SkillSpec(
            name="local-document-evidence",
            description="doc workflow",
            kind="workflow",
            priority=100,
            tools=("file_read",),
            triggers=("document",),
            scope="turn",
            body="Read local files first.",
        ),
        SkillSpec(
            name="web-verification",
            description="web workflow",
            kind="workflow",
            priority=90,
            tools=("jina_web_snapshot",),
            triggers=("verify",),
            scope="turn",
            body="Verify from sources.",
        ),
    ]

    prompt = (
        "Calculate the distance between the first two atoms in the attached structure.\n\n"
        "Evaluation execution guidance:\n"
        "- Prefer exact local files and precise structured tools before generic web discovery.\n"
        "- For scholarly or document tasks, prefer openalex_works and exact local files before perplexity_search or web snapshots.\n\n"
        "Required attachment files are preloaded in the workspace at:\n"
        "- eval/runs/sample/_attachments/5wb7.pdb\n"
        "Local files are primary evidence for this task.\n"
    )

    selected = select_turn_skills(prompt, ["file_read", "jina_web_snapshot"], skills)

    assert [skill.name for skill in selected] == ["local-document-evidence"]


def test_select_turn_skills_matches_structured_answer_verification_cues():
    skills = [
        SkillSpec(
            name="structured-answer-verification",
            description="comparison workflow",
            kind="workflow",
            priority=92,
            tools=("calculator", "jina_web_snapshot"),
            triggers=("compare", "difference"),
            scope="turn",
            body="Verify the eligible set before computing.",
        ),
        SkillSpec(
            name="web-verification",
            description="web workflow",
            kind="workflow",
            priority=90,
            tools=("jina_web_snapshot",),
            triggers=("verify",),
            scope="turn",
            body="Verify from sources.",
        ),
    ]

    selected = select_turn_skills(
        "Which ASEAN capital city pair is furthest apart, and what is the difference in distance compared with the nearest pair?",
        ["calculator", "jina_web_snapshot"],
        skills,
    )

    assert [skill.name for skill in selected] == ["structured-answer-verification"]


def test_render_skill_text_formats_named_blocks():
    text = render_skill_text(
        [
            SkillSpec(
                name="calculator-rules",
                description="calc rules",
                kind="tool_rules",
                priority=10,
                tools=("calculator",),
                triggers=("calc",),
                scope="system",
                body="Use calculator first.",
            )
        ]
    )

    assert text == "[calculator-rules]\nUse calculator first."
