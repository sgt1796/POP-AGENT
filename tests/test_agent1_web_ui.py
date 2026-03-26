from agent_build.agent1.web.ui_extraction import extract_structured_ui


def test_extract_result_table_from_markdown() -> None:
    spec = extract_structured_ui(
        """
Repository findings

| File | Status | Notes |
| --- | --- | --- |
| runtime.py | done | Core seam identified |
| web/app.py | pending | Still needs API adapter |
"""
    )
    assert spec is not None
    assert spec.type == "ResultTable"
    assert spec.props.columns == ["File", "Status", "Notes"]
    assert spec.props.rows[0] == ["runtime.py", "done", "Core seam identified"]


def test_extract_plan_checklist_from_checkboxes() -> None:
    spec = extract_structured_ui(
        """
Execution Plan:
- [x] Inspect runtime seam
- [~] Add API adapter
- [ ] Build CopilotKit frontend
"""
    )
    assert spec is not None
    assert spec.type == "PlanChecklist"
    assert [item.status for item in spec.props.items] == ["done", "in_progress", "pending"]


def test_extract_plan_checklist_from_markdown_heading_and_numbered_steps() -> None:
    spec = extract_structured_ui(
        """
### **Execution Plan**
1. Inspect runtime seam
2. Add API adapter
3. Build CopilotKit frontend
"""
    )
    assert spec is not None
    assert spec.type == "PlanChecklist"
    assert spec.props.title == "Execution Plan"
    assert [item.label for item in spec.props.items] == [
        "Inspect runtime seam",
        "Add API adapter",
        "Build CopilotKit frontend",
    ]


def test_ignore_markdown_summary_bullets_that_are_not_tasks() -> None:
    spec = extract_structured_ui(
        """
### **Market Overview (March 26, 2026)**
* **Dow Jones Industrial Average:** 45,960.11 (-1.01% | -469.38 pts)
* **S&P 500:** 6,477.16 (-1.74% | -114.74 pts)
* **Nasdaq Composite:** 21,408.08 (-2.38% | -521.75 pts)
"""
    )
    assert spec is None


def test_ignore_plain_paragraph_text() -> None:
    assert extract_structured_ui("This is a regular assistant reply without structured output.") is None
