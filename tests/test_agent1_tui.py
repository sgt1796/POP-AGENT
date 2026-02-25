from agent_build.agent1.tui import _InputHistory, _looks_like_markdown_text
from agent_build.agent1.usage_reporting import format_cumulative_usage_fragment


def test_looks_like_markdown_text_detects_headings_and_lists():
    text = "# Plan\n\n1. First item\n2. Second item"
    assert _looks_like_markdown_text(text) is True


def test_looks_like_markdown_text_detects_code_fence():
    text = "```python\nprint('hi')\n```"
    assert _looks_like_markdown_text(text) is True


def test_looks_like_markdown_text_ignores_plain_text():
    text = "This is a plain sentence with punctuation.\nSecond line of plain text."
    assert _looks_like_markdown_text(text) is False


def test_usage_fragment_for_status_line_is_compact():
    fragment = format_cumulative_usage_fragment(
        {
            "total_tokens": 200,
            "input_tokens": 150,
            "output_tokens": 50,
            "calls": 4,
            "provider_calls": 2,
            "estimated_calls": 1,
            "hybrid_calls": 1,
            "anomaly_calls": 0,
        }
    )
    assert fragment == "usage(total=200,in=150,out=50,calls=4,p/e/h=2/1/1,anom=0)"


def test_input_history_up_down_navigation_and_draft_restore():
    history = _InputHistory()
    history.add("first")
    history.add("second")

    assert history.move_up("draft now") == "second"
    assert history.move_up("ignored while browsing") == "first"
    assert history.move_down() == "second"
    assert history.move_down() == "draft now"
    assert history.move_down() is None


def test_input_history_ignores_blank_entries():
    history = _InputHistory()
    history.add("   ")
    history.add("")
    history.add("ok")

    assert history.entries == ["ok"]
