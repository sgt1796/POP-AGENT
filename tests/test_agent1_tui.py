from agent_build.agent1.tui import (
    _InputHistory,
    _insert_text_at_cursor,
    _looks_like_markdown_text,
    _normalize_pasted_chat_text,
    _render_transcript_entry,
    _show_pending_tool_line,
)
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


def test_render_transcript_entry_puts_bold_role_on_its_own_line_with_separator():
    rendered = []

    _render_transcript_entry(
        rendered.append,
        "User",
        "hello there",
        text_factory=lambda value, style=None: ("text", value, style),
        markdown_factory=lambda value: ("markdown", value),
    )

    assert rendered == [
        ("text", "User:", "bold"),
        ("text", "hello there", None),
        "",
    ]


def test_render_transcript_entry_uses_markdown_for_assistant_markdown_messages():
    rendered = []
    body = "# Plan\n\n1. First item\n2. Second item"

    _render_transcript_entry(
        rendered.append,
        "Assistant",
        body,
        text_factory=lambda value, style=None: ("text", value, style),
        markdown_factory=lambda value: ("markdown", value),
    )

    assert rendered == [
        ("text", "Assistant:", "bold"),
        ("markdown", body),
        "",
    ]


def test_show_pending_tool_line_hides_pending_output_for_quiet_and_simple():
    assert _show_pending_tool_line("quiet") is False
    assert _show_pending_tool_line("simple") is False
    assert _show_pending_tool_line("messages") is False
    assert _show_pending_tool_line("full") is True
    assert _show_pending_tool_line("stream") is True
    assert _show_pending_tool_line("debug") is True


def test_normalize_pasted_chat_text_replaces_newlines_with_literal_markers():
    assert _normalize_pasted_chat_text("line1\nline2") == "line1\\nline2"
    assert _normalize_pasted_chat_text("line1\r\nline2\rline3") == "line1\\nline2\\nline3"


def test_insert_text_at_cursor_inserts_in_middle_and_tracks_cursor():
    updated, cursor = _insert_text_at_cursor("hello world", 5, "\\n")
    assert updated == "hello\\n world"
    assert cursor == 7
