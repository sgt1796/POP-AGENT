from agent_build.agent1.tui import _looks_like_markdown_text


def test_looks_like_markdown_text_detects_headings_and_lists():
    text = "# Plan\n\n1. First item\n2. Second item"
    assert _looks_like_markdown_text(text) is True


def test_looks_like_markdown_text_detects_code_fence():
    text = "```python\nprint('hi')\n```"
    assert _looks_like_markdown_text(text) is True


def test_looks_like_markdown_text_ignores_plain_text():
    text = "This is a plain sentence with punctuation.\nSecond line of plain text."
    assert _looks_like_markdown_text(text) is False
