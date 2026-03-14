from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.tools.file_write_tool import FileWriteError, FileWriteTool, write


def _run(tool, params):
    return asyncio.run(tool.execute("tc1", params))


def test_write_creates_new_file(tmp_path: Path):
    payload = write("notes/out.txt", workspace_root=str(tmp_path), action="write", content="hello", create_dirs=True)

    assert payload["ok"] is True
    assert payload["created"] is True
    assert (tmp_path / "notes" / "out.txt").read_text(encoding="utf-8") == "hello"


def test_append_adds_text(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")

    payload = write("a.txt", workspace_root=str(tmp_path), action="append", content=" two")

    assert payload["ok"] is True
    assert payload["created"] is False
    assert path.read_text(encoding="utf-8") == "one two"


def test_replace_honors_count_limit(tmp_path: Path):
    path = tmp_path / "replace.txt"
    path.write_text("apple apple apple", encoding="utf-8")

    payload = write(
        "replace.txt",
        workspace_root=str(tmp_path),
        action="replace",
        find="apple",
        replace_with="orange",
        count=2,
    )

    assert payload["ok"] is True
    assert payload["replacements"] == 2
    assert path.read_text(encoding="utf-8") == "orange orange apple"


def test_write_rejects_outside_workspace(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    with pytest.raises(FileWriteError) as exc:
        write(str(outside), workspace_root=str(tmp_path), action="write", content="x")
    assert exc.value.code == "path_outside_workspace"


def test_write_allows_target_in_allowed_roots(tmp_path: Path):
    outside_root = tmp_path / "external"
    outside_root.mkdir()
    outside_file = outside_root / "out.txt"

    payload = write(
        str(outside_file),
        workspace_root=str(tmp_path),
        allowed_roots=[str(outside_root)],
        action="write",
        content="shared root",
    )

    assert payload["ok"] is True
    assert outside_file.read_text(encoding="utf-8") == "shared root"


def test_file_write_tool_success_envelope(tmp_path: Path):
    tool = FileWriteTool(workspace_root=str(tmp_path))

    result = _run(tool, {"path": "hello.txt", "action": "write", "content": "world"})

    assert result.details["ok"] is True
    payload = json.loads(result.content[0].text)
    assert payload["ok"] is True
    assert payload["action"] == "write"
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "world"


def test_file_write_tool_error_envelope(tmp_path: Path):
    tool = FileWriteTool(workspace_root=str(tmp_path))

    result = _run(tool, {"path": "hello.txt", "action": "replace", "find": "x", "replace_with": "y"})

    assert result.details["ok"] is False
    assert result.details["error"] == "path_not_found"
    assert result.content[0].text == "file_write error: file not found: hello.txt"
