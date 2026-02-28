from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import openpyxl
import pytest

from agent.tools.file_read_tool import FileReadError, FileReadTool, read


def _run(tool, params):
    return asyncio.run(tool.execute("tc1", params))


def test_read_txt_and_md(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("hello text", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# header", encoding="utf-8")

    txt = read("notes.txt", workspace_root=str(tmp_path))
    md = read("guide.md", workspace_root=str(tmp_path))

    assert txt["ok"] is True
    assert txt["kind"] == "text"
    assert txt["content"] == "hello text"
    assert txt["suffix"] == ".txt"

    assert md["ok"] is True
    assert md["kind"] == "text"
    assert md["content"] == "# header"
    assert md["suffix"] == ".md"


def test_read_json(tmp_path: Path):
    (tmp_path / "payload.json").write_text('{"a": 1, "b": "two"}', encoding="utf-8")

    result = read("payload.json", workspace_root=str(tmp_path))

    assert result["ok"] is True
    assert result["kind"] == "json"
    assert result["content"] == {"a": 1, "b": "two"}
    assert result["truncated"] is False


def test_read_csv(tmp_path: Path):
    (tmp_path / "table.csv").write_text("name,age\nAlice,30\nBob,31\n", encoding="utf-8")

    result = read("table.csv", workspace_root=str(tmp_path))

    assert result["ok"] is True
    assert result["kind"] == "csv"
    assert result["metadata"]["headers"] == ["name", "age"]
    assert result["metadata"]["row_count"] == 2
    assert result["content"] == [
        {"name": "Alice", "age": "30"},
        {"name": "Bob", "age": "31"},
    ]


def test_read_xlsx_json(tmp_path: Path):
    workbook = openpyxl.Workbook()
    movies = workbook.active
    movies.title = "Movies"
    movies.append(["title", "year", "in_stock"])
    movies.append(["Inception", 2010, True])
    movies.append(["Spirited Away", 2001, False])
    directors = workbook.create_sheet("Directors")
    directors.append(["name"])
    directors.append(["Christopher Nolan"])
    workbook.save(tmp_path / "inventory.xlsx")

    result = read("inventory.xlsx", workspace_root=str(tmp_path))

    assert result["ok"] is True
    assert result["kind"] == "xlsx"
    assert result["metadata"]["format"] == "json"
    assert result["metadata"]["sheet_count"] == 2
    assert result["metadata"]["total_row_count"] == 3
    sheets = result["content"]["sheets"]
    assert sheets[0]["name"] == "Movies"
    assert sheets[0]["rows"][0] == {"title": "Inception", "year": 2010, "in_stock": True}
    assert sheets[0]["rows"][1] == {"title": "Spirited Away", "year": 2001, "in_stock": False}
    assert sheets[1]["name"] == "Directors"
    assert sheets[1]["rows"] == [{"name": "Christopher Nolan"}]


def test_read_xlsx_csv(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Movies"
    sheet.append(["title", "year"])
    sheet.append(["Arrival", 2016])
    workbook.save(tmp_path / "movies.xlsx")

    result = read("movies.xlsx", workspace_root=str(tmp_path), xlsx_format="csv")

    assert result["ok"] is True
    assert result["kind"] == "xlsx"
    assert result["metadata"]["format"] == "csv"
    assert result["metadata"]["sheet_count"] == 1
    assert result["metadata"]["total_row_count"] == 1
    assert "title,year" in result["content"]
    assert "Arrival,2016" in result["content"]


def test_read_xlsx_rejects_invalid_format(tmp_path: Path):
    (tmp_path / "empty.xlsx").write_bytes(b"placeholder")

    with pytest.raises(FileReadError) as exc:
        read("empty.xlsx", workspace_root=str(tmp_path), xlsx_format="xml")

    assert exc.value.code == "parse_error"


def test_read_pdf_includes_page_metadata(tmp_path: Path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_path = tmp_path / "sample.pdf"
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    result = read("sample.pdf", workspace_root=str(tmp_path))

    assert result["ok"] is True
    assert result["kind"] == "pdf"
    assert result["metadata"]["page_count"] == 1
    assert "extracted_char_count" in result["metadata"]
    assert isinstance(result["content"], str)


def test_read_image_as_base64(tmp_path: Path):
    png_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6pYQAAAABJRU5ErkJggg==")
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(png_bytes)

    result = read("pixel.png", workspace_root=str(tmp_path))

    assert result["ok"] is True
    assert result["kind"] == "image"
    assert result["metadata"]["mime_type"] == "image/png"
    assert base64.b64decode(result["content"]) == png_bytes


def test_read_rejects_path_outside_workspace(tmp_path: Path):
    outside = tmp_path.parent / "outside_file.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(FileReadError) as exc:
        read(str(outside), workspace_root=str(tmp_path))

    assert exc.value.code == "path_outside_workspace"


def test_read_rejects_unsupported_suffix(tmp_path: Path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")

    with pytest.raises(FileReadError) as exc:
        read("blob.bin", workspace_root=str(tmp_path))

    assert exc.value.code == "unsupported_suffix"


def test_file_read_tool_success_envelope(tmp_path: Path):
    (tmp_path / "message.txt").write_text("hello tool", encoding="utf-8")
    tool = FileReadTool(workspace_root=str(tmp_path), max_output_chars=50_000)

    result = _run(tool, {"path": "message.txt"})

    assert result.details["ok"] is True
    assert result.details["kind"] == "text"
    payload = json.loads(result.content[0].text)
    assert payload["ok"] is True
    assert payload["content"] == "hello tool"


def test_file_read_tool_error_envelope(tmp_path: Path):
    tool = FileReadTool(workspace_root=str(tmp_path))
    result = _run(tool, {"path": "missing.txt"})

    assert result.details == {
        "ok": False,
        "error": "path_not_found",
        "message": "file not found: missing.txt",
        "path": "missing.txt",
    }
    assert result.content[0].text == "file_read error: file not found: missing.txt"
