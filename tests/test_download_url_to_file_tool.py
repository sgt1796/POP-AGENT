from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import requests

from agent.tools import DownloadUrlToFileTool


def _run(tool: Any, params: Dict[str, Any]):
    return asyncio.run(tool.execute("tc1", params))


class _FakeResponse:
    def __init__(
        self,
        *,
        chunks: Iterable[bytes],
        content_type: str = "application/pdf",
        url: str = "https://example.org/file.pdf",
        status_code: int = 200,
    ) -> None:
        self._chunks = [bytes(chunk) for chunk in chunks]
        self.headers = {"Content-Type": content_type}
        self.url = url
        self.status_code = int(status_code)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"status={self.status_code}")
            error.response = SimpleNamespace(status_code=self.status_code)
            raise error

    def iter_content(self, chunk_size: Optional[int] = None):
        del chunk_size
        for chunk in self._chunks:
            yield chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


def test_download_url_to_file_success(tmp_path: Path, monkeypatch):
    payload = [b"%PDF-1.7\n", b"paper-bytes\n"]

    def _fake_get(url, stream=False, timeout=None, allow_redirects=False):
        assert url == "https://example.org/paper.pdf"
        assert stream is True
        assert allow_redirects is True
        assert timeout == 12.0
        return _FakeResponse(chunks=payload, content_type="application/pdf", url=url)

    monkeypatch.setattr("agent.tools.download_url_to_file.requests.get", _fake_get)

    tool = DownloadUrlToFileTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "url": "https://example.org/paper.pdf",
            "output_path": "downloads/paper.pdf",
            "timeout_s": 12,
            "expected_content_type": "application/pdf",
        },
    )

    target = tmp_path / "downloads" / "paper.pdf"
    expected_bytes = b"".join(payload)

    assert result.details["ok"] is True
    assert result.details["saved_path"] == str(target.resolve())
    assert result.details["bytes_written"] == len(expected_bytes)
    assert result.details["content_type"] == "application/pdf"
    assert result.details["final_url"] == "https://example.org/paper.pdf"
    assert result.details["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert target.read_bytes() == expected_bytes


def test_download_url_to_file_tracks_redirect_final_url(tmp_path: Path, monkeypatch):
    def _fake_get(url, stream=False, timeout=None, allow_redirects=False):
        assert url == "https://example.org/start"
        del stream, timeout, allow_redirects
        return _FakeResponse(
            chunks=[b"%PDF-1.4\n", b"redirected"],
            content_type="application/pdf; charset=binary",
            url="https://cdn.example.org/final.pdf",
        )

    monkeypatch.setattr("agent.tools.download_url_to_file.requests.get", _fake_get)

    tool = DownloadUrlToFileTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "url": "https://example.org/start",
            "output_path": "paper.pdf",
        },
    )

    assert result.details["ok"] is True
    assert result.details["final_url"] == "https://cdn.example.org/final.pdf"
    assert result.details["content_type"] == "application/pdf"
    assert (tmp_path / "paper.pdf").exists()


def test_download_url_to_file_rejects_unexpected_content_type(tmp_path: Path, monkeypatch):
    def _fake_get(url, stream=False, timeout=None, allow_redirects=False):
        del url, stream, timeout, allow_redirects
        return _FakeResponse(
            chunks=[
                (
                    b"<html><head><title>Project MUSE - A Dark Trace</title></head>"
                    b"<body><a href=\"/pub/258/oa_monograph/book/24372/pdf\">Download PDF</a></body></html>"
                )
            ],
            content_type="text/html",
            url="https://muse.jhu.edu/pub/258/oa_monograph/book/24372/pdf",
        )

    monkeypatch.setattr("agent.tools.download_url_to_file.requests.get", _fake_get)

    tool = DownloadUrlToFileTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "url": "https://example.org/not-a-pdf",
            "output_path": "downloads/file.pdf",
            "expected_content_type": "application/pdf",
        },
    )

    assert result.details["ok"] is False
    assert result.details["error"] == "unexpected_content_type"
    assert "expected content type" in result.content[0].text
    assert result.details["html_title"] == "Project MUSE - A Dark Trace"
    assert result.details["pdf_link_candidates"] == ["https://muse.jhu.edu/pub/258/oa_monograph/book/24372/pdf"]
    assert "landing page title" in result.content[0].text
    assert "final_url" in result.content[0].text
    assert "content_preview" in result.content[0].text
    assert "recovery_hint" in result.content[0].text
    assert "retry one of the exact PDF links before broad search" in result.details["recovery_hint"]
    assert not (tmp_path / "downloads" / "file.pdf").exists()


def test_download_url_to_file_flags_verification_page_with_recovery_hint(tmp_path: Path, monkeypatch):
    def _fake_get(url, stream=False, timeout=None, allow_redirects=False):
        del url, stream, timeout, allow_redirects
        return _FakeResponse(
            chunks=[
                (
                    b"<!DOCTYPE html><html><head><title>Project MUSE -- Verification required!</title></head>"
                    b"<body>Please verify you are a human before continuing.</body></html>"
                )
            ],
            content_type="text/html",
            url="https://muse.jhu.edu/verify?url=%2Fpub%2F258%2Foa_monograph%2Fbook%2F24372%2Fpdf",
        )

    monkeypatch.setattr("agent.tools.download_url_to_file.requests.get", _fake_get)

    tool = DownloadUrlToFileTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "url": "https://example.org/not-a-pdf",
            "output_path": "downloads/file.pdf",
            "expected_content_type": "application/pdf",
        },
    )

    assert result.details["ok"] is False
    assert result.details["error"] == "unexpected_content_type"
    assert result.details["html_title"] == "Project MUSE -- Verification required!"
    assert "verification/interstitial page" in result.details["recovery_hint"]
    assert "source landing page before broad search" in result.details["recovery_hint"]
    assert "recovery_hint" in result.content[0].text


def test_download_url_to_file_enforces_max_bytes(tmp_path: Path, monkeypatch):
    def _fake_get(url, stream=False, timeout=None, allow_redirects=False):
        del url, stream, timeout, allow_redirects
        return _FakeResponse(chunks=[b"abcd", b"efgh"], content_type="application/pdf")

    monkeypatch.setattr("agent.tools.download_url_to_file.requests.get", _fake_get)

    tool = DownloadUrlToFileTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "url": "https://example.org/large.pdf",
            "output_path": "downloads/large.pdf",
            "max_bytes": 6,
        },
    )

    assert result.details["ok"] is False
    assert result.details["error"] == "file_too_large"
    assert result.details["bytes_written"] == 4
    assert not (tmp_path / "downloads" / "large.pdf").exists()


def test_download_url_to_file_rejects_output_path_outside_workspace(tmp_path: Path):
    tool = DownloadUrlToFileTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "url": "https://example.org/paper.pdf",
            "output_path": "/tmp/outside.pdf",
        },
    )

    assert result.details["ok"] is False
    assert result.details["error"] == "path_outside_workspace"


def test_download_url_to_file_allows_output_path_in_allowed_roots(tmp_path: Path, monkeypatch):
    payload = [b"report"]
    outside_root = tmp_path / "external"
    outside_root.mkdir()
    outside_file = outside_root / "report.txt"

    def _fake_get(url, stream=False, timeout=None, allow_redirects=False):
        del stream, timeout, allow_redirects
        assert url == "https://example.org/report.txt"
        return _FakeResponse(chunks=payload, content_type="text/plain", url=url)

    monkeypatch.setattr("agent.tools.download_url_to_file.requests.get", _fake_get)

    tool = DownloadUrlToFileTool(workspace_root=str(tmp_path), allowed_roots=[str(outside_root)])
    result = _run(
        tool,
        {
            "url": "https://example.org/report.txt",
            "output_path": str(outside_file),
        },
    )

    assert result.details["ok"] is True
    assert outside_file.read_bytes() == b"report"


def test_download_url_to_file_returns_network_error(tmp_path: Path, monkeypatch):
    def _fake_get(url, stream=False, timeout=None, allow_redirects=False):
        del url, stream, timeout, allow_redirects
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("agent.tools.download_url_to_file.requests.get", _fake_get)

    tool = DownloadUrlToFileTool(workspace_root=str(tmp_path))
    result = _run(
        tool,
        {
            "url": "https://example.org/paper.pdf",
            "output_path": "downloads/network.pdf",
        },
    )

    assert result.details["ok"] is False
    assert result.details["error"] == "network_error"
    assert "connection refused" in result.content[0].text
