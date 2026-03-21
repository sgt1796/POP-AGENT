from __future__ import annotations

import asyncio
import builtins
import io
import json
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict
from urllib import error as urllib_error
from urllib import parse as urllib_parse

import requests

from agent.tools import JinaWebSnapshotTool, OpenAlexWorksTool, PerplexitySearchTool, PerplexityWebSnapshotTool


def _run(tool: Any, params: Dict[str, Any]):
    return asyncio.run(tool.execute("tc1", params))


class _FakeUrlResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeUrlResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


def test_jina_web_snapshot_returns_error_when_url_missing():
    result = _run(JinaWebSnapshotTool(), {})
    assert result.details["ok"] is False
    assert "missing web_url" in result.content[0].text


def test_perplexity_search_returns_error_when_query_missing():
    result = _run(PerplexitySearchTool(), {})
    assert result.details["ok"] is False
    assert "missing query" in result.content[0].text


def test_perplexity_search_returns_error_for_invalid_date():
    result = _run(PerplexitySearchTool(), {"query": "market news", "from_date": "2026/02/25"})
    assert result.details["ok"] is False
    assert "YYYY-MM-DD" in result.content[0].text


def test_perplexity_search_returns_error_when_sdk_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "perplexity", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "perplexity":
            raise ImportError("missing perplexity")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    result = _run(PerplexitySearchTool(), {"query": "stock market today"})
    assert result.details["ok"] is False
    assert result.details["error"] == "missing_perplexity_sdk"


def test_perplexity_search_applies_default_limits_and_concise_rendering(monkeypatch):
    class _FakeSearchApi:
        def __init__(self) -> None:
            self.calls: list[Dict[str, Any]] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            long_snippet = "A" * 400
            return SimpleNamespace(
                results=[
                    {"title": f"Result {idx}", "url": f"https://example.com/{idx}", "snippet": long_snippet, "date": "2026-03-20", "updated_at": "2026-03-21"}
                    for idx in range(1, 5)
                ]
            )

    class _FakePerplexity:
        last_instance: "_FakePerplexity | None" = None

        def __init__(self) -> None:
            self.search = _FakeSearchApi()
            _FakePerplexity.last_instance = self

    fake_module = types.ModuleType("perplexity")
    fake_module.Perplexity = _FakePerplexity
    monkeypatch.setitem(sys.modules, "perplexity", fake_module)

    result = _run(PerplexitySearchTool(), {"query": "compact query"})

    instance = _FakePerplexity.last_instance
    assert instance is not None
    call = instance.search.calls[0]
    assert call["max_results"] == 3
    assert call["max_tokens_per_page"] == 1024

    assert result.details["ok"] is True
    assert result.details["count"] == 4
    assert len(result.details["results"]) == 4
    assert "1. Result 1" in result.content[0].text
    assert "3. Result 3" in result.content[0].text
    assert "4. Result 4" not in result.content[0].text
    assert "Updated:" not in result.content[0].text
    assert ("A" * 320) not in result.content[0].text


def test_perplexity_search_success_with_mocked_sdk(monkeypatch):
    class _FakeSearchApi:
        def __init__(self) -> None:
            self.calls: list[Dict[str, Any]] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        title="Result A",
                        url="https://example.com/a",
                        snippet="Alpha snippet",
                        date="2026-02-24",
                    ),
                    {
                        "title": "Result B",
                        "link": "https://example.com/b",
                        "text": "Beta snippet",
                        "updated_at": "2026-02-25",
                    },
                ]
            )

    class _FakePerplexity:
        last_instance: "_FakePerplexity | None" = None

        def __init__(self) -> None:
            self.search = _FakeSearchApi()
            _FakePerplexity.last_instance = self

    fake_module = types.ModuleType("perplexity")
    fake_module.Perplexity = _FakePerplexity
    monkeypatch.setitem(sys.modules, "perplexity", fake_module)

    tool = PerplexitySearchTool()
    result = _run(
        tool,
        {
            "query": "stock market today",
            "max_results": 5,
            "max_tokens_per_page": 4096,
            "country": "US",
            "search_domain_filter": ["example.com"],
            "search_recency_filter": "week",
            "from_date": "2026-02-01",
            "to_date": "2026-02-25",
        },
    )

    assert result.details["ok"] is True
    assert result.details["count"] == 2
    assert "1. Result A" in result.content[0].text
    assert "2. Result B" in result.content[0].text

    instance = _FakePerplexity.last_instance
    assert instance is not None
    call = instance.search.calls[0]
    assert call["query"] == "stock market today"
    assert call["max_results"] == 5
    assert call["max_tokens_per_page"] == 4096
    assert call["country"] == "US"
    assert call["search_domain_filter"] == ["example.com"]
    assert call["search_recency_filter"] == "week"
    assert call["from_date"] == "2026-02-01"
    assert call["to_date"] == "2026-02-25"


class _FakeRequestsResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = int(status_code)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")


def test_jina_web_snapshot_truncates_content_and_reports_metadata(monkeypatch):
    def _fake_get(url: str, headers: Dict[str, str]):
        assert url == "https://r.jina.ai/https://example.com/large"
        assert isinstance(headers, dict)
        return _FakeRequestsResponse(text="X" * 20_000)

    monkeypatch.setattr("agent.tools.search.jina_web_snapshot.requests.get", _fake_get)
    result = _run(JinaWebSnapshotTool(), {"web_url": "https://example.com/large"})

    assert result.details["ok"] is True
    assert result.details["url"] == "https://example.com/large"
    assert result.details["char_count"] == 20_000
    assert result.details["truncated"] is True
    assert result.details["max_chars"] == 12_000
    assert len(result.content[0].text) == 12_000


def test_jina_web_snapshot_stringifies_timeout_header(monkeypatch):
    captured = {}

    def _fake_get(url: str, headers: Dict[str, str]):
        captured["url"] = url
        captured["headers"] = dict(headers)
        return _FakeRequestsResponse(text="ok")

    monkeypatch.setattr("agent.tools.search.jina_web_snapshot.requests.get", _fake_get)
    result = _run(JinaWebSnapshotTool(), {"web_url": "https://example.com/page", "timeout": 20})

    assert result.details["ok"] is True
    assert captured["url"] == "https://r.jina.ai/https://example.com/page"
    assert captured["headers"]["X-Timeout"] == "20"


def test_perplexity_web_snapshot_stub_response():
    result = _run(PerplexityWebSnapshotTool(), {"url": "https://example.com"})
    assert result.details["ok"] is False
    assert result.details["implemented"] is False
    assert result.details["tool"] == "perplexity_web_snapshot"
    assert "not implemented yet" in result.content[0].text


def test_openalex_works_returns_error_for_invalid_action():
    result = _run(OpenAlexWorksTool(), {"action": "lookup"})
    assert result.details["ok"] is False
    assert result.details["error"] == "invalid_action"


def test_openalex_works_search_requires_query_or_filter():
    result = _run(OpenAlexWorksTool(), {"action": "search"})
    assert result.details["ok"] is False
    assert "requires query or at least one filter" in result.content[0].text


def test_openalex_works_search_rejects_large_per_page():
    result = _run(OpenAlexWorksTool(), {"action": "search", "query": "test", "per_page": 500})
    assert result.details["ok"] is False
    assert "per_page must be <= 200" in result.content[0].text


def test_openalex_works_search_success_with_filter_and_cursor(monkeypatch):
    from agent.tools.search import openalex_works as openalex_mod

    calls = []

    def _fake_urlopen(req, timeout=0):
        del timeout
        parsed = urllib_parse.urlparse(req.full_url)
        query = urllib_parse.parse_qs(parsed.query)
        calls.append({"path": parsed.path, "query": query})
        return _FakeUrlResponse(
            {
                "meta": {"count": 1, "next_cursor": "cursor-next"},
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "title": "A Better Retrieval Paper",
                        "publication_year": 2024,
                        "publication_date": "2024-05-02",
                        "cited_by_count": 42,
                        "doi": "https://doi.org/10.1234/example",
                        "authorships": [
                            {"author": {"display_name": "Ada Lovelace"}},
                            {"author": {"display_name": "Grace Hopper"}},
                        ],
                        "primary_location": {"source": {"display_name": "Journal of Testing"}},
                        "open_access": {"is_oa": True, "oa_url": "https://oa.example.org/paper"},
                        "best_oa_location": {
                            "landing_page_url": "https://example.org/landing",
                            "pdf_url": "https://example.org/paper.pdf",
                        },
                        "default_relevance_score": 73.1,
                        "abstract_inverted_index": {"research": [0], "paper": [1]},
                    }
                ],
            }
        )

    monkeypatch.setenv("OPENALEX_EMAIL", "team@example.com")
    monkeypatch.setenv("OPENALEX_API_KEY", "k_test_123")
    monkeypatch.setattr(openalex_mod.urllib_request, "urlopen", _fake_urlopen)

    result = _run(
        OpenAlexWorksTool(),
        {
            "action": "search",
            "query": "retrieval augmented generation",
            "per_page": 5,
            "cursor": "cursor:abc",
            "sort": "cited_by_count:desc",
            "select": ["id", "title", "doi"],
            "publication_year_from": 2020,
            "publication_year_to": 2024,
            "type": "article",
            "open_access_only": True,
            "has_doi": True,
            "filter": "is_paratext:false",
            "include_abstract": True,
        },
    )

    assert result.details["ok"] is True
    assert result.details["action"] == "search"
    assert result.details["count"] == 1
    assert result.details["next_cursor"] == "cursor-next"
    assert result.details["results"][0]["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert result.details["results"][0]["abstract"] == "research paper"
    assert "1. A Better Retrieval Paper" in result.content[0].text

    assert len(calls) == 1
    call = calls[0]
    assert call["path"] == "/works"
    assert call["query"]["search"] == ["retrieval augmented generation"]
    assert call["query"]["per-page"] == ["5"]
    assert call["query"]["cursor"] == ["cursor:abc"]
    assert call["query"]["sort"] == ["cited_by_count:desc"]
    assert call["query"]["select"] == ["id,title,doi"]
    assert call["query"]["mailto"] == ["team@example.com"]
    assert call["query"]["api_key"] == ["k_test_123"]
    assert call["query"]["filter"] == [
        "publication_year:2020-2024,type:article,is_oa:true,has_doi:true,is_paratext:false"
    ]


def test_openalex_works_fetch_openalex_id_success(monkeypatch):
    from agent.tools.search import openalex_works as openalex_mod

    calls = []

    def _fake_urlopen(req, timeout=0):
        del timeout
        parsed = urllib_parse.urlparse(req.full_url)
        query = urllib_parse.parse_qs(parsed.query)
        calls.append({"path": parsed.path, "query": query})
        return _FakeUrlResponse(
            {
                "id": "https://openalex.org/W555",
                "title": "OpenAlex Direct Fetch",
                "publication_year": 2023,
                "publication_date": "2023-11-01",
                "cited_by_count": 8,
                "doi": "https://doi.org/10.1000/direct",
                "authorships": [{"author": {"display_name": "Linus Torvalds"}}],
                "open_access": {"is_oa": False, "oa_url": ""},
            }
        )

    monkeypatch.setattr(openalex_mod.urllib_request, "urlopen", _fake_urlopen)
    result = _run(
        OpenAlexWorksTool(),
        {
            "action": "fetch_openalex_record",
            "work_id": "https://openalex.org/W555",
            "select": ["id", "title", "doi"],
        },
    )

    assert result.details["ok"] is True
    assert result.details["action"] == "fetch_openalex_record"
    assert result.details["mode"] == "openalex_id"
    assert result.details["record"]["id"] == "https://openalex.org/W555"
    assert result.details["record"]["authors"] == ["Linus Torvalds"]
    assert "OpenAlex record: OpenAlex Direct Fetch" in result.content[0].text

    call = calls[0]
    assert call["path"] == "/works/W555"
    assert call["query"]["select"] == ["id,title,doi"]


def test_openalex_works_fetch_doi_lookup_success(monkeypatch):
    from agent.tools.search import openalex_works as openalex_mod

    calls = []

    def _fake_urlopen(req, timeout=0):
        del timeout
        parsed = urllib_parse.urlparse(req.full_url)
        query = urllib_parse.parse_qs(parsed.query)
        calls.append({"path": parsed.path, "query": query})
        return _FakeUrlResponse(
            {
                "meta": {"count": 1},
                "results": [
                    {
                        "id": "https://openalex.org/W999",
                        "title": "DOI Lookup Result",
                        "publication_year": 2025,
                        "cited_by_count": 1,
                        "doi": "https://doi.org/10.1000/xyz",
                    }
                ],
            }
        )

    monkeypatch.setattr(openalex_mod.urllib_request, "urlopen", _fake_urlopen)
    result = _run(
        OpenAlexWorksTool(),
        {"action": "fetch_openalex_record", "work_id": "10.1000/xyz"},
    )

    assert result.details["ok"] is True
    assert result.details["mode"] == "doi_lookup"
    assert result.details["resolved_work_id"] == "10.1000/xyz"
    assert result.details["record"]["id"] == "https://openalex.org/W999"

    call = calls[0]
    assert call["path"] == "/works"
    assert call["query"]["filter"] == ["doi:https://doi.org/10.1000/xyz"]
    assert call["query"]["per-page"] == ["1"]


def test_openalex_works_fetch_doi_not_found(monkeypatch):
    from agent.tools.search import openalex_works as openalex_mod

    def _fake_urlopen(req, timeout=0):
        del req, timeout
        return _FakeUrlResponse({"meta": {"count": 0}, "results": []})

    monkeypatch.setattr(openalex_mod.urllib_request, "urlopen", _fake_urlopen)
    result = _run(
        OpenAlexWorksTool(),
        {"action": "fetch_openalex_record", "work_id": "doi:10.1000/missing"},
    )
    assert result.details["ok"] is False
    assert result.details["error"] == "doi_not_found"


def test_openalex_works_fetch_requires_work_id():
    result = _run(OpenAlexWorksTool(), {"action": "fetch_openalex_record"})
    assert result.details["ok"] is False
    assert "requires work_id" in result.content[0].text


def test_openalex_works_handles_http_error(monkeypatch):
    from agent.tools.search import openalex_works as openalex_mod

    def _fake_urlopen(req, timeout=0):
        raise urllib_error.HTTPError(
            url=req.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"rate_limited"}'),
        )

    monkeypatch.setattr(openalex_mod.urllib_request, "urlopen", _fake_urlopen)
    result = _run(OpenAlexWorksTool(), {"action": "search", "query": "rate limit test"})
    assert result.details["ok"] is False
    assert "HTTP 429" in result.content[0].text


def test_compatibility_imports_point_to_search_tools():
    from agent.tools.example_tools import WebSnapshotTool as ExampleWebSnapshotTool
    from agent.tools.perplexity_search import PerplexitySearchTool as CompatPerplexitySearchTool
    from agent.tools.search import (
        JinaWebSnapshotTool,
        OpenAlexWorksTool as SearchOpenAlexWorksTool,
        PerplexitySearchTool as SearchPerplexitySearchTool,
    )

    assert ExampleWebSnapshotTool is JinaWebSnapshotTool
    assert CompatPerplexitySearchTool is SearchPerplexitySearchTool
    assert SearchOpenAlexWorksTool().name == "openalex_works"
    assert ExampleWebSnapshotTool().name == "jina_web_snapshot"
