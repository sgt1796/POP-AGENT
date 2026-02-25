from __future__ import annotations

import asyncio
import builtins
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict

from agent.tools import JinaWebSnapshotTool, PerplexitySearchTool, PerplexityWebSnapshotTool


def _run(tool: Any, params: Dict[str, Any]):
    return asyncio.run(tool.execute("tc1", params))


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


def test_perplexity_web_snapshot_stub_response():
    result = _run(PerplexityWebSnapshotTool(), {"url": "https://example.com"})
    assert result.details["ok"] is False
    assert result.details["implemented"] is False
    assert result.details["tool"] == "perplexity_web_snapshot"
    assert "not implemented yet" in result.content[0].text


def test_compatibility_imports_point_to_search_tools():
    from agent.tools.example_tools import WebSnapshotTool as ExampleWebSnapshotTool
    from agent.tools.perplexity_search import PerplexitySearchTool as CompatPerplexitySearchTool
    from agent.tools.search import JinaWebSnapshotTool, PerplexitySearchTool as SearchPerplexitySearchTool

    assert ExampleWebSnapshotTool is JinaWebSnapshotTool
    assert CompatPerplexitySearchTool is SearchPerplexitySearchTool
    assert ExampleWebSnapshotTool().name == "jina_web_snapshot"
