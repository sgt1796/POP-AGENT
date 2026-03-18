import asyncio

from agent.tools.agent1_tools import MemorySearchTool


class _FakeRetriever:
    def __init__(self) -> None:
        self.default_session_id = "alpha"
        self.calls = []

    def retrieve(self, query, top_k=3, scope="both", session_id="default"):
        self.calls.append((query, top_k, scope, session_id))
        return ["hit"]


def test_memory_search_tool_defaults_session_id():
    retriever = _FakeRetriever()
    tool = MemorySearchTool(retriever=retriever)

    result = asyncio.run(tool.execute("call-1", {"query": "hello"}, None, None))

    assert retriever.calls == [("hello", 3, "both", "alpha")]
    assert result.details["session_id"] == "alpha"


class _FallbackRetriever:
    def __init__(self) -> None:
        self.default_session_id = "alpha"
        self.retrieve_calls = []
        self.fallback_calls = []

    def retrieve(self, query, top_k=3, scope="both", session_id="default"):
        self.retrieve_calls.append((query, top_k, scope, session_id))
        return ["session hit"]

    def retrieve_with_fallback(self, query, top_k=3, scope="both", session_id="default"):
        self.fallback_calls.append((query, top_k, scope, session_id))
        return ["global hit"]


def test_memory_search_tool_uses_fallback_for_implicit_session():
    retriever = _FallbackRetriever()
    tool = MemorySearchTool(retriever=retriever)

    result = asyncio.run(tool.execute("call-1", {"query": "server"}, None, None))

    assert retriever.fallback_calls == [("server", 3, "both", "alpha")]
    assert retriever.retrieve_calls == []
    assert "global hit" in result.content[0].text


def test_memory_search_tool_honors_explicit_session_without_fallback():
    retriever = _FallbackRetriever()
    tool = MemorySearchTool(retriever=retriever)

    result = asyncio.run(tool.execute("call-1", {"query": "server", "session_id": "beta"}, None, None))

    assert retriever.fallback_calls == []
    assert retriever.retrieve_calls == [("server", 3, "both", "beta")]
    assert result.details["session_id"] == "beta"
