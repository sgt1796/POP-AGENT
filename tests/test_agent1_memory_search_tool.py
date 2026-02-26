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
