from types import SimpleNamespace

from agent_build.agent1 import runtime


class _FakeAgent:
    def __init__(self, _opts=None):
        self._model = None
        self._timeout = None
        self._system_prompt = ""
        self._tools = []
        self._subscribers = []

    def set_model(self, model):
        self._model = model

    def set_timeout(self, timeout):
        self._timeout = timeout

    def set_system_prompt(self, prompt):
        self._system_prompt = prompt

    def set_tools(self, tools):
        self._tools = list(tools)

    def subscribe(self, fn):
        self._subscribers.append(fn)

        def _unsubscribe():
            if fn in self._subscribers:
                self._subscribers.remove(fn)

        return _unsubscribe


class _FakeWorker:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def start(self):
        return None

    async def flush(self):
        return None

    async def shutdown(self):
        return None


class _FakeRetriever:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def retrieve_sections(self, query, top_k=3, scope="both"):
        del query, top_k, scope
        return [], []


class _FakeBashExecTool:
    def __init__(self, config, approval_fn=None):
        self.name = "bash_exec"
        self.config = config
        self.approval_fn = approval_fn
        self.description = ""


def _dummy_tool(name: str):
    return SimpleNamespace(name=name)


def test_runtime_overrides_backward_compat(monkeypatch):
    monkeypatch.setattr(runtime, "Agent", _FakeAgent)
    monkeypatch.setattr(runtime, "Embedder", lambda use_api: object())
    monkeypatch.setattr(runtime, "ConversationMemory", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "DiskMemory", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "MemoryRetriever", _FakeRetriever)
    monkeypatch.setattr(runtime, "EmbeddingIngestionWorker", _FakeWorker)
    monkeypatch.setattr(runtime, "MemorySubscriber", lambda ingestion_worker: SimpleNamespace(on_event=lambda _e: None))

    monkeypatch.setattr(runtime, "MemorySearchTool", lambda retriever: _dummy_tool("memory_search"))
    monkeypatch.setattr(runtime, "ToolsmakerTool", lambda agent, allowed_capabilities: _dummy_tool("toolsmaker"))
    monkeypatch.setattr(runtime, "GmailFetchTool", lambda workspace_root: _dummy_tool("gmail_fetch"))
    monkeypatch.setattr(runtime, "PdfMergeTool", lambda workspace_root: _dummy_tool("pdf_merge"))
    monkeypatch.setattr(runtime, "JinaWebSnapshotTool", lambda: _dummy_tool("jina_web_snapshot"))
    monkeypatch.setattr(runtime, "PerplexitySearchTool", lambda: _dummy_tool("perplexity_search"))
    monkeypatch.setattr(runtime, "PerplexityWebSnapshotTool", lambda: _dummy_tool("perplexity_web_snapshot"))
    monkeypatch.setattr(runtime, "SlowTool", lambda: _dummy_tool("slow"))
    monkeypatch.setattr(runtime, "FastTool", lambda: _dummy_tool("fast"))

    monkeypatch.setattr(runtime, "BashExecConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(runtime, "BashExecTool", _FakeBashExecTool)
    monkeypatch.setattr(runtime, "build_system_prompt", lambda **kwargs: "prompt")

    # Backward compatibility: no overrides still works.
    session_default = runtime.create_runtime_session(enable_event_logger=False)
    assert session_default.agent._model["id"] == "gemini-3-flash-preview"

    session_overridden = runtime.create_runtime_session(
        enable_event_logger=False,
        overrides=runtime.RuntimeOverrides(
            enable_memory=False,
            include_tools=["jina_web_snapshot"],
            exclude_tools=["toolsmaker"],
            model_override={"provider": "openai", "id": "gpt-test", "api": None},
            bash_prompt_approval=False,
            toolsmaker_manual_approval=False,
            toolsmaker_auto_continue=False,
            log_level="quiet",
        ),
    )

    assert session_overridden.agent._model["id"] == "gpt-test"
    assert [tool.name for tool in session_overridden.agent._tools] == ["jina_web_snapshot"]
    assert session_overridden.bash_prompt_approval is False
    assert session_overridden.toolsmaker_manual_approval is False
    assert session_overridden.toolsmaker_auto_continue is False
