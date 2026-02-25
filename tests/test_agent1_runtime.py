import asyncio
from types import SimpleNamespace

from agent.agent_types import TextContent
from agent_build.agent1 import runtime
from agent_build.agent1.runtime import RuntimeSession, build_runtime_tools, run_user_turn, shutdown_runtime_session


def _dummy_tool(name: str):
    return SimpleNamespace(name=name)


def test_build_runtime_tools_excludes_demo_tools_by_default():
    tools = build_runtime_tools(
        memory_search_tool=_dummy_tool("memory_search"),  # type: ignore[arg-type]
        toolsmaker_tool=_dummy_tool("toolsmaker"),  # type: ignore[arg-type]
        bash_exec_tool=_dummy_tool("bash_exec"),  # type: ignore[arg-type]
        gmail_fetch_tool=_dummy_tool("gmail_fetch"),  # type: ignore[arg-type]
        pdf_merge_tool=_dummy_tool("pdf_merge"),  # type: ignore[arg-type]
        include_demo_tools=False,
    )

    names = [tool.name for tool in tools]
    assert names == ["websnapshot", "memory_search", "toolsmaker", "bash_exec", "gmail_fetch", "pdf_merge"]


def test_build_runtime_tools_includes_demo_tools_when_enabled():
    tools = build_runtime_tools(
        memory_search_tool=_dummy_tool("memory_search"),  # type: ignore[arg-type]
        toolsmaker_tool=_dummy_tool("toolsmaker"),  # type: ignore[arg-type]
        bash_exec_tool=_dummy_tool("bash_exec"),  # type: ignore[arg-type]
        gmail_fetch_tool=_dummy_tool("gmail_fetch"),  # type: ignore[arg-type]
        pdf_merge_tool=_dummy_tool("pdf_merge"),  # type: ignore[arg-type]
        include_demo_tools=True,
    )

    names = [tool.name for tool in tools]
    assert names == ["websnapshot", "memory_search", "toolsmaker", "bash_exec", "gmail_fetch", "pdf_merge", "slow", "fast"]


class _FakeAgent:
    def __init__(self, _opts=None):
        self._model = None
        self._timeout = None
        self._system_prompt = ""
        self._tools = []
        self._subscribers = []
        self.prompts = []
        self.state = SimpleNamespace(messages=[])

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

    async def prompt(self, text):
        self.prompts.append(text)
        self.state.messages.append(
            SimpleNamespace(role="assistant", content=[TextContent(type="text", text="ok")], error_message=None)
        )


class _FakeWorker:
    def __init__(self, *_, **__):
        self.started = False
        self.flushed = 0
        self.shutdowns = 0

    def start(self):
        self.started = True

    def set_active_session(self, _session_id):
        return None

    async def flush(self):
        self.flushed += 1

    async def shutdown(self):
        self.shutdowns += 1


class _FakeRetriever:
    def __init__(self, *_, **__):
        self.calls = []

    def retrieve_sections(self, query, top_k=3, scope="both", session_id="default"):
        self.calls.append((query, top_k, scope, session_id))
        return ["user: previous"], ["assistant: previous"]


class _FailingRetriever(_FakeRetriever):
    def retrieve_sections(self, query, top_k=3, scope="both", session_id="default"):
        self.calls.append((query, top_k, scope, session_id))
        raise RuntimeError("retrieval failed")


def test_create_runtime_session_builds_shared_runtime(monkeypatch):
    monkeypatch.setenv("POP_AGENT_TOOLSMAKER_PROMPT_APPROVAL", "true")
    monkeypatch.setenv("POP_AGENT_TOOLSMAKER_AUTO_ACTIVATE", "true")
    monkeypatch.setenv("POP_AGENT_TOOLSMAKER_AUTO_CONTINUE", "true")
    monkeypatch.setenv("POP_AGENT_BASH_PROMPT_APPROVAL", "true")
    monkeypatch.setenv("POP_AGENT_INCLUDE_DEMO_TOOLS", "false")
    monkeypatch.setenv("POP_AGENT_MEMORY_TOP_K", "3")

    monkeypatch.setattr(runtime, "Agent", _FakeAgent)
    monkeypatch.setattr(runtime, "Embedder", lambda use_api: object())
    monkeypatch.setattr(runtime, "SessionConversationMemory", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "DiskMemory", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "MemoryRetriever", _FakeRetriever)
    monkeypatch.setattr(runtime, "EmbeddingIngestionWorker", _FakeWorker)
    monkeypatch.setattr(runtime, "MemorySubscriber", lambda ingestion_worker: SimpleNamespace(on_event=lambda _e: None))
    monkeypatch.setattr(runtime, "ContextCompressor", lambda *a, **k: SimpleNamespace(maybe_compress=lambda *args, **kwargs: False))

    monkeypatch.setattr(runtime, "MemorySearchTool", lambda retriever: SimpleNamespace(name="memory_search"))
    monkeypatch.setattr(runtime, "ToolsmakerTool", lambda agent, allowed_capabilities: SimpleNamespace(name="toolsmaker"))
    monkeypatch.setattr(runtime, "GmailFetchTool", lambda workspace_root: SimpleNamespace(name="gmail_fetch"))
    monkeypatch.setattr(runtime, "PdfMergeTool", lambda workspace_root: SimpleNamespace(name="pdf_merge"))
    monkeypatch.setattr(runtime, "WebSnapshotTool", lambda: SimpleNamespace(name="websnapshot"))
    monkeypatch.setattr(runtime, "SlowTool", lambda: SimpleNamespace(name="slow"))
    monkeypatch.setattr(runtime, "FastTool", lambda: SimpleNamespace(name="fast"))

    monkeypatch.setattr(runtime, "BashExecConfig", lambda **kwargs: SimpleNamespace(**kwargs))

    class _FakeBashExecTool:
        def __init__(self, config, approval_fn=None):
            self.name = "bash_exec"
            self.config = config
            self.approval_fn = approval_fn
            self.description = ""

    monkeypatch.setattr(runtime, "BashExecTool", _FakeBashExecTool)
    monkeypatch.setattr(runtime, "build_system_prompt", lambda **kwargs: "prompt")

    session = runtime.create_runtime_session(enable_event_logger=False)

    assert isinstance(session, RuntimeSession)
    assert session.top_k >= 1
    assert session.toolsmaker_manual_approval is True
    assert session.bash_prompt_approval is True
    assert [tool.name for tool in session.agent._tools][:6] == [
        "websnapshot",
        "memory_search",
        "toolsmaker",
        "bash_exec",
        "gmail_fetch",
        "pdf_merge",
    ]


def test_run_user_turn_builds_augmented_prompt_and_flushes():
    agent = _FakeAgent()
    retriever = _FakeRetriever()
    worker = _FakeWorker()

    session = RuntimeSession(
        agent=agent,
        retriever=retriever,
        ingestion_worker=worker,
        active_session_id="default",
        context_compressor=SimpleNamespace(maybe_compress=lambda *a, **k: False),
        top_k=3,
        toolsmaker_manual_approval=True,
        toolsmaker_auto_activate=True,
        toolsmaker_auto_continue=True,
        bash_prompt_approval=True,
        execution_profile="balanced",
        include_demo_tools=False,
        unsubscribe_log=lambda: None,
        unsubscribe_memory=lambda: None,
        unsubscribe_approval=lambda: None,
    )

    reply = asyncio.run(run_user_turn(session, "hello"))

    assert reply == "ok"
    assert worker.flushed == 1
    assert retriever.calls == [("hello", 3, "both", "default")]
    assert "|Current user message|:\nhello" in agent.prompts[0]


def test_run_user_turn_reports_memory_warning_when_retrieval_fails():
    agent = _FakeAgent()
    retriever = _FailingRetriever()
    worker = _FakeWorker()
    warnings = []

    session = RuntimeSession(
        agent=agent,
        retriever=retriever,
        ingestion_worker=worker,
        active_session_id="default",
        context_compressor=SimpleNamespace(maybe_compress=lambda *a, **k: False),
        top_k=3,
        toolsmaker_manual_approval=True,
        toolsmaker_auto_activate=True,
        toolsmaker_auto_continue=True,
        bash_prompt_approval=True,
        execution_profile="balanced",
        include_demo_tools=False,
        unsubscribe_log=lambda: None,
        unsubscribe_memory=lambda: None,
        unsubscribe_approval=lambda: None,
    )

    reply = asyncio.run(run_user_turn(session, "hello", on_warning=warnings.append))

    assert reply == "ok"
    assert warnings == ["[memory] retrieval warning: retrieval failed"]
    assert "(no relevant memories)" in agent.prompts[0]


def test_shutdown_runtime_session_unsubscribes_even_when_worker_raises():
    calls = []

    class _FailingWorker(_FakeWorker):
        async def shutdown(self):
            self.shutdowns += 1
            raise RuntimeError("boom")

    session = RuntimeSession(
        agent=_FakeAgent(),
        retriever=_FakeRetriever(),
        ingestion_worker=_FailingWorker(),
        active_session_id="default",
        context_compressor=SimpleNamespace(maybe_compress=lambda *a, **k: False),
        top_k=3,
        toolsmaker_manual_approval=True,
        toolsmaker_auto_activate=True,
        toolsmaker_auto_continue=True,
        bash_prompt_approval=True,
        execution_profile="balanced",
        include_demo_tools=False,
        unsubscribe_log=lambda: calls.append("log"),
        unsubscribe_memory=lambda: calls.append("memory"),
        unsubscribe_approval=lambda: calls.append("approval"),
    )

    try:
        asyncio.run(shutdown_runtime_session(session))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == "boom"

    assert calls == ["memory", "log", "approval"]


def test_build_turn_usage_line_reports_delta_when_calls_increase():
    class _UsageAgent:
        def get_usage_summary(self):
            return {
                "calls": 3,
                "input_tokens": 120,
                "output_tokens": 40,
                "total_tokens": 160,
                "provider_calls": 2,
                "estimated_calls": 1,
                "hybrid_calls": 0,
                "anomaly_calls": 0,
            }

        def get_last_usage(self):
            return {"source": "provider"}

    before = {
        "calls": 2,
        "input_tokens": 100,
        "output_tokens": 30,
        "total_tokens": 130,
        "provider_calls": 1,
        "estimated_calls": 1,
        "hybrid_calls": 0,
        "anomaly_calls": 0,
    }
    line = runtime._build_turn_usage_line(_UsageAgent(), before)  # type: ignore[arg-type]

    assert line.startswith("[usage] turn")
    assert "calls=1" in line
    assert "total=30" in line
    assert "source=provider" in line


def test_build_turn_usage_line_returns_empty_when_no_usage_calls():
    class _UsageAgent:
        def get_usage_summary(self):
            return {"calls": 1, "total_tokens": 10}

        def get_last_usage(self):
            return {"source": "estimate"}

    line = runtime._build_turn_usage_line(_UsageAgent(), {"calls": 1, "total_tokens": 10})  # type: ignore[arg-type]
    assert line == ""
