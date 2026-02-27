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
        file_read_tool=_dummy_tool("file_read"),  # type: ignore[arg-type]
        gmail_fetch_tool=_dummy_tool("gmail_fetch"),  # type: ignore[arg-type]
        pdf_merge_tool=_dummy_tool("pdf_merge"),  # type: ignore[arg-type]
        include_demo_tools=False,
    )

    names = [tool.name for tool in tools]
    assert names == [
        "jina_web_snapshot",
        "perplexity_search",
        "perplexity_web_snapshot",
        "memory_search",
        "toolsmaker",
        "bash_exec",
        "file_read",
        "gmail_fetch",
        "pdf_merge",
    ]


def test_build_runtime_tools_includes_demo_tools_when_enabled():
    tools = build_runtime_tools(
        memory_search_tool=_dummy_tool("memory_search"),  # type: ignore[arg-type]
        toolsmaker_tool=_dummy_tool("toolsmaker"),  # type: ignore[arg-type]
        bash_exec_tool=_dummy_tool("bash_exec"),  # type: ignore[arg-type]
        file_read_tool=_dummy_tool("file_read"),  # type: ignore[arg-type]
        gmail_fetch_tool=_dummy_tool("gmail_fetch"),  # type: ignore[arg-type]
        pdf_merge_tool=_dummy_tool("pdf_merge"),  # type: ignore[arg-type]
        include_demo_tools=True,
    )

    names = [tool.name for tool in tools]
    assert names == [
        "jina_web_snapshot",
        "perplexity_search",
        "perplexity_web_snapshot",
        "memory_search",
        "toolsmaker",
        "bash_exec",
        "file_read",
        "gmail_fetch",
        "pdf_merge",
        "slow",
        "fast",
    ]


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
    monkeypatch.setattr(runtime, "ConversationMemory", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "DiskMemory", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "MemoryRetriever", _FakeRetriever)
    monkeypatch.setattr(runtime, "EmbeddingIngestionWorker", _FakeWorker)
    monkeypatch.setattr(runtime, "MemorySubscriber", lambda ingestion_worker: SimpleNamespace(on_event=lambda _e: None))
    monkeypatch.setattr(runtime, "ContextCompressor", lambda *a, **k: SimpleNamespace(maybe_compress=lambda *args, **kwargs: False))

    monkeypatch.setattr(runtime, "MemorySearchTool", lambda retriever: SimpleNamespace(name="memory_search"))
    monkeypatch.setattr(runtime, "ToolsmakerTool", lambda agent, allowed_capabilities: SimpleNamespace(name="toolsmaker"))
    monkeypatch.setattr(runtime, "FileReadTool", lambda workspace_root: SimpleNamespace(name="file_read"))
    monkeypatch.setattr(runtime, "GmailFetchTool", lambda workspace_root: SimpleNamespace(name="gmail_fetch"))
    monkeypatch.setattr(runtime, "PdfMergeTool", lambda workspace_root: SimpleNamespace(name="pdf_merge"))
    monkeypatch.setattr(runtime, "JinaWebSnapshotTool", lambda: SimpleNamespace(name="jina_web_snapshot"))
    monkeypatch.setattr(runtime, "PerplexitySearchTool", lambda: SimpleNamespace(name="perplexity_search"))
    monkeypatch.setattr(runtime, "PerplexityWebSnapshotTool", lambda: SimpleNamespace(name="perplexity_web_snapshot"))
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
    monkeypatch.setattr(runtime, "_generate_session_id", lambda: "session-test")

    session = runtime.create_runtime_session(enable_event_logger=False)

    assert isinstance(session, RuntimeSession)
    assert session.top_k >= 1
    assert session.toolsmaker_manual_approval is True
    assert session.bash_prompt_approval is True
    assert session.active_session_id == "session-test"
    assert session.auto_session_id == "session-test"
    assert session.auto_title_enabled is True
    assert [tool.name for tool in session.agent._tools][:9] == [
        "jina_web_snapshot",
        "perplexity_search",
        "perplexity_web_snapshot",
        "memory_search",
        "toolsmaker",
        "bash_exec",
        "file_read",
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


class _FakeMemory:
    def __init__(self, sessions):
        self._sessions = set(sessions)

    def has_session(self, session_id):
        return session_id in self._sessions


class _AutoTitleMemory:
    def __init__(self, sessions):
        self._sessions = set(sessions)
        self.renames = []

    def has_session(self, session_id):
        return session_id in self._sessions

    def rename_session(self, old_session_id, new_session_id):
        if old_session_id not in self._sessions:
            return False
        self._sessions.remove(old_session_id)
        self._sessions.add(new_session_id)
        self.renames.append((old_session_id, new_session_id))
        return True


class _AutoTitleRetriever:
    def __init__(self, short_memory=None, long_memory=None):
        self.short_term = short_memory
        self.long_term = long_memory
        self.default_sessions = []

    def set_default_session(self, session_id):
        self.default_sessions.append(session_id)


class _AutoTitleWorker:
    def __init__(self):
        self.flushed = 0
        self.active_sessions = []

    def set_active_session(self, session_id):
        self.active_sessions.append(session_id)

    async def flush(self):
        self.flushed += 1

    async def shutdown(self):
        return None


class _AutoTitleAgent:
    def __init__(self, provider="gemini", model_id="gemini-3-flash-preview", timeout_s=12.0):
        self.state = SimpleNamespace(model={"provider": provider, "id": model_id})
        self.request_timeout_s = timeout_s
        self.session_id = None


def _build_auto_title_session(
    *,
    active_session_id="session-a",
    provider="gemini",
    model_id="gemini-3-flash-preview",
    short_memory=None,
    long_memory=None,
    timeout_s=12.0,
):
    worker = _AutoTitleWorker()
    retriever = _AutoTitleRetriever(short_memory=short_memory, long_memory=long_memory)
    debug_lines = []
    session = RuntimeSession(
        agent=_AutoTitleAgent(provider=provider, model_id=model_id, timeout_s=timeout_s),
        retriever=retriever,
        ingestion_worker=worker,
        active_session_id=active_session_id,
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
        debug_log=debug_lines.append,
        auto_session_id=active_session_id,
        auto_title_enabled=True,
    )
    return session, worker, retriever, debug_lines


def test_normalize_session_title():
    assert runtime._normalize_session_title('  "Project kickoff"  ') == "Project kickoff"
    assert runtime._normalize_session_title("Hi") == ""
    assert runtime._normalize_session_title("A B C") == "A B C"


def test_ensure_unique_session_title_suffixes():
    short_memory = _FakeMemory({"Alpha"})
    assert runtime._ensure_unique_session_title("Alpha", short_memory=short_memory, long_memory=None) == "Alpha-2"
    assert runtime._ensure_unique_session_title("Beta", short_memory=short_memory, long_memory=None) == "Beta"


def test_auto_title_session_uses_prompt_function_without_subagent(monkeypatch):
    class _FakePromptFunction:
        init_calls = []
        execute_calls = []

        def __init__(self, sys_prompt="", prompt="", client=None):
            self.client = client
            _FakePromptFunction.init_calls.append((sys_prompt, prompt, client))

        def execute(self, *args, **kwargs):
            _FakePromptFunction.execute_calls.append((args, kwargs))
            return "Payroll Tax Filing Help"

    def _forbid_agent(*_args, **_kwargs):
        raise AssertionError("auto-title should not instantiate Agent")

    monkeypatch.setattr(runtime, "PromptFunction", _FakePromptFunction)
    monkeypatch.setattr(runtime, "Agent", _forbid_agent)

    short_memory = _AutoTitleMemory({"session-a"})
    long_memory = _AutoTitleMemory({"session-a"})
    session, worker, retriever, _ = _build_auto_title_session(short_memory=short_memory, long_memory=long_memory)

    asyncio.run(
        runtime._auto_title_session(
            session,
            user_message="Need help filing payroll tax",
            assistant_reply="I can walk you through the filing steps.",
        )
    )

    assert session.active_session_id == "Payroll Tax Filing Help"
    assert session.auto_session_id is None
    assert worker.flushed == 1
    assert short_memory.renames == [("session-a", "Payroll Tax Filing Help")]
    assert long_memory.renames == [("session-a", "Payroll Tax Filing Help")]
    assert retriever.default_sessions == ["Payroll Tax Filing Help"]
    assert session.agent.session_id == "Payroll Tax Filing Help"
    assert _FakePromptFunction.init_calls == [
        (
            "You generate concise session titles. Output a short, descriptive title only.",
            "",
            "gemini",
        )
    ]
    prompt_args, prompt_kwargs = _FakePromptFunction.execute_calls[0]
    assert "Generate a short session title (3-7 words)" in prompt_args[0]
    assert prompt_kwargs["tools"] == []
    assert prompt_kwargs["tool_choice"] == "auto"
    assert prompt_kwargs["model"] == "gemini-3-flash-preview"


def test_auto_title_session_caches_prompt_function_per_provider(monkeypatch):
    class _FakePromptFunction:
        init_clients = []
        execute_count = 0

        def __init__(self, sys_prompt="", prompt="", client=None):
            _FakePromptFunction.init_clients.append(client)

        def execute(self, *_args, **_kwargs):
            _FakePromptFunction.execute_count += 1
            if _FakePromptFunction.execute_count == 1:
                return "First Session Title"
            return "Second Session Title"

    monkeypatch.setattr(runtime, "PromptFunction", _FakePromptFunction)

    short_memory = _AutoTitleMemory({"session-a"})
    session, _, _, _ = _build_auto_title_session(short_memory=short_memory, long_memory=None)

    asyncio.run(
        runtime._auto_title_session(
            session,
            user_message="first request",
            assistant_reply="first reply",
        )
    )

    session.auto_title_enabled = True
    session.auto_session_id = session.active_session_id
    asyncio.run(
        runtime._auto_title_session(
            session,
            user_message="second request",
            assistant_reply="second reply",
        )
    )

    assert session.active_session_id == "Second Session Title"
    assert _FakePromptFunction.init_clients == ["gemini"]
    assert list(session.title_prompt_functions.keys()) == ["gemini"]


def test_auto_title_session_creates_new_prompt_function_when_provider_changes(monkeypatch):
    class _FakePromptFunction:
        init_clients = []
        execute_count = 0

        def __init__(self, sys_prompt="", prompt="", client=None):
            _FakePromptFunction.init_clients.append(client)

        def execute(self, *_args, **_kwargs):
            _FakePromptFunction.execute_count += 1
            if _FakePromptFunction.execute_count == 1:
                return "Gemini Session Title"
            return "OpenAI Session Title"

    monkeypatch.setattr(runtime, "PromptFunction", _FakePromptFunction)

    short_memory = _AutoTitleMemory({"session-a"})
    session, _, _, _ = _build_auto_title_session(short_memory=short_memory, long_memory=None)

    asyncio.run(
        runtime._auto_title_session(
            session,
            user_message="first request",
            assistant_reply="first reply",
        )
    )

    session.agent.state.model = {"provider": "openai", "id": "gpt-5-nano"}
    session.auto_title_enabled = True
    session.auto_session_id = session.active_session_id
    asyncio.run(
        runtime._auto_title_session(
            session,
            user_message="second request",
            assistant_reply="second reply",
        )
    )

    assert session.active_session_id == "OpenAI Session Title"
    assert _FakePromptFunction.init_clients == ["gemini", "openai"]
    assert sorted(session.title_prompt_functions.keys()) == ["gemini", "openai"]


def test_auto_title_session_failure_clears_auto_session_and_logs(monkeypatch):
    class _FailingPromptFunction:
        def __init__(self, sys_prompt="", prompt="", client=None):
            del sys_prompt, prompt, client

        def execute(self, *_args, **_kwargs):
            raise RuntimeError("prompt failure")

    monkeypatch.setattr(runtime, "PromptFunction", _FailingPromptFunction)

    session, _, _, debug_lines = _build_auto_title_session(short_memory=None, long_memory=None)
    warnings = []

    asyncio.run(
        runtime._auto_title_session(
            session,
            user_message="trigger failure",
            assistant_reply="trigger failure",
            on_warning=warnings.append,
        )
    )

    assert session.active_session_id == "session-a"
    assert session.auto_title_enabled is False
    assert session.auto_session_id is None
    assert warnings and warnings[0].startswith("[session] auto-title failed: ")
    assert any(line.startswith("[session] auto-title failed: ") for line in debug_lines)


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
