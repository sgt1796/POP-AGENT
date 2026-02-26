import asyncio
import os
import re
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set

from POP.embedder import Embedder
from POP.stream import stream

from agent import Agent
from agent.agent_types import AgentTool
from agent.tools import (
    BashExecConfig,
    BashExecTool,
    FastTool,
    GmailFetchTool,
    JinaWebSnapshotTool,
    MemorySearchTool,
    PerplexitySearchTool,
    PerplexityWebSnapshotTool,
    PdfMergeTool,
    SlowTool,
    ToolsmakerTool,
)

from .approvals import (
    BashExecApprovalPrompter,
    ToolsmakerApprovalSubscriber,
    ToolsmakerAutoContinueSubscriber,
)
from .constants import BASH_GIT_READ_SUBCOMMANDS, BASH_READ_COMMANDS, BASH_WRITE_COMMANDS, LOG_LEVELS
from .env_utils import (
    parse_bool_env,
    parse_float_env,
    parse_int_env,
    parse_path_list_env,
    parse_toolsmaker_allowed_capabilities,
    sorted_csv,
)
from .event_logger import make_event_logger, resolve_log_level
from .memory import (
    ContextCompressor,
    SessionConversationMemory,
    DiskMemory,
    EmbeddingIngestionWorker,
    MemoryRetriever,
    MemorySubscriber,
    build_augmented_prompt,
    format_memory_sections,
)
from .message_utils import extract_latest_assistant_text
from .prompting import build_system_prompt, resolve_execution_profile
from .usage_reporting import format_turn_usage_line, usage_delta

ConversationMemory = SessionConversationMemory


class ManualToolsmakerSubscriberFactory(Protocol):
    def __call__(self, agent: Agent) -> Any:
        ...


@dataclass
class RuntimeSession:
    agent: Agent
    retriever: MemoryRetriever
    ingestion_worker: EmbeddingIngestionWorker
    active_session_id: str
    context_compressor: ContextCompressor
    top_k: int
    toolsmaker_manual_approval: bool
    toolsmaker_auto_activate: bool
    toolsmaker_auto_continue: bool
    bash_prompt_approval: bool
    execution_profile: str
    include_demo_tools: bool
    unsubscribe_log: Callable[[], None]
    unsubscribe_memory: Callable[[], None]
    unsubscribe_approval: Callable[[], None]
    debug_log: Optional[Callable[[str], None]] = None
    auto_session_id: Optional[str] = None
    auto_title_enabled: bool = False
    auto_title_task: Optional[asyncio.Task[None]] = None


@dataclass
class RuntimeOverrides:
    long_memory_base_path: Optional[str] = None
    enable_memory: Optional[bool] = None
    include_tools: Optional[List[str]] = None
    exclude_tools: Optional[List[str]] = None
    model_override: Optional[Dict[str, Any]] = None
    bash_prompt_approval: Optional[bool] = None
    toolsmaker_manual_approval: Optional[bool] = None
    toolsmaker_auto_continue: Optional[bool] = None
    log_level: Optional[str] = None


class _NoopRetriever:
    def __init__(self, default_session_id: str = "default") -> None:
        self.default_session_id = str(default_session_id or "default").strip() or "default"
        self.short_term = None
        self.long_term = None

    def set_default_session(self, session_id: str) -> None:
        self.default_session_id = str(session_id or "default").strip() or "default"

    def retrieve_sections(
        self,
        query: str,
        top_k: int = 3,
        scope: str = "both",
        session_id: Optional[str] = None,
    ) -> tuple[list[str], list[str]]:
        del query, top_k, scope, session_id
        return [], []

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        scope: str = "both",
        session_id: Optional[str] = None,
    ) -> list[str]:
        del query, top_k, scope, session_id
        return []


class _NoopIngestionWorker:
    def start(self) -> None:
        return None

    def set_active_session(self, session_id: str) -> None:
        del session_id
        return None

    async def flush(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


def _resolve_bool_override(value: Optional[bool], default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _normalize_tool_name_set(values: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if values is None:
        return None
    names = {str(item).strip() for item in values if str(item).strip()}
    return names if names else None


def _filter_runtime_tools(
    tools: List[AgentTool],
    include_tools: Optional[Sequence[str]],
    exclude_tools: Optional[Sequence[str]],
) -> List[AgentTool]:
    include_names = _normalize_tool_name_set(include_tools)
    exclude_names = _normalize_tool_name_set(exclude_tools) or set()

    filtered = list(tools)
    if include_names is not None:
        filtered = [tool for tool in filtered if getattr(tool, "name", "") in include_names]
    if exclude_names:
        filtered = [tool for tool in filtered if getattr(tool, "name", "") not in exclude_names]
    return filtered


def _generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"session-{timestamp}-{suffix}"


def _normalize_session_title(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    value = " ".join(value.split())
    if not value:
        return ""
    value = value[:60].strip()
    if len(re.findall(r"[A-Za-z0-9]", value)) < 3:
        return ""
    return value


def _session_id_taken(
    candidate: str,
    *,
    short_memory: Optional[SessionConversationMemory],
    long_memory: Optional[DiskMemory],
) -> bool:
    if not candidate:
        return False
    if short_memory is not None and hasattr(short_memory, "has_session"):
        try:
            if short_memory.has_session(candidate):
                return True
        except Exception:
            pass
    if long_memory is not None and hasattr(long_memory, "has_session"):
        try:
            if long_memory.has_session(candidate):
                return True
        except Exception:
            pass
    return False


def _ensure_unique_session_title(
    title: str,
    *,
    short_memory: Optional[SessionConversationMemory],
    long_memory: Optional[DiskMemory],
) -> str:
    base = title
    if not _session_id_taken(base, short_memory=short_memory, long_memory=long_memory):
        return base
    suffix = 2
    while True:
        suffix_text = f"-{suffix}"
        max_len = max(1, 60 - len(suffix_text))
        trimmed = base[:max_len].rstrip()
        candidate = f"{trimmed}{suffix_text}"
        if not _session_id_taken(candidate, short_memory=short_memory, long_memory=long_memory):
            return candidate
        suffix += 1


async def _read_input(prompt: str) -> str:
    return input(prompt)


def build_runtime_tools(
    *,
    memory_search_tool: MemorySearchTool,
    toolsmaker_tool: ToolsmakerTool,
    bash_exec_tool: BashExecTool,
    gmail_fetch_tool: GmailFetchTool,
    pdf_merge_tool: PdfMergeTool,
    include_demo_tools: bool,
) -> List[AgentTool]:
    tools: List[AgentTool] = [
        JinaWebSnapshotTool(),
        PerplexitySearchTool(),
        PerplexityWebSnapshotTool(),
        memory_search_tool,
        toolsmaker_tool,
        bash_exec_tool,
        gmail_fetch_tool,
        pdf_merge_tool,
    ]
    if include_demo_tools:
        tools.extend([SlowTool(), FastTool()])
    return tools


def _build_title_prompt(user_message: str, assistant_reply: str) -> str:
    user_text = " ".join(str(user_message or "").strip().split())
    assistant_text = " ".join(str(assistant_reply or "").strip().split())
    if len(user_text) > 800:
        user_text = user_text[:800].rstrip()
    if len(assistant_text) > 800:
        assistant_text = assistant_text[:800].rstrip()
    return (
        "Generate a short session title (3-7 words) that captures the user's request.\n"
        "Return only the title, no quotes or extra text.\n\n"
        f"User: {user_text}\n"
        f"Assistant: {assistant_text}\n"
    )


async def _auto_title_session(
    session: RuntimeSession,
    *,
    user_message: str,
    assistant_reply: str,
    on_warning: Optional[Callable[[str], None]] = None,
) -> None:
    if not session.auto_title_enabled:
        return
    session.auto_title_enabled = False
    if session.auto_session_id is None:
        return
    if session.active_session_id != session.auto_session_id:
        return

    old_session_id = session.active_session_id
    title_prompt = _build_title_prompt(user_message, assistant_reply)

    title_agent = Agent({"stream_fn": stream})
    model = getattr(session.agent.state, "model", {}) or {}
    if isinstance(model, dict) and model:
        title_agent.set_model(dict(model))
    timeout = getattr(session.agent, "request_timeout_s", None)
    if timeout is None:
        timeout = 30.0
    try:
        title_agent.set_timeout(min(30.0, float(timeout)))
    except Exception:
        title_agent.set_timeout(30.0)
    title_agent.set_system_prompt(
        "You generate concise session titles. Output a short, descriptive title only."
    )
    title_agent.set_tools([])

    try:
        await title_agent.prompt(title_prompt)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if on_warning is not None:
            on_warning(f"[session] auto-title failed: {exc}")
        if session.debug_log is not None:
            session.debug_log(f"[session] auto-title failed: {exc}")
        session.auto_session_id = None
        return

    raw_title = extract_latest_assistant_text(title_agent)
    normalized = _normalize_session_title(raw_title)
    if not normalized:
        session.auto_session_id = None
        if session.debug_log is not None:
            session.debug_log("[session] auto-title failed: invalid title")
        return
    unique_title = _ensure_unique_session_title(
        normalized,
        short_memory=session.retriever.short_term,
        long_memory=session.retriever.long_term,
    )

    try:
        await session.ingestion_worker.flush()
    except Exception:
        pass

    short_memory = session.retriever.short_term
    long_memory = session.retriever.long_term
    short_needs_rename = False
    long_needs_rename = False
    if short_memory is not None and hasattr(short_memory, "has_session"):
        try:
            short_needs_rename = short_memory.has_session(old_session_id)
        except Exception:
            short_needs_rename = False
    if long_memory is not None and hasattr(long_memory, "has_session"):
        try:
            long_needs_rename = long_memory.has_session(old_session_id)
        except Exception:
            long_needs_rename = False

    short_renamed = False
    long_renamed = False
    rename_failed = False

    if short_needs_rename and hasattr(short_memory, "rename_session"):
        try:
            short_renamed = bool(short_memory.rename_session(old_session_id, unique_title))
        except Exception:
            short_renamed = False
        if not short_renamed:
            rename_failed = True

    if long_needs_rename and long_memory is not None and hasattr(long_memory, "rename_session"):
        try:
            long_renamed = bool(long_memory.rename_session(old_session_id, unique_title))
        except Exception:
            long_renamed = False
        if not long_renamed:
            rename_failed = True

    if rename_failed:
        if short_renamed and hasattr(short_memory, "rename_session"):
            try:
                short_memory.rename_session(unique_title, old_session_id)
            except Exception:
                pass
        if long_renamed and long_memory is not None and hasattr(long_memory, "rename_session"):
            try:
                long_memory.rename_session(unique_title, old_session_id)
            except Exception:
                pass
        session.auto_session_id = None
        if session.debug_log is not None:
            session.debug_log("[session] auto-title failed: rename failed")
        return

    _set_active_session(session, unique_title)
    session.auto_session_id = None
    if session.debug_log is not None:
        session.debug_log(f"[session] auto-titled: {old_session_id} -> {unique_title}")


def _track_auto_title_task(session: RuntimeSession, task: asyncio.Task[None]) -> None:
    session.auto_title_task = task

    def _clear(_task: asyncio.Task[None]) -> None:
        try:
            _task.exception()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        if session.auto_title_task is _task:
            session.auto_title_task = None

    task.add_done_callback(_clear)


def _set_active_session(session: RuntimeSession, new_id: str) -> None:
    sid = str(new_id or "").strip() or "default"
    session.active_session_id = sid
    try:
        session.ingestion_worker.set_active_session(sid)
    except Exception:
        pass
    if hasattr(session.retriever, "set_default_session"):
        try:
            session.retriever.set_default_session(sid)
        except Exception:
            pass
    try:
        session.agent.session_id = sid
    except Exception:
        pass


def switch_session(session: RuntimeSession, new_id: str) -> None:
    next_session = str(new_id or "").strip()
    if not next_session:
        return
    task = session.auto_title_task
    session.auto_title_task = None
    session.auto_title_enabled = False
    session.auto_session_id = None
    if task is not None and not task.done():
        task.cancel()
    _set_active_session(session, next_session)
    if session.debug_log is not None:
        session.debug_log(f"[session] switched to '{next_session}'")


def create_runtime_session(
    *,
    log_level: Optional[str] = None,
    enable_event_logger: bool = True,
    debug_log: Optional[Callable[[str], None]] = None,
    bash_approval_fn: Optional[Callable[[dict], Any]] = None,
    manual_toolsmaker_subscriber_factory: Optional[ManualToolsmakerSubscriberFactory] = None,
    overrides: Optional[RuntimeOverrides] = None,
) -> RuntimeSession:
    if overrides is None:
        overrides = RuntimeOverrides()

    agent = Agent({"stream_fn": stream})
    agent.set_model({"provider": "gemini", "id": "gemini-3-flash-preview", "api": None})
    if isinstance(overrides.model_override, dict) and overrides.model_override:
        agent.set_model(dict(overrides.model_override))
    agent.set_timeout(120)

    initial_session_id = _generate_session_id()
    memory_enabled = _resolve_bool_override(overrides.enable_memory, True)
    long_memory_base_path = str(overrides.long_memory_base_path or os.path.join("agent", "mem", "history.jsonl")).strip()
    if not long_memory_base_path:
        long_memory_base_path = os.path.join("agent", "mem", "history.jsonl")

    if memory_enabled:
        embedder = Embedder(use_api="openai")
        short_memory = ConversationMemory(embedder=embedder, max_entries_per_session=100, max_sessions=100)
        long_memory = DiskMemory(filepath=long_memory_base_path, embedder=embedder, max_entries=1000)
        retriever: Any = MemoryRetriever(
            short_term=short_memory,
            long_term=long_memory,
            default_session_id=initial_session_id,
        )
        ingestion_worker: Any = EmbeddingIngestionWorker(memory=short_memory, long_term=long_memory)
        if hasattr(ingestion_worker, "set_active_session"):
            try:
                ingestion_worker.set_active_session(initial_session_id)
            except Exception:
                pass
        ingestion_worker.start()
        memory_subscriber = MemorySubscriber(ingestion_worker=ingestion_worker)
    else:
        retriever = _NoopRetriever(default_session_id=initial_session_id)
        ingestion_worker = _NoopIngestionWorker()
        memory_subscriber = None

    memory_search_tool = MemorySearchTool(retriever=retriever)
    toolsmaker_caps = parse_toolsmaker_allowed_capabilities(os.getenv("POP_AGENT_TOOLSMAKER_ALLOWED_CAPS"))
    toolsmaker_tool = ToolsmakerTool(agent=agent, allowed_capabilities=toolsmaker_caps)
    workspace_root = os.path.realpath(os.getcwd())
    gmail_fetch_tool = GmailFetchTool(workspace_root=workspace_root)
    pdf_merge_tool = PdfMergeTool(workspace_root=workspace_root)

    bash_allowed_roots = parse_path_list_env(
        "POP_AGENT_BASH_ALLOWED_ROOTS",
        default_paths=[workspace_root],
        base_dir=workspace_root,
    )
    bash_writable_roots = parse_path_list_env(
        "POP_AGENT_BASH_WRITABLE_ROOTS",
        default_paths=[workspace_root],
        base_dir=workspace_root,
    )
    bash_timeout_s = parse_float_env("POP_AGENT_BASH_TIMEOUT_S", 15.0)
    bash_max_output_chars = parse_int_env("POP_AGENT_BASH_MAX_OUTPUT_CHARS", 20_000)
    bash_prompt_approval = parse_bool_env("POP_AGENT_BASH_PROMPT_APPROVAL", True)
    bash_prompt_approval = _resolve_bool_override(overrides.bash_prompt_approval, bash_prompt_approval)
    effective_bash_approval_fn = None
    if bash_prompt_approval:
        effective_bash_approval_fn = bash_approval_fn if bash_approval_fn is not None else BashExecApprovalPrompter()

    bash_exec_tool = BashExecTool(
        BashExecConfig(
            project_root=workspace_root,
            allowed_roots=bash_allowed_roots,
            writable_roots=bash_writable_roots,
            read_commands=BASH_READ_COMMANDS,
            write_commands=BASH_WRITE_COMMANDS,
            git_read_subcommands=BASH_GIT_READ_SUBCOMMANDS,
            default_timeout_s=bash_timeout_s,
            max_timeout_s=60.0,
            default_max_output_chars=bash_max_output_chars,
            max_output_chars_limit=100_000,
        ),
        approval_fn=effective_bash_approval_fn,
    )

    bash_read_csv = sorted_csv(BASH_READ_COMMANDS)
    bash_write_csv = sorted_csv(BASH_WRITE_COMMANDS)
    bash_git_csv = sorted_csv(BASH_GIT_READ_SUBCOMMANDS)
    execution_profile = resolve_execution_profile(os.getenv("POP_AGENT_EXECUTION_PROFILE", "balanced"))
    toolsmaker_manual_approval = parse_bool_env("POP_AGENT_TOOLSMAKER_PROMPT_APPROVAL", True)
    toolsmaker_auto_activate = parse_bool_env("POP_AGENT_TOOLSMAKER_AUTO_ACTIVATE", True)
    toolsmaker_auto_continue = parse_bool_env("POP_AGENT_TOOLSMAKER_AUTO_CONTINUE", True)
    toolsmaker_manual_approval = _resolve_bool_override(
        overrides.toolsmaker_manual_approval,
        toolsmaker_manual_approval,
    )
    toolsmaker_auto_continue = _resolve_bool_override(
        overrides.toolsmaker_auto_continue,
        toolsmaker_auto_continue,
    )
    include_demo_tools = parse_bool_env("POP_AGENT_INCLUDE_DEMO_TOOLS", False)

    bash_exec_tool.description = (
        "Run one safe shell command without a shell. "
        f"Allowed read commands: {bash_read_csv}. "
        f"Allowed write commands: {bash_write_csv}. "
        f"For git, allowed subcommands: {bash_git_csv}. "
        + (
            "Write commands require approval."
            if bash_prompt_approval
            else "Without approval prompts, medium/high-risk write commands are denied."
        )
    )

    agent.set_system_prompt(
        build_system_prompt(
            bash_read_csv=bash_read_csv,
            bash_write_csv=bash_write_csv,
            bash_git_csv=bash_git_csv,
            bash_prompt_approval=bash_prompt_approval,
            toolsmaker_manual_approval=toolsmaker_manual_approval,
            toolsmaker_auto_continue=toolsmaker_auto_continue,
            execution_profile=execution_profile,
            workspace_root=workspace_root,
        )
    )
    tools = build_runtime_tools(
        memory_search_tool=memory_search_tool,
        toolsmaker_tool=toolsmaker_tool,
        bash_exec_tool=bash_exec_tool,
        gmail_fetch_tool=gmail_fetch_tool,
        pdf_merge_tool=pdf_merge_tool,
        include_demo_tools=include_demo_tools,
    )
    tools = _filter_runtime_tools(
        tools,
        include_tools=overrides.include_tools,
        exclude_tools=overrides.exclude_tools,
    )
    agent.set_tools(tools)
    try:
        agent.session_id = initial_session_id
    except Exception:
        pass

    effective_log_level = log_level
    if effective_log_level is None:
        effective_log_level = os.getenv("POP_AGENT_LOG_LEVEL", "quiet")
    if overrides.log_level is not None:
        effective_log_level = overrides.log_level

    if enable_event_logger:
        unsubscribe_log = agent.subscribe(make_event_logger(effective_log_level))
    else:
        unsubscribe_log = lambda: None

    if memory_subscriber is not None:
        unsubscribe_memory = agent.subscribe(memory_subscriber.on_event)
    else:
        unsubscribe_memory = lambda: None

    if toolsmaker_manual_approval:
        if manual_toolsmaker_subscriber_factory is not None:
            candidate = manual_toolsmaker_subscriber_factory(agent)
            subscriber = getattr(candidate, "on_event", candidate)
            unsubscribe_approval = agent.subscribe(subscriber)
        else:
            approval_subscriber = ToolsmakerApprovalSubscriber(
                agent=agent,
                auto_activate_default=toolsmaker_auto_activate,
            )
            unsubscribe_approval = agent.subscribe(approval_subscriber.on_event)
    elif toolsmaker_auto_continue:
        auto_continue_subscriber = ToolsmakerAutoContinueSubscriber(agent=agent)
        unsubscribe_approval = agent.subscribe(auto_continue_subscriber.on_event)
    else:
        unsubscribe_approval = lambda: None

    try:
        top_k = max(1, int(os.getenv("POP_AGENT_MEMORY_TOP_K", "3") or "3"))
    except Exception:
        top_k = 3

    compressor_trigger_chars = parse_int_env("POP_AGENT_CONTEXT_TRIGGER_CHARS", 20_000)
    compressor_target_chars = parse_int_env("POP_AGENT_CONTEXT_TARGET_CHARS", 12_000)
    context_compressor = ContextCompressor(
        trigger_chars=compressor_trigger_chars,
        target_keep_chars=compressor_target_chars,
    )

    session = RuntimeSession(
        agent=agent,
        retriever=retriever,
        ingestion_worker=ingestion_worker,
        active_session_id=initial_session_id,
        context_compressor=context_compressor,
        top_k=top_k,
        toolsmaker_manual_approval=toolsmaker_manual_approval,
        toolsmaker_auto_activate=toolsmaker_auto_activate,
        toolsmaker_auto_continue=toolsmaker_auto_continue,
        bash_prompt_approval=bash_prompt_approval,
        execution_profile=execution_profile,
        include_demo_tools=include_demo_tools,
        unsubscribe_log=unsubscribe_log,
        unsubscribe_memory=unsubscribe_memory,
        unsubscribe_approval=unsubscribe_approval,
        debug_log=debug_log,
        auto_session_id=initial_session_id,
        auto_title_enabled=True,
        auto_title_task=None,
    )
    if debug_log is not None:
        debug_log(f"[session] created id={initial_session_id}")
    return session


async def run_user_turn(
    session: RuntimeSession,
    user_message: str,
    on_warning: Optional[Callable[[str], None]] = None,
) -> str:
    await session.ingestion_worker.flush()

    memory_text = "(no relevant memories)"
    try:
        short_hits, long_hits = session.retriever.retrieve_sections(
            user_message,
            top_k=session.top_k,
            scope="both",
            session_id=session.active_session_id,
        )
        memory_text = format_memory_sections(short_hits, long_hits)
        if session.debug_log is not None:
            session.debug_log(
                "[memory] retrieved "
                f"session={session.active_session_id} short={len(short_hits)} "
                f"long={len(long_hits)} top_k={session.top_k}"
            )
    except Exception as exc:
        memory_text = "(no relevant memories)"
        if on_warning is not None:
            on_warning(f"[memory] retrieval warning: {exc}")

    session.context_compressor.maybe_compress(
        session.agent,
        session.active_session_id,
        long_term=getattr(session.retriever, "long_term", None),
    )
    augmented_prompt = build_augmented_prompt(user_message, memory_text)
    await session.agent.prompt(augmented_prompt)

    reply = extract_latest_assistant_text(session.agent)
    if not reply:
        reply = "(no assistant text returned)"
    if (
        session.auto_title_enabled
        and session.auto_session_id is not None
        and session.active_session_id == session.auto_session_id
        and (session.auto_title_task is None or session.auto_title_task.done())
    ):
        task = asyncio.create_task(
            _auto_title_session(
                session,
                user_message=user_message,
                assistant_reply=reply,
                on_warning=on_warning,
            )
        )
        _track_auto_title_task(session, task)
    return reply


def _build_turn_usage_line(agent: Agent, before_summary: Dict[str, Any]) -> str:
    get_summary = getattr(agent, "get_usage_summary", None)
    if not callable(get_summary):
        return ""
    after_summary_raw = get_summary()
    if not isinstance(after_summary_raw, dict):
        return ""
    delta = usage_delta(before_summary, after_summary_raw)
    if int(delta.get("calls", 0)) <= 0:
        return ""
    get_last = getattr(agent, "get_last_usage", None)
    last_usage_raw = get_last() if callable(get_last) else None
    last_usage = last_usage_raw if isinstance(last_usage_raw, dict) else None
    return format_turn_usage_line(delta, last_usage)


async def shutdown_runtime_session(session: RuntimeSession) -> None:
    shutdown_error: Optional[Exception] = None
    try:
        await session.ingestion_worker.shutdown()
    except Exception as exc:
        shutdown_error = exc
    finally:
        task = session.auto_title_task
        session.auto_title_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        session.unsubscribe_memory()
        session.unsubscribe_log()
        session.unsubscribe_approval()

    if shutdown_error is not None:
        raise shutdown_error


async def main() -> None:
    log_level_env = os.getenv("POP_AGENT_LOG_LEVEL", "quiet")
    debug_log = None
    if resolve_log_level(log_level_env) >= LOG_LEVELS["debug"]:
        debug_log = print
    session = create_runtime_session(debug_log=debug_log)

    print("POP Chatroom Agent (tools + embedding memory)")
    if session.toolsmaker_manual_approval:
        print(
            "[toolsmaker] manual approval prompts: on "
            f"(default auto-activate={'on' if session.toolsmaker_auto_activate else 'off'})"
        )
        print("[toolsmaker] auto-continue: off (manual approval mode)")
    else:
        print("[toolsmaker] manual approval prompts: off")
        if session.toolsmaker_auto_continue:
            print("[toolsmaker] auto-continue: on")
        else:
            print("[toolsmaker] auto-continue: off")
    print(f"[agent] execution profile: {session.execution_profile}")
    print(f"[tools] demo tools: {'on' if session.include_demo_tools else 'off'}")
    if session.bash_prompt_approval:
        print("[bash_exec] approval prompts: on")
    else:
        print("[bash_exec] approval prompts: off (medium/high commands will be denied)")
    print(f"[memory] active session: {session.active_session_id}")
    print("Type 'exit' or 'quit' to stop. Use /session <name> to switch sessions.\n")

    try:
        while True:
            try:
                user_message = (await _read_input("User: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if user_message.lower().startswith("/session "):
                next_session = user_message.split(" ", 1)[1].strip()
                if not next_session:
                    print("[session] usage: /session <name>\n")
                    continue
                switch_session(session, next_session)
                print(f"[session] switched to '{next_session}'\n")
                continue

            before_summary: Dict[str, Any] = {}
            get_summary = getattr(session.agent, "get_usage_summary", None)
            if callable(get_summary):
                summary = get_summary()
                if isinstance(summary, dict):
                    before_summary = summary

            try:
                reply = await run_user_turn(session, user_message, on_warning=print)
            except Exception as exc:
                print(f"Assistant error: {exc}\n")
                continue

            print(f"Assistant: {reply}\n")
            usage_line = _build_turn_usage_line(session.agent, before_summary)
            if usage_line:
                print(f"{usage_line}\n")
    finally:
        try:
            await shutdown_runtime_session(session)
        except Exception as exc:
            print(f"[memory] shutdown warning: {exc}")
