import asyncio
import os
import re
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, TextIO

from POP import PromptFunction
from POP.embedder import Embedder
from POP.stream import stream

from agent import Agent
from agent.agent_types import AgentTool
from agent.tools import (
    AgentMailSendTool,
    BashExecConfig,
    BashExecTool,
    DownloadUrlToFileTool,
    FileReadTool,
    FileWriteTool,
    FastTool,
    GmailFetchTool,
    JinaWebSnapshotTool,
    MemorySearchTool,
    OpenAlexWorksTool,
    PerplexitySearchTool,
    PerplexityWebSnapshotTool,
    PdfMergeTool,
    SlowTool,
    TaskSchedulerTool,
)

from .approvals import BashExecApprovalPrompter
from .constants import BASH_GIT_READ_SUBCOMMANDS, BASH_READ_COMMANDS, BASH_WRITE_COMMANDS, LOG_LEVELS
from .env_utils import (
    parse_bool_env,
    parse_float_env,
    parse_int_env,
    parse_path_list_env,
    sorted_csv,
)
from .event_logger import make_event_logger, resolve_log_level
from agent.memory import (
    ContextCompressor,
    ConversationMemory,
    DiskMemory,
    EmbeddingIngestionWorker,
    MemoryRetriever,
    MemorySubscriber,
    build_augmented_prompt,
    format_memory_sections,
)
from .message_utils import extract_latest_assistant_text, extract_texts
from .prompting import build_system_prompt, resolve_execution_profile
from .usage_reporting import format_turn_usage_line, usage_delta


@dataclass
class RuntimeSession:
    agent: Agent
    retriever: MemoryRetriever
    ingestion_worker: EmbeddingIngestionWorker
    active_session_id: str
    context_compressor: ContextCompressor
    top_k: int
    bash_prompt_approval: bool
    execution_profile: str
    include_demo_tools: bool
    unsubscribe_log: Callable[[], None]
    unsubscribe_memory: Callable[[], None]
    unsubscribe_approval: Callable[[], None]
    debug_log: Optional[Callable[[str], None]] = None
    unsubscribe_debug_file_log: Callable[[], None] = lambda: None
    close_debug_log_file: Optional[Callable[[], None]] = None
    auto_session_id: Optional[str] = None
    auto_title_enabled: bool = False
    auto_title_task: Optional[asyncio.Task[None]] = None
    title_prompt_functions: Dict[str, PromptFunction] = field(default_factory=dict)


@dataclass
class RuntimeOverrides:
    long_memory_base_path: Optional[str] = None
    enable_memory: Optional[bool] = None
    enable_auto_title: Optional[bool] = None
    include_tools: Optional[List[str]] = None
    exclude_tools: Optional[List[str]] = None
    model_override: Optional[Dict[str, Any]] = None
    bash_prompt_approval: Optional[bool] = None
    log_level: Optional[str] = None
    execution_profile: Optional[str] = None
    memory_top_k: Optional[int] = None


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


def _combine_debug_logs(
    primary: Optional[Callable[[str], None]],
    secondary: Optional[Callable[[str], None]],
) -> Optional[Callable[[str], None]]:
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    def _log(text: str) -> None:
        try:
            primary(text)
        except Exception:
            pass
        try:
            secondary(text)
        except Exception:
            pass

    return _log


def _make_debug_file_sink(
    raw_path: Optional[str],
) -> tuple[Optional[Callable[[str], None]], Optional[Callable[[], None]], Optional[str], Optional[str]]:
    path_value = str(raw_path or "").strip()
    if not path_value:
        return None, None, None, None

    resolved_path = os.path.realpath(path_value)
    parent_dir = os.path.dirname(resolved_path)
    if parent_dir:
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except Exception as exc:
            return None, None, resolved_path, f"unable to create directory '{parent_dir}': {exc}"

    handle: Optional[TextIO] = None
    try:
        handle = open(resolved_path, "a", encoding="utf-8")
    except Exception as exc:
        return None, None, resolved_path, f"unable to open '{resolved_path}': {exc}"

    def _sink(text: str) -> None:
        try:
            if handle is None:
                return
            handle.write(f"{text}\n")
            handle.flush()
        except Exception:
            pass

    def _close() -> None:
        try:
            if handle is not None:
                handle.close()
        except Exception:
            pass

    return _sink, _close, resolved_path, None


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
    short_memory: Optional[ConversationMemory],
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
    short_memory: Optional[ConversationMemory],
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
    bash_exec_tool: BashExecTool,
    download_url_to_file_tool: AgentTool,
    gmail_fetch_tool: GmailFetchTool,
    pdf_merge_tool: PdfMergeTool,
    agentmail_send_tool: AgentTool,
    include_demo_tools: bool,
    task_scheduler_tool: Optional[AgentTool] = None,
    file_read_tool: Optional[AgentTool] = None,
    file_write_tool: Optional[AgentTool] = None,
) -> List[AgentTool]:
    tools: List[AgentTool] = [
        JinaWebSnapshotTool(),
        PerplexitySearchTool(),
        OpenAlexWorksTool(),
        download_url_to_file_tool,
        PerplexityWebSnapshotTool(),
        memory_search_tool,
    ]
    if task_scheduler_tool is not None:
        tools.append(task_scheduler_tool)
    tools.append(bash_exec_tool)
    if file_read_tool is not None:
        tools.append(file_read_tool)
    if file_write_tool is not None:
        tools.append(file_write_tool)
    tools.extend([gmail_fetch_tool, pdf_merge_tool, agentmail_send_tool])
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


def _resolve_title_model(session: RuntimeSession) -> tuple[str, Optional[str]]:
    model = getattr(session.agent.state, "model", {}) or {}
    if not isinstance(model, dict):
        return "gemini", None
    provider_value = model.get("provider") or model.get("api")
    provider = str(provider_value or "").strip().lower() or "gemini"
    model_value = model.get("id")
    model_id = str(model_value or "").strip() or None
    return provider, model_id


def _get_title_prompt_function(session: RuntimeSession, provider: str) -> PromptFunction:
    key = str(provider or "").strip().lower() or "gemini"
    cached = session.title_prompt_functions.get(key)
    if cached is not None:
        return cached
    prompt_fn = PromptFunction(
        sys_prompt="You generate concise session titles. Output a short, descriptive title only.",
        prompt="",
        client=key,
    )
    session.title_prompt_functions[key] = prompt_fn
    return prompt_fn


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
    provider, model_id = _resolve_title_model(session)
    title_kwargs: Dict[str, Any] = {
        "tools": [],
        "tool_choice": "auto",
    }
    if model_id:
        title_kwargs["model"] = model_id

    try:
        title_prompt_fn = _get_title_prompt_function(session, provider)
        raw_title = title_prompt_fn.execute(title_prompt, **title_kwargs)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if on_warning is not None:
            on_warning(f"[session] auto-title failed: {exc}")
        if session.debug_log is not None:
            session.debug_log(f"[session] auto-title failed: {exc}")
        session.auto_session_id = None
        return

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


def _scheduled_task_runtime_error(agent: Any) -> str:
    state = getattr(agent, "state", None)
    return str(getattr(state, "error", "") or "").strip()


def _scheduled_task_failed_tool_result(agent: Any) -> Optional[Dict[str, str]]:
    state = getattr(agent, "state", None)
    messages = getattr(state, "messages", None)
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if getattr(message, "role", None) != "toolResult":
            continue
        details = getattr(message, "details", None)
        ok_value = details.get("ok") if isinstance(details, dict) else None
        is_error = bool(getattr(message, "is_error", False))
        if not is_error and ok_value is not False:
            return None

        tool_name = str(getattr(message, "tool_name", "") or "").strip()
        summary = "\n".join([text for text in extract_texts(message) if text.strip()]).strip()
        if not summary and isinstance(details, dict):
            error_value = str(details.get("error") or "").strip()
            if error_value:
                summary = f"{tool_name or 'tool'} error: {error_value}"
        if not summary:
            summary = f"{tool_name or 'tool'} failed"
        return {
            "tool_name": tool_name,
            "summary": summary,
        }
    return None


def _build_scheduled_task_prompt(task: Dict[str, Any]) -> str:
    prompt = str(task.get("prompt") or "").strip()
    task_id = str(task.get("id") or "").strip() or "(unknown)"
    task_name = str(task.get("task_name") or "").strip() or "(unnamed)"
    schedule_type = str(task.get("schedule_type") or "").strip() or "one_time"
    run_at = str(task.get("run_at") or "").strip() or "(not set)"
    next_run_at_utc = str(task.get("next_run_at_utc") or "").strip() or "(due now)"
    timezone = str(task.get("timezone") or "").strip() or "UTC"
    return "\n".join(
        [
            "Scheduled task execution context:",
            f"- task_id: {task_id}",
            f"- task_name: {task_name}",
            f"- schedule_type: {schedule_type}",
            f"- run_at: {run_at}",
            f"- timezone: {timezone}",
            f"- next_run_at_utc: {next_run_at_utc}",
            "",
            "This task is already due and is being executed now by the scheduled runner.",
            "Carry out the task prompt now using the available tools.",
            (
                "Relative timing phrases in the stored prompt refer to the original scheduling request; "
                "they do not mean to delay or schedule the task again from the current time."
            ),
            (
                "Do not call task_scheduler to recreate, delay, or requeue this task unless the task prompt "
                "explicitly asks you to manage schedules as part of the work."
            ),
            "",
            "Task prompt:",
            prompt,
        ]
    )


def _build_scheduled_task_overrides(overrides: Optional[RuntimeOverrides]) -> RuntimeOverrides:
    base = overrides or RuntimeOverrides()
    include_tools = list(base.include_tools) if base.include_tools is not None else None
    exclude_tools = [str(item).strip() for item in list(base.exclude_tools or []) if str(item).strip()]
    if "task_scheduler" not in exclude_tools:
        exclude_tools.append("task_scheduler")
    model_override = dict(base.model_override) if isinstance(base.model_override, dict) else None
    return RuntimeOverrides(
        long_memory_base_path=base.long_memory_base_path,
        enable_memory=base.enable_memory,
        include_tools=include_tools,
        exclude_tools=exclude_tools,
        model_override=model_override,
        bash_prompt_approval=base.bash_prompt_approval,
        log_level=base.log_level,
        execution_profile=base.execution_profile,
        memory_top_k=base.memory_top_k,
    )


def build_runtime_overrides_from_session(session: RuntimeSession) -> RuntimeOverrides:
    model_override: Optional[Dict[str, Any]] = None
    raw_model = getattr(session.agent.state, "model", None)
    if isinstance(raw_model, dict) and raw_model:
        model_override = dict(raw_model)
    return RuntimeOverrides(
        model_override=model_override,
        bash_prompt_approval=session.bash_prompt_approval,
        execution_profile=session.execution_profile,
        memory_top_k=session.top_k,
    )


def create_runtime_session(
    *,
    log_level: Optional[str] = None,
    enable_event_logger: bool = True,
    debug_log: Optional[Callable[[str], None]] = None,
    bash_approval_fn: Optional[Callable[[dict], Any]] = None,
    run_scheduled_tasks_now_fn: Optional[Callable[[], Any]] = None,
    ensure_scheduler_daemon_fn: Optional[Callable[[], Any]] = None,
    overrides: Optional[RuntimeOverrides] = None,
) -> RuntimeSession:
    if overrides is None:
        overrides = RuntimeOverrides()

    agent = Agent({"stream_fn": stream})
    agent.set_model({"provider": "gemini", "id": "gemini-3-flash-preview", "api": None})
    if isinstance(overrides.model_override, dict) and overrides.model_override: # for eval overrides
        print(f"[config] applying model override: {overrides.model_override}")
        agent.set_model(dict(overrides.model_override))
    agent.set_timeout(120)

    initial_session_id = _generate_session_id()

    ## Memory
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

    ## Tools
    memory_search_tool = MemorySearchTool(retriever=retriever)
    task_scheduler_tool = TaskSchedulerTool(
        agent=agent,
        run_due_tasks_now_fn=run_scheduled_tasks_now_fn,
        ensure_scheduler_daemon_fn=ensure_scheduler_daemon_fn,
    )
    workspace_root = os.path.realpath(os.getcwd())
    bash_allowed_roots = parse_path_list_env(
        "POP_AGENT_BASH_ALLOWED_ROOTS",
        default_paths=[workspace_root],
        base_dir=workspace_root,
    )
    tool_allowed_roots = parse_path_list_env(
        "POP_AGENT_TOOL_ALLOWED_ROOTS",
        default_paths=bash_allowed_roots,
        base_dir=workspace_root,
    )
    file_read_tool = FileReadTool(workspace_root=workspace_root, allowed_roots=tool_allowed_roots)
    file_write_tool = FileWriteTool(workspace_root=workspace_root, allowed_roots=tool_allowed_roots)
    download_url_to_file_tool = DownloadUrlToFileTool(
        workspace_root=workspace_root,
        allowed_roots=tool_allowed_roots,
    )
    gmail_fetch_tool = GmailFetchTool(workspace_root=workspace_root)
    pdf_merge_tool = PdfMergeTool(workspace_root=workspace_root)
    agentmail_send_tool = AgentMailSendTool(
        workspace_root=workspace_root,
        allowed_roots=tool_allowed_roots,
    )

    ## Bash exec configuration
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
    if overrides.execution_profile is not None:
        execution_profile = resolve_execution_profile(overrides.execution_profile)
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
            execution_profile=execution_profile,
            workspace_root=workspace_root,
        )
    )
    tools = build_runtime_tools(
        memory_search_tool=memory_search_tool,
        task_scheduler_tool=task_scheduler_tool,
        bash_exec_tool=bash_exec_tool,
        download_url_to_file_tool=download_url_to_file_tool,
        file_read_tool=file_read_tool,
        file_write_tool=file_write_tool,
        gmail_fetch_tool=gmail_fetch_tool,
        pdf_merge_tool=pdf_merge_tool,
        agentmail_send_tool=agentmail_send_tool,
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

    debug_file_sink: Optional[Callable[[str], None]] = None
    close_debug_log_file: Optional[Callable[[], None]] = None
    debug_log_path_env = os.getenv("POP_AGENT_DEBUG_LOG_PATH", "")
    debug_log_path = None
    debug_log_path_error = None
    if str(debug_log_path_env).strip():
        debug_file_sink, close_debug_log_file, debug_log_path, debug_log_path_error = _make_debug_file_sink(
            debug_log_path_env
        )
        if debug_log_path_error:
            print(f"[debug] POP_AGENT_DEBUG_LOG_PATH warning: {debug_log_path_error}")

    effective_debug_log = _combine_debug_logs(debug_log, debug_file_sink)

    ## Event subscriptions
    if enable_event_logger:
        unsubscribe_log = agent.subscribe(make_event_logger(effective_log_level))
    else:
        unsubscribe_log = lambda: None

    if debug_file_sink is not None:
        unsubscribe_debug_file_log = agent.subscribe(make_event_logger("debug", sink=debug_file_sink))
    else:
        unsubscribe_debug_file_log = lambda: None

    if memory_subscriber is not None:
        unsubscribe_memory = agent.subscribe(memory_subscriber.on_event)
    else:
        unsubscribe_memory = lambda: None

    unsubscribe_approval = lambda: None

    top_k_raw: Any = overrides.memory_top_k if overrides.memory_top_k is not None else os.getenv("POP_AGENT_MEMORY_TOP_K", "3")
    try:
        top_k = max(1, int(top_k_raw or "3"))
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
        bash_prompt_approval=bash_prompt_approval,
        execution_profile=execution_profile,
        include_demo_tools=include_demo_tools,
        unsubscribe_log=unsubscribe_log,
        unsubscribe_memory=unsubscribe_memory,
        unsubscribe_approval=unsubscribe_approval,
        debug_log=effective_debug_log,
        unsubscribe_debug_file_log=unsubscribe_debug_file_log,
        close_debug_log_file=close_debug_log_file,
        auto_session_id=initial_session_id,
        auto_title_enabled=_resolve_bool_override(overrides.enable_auto_title, True),
        auto_title_task=None,
    )
    if effective_debug_log is not None:
        if debug_log_path is not None:
            effective_debug_log(f"[debug:file] POP_AGENT_DEBUG_LOG_PATH={debug_log_path}")
        effective_debug_log(f"[session] created id={initial_session_id}")
    return session


async def execute_scheduled_task(
    task: Dict[str, Any],
    *,
    enable_event_logger: bool = False,
    debug_log: Optional[Callable[[str], None]] = None,
    bash_approval_fn: Optional[Callable[[dict], Any]] = None,
    overrides: Optional[RuntimeOverrides] = None,
    on_warning: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    task_id = str(task.get("id") or "")
    prompt = _build_scheduled_task_prompt(task)
    effective_overrides = _build_scheduled_task_overrides(overrides)
    session: Optional[RuntimeSession] = None
    try:
        session = create_runtime_session(
            enable_event_logger=enable_event_logger,
            debug_log=debug_log,
            bash_approval_fn=bash_approval_fn,
            overrides=effective_overrides,
        )
        if task_id:
            switch_session(session, f"scheduled:{task_id}")
        reply = await run_user_turn(session, prompt, on_warning=on_warning)
        runtime_error = _scheduled_task_runtime_error(getattr(session, "agent", None))
        if runtime_error:
            result: Dict[str, Any] = {
                "status": "error",
                "summary": runtime_error,
                "error": runtime_error,
            }
            if reply and reply != runtime_error:
                result["reply"] = reply
            return result
        failed_tool = _scheduled_task_failed_tool_result(getattr(session, "agent", None))
        if failed_tool is not None:
            result = {
                "status": "error",
                "summary": failed_tool["summary"],
                "error": failed_tool["summary"],
            }
            if failed_tool.get("tool_name"):
                result["tool_name"] = failed_tool["tool_name"]
            if reply and reply != failed_tool["summary"]:
                result["reply"] = reply
            return result
        return {
            "status": "success",
            "summary": reply,
        }
    except Exception as exc:
        return {
            "status": "error",
            "summary": str(exc),
            "error": str(exc),
        }
    finally:
        if session is not None:
            try:
                await shutdown_runtime_session(session)
            except Exception as exc:
                if on_warning is not None:
                    on_warning(f"[scheduled_runner] shutdown warning: {exc}")


async def run_due_scheduled_tasks(
    agent: Agent,
    *,
    max_parallel: int = 3,
    enable_event_logger: bool = False,
    debug_log: Optional[Callable[[str], None]] = None,
    bash_approval_fn: Optional[Callable[[dict], Any]] = None,
    overrides: Optional[RuntimeOverrides] = None,
    on_warning: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    async def _executor(task: Dict[str, Any]) -> Dict[str, Any]:
        return await execute_scheduled_task(
            task,
            enable_event_logger=enable_event_logger,
            debug_log=debug_log,
            bash_approval_fn=bash_approval_fn,
            overrides=overrides,
            on_warning=on_warning,
        )

    return await agent.run_due_tasks(_executor, max_parallel=max_parallel)


async def run_user_turn(
    session: RuntimeSession,
    user_message: str,
    on_warning: Optional[Callable[[str], None]] = None,
) -> str:
    await session.ingestion_worker.flush()

    memory_text = "(no relevant memories)"
    try:
        retrieve_sections_with_fallback = getattr(session.retriever, "retrieve_sections_with_fallback", None)
        if callable(retrieve_sections_with_fallback):
            short_hits, long_hits = retrieve_sections_with_fallback(
                user_message,
                top_k=session.top_k,
                scope="both",
                session_id=session.active_session_id,
            )
        else:
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
        session.unsubscribe_debug_file_log()
        close_debug_log_file = session.close_debug_log_file
        if close_debug_log_file is not None:
            close_debug_log_file()
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
