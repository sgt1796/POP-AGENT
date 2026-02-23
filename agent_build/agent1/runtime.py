import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from POP.embedder import Embedder
from POP.stream import stream

from agent import Agent
from agent.agent_types import AgentTool
from agent.tools import (
    BashExecConfig,
    BashExecTool,
    FastTool,
    GmailFetchTool,
    MemorySearchTool,
    PdfMergeTool,
    SlowTool,
    ToolsmakerTool,
    WebSnapshotTool,
)

from .approvals import (
    BashExecApprovalPrompter,
    ToolsmakerApprovalSubscriber,
    ToolsmakerAutoContinueSubscriber,
)
from .constants import BASH_GIT_READ_SUBCOMMANDS, BASH_READ_COMMANDS, BASH_WRITE_COMMANDS
from .env_utils import (
    parse_bool_env,
    parse_float_env,
    parse_int_env,
    parse_path_list_env,
    parse_toolsmaker_allowed_capabilities,
    sorted_csv,
)
from .event_logger import make_event_logger
from .memory import (
    ConversationMemory,
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


class ManualToolsmakerSubscriberFactory(Protocol):
    def __call__(self, agent: Agent) -> Any:
        ...


@dataclass
class RuntimeSession:
    agent: Agent
    retriever: MemoryRetriever
    ingestion_worker: EmbeddingIngestionWorker
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
        WebSnapshotTool(),
        memory_search_tool,
        toolsmaker_tool,
        bash_exec_tool,
        gmail_fetch_tool,
        pdf_merge_tool,
    ]
    if include_demo_tools:
        tools.extend([SlowTool(), FastTool()])
    return tools


def create_runtime_session(
    *,
    log_level: Optional[str] = None,
    enable_event_logger: bool = True,
    bash_approval_fn: Optional[Callable[[dict], Any]] = None,
    manual_toolsmaker_subscriber_factory: Optional[ManualToolsmakerSubscriberFactory] = None,
) -> RuntimeSession:
    agent = Agent({"stream_fn": stream})
    agent.set_model({"provider": "gemini", "id": "gemini-3-flash-preview", "api": None})
    agent.set_timeout(120)

    embedder = Embedder(use_api="openai")
    short_memory = ConversationMemory(embedder=embedder, max_entries=100)
    long_memory = DiskMemory(filepath=os.path.join("agent", "mem", "chat"), embedder=embedder, max_entries=1000)
    retriever = MemoryRetriever(short_term=short_memory, long_term=long_memory)

    ingestion_worker = EmbeddingIngestionWorker(memory=short_memory, long_term=long_memory)
    ingestion_worker.start()
    memory_subscriber = MemorySubscriber(ingestion_worker=ingestion_worker)

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
    agent.set_tools(
        build_runtime_tools(
            memory_search_tool=memory_search_tool,
            toolsmaker_tool=toolsmaker_tool,
            bash_exec_tool=bash_exec_tool,
            gmail_fetch_tool=gmail_fetch_tool,
            pdf_merge_tool=pdf_merge_tool,
            include_demo_tools=include_demo_tools,
        )
    )

    if log_level is None:
        log_level = os.getenv("POP_AGENT_LOG_LEVEL", "quiet")
    if enable_event_logger:
        unsubscribe_log = agent.subscribe(make_event_logger(log_level))
    else:
        unsubscribe_log = lambda: None

    unsubscribe_memory = agent.subscribe(memory_subscriber.on_event)

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

    return RuntimeSession(
        agent=agent,
        retriever=retriever,
        ingestion_worker=ingestion_worker,
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
    )


async def run_user_turn(
    session: RuntimeSession,
    user_message: str,
    on_warning: Optional[Callable[[str], None]] = None,
) -> str:
    await session.ingestion_worker.flush()

    memory_text = "(no relevant memories)"
    try:
        short_hits, long_hits = session.retriever.retrieve_sections(user_message, top_k=session.top_k, scope="both")
        memory_text = format_memory_sections(short_hits, long_hits)
    except Exception as exc:
        memory_text = "(no relevant memories)"
        if on_warning is not None:
            on_warning(f"[memory] retrieval warning: {exc}")

    augmented_prompt = build_augmented_prompt(user_message, memory_text)
    await session.agent.prompt(augmented_prompt)

    reply = extract_latest_assistant_text(session.agent)
    if not reply:
        reply = "(no assistant text returned)"
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
        session.unsubscribe_memory()
        session.unsubscribe_log()
        session.unsubscribe_approval()

    if shutdown_error is not None:
        raise shutdown_error


async def main() -> None:
    session = create_runtime_session()

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
    print("Type 'exit' or 'quit' to stop.\n")

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
