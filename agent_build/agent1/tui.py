import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .approvals import DEFAULT_TOOL_REJECT_REASON
from .constants import BASH_GIT_READ_SUBCOMMANDS, BASH_READ_COMMANDS, BASH_WRITE_COMMANDS, LOG_LEVELS
from .env_utils import sorted_csv
from .message_utils import extract_bash_exec_command, extract_texts
from .prompting import VALID_EXECUTION_PROFILES, build_system_prompt
from .runtime import create_runtime_session, run_user_turn, shutdown_runtime_session
from .tui_runtime import (
    AsyncDecisionQueue,
    AsyncToolsmakerApprovalSubscriber,
    ToolsmakerDecision,
    format_activity_event,
)
from .usage_reporting import format_cumulative_usage_fragment, format_turn_usage_line, usage_delta


def _textual_import_error_hint() -> None:
    print("Textual is not installed in this environment.")
    print("Install it with: pip install textual")


def _looks_like_markdown_text(text: str) -> bool:
    value = str(text or "")
    if not value.strip():
        return False

    if "```" in value or "~~~" in value:
        return True

    if re.search(r"(?m)^\s{0,3}#{1,6}\s+\S", value):
        return True
    if re.search(r"(?m)^\s{0,3}>\s+\S", value):
        return True
    if re.search(r"(?m)^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$", value):
        return True

    has_table_row = re.search(r"(?m)^\s*\|.+\|\s*$", value) is not None
    has_table_sep = (
        re.search(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", value) is not None
    )
    if has_table_row and has_table_sep:
        return True

    list_matches = re.findall(r"(?m)^\s{0,3}(?:[-*+]|\d+[.)])\s+\S", value)
    if len(list_matches) >= 2:
        return True

    if "\n" in value and re.search(r"`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\[[^\]]+\]\([^)]+\)", value):
        return True

    return False


@dataclass
class _QueuedApproval:
    kind: str
    payload: dict
    future: asyncio.Future[Any]


@dataclass(frozen=True)
class _SettingsPayload:
    model_provider: str
    model_id: str
    model_api: str
    timeout_s: str
    execution_profile: str
    memory_top_k: str
    activity_level: str


@dataclass(frozen=True)
class _ValidatedSettings:
    model_provider: str
    model_id: Optional[str]
    model_api: Optional[str]
    timeout_s: float
    execution_profile: str
    memory_top_k: int
    activity_level: str


@dataclass(frozen=True)
class _PendingToolCallRecord:
    tool_name: str
    args_preview: str
    command: str


def run_tui() -> int:
    try:
        from rich.markdown import Markdown
        from rich.text import Text
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical, VerticalScroll
        from textual.screen import ModalScreen
        from textual.widgets import Button, Checkbox, Footer, Header, Input, RichLog, Static
        try:
            from textual.widgets import Select
        except (ImportError, ModuleNotFoundError):
            Select = None  # type: ignore[assignment]
    except (ImportError, ModuleNotFoundError):
        _textual_import_error_hint()
        return 1

    class BashApprovalScreen(ModalScreen[bool]):
        BINDINGS = [Binding("escape", "reject", "Reject")]

        def __init__(self, request: dict) -> None:
            super().__init__()
            self._request = request

        def compose(self) -> ComposeResult:
            command = str(self._request.get("command", "")).strip()
            cwd = str(self._request.get("cwd", "")).strip() or "(none)"
            risk = str(self._request.get("risk", "")).strip() or "unknown"
            justification = str(self._request.get("justification", "")).strip() or "(none)"

            with Vertical(id="approval_modal"):
                yield Static("bash_exec Approval", classes="modal_title")
                yield Static(f"Risk: {risk}")
                yield Static(f"CWD: {cwd}")
                yield Static(f"Command: {command}")
                yield Static(f"Justification: {justification}")
                with Horizontal(classes="modal_actions"):
                    yield Button("Approve", id="approve", variant="success")
                    yield Button("Reject", id="reject", variant="error")

        def action_reject(self) -> None:
            self.dismiss(False)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(event.button.id == "approve")

    class ToolsmakerApprovalScreen(ModalScreen[ToolsmakerDecision]):
        BINDINGS = [Binding("escape", "reject", "Reject")]

        def __init__(self, details: dict, *, default_activate: bool) -> None:
            super().__init__()
            self._details = details
            self._default_activate = default_activate

        def compose(self) -> ComposeResult:
            name = str(self._details.get("name", "")).strip() or "(unknown)"
            version = int(self._details.get("version", 0) or 0)
            review_path = str(self._details.get("review_path", "")).strip() or "(none)"
            caps = list(self._details.get("requested_capabilities") or [])
            caps_text = ", ".join([str(c) for c in caps]) if caps else "(none)"

            with Vertical(id="approval_modal"):
                yield Static("toolsmaker Approval", classes="modal_title")
                yield Static(f"Tool: {name}")
                yield Static(f"Version: {version}")
                yield Static(f"Review Path: {review_path}")
                yield Static(f"Requested Capabilities: {caps_text}")
                yield Checkbox("Activate after approve", id="activate_checkbox", value=self._default_activate)
                yield Input(value=DEFAULT_TOOL_REJECT_REASON, id="reason_input", placeholder="Reject reason")
                with Horizontal(classes="modal_actions"):
                    yield Button("Approve", id="approve", variant="success")
                    yield Button("Reject", id="reject", variant="error")

        def action_reject(self) -> None:
            reason_input = self.query_one("#reason_input", Input)
            reason = reason_input.value.strip() or DEFAULT_TOOL_REJECT_REASON
            self.dismiss(ToolsmakerDecision(approve=False, activate=False, reason=reason))

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "approve":
                activate = self.query_one("#activate_checkbox", Checkbox).value
                self.dismiss(ToolsmakerDecision(approve=True, activate=bool(activate), reason=""))
                return

            reason_input = self.query_one("#reason_input", Input)
            reason = reason_input.value.strip() or DEFAULT_TOOL_REJECT_REASON
            self.dismiss(ToolsmakerDecision(approve=False, activate=False, reason=reason))

    class SettingsScreen(ModalScreen[Optional[_SettingsPayload]]):
        BINDINGS = [Binding("escape", "cancel", "Cancel")]

        def __init__(self, current: _SettingsPayload) -> None:
            super().__init__()
            self._current = current

        def compose(self) -> ComposeResult:
            with VerticalScroll(id="settings_modal"):
                yield Static("Runtime Settings", classes="modal_title")
                yield Static("Model provider", classes="settings_label")
                yield Input(value=self._current.model_provider, id="settings_model_provider", placeholder="gemini")
                yield Static("Model id (blank for provider default)", classes="settings_label")
                yield Input(value=self._current.model_id, id="settings_model_id", placeholder="gemini-3-flash-preview")
                yield Static("Model api (optional)", classes="settings_label")
                yield Input(value=self._current.model_api, id="settings_model_api", placeholder="(optional)")
                yield Static("Request timeout seconds (>0)", classes="settings_label")
                yield Input(value=self._current.timeout_s, id="settings_timeout_s", placeholder="120")
                yield Static("Execution profile: balanced | aggressive | conservative", classes="settings_label")
                yield Input(value=self._current.execution_profile, id="settings_execution_profile", placeholder="balanced")
                yield Static("Memory retrieval top-k (>=1)", classes="settings_label")
                yield Input(value=self._current.memory_top_k, id="settings_memory_top_k", placeholder="3")
                yield Static("Activity level: quiet | messages | stream | debug", classes="settings_label")
                if Select is None:
                    yield Input(value=self._current.activity_level, id="settings_activity_level", placeholder="stream")
                else:
                    yield Select(
                        options=[
                            ("quiet", "quiet"),
                            ("messages", "messages"),
                            ("stream", "stream"),
                            ("debug", "debug"),
                        ],
                        value=self._current.activity_level,
                        id="settings_activity_level",
                    )
                with Horizontal(classes="modal_actions"):
                    yield Button("Apply", id="apply", variant="success")
                    yield Button("Cancel", id="cancel", variant="primary")

        def action_cancel(self) -> None:
            self.dismiss(None)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id != "apply":
                self.dismiss(None)
                return

            activity_widget = self.query_one("#settings_activity_level")
            activity_raw = getattr(activity_widget, "value", "")
            activity_level = "" if activity_raw is None else str(activity_raw).strip().lower()
            if not activity_level or activity_level.endswith("blank"):
                activity_level = self._current.activity_level

            payload = _SettingsPayload(
                model_provider=self.query_one("#settings_model_provider", Input).value,
                model_id=self.query_one("#settings_model_id", Input).value,
                model_api=self.query_one("#settings_model_api", Input).value,
                timeout_s=self.query_one("#settings_timeout_s", Input).value,
                execution_profile=self.query_one("#settings_execution_profile", Input).value,
                memory_top_k=self.query_one("#settings_memory_top_k", Input).value,
                activity_level=activity_level,
            )
            self.dismiss(payload)

    class AgentTuiApp(App[None]):
        CSS = """
        Screen {
            layout: vertical;
        }

        #status_line {
            height: 1;
            padding: 0 1;
            background: $panel;
            color: $text;
        }

        #body {
            height: 1fr;
            layout: horizontal;
        }

        #left_panel {
            width: 2fr;
            layout: vertical;
            border: solid $accent;
            padding: 0 1;
        }

        #right_panel {
            width: 1fr;
            layout: vertical;
            border: solid $primary;
            padding: 0 1;
        }

        #transcript {
            height: 1fr;
        }

        #stream_preview {
            min-height: 3;
            border: solid $boost;
            padding: 0 1;
        }

        #activity {
            height: 1fr;
        }

        #chat_input {
            dock: bottom;
        }

        #approval_modal {
            width: 80;
            height: auto;
            border: round $warning;
            background: $surface;
            padding: 1 2;
            layout: vertical;
        }

        #settings_modal {
            width: 92;
            height: 90%;
            max-height: 90%;
            border: round $accent;
            background: $surface;
            padding: 1 2;
            layout: vertical;
            overflow-y: auto;
        }

        .modal_title {
            text-style: bold;
            padding-bottom: 1;
        }

        .settings_label {
            padding-top: 1;
        }

        .modal_actions {
            padding-top: 1;
            height: auto;
            align: right middle;
        }
        """

        BINDINGS = [
            Binding("q", "quit_app", "Quit"),
            Binding("ctrl+s", "open_settings", "Settings"),
            Binding("ctrl+c", "abort_run", "Abort"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._session = None
            self._turn_task: Optional[asyncio.Task[None]] = None
            self._approval_task: Optional[asyncio.Task[None]] = None
            self._approval_queue: AsyncDecisionQueue[_QueuedApproval] = AsyncDecisionQueue()
            self._unsubscribe_ui_events = lambda: None
            self._cleanup_done = False
            self._stream_buffer = ""
            self._activity_log_level = self._normalize_activity_level(os.getenv("POP_AGENT_LOG_LEVEL", "stream"))
            self._pending_tool_calls: Dict[str, _PendingToolCallRecord] = {}
            self._turn_usage_before: Dict[str, Any] = {}

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            yield Static("Initializing...", id="status_line")
            with Horizontal(id="body"):
                with Vertical(id="left_panel"):
                    yield Static("Transcript")
                    yield RichLog(id="transcript", wrap=True, markup=False, highlight=False)
                    yield Static("Streaming Preview")
                    yield Static("", id="stream_preview")
                with Vertical(id="right_panel"):
                    yield Static("Activity")
                    yield RichLog(id="activity", wrap=True, markup=False, highlight=False)
            yield Input(id="chat_input", placeholder="Type a message and press Enter...")
            yield Footer()

        async def on_mount(self) -> None:
            self._status = self.query_one("#status_line", Static)
            self._transcript = self.query_one("#transcript", RichLog)
            self._activity = self.query_one("#activity", RichLog)
            self._stream_preview = self.query_one("#stream_preview", Static)
            self._chat_input = self.query_one("#chat_input", Input)

            self._set_status("Starting runtime session...")
            try:
                self._session = create_runtime_session(
                    enable_event_logger=False,
                    bash_approval_fn=self._request_bash_approval,
                    manual_toolsmaker_subscriber_factory=self._make_toolsmaker_subscriber,
                )
            except Exception as exc:
                self._append_activity(f"[runtime:error] {exc}")
                self._set_status("Runtime startup failed")
                self._chat_input.disabled = True
                return

            self._unsubscribe_ui_events = self._session.agent.subscribe(self._on_agent_event)
            self._approval_task = asyncio.create_task(self._approval_worker())
            self._append_activity(f"[settings] Ctrl+S to edit runtime settings (activity={self._activity_log_level})")
            self._refresh_status("Ready")
            self._chat_input.focus()

        @staticmethod
        def _normalize_activity_level(value: str) -> str:
            key = str(value or "").strip().lower()
            if key in LOG_LEVELS:
                return key
            return "stream"

        def _toolsmaker_mode_text(self) -> str:
            if self._session is None:
                return "unknown"
            if self._session.toolsmaker_manual_approval:
                return "manual"
            if self._session.toolsmaker_auto_continue:
                return "auto-continue"
            return "llm-managed"

        def _bash_mode_text(self) -> str:
            if self._session is None:
                return "unknown"
            return "prompt" if self._session.bash_prompt_approval else "policy-deny"

        def _model_summary(self) -> str:
            if self._session is None:
                return "unknown"
            model = getattr(self._session.agent.state, "model", {}) or {}
            provider = str(model.get("provider", "")).strip() or "unknown"
            model_id_raw = model.get("id")
            model_id = str(model_id_raw).strip() if model_id_raw is not None else ""
            if not model_id:
                return provider
            return f"{provider}:{model_id}"

        def _refresh_status(self, prefix: str = "Ready") -> None:
            if self._session is None:
                self._set_status(prefix)
                return
            usage_fragment = format_cumulative_usage_fragment(self._current_usage_summary())
            self._set_status(
                f"{prefix} | model={self._model_summary()} | profile={self._session.execution_profile} "
                f"| activity={self._activity_log_level} | toolsmaker={self._toolsmaker_mode_text()} "
                f"| bash={self._bash_mode_text()} | {usage_fragment}"
            )

        def _set_status(self, text: str) -> None:
            self._status.update(text)

        def _current_usage_summary(self) -> Dict[str, Any]:
            if self._session is None:
                return {}
            get_summary = getattr(self._session.agent, "get_usage_summary", None)
            if not callable(get_summary):
                return {}
            summary = get_summary()
            if not isinstance(summary, dict):
                return {}
            return summary

        def _append_transcript(self, role: str, text: str) -> None:
            if role == "Assistant" and _looks_like_markdown_text(text):
                self._transcript.write(Text(f"{role}:", style="bold"))
                self._transcript.write(Markdown(text))
                return
            self._transcript.write(f"{role}: {text}")

        def _append_transcript_note(self, text: Any) -> None:
            self._transcript.write(text)

        def _append_activity(self, text: Any) -> None:
            self._activity.write(text)

        def _update_stream_preview(self, text: str) -> None:
            if text and _looks_like_markdown_text(text):
                self._stream_preview.update(Markdown(text))
                return
            self._stream_preview.update(text)

        def _allow_activity_line(self, text: str) -> bool:
            if self._activity_log_level == "quiet":
                return False
            if self._activity_log_level == "messages":
                return not text.startswith("[stream]")
            return True

        def _is_turn_active(self) -> bool:
            return self._turn_task is not None and not self._turn_task.done()

        def _dismiss_open_approval_modal(self) -> bool:
            active_screen = self.screen
            if isinstance(active_screen, BashApprovalScreen):
                active_screen.dismiss(False)
                return True
            if isinstance(active_screen, ToolsmakerApprovalScreen):
                active_screen.dismiss(
                    ToolsmakerDecision(
                        approve=False,
                        activate=False,
                        reason="aborted_by_user",
                    )
                )
                return True
            return False

        def _reject_queued_approvals(self) -> int:
            rejected = 0
            for queued in self._approval_queue.drain():
                if queued.future.done():
                    continue
                if queued.kind == "toolsmaker":
                    queued.future.set_result(
                        ToolsmakerDecision(
                            approve=False,
                            activate=False,
                            reason="aborted_by_user",
                        )
                    )
                else:
                    queued.future.set_result(False)
                rejected += 1
            return rejected

        async def _push_screen_result(self, screen: Any) -> Any:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[Any] = loop.create_future()

            def _on_dismiss(result: Any) -> None:
                if not future.done():
                    future.set_result(result)

            self.push_screen(screen, callback=_on_dismiss)
            return await future

        def _rebuild_system_prompt(self) -> None:
            if self._session is None:
                return
            workspace_root = os.path.realpath(os.getcwd())
            self._session.agent.set_system_prompt(
                build_system_prompt(
                    bash_read_csv=sorted_csv(BASH_READ_COMMANDS),
                    bash_write_csv=sorted_csv(BASH_WRITE_COMMANDS),
                    bash_git_csv=sorted_csv(BASH_GIT_READ_SUBCOMMANDS),
                    bash_prompt_approval=self._session.bash_prompt_approval,
                    toolsmaker_manual_approval=self._session.toolsmaker_manual_approval,
                    toolsmaker_auto_continue=self._session.toolsmaker_auto_continue,
                    execution_profile=self._session.execution_profile,
                    workspace_root=workspace_root,
                )
            )

        def _read_settings(self) -> _SettingsPayload:
            model = {}
            timeout_s = 120.0
            execution_profile = "balanced"
            memory_top_k = 3
            if self._session is not None:
                model = getattr(self._session.agent.state, "model", {}) or {}
                timeout_value = self._session.agent.request_timeout_s
                timeout_s = float(timeout_value) if timeout_value is not None else 120.0
                execution_profile = self._session.execution_profile
                memory_top_k = self._session.top_k

            model_provider = str(model.get("provider", "")).strip()
            model_id_raw = model.get("id")
            model_api_raw = model.get("api")
            return _SettingsPayload(
                model_provider=model_provider,
                model_id="" if model_id_raw is None else str(model_id_raw),
                model_api="" if model_api_raw is None else str(model_api_raw),
                timeout_s=f"{timeout_s:g}",
                execution_profile=execution_profile,
                memory_top_k=str(memory_top_k),
                activity_level=self._activity_log_level,
            )

        def _validate_settings(self, payload: _SettingsPayload) -> Tuple[Optional[_ValidatedSettings], Optional[str]]:
            model_provider = payload.model_provider.strip().lower()
            if not model_provider:
                return None, "model provider is required"

            model_id_raw = payload.model_id.strip()
            model_api_raw = payload.model_api.strip()
            model_id = model_id_raw if model_id_raw else None
            model_api = model_api_raw if model_api_raw else None

            try:
                timeout_s = float(payload.timeout_s.strip())
            except Exception:
                return None, "request timeout must be a number"
            if timeout_s <= 0:
                return None, "request timeout must be > 0"

            execution_profile = payload.execution_profile.strip().lower()
            if execution_profile not in VALID_EXECUTION_PROFILES:
                allowed_profiles = ", ".join(sorted(VALID_EXECUTION_PROFILES))
                return None, f"execution profile must be one of: {allowed_profiles}"

            try:
                memory_top_k = int(payload.memory_top_k.strip())
            except Exception:
                return None, "memory top-k must be an integer"
            if memory_top_k < 1:
                return None, "memory top-k must be >= 1"

            activity_level = payload.activity_level.strip().lower()
            if activity_level not in LOG_LEVELS:
                allowed_levels = ", ".join(sorted(LOG_LEVELS.keys()))
                return None, f"activity level must be one of: {allowed_levels}"

            return (
                _ValidatedSettings(
                    model_provider=model_provider,
                    model_id=model_id,
                    model_api=model_api,
                    timeout_s=timeout_s,
                    execution_profile=execution_profile,
                    memory_top_k=memory_top_k,
                    activity_level=activity_level,
                ),
                None,
            )

        def _apply_settings(self, payload: _SettingsPayload) -> None:
            if self._session is None:
                self._append_activity("[settings:error] runtime session unavailable")
                return
            validated, error = self._validate_settings(payload)
            if validated is None:
                self._append_activity(f"[settings:error] {error}")
                return

            try:
                self._session.agent.set_model(
                    {
                        "provider": validated.model_provider,
                        "id": validated.model_id,
                        "api": validated.model_api,
                    }
                )
                self._session.agent.set_timeout(validated.timeout_s)
                self._session.top_k = validated.memory_top_k
                self._session.execution_profile = validated.execution_profile
                self._activity_log_level = validated.activity_level
                self._rebuild_system_prompt()
            except Exception as exc:
                self._append_activity(f"[settings:error] apply failed: {exc}")
                return

            model_id_text = validated.model_id if validated.model_id is not None else "(default)"
            self._append_activity(
                "[settings] applied "
                f"model={validated.model_provider}:{model_id_text} "
                f"timeout_s={validated.timeout_s:g} "
                f"profile={validated.execution_profile} "
                f"top_k={validated.memory_top_k} "
                f"activity={validated.activity_level}"
            )
            self._refresh_status("Ready")

        def _extract_message_text(self, message: Any) -> str:
            text = "\n".join([t for t in extract_texts(message) if str(t).strip()]).strip()
            return text

        @staticmethod
        def _message_has_tool_calls(message: Any) -> bool:
            content = getattr(message, "content", None) or []
            for item in content:
                item_type = getattr(item, "type", None)
                if item_type is None and isinstance(item, dict):
                    item_type = item.get("type")
                if str(item_type) == "toolCall":
                    return True
            return False

        @staticmethod
        def _tool_call_id(event: dict) -> str:
            call_id = event.get("toolCallId")
            if call_id is None:
                return ""
            return str(call_id).strip()

        @staticmethod
        def _format_args_preview(args: Any, *, max_len: int = 160) -> str:
            if args in (None, ""):
                return ""
            text = str(args).strip()
            if not text:
                return ""
            if len(text) > max_len:
                return f"{text[: max_len - 3]}..."
            return text

        @staticmethod
        def _tool_signature(tool_name: str, args_preview: str) -> str:
            if args_preview:
                return f"{tool_name}({args_preview})"
            return f"{tool_name}()"

        def _build_tool_start_line(self, record: _PendingToolCallRecord) -> str:
            if record.tool_name == "bash_exec":
                if record.command:
                    return f"[...] Running bash: {record.command}"
                return "[...] Running bash command"
            return f"[...] Calling {self._tool_signature(record.tool_name, record.args_preview)}"

        def _build_tool_end_line(self, record: _PendingToolCallRecord, *, is_error: bool) -> str:
            if record.tool_name == "bash_exec":
                command_text = record.command or "bash command"
                if is_error:
                    return f"Ran {command_text} (failed)"
                return f"Ran {command_text}"

            signature = self._tool_signature(record.tool_name, record.args_preview)
            if is_error:
                return f"{signature} (failed)"
            return signature

        def _append_tool_line(self, line: str, *, pending: bool, is_error: bool = False) -> None:
            if pending:
                style = "bold yellow"
            elif is_error:
                style = "dim red"
            else:
                style = "dim"

            self._append_transcript_note(Text(line, style=style))
            if self._allow_activity_line("[tool]"):
                self._append_activity(Text(line, style=style))

        def _handle_tool_activity_event(self, event: dict) -> None:
            etype = str(event.get("type", "")).strip()
            call_id = self._tool_call_id(event)

            if etype == "tool_execution_start":
                key = call_id or f"tool_{len(self._pending_tool_calls) + 1}"
                tool_name = str(event.get("toolName", "")).strip() or "unknown"
                args_preview = self._format_args_preview(event.get("args"))
                command = extract_bash_exec_command(event) if tool_name == "bash_exec" else ""
                record = _PendingToolCallRecord(
                    tool_name=tool_name,
                    args_preview=args_preview,
                    command=command,
                )
                self._pending_tool_calls[key] = record
                self._append_tool_line(self._build_tool_start_line(record), pending=True)
                return

            if etype != "tool_execution_end":
                return

            record: Optional[_PendingToolCallRecord] = None
            if call_id:
                record = self._pending_tool_calls.pop(call_id, None)
            if record is None:
                tool_name = str(event.get("toolName", "")).strip() or "unknown"
                args_preview = self._format_args_preview(event.get("args"))
                command = extract_bash_exec_command(event) if tool_name == "bash_exec" else ""
                record = _PendingToolCallRecord(
                    tool_name=tool_name,
                    args_preview=args_preview,
                    command=command,
                )

            is_error = bool(event.get("isError"))
            self._append_tool_line(self._build_tool_end_line(record, is_error=is_error), pending=False, is_error=is_error)

        def _flush_pending_tool_lines(self) -> None:
            if not self._pending_tool_calls:
                return

            for record in list(self._pending_tool_calls.values()):
                interrupted_line = f"{self._build_tool_end_line(record, is_error=True)} (interrupted)"
                self._append_tool_line(interrupted_line, pending=False, is_error=True)
            self._pending_tool_calls.clear()

        def _on_agent_event(self, event: dict) -> None:
            if self._activity_log_level == "debug":
                self._append_activity(f"[debug:event] {event}")
            etype = str(event.get("type", "")).strip()
            if etype in {"tool_execution_start", "tool_execution_end"}:
                self._handle_tool_activity_event(event)
            else:
                activity_text = format_activity_event(event)
                if activity_text and self._allow_activity_line(activity_text):
                    self._append_activity(activity_text)

            if etype == "message_start":
                message = event.get("message")
                if getattr(message, "role", None) == "assistant":
                    self._stream_buffer = ""
                    self._update_stream_preview("")
                return

            if etype == "message_update":
                message = event.get("message")
                if getattr(message, "role", None) == "assistant":
                    self._stream_buffer = self._extract_message_text(message)
                    self._update_stream_preview(self._stream_buffer)
                return

            if etype != "message_end":
                return

            message = event.get("message")
            role = getattr(message, "role", "")
            if role != "assistant":
                return

            error_message = getattr(message, "error_message", None)
            if error_message:
                self._append_transcript("Assistant", f"(error) {error_message}")
            else:
                final_text = self._extract_message_text(message)
                if final_text:
                    self._append_transcript("Assistant", final_text)
                elif not self._message_has_tool_calls(message):
                    self._append_transcript("Assistant", "(no assistant text returned)")

            self._stream_buffer = ""
            self._update_stream_preview("")

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            user_text = event.value.strip()
            event.input.value = ""

            if not user_text:
                return
            if self._session is None:
                self._append_activity("[runtime] session unavailable")
                return
            if self._turn_task is not None and not self._turn_task.done():
                self._append_activity("[runtime] turn already running")
                return

            self._append_transcript("User", user_text)
            self._turn_usage_before = self._current_usage_summary()
            self._chat_input.disabled = True
            self._set_status("Running...")
            self._turn_task = asyncio.create_task(self._run_turn(user_text))

        async def _run_turn(self, user_text: str) -> None:
            if self._session is None:
                return
            try:
                await run_user_turn(self._session, user_text, on_warning=self._append_activity)
            except asyncio.CancelledError:
                self._append_activity("[abort] turn cancelled")
            except Exception as exc:
                self._append_activity(f"[turn:error] {exc}")
                self._append_transcript("Assistant", f"(error) {exc}")
            finally:
                self._flush_pending_tool_lines()
                if self._session is not None:
                    after_summary = self._current_usage_summary()
                    delta = usage_delta(self._turn_usage_before, after_summary)
                    if int(delta.get("calls", 0)) > 0:
                        get_last = getattr(self._session.agent, "get_last_usage", None)
                        last_usage_raw = get_last() if callable(get_last) else None
                        last_usage = last_usage_raw if isinstance(last_usage_raw, dict) else None
                        usage_line = format_turn_usage_line(delta, last_usage)
                        if usage_line:
                            self._append_activity(usage_line)
                self._turn_usage_before = {}
                self._chat_input.disabled = False
                self._chat_input.focus()
                self._refresh_status("Ready")

        def _on_settings_dismissed(self, result: Optional[_SettingsPayload]) -> None:
            if isinstance(result, _SettingsPayload):
                self._apply_settings(result)

        def action_open_settings(self) -> None:
            if self._session is None:
                self._append_activity("[settings] runtime unavailable")
                return
            if self._turn_task is not None and not self._turn_task.done():
                self._append_activity("[settings] wait for current run to finish")
                return
            if self._session.agent.state.is_streaming:
                self._append_activity("[settings] cannot apply while streaming")
                return

            payload = self._read_settings()
            self.push_screen(SettingsScreen(payload), callback=self._on_settings_dismissed)

        async def _request_bash_approval(self, request: dict) -> bool:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[Any] = loop.create_future()
            self._approval_queue.put(_QueuedApproval(kind="bash", payload=request, future=future))
            decision = await future
            return bool(decision)

        async def _request_toolsmaker_decision(self, details: dict) -> ToolsmakerDecision:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[Any] = loop.create_future()
            self._approval_queue.put(_QueuedApproval(kind="toolsmaker", payload=details, future=future))
            result = await future
            if isinstance(result, ToolsmakerDecision):
                return result
            return ToolsmakerDecision(approve=False, activate=False, reason=DEFAULT_TOOL_REJECT_REASON)

        def _make_toolsmaker_subscriber(self, agent: Any) -> AsyncToolsmakerApprovalSubscriber:
            return AsyncToolsmakerApprovalSubscriber(
                agent=agent,
                resolve_decision=self._request_toolsmaker_decision,
                on_activity=self._append_activity,
            )

        async def _approval_worker(self) -> None:
            while True:
                queued = await self._approval_queue.get()

                if queued.kind == "bash":
                    request = queued.payload
                    self._append_activity(
                        f"[approval] bash_exec requested risk={request.get('risk')} cmd={request.get('command')}"
                    )
                    decision: Any = False
                    try:
                        decision = await self._push_screen_result(BashApprovalScreen(request))
                    except Exception as exc:
                        self._append_activity(f"[approval:error] bash_exec modal failed: {exc}")
                        decision = False
                    finally:
                        if not queued.future.done():
                            queued.future.set_result(bool(decision))
                    self._append_activity(
                        f"[approval] bash_exec {'approved' if bool(decision) else 'rejected'}"
                    )
                    continue

                details = queued.payload
                name = str(details.get("name", "")).strip() or "(unknown)"
                version = int(details.get("version", 0) or 0)
                self._append_activity(f"[approval] toolsmaker requested tool={name} version={version}")

                decision_obj: Any = ToolsmakerDecision(
                    approve=False,
                    activate=False,
                    reason=DEFAULT_TOOL_REJECT_REASON,
                )
                try:
                    default_activate = bool(self._session and self._session.toolsmaker_auto_activate)
                    decision_obj = await self._push_screen_result(
                        ToolsmakerApprovalScreen(details, default_activate=default_activate)
                    )
                except Exception as exc:
                    self._append_activity(f"[approval:error] toolsmaker modal failed: {exc}")
                finally:
                    if not isinstance(decision_obj, ToolsmakerDecision):
                        decision_obj = ToolsmakerDecision(
                            approve=False,
                            activate=False,
                            reason=DEFAULT_TOOL_REJECT_REASON,
                        )
                    if not queued.future.done():
                        queued.future.set_result(decision_obj)

                if decision_obj.approve:
                    if decision_obj.activate:
                        self._append_activity(f"[approval] toolsmaker approved+activate tool={name} version={version}")
                    else:
                        self._append_activity(f"[approval] toolsmaker approved tool={name} version={version}")
                else:
                    self._append_activity(
                        f"[approval] toolsmaker rejected tool={name} version={version} reason={decision_obj.reason}"
                    )

        def action_abort_run(self) -> None:
            if self._session is None:
                self._append_activity("[abort] runtime unavailable")
                return

            turn_active = self._is_turn_active()
            stream_active = bool(self._session.agent.state.is_streaming)
            if not turn_active and not stream_active:
                self._append_activity("[abort] no active run")
                return

            modal_dismissed = self._dismiss_open_approval_modal()
            queued_rejected = self._reject_queued_approvals()

            try:
                self._session.agent.abort()
            except Exception as exc:
                self._append_activity(f"[abort:warning] abort signal failed: {exc}")

            if turn_active and self._turn_task is not None and not self._turn_task.done():
                self._turn_task.cancel()

            self._set_status("Aborting...")
            notes = []
            if turn_active:
                notes.append("turn task cancelled")
            if stream_active:
                notes.append("stream abort signaled")
            if modal_dismissed:
                notes.append("approval dialog dismissed")
            if queued_rejected:
                notes.append(f"queued approvals rejected={queued_rejected}")
            suffix = f" ({', '.join(notes)})" if notes else ""
            self._append_activity(f"[abort] requested{suffix}")

        async def action_quit_app(self) -> None:
            await self._cleanup_runtime()
            self.exit()

        async def on_shutdown(self) -> None:
            await self._cleanup_runtime()

        async def _cleanup_runtime(self) -> None:
            if self._cleanup_done:
                return
            self._cleanup_done = True

            if self._turn_task is not None and not self._turn_task.done():
                if self._session is not None:
                    self._session.agent.abort()
                try:
                    await asyncio.wait_for(self._turn_task, timeout=5.0)
                except BaseException:
                    pass

            if self._approval_task is not None and not self._approval_task.done():
                self._approval_task.cancel()
                try:
                    await self._approval_task
                except BaseException:
                    pass

            self._unsubscribe_ui_events()

            if self._session is not None:
                try:
                    await shutdown_runtime_session(self._session)
                except Exception:
                    pass

    try:
        app = AgentTuiApp()
        app.run()
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(run_tui())
