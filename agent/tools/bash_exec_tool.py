from __future__ import annotations

import asyncio
import fnmatch
import inspect
import os
import shlex
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..agent_types import AgentTool, AgentToolResult, TextContent

ApprovalFn = Callable[[Dict[str, Any]], bool | Awaitable[bool]]


@dataclass
class BashExecConfig:
    project_root: str
    allowed_roots: List[str]
    writable_roots: List[str]
    read_commands: Set[str]
    write_commands: Set[str]
    git_read_subcommands: Set[str]
    default_timeout_s: float = 15.0
    max_timeout_s: float = 60.0
    default_max_output_chars: int = 20_000
    max_output_chars_limit: int = 100_000


class BashExecTool(AgentTool):
    name = "bash_exec"
    description = (
        "Run one safe shell command without a shell. "
        "Read commands auto-run; write commands require approval."
    )
    parameters = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "Single command only. Shell operators and command chaining are blocked.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory; must be within allowed roots.",
            },
            "timeout_s": {
                "type": "number",
                "description": "Execution timeout in seconds. Clamped to a safe range.",
            },
            "max_output_chars": {
                "type": "integer",
                "description": "Max combined stdout/stderr chars in final result.",
            },
            "stdout_max_chars": {
                "type": "integer",
                "description": "Optional max chars retained from stdout before combined cap.",
            },
            "stderr_max_chars": {
                "type": "integer",
                "description": "Optional max chars retained from stderr before combined cap.",
            },
            "result_detail": {
                "type": "string",
                "description": "Tool result text verbosity: summary or detailed (default detailed).",
                "enum": ["summary", "detailed"],
            },
            "justification": {
                "type": "string",
                "description": "Optional rationale for audit and approval context.",
            },
        },
        "required": ["cmd"],
    }
    label = "Bash Exec"

    _BLOCKED_META: Tuple[str, ...] = ("|", ";", ">", "<", "&")
    _FIND_BLOCKED: Set[str] = {
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
        "-delete",
        "-fprint",
        "-fprint0",
        "-fprintf",
        "-fls",
    }
    _RG_BLOCKED_PREFIXES: Tuple[str, ...] = ("--pre=", "--pre-glob=")
    _RG_BLOCKED: Set[str] = {"--pre", "--pre-glob"}
    _GIT_BLOCKED: Set[str] = {"-C", "--git-dir", "--work-tree", "-c", "--config-env", "--exec-path"}
    _GIT_BLOCKED_PREFIXES: Tuple[str, ...] = (
        "-C=",
        "--git-dir=",
        "--work-tree=",
        "-c=",
        "--config-env=",
        "--exec-path=",
    )
    _DEST_OPT_BLOCKED: Set[str] = {"-t", "--target-directory"}
    _DEST_OPT_BLOCKED_PREFIXES: Tuple[str, ...] = ("--target-directory=",)
    _RG_SHORT_VALUE_FLAGS: Set[str] = {
        "-A",
        "-B",
        "-C",
        "-E",
        "-M",
        "-f",
        "-g",
        "-m",
        "-e",
        "-r",
        "-s",
        "-t",
        "-T",
        "-u",
    }
    _RG_LONG_VALUE_FLAGS: Set[str] = {
        "--after-context",
        "--before-context",
        "--context",
        "--encoding",
        "--max-columns",
        "--max-count",
        "--file",
        "--glob",
        "--glob-case-insensitive",
        "--ignore-file",
        "--replace",
        "--type",
        "--type-not",
        "--threads",
    }

    def __init__(self, config: BashExecConfig, approval_fn: Optional[ApprovalFn] = None) -> None:
        project_root = self._normalize_path(config.project_root, config.project_root)
        allowed_roots = self._normalize_roots(config.allowed_roots, project_root)
        writable_roots = self._normalize_roots(config.writable_roots, project_root)
        read_commands = {str(x).strip() for x in config.read_commands if str(x).strip()}
        write_commands = {str(x).strip() for x in config.write_commands if str(x).strip()}
        git_subcommands = {str(x).strip() for x in config.git_read_subcommands if str(x).strip()}

        self.config = BashExecConfig(
            project_root=project_root,
            allowed_roots=allowed_roots or [project_root],
            writable_roots=writable_roots or [project_root],
            read_commands=read_commands,
            write_commands=write_commands,
            git_read_subcommands=git_subcommands,
            default_timeout_s=float(config.default_timeout_s or 15.0),
            max_timeout_s=max(1.0, float(config.max_timeout_s or 60.0)),
            default_max_output_chars=max(256, int(config.default_max_output_chars or 20_000)),
            max_output_chars_limit=max(256, int(config.max_output_chars_limit or 100_000)),
        )
        self._approval_fn = approval_fn

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        started = time.perf_counter()
        cmd_raw = params.get("cmd", "")
        cmd = str(cmd_raw).strip() if isinstance(cmd_raw, str) else ""
        justification = str(params.get("justification", "") or "").strip()
        timeout_s = self._coerce_timeout(params.get("timeout_s"))
        max_output_chars = self._coerce_max_output(params.get("max_output_chars"))
        stdout_max_chars = self._coerce_optional_max_output(params.get("stdout_max_chars"))
        stderr_max_chars = self._coerce_optional_max_output(params.get("stderr_max_chars"))
        result_detail = self._coerce_result_detail(params.get("result_detail"))

        if not cmd:
            return self._blocked_result(
                command=cmd,
                argv=[],
                cwd="",
                block_reason="missing_cmd",
                risk="low",
                approved=False,
                max_output_chars=max_output_chars,
                started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )
        if len(cmd) > 2000:
            return self._blocked_result(
                command=cmd,
                argv=[],
                cwd="",
                block_reason="command_too_long",
                risk="low",
                approved=False,
                max_output_chars=max_output_chars,
                started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )
        if self._contains_blocked_shell_operator(cmd):
            return self._blocked_result(
                command=cmd,
                argv=[],
                cwd="",
                block_reason="blocked_shell_operator",
                risk="low",
                approved=False,
                max_output_chars=max_output_chars,
                started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )

        try:
            argv = shlex.split(cmd)
        except ValueError:
            return self._blocked_result(
                command=cmd,
                argv=[],
                cwd="",
                block_reason="invalid_shell_syntax",
                risk="low",
                approved=False,
                max_output_chars=max_output_chars,
                started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )
        if not argv:
            return self._blocked_result(
                command=cmd,
                argv=[],
                cwd="",
                block_reason="missing_cmd",
                risk="low",
                approved=False,
                max_output_chars=max_output_chars,
                started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )

        try:
            exec_cwd = self._resolve_cwd(params.get("cwd"))
        except ValueError as exc:
            return self._blocked_result(
                command=cmd,
                argv=argv,
                cwd="",
                block_reason=str(exc),
                risk="low",
                approved=False,
                max_output_chars=max_output_chars,
                started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )

        command_name = argv[0]
        if command_name not in self.config.read_commands and command_name not in self.config.write_commands:
            return self._blocked_result(
                command=cmd,
                argv=argv,
                cwd=exec_cwd,
                block_reason="command_not_allowed",
                risk="low",
                approved=False,
                max_output_chars=max_output_chars,
                started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )

        policy_error = self._validate_command_policy(command_name, argv, exec_cwd)
        risk = self._classify_risk(command_name)
        if policy_error is not None:
            return self._blocked_result(
                command=cmd,
                argv=argv,
                cwd=exec_cwd,
                block_reason=policy_error,
                risk=risk,
                approved=False,
                max_output_chars=max_output_chars,
                started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )

        approved = True
        if risk in {"medium", "high"}:
            approved = False
            if self._approval_fn is not None:
                approval_payload = {
                    "command": cmd,
                    "argv": list(argv),
                    "cwd": exec_cwd,
                    "risk": risk,
                    "justification": justification,
                }
                try:
                    decision = self._approval_fn(approval_payload)
                    if inspect.isawaitable(decision):
                        decision = await decision
                    approved = bool(decision)
                except Exception:
                    approved = False
            if not approved:
                return self._blocked_result(
                    command=cmd,
                    argv=argv,
                    cwd=exec_cwd,
                    block_reason="approval_required_or_denied",
                    risk=risk,
                    approved=False,
                    max_output_chars=max_output_chars,
                    started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
                )

        timed_out = False
        exit_code: Optional[int] = None
        stdout_text = ""
        stderr_text = ""
        truncated = False
        stdout_raw_chars = 0
        stderr_raw_chars = 0
        stdout_truncated = False
        stderr_truncated = False
        combined_truncated = False

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=exec_cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_raw, stderr_raw = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                timed_out = True
                proc.kill()
                stdout_raw, stderr_raw = await proc.communicate()
            exit_code = proc.returncode
            stdout_text = stdout_raw.decode("utf-8", errors="replace")
            stderr_text = stderr_raw.decode("utf-8", errors="replace")
            if command_name == "find" and self._looks_like_windows_find_error(stderr_text):
                handled, fallback_exit_code, fallback_stdout, fallback_stderr = self._fallback_find(argv, exec_cwd)
                if handled:
                    exit_code = fallback_exit_code
                    stdout_text = fallback_stdout
                    stderr_text = fallback_stderr
            stdout_raw_chars = len(stdout_text)
            stderr_raw_chars = len(stderr_text)
            (
                stdout_text,
                stderr_text,
                combined_truncated,
                stdout_truncated,
                stderr_truncated,
            ) = self._truncate_output(
                stdout_text,
                stderr_text,
                max_output_chars,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )
            truncated = bool(combined_truncated or stdout_truncated or stderr_truncated)
        except FileNotFoundError:
            handled, exit_code, stdout_text, stderr_text = self._execute_host_fallback(
                command=command_name,
                argv=argv,
                cwd=exec_cwd,
            )
            if not handled:
                return self._blocked_result(
                    command=cmd,
                    argv=argv,
                    cwd=exec_cwd,
                    block_reason="command_not_available_on_host",
                    risk=risk,
                    approved=approved,
                    max_output_chars=max_output_chars,
                    started=started,
                result_detail=result_detail,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
                )
            stdout_raw_chars = len(stdout_text)
            stderr_raw_chars = len(stderr_text)
            (
                stdout_text,
                stderr_text,
                combined_truncated,
                stdout_truncated,
                stderr_truncated,
            ) = self._truncate_output(
                stdout_text,
                stderr_text,
                max_output_chars,
                stdout_max_chars=stdout_max_chars,
                stderr_max_chars=stderr_max_chars,
            )
            truncated = bool(combined_truncated or stdout_truncated or stderr_truncated)
        except Exception as exc:
            stderr_text = str(exc)
            exit_code = None
            stderr_raw_chars = len(stderr_text)
            stdout_raw_chars = 0

        ok = bool(exit_code == 0 and not timed_out)
        details = self._details(
            ok=ok,
            blocked=False,
            block_reason="",
            approved=approved,
            risk=risk,
            command=cmd,
            argv=argv,
            cwd=exec_cwd,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=self._duration_ms(started),
            stdout=stdout_text,
            stderr=stderr_text,
            truncated=truncated,
            max_output_chars=max_output_chars,
            output_meta={
                "result_detail": result_detail,
                "stdout_max_chars": stdout_max_chars,
                "stderr_max_chars": stderr_max_chars,
                "stdout_raw_chars": stdout_raw_chars,
                "stderr_raw_chars": stderr_raw_chars,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "combined_truncated": combined_truncated,
            },
        )
        if timed_out:
            summary = f"bash_exec timed out after {timeout_s:.1f}s"
        elif exit_code is None:
            if stderr_text:
                summary = f"bash_exec failed to execute: {stderr_text}"
            else:
                summary = "bash_exec failed to execute"
        elif exit_code != 0 and stderr_text:
            summary = f"bash_exec exit_code={exit_code} stderr={stderr_text}"
        else:
            summary = f"bash_exec exit_code={exit_code}"
        result_text = self._build_result_text(
            summary=summary,
            stdout=stdout_text,
            stderr=stderr_text,
            result_detail=result_detail,
        )
        return AgentToolResult(content=[TextContent(type="text", text=result_text)], details=details)

    def _execute_host_fallback(
        self,
        *,
        command: str,
        argv: Sequence[str],
        cwd: str,
    ) -> Tuple[bool, Optional[int], str, str]:
        if command == "pwd":
            return True, 0, f"{cwd}\n", ""
        if command == "ls":
            return self._fallback_ls(argv, cwd)
        if command == "find":
            return self._fallback_find(argv, cwd)
        return False, None, "", ""

    def _contains_blocked_shell_operator(self, cmd: str) -> bool:
        in_single = False
        in_double = False
        escaped = False

        for ch in str(cmd):
            if escaped:
                escaped = False
                continue
            if ch == "\\" and not in_single:
                escaped = True
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if not in_single and not in_double and ch in self._BLOCKED_META:
                return True
        return False

    def _fallback_ls(self, argv: Sequence[str], cwd: str) -> Tuple[bool, int, str, str]:
        show_hidden = False
        classify = False
        positional: List[str] = []
        force_positional = False
        for token in list(argv[1:]):
            if force_positional:
                positional.append(token)
                continue
            if token == "--":
                force_positional = True
                continue
            if token in {"-a", "-A", "--all", "--almost-all"}:
                show_hidden = True
                continue
            if token in {"-F", "--classify"}:
                classify = True
                continue
            if token.startswith("-"):
                continue
            positional.append(token)

        targets = self._resolve_tokens_as_paths(positional, cwd) if positional else [cwd]
        output_lines: List[str] = []
        error_lines: List[str] = []
        exit_code = 0
        multi = len(targets) > 1

        for i, target in enumerate(targets):
            if not os.path.exists(target):
                exit_code = 1
                error_lines.append(f"ls: cannot access '{target}': No such file or directory")
                continue
            if multi:
                if i > 0:
                    output_lines.append("")
                output_lines.append(f"{target}:")
            if os.path.isdir(target):
                try:
                    names = sorted(os.listdir(target))
                except OSError as exc:
                    exit_code = 1
                    error_lines.append(f"ls: cannot open directory '{target}': {exc}")
                    continue
                if not show_hidden:
                    names = [x for x in names if not x.startswith(".")]
                for name in names:
                    display = name
                    if classify:
                        full = os.path.join(target, name)
                        if os.path.islink(full):
                            display += "@"
                        elif os.path.isdir(full):
                            display += "/"
                        elif os.access(full, os.X_OK):
                            display += "*"
                    output_lines.append(display)
            else:
                name = os.path.basename(target) or target
                output_lines.append(name)

        stdout = ("\n".join(output_lines) + "\n") if output_lines else ""
        stderr = ("\n".join(error_lines) + "\n") if error_lines else ""
        return True, exit_code, stdout, stderr

    def _looks_like_windows_find_error(self, stderr_text: str) -> bool:
        lowered = str(stderr_text or "").strip().lower()
        return "parameter format not correct" in lowered

    def _fallback_find(self, argv: Sequence[str], cwd: str) -> Tuple[bool, int, str, str]:
        path_tokens, filters, parse_error = self._parse_find_fallback(argv)
        if parse_error is not None:
            return True, 2, "", f"{parse_error}\n"

        output_lines: List[str] = []
        error_lines: List[str] = []
        exit_code = 0
        resolved_paths = self._resolve_tokens_as_paths(path_tokens, cwd)

        for raw_token, start_path in zip(path_tokens, resolved_paths):
            if not os.path.exists(start_path):
                exit_code = 1
                error_lines.append(f"find: '{raw_token}': No such file or directory")
                continue

            for abs_path, depth in self._iter_find_paths(start_path):
                if not self._find_matches(abs_path, depth, filters):
                    continue
                output_lines.append(self._render_find_path(raw_token, start_path, abs_path))

        stdout = ("\n".join(output_lines) + "\n") if output_lines else ""
        stderr = ("\n".join(error_lines) + "\n") if error_lines else ""
        return True, exit_code, stdout, stderr

    def _parse_find_fallback(
        self,
        argv: Sequence[str],
    ) -> Tuple[List[str], Dict[str, Any], Optional[str]]:
        tokens = list(argv[1:])
        path_tokens: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in {"!", "(", ")"} or token.startswith("-"):
                break
            path_tokens.append(token)
            index += 1
        if not path_tokens:
            path_tokens = ["."]

        filters: Dict[str, Any] = {
            "mindepth": 0,
            "maxdepth": None,
            "type": None,
            "name": None,
            "case_insensitive": False,
        }
        while index < len(tokens):
            token = tokens[index]
            if token == "-print":
                index += 1
                continue
            if token in {"-name", "-iname"}:
                if index + 1 >= len(tokens):
                    return path_tokens, filters, f"find: missing pattern for {token}"
                filters["name"] = str(tokens[index + 1])
                filters["case_insensitive"] = token == "-iname"
                index += 2
                continue
            if token == "-type":
                if index + 1 >= len(tokens):
                    return path_tokens, filters, "find: missing type for -type"
                value = str(tokens[index + 1]).strip().lower()
                if value not in {"f", "d"}:
                    return path_tokens, filters, f"find: unsupported -type value: {value or '(empty)'}"
                filters["type"] = value
                index += 2
                continue
            if token in {"-maxdepth", "-mindepth"}:
                if index + 1 >= len(tokens):
                    return path_tokens, filters, f"find: missing integer for {token}"
                try:
                    value = int(tokens[index + 1])
                except Exception:
                    return path_tokens, filters, f"find: invalid integer for {token}: {tokens[index + 1]}"
                if value < 0:
                    return path_tokens, filters, f"find: {token} must be >= 0"
                filters[token[1:]] = value
                index += 2
                continue
            return path_tokens, filters, f"find: unsupported expression: {token}"

        return path_tokens, filters, None

    def _iter_find_paths(self, start_path: str) -> List[Tuple[str, int]]:
        items: List[Tuple[str, int]] = [(start_path, 0)]
        if not os.path.isdir(start_path):
            return items

        for root, dirs, files in os.walk(start_path):
            dirs.sort()
            files.sort()
            rel_root = os.path.relpath(root, start_path)
            root_depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            for name in dirs:
                items.append((os.path.join(root, name), root_depth + 1))
            for name in files:
                items.append((os.path.join(root, name), root_depth + 1))
        return items

    def _find_matches(self, abs_path: str, depth: int, filters: Dict[str, Any]) -> bool:
        mindepth = int(filters.get("mindepth") or 0)
        maxdepth = filters.get("maxdepth")
        if depth < mindepth:
            return False
        if maxdepth is not None and depth > int(maxdepth):
            return False

        type_filter = str(filters.get("type") or "").strip().lower()
        if type_filter == "f" and not os.path.isfile(abs_path):
            return False
        if type_filter == "d" and not os.path.isdir(abs_path):
            return False

        pattern = filters.get("name")
        if pattern:
            basename = os.path.basename(abs_path)
            if bool(filters.get("case_insensitive")):
                if not fnmatch.fnmatch(basename.lower(), str(pattern).lower()):
                    return False
            elif not fnmatch.fnmatchcase(basename, str(pattern)):
                return False
        return True

    def _render_find_path(self, raw_token: str, start_path: str, abs_path: str) -> str:
        rendered_root = raw_token or "."
        rel_path = os.path.relpath(abs_path, start_path)
        if rel_path == ".":
            return rendered_root
        separator = "\\" if "\\" in rendered_root and "/" not in rendered_root else "/"
        if rendered_root in {".", "./"}:
            return f".{separator}{rel_path.replace(os.sep, separator)}"
        rendered_root = rendered_root.rstrip("/\\")
        return f"{rendered_root}{separator}{rel_path.replace(os.sep, separator)}"

    def _validate_command_policy(self, command: str, argv: Sequence[str], cwd: str) -> Optional[str]:
        if command == "find":
            for token in argv[1:]:
                if token in self._FIND_BLOCKED:
                    return "find_option_not_allowed"
        elif command == "rg":
            for token in argv[1:]:
                if token in self._RG_BLOCKED or any(token.startswith(prefix) for prefix in self._RG_BLOCKED_PREFIXES):
                    return "rg_option_not_allowed"
        elif command == "git":
            for token in argv[1:]:
                if (token.startswith("-C") and token != "-C") or (token.startswith("-c") and token != "-c"):
                    return "git_option_not_allowed"
                if token in self._GIT_BLOCKED or any(token.startswith(prefix) for prefix in self._GIT_BLOCKED_PREFIXES):
                    return "git_option_not_allowed"
            subcommand = self._git_subcommand(argv)
            if not subcommand:
                return "git_subcommand_missing"
            if subcommand not in self.config.git_read_subcommands:
                return "git_subcommand_not_allowed"
        elif command in {"cp", "mv"}:
            for token in argv[1:]:
                if token.startswith("-t") and token != "-t":
                    return "destination_option_not_allowed"
                if token in self._DEST_OPT_BLOCKED or any(
                    token.startswith(prefix) for prefix in self._DEST_OPT_BLOCKED_PREFIXES
                ):
                    return "destination_option_not_allowed"

        read_paths, write_paths, parse_error = self._extract_paths(command, argv, cwd)
        if parse_error is not None:
            return parse_error
        for path in read_paths:
            if not self._path_in_roots(path, self.config.allowed_roots):
                return "path_outside_allowed_roots"
        for path in write_paths:
            if not self._path_in_roots(path, self.config.writable_roots):
                return "path_outside_writable_roots"
            if any(path == root for root in self.config.writable_roots):
                return "write_target_is_root_blocked"
        return None

    def _extract_paths(self, command: str, argv: Sequence[str], cwd: str) -> Tuple[List[str], List[str], Optional[str]]:
        if command == "pwd":
            return [], [], None

        if command in {"ls", "cat", "head", "tail", "wc"}:
            short_flags: Set[str] = set()
            long_flags: Set[str] = set()
            if command in {"head", "tail"}:
                short_flags = {"-n", "-c"}
                long_flags = {"--lines", "--bytes"}
            elif command == "wc":
                long_flags = {"--files0-from"}
            positionals = self._collect_positionals(argv[1:], short_flags, long_flags)
            return self._resolve_tokens_as_paths(positionals, cwd), [], None

        if command == "find":
            tokens = argv[1:]
            path_tokens: List[str] = []
            for token in tokens:
                if token in {"!", "(", ")"} or token.startswith("-"):
                    break
                path_tokens.append(token)
            if not path_tokens:
                path_tokens = ["."]
            return self._resolve_tokens_as_paths(path_tokens, cwd), [], None

        if command == "rg":
            positionals = self._collect_positionals(argv[1:], self._RG_SHORT_VALUE_FLAGS, self._RG_LONG_VALUE_FLAGS)
            if len(positionals) <= 1:
                return [], [], None
            return self._resolve_tokens_as_paths(positionals[1:], cwd), [], None

        if command == "git":
            pathspec = self._git_pathspec(argv)
            return self._resolve_tokens_as_paths(pathspec, cwd), [], None

        if command in {"mkdir", "touch", "rm"}:
            short_flags: Set[str] = set()
            long_flags: Set[str] = set()
            if command == "mkdir":
                short_flags = {"-m"}
                long_flags = {"--mode"}
            elif command == "touch":
                short_flags = {"-d", "-t", "-r"}
                long_flags = {"--date", "--reference"}
            targets = self._collect_positionals(argv[1:], short_flags, long_flags)
            return [], self._resolve_tokens_as_paths(targets, cwd), None

        if command in {"cp", "mv"}:
            positionals = self._collect_positionals(argv[1:], short_value_flags=set(), long_value_flags=set())
            if len(positionals) < 2:
                return [], [], None
            src_tokens = positionals[:-1]
            dst_token = positionals[-1]
            read_paths = self._resolve_tokens_as_paths(src_tokens, cwd)
            write_paths = self._resolve_tokens_as_paths([dst_token], cwd)
            if command == "mv":
                write_paths.extend(read_paths)
            return read_paths, write_paths, None

        return [], [], None

    def _collect_positionals(
        self,
        tokens: Sequence[str],
        short_value_flags: Optional[Set[str]] = None,
        long_value_flags: Optional[Set[str]] = None,
    ) -> List[str]:
        short_flags = short_value_flags or set()
        long_flags = long_value_flags or set()
        result: List[str] = []
        expect_value = False
        force_positional = False

        for token in tokens:
            if expect_value:
                expect_value = False
                continue
            if force_positional:
                result.append(token)
                continue
            if token == "--":
                force_positional = True
                continue
            if token == "-":
                result.append(token)
                continue
            if token.startswith("--"):
                if "=" in token:
                    continue
                if token in long_flags:
                    expect_value = True
                continue
            if token.startswith("-"):
                if token in short_flags:
                    expect_value = True
                continue
            result.append(token)
        return result

    def _git_subcommand(self, argv: Sequence[str]) -> str:
        tokens = list(argv[1:])
        if not tokens:
            return ""
        for token in tokens:
            if token == "--":
                continue
            if token.startswith("-"):
                continue
            return token
        return ""

    def _git_pathspec(self, argv: Sequence[str]) -> List[str]:
        tokens = list(argv[1:])
        if not tokens:
            return []
        subcommand_seen = False
        for i, token in enumerate(tokens):
            if not subcommand_seen:
                if token == "--" or token.startswith("-"):
                    continue
                subcommand_seen = True
                continue
            if token == "--":
                return tokens[i + 1 :]
        return []

    def _resolve_tokens_as_paths(self, tokens: Sequence[str], cwd: str) -> List[str]:
        paths: List[str] = []
        for token in tokens:
            stripped = str(token).strip()
            if not stripped or stripped == "-":
                continue
            paths.append(self._normalize_path(stripped, cwd))
        return paths

    def _resolve_cwd(self, cwd_value: Any) -> str:
        if cwd_value is None or str(cwd_value).strip() == "":
            candidate = self.config.project_root
        elif isinstance(cwd_value, str):
            candidate = self._normalize_path(cwd_value, self.config.project_root)
        else:
            raise ValueError("invalid_cwd")
        if not self._path_in_roots(candidate, self.config.allowed_roots):
            raise ValueError("cwd_outside_allowed_roots")
        return candidate

    def _classify_risk(self, command: str) -> str:
        if command == "rm":
            return "high"
        if command in {"mkdir", "touch", "cp", "mv"}:
            return "medium"
        return "low"

    def _coerce_timeout(self, value: Any) -> float:
        try:
            timeout = float(value if value is not None else self.config.default_timeout_s)
        except Exception:
            timeout = float(self.config.default_timeout_s)
        timeout = min(timeout, float(self.config.max_timeout_s))
        timeout = max(1.0, timeout)
        return timeout

    def _coerce_max_output(self, value: Any) -> int:
        try:
            max_chars = int(value if value is not None else self.config.default_max_output_chars)
        except Exception:
            max_chars = int(self.config.default_max_output_chars)
        max_chars = min(max_chars, int(self.config.max_output_chars_limit))
        max_chars = max(256, max_chars)
        return max_chars

    def _coerce_optional_max_output(self, value: Any) -> Optional[int]:
        if value is None or str(value).strip() == "":
            return None
        try:
            max_chars = int(value)
        except Exception:
            return None
        if max_chars <= 0:
            return None
        max_chars = min(max_chars, int(self.config.max_output_chars_limit))
        return max_chars

    @staticmethod
    def _coerce_result_detail(value: Any) -> str:
        detail = str(value if value is not None else "detailed").strip().lower()
        if detail not in {"summary", "detailed"}:
            return "detailed"
        return detail

    @staticmethod
    def _build_result_text(summary: str, stdout: str, stderr: str, result_detail: str) -> str:
        if result_detail == "summary":
            return summary
        lines = [summary, "stdout:", stdout if stdout else "(empty)", "stderr:", stderr if stderr else "(empty)"]
        return "\n".join(lines)

    def _truncate_output(
        self,
        stdout: str,
        stderr: str,
        max_chars: int,
        *,
        stdout_max_chars: Optional[int] = None,
        stderr_max_chars: Optional[int] = None,
    ) -> Tuple[str, str, bool, bool, bool]:
        out = stdout
        err = stderr
        stdout_truncated = False
        stderr_truncated = False

        if stdout_max_chars is not None and len(out) > stdout_max_chars:
            out = out[:stdout_max_chars]
            stdout_truncated = True
        if stderr_max_chars is not None and len(err) > stderr_max_chars:
            err = err[:stderr_max_chars]
            stderr_truncated = True

        total = len(out) + len(err)
        if total <= max_chars:
            return out, err, False, stdout_truncated, stderr_truncated

        out_len = len(out)
        err_len = len(err)
        combined_truncated = True

        if out_len == 0:
            err = err[:max_chars]
        elif err_len == 0:
            out = out[:max_chars]
        elif max_chars <= 1:
            err = err[:1]
            out = ""
        else:
            out_budget = min(out_len, max_chars // 2)
            err_budget = min(err_len, max_chars - out_budget)
            remaining = max_chars - (out_budget + err_budget)
            if remaining > 0:
                add_out = min(remaining, out_len - out_budget)
                out_budget += add_out
                remaining -= add_out
            if remaining > 0:
                err_budget += min(remaining, err_len - err_budget)
            out = out[:out_budget]
            err = err[:err_budget]

        stdout_truncated = stdout_truncated or (len(out) < out_len)
        stderr_truncated = stderr_truncated or (len(err) < err_len)
        return out, err, combined_truncated, stdout_truncated, stderr_truncated

    @staticmethod
    def _duration_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    def _blocked_result(
        self,
        *,
        command: str,
        argv: Sequence[str],
        cwd: str,
        block_reason: str,
        risk: str,
        approved: bool,
        max_output_chars: int,
        started: float,
        result_detail: str = "detailed",
        stdout_max_chars: Optional[int] = None,
        stderr_max_chars: Optional[int] = None,
    ) -> AgentToolResult:
        details = self._details(
            ok=False,
            blocked=True,
            block_reason=block_reason,
            approved=approved,
            risk=risk,
            command=command,
            argv=argv,
            cwd=cwd,
            exit_code=None,
            timed_out=False,
            duration_ms=self._duration_ms(started),
            stdout="",
            stderr="",
            truncated=False,
            max_output_chars=max_output_chars,
            output_meta={
                "result_detail": result_detail,
                "stdout_max_chars": stdout_max_chars,
                "stderr_max_chars": stderr_max_chars,
                "stdout_raw_chars": 0,
                "stderr_raw_chars": 0,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "combined_truncated": False,
            },
        )
        text = f"bash_exec blocked: {block_reason}"
        return AgentToolResult(content=[TextContent(type="text", text=text)], details=details)

    @staticmethod
    def _details(
        *,
        ok: bool,
        blocked: bool,
        block_reason: str,
        approved: bool,
        risk: str,
        command: str,
        argv: Sequence[str],
        cwd: str,
        exit_code: Optional[int],
        timed_out: bool,
        duration_ms: int,
        stdout: str,
        stderr: str,
        truncated: bool,
        max_output_chars: int,
        output_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        details = {
            "ok": ok,
            "blocked": blocked,
            "block_reason": block_reason,
            "approved": approved,
            "risk": risk,
            "command": command,
            "argv": list(argv),
            "cwd": cwd,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
            "max_output_chars": max_output_chars,
        }
        if output_meta:
            details.update(output_meta)
        return details

    @staticmethod
    def _normalize_roots(values: Sequence[str], base: str) -> List[str]:
        roots: List[str] = []
        seen: Set[str] = set()
        for value in values:
            normalized = BashExecTool._normalize_path(value, base)
            if not normalized or normalized in seen:
                continue
            roots.append(normalized)
            seen.add(normalized)
        return roots

    @staticmethod
    def _normalize_path(path: str, base: str) -> str:
        value = str(path or "").strip()
        if not value:
            return ""
        candidate = value if os.path.isabs(value) else os.path.join(base, value)
        return os.path.realpath(candidate)

    @staticmethod
    def _path_in_roots(path: str, roots: Sequence[str]) -> bool:
        target = os.path.realpath(path)
        for root in roots:
            try:
                if os.path.commonpath([target, root]) == root:
                    return True
            except Exception:
                continue
        return False
