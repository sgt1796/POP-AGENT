from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from agent.agent_types import AgentMessage, TextContent
from eval.core.contracts import AgentExecutor, BenchmarkSample, ExecutionResult


_GENERIC_WEB_DISCOVERY_TOOLS = {
    "perplexity_search",
    "jina_web_snapshot",
    "perplexity_web_snapshot",
}
_HARD_BASH_BLOCK_REASONS = {
    "command_not_allowed",
    "blocked_shell_operator",
    "command_not_available_on_host",
    "approval_required_or_denied",
}
_CALCULATOR_IO_ERROR_SNIPPETS = (
    "only direct function calls are allowed",
    "function not allowed",
)
_HTTP_STATUS_RE = re.compile(r"\b([45]\d{2})\b")


def _event_result_details(event: Dict[str, Any]) -> Dict[str, Any]:
    raw = event.get("result")
    if isinstance(raw, dict):
        details = raw.get("details")
        if isinstance(details, dict):
            return details
    details = getattr(raw, "details", None)
    if isinstance(details, dict):
        return details
    return {}


def _extract_http_status_code(details: Dict[str, Any]) -> Optional[int]:
    for key in ("status_code", "jina_status_code"):
        try:
            value = details.get(key)
            if value is not None:
                return int(value)
        except Exception:
            pass
    error_text = str(details.get("error") or "").strip()
    match = _HTTP_STATUS_RE.search(error_text)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


class _EvalSteeringGuard:
    def __init__(self, agent: Any, *, generic_web_budget: int = 4) -> None:
        self._agent = agent
        self._generic_web_budget = max(1, int(generic_web_budget))
        self._generic_web_calls = 0
        self._sent_keys: set[str] = set()

    def on_event(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        if str(event.get("type") or "") != "tool_execution_end":
            return

        tool_name = str(event.get("toolName") or "").strip()
        details = _event_result_details(event)

        if tool_name in _GENERIC_WEB_DISCOVERY_TOOLS:
            self._generic_web_calls += 1
            if self._generic_web_calls >= self._generic_web_budget:
                self._steer_once(
                    "generic-web-budget",
                    "Evaluation steering:\n"
                    "- The generic web discovery budget is exhausted for this sample.\n"
                    "- Stop reformulating broad searches or reopening the same source family.\n"
                    "- If the likely answer is already in a downloaded or local document, use file_read with query and bounded context on the exact phrase or heading.\n"
                    "- Otherwise use the strongest exact source already found, spend at most one targeted verification step, then answer.",
                )

        if tool_name == "bash_exec":
            block_reason = str(details.get("block_reason") or "").strip()
            if block_reason.startswith("path_outside_") or block_reason in _HARD_BASH_BLOCK_REASONS:
                self._steer_once(
                    "bash-hard-block",
                    "Evaluation steering:\n"
                    "- bash_exec is hard-blocked in this environment.\n"
                    "- Do not retry shell commands or syntax variants.\n"
                    "- Switch to non-shell tools only.\n"
                    "- If the target is already local, use file_read with query and context instead of shell grep.\n"
                    "- If you already have a likely source, verify it directly and answer.",
                )

        if tool_name == "file_read" and str(details.get("error") or "").strip() == "parse_error":
            self._steer_once(
                "file-read-parse-error",
                "Evaluation steering:\n"
                "- The local artifact could not be parsed as requested.\n"
                "- If it came from download_url_to_file, inspect final_url, pdf_link_candidates, content_preview, or saved_landing_page_path, then save the landing page as .html if needed.\n"
                "- Once you have the landing page or local document, use file_read with query and bounded context on the exact phrase or chapter heading instead of shelling out.\n"
                "- Recover one exact document path, then resume bounded local reading.",
            )

        if tool_name == "jina_web_snapshot":
            status_code = _extract_http_status_code(details)
            if status_code is not None and 400 <= status_code < 500:
                self._steer_once(
                    "jina-client-error",
                    "Evaluation steering:\n"
                    "- jina_web_snapshot hit a proxy or access error on the snapshot service.\n"
                    "- Do not keep reformulating the same generic search.\n"
                    "- Try the original URL directly or use download_url_to_file to save it as local .html, then use file_read with query and bounded context on the exact phrase, heading, or version section.\n"
                    "- If the source is genuinely gated, stay anchored to the exact DOI, title, or domain and spend at most one targeted alternative retrieval step before answering.",
                )

        if tool_name == "download_url_to_file" and str(details.get("error") or "").strip() == "unexpected_content_type":
            self._steer_once(
                "download-unexpected-content-type",
                "Evaluation steering:\n"
                "- The requested file resolved to a different content type than expected, often a landing or verification page.\n"
                "- Use final_url, pdf_link_candidates, content_preview, saved_landing_page_path, or the source landing page as the next step.\n"
                "- If you save the landing page locally, use file_read with query and bounded context on the exact phrase or chapter heading.\n"
                "- Do not retry the same PDF URL without a new concrete lead.",
            )

        if tool_name == "calculator":
            error_text = str(details.get("error") or "").strip().lower()
            if any(snippet in error_text for snippet in _CALCULATOR_IO_ERROR_SNIPPETS):
                self._steer_once(
                    "calculator-io-misuse",
                    "Evaluation steering:\n"
                    "- Calculator is arithmetic-only here.\n"
                    "- Do not use it to open files, inspect text, or emulate a scripting environment.\n"
                    "- Extract the exact values with file_read or retrieval tools first, then compute with one direct expression.",
                )

    def _steer_once(self, key: str, text: str) -> None:
        if key in self._sent_keys:
            return
        self._sent_keys.add(key)
        self._agent.steer(
            AgentMessage(
                role="user",
                content=[TextContent(type="text", text=text)],
                timestamp=time.time(),
            )
        )


class Agent1RuntimeExecutor(AgentExecutor):
    async def run_sample(
        self,
        sample: BenchmarkSample,
        *,
        timeout_s: float,
        sample_index: int,
        run_id: str,
        run_dir: str,
        executor_options: Dict[str, Any],
    ) -> ExecutionResult:
        from agent_build.agent1 import runtime as agent_runtime
        from agent_build.agent1.usage_reporting import usage_delta

        opts = dict(executor_options or {})
        warnings: List[str] = []
        events: List[Dict[str, Any]] = []

        memory_dir = os.path.join(
            run_dir,
            "_memory",
            f"{run_id}_sample_{sample_index:06d}_{self._safe_id(sample.sample_id)}",
        )

        overrides = agent_runtime.RuntimeOverrides(
            long_memory_base_path=str(opts.get("long_memory_base_path") or memory_dir),
            enable_memory=self._coerce_optional_bool(opts.get("enable_memory"), default=True),
            enable_auto_title=self._coerce_optional_bool(opts.get("enable_auto_title"), default=False),
            include_tools=self._coerce_tool_list(opts.get("include_tools")),
            exclude_tools=self._coerce_tool_list(opts.get("exclude_tools")),
            model_override=self._coerce_optional_dict(opts.get("model_override")),
            bash_prompt_approval=self._coerce_optional_bool(opts.get("bash_prompt_approval"), default=False),
            log_level=self._coerce_optional_str(opts.get("log_level"), default="quiet"),
        )
        enable_event_logger = self._coerce_optional_bool(opts.get("enable_event_logger"), default=True)

        session = agent_runtime.create_runtime_session(
            log_level=overrides.log_level,
            enable_event_logger=bool(enable_event_logger),
            overrides=overrides,
        )
        steering_guard = _EvalSteeringGuard(session.agent)

        def _capture(event: Dict[str, Any]) -> None:
            events.append(self._to_jsonable(event))

        unsubscribe_trace = session.agent.subscribe(_capture)
        unsubscribe_guard = session.agent.subscribe(steering_guard.on_event)

        before_usage = self._coerce_optional_dict(getattr(session.agent, "get_usage_summary", lambda: {})()) or {}
        last_usage = None
        status = "ok"
        prediction = ""
        error: Optional[str] = None
        started = time.perf_counter()
        staged_files = self._stage_required_files(
            sample=sample,
            run_dir=run_dir,
            sample_index=sample_index,
            run_id=run_id,
            warnings=warnings,
        )
        prompt = self._augment_eval_prompt(sample.prompt, staged_files)

        try:
            if timeout_s and float(timeout_s) > 0:
                prediction = await asyncio.wait_for(
                    agent_runtime.run_user_turn(session, prompt, on_warning=warnings.append),
                    timeout=float(timeout_s),
                )
            else:
                prediction = await agent_runtime.run_user_turn(session, prompt, on_warning=warnings.append)
            get_last = getattr(session.agent, "get_last_usage", None)
            if callable(get_last):
                last_usage = self._coerce_optional_dict(get_last())
        except Exception as exc:
            status = "error"
            error = self._format_error(exc, timeout_s=timeout_s)
        finally:
            unsubscribe_trace()
            unsubscribe_guard()
            try:
                await agent_runtime.shutdown_runtime_session(session)
            except Exception as shutdown_exc:
                if error is None:
                    status = "error"
                    error = f"shutdown error: {shutdown_exc}"

        latency_ms = (time.perf_counter() - started) * 1000.0
        after_usage = self._coerce_optional_dict(getattr(session.agent, "get_usage_summary", lambda: {})()) or {}
        usage = {
            "before": before_usage,
            "after": after_usage,
            "delta": usage_delta(before_usage, after_usage),
            "last": last_usage,
            "warnings": warnings,
        }
        if staged_files:
            usage["attachments"] = staged_files

        if warnings and error is None:
            usage["warning_count"] = len(warnings)

        return ExecutionResult(
            status=status,
            prediction=str(prediction or ""),
            usage=usage,
            latency_ms=latency_ms,
            error=error,
            events=events,
            trace_ref=f"events.jsonl#sample_id={sample.sample_id}",
        )

    def _safe_id(self, value: str) -> str:
        text = str(value or "sample")
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)

    def _collect_required_files(self, sample: BenchmarkSample) -> List[Dict[str, str]]:
        assets = sample.assets if isinstance(sample.assets, dict) else {}
        required_files = assets.get("required_files")
        files: List[Dict[str, str]] = []
        if isinstance(required_files, list):
            for item in required_files:
                normalized = self._normalize_required_file_item(item)
                if normalized is not None:
                    files.append(normalized)

        if files:
            return files

        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        legacy_path = str(metadata.get("file_path", "") or "").strip()
        if not legacy_path:
            return []
        legacy_name = str(metadata.get("file_name", "") or "").strip()
        return [
            {
                "name": legacy_name,
                "dataset_path": legacy_path,
                "source_uri": legacy_path,
            }
        ]

    def _normalize_required_file_item(self, value: Any) -> Optional[Dict[str, str]]:
        if not isinstance(value, dict):
            return None
        dataset_path = str(value.get("dataset_path") or value.get("file_path") or "").strip()
        source_uri = str(value.get("source_uri") or value.get("source_path") or "").strip()
        name = str(value.get("name") or value.get("file_name") or "").strip()
        if not (dataset_path or source_uri):
            return None
        return {
            "name": name,
            "dataset_path": dataset_path,
            "source_uri": source_uri,
        }

    def _stage_required_files(
        self,
        *,
        sample: BenchmarkSample,
        run_dir: str,
        sample_index: int,
        run_id: str,
        warnings: List[str],
    ) -> List[Dict[str, str]]:
        required_files = self._collect_required_files(sample)
        if not required_files:
            return []

        sample_suffix = f"{run_id}_sample_{sample_index:06d}_{self._safe_id(sample.sample_id)}"
        stage_dir = os.path.join(run_dir, "_attachments", sample_suffix)
        os.makedirs(stage_dir, exist_ok=True)

        staged: List[Dict[str, str]] = []
        for index, item in enumerate(required_files):
            dataset_path = str(item.get("dataset_path", "") or "").strip()
            source_uri = str(item.get("source_uri", "") or "").strip()
            source = source_uri or dataset_path
            if not source:
                warnings.append(f"[attachments] sample {sample.sample_id}: missing source for required file #{index + 1}.")
                continue

            default_name = os.path.basename(dataset_path or source) or f"attachment_{index + 1}"
            safe_name = self._safe_filename(str(item.get("name", "") or default_name), fallback=default_name)
            destination = self._next_available_path(stage_dir, safe_name)
            try:
                self._copy_required_file(source, destination)
            except Exception as exc:
                warnings.append(f"[attachments] sample {sample.sample_id}: failed to stage '{source}': {exc}")
                continue

            staged.append(
                {
                    "name": os.path.basename(destination),
                    "dataset_path": dataset_path,
                    "source_uri": source,
                    "local_path": destination,
                    "workspace_path": self._workspace_relative_path(destination),
                }
            )

        return staged

    def _augment_eval_prompt(self, prompt: str, staged_files: List[Dict[str, str]]) -> str:
        lines = [
            str(prompt or "").rstrip(),
            "",
            "Evaluation execution guidance:",
            "- Prefer exact local files and precise structured tools before generic web discovery.",
            "- For scholarly or document tasks, prefer openalex_works and exact local files before perplexity_search or web snapshots.",
            "- Do not answer from search-result snippets alone if you can open the cited page or a local artifact and verify the exact field.",
            "- If a page is relevant but dense, extract the target field from the exact nearby passage instead of relying on a broad summary of the page.",
            "- For quote-in-document tasks, recover the exact phrase in the exact title, chapter, page, or preview path and answer from the nearby passage only.",
            "- For large local text, HTML, RST, or PDF documents, use file_read with query and bounded context instead of sequential scanning or shell grep.",
            "- If a tool returns concrete recovery hints such as final_url, pdf_link_candidates, or content_preview, plus saved_landing_page_path when present, use those exact leads before broad search.",
            "- If jina_web_snapshot fails with a 4xx or proxy access error on a known page URL, try the original URL directly or use download_url_to_file to save it as local .html, then inspect it with file_read before broadening search.",
            "- If an exact-source fetch fails and later results only surface tangential names, generic summaries, or unverified numbers, do not turn that drift into the final answer; stay anchored to the DOI, title, quote, domain, or entity chain.",
            (
                "- Generic web discovery budget for this sample: at most 4 total calls across "
                "perplexity_search, jina_web_snapshot, and perplexity_web_snapshot unless a new call "
                "adds a concrete new constraint such as a domain filter or a distinct source family."
            ),
            "- Once that budget is spent, give the best supported final answer instead of reformulating the same search.",
            "- Treat bash_exec as local-shell inspection only. Do not use it for network fetches, Python one-liners, grep pipelines, or shell syntax variants.",
            "- If bash_exec is blocked, that is a hard constraint for this sample; switch tools immediately instead of retrying shell alternatives.",
            "- For calculator, use a single expression with direct function calls and bindings; do not use import, lambda, __import__, or attribute access like math.sin.",
            "- If calculator returns an unsupported syntax/function error, rewrite the expression with direct allowed calls or bindings and retry once before answering.",
            "- Do not use calculator to open files, inspect text, or simulate scripting; extract evidence first, then compute from the explicit values only.",
            "- For counts, distances, comparisons, and max/min selection over a bounded set, extract the concrete inputs and eligible candidates first and compute from those explicit values rather than mental arithmetic or a guessed set.",
            "- Before using calculator for a count or difference, write down the exact source-backed operands and make sure they follow the task's counting convention, such as unique winners, same-line stops, or season-specific measurements.",
            "- Before answering, verify the requested output field and counting convention: exact entity, requested unit, item index, inclusive vs exclusive counts, and requested precision or rounding rule.",
            "- Translate the requested precision into the final numeric format before answering; for example, nearest 0.001 of the reported unit requires three decimals.",
            "- When the prompt asks for answer text or number only, output only the filled answer field; do not echo the format template, labels, or extra units unless explicitly requested.",
            "- If the candidate answer is still a placeholder, copied template, or generic filler token, treat the field as unverified and use the strongest targeted verification call before answering.",
            "- If the exact requested field is still unverified, spend one targeted verification call on the strongest candidate source before answering.",
        ]
        if staged_files:
            lines.extend(
                [
                    "",
                    "Required attachment files are preloaded in the workspace at:",
                ]
            )
            for item in staged_files:
                workspace_path = str(item.get("workspace_path", "") or item.get("local_path", "")).strip()
                if workspace_path:
                    lines.append(f"- {workspace_path}")
            lines.extend(
                [
                    "Local files are primary evidence for this task.",
                    "Open these exact local paths first with bounded local reads.",
                    "Use file_read first for scientific text files such as .pdb, .cif, and .mmcif.",
                    "Do not fetch remote copies or snapshots of these files unless the local path fails or is incomplete.",
                    "Use these exact paths when opening attached files.",
                ]
            )
        return "\n".join(lines).strip()

    def _copy_required_file(self, source: str, destination: str) -> None:
        raw_source = str(source or "").strip()
        if not raw_source:
            raise ValueError("empty attachment source")

        if self._looks_like_uri(raw_source):
            try:
                import fsspec
            except Exception as exc:
                raise RuntimeError("fsspec is required to fetch URI-based attachments") from exc
            with fsspec.open(raw_source, "rb") as src:
                with open(destination, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            return

        source_path = raw_source
        if not os.path.isabs(source_path):
            source_path = os.path.join(os.getcwd(), source_path)
        source_path = os.path.abspath(source_path)
        destination_path = os.path.abspath(destination)
        if source_path == destination_path:
            return
        if not os.path.isfile(source_path):
            raise FileNotFoundError(source_path)
        shutil.copyfile(source_path, destination_path)

    def _looks_like_uri(self, value: str) -> bool:
        return "://" in str(value or "")

    def _safe_filename(self, value: str, *, fallback: str) -> str:
        raw = str(value or "").strip().replace("\\", "/")
        name = raw.rsplit("/", 1)[-1].strip()
        safe = "".join(ch if ch.isalnum() or ch in {".", "-", "_"} else "_" for ch in name)
        safe = safe.strip("._")
        if safe:
            return safe
        raw_fallback = str(fallback or "attachment").strip().replace("\\", "/")
        fallback_name = raw_fallback.rsplit("/", 1)[-1].strip() or "attachment"
        return "".join(ch if ch.isalnum() or ch in {".", "-", "_"} else "_" for ch in fallback_name)

    def _next_available_path(self, directory: str, filename: str) -> str:
        candidate = os.path.join(directory, filename)
        if not os.path.exists(candidate):
            return candidate

        stem, ext = os.path.splitext(filename)
        suffix = 2
        while True:
            candidate = os.path.join(directory, f"{stem}_{suffix}{ext}")
            if not os.path.exists(candidate):
                return candidate
            suffix += 1

    def _workspace_relative_path(self, path: str) -> str:
        try:
            rel = os.path.relpath(path, os.getcwd())
        except Exception:
            rel = path
        return rel.replace("\\", "/")

    def _coerce_tool_list(self, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return None

    def _coerce_optional_bool(self, value: Any, *, default: Optional[bool]) -> Optional[bool]:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def _coerce_optional_dict(self, value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            return dict(value)
        return None

    def _coerce_optional_str(self, value: Any, *, default: Optional[str]) -> Optional[str]:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    def _to_jsonable(self, value: Any) -> Any:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self._to_jsonable(value.to_dict())
            except Exception:
                pass

        if is_dataclass(value):
            return self._to_jsonable(asdict(value))

        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]

        if isinstance(value, tuple):
            return [self._to_jsonable(item) for item in value]

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    def _format_error(self, exc: Exception, *, timeout_s: float) -> str:
        if isinstance(exc, asyncio.TimeoutError):
            timeout_value = float(timeout_s) if timeout_s and float(timeout_s) > 0 else 0.0
            return f"timeout after {timeout_value:g}s"

        text = str(exc).strip()
        if text:
            return text
        return exc.__class__.__name__
