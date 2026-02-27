from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from eval.core.contracts import AgentExecutor, BenchmarkSample, ExecutionResult


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
            include_tools=self._coerce_tool_list(opts.get("include_tools")),
            exclude_tools=self._coerce_tool_list(opts.get("exclude_tools")),
            model_override=self._coerce_optional_dict(opts.get("model_override")),
            bash_prompt_approval=self._coerce_optional_bool(opts.get("bash_prompt_approval"), default=False),
            toolsmaker_manual_approval=self._coerce_optional_bool(
                opts.get("toolsmaker_manual_approval"), default=False
            ),
            toolsmaker_auto_continue=self._coerce_optional_bool(opts.get("toolsmaker_auto_continue"), default=False),
            log_level=self._coerce_optional_str(opts.get("log_level"), default="quiet"),
        )
        enable_event_logger = self._coerce_optional_bool(opts.get("enable_event_logger"), default=True)

        session = agent_runtime.create_runtime_session(
            log_level=overrides.log_level,
            enable_event_logger=bool(enable_event_logger),
            overrides=overrides,
        )

        def _capture(event: Dict[str, Any]) -> None:
            events.append(self._to_jsonable(event))

        unsubscribe_trace = session.agent.subscribe(_capture)

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
        prompt = self._inject_attachment_context(sample.prompt, staged_files)

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

    def _inject_attachment_context(self, prompt: str, staged_files: List[Dict[str, str]]) -> str:
        if not staged_files:
            return prompt
        lines = [
            str(prompt or "").rstrip(),
            "",
            "Required attachment files are preloaded in the workspace at:",
        ]
        for item in staged_files:
            workspace_path = str(item.get("workspace_path", "") or item.get("local_path", "")).strip()
            if workspace_path:
                lines.append(f"- {workspace_path}")
        lines.append("Use these exact paths when opening attached files.")
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
