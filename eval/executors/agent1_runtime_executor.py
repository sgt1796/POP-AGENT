from __future__ import annotations

import asyncio
import os
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

        session = agent_runtime.create_runtime_session(
            log_level=overrides.log_level,
            enable_event_logger=False,
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

        try:
            if timeout_s and float(timeout_s) > 0:
                prediction = await asyncio.wait_for(
                    agent_runtime.run_user_turn(session, sample.prompt, on_warning=warnings.append),
                    timeout=float(timeout_s),
                )
            else:
                prediction = await agent_runtime.run_user_turn(session, sample.prompt, on_warning=warnings.append)
            get_last = getattr(session.agent, "get_last_usage", None)
            if callable(get_last):
                last_usage = self._coerce_optional_dict(get_last())
        except Exception as exc:
            status = "error"
            error = str(exc)
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
