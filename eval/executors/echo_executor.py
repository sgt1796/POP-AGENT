from __future__ import annotations

import time
from typing import Any, Dict

from eval.core.contracts import AgentExecutor, BenchmarkSample, ExecutionResult


class EchoExecutor(AgentExecutor):
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
        del timeout_s, sample_index, run_id, run_dir, executor_options
        started = time.perf_counter()
        prediction = sample.prompt
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ExecutionResult(
            status="ok",
            prediction=prediction,
            usage={"delta": {"calls": 0}},
            latency_ms=latency_ms,
            error=None,
            events=[{"type": "echo", "prompt": sample.prompt}],
            trace_ref=f"events.jsonl#sample_id={sample.sample_id}",
        )
