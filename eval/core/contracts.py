from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


@dataclass
class EvalConfig:
    benchmark: str = "gaia"
    split: str = "validation"
    limit: Optional[int] = None
    seed: int = 0
    timeout_s: float = 120.0
    output_root: str = "eval/runs"
    run_id: Optional[str] = None
    benchmark_options: Dict[str, Any] = field(default_factory=dict)
    executor: str = "agent1"
    executor_options: Dict[str, Any] = field(default_factory=dict)
    redact_patterns: List[str] = field(default_factory=list)
    continue_on_error: bool = True
    concurrency: int = 1


@dataclass
class BenchmarkSample:
    sample_id: str
    prompt: str
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    assets: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    correct: bool
    score: float
    reason: str
    normalized_prediction: Any
    normalized_ground_truth: Any


@dataclass
class ExecutionResult:
    status: str
    prediction: str
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    trace_ref: Optional[str] = None


@dataclass
class SampleResult:
    sample_id: str
    status: str
    prediction: str
    score_result: ScoreResult
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    trace_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    run_id: str
    benchmark: str
    split: str
    total: int
    correct: int
    error_count: int
    accuracy: float
    run_dir: str
    started_at: str
    ended_at: str
    duration_s: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    name: str

    def load_samples(
        self,
        *,
        split: str,
        limit: Optional[int],
        seed: int,
        options: Dict[str, Any],
    ) -> List[BenchmarkSample]:
        ...

    def build_prompt(self, sample: BenchmarkSample) -> str:
        ...

    def score(self, prediction: str, ground_truth: str, sample: BenchmarkSample) -> ScoreResult:
        ...

    def aggregate(self, sample_results: Sequence[SampleResult]) -> Dict[str, Any]:
        ...


class AgentExecutor(Protocol):
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
        ...


class ArtifactWriter(Protocol):
    def write_manifest(self, payload: Dict[str, Any]) -> str:
        ...

    def write_sample(self, payload: Dict[str, Any]) -> None:
        ...

    def write_summary(self, payload: Dict[str, Any]) -> str:
        ...
