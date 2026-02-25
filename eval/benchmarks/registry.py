from __future__ import annotations

from typing import Callable, Dict, List

from eval.benchmarks.gaia.adapter import GaiaAdapter
from eval.core.contracts import BenchmarkAdapter


_REGISTRY: Dict[str, Callable[[], BenchmarkAdapter]] = {}


def register_adapter(name: str, factory: Callable[[], BenchmarkAdapter]) -> None:
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("Benchmark adapter name cannot be empty")
    _REGISTRY[key] = factory


def get_adapter(name: str) -> BenchmarkAdapter:
    key = str(name or "").strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown benchmark adapter: {name}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[key]()


def list_adapters() -> List[str]:
    return sorted(_REGISTRY.keys())


register_adapter("gaia", GaiaAdapter)
