import os
from typing import List, Sequence

def parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    key = str(value).strip().lower()
    if key in {"1", "true", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def parse_path_list_env(name: str, default_paths: Sequence[str], base_dir: str) -> List[str]:
    value = os.getenv(name)
    raw_items = []
    if value is None:
        raw_items = [str(item) for item in default_paths]
    else:
        raw_items = [x.strip() for x in str(value).split(",") if x.strip()]
        if not raw_items:
            raw_items = [str(item) for item in default_paths]

    normalized: List[str] = []
    seen = set()
    for item in raw_items:
        candidate = item if os.path.isabs(item) else os.path.join(base_dir, item)
        root = os.path.realpath(candidate)
        if root in seen:
            continue
        normalized.append(root)
        seen.add(root)
    return normalized


def sorted_csv(values: Sequence[str]) -> str:
    cleaned = {str(item).strip() for item in values if str(item).strip()}
    return ", ".join(sorted(cleaned))
