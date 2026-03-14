from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple


def path_in_roots(path: str, roots: Sequence[str]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([root, path]) == root:
                return True
        except ValueError:
            continue
    return False


def normalize_allowed_roots(
    workspace_root: Optional[str] = None,
    allowed_roots: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str]]:
    root = os.path.realpath(str(workspace_root or os.getcwd()))
    normalized: List[str] = [root]
    seen = {root}

    for item in list(allowed_roots or []):
        raw = str(item or "").strip()
        if not raw:
            continue
        candidate = raw if os.path.isabs(raw) else os.path.join(root, raw)
        resolved = os.path.realpath(candidate)
        if resolved in seen:
            continue
        normalized.append(resolved)
        seen.add(resolved)

    return root, normalized
