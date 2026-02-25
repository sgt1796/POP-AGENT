from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from eval.benchmarks.gaia.scorer import score_gaia_prediction
from eval.core.contracts import BenchmarkSample, SampleResult, ScoreResult


class GaiaAdapter:
    name = "gaia"

    _SPLITS = {
        "test": "2023/test/metadata.parquet",
        "validation": "2023/validation/metadata.parquet",
    }
    _PROMPT_COLS = ("question", "Question")
    _GROUND_TRUTH_COLS = ("final_answer", "Final answer", "answer")
    _ID_COLS = ("task_id", "id")

    def load_samples(
        self,
        *,
        split: str,
        limit: Optional[int],
        seed: int,
        options: Dict[str, Any],
    ) -> List[BenchmarkSample]:
        split_key = str(split or "validation").strip().lower()
        if split_key not in self._SPLITS:
            raise ValueError(f"Unsupported GAIA split: {split}. Supported: {sorted(self._SPLITS.keys())}")

        parquet_path = str(options.get("parquet_path", "")).strip()
        if not parquet_path:
            base_uri = str(options.get("hf_base_uri", "hf://datasets/gaia-benchmark/GAIA/")).strip()
            if not base_uri.endswith("/"):
                base_uri += "/"
            parquet_path = base_uri + self._SPLITS[split_key]

        frame = pd.read_parquet(parquet_path)
        columns = list(frame.columns)

        prompt_col = self._pick_column(columns, self._PROMPT_COLS)
        gt_col = self._pick_column(columns, self._GROUND_TRUTH_COLS)
        id_col = self._pick_column(columns, self._ID_COLS)

        missing = []
        if prompt_col is None:
            missing.append(f"prompt candidates={self._PROMPT_COLS}")
        if gt_col is None:
            missing.append(f"ground-truth candidates={self._GROUND_TRUTH_COLS}")
        if missing:
            raise ValueError(
                "GAIA schema error: "
                + "; ".join(missing)
                + f"; discovered columns={columns}"
            )

        should_shuffle = bool(options.get("shuffle", False))
        if should_shuffle:
            frame = frame.sample(frac=1.0, random_state=int(seed or 0)).reset_index(drop=True)

        if limit is not None:
            n = max(0, int(limit))
            frame = frame.head(n)

        samples: List[BenchmarkSample] = []
        for idx, row in frame.iterrows():
            row_dict = {str(k): self._coerce_scalar(v) for k, v in row.to_dict().items()}
            sample_id_raw = row_dict.get(id_col) if id_col else idx
            sample_id = str(sample_id_raw)
            prompt = str(row_dict.get(prompt_col, "") or "")
            ground_truth = str(row_dict.get(gt_col, "") or "")

            metadata = {
                key: value
                for key, value in row_dict.items()
                if key not in {prompt_col, gt_col} and (id_col is None or key != id_col)
            }

            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    prompt=prompt,
                    ground_truth=ground_truth,
                    metadata=metadata,
                    assets={},
                )
            )

        return samples

    def build_prompt(self, sample: BenchmarkSample) -> str:
        return sample.prompt

    def score(self, prediction: str, ground_truth: str, sample: BenchmarkSample) -> ScoreResult:
        del sample
        return score_gaia_prediction(prediction=prediction, ground_truth=ground_truth)

    def aggregate(self, sample_results: Sequence[SampleResult]) -> Dict[str, Any]:
        total = len(sample_results)
        correct = sum(1 for item in sample_results if item.score_result.correct)
        error_count = sum(1 for item in sample_results if item.status == "error")
        accuracy = (float(correct) / float(total)) if total else 0.0

        by_level: Dict[str, Dict[str, Any]] = {}
        for result in sample_results:
            level = self._resolve_level(result.metadata)
            if level is None:
                continue
            bucket = by_level.setdefault(level, {"total": 0, "correct": 0, "error_count": 0, "accuracy": 0.0})
            bucket["total"] += 1
            if result.score_result.correct:
                bucket["correct"] += 1
            if result.status == "error":
                bucket["error_count"] += 1

        for level, values in by_level.items():
            level_total = int(values.get("total", 0))
            level_correct = int(values.get("correct", 0))
            values["accuracy"] = (float(level_correct) / float(level_total)) if level_total else 0.0
            by_level[level] = values

        return {
            "total": total,
            "correct": correct,
            "error_count": error_count,
            "accuracy": accuracy,
            "by_level": by_level,
        }

    def _pick_column(self, columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    def _resolve_level(self, metadata: Dict[str, Any]) -> Optional[str]:
        for key in ("Level", "level", "difficulty"):
            if key in metadata:
                value = str(metadata.get(key, "")).strip()
                if value:
                    return value
        return None

    def _coerce_scalar(self, value: Any) -> Any:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value
