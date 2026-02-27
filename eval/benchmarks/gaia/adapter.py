from __future__ import annotations

import json
import os
import re
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
    _FILE_NAME_COLS = ("file_name", "File Name", "filename")
    _FILE_PATH_COLS = ("file_path", "File Path", "filepath")
    _URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")

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
        file_base_uri = self._resolve_file_base_uri(
            split_key=split_key,
            parquet_path=parquet_path,
            options=options,
        )

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
            required_files = self._extract_required_files(row_dict, file_base_uri=file_base_uri)
            assets = {"required_files": required_files} if required_files else {}

            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    prompt=prompt,
                    ground_truth=ground_truth,
                    metadata=metadata,
                    assets=assets,
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

    def _resolve_file_base_uri(self, *, split_key: str, parquet_path: str, options: Dict[str, Any]) -> str:
        explicit = str(options.get("file_base_uri", "") or options.get("asset_base_uri", "")).strip()
        if explicit:
            return self._ensure_trailing_slash(explicit)

        hf_base_uri = str(options.get("hf_base_uri", "")).strip()
        if hf_base_uri:
            return self._ensure_trailing_slash(hf_base_uri)

        normalized_parquet = str(parquet_path or "").replace("\\", "/")
        split_suffix = self._SPLITS.get(split_key, "")
        if split_suffix and normalized_parquet.endswith(split_suffix):
            base = normalized_parquet[: -len(split_suffix)]
            return self._ensure_trailing_slash(base)

        if self._looks_like_uri(normalized_parquet):
            parent = normalized_parquet.rsplit("/", 1)[0]
            return self._ensure_trailing_slash(parent)

        return ""

    def _extract_required_files(self, row_dict: Dict[str, Any], *, file_base_uri: str) -> List[Dict[str, str]]:
        names_raw = self._first_present_value(row_dict, self._FILE_NAME_COLS)
        paths_raw = self._first_present_value(row_dict, self._FILE_PATH_COLS)

        names = self._normalize_string_list(names_raw)
        paths = self._normalize_string_list(paths_raw)
        if not paths:
            return []

        required_files: List[Dict[str, str]] = []
        for index, dataset_path in enumerate(paths):
            name_value = names[index] if index < len(names) else ""
            source_uri = self._resolve_file_source_uri(dataset_path, file_base_uri=file_base_uri)
            name = str(name_value or os.path.basename(dataset_path) or f"attachment_{index + 1}").strip()
            required_files.append(
                {
                    "name": name,
                    "dataset_path": dataset_path,
                    "source_uri": source_uri,
                }
            )

        return required_files

    def _first_present_value(self, row_dict: Dict[str, Any], candidates: Sequence[str]) -> Any:
        for key in candidates:
            if key in row_dict:
                return row_dict.get(key)
        return None

    def _normalize_string_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]

        text = str(value).strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]

        separators = ("\n", ";", "|")
        for separator in separators:
            if separator in text:
                return [item.strip() for item in text.split(separator) if item.strip()]

        return [text]

    def _resolve_file_source_uri(self, dataset_path: str, *, file_base_uri: str) -> str:
        raw_path = str(dataset_path or "").strip()
        if not raw_path:
            return ""
        if self._looks_like_uri(raw_path) or os.path.isabs(raw_path):
            return raw_path
        if file_base_uri:
            return self._join_uri(file_base_uri, raw_path)
        return raw_path

    def _looks_like_uri(self, value: str) -> bool:
        return bool(self._URI_SCHEME_RE.match(str(value or "").strip()))

    def _join_uri(self, base: str, suffix: str) -> str:
        base_clean = str(base or "").rstrip("/")
        suffix_clean = str(suffix or "").lstrip("/")
        if not base_clean:
            return suffix_clean
        if not suffix_clean:
            return base_clean + "/"
        return f"{base_clean}/{suffix_clean}"

    def _ensure_trailing_slash(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.endswith("/"):
            return text
        return text + "/"

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
