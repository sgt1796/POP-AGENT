from __future__ import annotations

from typing import Any, List

from eval.core.contracts import ScoreResult
from eval.score import normalize_number_str, normalize_str, question_scorer, split_string


def _is_float_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _normalize_value(value: Any, *, remove_punct: bool = True) -> Any:
    text = "" if value is None else str(value)
    if _is_float_like(text):
        return normalize_number_str(text)
    if any(ch in text for ch in [",", ";"]):
        normalized: List[Any] = []
        for element in split_string(text):
            if _is_float_like(element):
                normalized.append(normalize_number_str(element))
            else:
                normalized.append(normalize_str(element, remove_punct=remove_punct))
        return normalized
    return normalize_str(text, remove_punct=remove_punct)


def score_gaia_prediction(prediction: str, ground_truth: str) -> ScoreResult:
    pred_text = "" if prediction is None else str(prediction)
    gt_text = "" if ground_truth is None else str(ground_truth)

    correct = bool(question_scorer(model_answer=pred_text, ground_truth=gt_text))
    reason = "exact_match" if correct else "mismatch"

    # Keep punctuation for list elements to mirror GAIA list behavior.
    remove_punct = not any(ch in gt_text for ch in [",", ";"])
    return ScoreResult(
        correct=correct,
        score=1.0 if correct else 0.0,
        reason=reason,
        normalized_prediction=_normalize_value(pred_text, remove_punct=remove_punct),
        normalized_ground_truth=_normalize_value(gt_text, remove_punct=remove_punct),
    )
