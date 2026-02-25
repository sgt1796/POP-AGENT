from eval.benchmarks.gaia.scorer import score_gaia_prediction


def test_gaia_scorer_numeric_and_string_cases():
    numeric = score_gaia_prediction(prediction="$4.00", ground_truth="4")
    assert numeric.correct is True
    assert numeric.score == 1.0
    assert numeric.normalized_prediction == 4.0

    string_mismatch = score_gaia_prediction(prediction="Alpha", ground_truth="beta")
    assert string_mismatch.correct is False
    assert string_mismatch.score == 0.0
    assert string_mismatch.reason == "mismatch"


def test_gaia_scorer_list_case():
    result = score_gaia_prediction(prediction="red, blue", ground_truth="red,blue")
    assert result.correct is True
    assert isinstance(result.normalized_prediction, list)
    assert result.normalized_prediction == result.normalized_ground_truth
