import numpy as np

from src.baseline_features import (
    flattenedFeatures,
    globalSummaryFeatures,
    windowSummaryFeatures,
)
from src.ocsvm_baseline import fitOcsvmBaseline


def test_baseline_feature_shapes_for_wafer_length() -> None:
    sequences = np.random.default_rng(1405).normal(size=(7, 152))
    assert flattenedFeatures(sequences).shape == (7, 152)
    assert globalSummaryFeatures(sequences).shape == (7, 5)
    assert windowSummaryFeatures(sequences, nWindows=5).shape == (7, 25)


def test_window_features_support_non_divisible_length() -> None:
    sequences = np.arange(2 * 152, dtype=float).reshape(2, 152)
    features = windowSummaryFeatures(sequences, nWindows=5)
    assert features.shape == (2, 25)
    assert np.all(np.isfinite(features))


def test_ocsvm_scaler_and_scores_are_finite() -> None:
    rng = np.random.default_rng(1405)
    train = globalSummaryFeatures(rng.normal(size=(40, 20)))
    test = globalSummaryFeatures(rng.normal(size=(10, 20)))
    baseline = fitOcsvmBaseline(
        train, representation="global_summary", nu=0.1, gamma="scale"
    )
    scores = baseline.decisionFunction(test)
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))
