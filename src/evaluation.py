from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluateNormalityScores(
    labels: np.ndarray,
    normalityScores: np.ndarray,
) -> dict[str, float]:
    labelArray = np.asarray(labels, dtype=int).reshape(-1)
    scoreArray = np.asarray(normalityScores, dtype=float).reshape(-1)

    if labelArray.shape != scoreArray.shape:
        raise ValueError("labels and normalityScores must have equal lengths")

    if not np.all(np.isin(labelArray, [0, 1])):
        raise ValueError("labels must use 0 = normal and 1 = anomaly")

    if not np.all(np.isfinite(scoreArray)):
        raise ValueError("normalityScores contain non-finite values")

    anomalyScores = -scoreArray
    predictions = (scoreArray < 0.0).astype(int)

    return {
        "rocAuc": float(
            roc_auc_score(labelArray, anomalyScores)
        ),
        "precision": float(
            precision_score(
                labelArray,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labelArray,
                predictions,
                zero_division=0,
            )
        ),
        "f1Score": float(
            f1_score(
                labelArray,
                predictions,
                zero_division=0,
            )
        ),
    }