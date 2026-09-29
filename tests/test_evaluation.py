import numpy as np
import pytest

from src.evaluation import evaluateNormalityScores


def testPerfectNormalityScoreOrientation():
    labels = np.array([0, 0, 1, 1])
    normalityScores = np.array([2.0, 1.0, -1.0, -2.0])

    metrics = evaluateNormalityScores(
        labels,
        normalityScores,
    )

    assert metrics["rocAuc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1Score"] == 1.0


def testRejectsInvalidLabels():
    labels = np.array([-1, 1])
    normalityScores = np.array([1.0, -1.0])

    with pytest.raises(ValueError):
        evaluateNormalityScores(
            labels,
            normalityScores,
        )


def testRejectsDifferentLengths():
    labels = np.array([0, 1])
    normalityScores = np.array([1.0])

    with pytest.raises(ValueError):
        evaluateNormalityScores(
            labels,
            normalityScores,
        )


def testRejectsNonFiniteScores():
    labels = np.array([0, 1])
    normalityScores = np.array([1.0, np.nan])

    with pytest.raises(ValueError):
        evaluateNormalityScores(
            labels,
            normalityScores,
        )