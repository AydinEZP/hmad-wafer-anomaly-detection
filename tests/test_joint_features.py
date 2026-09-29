import numpy as np

from src.joint_features import batchJointFeatures, computeJointFeatures


def testJointFeatureShapeAndTransitionNormalization() -> None:
    sequence = np.array([0.0, 0.5, 3.5, 4.0, 0.2])
    states = np.array([0, 0, 1, 1, 0])
    features = computeJointFeatures(sequence, states, 2)
    assert features.shape == (6,)
    assert np.isclose(features[:4].sum(), 1.0)
    assert np.all(np.isfinite(features))


def testOneBasedAndZeroBasedStatesMatch() -> None:
    sequence = np.array([-0.5, 0.2, 3.8, 4.1])
    zeroBased = np.array([0, 0, 1, 1])
    oneBased = zeroBased + 1
    assert np.allclose(
        computeJointFeatures(sequence, zeroBased, 2),
        computeJointFeatures(sequence, oneBased, 2),
    )


def testEmptyStateProducesFiniteMeanFeature() -> None:
    sequence = np.array([0.1, -0.2, 0.3])
    states = np.zeros(3, dtype=int)
    features = computeJointFeatures(sequence, states, 2)
    assert features[-1] == 0.0
    assert np.all(np.isfinite(features))


def testBatchJointFeaturesShape() -> None:
    sequences = np.array([[0.0, 1.0, 3.0], [0.2, -0.1, 4.0]])
    states = np.array([[0, 0, 1], [0, 0, 1]])
    features = batchJointFeatures(sequences, states, 2)
    assert features.shape == (2, 6)


def testContiguousAndShuffledSequencesDifferInTransitions() -> None:
    randomGenerator = np.random.default_rng(1405)
    values = np.concatenate(
        (randomGenerator.normal(0.0, 1.0, 50), randomGenerator.normal(4.0, 1.0, 50))
    )
    shuffledValues = values.copy()
    randomGenerator.shuffle(shuffledValues)
    contiguousStates = np.where(values < 2.0, 0, 1)
    shuffledStates = np.where(shuffledValues < 2.0, 0, 1)
    contiguousFeatures = computeJointFeatures(values, contiguousStates, 2)
    shuffledFeatures = computeJointFeatures(shuffledValues, shuffledStates, 2)
    assert np.max(np.abs(contiguousFeatures[:4] - shuffledFeatures[:4])) > 0.15
