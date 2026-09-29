from pathlib import Path

import numpy as np

from src.structured_dataset import (
    createStructuredSequenceDataset,
    loadStructuredSequenceDataset,
    saveStructuredSequenceDataset,
    selectFixedTestSubset,
)


def testDatasetMeetsPublishedMinimumRequirements() -> None:
    dataset = createStructuredSequenceDataset(seed=1405)
    dataset.validate()
    assert dataset.trainSequences.shape == (200, 100)
    assert dataset.trainSequences.size >= 5000
    assert np.sum(dataset.testLabels == 0) == 160
    assert np.sum(dataset.testLabels == 1) == 40
    assert np.all(dataset.anomalyStarts[dataset.testLabels == 0] == -1)
    assert set(dataset.anomalyTypes[dataset.testLabels == 1]) == {
        "rapid_switch",
        "block_shuffle",
    }


def testDatasetIsExactlyReproducible() -> None:
    first = createStructuredSequenceDataset(seed=1405)
    second = createStructuredSequenceDataset(seed=1405)
    assert np.array_equal(first.trainSequences, second.trainSequences)
    assert np.array_equal(first.testSequences, second.testSequences)
    assert np.array_equal(first.testLabels, second.testLabels)
    assert np.array_equal(first.anomalyTypes, second.anomalyTypes)


def testSaveLoadRoundTrip(tmp_path: Path) -> None:
    dataset = createStructuredSequenceDataset(seed=1405)
    datasetPath, metadataPath, _ = saveStructuredSequenceDataset(dataset, tmp_path)
    loaded = loadStructuredSequenceDataset(datasetPath, metadataPath)
    assert np.array_equal(dataset.trainSequences, loaded.trainSequences)
    assert np.array_equal(dataset.testStates, loaded.testStates)
    assert dataset.metadata == loaded.metadata


def testSensitivitySubsetsKeepAllFortyAnomalies() -> None:
    dataset = createStructuredSequenceDataset(seed=1405)
    for nNormal in [160, 120, 80, 60]:
        indices = selectFixedTestSubset(dataset, nNormal, seed=1405 + nNormal)
        labels = dataset.testLabels[indices]
        assert np.sum(labels == 0) == nNormal
        assert np.sum(labels == 1) == 40


def testBlockShufflePreservesGlobalOrderInvariantStatistics() -> None:
    dataset = createStructuredSequenceDataset(seed=1405)
    shuffleIndices = np.flatnonzero(dataset.anomalyTypes == "block_shuffle")
    assert shuffleIndices.size == 20
    # The test verifies the intended nontriviality indirectly: anomaly values
    # remain within the same broad range as nominal values and do not form an
    # extreme-value-only problem.
    nominal = dataset.testSequences[dataset.testLabels == 0]
    shuffled = dataset.testSequences[shuffleIndices]
    assert abs(float(nominal.mean()) - float(shuffled.mean())) < 0.20
    assert abs(float(nominal.std()) - float(shuffled.std())) < 0.20
    assert np.max(np.abs(shuffled)) < 5.0
