import numpy as np
import pytest

from src.gaussian_required_data import (
    insertAnomalyBlocks,
    makeExactFractionSubsetFromPool,
    makeGaussianTestSet,
    makeGaussianTrainingSet,
    makeQuestion9Dataset,
    makeQuestion11TestSet,
    makeSensitivityMasterPool,
)


def testGaussianTrainingSetShapeAndFiniteness():
    trainSequences = makeGaussianTrainingSet()

    assert trainSequences.shape == (200, 100)
    assert np.isfinite(trainSequences).all()


def testQuestion9UsesExactlyOneBlock():
    dataset = makeQuestion9Dataset()

    assert dataset.sequences.shape == (
        250,
        100,
    )

    assert np.sum(dataset.labels == 0) == 200
    assert np.sum(dataset.labels == 1) == 50

    anomalyMask = dataset.labels == 1

    assert np.all(
        dataset.nAnomalyBlocks[
            anomalyMask
        ]
        == 1
    )

    assert dataset.anomalyStarts.shape == (
        250,
        1,
    )

    assert np.all(
        dataset.anomalyEnd[
            anomalyMask
        ]
        - dataset.anomalyStart[
            anomalyMask
        ]
        == 20
    )


def testQuestion11UsesExactlyOneBlock():
    dataset = makeQuestion11TestSet(
        nNormal=90,
        nAnomaly=10,
        seed=2000,
    )

    anomalyMask = dataset.labels == 1

    assert np.all(
        dataset.nAnomalyBlocks[
            anomalyMask
        ]
        == 1
    )

    assert np.all(
        dataset.anomalyEnd[
            anomalyMask
        ]
        - dataset.anomalyStart[
            anomalyMask
        ]
        == 20
    )


def testMultipleAnomalyBlocksAreSeparated():
    randomGenerator = (
        np.random.default_rng(123)
    )

    sequence = randomGenerator.normal(
        0.0,
        1.0,
        size=100,
    )

    (
        modifiedSequence,
        starts,
        ends,
    ) = insertAnomalyBlocks(
        sequence,
        randomGenerator,
        nBlocks=3,
        anomalyBlockLength=15,
        minimumGap=2,
    )

    assert modifiedSequence.shape == (100,)
    assert starts.shape == (3,)
    assert ends.shape == (3,)
    assert np.all(ends - starts == 15)
    assert np.all(starts[1:] - ends[:-1] >= 2)


def testMultipleBlocksFailWhenTheyCannotFit():
    randomGenerator = (
        np.random.default_rng(123)
    )

    sequence = np.zeros(100)

    with pytest.raises(ValueError):
        insertAnomalyBlocks(
            sequence,
            randomGenerator,
            nBlocks=5,
            anomalyBlockLength=20,
            minimumGap=1,
        )


def testGaussianTestSetSupportsThreeBlocks():
    dataset = makeGaussianTestSet(
        nNormal=20,
        nAnomaly=10,
        nBlocks=3,
        anomalyBlockLength=15,
        minimumGap=2,
        seed=3000,
    )

    anomalyMask = dataset.labels == 1

    assert dataset.anomalyStarts.shape == (
        30,
        3,
    )

    assert np.all(
        dataset.nAnomalyBlocks[
            anomalyMask
        ]
        == 3
    )

    assert np.all(
        dataset.anomalyEnds[
            anomalyMask
        ]
        - dataset.anomalyStarts[
            anomalyMask
        ]
        == 15
    )


def testSharedSensitivityPoolExactFractions():
    pool = makeSensitivityMasterPool(
        nBlocks=1,
        minimumGap=0,
    )

    expectedCounts = {
        0.01: (990, 10),
        0.05: (950, 50),
        0.10: (900, 100),
        0.20: (800, 200),
    }

    for fraction, (
        expectedNormal,
        expectedAnomaly,
    ) in expectedCounts.items():
        dataset = (
            makeExactFractionSubsetFromPool(
                pool,
                fraction,
                nTotal=1000,
                shuffleSeed=4000,
            )
        )

        assert dataset.sequences.shape == (
            1000,
            100,
        )

        assert np.sum(
            dataset.labels == 0
        ) == expectedNormal

        assert np.sum(
            dataset.labels == 1
        ) == expectedAnomaly

        assert np.mean(
            dataset.labels
        ) == fraction


def testSensitivitySubsetsReuseMasterSequences():
    pool = makeSensitivityMasterPool(
        nBlocks=1,
        minimumGap=0,
        seed=5000,
    )

    lowFraction = (
        makeExactFractionSubsetFromPool(
            pool,
            0.01,
            shuffleSeed=5100,
        )
    )

    highFraction = (
        makeExactFractionSubsetFromPool(
            pool,
            0.20,
            shuffleSeed=5200,
        )
    )

    lowAnomalies = lowFraction.sequences[
        lowFraction.labels == 1
    ]

    highAnomalies = highFraction.sequences[
        highFraction.labels == 1
    ]

    # Every anomaly used in the 1% subset must also belong to
    # the common 20% master-derived subset.
    highAnomalyRows = {
        row.tobytes()
        for row in highAnomalies
    }

    assert all(
        row.tobytes() in highAnomalyRows
        for row in lowAnomalies
    )


def testGaussianGenerationIsReproducible():
    firstDataset = makeQuestion9Dataset(
        seed=1406
    )

    secondDataset = makeQuestion9Dataset(
        seed=1406
    )

    assert np.array_equal(
        firstDataset.sequences,
        secondDataset.sequences,
    )

    assert np.array_equal(
        firstDataset.labels,
        secondDataset.labels,
    )

    assert np.array_equal(
        firstDataset.anomalyStarts,
        secondDataset.anomalyStarts,
    )

    assert np.array_equal(
        firstDataset.anomalyEnds,
        secondDataset.anomalyEnds,
    )

    assert np.array_equal(
        firstDataset.nAnomalyBlocks,
        secondDataset.nAnomalyBlocks,
    )

def testSingleBlockSamplingCoversFullValidRange():
    sampledStarts = []

    for seed in range(200):
        randomGenerator = (
            np.random.default_rng(seed)
        )

        sequence = np.zeros(
            100,
            dtype=float,
        )

        (
            _,
            starts,
            ends,
        ) = insertAnomalyBlocks(
            sequence,
            randomGenerator,
            nBlocks=1,
            anomalyBlockLength=20,
            minimumGap=0,
        )

        sampledStarts.append(
            int(starts[0])
        )

        assert int(ends[0]) - int(starts[0]) == 20

    assert min(sampledStarts) <= 5
    assert max(sampledStarts) >= 75