from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np


# فعلاً سیاست seed را تغییر نمی‌دهیم تا TA پاسخ دهد.
BASE_SEED = 1405
Q9_SEED = 1406
SENSITIVITY_POOL_SEED = 1407
SUPPLEMENTARY_SEED = 1408


@dataclass(frozen=True)
class GaussianTestSet:
    """Gaussian sequence-level anomaly-detection dataset.

    Label convention:
        0 = normal
        1 = anomaly

    Interval convention:
        starts are inclusive
        ends are exclusive

    anomalyStarts and anomalyEnds have shape:
        (n_sequences, max_blocks)

    Unused entries and all entries belonging to normal sequences
    are represented by -1.
    """

    sequences: np.ndarray
    labels: np.ndarray
    anomalyStarts: np.ndarray
    anomalyEnds: np.ndarray
    nAnomalyBlocks: np.ndarray

    @property
    def anomalyStart(self) -> np.ndarray:
        """First anomaly start for backward compatibility."""

        return self.anomalyStarts[:, 0]

    @property
    def anomalyEnd(self) -> np.ndarray:
        """First anomaly end for backward compatibility."""

        return self.anomalyEnds[:, 0]

    @property
    def maxBlocks(self) -> int:
        return int(self.anomalyStarts.shape[1])

    def validate(
        self,
        *,
        sequenceLength: int,
        anomalyBlockLength: int,
        minimumGap: int = 1,
    ) -> None:
        nSequences = self.sequences.shape[0]

        if self.sequences.shape != (
            nSequences,
            sequenceLength,
        ):
            raise ValueError(
                "Unexpected Gaussian sequence shape: "
                f"{self.sequences.shape}"
            )

        expectedVectorShape = (nSequences,)

        if self.labels.shape != expectedVectorShape:
            raise ValueError("labels have an unexpected shape")

        if self.nAnomalyBlocks.shape != expectedVectorShape:
            raise ValueError(
                "nAnomalyBlocks has an unexpected shape"
            )

        if self.anomalyStarts.ndim != 2:
            raise ValueError(
                "anomalyStarts must be a two-dimensional matrix"
            )

        if self.anomalyEnds.shape != self.anomalyStarts.shape:
            raise ValueError(
                "anomalyStarts and anomalyEnds must have equal shapes"
            )

        if self.anomalyStarts.shape[0] != nSequences:
            raise ValueError(
                "Anomaly-interval matrices have an unexpected row count"
            )

        if not np.all(np.isfinite(self.sequences)):
            raise ValueError(
                "Gaussian sequences contain non-finite values"
            )

        if not np.all(np.isin(self.labels, [0, 1])):
            raise ValueError(
                "labels must use 0 = normal and 1 = anomaly"
            )

        normalMask = self.labels == 0
        anomalyMask = self.labels == 1

        if not np.all(self.nAnomalyBlocks[normalMask] == 0):
            raise ValueError(
                "Normal sequences must have zero anomaly blocks"
            )

        if not np.all(self.anomalyStarts[normalMask] == -1):
            raise ValueError(
                "Normal sequences must have anomalyStarts = -1"
            )

        if not np.all(self.anomalyEnds[normalMask] == -1):
            raise ValueError(
                "Normal sequences must have anomalyEnds = -1"
            )

        if np.any(self.nAnomalyBlocks[anomalyMask] <= 0):
            raise ValueError(
                "Every anomalous sequence must contain at least one block"
            )

        if np.any(
            self.nAnomalyBlocks
            > self.anomalyStarts.shape[1]
        ):
            raise ValueError(
                "nAnomalyBlocks exceeds the interval-matrix width"
            )

        for sequenceIndex in np.flatnonzero(anomalyMask):
            blockCount = int(
                self.nAnomalyBlocks[sequenceIndex]
            )

            starts = self.anomalyStarts[
                sequenceIndex,
                :blockCount,
            ]

            ends = self.anomalyEnds[
                sequenceIndex,
                :blockCount,
            ]

            unusedStarts = self.anomalyStarts[
                sequenceIndex,
                blockCount:,
            ]

            unusedEnds = self.anomalyEnds[
                sequenceIndex,
                blockCount:,
            ]

            if not np.all(unusedStarts == -1):
                raise ValueError(
                    "Unused anomaly starts must be -1"
                )

            if not np.all(unusedEnds == -1):
                raise ValueError(
                    "Unused anomaly ends must be -1"
                )

            if not np.all(starts >= 0):
                raise ValueError(
                    "Anomaly starts must be nonnegative"
                )

            if not np.all(ends <= sequenceLength):
                raise ValueError(
                    "Anomaly ends exceed the sequence length"
                )

            if not np.all(
                ends - starts == anomalyBlockLength
            ):
                raise ValueError(
                    "Anomaly blocks have incorrect lengths"
                )

            if not np.all(
                starts[1:] > starts[:-1]
            ):
                raise ValueError(
                    "Anomaly blocks must be sorted"
                )

            if blockCount > 1:
                actualGaps = (
                    starts[1:] - ends[:-1]
                )

                if not np.all(
                    actualGaps >= minimumGap
                ):
                    raise ValueError(
                        "Anomaly blocks overlap or violate "
                        "the required minimum gap"
                    )


@dataclass(frozen=True)
class GaussianSensitivityPool:
    """Reusable class-conditional master pool for sensitivity tests."""

    normalSequences: np.ndarray
    anomalousSequences: np.ndarray
    anomalyStarts: np.ndarray
    anomalyEnds: np.ndarray
    nAnomalyBlocks: np.ndarray

    def validate(
        self,
        *,
        sequenceLength: int,
        anomalyBlockLength: int,
        minimumGap: int = 1,
    ) -> None:
        if self.normalSequences.ndim != 2:
            raise ValueError(
                "normalSequences must be a matrix"
            )

        if self.anomalousSequences.ndim != 2:
            raise ValueError(
                "anomalousSequences must be a matrix"
            )

        if self.normalSequences.shape[1] != sequenceLength:
            raise ValueError(
                "Normal master-pool sequence length is incorrect"
            )

        if self.anomalousSequences.shape[1] != sequenceLength:
            raise ValueError(
                "Anomalous master-pool sequence length is incorrect"
            )

        temporaryDataset = GaussianTestSet(
            sequences=self.anomalousSequences,
            labels=np.ones(
                self.anomalousSequences.shape[0],
                dtype=int,
            ),
            anomalyStarts=self.anomalyStarts,
            anomalyEnds=self.anomalyEnds,
            nAnomalyBlocks=self.nAnomalyBlocks,
        )

        temporaryDataset.validate(
            sequenceLength=sequenceLength,
            anomalyBlockLength=anomalyBlockLength,
            minimumGap=minimumGap,
        )

        if not np.all(np.isfinite(self.normalSequences)):
            raise ValueError(
                "Normal master-pool sequences contain "
                "non-finite values"
            )


def generateNominalSequences(
    nSequences: int,
    sequenceLength: int,
    randomGenerator: np.random.Generator,
    *,
    nominalMean: float = 0.0,
    nominalStd: float = 1.0,
) -> np.ndarray:
    """Generate independent nominal Gaussian sequences."""

    if nSequences <= 0:
        raise ValueError("nSequences must be positive")

    if sequenceLength <= 0:
        raise ValueError("sequenceLength must be positive")

    if nominalStd <= 0.0:
        raise ValueError("nominalStd must be positive")

    return randomGenerator.normal(
        loc=nominalMean,
        scale=nominalStd,
        size=(nSequences, sequenceLength),
    ).astype(float)

def _sampleSeparatedBlockStarts(
    *,
    sequenceLength: int,
    anomalyBlockLength: int,
    nBlocks: int,
    minimumGap: int,
    randomGenerator: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample sorted, non-overlapping block starts."""

    if nBlocks <= 0:
        raise ValueError(
            "nBlocks must be positive"
        )

    if minimumGap < 0:
        raise ValueError(
            "minimumGap must be nonnegative"
        )

    requiredLength = (
        nBlocks * anomalyBlockLength
        + (nBlocks - 1) * minimumGap
    )

    if requiredLength > sequenceLength:
        raise ValueError(
            "The requested anomaly blocks do not fit "
            "inside the sequence"
        )

    extraNominalSteps = (
        sequenceLength - requiredLength
    )

    # Sampling combinations with repetition through
    # an equivalent without-replacement transformation.
    transformedSlots = np.sort(
        randomGenerator.choice(
            extraNominalSteps + nBlocks,
            size=nBlocks,
            replace=False,
        )
    )

    compressedStarts = (
        transformedSlots
        - np.arange(
            nBlocks,
            dtype=int,
        )
    )

    starts = (
        compressedStarts
        + np.arange(
            nBlocks,
            dtype=int,
        )
        * (
            anomalyBlockLength
            + minimumGap
        )
    )

    return starts.astype(int)

def insertAnomalyBlocks(
    sequence: np.ndarray,
    randomGenerator: np.random.Generator,
    *,
    nBlocks: int = 1,
    anomalyBlockLength: int = 20,
    anomalyMean: float = 4.0,
    anomalyStd: float = 1.0,
    minimumGap: int = 1,
    starts: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert one or more separated shifted-mean blocks."""

    sequenceArray = np.asarray(
        sequence,
        dtype=float,
    ).copy()

    if sequenceArray.ndim != 1:
        raise ValueError(
            "sequence must be one-dimensional"
        )

    sequenceLength = sequenceArray.size

    if not (
        1
        <= anomalyBlockLength
        <= sequenceLength
    ):
        raise ValueError(
            "anomalyBlockLength must lie between "
            "1 and the sequence length"
        )

    if anomalyStd <= 0.0:
        raise ValueError(
            "anomalyStd must be positive"
        )

    if starts is None:
        blockStarts = _sampleSeparatedBlockStarts(
            sequenceLength=sequenceLength,
            anomalyBlockLength=anomalyBlockLength,
            nBlocks=nBlocks,
            minimumGap=minimumGap,
            randomGenerator=randomGenerator,
        )
    else:
        blockStarts = np.asarray(
            starts,
            dtype=int,
        ).reshape(-1)

        if blockStarts.size != nBlocks:
            raise ValueError(
                "The number of supplied starts must equal nBlocks"
            )

        blockStarts = np.sort(blockStarts)

        blockEnds = (
            blockStarts
            + anomalyBlockLength
        )

        if np.any(blockStarts < 0):
            raise ValueError(
                "Block starts must be nonnegative"
            )

        if np.any(blockEnds > sequenceLength):
            raise ValueError(
                "A supplied anomaly block exceeds "
                "the sequence length"
            )

        if nBlocks > 1:
            if np.any(
                blockStarts[1:]
                - blockEnds[:-1]
                < minimumGap
            ):
                raise ValueError(
                    "Supplied anomaly blocks overlap or "
                    "violate minimumGap"
                )

    blockEnds = (
        blockStarts
        + anomalyBlockLength
    )

    for blockStart, blockEnd in zip(
        blockStarts,
        blockEnds,
        strict=True,
    ):
        sequenceArray[
            blockStart:blockEnd
        ] = randomGenerator.normal(
            loc=anomalyMean,
            scale=anomalyStd,
            size=anomalyBlockLength,
        )

    return (
        sequenceArray,
        blockStarts,
        blockEnds,
    )


def insertAnomalyBlock(
    sequence: np.ndarray,
    randomGenerator: np.random.Generator,
    *,
    anomalyBlockLength: int = 20,
    anomalyMean: float = 4.0,
    anomalyStd: float = 1.0,
    start: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Backward-compatible single-block wrapper."""

    suppliedStarts = (
        None
        if start is None
        else np.array([start], dtype=int)
    )

    (
        modifiedSequence,
        blockStarts,
        blockEnds,
    ) = insertAnomalyBlocks(
        sequence,
        randomGenerator,
        nBlocks=1,
        anomalyBlockLength=anomalyBlockLength,
        anomalyMean=anomalyMean,
        anomalyStd=anomalyStd,
        minimumGap=0,
        starts=suppliedStarts,
    )

    return (
        modifiedSequence,
        int(blockStarts[0]),
        int(blockEnds[0]),
    )


def makeGaussianTrainingSet(
    *,
    nTrainNormal: int = 200,
    sequenceLength: int = 100,
    seed: int = BASE_SEED,
) -> np.ndarray:
    """Generate the required clean nominal training pool."""

    randomGenerator = np.random.default_rng(seed)

    return generateNominalSequences(
        nTrainNormal,
        sequenceLength,
        randomGenerator,
    )


def makeGaussianTestSet(
    *,
    nNormal: int,
    nAnomaly: int,
    sequenceLength: int = 100,
    anomalyBlockLength: int = 20,
    anomalyMean: float = 4.0,
    anomalyStd: float = 1.0,
    nBlocks: int = 1,
    minimumGap: int = 1,
    seed: int = Q9_SEED,
    shuffle: bool = True,
) -> GaussianTestSet:
    """Generate a Gaussian test set with sequence-level labels."""

    if nNormal <= 0:
        raise ValueError("nNormal must be positive")

    if nAnomaly <= 0:
        raise ValueError("nAnomaly must be positive")

    randomGenerator = np.random.default_rng(seed)

    normalSequences = generateNominalSequences(
        nNormal,
        sequenceLength,
        randomGenerator,
    )

    anomalousSequences = generateNominalSequences(
        nAnomaly,
        sequenceLength,
        randomGenerator,
    )

    anomalyStarts = np.full(
        (nAnomaly, nBlocks),
        -1,
        dtype=int,
    )

    anomalyEnds = np.full(
        (nAnomaly, nBlocks),
        -1,
        dtype=int,
    )

    for anomalyIndex in range(nAnomaly):
        (
            anomalousSequences[anomalyIndex],
            anomalyStarts[anomalyIndex],
            anomalyEnds[anomalyIndex],
        ) = insertAnomalyBlocks(
            anomalousSequences[anomalyIndex],
            randomGenerator,
            nBlocks=nBlocks,
            anomalyBlockLength=anomalyBlockLength,
            anomalyMean=anomalyMean,
            anomalyStd=anomalyStd,
            minimumGap=minimumGap,
        )

    sequences = np.concatenate(
        (normalSequences, anomalousSequences),
        axis=0,
    )

    labels = np.concatenate(
        (
            np.zeros(nNormal, dtype=int),
            np.ones(nAnomaly, dtype=int),
        )
    )

    fullAnomalyStarts = np.concatenate(
        (
            np.full(
                (nNormal, nBlocks),
                -1,
                dtype=int,
            ),
            anomalyStarts,
        ),
        axis=0,
    )

    fullAnomalyEnds = np.concatenate(
        (
            np.full(
                (nNormal, nBlocks),
                -1,
                dtype=int,
            ),
            anomalyEnds,
        ),
        axis=0,
    )

    fullBlockCounts = np.concatenate(
        (
            np.zeros(nNormal, dtype=int),
            np.full(
                nAnomaly,
                nBlocks,
                dtype=int,
            ),
        )
    )

    if shuffle:
        permutation = randomGenerator.permutation(
            sequences.shape[0]
        )

        sequences = sequences[permutation]
        labels = labels[permutation]
        fullAnomalyStarts = fullAnomalyStarts[
            permutation
        ]
        fullAnomalyEnds = fullAnomalyEnds[
            permutation
        ]
        fullBlockCounts = fullBlockCounts[
            permutation
        ]

    dataset = GaussianTestSet(
        sequences=sequences,
        labels=labels,
        anomalyStarts=fullAnomalyStarts,
        anomalyEnds=fullAnomalyEnds,
        nAnomalyBlocks=fullBlockCounts,
    )

    dataset.validate(
        sequenceLength=sequenceLength,
        anomalyBlockLength=anomalyBlockLength,
        minimumGap=minimumGap,
    )

    return dataset


def makeQuestion9Dataset(
    *,
    seed: int = Q9_SEED,
) -> GaussianTestSet:
    """Exact Question 9 dataset.

    Important:
        Q9 uses exactly one anomaly block per anomalous sequence.
    """

    return makeGaussianTestSet(
        nNormal=200,
        nAnomaly=50,
        sequenceLength=100,
        anomalyBlockLength=20,
        anomalyMean=4.0,
        anomalyStd=1.0,
        nBlocks=1,
        minimumGap=0,
        seed=seed,
        shuffle=False,
    )


def makeQuestion11TestSet(
    *,
    nNormal: int,
    nAnomaly: int,
    seed: int,
    shuffle: bool = True,
) -> GaussianTestSet:
    """Required Q11 single-block held-out test set."""

    return makeGaussianTestSet(
        nNormal=nNormal,
        nAnomaly=nAnomaly,
        sequenceLength=100,
        anomalyBlockLength=20,
        anomalyMean=4.0,
        anomalyStd=1.0,
        nBlocks=1,
        minimumGap=0,
        seed=seed,
        shuffle=shuffle,
    )


def makeSensitivityMasterPool(
    *,
    nNormalPool: int = 990,
    nAnomalyPool: int = 200,
    sequenceLength: int = 100,
    anomalyBlockLength: int = 20,
    nBlocks: int = 1,
    minimumGap: int = 1,
    seed: int = SENSITIVITY_POOL_SEED,
) -> GaussianSensitivityPool:
    """Generate one reusable master pool for all sensitivity ratios."""

    randomGenerator = np.random.default_rng(seed)

    normalSequences = generateNominalSequences(
        nNormalPool,
        sequenceLength,
        randomGenerator,
    )

    anomalousSequences = generateNominalSequences(
        nAnomalyPool,
        sequenceLength,
        randomGenerator,
    )

    anomalyStarts = np.full(
        (nAnomalyPool, nBlocks),
        -1,
        dtype=int,
    )

    anomalyEnds = np.full(
        (nAnomalyPool, nBlocks),
        -1,
        dtype=int,
    )

    for anomalyIndex in range(nAnomalyPool):
        (
            anomalousSequences[anomalyIndex],
            anomalyStarts[anomalyIndex],
            anomalyEnds[anomalyIndex],
        ) = insertAnomalyBlocks(
            anomalousSequences[anomalyIndex],
            randomGenerator,
            nBlocks=nBlocks,
            anomalyBlockLength=anomalyBlockLength,
            anomalyMean=4.0,
            anomalyStd=1.0,
            minimumGap=minimumGap,
        )

    normalOrder = randomGenerator.permutation(
        nNormalPool
    )

    anomalyOrder = randomGenerator.permutation(
        nAnomalyPool
    )

    pool = GaussianSensitivityPool(
        normalSequences=normalSequences[
            normalOrder
        ],
        anomalousSequences=anomalousSequences[
            anomalyOrder
        ],
        anomalyStarts=anomalyStarts[
            anomalyOrder
        ],
        anomalyEnds=anomalyEnds[
            anomalyOrder
        ],
        nAnomalyBlocks=np.full(
            nAnomalyPool,
            nBlocks,
            dtype=int,
        ),
    )

    pool.validate(
        sequenceLength=sequenceLength,
        anomalyBlockLength=anomalyBlockLength,
        minimumGap=minimumGap,
    )

    return pool


def makeExactFractionSubsetFromPool(
    pool: GaussianSensitivityPool,
    anomalyFraction: float,
    *,
    nTotal: int = 1000,
    shuffleSeed: int = 1500,
) -> GaussianTestSet:
    """Construct an exact-ratio subset from one shared master pool."""

    if not 0.0 < anomalyFraction < 1.0:
        raise ValueError(
            "anomalyFraction must lie in (0, 1)"
        )

    exactAnomalyCount = (
        nTotal * anomalyFraction
    )

    nAnomaly = int(round(exactAnomalyCount))

    if not np.isclose(
        exactAnomalyCount,
        nAnomaly,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "nTotal does not permit the requested "
            "anomaly fraction exactly"
        )

    nNormal = nTotal - nAnomaly

    if nNormal > pool.normalSequences.shape[0]:
        raise ValueError(
            "The master pool does not contain enough normal sequences"
        )

    if nAnomaly > pool.anomalousSequences.shape[0]:
        raise ValueError(
            "The master pool does not contain enough anomalies"
        )

    selectedNormalSequences = (
        pool.normalSequences[:nNormal]
    )

    selectedAnomalySequences = (
        pool.anomalousSequences[:nAnomaly]
    )

    maxBlocks = pool.anomalyStarts.shape[1]

    sequences = np.concatenate(
        (
            selectedNormalSequences,
            selectedAnomalySequences,
        ),
        axis=0,
    )

    labels = np.concatenate(
        (
            np.zeros(nNormal, dtype=int),
            np.ones(nAnomaly, dtype=int),
        )
    )

    starts = np.concatenate(
        (
            np.full(
                (nNormal, maxBlocks),
                -1,
                dtype=int,
            ),
            pool.anomalyStarts[:nAnomaly],
        ),
        axis=0,
    )

    ends = np.concatenate(
        (
            np.full(
                (nNormal, maxBlocks),
                -1,
                dtype=int,
            ),
            pool.anomalyEnds[:nAnomaly],
        ),
        axis=0,
    )

    blockCounts = np.concatenate(
        (
            np.zeros(nNormal, dtype=int),
            pool.nAnomalyBlocks[:nAnomaly],
        )
    )

    randomGenerator = np.random.default_rng(
        shuffleSeed
    )

    permutation = randomGenerator.permutation(
        nTotal
    )

    dataset = GaussianTestSet(
        sequences=sequences[permutation],
        labels=labels[permutation],
        anomalyStarts=starts[permutation],
        anomalyEnds=ends[permutation],
        nAnomalyBlocks=blockCounts[
            permutation
        ],
    )

    # The required sensitivity experiment uses one 20-step block.
    dataset.validate(
        sequenceLength=100,
        anomalyBlockLength=20,
        minimumGap=0,
    )

    return dataset


def makeExactFractionTestSet(
    anomalyFraction: float,
    *,
    nTotal: int = 1000,
    seed: int = SENSITIVITY_POOL_SEED,
) -> GaussianTestSet:
    """Backward-compatible exact-fraction constructor.

    The final runner should prefer:
        makeSensitivityMasterPool
        makeExactFractionSubsetFromPool
    """

    pool = makeSensitivityMasterPool(
        nNormalPool=990,
        nAnomalyPool=200,
        nBlocks=1,
        minimumGap=0,
        seed=seed,
    )

    return makeExactFractionSubsetFromPool(
        pool,
        anomalyFraction,
        nTotal=nTotal,
        shuffleSeed=seed + 100,
    )


def saveGaussianTestSet(
    dataset: GaussianTestSet,
    outputDirectory: str | Path,
    *,
    prefix: str,
) -> dict[str, object]:
    """Serialize one generated dataset and its metadata."""

    outputPath = Path(outputDirectory)

    outputPath.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        outputPath / f"{prefix}_sequences.npy",
        dataset.sequences,
    )

    np.save(
        outputPath / f"{prefix}_labels.npy",
        dataset.labels,
    )

    np.save(
        outputPath / f"{prefix}_anomaly_starts.npy",
        dataset.anomalyStarts,
    )

    np.save(
        outputPath / f"{prefix}_anomaly_ends.npy",
        dataset.anomalyEnds,
    )

    np.save(
        outputPath / f"{prefix}_n_blocks.npy",
        dataset.nAnomalyBlocks,
    )

    metadata = {
        "prefix": prefix,
        "nSequences": int(
            dataset.sequences.shape[0]
        ),
        "sequenceLength": int(
            dataset.sequences.shape[1]
        ),
        "nNormal": int(
            np.sum(dataset.labels == 0)
        ),
        "nAnomaly": int(
            np.sum(dataset.labels == 1)
        ),
        "anomalyFraction": float(
            np.mean(dataset.labels)
        ),
        "maxBlocks": int(
            dataset.maxBlocks
        ),
        "blockCountDistribution": {
            str(int(blockCount)): int(count)
            for blockCount, count in zip(
                *np.unique(
                    dataset.nAnomalyBlocks,
                    return_counts=True,
                ),
                strict=True,
            )
        },
        "intervalConvention": {
            "start": "inclusive",
            "end": "exclusive",
            "unused": -1,
        },
    }

    metadataPath = (
        outputPath
        / f"{prefix}_metadata.json"
    )

    metadataPath.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return metadata