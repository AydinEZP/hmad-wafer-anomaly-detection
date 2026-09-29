from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructuredSequenceDataset:
    """Fixed-length sequence dataset used by the integrated HMAD project.

    Labels are sequence-level: 0 denotes nominal and 1 denotes anomalous.
    Hidden state paths are stored only for dataset validation and qualitative
    analysis. They are never supplied to OC-SVM or HMAD during training or
    scoring.
    """

    trainSequences: np.ndarray
    trainStates: np.ndarray
    testSequences: np.ndarray
    testStates: np.ndarray
    testLabels: np.ndarray
    anomalyStarts: np.ndarray
    anomalyEnds: np.ndarray
    anomalyTypes: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        trainSequences = np.asarray(self.trainSequences, dtype=float)
        trainStates = np.asarray(self.trainStates, dtype=int)
        testSequences = np.asarray(self.testSequences, dtype=float)
        testStates = np.asarray(self.testStates, dtype=int)
        testLabels = np.asarray(self.testLabels, dtype=int).reshape(-1)
        anomalyStarts = np.asarray(self.anomalyStarts, dtype=int).reshape(-1)
        anomalyEnds = np.asarray(self.anomalyEnds, dtype=int).reshape(-1)
        anomalyTypes = np.asarray(self.anomalyTypes).reshape(-1)

        if trainSequences.ndim != 2 or testSequences.ndim != 2:
            raise ValueError("trainSequences and testSequences must be two-dimensional")
        if trainSequences.shape != trainStates.shape:
            raise ValueError("train sequence and hidden-state arrays must have equal shape")
        if testSequences.shape != testStates.shape:
            raise ValueError("test sequence and hidden-state arrays must have equal shape")
        if testSequences.shape[0] != testLabels.size:
            raise ValueError("one sequence-level label is required for each test sequence")
        if not (
            testLabels.size
            == anomalyStarts.size
            == anomalyEnds.size
            == anomalyTypes.size
        ):
            raise ValueError("test metadata arrays must have equal length")
        if not np.all(np.isfinite(trainSequences)) or not np.all(np.isfinite(testSequences)):
            raise ValueError("dataset contains non-finite observations")
        if not np.all(np.isin(testLabels, [0, 1])):
            raise ValueError("testLabels must use 0 for nominal and 1 for anomaly")
        if np.any((trainStates < 0) | (trainStates > 1)) or np.any(
            (testStates < 0) | (testStates > 1)
        ):
            raise ValueError("hidden states must use zero-based labels {0, 1}")

        nominalMask = testLabels == 0
        anomalyMask = testLabels == 1
        if np.any(anomalyStarts[nominalMask] != -1) or np.any(anomalyEnds[nominalMask] != -1):
            raise ValueError("nominal test sequences must use -1 anomaly boundaries")
        if np.any(anomalyStarts[anomalyMask] < 0):
            raise ValueError("anomalous sequences require a non-negative anomaly start")
        if np.any(anomalyEnds[anomalyMask] <= anomalyStarts[anomalyMask]):
            raise ValueError("anomaly end must be greater than anomaly start")
        if np.any(anomalyEnds[anomalyMask] > testSequences.shape[1]):
            raise ValueError("anomaly intervals must lie inside the sequence")


def _sample_normal_sequence(
    sequenceLength: int,
    initialProbabilities: np.ndarray,
    transitionMatrix: np.ndarray,
    emissionMeans: np.ndarray,
    emissionStandardDeviations: np.ndarray,
    randomGenerator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.empty(sequenceLength, dtype=np.int8)
    observations = np.empty(sequenceLength, dtype=np.float64)
    states[0] = randomGenerator.choice(2, p=initialProbabilities)
    observations[0] = randomGenerator.normal(
        emissionMeans[states[0]], emissionStandardDeviations[states[0]]
    )
    for timeIndex in range(1, sequenceLength):
        states[timeIndex] = randomGenerator.choice(
            2, p=transitionMatrix[states[timeIndex - 1]]
        )
        observations[timeIndex] = randomGenerator.normal(
            emissionMeans[states[timeIndex]],
            emissionStandardDeviations[states[timeIndex]],
        )
    return observations, states


def _inject_rapid_switch_anomaly(
    observations: np.ndarray,
    states: np.ndarray,
    blockLength: int,
    emissionMeans: np.ndarray,
    emissionStandardDeviations: np.ndarray,
    randomGenerator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    anomalousObservations = np.asarray(observations, dtype=float).copy()
    anomalousStates = np.asarray(states, dtype=np.int8).copy()
    sequenceLength = anomalousObservations.size
    startIndex = int(randomGenerator.integers(15, sequenceLength - blockLength - 4))
    endIndex = startIndex + blockLength

    # The values remain valid emissions from the same two nominal regimes. The
    # anomaly is the implausibly frequent alternation between regimes.
    firstState = int(randomGenerator.integers(0, 2))
    anomalousStates[startIndex:endIndex] = (
        firstState + np.arange(blockLength, dtype=int)
    ) % 2
    anomalousObservations[startIndex:endIndex] = randomGenerator.normal(
        emissionMeans[anomalousStates[startIndex:endIndex]],
        emissionStandardDeviations[anomalousStates[startIndex:endIndex]],
    )
    return anomalousObservations, anomalousStates, startIndex, endIndex


def _inject_block_shuffle_anomaly(
    observations: np.ndarray,
    states: np.ndarray,
    blockLength: int,
    randomGenerator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    anomalousObservations = np.asarray(observations, dtype=float).copy()
    anomalousStates = np.asarray(states, dtype=np.int8).copy()
    sequenceLength = anomalousObservations.size
    startIndex = int(randomGenerator.integers(10, sequenceLength - blockLength - 9))
    endIndex = startIndex + blockLength
    permutation = randomGenerator.permutation(blockLength)

    # This anomaly preserves the exact multiset of observations and therefore
    # preserves every global order-invariant statistic. Only local ordering and
    # transition structure are changed.
    anomalousObservations[startIndex:endIndex] = anomalousObservations[
        startIndex:endIndex
    ][permutation]
    anomalousStates[startIndex:endIndex] = anomalousStates[startIndex:endIndex][
        permutation
    ]
    return anomalousObservations, anomalousStates, startIndex, endIndex


def createStructuredSequenceDataset(
    *,
    seed: int = 1405,
    nTrainNormal: int = 200,
    nTestNormal: int = 160,
    nRapidSwitchAnomalies: int = 20,
    nBlockShuffleAnomalies: int = 20,
    sequenceLength: int = 100,
    rapidSwitchBlockLength: int = 24,
    shuffleBlockLength: int = 30,
) -> StructuredSequenceDataset:
    """Create the reproducible Structured Regime-Switching Sequence Dataset.

    The nominal process is a persistent two-state Gaussian HMM. Anomalies use
    exactly the same emission distributions as the nominal process and are
    defined by locally implausible state transitions or order permutations.
    Consequently, individual values remain plausible and the anomaly is not a
    trivial extreme-value detection problem.
    """

    if nTrainNormal < 100:
        raise ValueError("nTrainNormal must be at least 100")
    if nRapidSwitchAnomalies + nBlockShuffleAnomalies < 20:
        raise ValueError("the test set must include at least 20 anomalies")
    if sequenceLength < max(rapidSwitchBlockLength + 25, shuffleBlockLength + 20):
        raise ValueError("sequenceLength is too short for the requested anomaly blocks")

    randomGenerator = np.random.default_rng(seed)
    initialProbabilities = np.array([0.55, 0.45], dtype=float)
    transitionMatrix = np.array([[0.97, 0.03], [0.04, 0.96]], dtype=float)
    emissionMeans = np.array([-1.20, 1.20], dtype=float)
    emissionStandardDeviations = np.array([0.65, 0.65], dtype=float)

    trainObservations: list[np.ndarray] = []
    trainStates: list[np.ndarray] = []
    for _ in range(nTrainNormal):
        observations, states = _sample_normal_sequence(
            sequenceLength,
            initialProbabilities,
            transitionMatrix,
            emissionMeans,
            emissionStandardDeviations,
            randomGenerator,
        )
        trainObservations.append(observations)
        trainStates.append(states)

    testObservations: list[np.ndarray] = []
    testStates: list[np.ndarray] = []
    testLabels: list[int] = []
    anomalyStarts: list[int] = []
    anomalyEnds: list[int] = []
    anomalyTypes: list[str] = []

    for _ in range(nTestNormal):
        observations, states = _sample_normal_sequence(
            sequenceLength,
            initialProbabilities,
            transitionMatrix,
            emissionMeans,
            emissionStandardDeviations,
            randomGenerator,
        )
        testObservations.append(observations)
        testStates.append(states)
        testLabels.append(0)
        anomalyStarts.append(-1)
        anomalyEnds.append(-1)
        anomalyTypes.append("nominal")

    for _ in range(nRapidSwitchAnomalies):
        observations, states = _sample_normal_sequence(
            sequenceLength,
            initialProbabilities,
            transitionMatrix,
            emissionMeans,
            emissionStandardDeviations,
            randomGenerator,
        )
        observations, states, startIndex, endIndex = _inject_rapid_switch_anomaly(
            observations,
            states,
            rapidSwitchBlockLength,
            emissionMeans,
            emissionStandardDeviations,
            randomGenerator,
        )
        testObservations.append(observations)
        testStates.append(states)
        testLabels.append(1)
        anomalyStarts.append(startIndex)
        anomalyEnds.append(endIndex)
        anomalyTypes.append("rapid_switch")

    for _ in range(nBlockShuffleAnomalies):
        observations, states = _sample_normal_sequence(
            sequenceLength,
            initialProbabilities,
            transitionMatrix,
            emissionMeans,
            emissionStandardDeviations,
            randomGenerator,
        )
        observations, states, startIndex, endIndex = _inject_block_shuffle_anomaly(
            observations,
            states,
            shuffleBlockLength,
            randomGenerator,
        )
        testObservations.append(observations)
        testStates.append(states)
        testLabels.append(1)
        anomalyStarts.append(startIndex)
        anomalyEnds.append(endIndex)
        anomalyTypes.append("block_shuffle")

    testObservationsArray = np.vstack(testObservations)
    testStatesArray = np.vstack(testStates)
    testLabelsArray = np.asarray(testLabels, dtype=np.int8)
    anomalyStartsArray = np.asarray(anomalyStarts, dtype=np.int16)
    anomalyEndsArray = np.asarray(anomalyEnds, dtype=np.int16)
    anomalyTypesArray = np.asarray(anomalyTypes, dtype="U24")

    # Shuffle once before any model is trained. The fixed permutation is part of
    # the dataset definition and is reproducible from the public seed.
    order = randomGenerator.permutation(testLabelsArray.size)
    testObservationsArray = testObservationsArray[order]
    testStatesArray = testStatesArray[order]
    testLabelsArray = testLabelsArray[order]
    anomalyStartsArray = anomalyStartsArray[order]
    anomalyEndsArray = anomalyEndsArray[order]
    anomalyTypesArray = anomalyTypesArray[order]

    metadata: dict[str, Any] = {
        "datasetName": "Structured Regime-Switching Sequence Dataset",
        "datasetVersion": "1.0",
        "source": "Programmatically generated by the project team",
        "randomSeed": seed,
        "sequenceDefinition": "One independent length-100 trajectory from a persistent two-state Gaussian HMM",
        "labelDefinition": "0 = nominal sequence, 1 = sequence containing a temporal-structure anomaly",
        "nTrainNormal": nTrainNormal,
        "nTestNormal": nTestNormal,
        "nTestAnomaly": nRapidSwitchAnomalies + nBlockShuffleAnomalies,
        "sequenceLength": sequenceLength,
        "totalTrainingTimeSteps": nTrainNormal * sequenceLength,
        "totalTestTimeSteps": (
            nTestNormal + nRapidSwitchAnomalies + nBlockShuffleAnomalies
        )
        * sequenceLength,
        "initialProbabilities": initialProbabilities.tolist(),
        "nominalTransitionMatrix": transitionMatrix.tolist(),
        "emissionMeans": emissionMeans.tolist(),
        "emissionStandardDeviations": emissionStandardDeviations.tolist(),
        "anomalyTypes": {
            "rapid_switch": {
                "count": nRapidSwitchAnomalies,
                "blockLength": rapidSwitchBlockLength,
                "description": "A local block alternates hidden regimes almost every time step while using nominal state-conditional emissions.",
            },
            "block_shuffle": {
                "count": nBlockShuffleAnomalies,
                "blockLength": shuffleBlockLength,
                "description": "A local block is randomly permuted, preserving its exact observations and all global order-invariant statistics while disrupting temporal order.",
            },
        },
        "antiLeakagePolicy": "The train/test split, anomaly intervals, anomaly types, and random seed are fixed before model fitting. Hidden state paths are not used as model inputs.",
    }

    dataset = StructuredSequenceDataset(
        trainSequences=np.vstack(trainObservations),
        trainStates=np.vstack(trainStates),
        testSequences=testObservationsArray,
        testStates=testStatesArray,
        testLabels=testLabelsArray,
        anomalyStarts=anomalyStartsArray,
        anomalyEnds=anomalyEndsArray,
        anomalyTypes=anomalyTypesArray,
        metadata=metadata,
    )
    dataset.validate()
    return dataset


def saveStructuredSequenceDataset(
    dataset: StructuredSequenceDataset,
    outputDirectory: str | Path,
) -> tuple[Path, Path, Path]:
    dataset.validate()
    outputPath = Path(outputDirectory)
    outputPath.mkdir(parents=True, exist_ok=True)
    datasetPath = outputPath / "structured_regime_sequence_dataset.npz"
    metadataPath = outputPath / "structured_regime_sequence_dataset_metadata.json"
    manifestPath = outputPath / "structured_regime_sequence_manifest.csv"

    np.savez_compressed(
        datasetPath,
        trainSequences=dataset.trainSequences,
        trainStates=dataset.trainStates,
        testSequences=dataset.testSequences,
        testStates=dataset.testStates,
        testLabels=dataset.testLabels,
        anomalyStarts=dataset.anomalyStarts,
        anomalyEnds=dataset.anomalyEnds,
        anomalyTypes=dataset.anomalyTypes,
    )
    metadataPath.write_text(
        json.dumps(dataset.metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifestRows: list[dict[str, Any]] = []
    for sequenceIndex in range(dataset.trainSequences.shape[0]):
        manifestRows.append(
            {
                "sequenceId": f"train_{sequenceIndex:04d}",
                "split": "train",
                "length": int(dataset.trainSequences.shape[1]),
                "label": 0,
                "anomalyType": "nominal",
                "anomalyStart": -1,
                "anomalyEnd": -1,
            }
        )
    for sequenceIndex in range(dataset.testSequences.shape[0]):
        manifestRows.append(
            {
                "sequenceId": f"test_{sequenceIndex:04d}",
                "split": "test",
                "length": int(dataset.testSequences.shape[1]),
                "label": int(dataset.testLabels[sequenceIndex]),
                "anomalyType": str(dataset.anomalyTypes[sequenceIndex]),
                "anomalyStart": int(dataset.anomalyStarts[sequenceIndex]),
                "anomalyEnd": int(dataset.anomalyEnds[sequenceIndex]),
            }
        )
    pd.DataFrame(manifestRows).to_csv(manifestPath, index=False)
    return datasetPath, metadataPath, manifestPath


def loadStructuredSequenceDataset(
    datasetPath: str | Path,
    metadataPath: str | Path | None = None,
) -> StructuredSequenceDataset:
    datasetFile = Path(datasetPath)
    if metadataPath is None:
        metadataFile = datasetFile.with_name(
            "structured_regime_sequence_dataset_metadata.json"
        )
    else:
        metadataFile = Path(metadataPath)
    with np.load(datasetFile, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(metadataFile.read_text(encoding="utf-8"))
    dataset = StructuredSequenceDataset(metadata=metadata, **arrays)
    dataset.validate()
    return dataset


def selectFixedTestSubset(
    dataset: StructuredSequenceDataset,
    nNormal: int,
    *,
    seed: int = 1405,
) -> np.ndarray:
    """Return fixed test indices with all anomalies and a seeded normal subset.

    Every subset contains the complete, pre-declared anomaly pool (40 anomaly
    sequences in the default dataset), so sensitivity analysis never violates
    the minimum-anomaly requirement or cherry-picks easy anomalies.
    """

    dataset.validate()
    normalIndices = np.flatnonzero(dataset.testLabels == 0)
    anomalyIndices = np.flatnonzero(dataset.testLabels == 1)
    if not 1 <= nNormal <= normalIndices.size:
        raise ValueError("nNormal must be between 1 and the available normal count")
    randomGenerator = np.random.default_rng(seed)
    selectedNormal = randomGenerator.permutation(normalIndices)[:nNormal]
    selected = np.concatenate((selectedNormal, anomalyIndices))
    return randomGenerator.permutation(selected)


# Snake-case aliases for callers that prefer PEP 8 naming.
create_structured_sequence_dataset = createStructuredSequenceDataset
save_structured_sequence_dataset = saveStructuredSequenceDataset
load_structured_sequence_dataset = loadStructuredSequenceDataset
select_fixed_test_subset = selectFixedTestSubset
