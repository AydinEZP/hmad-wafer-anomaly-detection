from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class TestSet:
    sequences: np.ndarray
    labels: np.ndarray
    anomalyStarts: np.ndarray
    anomalyEnds: np.ndarray


def logSpaceViterbi(
    sequence: np.ndarray,
    initialProbabilities: np.ndarray,
    transitionMatrix: np.ndarray,
    emissionMeans: np.ndarray,
    emissionVariances: np.ndarray,
) -> np.ndarray:
    sequenceArray = np.asarray(sequence, dtype=float).reshape(-1)
    initialArray = np.asarray(initialProbabilities, dtype=float).reshape(-1)
    transitionArray = np.asarray(transitionMatrix, dtype=float)
    meanArray = np.asarray(emissionMeans, dtype=float).reshape(-1)
    varianceArray = np.asarray(emissionVariances, dtype=float).reshape(-1)
    nStates = meanArray.size
    sequenceLength = sequenceArray.size

    logInitial = np.log(np.clip(initialArray, 1.0e-300, None))
    logTransition = np.log(np.clip(transitionArray, 1.0e-300, None))
    logEmissions = -0.5 * (
        np.log(2.0 * np.pi * varianceArray)[None, :]
        + (sequenceArray[:, None] - meanArray[None, :]) ** 2 / varianceArray[None, :]
    )

    bestScores = np.empty((sequenceLength, nStates), dtype=float)
    backPointers = np.zeros((sequenceLength, nStates), dtype=int)
    bestScores[0] = logInitial + logEmissions[0]

    for timeIndex in range(1, sequenceLength):
        candidateScores = bestScores[timeIndex - 1][:, None] + logTransition
        backPointers[timeIndex] = np.argmax(candidateScores, axis=0)
        bestScores[timeIndex] = (
            candidateScores[backPointers[timeIndex], np.arange(nStates)]
            + logEmissions[timeIndex]
        )

    statePath = np.empty(sequenceLength, dtype=int)
    statePath[-1] = int(np.argmax(bestScores[-1]))
    for timeIndex in range(sequenceLength - 2, -1, -1):
        statePath[timeIndex] = backPointers[timeIndex + 1, statePath[timeIndex + 1]]
    return statePath


def generateNominalSequences(
    nSequences: int,
    sequenceLength: int,
    randomGenerator: np.random.Generator,
) -> np.ndarray:
    return randomGenerator.normal(0.0, 1.0, size=(nSequences, sequenceLength))


def insertAnomalyBlock(
    sequence: np.ndarray,
    blockLength: int,
    anomalyMean: float,
    anomalyStd: float,
    randomGenerator: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    sequenceArray = np.asarray(sequence, dtype=float).copy()
    startIndex = int(randomGenerator.integers(0, sequenceArray.size - blockLength + 1))
    endIndex = startIndex + blockLength
    sequenceArray[startIndex:endIndex] = randomGenerator.normal(
        anomalyMean, anomalyStd, size=blockLength
    )
    return sequenceArray, startIndex, endIndex


def makeTrainSet(
    nTrain: int,
    sequenceLength: int,
    seed: int,
) -> np.ndarray:
    randomGenerator = np.random.default_rng(seed)
    return generateNominalSequences(nTrain, sequenceLength, randomGenerator)


def makeEvaluationPools(
    nNormal: int,
    nAnomaly: int,
    sequenceLength: int,
    blockLength: int,
    anomalyMean: float,
    anomalyStd: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    randomGenerator = np.random.default_rng(seed)
    normalSequences = generateNominalSequences(nNormal, sequenceLength, randomGenerator)
    anomalousSequences = generateNominalSequences(nAnomaly, sequenceLength, randomGenerator)
    anomalyStarts = np.empty(nAnomaly, dtype=int)
    anomalyEnds = np.empty(nAnomaly, dtype=int)
    for sequenceIndex in range(nAnomaly):
        anomalousSequences[sequenceIndex], anomalyStarts[sequenceIndex], anomalyEnds[sequenceIndex] = insertAnomalyBlock(
            anomalousSequences[sequenceIndex],
            blockLength,
            anomalyMean,
            anomalyStd,
            randomGenerator,
        )
    return normalSequences, anomalousSequences, anomalyStarts, anomalyEnds


def makeTestSetFromPools(
    normalSequences: np.ndarray,
    anomalousSequences: np.ndarray,
    anomalyStarts: np.ndarray,
    anomalyEnds: np.ndarray,
    anomalyFraction: float,
    seed: int,
) -> TestSet:
    nNormal = normalSequences.shape[0]
    nAnomaly = int(round(anomalyFraction / (1.0 - anomalyFraction) * nNormal))
    selectedAnomalies = anomalousSequences[:nAnomaly]
    selectedStarts = anomalyStarts[:nAnomaly]
    selectedEnds = anomalyEnds[:nAnomaly]
    sequences = np.vstack((normalSequences, selectedAnomalies))
    labels = np.concatenate((np.zeros(nNormal, dtype=int), np.ones(nAnomaly, dtype=int)))
    starts = np.concatenate((np.full(nNormal, -1, dtype=int), selectedStarts))
    ends = np.concatenate((np.full(nNormal, -1, dtype=int), selectedEnds))
    randomGenerator = np.random.default_rng(seed)
    order = randomGenerator.permutation(labels.size)
    return TestSet(sequences[order], labels[order], starts[order], ends[order])


def globalSummaryFeatures(sequences: np.ndarray) -> np.ndarray:
    sequenceArray = np.asarray(sequences, dtype=float)
    return np.column_stack(
        (
            sequenceArray.mean(axis=1),
            sequenceArray.std(axis=1),
            sequenceArray.min(axis=1),
            sequenceArray.max(axis=1),
            np.median(sequenceArray, axis=1),
        )
    )


def fitStandaloneBaseline(
    trainSequences: np.ndarray,
    nu: float,
    gamma: str | float,
) -> tuple[StandardScaler, OneClassSVM]:
    trainFeatures = globalSummaryFeatures(trainSequences)
    scaler = StandardScaler().fit(trainFeatures)
    model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
    model.fit(scaler.transform(trainFeatures))
    return scaler, model


def scoreStandaloneBaseline(
    sequences: np.ndarray,
    scaler: StandardScaler,
    model: OneClassSVM,
) -> np.ndarray:
    features = globalSummaryFeatures(sequences)
    return model.decision_function(scaler.transform(features)).reshape(-1)


def evaluateNormalityScores(labels: np.ndarray, normalityScores: np.ndarray) -> dict[str, float]:
    labelArray = np.asarray(labels, dtype=int).reshape(-1)
    scoreArray = np.asarray(normalityScores, dtype=float).reshape(-1)
    anomalyScores = -scoreArray
    predictions = (scoreArray < 0.0).astype(int)
    precision, recall, f1Score, _ = precision_recall_fscore_support(
        labelArray,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "rocAuc": float(roc_auc_score(labelArray, anomalyScores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1Score": float(f1Score),
    }
