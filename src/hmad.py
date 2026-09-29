from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .joint_features import batchJointFeatures, normalizeStateLabels
from .adapters import viterbi_path_adapter


ViterbiDecoder = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]


class SimplifiedHMAD:
    def __init__(
        self,
        nStates: int = 2,
        nu: float = 0.1,
        gamma: str | float = "scale",
        maxIterations: int = 10,
        convergenceThreshold: float = 0.01,
        initialSelfTransition: float = 0.9,
        varianceFloor: float = 1.0e-6,
        transitionSmoothing: float = 1.0e-3,
        randomState: int = 1405,
        viterbiDecoder: ViterbiDecoder | None = None,
    ) -> None:
        if nStates < 2:
            raise ValueError("nStates must be at least 2")
        if not 0.0 < nu <= 1.0:
            raise ValueError("nu must be in (0, 1]")
        if maxIterations < 1:
            raise ValueError("maxIterations must be positive")
        if not 0.0 < initialSelfTransition < 1.0:
            raise ValueError("initialSelfTransition must be in (0, 1)")
        self.nStates = nStates
        self.nu = nu
        self.gamma = gamma
        self.maxIterations = maxIterations
        self.convergenceThreshold = convergenceThreshold
        self.initialSelfTransition = initialSelfTransition
        self.varianceFloor = varianceFloor
        self.transitionSmoothing = transitionSmoothing
        self.randomState = randomState
        self.viterbiDecoder = viterbiDecoder or viterbi_path_adapter

    def fit(self, sequences: np.ndarray) -> "SimplifiedHMAD":
        sequenceArray = self._validateSequences(sequences)
        previousStates = self._initializeParameters(sequenceArray)
        convergenceHistory: list[dict[str, float]] = []

        for iterationIndex in range(1, self.maxIterations + 1):
            decodedStates = self._decodeBatch(sequenceArray)
            updatedParameters = self._estimateParameters(sequenceArray, decodedStates)
            decodedStates = self._applyStateOrder(decodedStates, updatedParameters)
            jointFeatures = batchJointFeatures(sequenceArray, decodedStates, self.nStates)
            self.featureScaler_ = StandardScaler().fit(jointFeatures)
            scaledFeatures = self.featureScaler_.transform(jointFeatures)
            self.oneClassModel_ = OneClassSVM(kernel="rbf", gamma=self.gamma, nu=self.nu)
            self.oneClassModel_.fit(scaledFeatures)
            changedFraction = float(np.mean(decodedStates != previousStates))
            meanDecisionScore = float(
                np.mean(self.oneClassModel_.decision_function(scaledFeatures))
            )
            convergenceHistory.append(
                {
                    "iteration": float(iterationIndex),
                    "changedFraction": changedFraction,
                    "meanDecisionScore": meanDecisionScore,
                }
            )
            previousStates = decodedStates
            if changedFraction < self.convergenceThreshold:
                break

        self.trainingStates_ = previousStates
        self.trainingFeatures_ = batchJointFeatures(
            sequenceArray, self.trainingStates_, self.nStates
        )
        self.convergenceHistory_ = convergenceHistory
        self.nIterations_ = len(convergenceHistory)
        self.converged_ = convergenceHistory[-1]["changedFraction"] < self.convergenceThreshold
        self.isFitted_ = True
        return self

    def decodeSequence(self, sequence: np.ndarray) -> np.ndarray:
        self._checkFitted()
        return self._decodeOne(np.asarray(sequence, dtype=float).reshape(-1))

    def decodeSequences(self, sequences: np.ndarray) -> np.ndarray:
        self._checkFitted()
        return self._decodeBatch(self._validateSequences(sequences))

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        self._checkFitted()
        sequenceArray = self._validateSequences(sequences)
        stateSequences = self._decodeBatch(sequenceArray)
        return batchJointFeatures(sequenceArray, stateSequences, self.nStates)

    def decisionFunction(self, sequences: np.ndarray) -> np.ndarray:
        jointFeatures = self.transform(sequences)
        scaledFeatures = self.featureScaler_.transform(jointFeatures)
        return self.oneClassModel_.decision_function(scaledFeatures).reshape(-1)

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        return np.where(self.decisionFunction(sequences) >= 0.0, 1, -1)

    def decisionFunctionWithTrivialStates(
        self,
        sequences: np.ndarray,
        stateIndex: int = 0,
    ) -> np.ndarray:
        self._checkFitted()
        sequenceArray = self._validateSequences(sequences)
        if not 0 <= stateIndex < self.nStates:
            raise ValueError("stateIndex is outside the valid range")
        stateSequences = np.full(sequenceArray.shape, stateIndex, dtype=int)
        jointFeatures = batchJointFeatures(sequenceArray, stateSequences, self.nStates)
        scaledFeatures = self.featureScaler_.transform(jointFeatures)
        return self.oneClassModel_.decision_function(scaledFeatures).reshape(-1)

    def _initializeParameters(self, sequences: np.ndarray) -> np.ndarray:
        pooledObservations = sequences.reshape(-1, 1)
        kMeansModel = KMeans(
            n_clusters=self.nStates,
            n_init=10,
            random_state=self.randomState,
        ).fit(pooledObservations)
        originalMeans = kMeansModel.cluster_centers_.reshape(-1)
        stateOrder = np.argsort(originalMeans)
        oldToNew = np.empty(self.nStates, dtype=int)
        oldToNew[stateOrder] = np.arange(self.nStates)
        initialStates = oldToNew[kMeansModel.labels_].reshape(sequences.shape)
        self.emissionMeans_ = originalMeans[stateOrder]
        self.emissionVariances_ = np.array(
            [
                max(
                    float(np.var(sequences[initialStates == stateIndex])),
                    self.varianceFloor,
                )
                for stateIndex in range(self.nStates)
            ]
        )
        self.initialProbabilities_ = np.full(self.nStates, 1.0 / self.nStates)
        offDiagonalProbability = (1.0 - self.initialSelfTransition) / (self.nStates - 1)
        self.transitionMatrix_ = np.full(
            (self.nStates, self.nStates), offDiagonalProbability, dtype=float
        )
        np.fill_diagonal(self.transitionMatrix_, self.initialSelfTransition)
        return initialStates

    def _decodeOne(self, sequence: np.ndarray) -> np.ndarray:
        decoderOutput = self.viterbiDecoder(
            sequence,
            self.initialProbabilities_,
            self.transitionMatrix_,
            self.emissionMeans_,
            self.emissionVariances_,
        )
        if isinstance(decoderOutput, tuple):
            decoderOutput = decoderOutput[0]
        return normalizeStateLabels(np.asarray(decoderOutput), self.nStates)

    def _decodeBatch(self, sequences: np.ndarray) -> np.ndarray:
        return np.vstack([self._decodeOne(sequence) for sequence in sequences])

    def _estimateParameters(
        self,
        sequences: np.ndarray,
        stateSequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        emissionMeans = self.emissionMeans_.copy()
        emissionVariances = self.emissionVariances_.copy()
        for stateIndex in range(self.nStates):
            stateValues = sequences[stateSequences == stateIndex]
            if stateValues.size > 0:
                emissionMeans[stateIndex] = float(np.mean(stateValues))
                emissionVariances[stateIndex] = max(
                    float(np.var(stateValues)), self.varianceFloor
                )

        transitionCounts = np.full(
            (self.nStates, self.nStates), self.transitionSmoothing, dtype=float
        )
        initialCounts = np.full(self.nStates, self.transitionSmoothing, dtype=float)
        for states in stateSequences:
            initialCounts[states[0]] += 1.0
            np.add.at(transitionCounts, (states[:-1], states[1:]), 1.0)

        initialProbabilities = initialCounts / initialCounts.sum()
        transitionMatrix = transitionCounts / transitionCounts.sum(axis=1, keepdims=True)
        return emissionMeans, emissionVariances, initialProbabilities, transitionMatrix

    def _applyStateOrder(
        self,
        stateSequences: np.ndarray,
        updatedParameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        emissionMeans, emissionVariances, initialProbabilities, transitionMatrix = updatedParameters
        stateOrder = np.argsort(emissionMeans)
        oldToNew = np.empty(self.nStates, dtype=int)
        oldToNew[stateOrder] = np.arange(self.nStates)
        orderedStates = oldToNew[stateSequences]
        self.emissionMeans_ = emissionMeans[stateOrder]
        self.emissionVariances_ = np.maximum(
            emissionVariances[stateOrder], self.varianceFloor
        )
        self.initialProbabilities_ = initialProbabilities[stateOrder]
        self.transitionMatrix_ = transitionMatrix[np.ix_(stateOrder, stateOrder)]
        return orderedStates

    def _validateSequences(self, sequences: np.ndarray) -> np.ndarray:
        sequenceArray = np.asarray(sequences, dtype=float)
        if sequenceArray.ndim != 2:
            raise ValueError("sequences must have shape (nSequences, sequenceLength)")
        if sequenceArray.shape[0] == 0 or sequenceArray.shape[1] == 0:
            raise ValueError("sequences must not be empty")
        if not np.all(np.isfinite(sequenceArray)):
            raise ValueError("sequences contain non-finite values")
        return sequenceArray

    def _checkFitted(self) -> None:
        if not getattr(self, "isFitted_", False):
            raise RuntimeError("fit must be called before prediction")
