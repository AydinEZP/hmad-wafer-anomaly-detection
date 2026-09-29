from __future__ import annotations

import numpy as np


def normalizeStateLabels(states: np.ndarray, nStates: int) -> np.ndarray:
    statesArray = np.asarray(states, dtype=int).reshape(-1)
    if statesArray.size == 0:
        raise ValueError("states must not be empty")
    if statesArray.min() >= 0 and statesArray.max() < nStates:
        return statesArray
    if statesArray.min() >= 1 and statesArray.max() <= nStates:
        return statesArray - 1
    raise ValueError("states must use either zero-based or one-based labels")


def computeJointFeatures(
    sequence: np.ndarray,
    states: np.ndarray,
    nStates: int = 2,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    sequenceArray = np.asarray(sequence, dtype=float).reshape(-1)
    stateArray = normalizeStateLabels(states, nStates)
    if sequenceArray.size != stateArray.size:
        raise ValueError("sequence and states must have the same length")
    if sequenceArray.size == 0:
        raise ValueError("sequence must not be empty")

    transitionFeatures = np.zeros((nStates, nStates), dtype=float)
    if sequenceArray.size > 1:
        np.add.at(transitionFeatures, (stateArray[:-1], stateArray[1:]), 1.0)
        transitionFeatures /= sequenceArray.size - 1

    emissionFeatures = np.zeros(nStates, dtype=float)
    for stateIndex in range(nStates):
        stateMask = stateArray == stateIndex
        emissionFeatures[stateIndex] = sequenceArray[stateMask].sum() / (stateMask.sum() + epsilon)

    jointFeatures = np.concatenate((transitionFeatures.reshape(-1), emissionFeatures))
    if not np.all(np.isfinite(jointFeatures)):
        raise ValueError("joint features contain non-finite values")
    return jointFeatures


def batchJointFeatures(
    sequences: np.ndarray,
    stateSequences: np.ndarray,
    nStates: int = 2,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    sequenceArray = np.asarray(sequences, dtype=float)
    stateArray = np.asarray(stateSequences, dtype=int)
    if sequenceArray.ndim != 2 or stateArray.ndim != 2:
        raise ValueError("sequences and stateSequences must be two-dimensional")
    if sequenceArray.shape != stateArray.shape:
        raise ValueError("sequences and stateSequences must have the same shape")
    return np.vstack(
        [
            computeJointFeatures(sequence, states, nStates, epsilon)
            for sequence, states in zip(sequenceArray, stateArray)
        ]
    )


compute_joint_features = computeJointFeatures
