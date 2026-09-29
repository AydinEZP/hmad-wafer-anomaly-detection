from __future__ import annotations

import numpy as np


def _validate_sequences(sequences: np.ndarray) -> np.ndarray:
    sequenceArray = np.asarray(sequences, dtype=float)
    if sequenceArray.ndim != 2:
        raise ValueError("sequences must have shape (nSequences, sequenceLength)")
    if sequenceArray.shape[0] == 0 or sequenceArray.shape[1] == 0:
        raise ValueError("sequences must not be empty")
    if not np.all(np.isfinite(sequenceArray)):
        raise ValueError("sequences contain non-finite values")
    return sequenceArray


def flattenedFeatures(sequences: np.ndarray) -> np.ndarray:
    """Preserve the absolute position of every time step."""

    return _validate_sequences(sequences).copy()


def globalSummaryFeatures(sequences: np.ndarray) -> np.ndarray:
    """Return mean, standard deviation, minimum, maximum, and median."""

    sequenceArray = _validate_sequences(sequences)
    return np.column_stack(
        (
            sequenceArray.mean(axis=1),
            sequenceArray.std(axis=1),
            sequenceArray.min(axis=1),
            sequenceArray.max(axis=1),
            np.median(sequenceArray, axis=1),
        )
    )


def windowSummaryFeatures(
    sequences: np.ndarray,
    nWindows: int = 5,
) -> np.ndarray:
    """Compute five summary statistics in each relative sequence window.

    ``numpy.array_split`` is used so Wafer's length 152 does not need to be
    divisible by the number of windows. Every sequence receives the same fixed
    feature dimension ``5 * nWindows``.
    """

    sequenceArray = _validate_sequences(sequences)
    if not 1 <= nWindows <= sequenceArray.shape[1]:
        raise ValueError("nWindows must be between 1 and the sequence length")

    blocks: list[np.ndarray] = []
    for window in np.array_split(sequenceArray, nWindows, axis=1):
        blocks.append(
            np.column_stack(
                (
                    window.mean(axis=1),
                    window.std(axis=1),
                    window.min(axis=1),
                    window.max(axis=1),
                    np.median(window, axis=1),
                )
            )
        )
    return np.hstack(blocks)


def buildFeatureRepresentation(
    sequences: np.ndarray,
    representation: str,
    *,
    nWindows: int = 5,
) -> np.ndarray:
    normalizedName = representation.strip().lower()
    if normalizedName == "flattened":
        return flattenedFeatures(sequences)
    if normalizedName in {"global", "global_summary", "summary"}:
        return globalSummaryFeatures(sequences)
    if normalizedName in {"window", "window_summary", "windowed"}:
        return windowSummaryFeatures(sequences, nWindows=nWindows)
    raise ValueError(f"Unknown OC-SVM representation: {representation}")
