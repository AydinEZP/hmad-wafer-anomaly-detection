"""Markov-chain probability utilities.

Code convention
---------------
States are represented by zero-based integer indices ``0, ..., K-1``.
The report may use one-based mathematical labels ``1, ..., K``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_probability_vector(pi: ArrayLike, *, atol: float = 1e-10) -> NDArray[np.float64]:
    arr = np.asarray(pi, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("pi must be a non-empty one-dimensional vector")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("pi must contain finite non-negative probabilities")
    if not np.isclose(arr.sum(), 1.0, atol=atol):
        raise ValueError(f"pi must sum to one; received {arr.sum():.12g}")
    return arr


def _validate_transition_matrix(
    A: ArrayLike, *, n_states: int | None = None, atol: float = 1e-10
) -> NDArray[np.float64]:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError("A must be a non-empty square matrix")
    if n_states is not None and arr.shape != (n_states, n_states):
        raise ValueError(f"A must have shape {(n_states, n_states)}, received {arr.shape}")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("A must contain finite non-negative probabilities")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=atol):
        raise ValueError("each row of A must sum to one")
    return arr


def validate_markov_parameters(
    pi: ArrayLike, A: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return a copied initial distribution and transition matrix."""
    pi_arr = _validate_probability_vector(pi)
    A_arr = _validate_transition_matrix(A, n_states=pi_arr.size)
    return pi_arr.copy(), A_arr.copy()


def validate_state_sequence(z: ArrayLike, n_states: int) -> NDArray[np.int64]:
    """Validate a non-empty zero-based integer state sequence."""
    arr = np.asarray(z)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("z must be a non-empty one-dimensional state sequence")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(arr, np.floor(arr))):
            raise ValueError("z must contain integer state indices")
    arr = arr.astype(np.int64, copy=False)
    if np.any(arr < 0) or np.any(arr >= n_states):
        raise ValueError(f"state indices must lie in [0, {n_states - 1}]")
    return arr


def markov_log_probability(z: ArrayLike, pi: ArrayLike, A: ArrayLike) -> float:
    r"""Return the log-probability of a state path.

    Computes

    .. math::
       \log P(z) = \log \pi_{z_1} + \sum_{t=2}^{T}\log A_{z_{t-1},z_t}.

    Impossible paths return ``-np.inf`` rather than raising a floating-point error.
    """
    pi_arr, A_arr = validate_markov_parameters(pi, A)
    z_arr = validate_state_sequence(z, pi_arr.size)

    with np.errstate(divide="ignore"):
        log_prob = np.log(pi_arr[z_arr[0]])
        if z_arr.size > 1:
            log_prob += np.log(A_arr[z_arr[:-1], z_arr[1:]]).sum()
    return float(log_prob)


def markov_probability(z: ArrayLike, pi: ArrayLike, A: ArrayLike) -> float:
    """Return the raw path probability; intended only for short diagnostic examples."""
    return float(np.exp(markov_log_probability(z, pi, A)))
