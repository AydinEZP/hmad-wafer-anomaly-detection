"""Log-space Viterbi decoding and validation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .hmm import gaussian_logpdf, joint_log_probability, validate_gaussian_hmm_parameters


@dataclass(frozen=True)
class ViterbiResult:
    path: NDArray[np.int64]
    log_score: float
    delta: NDArray[np.float64] | None = None
    backpointers: NDArray[np.int64] | None = None


def viterbi_decode(
    x: ArrayLike,
    pi: ArrayLike,
    A: ArrayLike,
    means: ArrayLike,
    variances: ArrayLike,
    *,
    return_tables: bool = False,
) -> ViterbiResult:
    r"""Decode the globally most likely hidden path in log space.

    The dynamic program is

    .. math::
       \delta_t(j) = \max_i[\delta_{t-1}(i)+\log A_{ij}]
                     + \log p(x_t\mid z_t=j).
    """
    params = validate_gaussian_hmm_parameters(pi, A, means, variances)
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError("x must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(x_arr)):
        raise ValueError("x must contain finite observations")

    T = x_arr.size
    K = params.n_states
    emissions = gaussian_logpdf(x_arr, params.means, params.variances)
    with np.errstate(divide="ignore"):
        log_pi = np.log(params.pi)
        log_A = np.log(params.A)

    delta = np.full((T, K), -np.inf, dtype=float)
    back = np.full((T, K), -1, dtype=np.int64)
    delta[0] = log_pi + emissions[0]

    for t in range(1, T):
        candidates = delta[t - 1][:, None] + log_A
        back[t] = np.argmax(candidates, axis=0)
        delta[t] = candidates[back[t], np.arange(K)] + emissions[t]

    last_state = int(np.argmax(delta[-1]))
    best_score = float(delta[-1, last_state])
    if not np.isfinite(best_score):
        raise ValueError("no finite-probability state path exists for this observation sequence")

    path = np.empty(T, dtype=np.int64)
    path[-1] = last_state
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    return ViterbiResult(
        path=path,
        log_score=best_score,
        delta=delta if return_tables else None,
        backpointers=back if return_tables else None,
    )


def brute_force_decode(
    x: ArrayLike,
    pi: ArrayLike,
    A: ArrayLike,
    means: ArrayLike,
    variances: ArrayLike,
    *,
    max_paths: int = 1_000_000,
) -> ViterbiResult:
    """Enumerate every path; use only for short correctness tests."""
    params = validate_gaussian_hmm_parameters(pi, A, means, variances)
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError("x must be a non-empty one-dimensional sequence")
    n_paths = params.n_states ** x_arr.size
    if n_paths > max_paths:
        raise ValueError(f"refusing to enumerate {n_paths:,} paths; increase max_paths explicitly")

    best_path: tuple[int, ...] | None = None
    best_score = -np.inf
    for path_tuple in product(range(params.n_states), repeat=x_arr.size):
        path = np.fromiter(path_tuple, dtype=np.int64)
        score = joint_log_probability(
            x_arr, path, params.pi, params.A, params.means, params.variances
        )
        if score > best_score:
            best_score = score
            best_path = path_tuple
    assert best_path is not None
    return ViterbiResult(np.asarray(best_path, dtype=np.int64), float(best_score))


def state_accuracy(true_states: ArrayLike, decoded_states: ArrayLike) -> float:
    """Return direct state-label accuracy when labels have a fixed semantic order."""
    true_arr = np.asarray(true_states, dtype=np.int64)
    decoded_arr = np.asarray(decoded_states, dtype=np.int64)
    if true_arr.shape != decoded_arr.shape or true_arr.ndim != 1 or true_arr.size == 0:
        raise ValueError("state sequences must be non-empty one-dimensional arrays of equal shape")
    return float(np.mean(true_arr == decoded_arr))
