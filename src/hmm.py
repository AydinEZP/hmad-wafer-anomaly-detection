"""Gaussian-emission hidden Markov model helpers."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .markov import validate_markov_parameters


@dataclass(frozen=True)
class GaussianHMMParameters:
    """Validated scalar Gaussian-emission HMM parameters."""

    pi: NDArray[np.float64]
    A: NDArray[np.float64]
    means: NDArray[np.float64]
    variances: NDArray[np.float64]

    @property
    def n_states(self) -> int:
        return int(self.pi.size)


def validate_gaussian_hmm_parameters(
    pi: ArrayLike,
    A: ArrayLike,
    means: ArrayLike,
    variances: ArrayLike,
) -> GaussianHMMParameters:
    pi_arr, A_arr = validate_markov_parameters(pi, A)
    means_arr = np.asarray(means, dtype=float)
    vars_arr = np.asarray(variances, dtype=float)
    expected = (pi_arr.size,)
    if means_arr.shape != expected or vars_arr.shape != expected:
        raise ValueError(
            f"means and variances must both have shape {expected}; "
            f"received {means_arr.shape} and {vars_arr.shape}"
        )
    if np.any(~np.isfinite(means_arr)):
        raise ValueError("means must be finite")
    if np.any(~np.isfinite(vars_arr)) or np.any(vars_arr <= 0.0):
        raise ValueError("variances must be finite and strictly positive")
    return GaussianHMMParameters(
        pi=pi_arr.copy(),
        A=A_arr.copy(),
        means=means_arr.copy(),
        variances=vars_arr.copy(),
    )


def gaussian_logpdf(
    x: ArrayLike, means: ArrayLike, variances: ArrayLike
) -> NDArray[np.float64]:
    """Evaluate scalar Gaussian log densities for all x/state combinations.

    If ``x`` has shape ``(T,)`` and the parameter vectors have shape ``(K,)``,
    the returned array has shape ``(T, K)``.
    """
    x_arr = np.asarray(x, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    vars_arr = np.asarray(variances, dtype=float)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError("x must be a non-empty one-dimensional sequence")
    if means_arr.ndim != 1 or vars_arr.shape != means_arr.shape:
        raise ValueError("means and variances must be one-dimensional vectors of equal size")
    if np.any(~np.isfinite(x_arr)) or np.any(~np.isfinite(means_arr)):
        raise ValueError("x and means must be finite")
    if np.any(~np.isfinite(vars_arr)) or np.any(vars_arr <= 0.0):
        raise ValueError("variances must be finite and strictly positive")

    centered = x_arr[:, None] - means_arr[None, :]
    return -0.5 * (
        np.log(2.0 * np.pi * vars_arr)[None, :]
        + centered**2 / vars_arr[None, :]
    )


def sample_gaussian_hmm(
    pi: ArrayLike,
    A: ArrayLike,
    means: ArrayLike,
    variances: ArrayLike,
    length: int,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Generate hidden states and scalar observations from a Gaussian HMM."""
    params = validate_gaussian_hmm_parameters(pi, A, means, variances)
    if not isinstance(length, (int, np.integer)) or length <= 0:
        raise ValueError("length must be a positive integer")
    if rng is None:
        rng = np.random.default_rng()

    z = np.empty(length, dtype=np.int64)
    x = np.empty(length, dtype=float)
    states = np.arange(params.n_states)

    z[0] = rng.choice(states, p=params.pi)
    x[0] = rng.normal(params.means[z[0]], np.sqrt(params.variances[z[0]]))
    for t in range(1, length):
        z[t] = rng.choice(states, p=params.A[z[t - 1]])
        x[t] = rng.normal(params.means[z[t]], np.sqrt(params.variances[z[t]]))
    return z, x


def joint_log_probability(
    x: ArrayLike,
    z: ArrayLike,
    pi: ArrayLike,
    A: ArrayLike,
    means: ArrayLike,
    variances: ArrayLike,
) -> float:
    """Compute log P(x, z) for a scalar Gaussian-emission HMM."""
    from .markov import markov_log_probability, validate_state_sequence

    params = validate_gaussian_hmm_parameters(pi, A, means, variances)
    x_arr = np.asarray(x, dtype=float)
    z_arr = validate_state_sequence(z, params.n_states)
    if x_arr.ndim != 1 or x_arr.size != z_arr.size:
        raise ValueError("x and z must be one-dimensional sequences of equal non-zero length")
    log_emissions = gaussian_logpdf(x_arr, params.means, params.variances)
    return float(
        markov_log_probability(z_arr, params.pi, params.A)
        + log_emissions[np.arange(x_arr.size), z_arr].sum()
    )
