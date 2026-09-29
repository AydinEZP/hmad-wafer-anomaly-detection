"""Adapters connecting model components without changing their public APIs."""
from __future__ import annotations

import numpy as np

from .viterbi import viterbi_decode


def viterbi_path_adapter(
    sequence: np.ndarray,
    initial_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    emission_means: np.ndarray,
    emission_variances: np.ndarray,
) -> np.ndarray:
    """Return only the zero-based path expected by ``SimplifiedHMAD``.

    The reviewed decoder returns a ``ViterbiResult`` containing the path, log
    score, and optional dynamic-programming tables. The HMAD injection point
    expects a state array (or tuple), so this adapter exposes the required
    state-only contract.
    """
    result = viterbi_decode(
        sequence,
        initial_probabilities,
        transition_matrix,
        emission_means,
        emission_variances,
    )
    return result.path
