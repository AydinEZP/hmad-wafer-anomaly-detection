import numpy as np

from src.adapters import viterbi_path_adapter
from src.viterbi import viterbi_decode


def test_adapter_returns_same_path_as_reviewed_decoder() -> None:
    x = np.array([-0.2, 0.1, 3.8, 4.2, 0.0])
    pi = np.array([0.6, 0.4])
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([0.0, 4.0])
    variances = np.array([1.0, 1.0])
    expected = viterbi_decode(x, pi, A, means, variances).path
    actual = viterbi_path_adapter(x, pi, A, means, variances)
    assert np.array_equal(actual, expected)
    assert actual.dtype.kind in {'i', 'u'}
