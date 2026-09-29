import numpy as np
import pytest

from src.markov import markov_log_probability
from src.hmm import gaussian_logpdf, sample_gaussian_hmm
from src.viterbi import brute_force_decode, viterbi_decode

PI = np.array([0.6, 0.4])
A = np.array([[0.9, 0.1], [0.2, 0.8]])
MEANS = np.array([0.0, 3.0])
VARIANCES = np.array([1.0, 1.5])


def test_markov_log_probability_matches_manual_value():
    z = np.array([0, 0, 1, 1])
    expected = np.log(0.6) + np.log(0.9) + np.log(0.1) + np.log(0.8)
    assert np.isclose(markov_log_probability(z, PI, A), expected)


def test_impossible_path_returns_negative_infinity():
    A_zero = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert markov_log_probability([0, 1], PI, A_zero) == -np.inf


def test_generator_is_reproducible_and_has_correct_shapes():
    z1, x1 = sample_gaussian_hmm(
        PI, A, MEANS, VARIANCES, 50, rng=np.random.default_rng(1405)
    )
    z2, x2 = sample_gaussian_hmm(
        PI, A, MEANS, VARIANCES, 50, rng=np.random.default_rng(1405)
    )
    assert z1.shape == x1.shape == (50,)
    assert np.array_equal(z1, z2)
    assert np.allclose(x1, x2)
    assert np.all((0 <= z1) & (z1 < 2))


def test_gaussian_logpdf_shape_and_finiteness():
    logp = gaussian_logpdf(np.array([-1.0, 0.0, 2.0]), MEANS, VARIANCES)
    assert logp.shape == (3, 2)
    assert np.all(np.isfinite(logp))


@pytest.mark.parametrize("seed", range(10))
def test_viterbi_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    _, x = sample_gaussian_hmm(PI, A, MEANS, VARIANCES, 7, rng=rng)
    vit = viterbi_decode(x, PI, A, MEANS, VARIANCES)
    brute = brute_force_decode(x, PI, A, MEANS, VARIANCES)
    assert np.array_equal(vit.path, brute.path)
    assert np.isclose(vit.log_score, brute.log_score)


def test_viterbi_tables_have_expected_shapes():
    _, x = sample_gaussian_hmm(
        PI, A, MEANS, VARIANCES, 12, rng=np.random.default_rng(4)
    )
    result = viterbi_decode(x, PI, A, MEANS, VARIANCES, return_tables=True)
    assert result.path.shape == (12,)
    assert result.delta is not None and result.delta.shape == (12, 2)
    assert result.backpointers is not None and result.backpointers.shape == (12, 2)
    assert np.isfinite(result.log_score)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        markov_log_probability([0, 1], [0.4, 0.4], A)
    with pytest.raises(ValueError):
        sample_gaussian_hmm(PI, A, MEANS, [1.0, 0.0], 10)
