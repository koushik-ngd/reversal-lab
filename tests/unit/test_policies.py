from __future__ import annotations

import numpy as np
import pytest

from reversallab.exceptions import ConfigurationError, DataValidationError
from reversallab.policies import (
    EpsilonGreedyLinearPolicy,
    LinearSoftmaxPolicy,
    Policy,
    UniformPolicy,
    sample_actions,
)

RNG = np.random.default_rng(0)
W = RNG.normal(size=(3, 4))
X = RNG.normal(size=(50, 3))


@pytest.mark.parametrize(
    "policy",
    [
        UniformPolicy(4),
        LinearSoftmaxPolicy(W, inverse_temperature=2.0),
        EpsilonGreedyLinearPolicy(W, epsilon=0.3),
    ],
)
def test_distributions_are_valid(policy: Policy) -> None:
    assert isinstance(policy, Policy)
    dist = policy.action_distribution(X)
    assert dist.shape == (50, 4)
    assert np.all(dist >= 0)
    np.testing.assert_allclose(dist.sum(axis=1), 1.0, atol=1e-12)


def test_softmax_zero_temperature_is_uniform() -> None:
    dist = LinearSoftmaxPolicy(W, inverse_temperature=0.0).action_distribution(X)
    np.testing.assert_allclose(dist, 0.25)


def test_softmax_matches_definition() -> None:
    bias = np.array([0.1, -0.2, 0.0, 0.3])
    dist = LinearSoftmaxPolicy(W, bias, inverse_temperature=1.5).action_distribution(X)
    z = np.exp(1.5 * (X @ W + bias))
    np.testing.assert_allclose(dist, z / z.sum(axis=1, keepdims=True), rtol=1e-12)


def test_softmax_is_numerically_stable_for_large_scores() -> None:
    dist = LinearSoftmaxPolicy(W * 1e3, inverse_temperature=1.0).action_distribution(X)
    assert np.all(np.isfinite(dist))


def test_epsilon_greedy_values() -> None:
    dist = EpsilonGreedyLinearPolicy(W, epsilon=0.2).action_distribution(X)
    greedy = np.argmax(X @ W, axis=1)
    np.testing.assert_allclose(dist[np.arange(50), greedy], 0.8 + 0.05)
    assert np.sum(np.isclose(dist, 0.05)) == 50 * 3


def test_greedy_is_deterministic_one_hot() -> None:
    dist = EpsilonGreedyLinearPolicy(W, epsilon=0.0).action_distribution(X)
    assert set(np.unique(dist)) == {0.0, 1.0}
    np.testing.assert_array_equal(dist.sum(axis=1), 1.0)


def test_context_dimension_checked() -> None:
    with pytest.raises(DataValidationError, match="context dimension"):
        LinearSoftmaxPolicy(W).action_distribution(np.zeros((2, 5)))


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ConfigurationError):
        EpsilonGreedyLinearPolicy(W, epsilon=epsilon)


def test_sampling_never_draws_zero_probability_actions() -> None:
    dist = np.tile([0.0, 0.5, 0.0, 0.5, 0.0], (20_000, 1))
    actions = sample_actions(dist, np.random.default_rng(1))
    assert set(np.unique(actions)) == {1, 3}


def test_sampling_frequencies_match_probabilities() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.4])
    n = 200_000
    actions = sample_actions(np.tile(p, (n, 1)), np.random.default_rng(2))
    freq = np.bincount(actions, minlength=4) / n
    # 5 binomial standard errors per action.
    np.testing.assert_array_less(np.abs(freq - p), 5 * np.sqrt(p * (1 - p) / n))


def test_sampling_is_reproducible() -> None:
    dist = UniformPolicy(6).action_distribution(np.zeros((100, 1)))
    a = sample_actions(dist, np.random.default_rng(3))
    b = sample_actions(dist, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_sampling_rejects_invalid_distribution() -> None:
    with pytest.raises(DataValidationError):
        sample_actions(np.array([[0.5, 0.6]]), np.random.default_rng(0))
