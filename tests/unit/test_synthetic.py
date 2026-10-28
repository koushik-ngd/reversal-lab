from __future__ import annotations

import numpy as np
import pytest

from reversallab.data import RewardType, SyntheticBanditEnvironment
from reversallab.exceptions import ConfigurationError
from reversallab.policies import LinearSoftmaxPolicy, UniformPolicy


def _env(reward_type: RewardType = RewardType.BINARY) -> SyntheticBanditEnvironment:
    return SyntheticBanditEnvironment.random(
        n_actions=4,
        context_dim=3,
        rng=np.random.default_rng(0),
        reward_type=reward_type,
        reward_scale=1.0,
        bias_scale=0.5,
    )


def test_logged_data_shapes_and_contract() -> None:
    env = _env()
    policy = LinearSoftmaxPolicy(env.reward_weights, env.reward_bias)
    data = env.generate_logged_data(200, policy, np.random.default_rng(1))
    assert data.context.shape == (200, 3)
    assert data.logging_action_dist is not None
    assert set(np.unique(data.reward)) <= {0.0, 1.0}
    np.testing.assert_allclose(
        data.logging_propensity,
        policy.action_distribution(data.context)[np.arange(200), data.action],
    )


def test_logging_dist_can_be_omitted() -> None:
    env = _env()
    data = env.generate_logged_data(
        10, UniformPolicy(4), np.random.default_rng(1), include_logging_action_dist=False
    )
    assert data.logging_action_dist is None


def test_same_seed_same_data() -> None:
    env = _env(RewardType.GAUSSIAN)
    a = env.generate_logged_data(100, UniformPolicy(4), np.random.default_rng(5))
    b = env.generate_logged_data(100, UniformPolicy(4), np.random.default_rng(5))
    for field in ("context", "action", "reward", "logging_propensity"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_expected_reward_binary_in_unit_interval() -> None:
    env = _env()
    q = env.expected_reward(env.sample_contexts(1000, np.random.default_rng(2)))
    assert q.shape == (1000, 4)
    assert np.all((q > 0) & (q < 1))


def test_conditional_value_matches_hand_calculation() -> None:
    env = SyntheticBanditEnvironment(
        reward_weights=[[1.0, -1.0]], reward_bias=[0.0, 0.5], reward_type=RewardType.GAUSSIAN
    )
    x = np.array([[0.0], [2.0]])
    # q = [[0, 0.5], [2, -1.5]]; uniform policy averages each row: [0.25, 0.25].
    assert env.conditional_policy_value(x, UniformPolicy(2)) == pytest.approx(0.25)


def test_policy_value_matches_closed_form() -> None:
    """Gaussian linear rewards, uniform policy: V = mean(bias) because E[x] = 0."""
    env = _env(RewardType.GAUSSIAN)
    ref = env.policy_value(UniformPolicy(4), np.random.default_rng(3), n_contexts=200_000)
    assert abs(ref.value - env.reward_bias.mean()) < 4 * ref.monte_carlo_standard_error
    assert ref.monte_carlo_standard_error < 0.01


def test_policy_value_is_unchanged_by_chunk_size() -> None:
    """Chunking is only a memory device. It must not change the Monte Carlo estimate.

    Partial sums are not bitwise identical, so the comparison allows a few ulps.
    """
    env = _env()
    policy = LinearSoftmaxPolicy(env.reward_weights, env.reward_bias, inverse_temperature=2.0)
    whole = env.policy_value(policy, np.random.default_rng(4), n_contexts=10_000, chunk_size=10_000)
    chunked = env.policy_value(
        policy, np.random.default_rng(4), n_contexts=10_000, chunk_size=3_000
    )
    assert chunked.n_contexts == whole.n_contexts
    assert chunked.value == pytest.approx(whole.value, abs=1e-12)
    assert chunked.monte_carlo_standard_error == pytest.approx(
        whole.monte_carlo_standard_error, abs=1e-12
    )


def test_action_count_mismatch_rejected() -> None:
    with pytest.raises(ConfigurationError, match="actions"):
        _env().generate_logged_data(10, UniformPolicy(3), np.random.default_rng(0))


def test_invalid_environment_rejected() -> None:
    with pytest.raises(ConfigurationError):
        SyntheticBanditEnvironment(reward_weights=[[1.0]], reward_bias=[0.0])
