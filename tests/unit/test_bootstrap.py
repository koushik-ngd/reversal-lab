"""Unit tests for nonparametric bootstrap evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.data.synthetic import RewardType, SyntheticBanditEnvironment
from reversallab.evaluation.bootstrap import (
    bootstrap_evaluate,
    construct_bootstrap_intervals,
    resample_logged_data,
)
from reversallab.evaluation.reward_models import StratifiedLinearRewardModel
from reversallab.policies.linear import UniformPolicy


@pytest.fixture
def small_data() -> tuple[LoggedBanditData, np.ndarray]:
    rng = np.random.default_rng(42)
    env = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=rng, reward_type=RewardType.BINARY
    )
    policy = UniformPolicy(n_actions=3)
    data = env.generate_logged_data(50, policy, rng)
    eval_dist = np.full((50, 3), 1.0 / 3.0)
    return data, eval_dist


def test_resample_logged_data_row_alignment(
    small_data: tuple[LoggedBanditData, np.ndarray],
) -> None:
    data, _ = small_data
    rng = np.random.default_rng(123)
    resampled, indices = resample_logged_data(data, rng)

    assert resampled.n_samples == data.n_samples
    assert resampled.n_actions == data.n_actions
    assert len(indices) == data.n_samples

    # Check row alignment: every row in resampled matches data at the sampled index
    for j, idx in enumerate(indices):
        np.testing.assert_array_equal(resampled.context[j], data.context[idx])
        assert resampled.action[j] == data.action[idx]
        assert resampled.reward[j] == data.reward[idx]
        assert resampled.logging_propensity[j] == data.logging_propensity[idx]
        if data.logging_action_dist is not None:
            assert resampled.logging_action_dist is not None
            np.testing.assert_array_equal(
                resampled.logging_action_dist[j], data.logging_action_dist[idx]
            )


def test_bootstrap_deterministic_seed(small_data: tuple[LoggedBanditData, np.ndarray]) -> None:
    data, eval_dist = small_data
    res1 = bootstrap_evaluate("ips", data, eval_dist, n_bootstraps=20, seed=999)
    res2 = bootstrap_evaluate("ips", data, eval_dist, n_bootstraps=20, seed=999)

    assert res1.point_estimate == res2.point_estimate
    np.testing.assert_array_equal(res1.bootstrap_estimates, res2.bootstrap_estimates)
    assert res1.percentile_ci == res2.percentile_ci
    assert res1.basic_ci == res2.basic_ci
    assert res1.normal_ci == res2.normal_ci


def test_construct_bootstrap_intervals_known_quantiles() -> None:
    # 100 evenly spaced estimates from 1.0 to 100.0
    estimates = np.linspace(1.0, 100.0, 100)
    point_est = 50.0

    # 90% confidence level -> alpha = 0.10 -> percentiles at 5% and 95%
    perc_ci, basic_ci, norm_ci = construct_bootstrap_intervals(
        point_est, estimates, confidence_level=0.90
    )

    # 5th percentile is approx 5.95, 95th is approx 95.05
    assert perc_ci[0] == pytest.approx(5.95, abs=0.1)
    assert perc_ci[1] == pytest.approx(95.05, abs=0.1)

    # Basic CI is [2*50 - 95.05, 2*50 - 5.95] = [4.95, 94.05]
    assert basic_ci[0] == pytest.approx(4.95, abs=0.1)
    assert basic_ci[1] == pytest.approx(94.05, abs=0.1)

    # Normal CI is symmetric around point_est
    assert norm_ci[0] < point_est < norm_ci[1]
    assert point_est - norm_ci[0] == pytest.approx(norm_ci[1] - point_est)


def test_bootstrap_fixed_model_vs_refit(small_data: tuple[LoggedBanditData, np.ndarray]) -> None:
    data, eval_dist = small_data
    q_hat = np.full((data.n_samples, data.n_actions), 0.5)

    # Fixed model bootstrap: does not refit
    res_fixed = bootstrap_evaluate(
        "dm",
        data,
        eval_dist,
        predicted_rewards=q_hat,
        n_bootstraps=15,
        refit=False,
        seed=1,
    )
    assert not res_fixed.refit_model
    # For constant q_hat = 0.5, DM is identically 0.5 on every sample!
    np.testing.assert_allclose(res_fixed.bootstrap_estimates, 0.5)
    assert res_fixed.bootstrap_standard_error == pytest.approx(0.0)

    # Refit bootstrap: refits StratifiedLinearRewardModel each time, so estimates will vary
    res_refit = bootstrap_evaluate(
        "dm",
        data,
        eval_dist,
        model_factory=lambda: StratifiedLinearRewardModel(
            n_actions=3, reward_type=RewardType.BINARY
        ),
        n_bootstraps=10,
        n_folds=2,
        refit=True,
        seed=1,
    )
    assert res_refit.refit_model
    assert len(res_refit.bootstrap_estimates) == 10
    assert np.all(np.isfinite(res_refit.bootstrap_estimates))


def test_bootstrap_all_estimators_smoke(small_data: tuple[LoggedBanditData, np.ndarray]) -> None:
    data, eval_dist = small_data

    def factory() -> StratifiedLinearRewardModel:
        return StratifiedLinearRewardModel(n_actions=3, reward_type=RewardType.BINARY)

    for est in ["ips", "snips", "dm", "dr"]:
        res = bootstrap_evaluate(
            est,
            data,
            eval_dist,
            model_factory=factory,
            n_bootstraps=5,
            n_folds=2,
            refit=True,
            seed=42,
        )
        assert res.estimator == est
        assert np.isfinite(res.point_estimate)
        assert len(res.bootstrap_estimates) == 5
        assert res.percentile_ci[0] <= res.percentile_ci[1]


def test_bootstrap_handles_degenerate_weights_gracefully() -> None:
    """Verify that DegenerateWeightsError on base estimate is propagated and NaNs are handled."""
    from reversallab.exceptions import DegenerateWeightsError, SupportViolationError

    data = LoggedBanditData(
        context=np.array([[1.0], [2.0]]),
        action=np.array([0, 0]),
        reward=np.array([1.0, 1.0]),
        logging_propensity=np.array([1.0, 1.0]),
        n_actions=2,
    )
    # Action 1 evaluated, but only action 0 is logged and observed -> weights are all 0
    # SNIPS raises DegenerateWeightsError on base estimate
    with pytest.raises(DegenerateWeightsError):
        bootstrap_evaluate("snips", data, np.array([[0.0, 1.0], [0.0, 1.0]]))

    # If full logging distribution has zero mass, raises SupportViolationError
    data_with_dist = LoggedBanditData(
        context=np.array([[1.0], [2.0]]),
        action=np.array([0, 0]),
        reward=np.array([1.0, 1.0]),
        logging_propensity=np.array([1.0, 1.0]),
        n_actions=2,
        logging_action_dist=np.array([[1.0, 0.0], [1.0, 0.0]]),
    )
    with pytest.raises(SupportViolationError):
        bootstrap_evaluate("snips", data_with_dist, np.array([[0.0, 1.0], [0.0, 1.0]]))

    # construct_bootstrap_intervals with < 2 clean estimates returns NaNs
    p_ci, b_ci, n_ci = construct_bootstrap_intervals(1.0, np.array([float("nan"), float("nan")]))
    assert np.isnan(p_ci[0])
    assert np.isnan(p_ci[1])
    assert np.isnan(b_ci[0])
    assert np.isnan(b_ci[1])
    assert np.isnan(n_ci[0])
    assert np.isnan(n_ci[1])
