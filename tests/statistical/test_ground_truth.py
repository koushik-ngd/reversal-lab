"""Monte Carlo checks of IPS/SNIPS against synthetic ground truth.

Tolerances are expressed in Monte Carlo standard errors of the quantity being
tested, so a failure indicates a discrepancy well beyond replication noise. Seeds
are fixed, so these tests are deterministic; the tolerances document how far the
result could move under a different seed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reversallab.data import SyntheticBanditEnvironment
from reversallab.evaluation import ips, snips
from reversallab.evaluation.estimators import Estimator
from reversallab.policies import (
    EpsilonGreedyLinearPolicy,
    LinearSoftmaxPolicy,
    Policy,
    UniformPolicy,
)

pytestmark = pytest.mark.statistical

ENV = SyntheticBanditEnvironment.random(
    n_actions=5, context_dim=3, rng=np.random.default_rng(123), reward_scale=1.5, bias_scale=0.5
)
INFORMED = LinearSoftmaxPolicy(ENV.reward_weights, ENV.reward_bias, inverse_temperature=2.0)
GREEDY = EpsilonGreedyLinearPolicy(ENV.reward_weights, ENV.reward_bias, epsilon=0.0)
LOGGER = LinearSoftmaxPolicy(ENV.reward_weights, ENV.reward_bias, inverse_temperature=0.5)


def _true_value(policy: Policy) -> tuple[float, float]:
    ref = ENV.policy_value(policy, np.random.default_rng(999), n_contexts=1_000_000)
    return ref.value, ref.monte_carlo_standard_error


def _replicate(
    estimator: Estimator, logging: Policy, target: Policy, n: int, reps: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, lo, hi = [], [], []
    for r in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, r]))
        data = ENV.generate_logged_data(n, logging, rng)
        est = estimator(data, target.action_distribution(data.context))
        values.append(est.value)
        lo.append(est.ci_lower)
        hi.append(est.ci_upper)
    return np.array(values), np.array(lo), np.array(hi)


@pytest.mark.parametrize(
    ("estimator", "logging", "target", "n"),
    [
        (ips, UniformPolicy(5), INFORMED, 1000),
        (ips, LOGGER, INFORMED, 1000),
        (ips, UniformPolicy(5), GREEDY, 1000),
        (ips, LOGGER, LOGGER, 200),
        (snips, UniformPolicy(5), INFORMED, 2000),
        (snips, LOGGER, GREEDY, 2000),
    ],
)
def test_estimator_is_centred_on_true_value(
    estimator: Estimator, logging: Policy, target: Policy, n: int
) -> None:
    true, ref_se = _true_value(target)
    values, _, _ = _replicate(estimator, logging, target, n, reps=400, seed=1)
    mc_se = math.sqrt(values.var(ddof=1) / len(values) + ref_se**2)
    assert abs(values.mean() - true) < 4 * mc_se, (values.mean(), true, mc_se)


def test_ips_wald_coverage_near_nominal_with_good_overlap() -> None:
    true, _ = _true_value(INFORMED)
    _, lo, hi = _replicate(ips, UniformPolicy(5), INFORMED, n=2000, reps=600, seed=2)
    coverage = float(np.mean((lo <= true) & (true <= hi)))
    se = math.sqrt(0.95 * 0.05 / 600)
    assert abs(coverage - 0.95) < 4 * se, coverage


@pytest.mark.parametrize("estimator", [ips, snips])
def test_error_shrinks_at_root_n_rate(estimator: Estimator) -> None:
    true, _ = _true_value(INFORMED)
    small, _, _ = _replicate(estimator, UniformPolicy(5), INFORMED, n=250, reps=300, seed=3)
    large, _, _ = _replicate(estimator, UniformPolicy(5), INFORMED, n=4000, reps=300, seed=3)
    rmse_small = math.sqrt(np.mean((small - true) ** 2))
    rmse_large = math.sqrt(np.mean((large - true) ** 2))
    # A 16x larger sample should cut RMSE by ~4x; require at least a 3x reduction.
    assert rmse_large < rmse_small / 3, (rmse_small, rmse_large)


def test_snips_has_lower_variance_than_ips_with_heavy_weights() -> None:
    """Known property of self-normalisation for rewards in [0, 1] (not a theorem for all cases)."""
    v_ips, _, _ = _replicate(ips, LOGGER, GREEDY, n=500, reps=400, seed=4)
    v_snips, _, _ = _replicate(snips, LOGGER, GREEDY, n=500, reps=400, seed=4)
    assert v_snips.var() < v_ips.var()
