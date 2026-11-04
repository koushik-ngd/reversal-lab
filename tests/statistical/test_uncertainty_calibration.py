"""Statistical tests for CI coverage and calibration against synthetic ground truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from reversallab.data.synthetic import RewardType, SyntheticBanditEnvironment
from reversallab.evaluation.bootstrap import bootstrap_evaluate
from reversallab.evaluation.cross_fitting import cross_fit_reward_model
from reversallab.evaluation.diagnostics import DiagnosticThresholds, diagnose
from reversallab.evaluation.estimators import dm
from reversallab.evaluation.influence_scores import evaluate_influence_inference
from reversallab.evaluation.reward_models import (
    MisspecifiedLinearRewardModel,
    StratifiedLinearRewardModel,
)
from reversallab.policies.linear import LinearSoftmaxPolicy, UniformPolicy

pytestmark = pytest.mark.statistical

ENV = SyntheticBanditEnvironment.random(
    n_actions=4,
    context_dim=3,
    rng=np.random.default_rng(20261030),
    reward_type=RewardType.BINARY,
    reward_scale=1.0,
    bias_scale=0.2,
)

INFORMED = LinearSoftmaxPolicy(ENV.reward_weights, ENV.reward_bias, inverse_temperature=1.0)


def _true_value(policy: LinearSoftmaxPolicy) -> float:
    rng = np.random.default_rng(20261030)
    ref = ENV.policy_value(policy, rng, n_contexts=100_000)
    return ref.value


def test_dr_influence_wald_coverage_good_overlap() -> None:
    """Under good overlap and correct model, DR influence Wald coverage is near 95%."""
    true_v = _true_value(INFORMED)
    n = 400
    reps = 60
    covered = 0

    for r in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence([101, n, r]))
        data = ENV.generate_logged_data(n, UniformPolicy(4), rng, include_logging_action_dist=True)
        eval_dist = INFORMED.action_distribution(data.context)

        q_hat = cross_fit_reward_model(
            data,
            lambda: StratifiedLinearRewardModel(n_actions=4, reward_type=ENV.reward_type),
            n_folds=3,
            seed=r,
        )

        res = evaluate_influence_inference("dr", data, eval_dist, q_hat, confidence_level=0.95)
        if res.ci_lower <= true_v <= res.ci_upper:
            covered += 1

    cov = covered / reps
    mc_se = math.sqrt(cov * (1.0 - cov) / reps)
    # Coverage should be within 3 binomial standard errors of nominal 0.95
    assert cov >= 0.88, f"DR influence Wald coverage too low: {cov:.3f} (mc_se={mc_se:.3f})"


def test_dm_naive_wald_undercovers_vs_bootstrap() -> None:
    """Naive DM Wald interval severely undercovers compared to full-refit bootstrap."""
    true_v = _true_value(INFORMED)
    n = 200
    reps = 40
    naive_covered = 0
    boot_covered = 0

    for r in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence([102, n, r]))
        data = ENV.generate_logged_data(n, UniformPolicy(4), rng, include_logging_action_dist=True)
        eval_dist = INFORMED.action_distribution(data.context)

        def factory() -> StratifiedLinearRewardModel:
            return StratifiedLinearRewardModel(n_actions=4, reward_type=ENV.reward_type)

        q_hat = cross_fit_reward_model(data, factory, n_folds=2, seed=r)

        est_naive = dm(data, eval_dist, q_hat, confidence_level=0.95)
        if est_naive.ci_lower <= true_v <= est_naive.ci_upper:
            naive_covered += 1

        est_boot = bootstrap_evaluate(
            "dm",
            data,
            eval_dist,
            model_factory=factory,
            n_bootstraps=25,
            n_folds=2,
            refit=True,
            seed=r,
        )
        if est_boot.percentile_ci[0] <= true_v <= est_boot.percentile_ci[1]:
            boot_covered += 1

    naive_cov = naive_covered / reps
    boot_cov = boot_covered / reps

    # Naive DM Wald misses model uncertainty and undercovers significantly relative to bootstrap
    assert boot_cov > naive_cov, (
        f"Expected refit bootstrap coverage ({boot_cov:.3f}) > naive Wald ({naive_cov:.3f})"
    )


def test_dr_coverage_under_misspecified_model() -> None:
    """DR influence Wald interval maintains high coverage even under model misspecification."""
    true_v = _true_value(INFORMED)
    n = 400
    reps = 50
    covered = 0

    for r in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence([103, n, r]))
        data = ENV.generate_logged_data(n, UniformPolicy(4), rng, include_logging_action_dist=True)
        eval_dist = INFORMED.action_distribution(data.context)

        # Deliberately misspecified model
        q_hat = cross_fit_reward_model(
            data,
            lambda: MisspecifiedLinearRewardModel(n_actions=4, reward_type=ENV.reward_type),
            n_folds=2,
            seed=r,
        )

        res = evaluate_influence_inference("dr", data, eval_dist, q_hat, confidence_level=0.95)
        if res.ci_lower <= true_v <= res.ci_upper:
            covered += 1

    cov = covered / reps
    # DR should still cover at >= 88% because propensities are correct
    assert cov >= 0.88, f"DR under misspecified model had insufficient coverage: {cov:.3f}"


def test_poor_overlap_emits_diagnostics() -> None:
    """Poor overlap scenario emits low ESS and concentrated weights warnings."""
    rng = np.random.default_rng(999)
    # Highly confident logger independent of reward
    random_logits = rng.standard_normal((3, 4))
    logger_policy = LinearSoftmaxPolicy(random_logits, np.zeros(4), inverse_temperature=3.0)
    data = ENV.generate_logged_data(300, logger_policy, rng, include_logging_action_dist=True)
    eval_dist = INFORMED.action_distribution(data.context)

    report = diagnose(data, eval_dist, DiagnosticThresholds(min_ess_fraction=0.10))
    assert (
        "low_effective_sample_size" in report.warning_codes
        or "dominant_weight" in report.warning_codes
        or "concentrated_weights" in report.warning_codes
    )
