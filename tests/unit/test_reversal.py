"""Unit tests for the decision-reversal subpackage."""

from __future__ import annotations

import numpy as np
import pytest

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.policies.linear import LinearSoftmaxPolicy, UniformPolicy
from reversallab.reversal.decision import (
    DecisionOutcome,
    evaluate_decision,
    evaluate_decision_from_paired_result,
)
from reversallab.reversal.paired import (
    compute_paired_difference_scores,
    evaluate_paired_difference,
)
from reversallab.reversal.search import (
    bisect_decision_boundary,
    find_decision_reversal,
)
from reversallab.reversal.strata import (
    StratumPartition,
    create_two_stratum_model,
)


@pytest.fixture
def synthetic_batch() -> tuple[LoggedBanditData, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a small deterministic synthetic batch for unit testing."""
    rng = np.random.default_rng(42)
    env = SyntheticBanditEnvironment.random(
        n_actions=3,
        context_dim=2,
        rng=rng,
        reward_type="binary",
    )
    logging_policy = UniformPolicy(n_actions=3)
    data = env.generate_logged_data(
        n_samples=200, logging_policy=logging_policy, rng=np.random.default_rng(100)
    )
    policy_cand = LinearSoftmaxPolicy(
        weights=env.reward_weights, bias=env.reward_bias, inverse_temperature=1.0
    )
    policy_base = LinearSoftmaxPolicy(
        weights=env.reward_weights, bias=env.reward_bias, inverse_temperature=0.2
    )
    cand_dist = policy_cand.action_distribution(data.context)
    base_dist = policy_base.action_distribution(data.context)
    q_hat = np.full((data.n_samples, data.n_actions), 0.5)
    return data, cand_dist, base_dist, q_hat


class TestPairedDifference:
    """Test paired per-observation difference scoring and inference."""

    def test_paired_dr_difference_scores(
        self, synthetic_batch: tuple[LoggedBanditData, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        data, cand_dist, base_dist, q_hat = synthetic_batch
        diff_scores, psi_e, psi_b = compute_paired_difference_scores(
            "dr", data, cand_dist, base_dist, predicted_rewards=q_hat
        )
        assert len(diff_scores) == data.n_samples
        np.testing.assert_allclose(diff_scores, psi_e - psi_b)

        res = evaluate_paired_difference("dr", data, cand_dist, base_dist, predicted_rewards=q_hat)
        assert res.estimator == "dr"
        assert res.n_samples == data.n_samples
        np.testing.assert_allclose(res.point_estimate, float(np.mean(diff_scores)))
        np.testing.assert_allclose(
            res.point_estimate, res.candidate_point_estimate - res.baseline_point_estimate
        )

    def test_paired_ips_difference_scores(
        self, synthetic_batch: tuple[LoggedBanditData, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        data, cand_dist, base_dist, _ = synthetic_batch
        res = evaluate_paired_difference("ips", data, cand_dist, base_dist)
        assert res.estimator == "ips"
        np.testing.assert_allclose(
            res.point_estimate, res.candidate_point_estimate - res.baseline_point_estimate
        )

    def test_paired_dm_difference_scores(
        self, synthetic_batch: tuple[LoggedBanditData, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        data, cand_dist, base_dist, q_hat = synthetic_batch
        res = evaluate_paired_difference("dm", data, cand_dist, base_dist, predicted_rewards=q_hat)
        assert res.estimator == "dm"
        np.testing.assert_allclose(
            res.point_estimate,
            res.candidate_point_estimate - res.baseline_point_estimate,
            atol=1e-12,
        )

    def test_paired_snips_difference_scores(
        self, synthetic_batch: tuple[LoggedBanditData, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        data, cand_dist, base_dist, _ = synthetic_batch
        res = evaluate_paired_difference("snips", data, cand_dist, base_dist)
        assert res.estimator == "snips"
        np.testing.assert_allclose(
            res.point_estimate, res.candidate_point_estimate - res.baseline_point_estimate
        )

    def test_positive_covariance_reduces_variance(
        self, synthetic_batch: tuple[LoggedBanditData, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        data, cand_dist, base_dist, q_hat = synthetic_batch
        res = evaluate_paired_difference("dr", data, cand_dist, base_dist, predicted_rewards=q_hat)
        # Because candidate and baseline are positively correlated, covariance > 0
        assert res.covariance > 0.0
        assert res.empirical_variance < res.independent_variance
        assert res.standard_error < res.independent_standard_error
        assert res.variance_reduction_pct > 0.0

    def test_predicted_rewards_required_for_dr_and_dm(
        self, synthetic_batch: tuple[LoggedBanditData, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        data, cand_dist, base_dist, _ = synthetic_batch
        with pytest.raises(ValueError, match="predicted_rewards is required"):
            evaluate_paired_difference("dr", data, cand_dist, base_dist, predicted_rewards=None)
        with pytest.raises(ValueError, match="predicted_rewards is required"):
            evaluate_paired_difference("dm", data, cand_dist, base_dist, predicted_rewards=None)


class TestDecisionRule:
    """Test deployment decision rule and boundary conventions."""

    def test_recommend_when_lower_bound_exceeds_margin(self) -> None:
        res = evaluate_decision(
            ci_lower=0.06,
            ci_upper=0.10,
            point_estimate=0.08,
            margin=0.05,
        )
        assert res.outcome == DecisionOutcome.RECOMMEND
        assert res.is_actionable
        assert res.naive_matches_evidence

    def test_hold_when_upper_bound_below_margin(self) -> None:
        res = evaluate_decision(
            ci_lower=-0.02,
            ci_upper=0.04,
            point_estimate=0.01,
            margin=0.05,
        )
        assert res.outcome == DecisionOutcome.HOLD
        assert res.is_actionable
        assert res.naive_matches_evidence

    def test_inconclusive_when_interval_covers_margin(self) -> None:
        res = evaluate_decision(
            ci_lower=0.02,
            ci_upper=0.08,
            point_estimate=0.06,
            margin=0.05,
        )
        assert res.outcome == DecisionOutcome.INCONCLUSIVE
        assert not res.is_actionable
        # Naive point estimate would dangerously recommend deployment!
        assert res.naive_outcome == DecisionOutcome.RECOMMEND
        assert not res.naive_matches_evidence

    def test_conservative_boundary_convention(self) -> None:
        # Exact lower bound equality
        res_eq_lower = evaluate_decision(
            ci_lower=0.05,
            ci_upper=0.10,
            point_estimate=0.075,
            margin=0.05,
        )
        assert res_eq_lower.outcome == DecisionOutcome.INCONCLUSIVE

        # Exact upper bound equality
        res_eq_upper = evaluate_decision(
            ci_lower=0.01,
            ci_upper=0.05,
            point_estimate=0.03,
            margin=0.05,
        )
        assert res_eq_upper.outcome == DecisionOutcome.INCONCLUSIVE

    def test_diagnostic_gate_forces_inconclusive(
        self, synthetic_batch: tuple[LoggedBanditData, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        data, cand_dist, base_dist, q_hat = synthetic_batch
        paired = evaluate_paired_difference(
            "dr", data, cand_dist, base_dist, predicted_rewards=q_hat
        )
        # Force min_propensity violation
        res = evaluate_decision_from_paired_result(
            paired,
            margin=0.01,
            min_propensity=1e-6,
            min_propensity_threshold=1e-4,
        )
        assert res.outcome == DecisionOutcome.INCONCLUSIVE
        assert not res.diagnostic_passed
        assert "minimum propensity" in res.reason


class TestTwoStratumModel:
    """Test two-stratum context mixture model under total variation perturbation."""

    def test_stratum_partition_assignment(self) -> None:
        partition = StratumPartition(split_dim=0, split_threshold=0.0)
        contexts = np.array([[-1.0, 0.5], [0.0, 1.0], [2.0, -0.5], [-0.5, 0.0]])
        assignments = partition.assign(contexts)
        np.testing.assert_array_equal(assignments, [0, 1, 1, 0])

    def test_model_construction_and_mixture_evaluation(self) -> None:
        contexts = np.array([[-1.0, 0.0], [-2.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        diff_scores = np.array([0.10, 0.12, -0.04, -0.06])
        model = create_two_stratum_model(contexts, diff_scores)

        assert model.stratum_0.n_samples == 2
        assert model.stratum_1.n_samples == 2
        assert model.baseline_q0 == 0.5
        np.testing.assert_allclose(model.stratum_0.mean_diff, 0.11)
        np.testing.assert_allclose(model.stratum_1.mean_diff, -0.05)

        # Baseline mixture: 0.5 * 0.11 + 0.5 * (-0.05) = 0.03
        pt, _se, ci_lower, ci_upper = model.evaluate_mixture(0.5)
        np.testing.assert_allclose(pt, 0.03)
        assert ci_lower < pt < ci_upper


class TestReversalSearch:
    """Test 1-D decision-reversal search with numerical bisection."""

    def test_robust_scenario_no_reversal_within_bounds(self) -> None:
        # Both strata strongly favor candidate (> margin = 0.02)
        contexts = np.random.default_rng(42).normal(size=(1000, 2))
        # Stratum 0 has mean 0.10, Stratum 1 has mean 0.08
        diff_scores = np.where(contexts[:, 0] < 0, 0.10, 0.08) + np.random.default_rng(42).normal(
            scale=0.02, size=1000
        )
        model = create_two_stratum_model(contexts, diff_scores)

        res = find_decision_reversal(model, margin=0.02, max_epsilon=0.40)
        assert res.initial_decision == DecisionOutcome.RECOMMEND
        assert not res.is_reversed
        assert res.status == "NO_REVERSAL_FOUND_WITHIN_BOUNDS"
        assert res.reversal_distance is None
        assert res.fine_grid_verified

    def test_fragile_scenario_reversal_detected(self) -> None:
        # Stratum 0 has mean +0.15, Stratum 1 has mean -0.08
        # Margin = 0.02. Baseline q0=0.5: mixture mean = 0.5*0.15 + 0.5*(-0.08) = 0.035 > 0.02
        rng = np.random.default_rng(123)
        contexts = rng.normal(size=(2000, 2))
        diff_scores = np.where(contexts[:, 0] < 0, 0.15, -0.08) + rng.normal(scale=0.02, size=2000)
        model = create_two_stratum_model(contexts, diff_scores)

        res = find_decision_reversal(model, margin=0.02, max_epsilon=0.40, tolerance=1e-5)
        assert res.initial_decision == DecisionOutcome.RECOMMEND
        assert res.is_reversed
        assert res.reversal_distance is not None
        assert 0.0 < res.reversal_distance <= 0.40
        assert res.transition == "RECOMMEND -> INCONCLUSIVE"
        assert res.status == "REVERSAL_FOUND"
        assert res.fine_grid_verified

    def test_bisection_accuracy(self) -> None:
        # Check bisection stops within requested tolerance
        rng = np.random.default_rng(456)
        contexts = rng.normal(size=(1000, 2))
        diff_scores = np.where(contexts[:, 0] < 0, 0.12, -0.05) + rng.normal(scale=0.01, size=1000)
        model = create_two_stratum_model(contexts, diff_scores)

        q_rev, dec_rev = bisect_decision_boundary(
            model,
            q_start=model.baseline_q0,
            q_end=0.0,
            margin=0.02,
            tolerance=1e-6,
        )
        assert dec_rev != DecisionOutcome.RECOMMEND
        # Evaluate just epsilon before q_rev and verify it was RECOMMEND
        step = 1e-4
        if q_rev + step <= model.baseline_q0:
            res_before = model.evaluate_mixture_decision(q_rev + step, margin=0.02)
            assert res_before.outcome == DecisionOutcome.RECOMMEND
