"""Statistical validation tests for Milestone M4A Decision-Reversal Prototype."""

from __future__ import annotations

from typing import Any

import pytest

from reversallab.experiments.config import load_reversal_config
from reversallab.experiments.reversal_prototype import run_reversal_prototype
from reversallab.reversal.decision import DecisionOutcome

pytestmark = pytest.mark.statistical


@pytest.fixture(scope="module")
def reversal_run_results() -> dict[str, Any]:
    """Execute Experiment 005 once for statistical assertions."""
    config = load_reversal_config("configs/experiments/005_reversal_prototype.yaml")
    output = run_reversal_prototype(config)
    return output.results


class TestReversalPrototypeStatisticalProperties:
    """Statistical assertions against known synthetic ground truth."""

    def test_estimation_accuracy_against_ground_truth(
        self, reversal_run_results: dict[str, Any]
    ) -> None:
        """Point estimate must fall within 3 standard errors of the ground truth."""
        for sc in reversal_run_results["scenarios"]:
            true_delta = sc["ground_truth"]["true_delta"]
            est_delta = sc["paired_estimation"]["point_estimate"]
            se = sc["paired_estimation"]["standard_error"]
            # Within 3 standard errors
            assert abs(est_delta - true_delta) <= 3.0 * se, (
                f"{sc['scenario']}: est ({est_delta:.4f}) deviates from true ({true_delta:.4f}) "
                f"by more than 3 se ({se:.4f})"
            )

    def test_robust_superiority_scenario(self, reversal_run_results: dict[str, Any]) -> None:
        """Robust scenario must recommend deployment and have no reversal within bounds."""
        sc = next(
            s for s in reversal_run_results["scenarios"] if s["scenario"] == "robust_superiority"
        )
        assert sc["decisions"]["baseline_confidence_bound"] == DecisionOutcome.RECOMMEND.value
        assert sc["decisions"]["naive_point_estimate"] == DecisionOutcome.RECOMMEND.value
        assert sc["reversal_search"]["status"] == "NO_REVERSAL_FOUND_WITHIN_BOUNDS"
        assert sc["reversal_search"]["reversal_distance"] is None
        assert sc["reversal_search"]["fine_grid_verified"] is True

    def test_fragile_superiority_scenario(self, reversal_run_results: dict[str, Any]) -> None:
        """Fragile scenario must recommend initially but reverse under small perturbation."""
        sc = next(
            s for s in reversal_run_results["scenarios"] if s["scenario"] == "fragile_superiority"
        )
        assert sc["decisions"]["baseline_confidence_bound"] == DecisionOutcome.RECOMMEND.value
        assert sc["reversal_search"]["status"] == "REVERSAL_FOUND"
        assert sc["reversal_search"]["transition"] == "RECOMMEND -> INCONCLUSIVE"
        rev_dist = sc["reversal_search"]["reversal_distance"]
        assert rev_dist is not None
        assert 0.01 <= rev_dist <= 0.20
        assert sc["reversal_search"]["fine_grid_verified"] is True

    def test_high_uncertainty_scenario_protects_from_naive_deployment(
        self, reversal_run_results: dict[str, Any]
    ) -> None:
        """High uncertainty scenario must trigger INCONCLUSIVE despite positive naive estimate."""
        sc = next(
            s for s in reversal_run_results["scenarios"] if s["scenario"] == "high_uncertainty"
        )
        # Naive rule falsely recommends
        assert sc["decisions"]["naive_point_estimate"] == DecisionOutcome.RECOMMEND.value
        # Uncertainty-aware rule holds back as INCONCLUSIVE
        assert sc["decisions"]["baseline_confidence_bound"] == DecisionOutcome.INCONCLUSIVE.value
        ci_lower = sc["paired_estimation"]["ci_lower"]
        ci_upper = sc["paired_estimation"]["ci_upper"]
        margin = sc["margin"]
        assert ci_lower <= margin <= ci_upper

    def test_poor_overlap_scenario_triggers_diagnostic_failure(
        self, reversal_run_results: dict[str, Any]
    ) -> None:
        """Poor overlap must fail diagnostic gate and block perturbation search."""
        sc = next(s for s in reversal_run_results["scenarios"] if s["scenario"] == "poor_overlap")
        assert sc["decisions"]["diagnostic_passed"] is False
        assert sc["decisions"]["baseline_confidence_bound"] == DecisionOutcome.INCONCLUSIVE.value
        assert sc["reversal_search"]["status"] == "BASELINE_DIAGNOSTIC_FAILURE"
        assert sc["reversal_search"]["reversal_distance"] is None

    def test_paired_variance_reduction_is_positive(
        self, reversal_run_results: dict[str, Any]
    ) -> None:
        """Evaluating on same observations must produce positive covariance and reduction."""
        for sc in reversal_run_results["scenarios"]:
            pe = sc["paired_estimation"]
            assert pe["covariance"] > 0.0
            assert pe["standard_error"] < pe["independent_standard_error"]
            assert pe["variance_reduction_pct"] > 0.0
