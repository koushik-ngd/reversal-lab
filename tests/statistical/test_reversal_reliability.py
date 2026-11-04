"""Statistical assertions for Milestone M4B-1 Decision-Reversal Reliability Benchmark."""

from __future__ import annotations

from typing import Any

import pytest

from reversallab.experiments.config import load_reliability_config
from reversallab.experiments.reversal_reliability import run_reversal_reliability


@pytest.fixture(scope="module")
def reliability_run_results() -> dict[str, Any]:
    """Execute Experiment 006 with 5 replications for statistical validation."""
    config = load_reliability_config("configs/experiments/006_reversal_reliability.yaml")
    # Use 5 replications for test suite speed while preserving statistical properties
    config = type(config).model_validate({**config.model_dump(), "n_replications": 5})
    output = run_reversal_reliability(config)
    return output.results


class TestReversalReliabilityStatisticalProperties:
    """Statistical verification of the decision-reversal reliability benchmark."""

    def test_analytical_and_numerical_agreement(
        self, reliability_run_results: dict[str, Any]
    ) -> None:
        """Numerical bisection and exact closed-form analytical reference must match within 1e-4."""
        for sc in reliability_run_results["scenarios"]:
            val = sc["reversal_search_validation"]
            max_disc = val["max_analytical_discrepancy"]
            assert val["discrepancy_within_tolerance"] is True, (
                f"{sc['scenario']}: discrepancy {max_disc} exceeds 1e-4"
            )
            assert max_disc <= 1e-4

    def test_robust_scenario_recommends_consistently(
        self, reliability_run_results: dict[str, Any]
    ) -> None:
        """In robust improvement, all methods (A through E) recommend deployment."""
        sc = next(
            s
            for s in reliability_run_results["scenarios"]
            if s["scenario"] == "robust_true_improvement"
        )
        methods = sc["methods"]
        for m_key in [
            "method_a_point_estimate",
            "method_b_baseline_ci",
            "method_c_worst_stratum",
            "method_d_pointwise_reversal",
            "method_e_simultaneous_band",
        ]:
            assert methods[m_key]["recommend_rate"] == 1.0, f"{m_key} did not recommend 100%"
        assert sc["diagnostics"]["diagnostic_gate_failed_rate"] == 0.0

    def test_poor_overlap_trips_diagnostic_gate_and_blocks_deployment(
        self, reliability_run_results: dict[str, Any]
    ) -> None:
        """In poor overlap, diagnostic gate must trip 100% of the time, blocking deployment."""
        sc = next(
            s for s in reliability_run_results["scenarios"] if s["scenario"] == "poor_overlap"
        )
        assert sc["diagnostics"]["diagnostic_gate_failed_rate"] == 1.0
        methods = sc["methods"]
        # Uncertainty-aware methods B through E must all have 0% recommendation
        assert methods["method_b_baseline_ci"]["recommend_rate"] == 0.0
        assert methods["method_c_worst_stratum"]["recommend_rate"] == 0.0
        assert methods["method_d_pointwise_reversal"]["recommend_rate"] == 0.0
        assert methods["method_e_simultaneous_band"]["recommend_rate"] == 0.0

    def test_performance_vs_margin_reversal_taxonomy(
        self, reliability_run_results: dict[str, Any]
    ) -> None:
        """Verify strict taxonomy separation between margin reversals and performance reversals."""
        # 1. Genuine performance reversal: has performance reversal
        sc_perf = next(
            s
            for s in reliability_run_results["scenarios"]
            if s["scenario"] == "genuine_performance_reversal"
        )
        assert sc_perf["taxonomy"]["has_performance_reversal"] is True
        assert sc_perf["ground_truth"]["true_performance_reversal_distance"] is not None

        # 2. Decision without performance reversal:
        # has margin reversal but NEVER performance reversal
        sc_margin = next(
            s
            for s in reliability_run_results["scenarios"]
            if s["scenario"] == "decision_without_perf_reversal"
        )
        assert sc_margin["taxonomy"]["has_margin_reversal"] is True
        assert sc_margin["taxonomy"]["has_performance_reversal"] is False
        assert sc_margin["ground_truth"]["true_margin_reversal_distance"] is not None
        assert sc_margin["ground_truth"]["true_performance_reversal_distance"] is None
        # Both stratum true deltas must be strictly positive
        assert sc_margin["ground_truth"]["true_delta_0"] > 0.0
        assert sc_margin["ground_truth"]["true_delta_1"] > 0.0

    def test_equal_strata_zero_slope_linear_degeneracy(
        self, reliability_run_results: dict[str, Any]
    ) -> None:
        """When strata have identical effect, slope is zero and solver handles linear degeneracy."""
        sc = next(
            s
            for s in reliability_run_results["scenarios"]
            if s["scenario"] == "equal_strata_near_boundary"
        )
        assert abs(sc["ground_truth"]["true_delta_0"] - sc["ground_truth"]["true_delta_1"]) < 0.01
        assert sc["reversal_search_validation"]["max_analytical_discrepancy"] <= 1e-4
