"""Statistical assertions for Milestone M4C Independent Reliability Benchmark."""

from __future__ import annotations

from typing import Any

import pytest

from reversallab.experiments.config import load_independent_reliability_config
from reversallab.experiments.independent_reliability import run_independent_reliability


@pytest.fixture(scope="module")
def independent_run_results() -> dict[str, Any]:
    """Execute Experiment 007 with 5 replications for fast test suite verification."""
    config = load_independent_reliability_config(
        "configs/experiments/007_independent_reliability.yaml"
    )
    config = type(config).model_validate({**config.model_dump(), "n_replications": 5})
    output = run_independent_reliability(config)
    return output.results


class TestIndependentReliabilityStatisticalProperties:
    """Statistical verification of the M4C independent reliability benchmark."""

    def test_mathematical_duality_d_and_f(self, independent_run_results: dict[str, Any]) -> None:
        """Method D (inverse search) and Method F (forward DRO) must produce identical decisions."""
        for sc in independent_run_results["scenarios"]:
            duality = sc["method_d_f_duality_validation"]
            assert duality["exact_decision_equivalence"] is True, (
                f"{sc['scenario']}: Method D and F decisions differed "
                f"(agreement {duality['agreement_rate']})"
            )
            assert duality["agreement_rate"] == 1.0

    def test_analytical_and_numerical_agreement(
        self, independent_run_results: dict[str, Any]
    ) -> None:
        """Numerical bisection and exact closed-form analytical roots match within 1e-4."""
        for sc in independent_run_results["scenarios"]:
            val = sc["reversal_distance_evaluation"]
            max_disc = val["max_analytical_discrepancy"]
            assert val["discrepancy_within_tolerance"] is True, (
                f"{sc['scenario']}: discrepancy {max_disc} exceeds 1e-4"
            )
            assert max_disc <= 1e-4

    def test_simultaneous_coverage_property(self, independent_run_results: dict[str, Any]) -> None:
        """Simultaneous confidence band achieves conservative coverage over Q_epsilon."""
        for sc in independent_run_results["scenarios"]:
            sim_cov = sc["simultaneous_inference_validation"]["simultaneous_interval_coverage"]
            # With M=5, coverage is either 0.8 or 1.0
            assert sim_cov >= 0.80, f"{sc['scenario']}: simultaneous coverage too low ({sim_cov})"

    def test_harmful_candidate_safety(self, independent_run_results: dict[str, Any]) -> None:
        """Truly harmful candidate must have 0% recommendation and 0% harmful deployment."""
        sc = next(
            s
            for s in independent_run_results["scenarios"]
            if s["scenario"] == "truly_harmful_candidate"
        )
        for m_name, m_data in sc["methods"].items():
            assert m_data["recommend_rate"] == 0.0, f"{m_name} recommended harmful policy"
            assert m_data["harmful_deployment_rate"] == 0.0

    def test_null_advantage_type_i_control(self, independent_run_results: dict[str, Any]) -> None:
        """Near-zero null advantage must not produce confident recommendations."""
        sc = next(
            s
            for s in independent_run_results["scenarios"]
            if s["scenario"] == "near_zero_null_advantage"
        )
        # Methods B through F must not recommend under the null
        for m_name in [
            "method_b_baseline_ci",
            "method_c_worst_stratum",
            "method_d_pointwise_reversal",
            "method_e_simultaneous_band",
            "method_f_robust_mixture",
        ]:
            assert sc["methods"][m_name]["recommend_rate"] == 0.0
            assert sc["methods"][m_name]["baseline_false_recommendation_rate"] == 0.0

    def test_poor_overlap_gate_failure(self, independent_run_results: dict[str, Any]) -> None:
        """Poor overlap trips diagnostic gate 100% of the time, blocking deployment."""
        sc = next(
            s
            for s in independent_run_results["scenarios"]
            if s["scenario"] == "poor_overlap_propensity_deficit"
        )
        assert sc["diagnostics"]["diagnostic_gate_failed_rate"] == 1.0
        for m_name in [
            "method_b_baseline_ci",
            "method_c_worst_stratum",
            "method_d_pointwise_reversal",
            "method_e_simultaneous_band",
            "method_f_robust_mixture",
        ]:
            assert sc["methods"][m_name]["recommend_rate"] == 0.0

    def test_decision_without_perf_reversal_taxonomy(
        self, independent_run_results: dict[str, Any]
    ) -> None:
        """Verify margin reversal occurs without performance reversal (both strata positive)."""
        sc = next(
            s
            for s in independent_run_results["scenarios"]
            if s["scenario"] == "decision_without_perf_reversal"
        )
        assert sc["taxonomy"]["has_margin_reversal"] is True
        assert sc["taxonomy"]["has_performance_reversal"] is False
        assert sc["ground_truth"]["true_margin_reversal_distance"] is not None
        assert sc["ground_truth"]["true_performance_reversal_distance"] is None
        assert sc["ground_truth"]["true_delta_0"] > 0.0
        assert sc["ground_truth"]["true_delta_1"] > 0.0
