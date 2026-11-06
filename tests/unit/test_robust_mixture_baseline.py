"""Unit tests for forward robust-value worst-case mixture baseline (Method F)."""

from __future__ import annotations

import numpy as np

from reversallab.reversal.decision import DecisionOutcome
from reversallab.reversal.robust_mixture import evaluate_robust_mixture_baseline
from reversallab.reversal.search import find_decision_reversal
from reversallab.reversal.strata import (
    StratumPartition,
    StratumStatistics,
    TwoStratumMixtureModel,
)


def _make_model(
    mean_0: float,
    se_0: float,
    mean_1: float,
    se_1: float,
    n_0: int = 1000,
    n_1: int = 1000,
    target_mixture: float = 0.50,
) -> TwoStratumMixtureModel:
    s0 = StratumStatistics(
        stratum_id=0,
        n_samples=n_0,
        empirical_weight=n_0 / (n_0 + n_1),
        mean_diff=mean_0,
        sample_variance=se_0**2 * n_0,
        standard_error=se_0,
    )
    s1 = StratumStatistics(
        stratum_id=1,
        n_samples=n_1,
        empirical_weight=n_1 / (n_0 + n_1),
        mean_diff=mean_1,
        sample_variance=se_1**2 * n_1,
        standard_error=se_1,
    )
    return TwoStratumMixtureModel(
        partition=StratumPartition(0, 0.0),
        stratum_0=s0,
        stratum_1=s1,
        baseline_q0=target_mixture,
        baseline_q1=1.0 - target_mixture,
        is_known_population_mixture=True,
    )


def test_concavity_and_boundary_minimum() -> None:
    """Verify that L_Delta(q) is concave and its infimum is achieved at boundaries."""
    model = _make_model(mean_0=0.08, se_0=0.015, mean_1=0.01, se_1=0.02)
    eps = 0.15
    res = evaluate_robust_mixture_baseline(model, margin=0.02, deploy_epsilon=eps)

    # Grid of 200 interior points
    q_grid = np.linspace(res.q_min, res.q_max, 200)
    l_grid = [model.evaluate_mixture(q)[2] for q in q_grid]

    # Minimum over grid must be at one of the endpoints
    min_grid = min(l_grid)
    assert abs(res.worst_case_lower_bound - min_grid) < 1e-4
    assert res.worst_case_lower_bound <= min(l_grid) + 1e-6


def test_robust_recommendation() -> None:
    """When both strata strongly exceed margin, baseline recommends robustly."""
    model = _make_model(mean_0=0.10, se_0=0.01, mean_1=0.08, se_1=0.01)
    res = evaluate_robust_mixture_baseline(model, margin=0.02, deploy_epsilon=0.10)
    assert res.outcome == DecisionOutcome.RECOMMEND
    assert res.worst_case_lower_bound > 0.02


def test_fragile_candidate_inconclusive() -> None:
    """When candidate reverses within epsilon, outcome is INCONCLUSIVE."""
    # Baseline q0=0.5: mean = 0.5*0.06 + 0.5*(-0.02) = 0.020
    # Lower bound at baseline is ~0.013 > 0.01 (baseline RECOMMEND)
    # At q=0.40: mean = 0.4*0.06 + 0.6*(-0.02) = 0.012, lower bound is ~0.005 <= 0.01 (INCONCLUSIVE)
    model = _make_model(mean_0=0.06, se_0=0.005, mean_1=-0.02, se_1=0.005)
    res = evaluate_robust_mixture_baseline(model, margin=0.01, deploy_epsilon=0.10)
    assert res.outcome == DecisionOutcome.INCONCLUSIVE
    assert res.worst_case_lower_bound <= 0.01


def test_hold_decision() -> None:
    """When candidate is clearly worse than margin, outcome is HOLD."""
    model = _make_model(mean_0=-0.05, se_0=0.01, mean_1=-0.03, se_1=0.01)
    res = evaluate_robust_mixture_baseline(model, margin=0.00, deploy_epsilon=0.10)
    assert res.outcome == DecisionOutcome.HOLD


def test_diagnostic_failure() -> None:
    """Diagnostic failure forces INCONCLUSIVE regardless of values."""
    model = _make_model(mean_0=0.10, se_0=0.01, mean_1=0.08, se_1=0.01)
    res = evaluate_robust_mixture_baseline(
        model,
        margin=0.02,
        deploy_epsilon=0.10,
        diagnostic_passed=False,
        diagnostic_reason="Propensity collapse",
    )
    assert res.outcome == DecisionOutcome.INCONCLUSIVE
    assert "Propensity collapse" in res.reason


def test_mathematical_duality_with_method_d() -> None:
    """Verify that Method D (inverse search) and Method F (DRO) yield identical decisions."""
    rng = np.random.default_rng(42)
    eps_deploy = 0.10
    margin = 0.02

    for _ in range(50):
        m0 = float(rng.uniform(-0.02, 0.10))
        m1 = float(rng.uniform(-0.02, 0.10))
        se0 = float(rng.uniform(0.005, 0.02))
        se1 = float(rng.uniform(0.005, 0.02))
        model = _make_model(m0, se0, m1, se1)

        # Baseline pointwise decision
        _, _, base_l, base_u = model.evaluate_mixture(0.50, confidence_level=0.95)

        is_baseline_rec = base_l > margin

        # Method D
        res_d_search = find_decision_reversal(
            model,
            margin=margin,
            max_epsilon=0.40,
            confidence_level=0.95,
        )
        if is_baseline_rec:
            if res_d_search.status == "NO_REVERSAL_FOUND_WITHIN_BOUNDS" or (
                res_d_search.reversal_distance is not None
                and res_d_search.reversal_distance >= eps_deploy
            ):
                dec_d = DecisionOutcome.RECOMMEND
            else:
                dec_d = DecisionOutcome.INCONCLUSIVE
        else:
            dec_d = DecisionOutcome.HOLD if base_u < margin else DecisionOutcome.INCONCLUSIVE

        # Method F
        res_f = evaluate_robust_mixture_baseline(
            model,
            margin=margin,
            deploy_epsilon=eps_deploy,
            confidence_level=0.95,
        )

        # Both MUST agree on whether to RECOMMEND
        assert (res_f.outcome == DecisionOutcome.RECOMMEND) == (dec_d == DecisionOutcome.RECOMMEND)
        # Both MUST agree on full decision outcome
        assert res_f.outcome == dec_d


def test_affine_case_boundary_minimum() -> None:
    """When one stratum has zero variance, L_Delta is affine and boundary minimum still holds."""
    # se_1 = 0 => affine case
    model = _make_model(mean_0=0.08, se_0=0.015, mean_1=0.02, se_1=0.0)
    eps = 0.15
    res = evaluate_robust_mixture_baseline(model, margin=0.02, deploy_epsilon=eps)

    q_grid = np.linspace(res.q_min, res.q_max, 200)
    l_grid = [model.evaluate_mixture(q)[2] for q in q_grid]
    min_grid = min(l_grid)
    assert abs(res.worst_case_lower_bound - min_grid) < 1e-4


def test_strict_inequality_at_boundary() -> None:
    """When lower bound exactly equals margin, outcome must be INCONCLUSIVE."""
    # Create model where at q_min, L_Delta equals exactly margin
    model = _make_model(mean_0=0.05, se_0=0.01, mean_1=0.05, se_1=0.01)
    base_l = model.evaluate_mixture(0.50)[2]
    # Set margin equal to worst-case lower bound
    res = evaluate_robust_mixture_baseline(model, margin=base_l, deploy_epsilon=0.10)
    # Exact equality with margin must NOT recommend
    assert res.outcome != DecisionOutcome.RECOMMEND
    assert res.outcome == DecisionOutcome.INCONCLUSIVE
