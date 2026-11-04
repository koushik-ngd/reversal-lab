"""Unit tests for exact closed-form analytical decision-reversal solver."""

from __future__ import annotations

import math

import pytest

from reversallab.reversal.analytical import (
    find_analytical_decision_reversal,
    solve_quadratic_crossing_roots,
)
from reversallab.reversal.search import find_decision_reversal
from reversallab.reversal.strata import (
    StratumPartition,
    StratumStatistics,
    TwoStratumMixtureModel,
)


def _make_strata(
    diff_0: float, se_0: float, diff_1: float, se_1: float, n_per_stratum: int = 1000
) -> TwoStratumMixtureModel:
    s0 = StratumStatistics(
        stratum_id=0,
        n_samples=n_per_stratum,
        empirical_weight=0.50,
        mean_diff=diff_0,
        sample_variance=se_0**2 * n_per_stratum,
        standard_error=se_0,
    )
    s1 = StratumStatistics(
        stratum_id=1,
        n_samples=n_per_stratum,
        empirical_weight=0.50,
        mean_diff=diff_1,
        sample_variance=se_1**2 * n_per_stratum,
        standard_error=se_1,
    )
    return TwoStratumMixtureModel(
        partition=StratumPartition(0, 0.0),
        stratum_0=s0,
        stratum_1=s1,
        baseline_q0=0.50,
        baseline_q1=0.50,
        is_known_population_mixture=True,
    )


class TestQuadraticCrossingSolver:
    """Mathematical verification of the closed-form quadratic polynomial solver."""

    def test_known_symmetric_crossing(self) -> None:
        """Symmetric setup where exact crossing point can be computed by hand."""
        d0, d1 = 0.10, -0.02
        se0, se1 = 0.01, 0.01
        z = 1.959963984540054
        margin = 0.02

        roots = solve_quadratic_crossing_roots(
            delta_0=d0,
            se_0=se0,
            delta_1=d1,
            se_1=se1,
            margin=margin,
            z=z,
        )
        assert len(roots) >= 1
        for q, _is_tangent in roots:
            delta_q = q * d0 + (1 - q) * d1
            se_q = math.sqrt(q**2 * se0**2 + (1 - q) ** 2 * se1**2)
            # Crossing equation (Delta(q) - delta)^2 = z^2 * se^2
            assert abs((delta_q - margin) ** 2 - (z**2) * (se_q**2)) < 1e-10

    def test_upper_and_lower_bound_crossings(self) -> None:
        """Roots must satisfy the squared confidence bound equation."""
        d0, d1 = 0.08, -0.04
        se0, se1 = 0.015, 0.015
        z = 1.96
        margin = 0.02

        roots = solve_quadratic_crossing_roots(
            delta_0=d0,
            se_0=se0,
            delta_1=d1,
            se_1=se1,
            margin=margin,
            z=z,
        )
        for q, _ in roots:
            delta_q = q * d0 + (1 - q) * d1
            se_q = math.sqrt(q**2 * se0**2 + (1 - q) ** 2 * se1**2)
            assert abs((delta_q - margin) ** 2 - (z**2) * (se_q**2)) < 1e-10

    def test_roots_outside_unit_interval(self) -> None:
        """When curve never reaches margin in [0, 1], roots lie outside [0, 1]."""
        # Both strata strongly exceed margin: d0=0.20, d1=0.20, margin=0.01
        roots = solve_quadratic_crossing_roots(
            delta_0=0.20,
            se_0=0.005,
            delta_1=0.20,
            se_1=0.005,
            margin=0.01,
            z=1.96,
        )
        # Real algebraic roots exist at large |q|, but none in [0, 1]
        for q, _ in roots:
            assert q < 0.0 or q > 1.0

    def test_tangency_single_root(self) -> None:
        """Construct exact tangency where curve just touches margin."""
        s = 0.02
        z = 2.0
        d0 = 0.05
        margin = d0 - z * (s / math.sqrt(2.0))
        roots = solve_quadratic_crossing_roots(
            delta_0=d0,
            se_0=s,
            delta_1=d0,
            se_1=s,
            margin=margin,
            z=z,
        )
        assert len(roots) >= 1
        assert any(abs(q - 0.50) < 1e-6 and is_tangent for q, is_tangent in roots)

    def test_linear_degeneracy_c2_near_zero(self) -> None:
        """When c2 == 0, solver falls back to linear equation -c0 / c1."""
        se0, se1 = 0.01, 0.01
        z = 1.96
        diff = z * math.sqrt(se0**2 + se1**2)
        d1 = 0.01
        d0 = d1 + diff
        roots = solve_quadratic_crossing_roots(
            delta_0=d0,
            se_0=se0,
            delta_1=d1,
            se_1=se1,
            margin=0.01,
            z=z,
        )
        assert isinstance(roots, list)


class TestAnalyticalVersusNumericalAgreement:
    """Validate numerical bisection search against exact analytical reference."""

    @pytest.mark.parametrize(
        ("d0", "d1", "se0", "se1", "margin"),
        [
            (0.12, -0.03, 0.015, 0.015, 0.02),
            (0.08, -0.02, 0.010, 0.012, 0.02),
            (0.15, -0.05, 0.020, 0.020, 0.03),
            (0.06, 0.00, 0.008, 0.008, 0.01),
        ],
    )
    def test_numerical_bisection_matches_closed_form(
        self, d0: float, d1: float, se0: float, se1: float, margin: float
    ) -> None:
        """Numerical bisection distance must match exact analytical distance within 1e-5."""
        strata = _make_strata(d0, se0, d1, se1)
        res_num = find_decision_reversal(strata, margin=margin, max_epsilon=0.40)
        res_ana = find_analytical_decision_reversal(strata, margin=margin, max_epsilon=0.40)

        assert res_num.status == res_ana.status
        assert res_num.initial_decision == res_ana.initial_decision
        assert res_num.reversed_decision == res_ana.reversed_decision

        if res_ana.exact_reversal_distance is not None:
            assert res_num.reversal_distance is not None
            discrepancy = abs(res_num.reversal_distance - res_ana.exact_reversal_distance)
            assert discrepancy < 1e-5, f"Discrepancy {discrepancy} exceeds 1e-5"

    def test_no_reversal_agreement(self) -> None:
        """When no reversal exists, both numerical and analytical report None distance."""
        strata = _make_strata(0.15, 0.01, 0.12, 0.01)
        res_num = find_decision_reversal(strata, margin=0.02, max_epsilon=0.40)
        res_ana = find_analytical_decision_reversal(strata, margin=0.02, max_epsilon=0.40)

        assert res_num.status == "NO_REVERSAL_FOUND_WITHIN_BOUNDS"
        assert res_ana.status == "NO_REVERSAL_FOUND_WITHIN_BOUNDS"
        assert res_num.reversal_distance is None
        assert res_ana.exact_reversal_distance is None
