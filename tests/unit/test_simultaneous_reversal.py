"""Unit tests for simultaneous confidence bands and adaptive search inference."""

from __future__ import annotations

import pytest

from reversallab.reversal.decision import DecisionOutcome
from reversallab.reversal.simultaneous import TwoStratumSimultaneousBand
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


class TestSimultaneousBandProperties:
    """Mathematical verification of simultaneous confidence bands."""

    def test_critical_value_ordering(self) -> None:
        """Simultaneous critical value z_simult must strictly exceed pointwise z."""
        strata = _make_strata(0.08, 0.01, 0.02, 0.01)
        band = TwoStratumSimultaneousBand(strata, confidence_level=0.95)

        # For 0.95 level, pointwise z = 1.96, Bonferroni z_simult = 2.2414
        assert band.z_simult > band.z_pointwise
        assert abs(band.z_pointwise - 1.95996) < 1e-4
        assert abs(band.z_simult - 2.24140) < 1e-4

    def test_simultaneous_band_strictly_wider_than_pointwise(self) -> None:
        """Simultaneous band must strictly contain the pointwise confidence interval."""
        strata = _make_strata(0.10, 0.015, -0.02, 0.015)
        band = TwoStratumSimultaneousBand(strata, confidence_level=0.95)

        for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
            res = band.evaluate_band(q, margin=0.02)
            assert res.simultaneous_ci_lower < res.pointwise_ci_lower
            assert res.simultaneous_ci_upper > res.pointwise_ci_upper
            assert res.band_expansion_ratio > 1.0

    def test_linear_boundary_propagation(self) -> None:
        """Simultaneous bounds must propagate exactly linearly in mixture weight q."""
        strata = _make_strata(0.12, 0.02, 0.04, 0.01)
        band = TwoStratumSimultaneousBand(strata, confidence_level=0.95)

        l0 = band.stratum_0_lower
        l1 = band.stratum_1_lower
        u0 = band.stratum_0_upper
        u1 = band.stratum_1_upper

        for q in [0.0, 0.25, 0.50, 0.75, 1.0]:
            res = band.evaluate_band(q, margin=0.02)
            expected_lower = q * l0 + (1 - q) * l1
            expected_upper = q * u0 + (1 - q) * u1
            assert abs(res.simultaneous_ci_lower - expected_lower) < 1e-12
            assert abs(res.simultaneous_ci_upper - expected_upper) < 1e-12

    def test_endpoint_infimum_theorem(self) -> None:
        """Infimum of linear lower bound over [q_min, q_max] must equal min of endpoints."""
        strata = _make_strata(0.12, 0.02, -0.04, 0.015)
        band = TwoStratumSimultaneousBand(strata, confidence_level=0.95)

        q_min, q_max = 0.10, 0.90
        res_min = band.evaluate_band(q_min, margin=0.02)
        res_max = band.evaluate_band(q_max, margin=0.02)
        min_endpoint = min(res_min.simultaneous_ci_lower, res_max.simultaneous_ci_lower)

        # Check a dense grid inside [q_min, q_max]
        for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            res_q = band.evaluate_band(q, margin=0.02)
            assert res_q.simultaneous_ci_lower >= min_endpoint - 1e-12

    def test_simultaneous_reversal_conservatism(self) -> None:
        """Simultaneous reversal distance must be weakly smaller than pointwise distance.

        Because the simultaneous band is wider, its lower bound is strictly lower,
        meaning it crosses the deployment threshold margin sooner than the pointwise bound.
        """
        strata = _make_strata(0.16, 0.005, -0.04, 0.005)
        band = TwoStratumSimultaneousBand(strata, confidence_level=0.95)

        res = band.find_simultaneous_reversal(margin=0.02, max_epsilon=0.40)
        assert res.initial_decision == DecisionOutcome.RECOMMEND
        assert res.status == "REVERSAL_FOUND"
        assert res.simultaneous_reversal_distance is not None
        assert res.pointwise_reversal_distance is not None
        # Simultaneous distance must be <= pointwise distance
        assert res.simultaneous_reversal_distance <= res.pointwise_reversal_distance + 1e-8


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("sign", [-1, 1])
def test_rounded_crossing_returns_nearest_changed_decision(direction: int, sign: int) -> None:
    margin = 0.02
    means = [margin + sign * 0.14, margin - sign * 0.04]
    if direction == 1:
        means.reverse()
    band = TwoStratumSimultaneousBand(_make_strata(means[0], 0.02, means[1], 0.02))
    result = band.find_simultaneous_reversal(margin, max_epsilon=0.4)
    assert result.simultaneous_reversal_distance == pytest.approx(0.028733030266117)
    assert result.simultaneous_reversed_q0 is not None
    assert direction * (result.simultaneous_reversed_q0 - 0.5) > 0
    assert band.evaluate_band(result.simultaneous_reversed_q0, margin).decision.outcome == (
        result.reversed_decision
    )
    assert result.reversed_decision == DecisionOutcome.INCONCLUSIVE
