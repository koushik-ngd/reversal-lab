from __future__ import annotations

import numpy as np
import pytest

from reversallab.contracts import FloatArray, LoggedBanditData
from reversallab.evaluation import (
    DiagnosticThresholds,
    diagnose,
    estimator_disagreement,
    ips,
    snips,
    weight_diagnostics,
)


def test_ess_equal_weights() -> None:
    wd = weight_diagnostics(np.full(10, 3.0))
    assert wd.effective_sample_size == pytest.approx(10)
    assert wd.effective_sample_size_fraction == pytest.approx(1.0)
    assert wd.max_weight_share == pytest.approx(0.1)


def test_ess_single_nonzero_weight() -> None:
    wd = weight_diagnostics(np.array([0.0, 0.0, 5.0, 0.0]))
    assert wd.effective_sample_size == pytest.approx(1.0)
    assert wd.max_weight_share == pytest.approx(1.0)
    assert wd.zero_weight_fraction == pytest.approx(0.75)


def test_ess_all_zero() -> None:
    wd = weight_diagnostics(np.zeros(5))
    assert wd.effective_sample_size == 0.0
    assert np.isnan(wd.max_weight_share)


def test_hand_example_report(hand_data: LoggedBanditData, hand_eval_dist: FloatArray) -> None:
    report = diagnose(hand_data, hand_eval_dist)
    assert report.weights.mean == pytest.approx(8.5 / 4)
    assert report.weights.effective_sample_size == pytest.approx(8.5**2 / 18.25)
    assert report.support.support_verifiable
    assert report.support.violations is not None
    assert report.support.violations.n_violating_pairs == 0
    assert report.support.logged_action_counts == [2, 2]
    np.testing.assert_allclose(report.support.evaluation_action_mass, [0.5, 0.5])
    # Four observations: one weight carries 2.5 / 8.5 of the total.
    assert "dominant_weight" in report.warning_codes


def test_support_violation_reported_not_raised() -> None:
    data = LoggedBanditData(
        context=np.zeros((2, 1)),
        action=np.array([0, 0]),
        reward=np.ones(2),
        logging_propensity=np.ones(2),
        n_actions=2,
        logging_action_dist=np.array([[1.0, 0.0]] * 2),
    )
    report = diagnose(data, np.full((2, 2), 0.5))
    assert report.support.violations is not None
    assert report.support.violations.n_violating_pairs == 2
    assert report.support.violations.mean_unsupported_evaluation_mass == pytest.approx(0.5)
    assert {"support_violation", "unlogged_action_mass", "mean_weight_deviation"} <= set(
        report.warning_codes
    )


def test_support_unverifiable_warning() -> None:
    data = LoggedBanditData(
        context=np.zeros((2, 1)),
        action=np.array([0, 1]),
        reward=np.ones(2),
        logging_propensity=np.full(2, 0.5),
        n_actions=2,
    )
    report = diagnose(data, np.full((2, 2), 0.5))
    assert not report.support.support_verifiable
    assert "support_unverifiable" in report.warning_codes


def test_all_weights_zero_warning() -> None:
    data = LoggedBanditData(
        context=np.zeros((3, 1)),
        action=np.zeros(3, dtype=int),
        reward=np.ones(3),
        logging_propensity=np.full(3, 0.5),
        n_actions=2,
        logging_action_dist=np.full((3, 2), 0.5),
    )
    report = diagnose(data, np.array([[0.0, 1.0]] * 3))
    assert "all_weights_zero" in report.warning_codes
    assert "low_effective_sample_size" in report.warning_codes


def test_misspecified_propensities_detected() -> None:
    """Logged propensities that are systematically too small inflate the mean weight."""
    rng = np.random.default_rng(0)
    n = 5000
    action = rng.integers(0, 2, size=n)
    data = LoggedBanditData(
        context=np.zeros((n, 1)),
        action=action,
        reward=np.ones(n),
        logging_propensity=np.full(n, 0.4),
        n_actions=2,  # true propensity is 0.5
    )
    report = diagnose(data, np.full((n, 2), 0.5))
    assert report.weights.mean == pytest.approx(1.25)
    assert "mean_weight_deviation" in report.warning_codes


def test_clean_case_has_no_weight_warnings() -> None:
    rng = np.random.default_rng(1)
    n = 5000
    action = rng.integers(0, 4, size=n)
    data = LoggedBanditData(
        context=np.zeros((n, 1)),
        action=action,
        reward=rng.random(n),
        logging_propensity=np.full(n, 0.25),
        n_actions=4,
        logging_action_dist=np.full((n, 4), 0.25),
    )
    report = diagnose(data, np.tile([0.4, 0.3, 0.2, 0.1], (n, 1)), DiagnosticThresholds())
    assert report.warning_codes == ()


def test_estimator_disagreement(hand_data: LoggedBanditData, hand_eval_dist: FloatArray) -> None:
    estimates = [ips(hand_data, hand_eval_dist), snips(hand_data, hand_eval_dist)]
    assert estimator_disagreement(estimates) == pytest.approx(1.5 - 12 / 17)
    assert np.isnan(estimator_disagreement(estimates[:1]))


def test_diagnose_interval_valid_and_failures() -> None:
    from reversallab.evaluation.diagnostics import diagnose_interval

    # Normal valid interval: width 0.2 < 1.0, se > 0
    w_valid = diagnose_interval((0.4, 0.6), 0.05)
    assert w_valid == []

    # Excessive width: width 1.5 > 1.0
    w_wide = diagnose_interval((0.0, 1.5), 0.3)
    codes_wide = [w.code for w in w_wide]
    assert "excessive_interval_width" in codes_wide

    # Degenerate interval: inverted lower > upper
    w_inv = diagnose_interval((0.8, 0.2), 0.1)
    codes_inv = [w.code for w in w_inv]
    assert "degenerate_interval" in codes_inv

    # Non-finite bounds and uncertainty
    w_nan = diagnose_interval((float("nan"), 1.0), float("nan"))
    codes_nan = [w.code for w in w_nan]
    assert "degenerate_interval" in codes_nan
    assert "nonfinite_uncertainty" in codes_nan


def test_concentrated_weights_and_invalid_inference_warnings() -> None:
    # 100 observations with 1 extreme weight of 1000.0, 99 of 0.01
    w = np.full(100, 0.01)
    w[0] = 1000.0
    wd = weight_diagnostics(w)
    assert wd.top_1pct_weight_share > 0.90
    assert wd.effective_sample_size_fraction < 0.02

    # Fake dataset with extreme weights
    n = 100
    data = LoggedBanditData(
        context=np.zeros((n, 1)),
        action=np.array([0] + [1] * (n - 1), dtype=np.int64),
        reward=np.ones(n),
        logging_propensity=np.array([0.001] + [0.999] * (n - 1)),
        n_actions=2,
        logging_action_dist=np.tile([0.001, 0.999], (n, 1)),
    )
    # Target puts 0.999 on action 0 -> weight = 0.999 / 0.001 = 999
    eval_dist = np.tile([0.999, 0.001], (n, 1))
    report = diagnose(data, eval_dist)
    assert "concentrated_weights" in report.warning_codes
    assert "invalid_inference_assumptions" in report.warning_codes
