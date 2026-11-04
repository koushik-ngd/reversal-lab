"""Unit tests for per-observation influence scores and asymptotic Wald inference."""

from __future__ import annotations

import numpy as np
import pytest

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.evaluation.estimators import dm, dr, ips, snips
from reversallab.evaluation.influence_scores import (
    compute_dm_pseudo_outcomes,
    compute_dr_influence_scores,
    compute_ips_influence_scores,
    compute_snips_influence_scores,
    evaluate_influence_inference,
)
from reversallab.exceptions import DataValidationError, SupportViolationError


@pytest.fixture
def hand_data() -> tuple[LoggedBanditData, np.ndarray, np.ndarray]:
    """Three observations, two actions. Hand-computed test case."""
    context = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    action = np.array([0, 1, 0], dtype=np.int64)
    reward = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    propensity = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    logging_dist = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64)

    data = LoggedBanditData(
        context=context,
        action=action,
        reward=reward,
        logging_propensity=propensity,
        n_actions=2,
        logging_action_dist=logging_dist,
    )
    # Target policy: action 0 with prob 0.8, action 1 with prob 0.2
    eval_dist = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]], dtype=np.float64)

    # Predicted reward matrix Q:
    # obs 0: [0.6, 0.4] -> m(x_0) = 0.8*0.6 + 0.2*0.4 = 0.48 + 0.08 = 0.56
    # obs 1: [0.5, 0.5] -> m(x_1) = 0.8*0.5 + 0.2*0.5 = 0.50
    # obs 2: [0.7, 0.3] -> m(x_2) = 0.8*0.7 + 0.2*0.3 = 0.56 + 0.06 = 0.62
    q_hat = np.array([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]], dtype=np.float64)

    return data, eval_dist, q_hat


def test_hand_computed_dr_influence_scores(
    hand_data: tuple[LoggedBanditData, np.ndarray, np.ndarray],
) -> None:
    data, eval_dist, q_hat = hand_data

    # obs 0: action=0, r=1.0, w=0.8/0.5=1.6. residual = 1.0 - 0.6 = 0.4.
    # psi_0 = 0.56 + 1.6 * 0.4 = 0.56 + 0.64 = 1.20
    # obs 1: action=1, r=0.0, w=0.2/0.5=0.4. residual = 0.0 - 0.5 = -0.5.
    # psi_1 = 0.50 + 0.4 * (-0.5) = 0.50 - 0.20 = 0.30
    # obs 2: action=0, r=1.0, w=0.8/0.5=1.6. residual = 1.0 - 0.7 = 0.3.
    # psi_2 = 0.62 + 1.6 * 0.3 = 0.62 + 0.48 = 1.10
    expected_scores = np.array([1.20, 0.30, 1.10], dtype=np.float64)

    scores = compute_dr_influence_scores(data, eval_dist, q_hat)
    np.testing.assert_allclose(scores, expected_scores, atol=1e-12)

    # Check that mean equals DR point estimate from estimators.py
    est_dr = dr(data, eval_dist, q_hat)
    assert np.isclose(np.mean(scores), est_dr.value, atol=1e-12)

    # Check inference result
    res = evaluate_influence_inference("dr", data, eval_dist, q_hat)
    assert np.isclose(res.point_estimate, est_dr.value, atol=1e-12)
    assert np.isclose(res.standard_error, est_dr.standard_error, atol=1e-12)
    assert np.isclose(res.ci_lower, est_dr.ci_lower, atol=1e-12)
    assert np.isclose(res.ci_upper, est_dr.ci_upper, atol=1e-12)


def test_hand_computed_dm_pseudo_outcomes(
    hand_data: tuple[LoggedBanditData, np.ndarray, np.ndarray],
) -> None:
    data, eval_dist, q_hat = hand_data
    expected_scores = np.array([0.56, 0.50, 0.62], dtype=np.float64)

    scores = compute_dm_pseudo_outcomes(data, eval_dist, q_hat)
    np.testing.assert_allclose(scores, expected_scores, atol=1e-12)

    est_dm = dm(data, eval_dist, q_hat)
    assert np.isclose(np.mean(scores), est_dm.value, atol=1e-12)

    res = evaluate_influence_inference("dm", data, eval_dist, q_hat)
    assert np.isclose(res.point_estimate, est_dm.value, atol=1e-12)
    assert np.isclose(res.standard_error, est_dm.standard_error, atol=1e-12)


def test_hand_computed_ips_influence_scores(
    hand_data: tuple[LoggedBanditData, np.ndarray, np.ndarray],
) -> None:
    data, eval_dist, _ = hand_data
    # w = [1.6, 0.4, 1.6], r = [1.0, 0.0, 1.0]
    expected_scores = np.array([1.6, 0.0, 1.6], dtype=np.float64)

    scores = compute_ips_influence_scores(data, eval_dist)
    np.testing.assert_allclose(scores, expected_scores, atol=1e-12)

    est_ips = ips(data, eval_dist)
    assert np.isclose(np.mean(scores), est_ips.value, atol=1e-12)

    res = evaluate_influence_inference("ips", data, eval_dist)
    assert np.isclose(res.point_estimate, est_ips.value, atol=1e-12)
    assert np.isclose(res.standard_error, est_ips.standard_error, atol=1e-12)


def test_hand_computed_snips_influence_scores(
    hand_data: tuple[LoggedBanditData, np.ndarray, np.ndarray],
) -> None:
    data, eval_dist, _ = hand_data
    est_sn = snips(data, eval_dist)

    scores = compute_snips_influence_scores(data, eval_dist)
    assert len(scores) == len(data.action)
    res = evaluate_influence_inference("snips", data, eval_dist)

    assert np.isclose(res.point_estimate, est_sn.value, atol=1e-12)
    assert np.isclose(res.standard_error, est_sn.standard_error, atol=1e-12)
    assert np.isclose(res.ci_lower, est_sn.ci_lower, atol=1e-12)
    assert np.isclose(res.ci_upper, est_sn.ci_upper, atol=1e-12)


def test_influence_scores_support_violation() -> None:
    bad_data = LoggedBanditData(
        context=np.array([[1.0], [2.0]]),
        action=np.array([0, 0]),
        reward=np.array([1.0, 1.0]),
        logging_propensity=np.array([1.0, 1.0]),
        n_actions=2,
        logging_action_dist=np.array([[1.0, 0.0], [1.0, 0.0]]),
    )
    unsupported_eval = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(SupportViolationError):
        compute_dr_influence_scores(bad_data, unsupported_eval, np.ones((2, 2)))


def test_influence_scores_invalid_inputs(
    hand_data: tuple[LoggedBanditData, np.ndarray, np.ndarray],
) -> None:
    data, eval_dist, _ = hand_data
    with pytest.raises(DataValidationError):
        compute_dr_influence_scores(data, eval_dist, np.ones((2, 2)))  # wrong shape

    with pytest.raises(ValueError, match="predicted_rewards is required"):
        evaluate_influence_inference("dr", data, eval_dist, predicted_rewards=None)

    with pytest.raises(ValueError, match="unknown estimator"):
        evaluate_influence_inference("unknown", data, eval_dist)
