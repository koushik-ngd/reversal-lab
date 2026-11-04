from __future__ import annotations

import math

import numpy as np
import pytest

from reversallab.contracts import FloatArray, LoggedBanditData
from reversallab.evaluation import dm, dr, importance_weights, ips, snips
from reversallab.exceptions import (
    ConfigurationError,
    DataValidationError,
    DegenerateWeightsError,
    InsufficientDataError,
    NumericalError,
    SupportViolationError,
)


def test_weights_hand_computed(hand_data: LoggedBanditData, hand_eval_dist: FloatArray) -> None:
    np.testing.assert_allclose(importance_weights(hand_data, hand_eval_dist), [2.0, 2.5, 2.0, 2.0])


def test_ips_hand_computed(hand_data: LoggedBanditData, hand_eval_dist: FloatArray) -> None:
    # w * r = [2, 0, 2, 2]: mean 1.5, sample sd 1, se 1/sqrt(4).
    est = ips(hand_data, hand_eval_dist)
    assert est.value == pytest.approx(1.5)
    assert est.standard_error == pytest.approx(0.5)
    assert est.ci_lower == pytest.approx(1.5 - 1.959963984540054 * 0.5)
    assert est.ci_upper == pytest.approx(1.5 + 1.959963984540054 * 0.5)
    assert est.n_samples == 4


def test_snips_hand_computed(hand_data: LoggedBanditData, hand_eval_dist: FloatArray) -> None:
    # sum(w r) = 6, sum(w) = 8.5; residuals r - 12/17 give sum (w * res)^2 = 1200 / 289.
    est = snips(hand_data, hand_eval_dist)
    assert est.value == pytest.approx(12 / 17)
    assert est.standard_error == pytest.approx(math.sqrt(1200) / 17 / 8.5)


def test_dm_hand_computed(hand_data: LoggedBanditData, hand_eval_dist: FloatArray) -> None:
    # hand_eval_dist: [[1, 0], [0.5, 0.5], [0.5, 0.5], [0, 1]]
    # q_hat:
    q_hat = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.1, 0.9]])
    # DM round values:
    # row 0: 1.0 * 0.8 + 0.0 * 0.2 = 0.8
    # row 1: 0.5 * 0.6 + 0.5 * 0.4 = 0.5
    # row 2: 0.5 * 0.7 + 0.5 * 0.3 = 0.5
    # row 3: 0.0 * 0.1 + 1.0 * 0.9 = 0.9
    # mean = (0.8 + 0.5 + 0.5 + 0.9) / 4 = 2.7 / 4 = 0.675
    # variance = ((0.8 - 0.675)^2 + 2 * (0.5 - 0.675)^2 + (0.9 - 0.675)^2) / 3
    #          = (0.015625 + 2 * 0.030625 + 0.050625) / 3 = 0.1275 / 3 = 0.0425
    # se = sqrt(0.0425) / 2 = 0.10307764064044151
    est = dm(hand_data, hand_eval_dist, q_hat)
    assert est.value == pytest.approx(0.675)
    assert est.standard_error == pytest.approx(math.sqrt(0.0425) / 2)
    assert est.n_samples == 4


def test_dr_hand_computed(hand_data: LoggedBanditData, hand_eval_dist: FloatArray) -> None:
    # hand_data actions: [0, 1, 0, 1], rewards: [1, 0, 1, 1], w: [2.0, 2.5, 2.0, 2.0]
    q_hat = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.1, 0.9]])
    # dm terms: [0.8, 0.5, 0.5, 0.9]
    # factual q_hat: [0.8, 0.4, 0.7, 0.9]
    # residuals (r - q_hat_factual): [0.2, -0.4, 0.3, 0.1]
    # weighted residuals: [0.4, -1.0, 0.6, 0.2]
    # round values = dm + weighted residual: [1.2, -0.5, 1.1, 1.1]
    # mean = 2.9 / 4 = 0.725
    # sample variance = 2.0075 / 3 = 0.6691666666666667
    # se = sqrt(0.6691666666666667) / 2 = 0.40898858990141384
    est = dr(hand_data, hand_eval_dist, q_hat)
    assert est.value == pytest.approx(0.725)
    assert est.standard_error == pytest.approx(math.sqrt(2.0075 / 3) / 2)
    assert est.n_samples == 4


def test_dr_reduces_to_ips_when_q_hat_zero(
    hand_data: LoggedBanditData, hand_eval_dist: FloatArray
) -> None:
    q_hat_zero = np.zeros((4, 2))
    est_ips = ips(hand_data, hand_eval_dist)
    est_dr = dr(hand_data, hand_eval_dist, q_hat_zero)
    assert est_dr.value == pytest.approx(est_ips.value, abs=1e-14)
    assert est_dr.standard_error == pytest.approx(est_ips.standard_error, abs=1e-14)


def test_dr_matches_dm_when_factual_residuals_zero(
    hand_data: LoggedBanditData, hand_eval_dist: FloatArray
) -> None:
    # If q_hat matches observed rewards exactly for chosen actions, the residual term is zero
    q_hat = np.zeros((4, 2))
    for i in range(4):
        q_hat[i, hand_data.action[i]] = hand_data.reward[i]
    est_dm = dm(hand_data, hand_eval_dist, q_hat)
    est_dr = dr(hand_data, hand_eval_dist, q_hat)
    assert est_dr.value == pytest.approx(est_dm.value, abs=1e-14)


def test_on_policy_ips_equals_sample_mean() -> None:
    rng = np.random.default_rng(0)
    n, k = 300, 4
    dist = rng.dirichlet(np.ones(k), size=n)
    action = np.array([rng.choice(k, p=row) for row in dist])
    reward = rng.random(n)
    data = LoggedBanditData(
        context=np.zeros((n, 1)),
        action=action,
        reward=reward,
        logging_propensity=dist[np.arange(n), action],
        n_actions=k,
        logging_action_dist=dist,
    )
    est_ips, est_snips = ips(data, dist), snips(data, dist)
    assert est_ips.value == pytest.approx(reward.mean(), rel=1e-12)
    assert est_snips.value == pytest.approx(reward.mean(), rel=1e-12)
    assert est_ips.standard_error == pytest.approx(reward.std(ddof=1) / math.sqrt(n), rel=1e-12)


def test_snips_is_translation_equivariant(
    hand_data: LoggedBanditData, hand_eval_dist: FloatArray
) -> None:
    shifted = LoggedBanditData(
        context=hand_data.context,
        action=hand_data.action,
        reward=hand_data.reward + 3.0,
        logging_propensity=hand_data.logging_propensity,
        n_actions=2,
        logging_action_dist=hand_data.logging_action_dist,
    )
    base, moved = snips(hand_data, hand_eval_dist), snips(shifted, hand_eval_dist)
    assert moved.value == pytest.approx(base.value + 3.0)
    assert moved.standard_error == pytest.approx(base.standard_error)


def test_support_violation_raises_when_verifiable() -> None:
    logging_dist = np.array([[1.0, 0.0], [0.5, 0.5]])
    data = LoggedBanditData(
        context=np.zeros((2, 1)),
        action=np.array([0, 1]),
        reward=np.array([1.0, 0.0]),
        logging_propensity=np.array([1.0, 0.5]),
        n_actions=2,
        logging_action_dist=logging_dist,
    )
    eval_dist = np.array([[0.5, 0.5], [0.5, 0.5]])
    q_hat = np.zeros((2, 2))
    for estimator in (ips, snips):
        with pytest.raises(SupportViolationError, match="1 \\(context, action\\) pair"):
            estimator(data, eval_dist)
    with pytest.raises(SupportViolationError, match="1 \\(context, action\\) pair"):
        dr(data, eval_dist, q_hat)


def test_support_unverifiable_without_logging_dist() -> None:
    data = LoggedBanditData(
        context=np.zeros((2, 1)),
        action=np.array([0, 1]),
        reward=np.array([1.0, 0.0]),
        logging_propensity=np.array([1.0, 0.5]),
        n_actions=2,
    )
    est = ips(data, np.array([[0.5, 0.5], [0.5, 0.5]]))
    assert est.value == pytest.approx(0.25)
    est_dr = dr(data, np.array([[0.5, 0.5], [0.5, 0.5]]), np.zeros((2, 2)))
    assert est_dr.value == pytest.approx(0.25)


def test_deterministic_evaluation_policy(hand_data: LoggedBanditData) -> None:
    always_zero = np.array([[1.0, 0.0]] * 4)
    # w = [2, 0, 4, 0]; w r = [2, 0, 4, 0].
    assert ips(hand_data, always_zero).value == pytest.approx(1.5)
    assert snips(hand_data, always_zero).value == pytest.approx(1.0)
    q_hat = np.ones((4, 2)) * 0.5
    assert dm(hand_data, always_zero, q_hat).value == pytest.approx(0.5)


def test_snips_degenerate_weights_raise() -> None:
    data = LoggedBanditData(
        context=np.zeros((3, 1)),
        action=np.array([0, 0, 0]),
        reward=np.ones(3),
        logging_propensity=np.full(3, 0.5),
        n_actions=2,
        logging_action_dist=np.full((3, 2), 0.5),
    )
    never_logged_action = np.array([[0.0, 1.0]] * 3)
    with pytest.raises(DegenerateWeightsError, match="all importance weights are zero"):
        snips(data, never_logged_action)
    assert ips(data, never_logged_action).value == 0.0


@pytest.mark.parametrize(
    ("eval_dist", "match"),
    [
        (np.array([[0.5, 0.6]] * 4), "sum to 1"),
        (np.array([[1.5, -0.5]] * 4), "outside"),
        (np.array([[np.nan, 1.0]] * 4), "non-finite"),
        (np.full((3, 2), 0.5), "shape"),
        (np.full((4, 3), 1 / 3), "shape"),
    ],
)
def test_invalid_evaluation_distribution(
    hand_data: LoggedBanditData, eval_dist: FloatArray, match: str
) -> None:
    for estimator in (ips, snips):
        with pytest.raises(DataValidationError, match=match):
            estimator(hand_data, eval_dist)
    q_hat = np.zeros((4, 2))
    with pytest.raises(DataValidationError, match=match):
        dm(hand_data, eval_dist, q_hat)
    with pytest.raises(DataValidationError, match=match):
        dr(hand_data, eval_dist, q_hat)


@pytest.mark.parametrize(
    ("q_hat", "match"),
    [
        (np.full((3, 2), 0.5), "shape"),
        (np.full((4, 3), 0.5), "shape"),
        (np.array([[np.nan, 0.5]] * 4), "non-finite"),
        (np.array([[np.inf, 0.5]] * 4), "non-finite"),
        (np.zeros((4,)), "2-dimensional"),
    ],
)
def test_dm_dr_invalid_predicted_rewards(
    hand_data: LoggedBanditData, hand_eval_dist: FloatArray, q_hat: FloatArray, match: str
) -> None:
    with pytest.raises(DataValidationError, match=match):
        dm(hand_data, hand_eval_dist, q_hat)
    with pytest.raises(DataValidationError, match=match):
        dr(hand_data, hand_eval_dist, q_hat)


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.1])
def test_invalid_confidence_level(
    hand_data: LoggedBanditData, hand_eval_dist: FloatArray, level: float
) -> None:
    with pytest.raises(ConfigurationError):
        ips(hand_data, hand_eval_dist, confidence_level=level)
    q_hat = np.zeros((4, 2))
    with pytest.raises(ConfigurationError):
        dm(hand_data, hand_eval_dist, q_hat, confidence_level=level)
    with pytest.raises(ConfigurationError):
        dr(hand_data, hand_eval_dist, q_hat, confidence_level=level)


def test_overflowing_importance_weight_is_rejected() -> None:
    propensity = np.array([0.5, np.nextafter(0.0, 1.0)])
    assert propensity[1] > 0.0
    logging_dist = np.column_stack([propensity, 1.0 - propensity])
    data = LoggedBanditData(
        context=np.zeros((2, 1)),
        action=np.array([0, 0]),
        reward=np.ones(2),
        logging_propensity=propensity,
        n_actions=2,
        logging_action_dist=logging_dist,
    )
    evaluation = np.array([[1.0, 0.0], [1.0, 0.0]])
    q_hat = np.zeros((2, 2))
    with pytest.raises(NumericalError, match="non-finite"):
        ips(data, evaluation)
    with pytest.raises(NumericalError, match="non-finite"):
        dr(data, evaluation, q_hat)


def test_single_observation_insufficient() -> None:
    data = LoggedBanditData(
        context=np.zeros((1, 1)),
        action=np.array([0]),
        reward=np.array([1.0]),
        logging_propensity=np.array([0.5]),
        n_actions=2,
    )
    with pytest.raises(InsufficientDataError):
        ips(data, np.array([[0.5, 0.5]]))
    with pytest.raises(InsufficientDataError):
        dm(data, np.array([[0.5, 0.5]]), np.zeros((1, 2)))
    with pytest.raises(InsufficientDataError):
        dr(data, np.array([[0.5, 0.5]]), np.zeros((1, 2)))


def test_continuous_rewards_dm_dr() -> None:
    # Test that DM and DR handle continuous, negative, and unbounded rewards correctly
    data = LoggedBanditData(
        context=np.array([[1.0], [2.0], [3.0]]),
        action=np.array([0, 1, 0]),
        reward=np.array([-5.2, 10.4, -1.1]),
        logging_propensity=np.array([0.5, 0.5, 0.5]),
        n_actions=2,
        logging_action_dist=np.full((3, 2), 0.5),
    )
    eval_dist = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])
    q_hat = np.array([[-5.0, 2.0], [1.0, 10.0], [-1.0, 3.0]])

    est_dm = dm(data, eval_dist, q_hat)
    est_dr = dr(data, eval_dist, q_hat)

    assert math.isfinite(est_dm.value)
    assert math.isfinite(est_dr.value)
    assert est_dm.standard_error > 0.0
    assert est_dr.standard_error > 0.0
