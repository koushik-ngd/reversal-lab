r"""Influence-score-based uncertainty and asymptotic inference for off-policy estimators.

For logged contextual-bandit observations :math:`\{(x_i, a_i, r_i, p_i)\}_{i=1}^N` drawn
i.i.d. from logging policy :math:`\pi_0`, candidate policy :math:`\pi_e`, importance
weights :math:`w_i = \pi_e(a_i \mid x_i) / p_i`, and predicted rewards :math:`\hat{q}(x, a)`:

1. Doubly Robust (DR) per-observation uncentered influence score:
   .. math::
       \psi_i^{\mathrm{DR}} = \sum_{a=1}^K \pi_e(a \mid x_i) \, \hat{q}(x_i, a)
       + w_i \big(r_i - \hat{q}(x_i, a_i)\big).

   Under Neyman orthogonality and :math:`K`-fold cross-fitting, when nuisance estimates
   converge at rate :math:`o_P(N^{-1/4})` and overlap holds, the sample variance of
   these scores provides an asymptotically valid estimate of the semiparametric
   variance bound.

2. Direct Method (DM) per-observation pseudo-outcome:
   .. math::
       \psi_i^{\mathrm{DM}} = \sum_{a=1}^K \pi_e(a \mid x_i) \, \hat{q}(x_i, a).

   .. note::
       The sample variance of :math:`\psi_i^{\mathrm{DM}}` captures *only* the context-sampling
       variation :math:`\mathrm{Var}_x(\mathbb{E}_{\pi_e}[q(x, a)])`. It treats :math:`\hat{q}`
       as fixed and ignores the estimation uncertainty of the reward model. Consequently,
       asymptotic Wald intervals using this variance under-cover in finite samples.

3. Inverse Probability Scoring (IPS):
   .. math::
       \psi_i^{\mathrm{IPS}} = w_i \, r_i.

4. Self-Normalized IPS (SNIPS):
   .. math::
       \psi_i^{\mathrm{SNIPS}} = \frac{w_i}{\bar{w}} \, (r_i - \hat V_{\mathrm{SNIPS}}),
       \qquad \bar{w} = \frac{1}{N} \sum_{j=1}^N w_j.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import (
    FloatArray,
    validate_predicted_rewards,
)
from reversallab.evaluation.estimators import (
    _normal_quantile,
    _require_two_samples,
)
from reversallab.evaluation.weights import (
    importance_weights,
    validate_evaluation_action_dist,
)
from reversallab.exceptions import DegenerateWeightsError


@dataclass(frozen=True)
class InfluenceScoreResult:
    """Summary of influence-score-based inference for an off-policy estimator."""

    estimator: str
    scores: FloatArray
    point_estimate: float
    empirical_variance: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return (self.ci_lower, self.ci_upper)


def compute_dr_influence_scores(
    data: LoggedBanditData,
    evaluation_action_dist: FloatArray,
    predicted_rewards: FloatArray,
) -> FloatArray:
    r"""Compute per-observation uncentered influence scores for Doubly Robust.

    .. math::
        \psi_i = \sum_{a=1}^K \pi_e(a \mid x_i) \hat{q}(x_i, a)
        + \frac{\pi_e(a_i \mid x_i)}{p_i} \big(r_i - \hat{q}(x_i, a_i)\big).

    Parameters
    ----------
    data:
        Validated logged bandit observations.
    evaluation_action_dist:
        Candidate policy action distribution of shape ``(N, K)``.
    predicted_rewards:
        Predicted reward matrix of shape ``(N, K)``.

    Returns
    -------
    FloatArray:
        1D array of influence scores of shape ``(N,)``.
    """
    eval_dist = validate_evaluation_action_dist(data, evaluation_action_dist)
    q_hat = validate_predicted_rewards(
        predicted_rewards, n_samples=data.n_samples, n_actions=data.n_actions
    )
    weights = importance_weights(data, eval_dist)
    exp_per_obs = np.sum(eval_dist * q_hat, axis=1)
    chosen_q = q_hat[np.arange(data.n_samples), data.action]
    residuals = data.reward - chosen_q
    scores: FloatArray = exp_per_obs + weights * residuals
    return scores


def compute_dm_pseudo_outcomes(
    data: LoggedBanditData,
    evaluation_action_dist: FloatArray,
    predicted_rewards: FloatArray,
) -> FloatArray:
    r"""Compute per-observation conditional expected rewards for Direct Method.

    .. math::
        m(x_i) = \sum_{a=1}^K \pi_e(a \mid x_i) \hat{q}(x_i, a).

    .. warning::
        This captures context-sampling variation only and does NOT account for
        uncertainty from fitting the reward model :math:`\hat{q}`.

    Parameters
    ----------
    data:
        Validated logged bandit observations.
    evaluation_action_dist:
        Candidate policy action distribution of shape ``(N, K)``.
    predicted_rewards:
        Predicted reward matrix of shape ``(N, K)``.

    Returns
    -------
    FloatArray:
        1D array of expected rewards of shape ``(N,)``.
    """
    eval_dist = validate_evaluation_action_dist(data, evaluation_action_dist)
    q_hat = validate_predicted_rewards(
        predicted_rewards, n_samples=data.n_samples, n_actions=data.n_actions
    )
    scores: FloatArray = np.sum(eval_dist * q_hat, axis=1)
    return scores


def compute_ips_influence_scores(
    data: LoggedBanditData,
    evaluation_action_dist: FloatArray,
) -> FloatArray:
    r"""Compute per-observation uncentered influence scores for IPS.

    .. math::
        \psi_i = w_i \, r_i.
    """
    eval_dist = validate_evaluation_action_dist(data, evaluation_action_dist)
    weights = importance_weights(data, eval_dist)
    scores: FloatArray = weights * data.reward
    return scores


def compute_snips_influence_scores(
    data: LoggedBanditData,
    evaluation_action_dist: FloatArray,
) -> FloatArray:
    r"""Compute per-observation delta-method influence scores for SNIPS.

    .. math::
        \psi_i = \frac{w_i}{\bar{w}} \, (r_i - \hat V_{\mathrm{SNIPS}}).
    """
    eval_dist = validate_evaluation_action_dist(data, evaluation_action_dist)
    weights = importance_weights(data, eval_dist)
    w_sum = float(np.sum(weights))
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        raise DegenerateWeightsError(
            f"sum of importance weights is non-positive or non-finite: {w_sum}"
        )
    val = float(np.sum(weights * data.reward) / w_sum)
    mean_w = w_sum / data.n_samples
    scores: FloatArray = (weights / mean_w) * (data.reward - val)
    return scores


def infer_from_influence_scores(
    estimator_name: str,
    scores: FloatArray,
    *,
    confidence_level: float = 0.95,
) -> InfluenceScoreResult:
    """Construct asymptotic Wald inference from an array of per-observation influence scores."""
    z = _normal_quantile(confidence_level)
    n = len(scores)
    if n < 2:
        raise ValueError(f"at least 2 observations required for inference; got {n}")

    point_estimate = float(np.mean(scores))
    var = float(np.var(scores, ddof=1))
    se = float(math.sqrt(var / n))
    ci_lower = point_estimate - z * se
    ci_upper = point_estimate + z * se

    return InfluenceScoreResult(
        estimator=estimator_name,
        scores=scores,
        point_estimate=point_estimate,
        empirical_variance=var,
        standard_error=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
    )


def evaluate_influence_inference(
    estimator_name: str,
    data: LoggedBanditData,
    evaluation_action_dist: FloatArray,
    predicted_rewards: FloatArray | None = None,
    *,
    confidence_level: float = 0.95,
) -> InfluenceScoreResult:
    """Compute per-observation scores and construct asymptotic Wald inference.

    Parameters
    ----------
    estimator_name:
        One of ``'ips'``, ``'snips'``, ``'dm'``, or ``'dr'``.
    data:
        Logged contextual bandit batch.
    evaluation_action_dist:
        Evaluation policy distribution of shape ``(N, K)``.
    predicted_rewards:
        Predicted reward matrix of shape ``(N, K)`` (required for DM and DR).
    confidence_level:
        Nominal coverage level in (0, 1). Default 0.95.
    """
    _require_two_samples(data)
    name = estimator_name.lower().strip()

    if name == "ips":
        scores = compute_ips_influence_scores(data, evaluation_action_dist)
        return infer_from_influence_scores(name, scores, confidence_level=confidence_level)

    if name == "snips":
        scores = compute_snips_influence_scores(data, evaluation_action_dist)
        # Note: SNIPS point estimate is mean(w*r)/mean(w), delta scores have sample mean 0
        eval_dist = validate_evaluation_action_dist(data, evaluation_action_dist)
        weights = importance_weights(data, eval_dist)
        w_sum = float(np.sum(weights))
        val = float(np.sum(weights * data.reward) / w_sum)
        z = _normal_quantile(confidence_level)
        var = float(np.var(scores, ddof=1))
        se = float(math.sqrt(np.sum(weights**2 * (data.reward - val) ** 2)) / w_sum)
        ci_lower = val - z * se
        ci_upper = val + z * se
        return InfluenceScoreResult(
            estimator=name,
            scores=scores,
            point_estimate=val,
            empirical_variance=var,
            standard_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
        )

    if name in {"dm", "dr"}:
        if predicted_rewards is None:
            raise ValueError(f"predicted_rewards is required for estimator '{name}'")
        if name == "dm":
            scores = compute_dm_pseudo_outcomes(data, evaluation_action_dist, predicted_rewards)
            return infer_from_influence_scores(name, scores, confidence_level=confidence_level)
        scores = compute_dr_influence_scores(data, evaluation_action_dist, predicted_rewards)
        return infer_from_influence_scores(name, scores, confidence_level=confidence_level)

    raise ValueError(f"unknown estimator: '{estimator_name}'; expected ips, snips, dm, or dr")
