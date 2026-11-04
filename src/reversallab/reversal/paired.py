r"""Paired per-observation policy-value difference scoring and inference.

When comparing a candidate policy :math:`\pi_e` against a baseline policy :math:`\pi_b`
on the same logged bandit dataset :math:`\mathcal{D} = \{(x_i, a_i, r_i, p_i)\}_{i=1}^N`,
the estimators :math:`\hat V(\pi_e)` and :math:`\hat V(\pi_b)` share substantial positive
covariance. Assuming independence between the two estimates overstates the uncertainty
of their difference :math:`\Delta = V(\pi_e) - V(\pi_b)`.

This module implements paired per-observation difference scores:
.. math::
    \psi_{\Delta, i} = \psi_{e, i} - \psi_{b, i}

and covariance-aware asymptotic Wald inference:
.. math::
    \widehat{\mathrm{Var}}(\hat \Delta) = \frac{1}{N} \big( s_e^2 + s_b^2 - 2 s_{e, b} \big).

Pre-specification restriction:
Candidate and baseline policies must be fixed prior to evaluation. Selecting a
candidate policy from a family evaluated on the same rewards introduces selection
bias (the winner's curse), which is outside the scope of M4A.
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
from reversallab.evaluation.influence_scores import (
    compute_dm_pseudo_outcomes,
    compute_dr_influence_scores,
    compute_ips_influence_scores,
    compute_snips_influence_scores,
)
from reversallab.evaluation.weights import (
    importance_weights,
    validate_evaluation_action_dist,
)


@dataclass(frozen=True)
class PairedDifferenceResult:
    """Statistical summary of paired policy-value difference evaluation."""

    estimator: str
    diff_scores: FloatArray
    candidate_scores: FloatArray
    baseline_scores: FloatArray
    point_estimate: float
    candidate_point_estimate: float
    baseline_point_estimate: float
    empirical_variance: float
    standard_error: float
    covariance: float
    independent_variance: float
    independent_standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_samples: int
    variance_reduction_pct: float

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Two-sided confidence interval for Delta."""
        return (self.ci_lower, self.ci_upper)


def compute_paired_difference_scores(
    estimator_name: str,
    data: LoggedBanditData,
    candidate_action_dist: FloatArray,
    baseline_action_dist: FloatArray,
    predicted_rewards: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute per-observation paired difference scores :math:`\psi_{\Delta, i}`.

    Parameters
    ----------
    estimator_name:
        One of ``'dr'``, ``'ips'``, ``'dm'``, or ``'snips'``.
    data:
        Validated logged bandit observations.
    candidate_action_dist:
        Candidate policy action distribution of shape ``(N, K)``.
    baseline_action_dist:
        Baseline policy action distribution of shape ``(N, K)``.
    predicted_rewards:
        Predicted reward matrix of shape ``(N, K)`` (required for DM and DR).

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]:
        ``(diff_scores, candidate_scores, baseline_scores)``, each of shape ``(N,)``.
    """
    _require_two_samples(data)
    name = estimator_name.lower().strip()
    cand_dist = validate_evaluation_action_dist(data, candidate_action_dist)
    base_dist = validate_evaluation_action_dist(data, baseline_action_dist)

    if name == "dr":
        if predicted_rewards is None:
            raise ValueError("predicted_rewards is required for Doubly Robust estimation.")
        q_hat = validate_predicted_rewards(
            predicted_rewards, n_samples=data.n_samples, n_actions=data.n_actions
        )
        psi_e = compute_dr_influence_scores(data, cand_dist, q_hat)
        psi_b = compute_dr_influence_scores(data, base_dist, q_hat)
        diff_scores = psi_e - psi_b
        return diff_scores, psi_e, psi_b

    if name == "ips":
        psi_e = compute_ips_influence_scores(data, cand_dist)
        psi_b = compute_ips_influence_scores(data, base_dist)
        diff_scores = psi_e - psi_b
        return diff_scores, psi_e, psi_b

    if name == "dm":
        if predicted_rewards is None:
            raise ValueError("predicted_rewards is required for Direct Method estimation.")
        q_hat = validate_predicted_rewards(
            predicted_rewards, n_samples=data.n_samples, n_actions=data.n_actions
        )
        psi_e = compute_dm_pseudo_outcomes(data, cand_dist, q_hat)
        psi_b = compute_dm_pseudo_outcomes(data, base_dist, q_hat)
        diff_scores = psi_e - psi_b
        return diff_scores, psi_e, psi_b

    if name == "snips":
        weights_e = importance_weights(data, cand_dist)
        weights_b = importance_weights(data, base_dist)
        w_sum_e = float(np.sum(weights_e))
        w_sum_b = float(np.sum(weights_b))
        if w_sum_e <= 0.0 or not np.isfinite(w_sum_e):
            raise ValueError(
                "Candidate policy SNIPS importance weight sum non-positive or non-finite."
            )
        if w_sum_b <= 0.0 or not np.isfinite(w_sum_b):
            raise ValueError(
                "Baseline policy SNIPS importance weight sum non-positive or non-finite."
            )

        val_e = float(np.sum(weights_e * data.reward) / w_sum_e)
        val_b = float(np.sum(weights_b * data.reward) / w_sum_b)
        delta_val = val_e - val_b

        # Uncentered SNIPS score: psi_i = val + (weights / mean_w) * (reward - val)
        # whose sample mean is exactly val.
        scores_e_delta = compute_snips_influence_scores(data, cand_dist)
        scores_b_delta = compute_snips_influence_scores(data, base_dist)
        psi_e = val_e + scores_e_delta
        psi_b = val_b + scores_b_delta
        diff_scores = delta_val + (scores_e_delta - scores_b_delta)
        return diff_scores, psi_e, psi_b

    raise ValueError(f"Unsupported estimator for paired difference: '{estimator_name}'.")


def evaluate_paired_difference(
    estimator_name: str,
    data: LoggedBanditData,
    candidate_action_dist: FloatArray,
    baseline_action_dist: FloatArray,
    predicted_rewards: FloatArray | None = None,
    *,
    confidence_level: float = 0.95,
) -> PairedDifferenceResult:
    r"""Evaluate paired policy-value difference :math:`\Delta = V(\pi_e) - V(\pi_b)` with inference.

    Parameters
    ----------
    estimator_name:
        One of ``'dr'``, ``'ips'``, ``'dm'``, or ``'snips'``.
    data:
        Validated logged bandit observations.
    candidate_action_dist:
        Candidate policy action distribution of shape ``(N, K)``.
    baseline_action_dist:
        Baseline policy action distribution of shape ``(N, K)``.
    predicted_rewards:
        Predicted reward matrix of shape ``(N, K)`` (required for DM and DR).
    confidence_level:
        Nominal coverage level in (0, 1). Default 0.95.

    Returns
    -------
    PairedDifferenceResult:
        Point estimates, paired and independent standard errors, covariance,
        confidence bounds, and variance reduction percentage.
    """
    diff_scores, psi_e, psi_b = compute_paired_difference_scores(
        estimator_name=estimator_name,
        data=data,
        candidate_action_dist=candidate_action_dist,
        baseline_action_dist=baseline_action_dist,
        predicted_rewards=predicted_rewards,
    )

    n = len(diff_scores)
    point_est = float(np.mean(diff_scores))
    cand_est = float(np.mean(psi_e))
    base_est = float(np.mean(psi_b))

    var_diff = float(np.var(diff_scores, ddof=1))
    var_e = float(np.var(psi_e, ddof=1))
    var_b = float(np.var(psi_b, ddof=1))

    # Empirical covariance between scores
    cov_eb = float(np.cov(psi_e, psi_b, ddof=1)[0, 1])

    indep_var = var_e + var_b
    se_diff = float(math.sqrt(var_diff / n))
    se_indep = float(math.sqrt(indep_var / n))

    z = _normal_quantile(confidence_level)
    ci_lower = point_est - z * se_diff
    ci_upper = point_est + z * se_diff

    # Variance change percentage: positive means pairing reduced variance;
    # negative means candidate and baseline have negative covariance, inflating variance.
    reduction_pct = 0.0
    if indep_var > 0:
        reduction_pct = float((1.0 - var_diff / indep_var) * 100.0)

    return PairedDifferenceResult(
        estimator=estimator_name.lower().strip(),
        diff_scores=diff_scores,
        candidate_scores=psi_e,
        baseline_scores=psi_b,
        point_estimate=point_est,
        candidate_point_estimate=cand_est,
        baseline_point_estimate=base_est,
        empirical_variance=var_diff,
        standard_error=se_diff,
        covariance=cov_eb,
        independent_variance=indep_var,
        independent_standard_error=se_indep,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_samples=n,
        variance_reduction_pct=reduction_pct,
    )
