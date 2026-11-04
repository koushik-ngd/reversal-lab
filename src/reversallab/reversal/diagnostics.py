r"""Comprehensive diagnostic gate evaluation for off-policy decision-reversal.

A simple heuristic gate based solely on minimum observed logging propensity
:math:`\min_i p_i < \tau` is subject to both false rejections and missed risks:
1. **False Alarm (Inactive Deficit):** An observation has very small logging propensity
   :math:`p_i < \tau`, but the candidate policy assigns zero probability
   :math:`\pi_e(a_i \mid x_i) = 0`. The resulting importance weight is exactly zero
   and does not destabilize estimation variance.
2. **Missed Risk (Active Concentration):** Propensities are safely above :math:`\tau`,
   but the candidate policy repeatedly assigns probability 1.0 to moderately rare
   actions, causing effective sample size collapse (:math:`\mathrm{ESS} / N < 0.05`)
   or extreme weight concentration (:math:`\max_i w_i / \sum w_i > 0.15`).
3. **Unverified Support vs Proven Violation:** When the full logging distribution
   :math:`\pi_0(\cdot \mid x)` is unobserved, support cannot be verified across
   unselected actions. This must be reported as unverified support rather than
   conflated with a proven mathematical support violation.

This module provides a multi-factor diagnostic report that evaluates support,
effective sample size, weight concentration, and active vs inactive propensity deficits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import FloatArray
from reversallab.evaluation.weights import (
    find_support_violations,
    importance_weights,
    validate_evaluation_action_dist,
)


@dataclass(frozen=True)
class DiagnosticGateThresholds:
    """Thresholds for diagnostic gate evaluation."""

    min_propensity: float = 1e-4
    min_ess_fraction: float = 0.05
    max_weight_share: float = 0.20
    require_full_support_verification: bool = False


@dataclass(frozen=True)
class DiagnosticGateReport:
    """Detailed evaluation of operational inference assumptions."""

    passed: bool
    rejection_reasons: list[str]
    warnings: list[str]
    min_propensity: float
    is_propensity_active: bool
    ess_candidate: float
    ess_fraction_candidate: float
    max_weight_share_candidate: float
    ess_baseline: float
    ess_fraction_baseline: float
    max_weight_share_baseline: float
    support_status: str  # "verified_full_support", "proven_support_violation", "unverified_support"
    n_samples: int


def evaluate_diagnostic_gate(
    data: LoggedBanditData,
    candidate_action_dist: FloatArray,
    baseline_action_dist: FloatArray,
    thresholds: DiagnosticGateThresholds | None = None,
) -> DiagnosticGateReport:
    """Evaluate inference assumptions against multi-criteria operational diagnostics."""
    if thresholds is None:
        thresholds = DiagnosticGateThresholds()

    n = data.n_samples
    reasons: list[str] = []
    warnings: list[str] = []

    cand_dist = validate_evaluation_action_dist(data, candidate_action_dist)
    base_dist = validate_evaluation_action_dist(data, baseline_action_dist)

    weights_cand = importance_weights(data, cand_dist, require_support=False)
    weights_base = importance_weights(data, base_dist, require_support=False)

    # 1. Support Analysis
    support_status = "unverified_support"
    if data.logging_action_dist is not None:
        cand_viol = find_support_violations(data, cand_dist)
        base_viol = find_support_violations(data, base_dist)
        cand_n = cand_viol.n_violating_pairs if cand_viol else 0
        base_n = base_viol.n_violating_pairs if base_viol else 0
        if cand_n > 0 or base_n > 0:
            support_status = "proven_support_violation"
            reasons.append(
                f"Proven support violation: candidate ({cand_n}) "
                f"or baseline ({base_n}) places probability mass "
                "where logging policy is zero."
            )
        else:
            support_status = "verified_full_support"
    else:
        warnings.append(
            "Logging action distribution not provided; full policy support is unverified."
        )
        if thresholds.require_full_support_verification:
            reasons.append("Full support verification required but logging distribution omitted.")

    # 2. Minimum Propensity & Active Check
    min_p = float(np.min(data.logging_propensity))
    min_idx = int(np.argmin(data.logging_propensity))
    is_active = float(weights_cand[min_idx]) > 0.0 or float(weights_base[min_idx]) > 0.0

    if min_p < thresholds.min_propensity:
        if is_active:
            reasons.append(
                f"Active propensity deficit: min propensity ({min_p:.2e}) < threshold "
                f"({thresholds.min_propensity:.2e}) on action actively evaluated by policy."
            )
        else:
            warnings.append(
                f"Inactive propensity deficit: min propensity ({min_p:.2e}) < threshold "
                f"({thresholds.min_propensity:.2e}), but policy assigns zero weight to it."
            )

    # 3. Effective Sample Size
    sum_w_c = float(np.sum(weights_cand))
    sum_w2_c = float(np.sum(weights_cand**2))
    ess_c = float((sum_w_c**2) / sum_w2_c) if sum_w2_c > 0 else 0.0
    ess_frac_c = ess_c / n if n > 0 else 0.0

    sum_w_b = float(np.sum(weights_base))
    sum_w2_b = float(np.sum(weights_base**2))
    ess_b = float((sum_w_b**2) / sum_w2_b) if sum_w2_b > 0 else 0.0
    ess_frac_b = ess_b / n if n > 0 else 0.0

    if ess_frac_c < thresholds.min_ess_fraction:
        reasons.append(
            f"ESS fraction collapse (cand): {ess_frac_c:.3f} < {thresholds.min_ess_fraction:.3f}."
        )
    if ess_frac_b < thresholds.min_ess_fraction:
        reasons.append(
            f"ESS fraction collapse (base): {ess_frac_b:.3f} < {thresholds.min_ess_fraction:.3f}."
        )

    # 4. Weight Concentration
    max_share_c = float(np.max(weights_cand) / sum_w_c) if sum_w_c > 0 else 1.0
    max_share_b = float(np.max(weights_base) / sum_w_b) if sum_w_b > 0 else 1.0

    if max_share_c > thresholds.max_weight_share:
        reasons.append(
            f"Excessive candidate weight share: max observation carries {max_share_c:.1%} "
            f"of total weight (> {thresholds.max_weight_share:.1%})."
        )
    if max_share_b > thresholds.max_weight_share:
        reasons.append(
            f"Excessive baseline weight share: max observation carries {max_share_b:.1%} "
            f"of total weight (> {thresholds.max_weight_share:.1%})."
        )

    passed = len(reasons) == 0

    return DiagnosticGateReport(
        passed=passed,
        rejection_reasons=reasons,
        warnings=warnings,
        min_propensity=min_p,
        is_propensity_active=is_active,
        ess_candidate=ess_c,
        ess_fraction_candidate=ess_frac_c,
        max_weight_share_candidate=max_share_c,
        ess_baseline=ess_b,
        ess_fraction_baseline=ess_frac_b,
        max_weight_share_baseline=max_share_b,
        support_status=support_status,
        n_samples=n,
    )
