r"""Formal deployment decision rule and outcome classifications.

In high-stakes ML policy deployment, choosing whether to replace an existing baseline
policy :math:`\pi_b` with a candidate policy :math:`\pi_e` requires evidence that the
performance improvement exceeds an operational margin :math:`\delta \ge 0`.

Decision Rules:
1. **RECOMMEND**: Sufficient evidence that improvement strictly exceeds the margin:
   .. math::
       L_\Delta > \delta.
2. **HOLD**: Sufficient evidence that improvement strictly fails to reach the margin:
   .. math::
       U_\Delta < \delta.
3. **INCONCLUSIVE**: Insufficient sample power or effect too close to the boundary:
   .. math::
       L_\Delta \le \delta \le U_\Delta
   or when operational diagnostics fail (e.g. propensity collapse, high truncation).

Boundary Convention:
Strict inequalities are required for action. Exact equality with :math:`\delta` is
classified conservatively as **INCONCLUSIVE**.

Comparison with Naive Decision Rule:
A naive rule deploys if :math:`\hat \Delta > \delta` regardless of uncertainty. In small-sample
or high-variance regimes, naive decisions suffer high false-recommendation rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from reversallab.reversal.paired import PairedDifferenceResult


class DecisionOutcome(StrEnum):
    """Possible outcomes of an evidence-based deployment decision."""

    RECOMMEND = "RECOMMEND"
    HOLD = "HOLD"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class DecisionResult:
    """Outcome and explanation of an off-policy deployment decision."""

    outcome: DecisionOutcome
    margin: float
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    reason: str
    diagnostic_passed: bool
    naive_outcome: DecisionOutcome

    @property
    def is_actionable(self) -> bool:
        """True if the decision is clear (RECOMMEND or HOLD), False if INCONCLUSIVE."""
        return self.outcome in (DecisionOutcome.RECOMMEND, DecisionOutcome.HOLD)

    @property
    def naive_matches_evidence(self) -> bool:
        """True if the naive point-estimate rule agreed with the uncertainty-aware rule."""
        return self.outcome == self.naive_outcome


def evaluate_decision(
    ci_lower: float,
    ci_upper: float,
    point_estimate: float,
    margin: float,
    *,
    confidence_level: float = 0.95,
    diagnostic_passed: bool = True,
    diagnostic_reason: str | None = None,
) -> DecisionResult:
    r"""Evaluate a deployment decision against minimum improvement margin :math:`\delta`.

    Parameters
    ----------
    ci_lower:
        Lower bound :math:`L_\Delta` of the :math:`(1 - \alpha)` confidence interval.
    ci_upper:
        Upper bound :math:`U_\Delta` of the :math:`(1 - \alpha)` confidence interval.
    point_estimate:
        Point estimate :math:`\hat \Delta`.
    margin:
        Minimum meaningful improvement margin :math:`\delta \ge 0`.
    confidence_level:
        Confidence level :math:`1 - \alpha` in (0, 1). Default 0.95.
    diagnostic_passed:
        Whether operational diagnostics indicate valid inference assumptions.
    diagnostic_reason:
        Optional description of diagnostic failure if ``diagnostic_passed`` is False.

    Returns
    -------
    DecisionResult:
        Decision outcome, bounds, reason, and naive comparison.
    """
    if ci_lower > ci_upper:
        raise ValueError(f"ci_lower ({ci_lower}) cannot exceed ci_upper ({ci_upper}).")

    # Evaluate naive point estimate
    if point_estimate > margin:
        naive = DecisionOutcome.RECOMMEND
    elif point_estimate < margin:
        naive = DecisionOutcome.HOLD
    else:
        naive = DecisionOutcome.INCONCLUSIVE

    # Diagnostic gate check
    if not diagnostic_passed:
        desc = diagnostic_reason or "Diagnostics indicated invalid inference assumptions."
        return DecisionResult(
            outcome=DecisionOutcome.INCONCLUSIVE,
            margin=margin,
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            reason=f"Diagnostic gate failure: {desc}",
            diagnostic_passed=False,
            naive_outcome=naive,
        )

    # Uncertainty-aware decision rule
    if ci_lower > margin:
        outcome = DecisionOutcome.RECOMMEND
        reason = (
            f"Statistically significant superiority: Lower confidence bound ({ci_lower:.4f}) "
            f"strictly exceeds margin ({margin:.4f})."
        )
    elif ci_upper < margin:
        outcome = DecisionOutcome.HOLD
        reason = (
            f"Statistically significant non-superiority: Upper confidence bound ({ci_upper:.4f}) "
            f"strictly below margin ({margin:.4f})."
        )
    else:
        outcome = DecisionOutcome.INCONCLUSIVE
        if ci_lower == margin or ci_upper == margin:
            reason = (
                f"Confidence bound exactly touches margin ({margin:.4f}); "
                "conservative boundary convention classifies as INCONCLUSIVE."
            )
        else:
            reason = (
                f"Inconclusive evidence: Confidence interval [{ci_lower:.4f}, {ci_upper:.4f}] "
                f"covers margin ({margin:.4f})."
            )

    return DecisionResult(
        outcome=outcome,
        margin=margin,
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        reason=reason,
        diagnostic_passed=True,
        naive_outcome=naive,
    )


def evaluate_decision_from_paired_result(
    result: PairedDifferenceResult,
    margin: float,
    *,
    min_propensity: float | None = None,
    min_propensity_threshold: float = 1e-4,
) -> DecisionResult:
    """Evaluate deployment decision directly from a PairedDifferenceResult."""
    diag_passed = True
    diag_reason = None
    if min_propensity is not None and min_propensity < min_propensity_threshold:
        diag_passed = False
        diag_reason = (
            f"minimum propensity ({min_propensity:.2e}) below threshold "
            f"({min_propensity_threshold:.2e})"
        )

    return evaluate_decision(
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        point_estimate=result.point_estimate,
        margin=margin,
        confidence_level=result.confidence_level,
        diagnostic_passed=diag_passed,
        diagnostic_reason=diag_reason,
    )
