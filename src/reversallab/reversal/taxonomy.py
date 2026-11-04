r"""Taxonomy and separation of evidence reversals from true policy-value reversals.

Decision-reversal analysis distinguishes three fundamentally different reversal events:

1. **Evidence Reversal (:math:`d_{\mathrm{ev}}^*`):**
   The sample-based confidence-bound deployment decision changes (e.g. from
   ``RECOMMEND`` to ``INCONCLUSIVE``). This reflects a loss of statistical certitude
   under demographic shift. The true expected policy advantage may still exceed
   :math:`\delta`!
   - Recommendation Withdrawal: ``RECOMMEND -> INCONCLUSIVE`` (uncertainty spans margin).
   - Recommendation Inversion: ``RECOMMEND -> HOLD`` (challenger significantly worse).
   - Recommendation Emergence: ``INCONCLUSIVE -> RECOMMEND`` (subgroup shift favors candidate).
   - Rejection Emergence: ``INCONCLUSIVE -> HOLD`` (subgroup shift confirms inferiority).

2. **Margin Reversal (:math:`d_{\mathrm{margin}}^*`):**
   The true population policy advantage :math:`\Delta_{\text{true}}(q)` crosses the
   meaningful improvement threshold :math:`\delta`. The candidate policy is no longer
   operationally worthwhile, but may still be slightly better than baseline.

3. **Performance Reversal (:math:`d_{\mathrm{perf}}^*`):**
   The true population policy advantage :math:`\Delta_{\text{true}}(q)` crosses zero.
   The candidate policy actively harms users compared to the baseline policy.

Conflating an evidence reversal with a performance reversal is a critical category error:
losing statistical significance does not imply the candidate has become harmful.
"""

from __future__ import annotations

from dataclasses import dataclass

from reversallab.reversal.decision import DecisionOutcome


@dataclass(frozen=True)
class ReversalTaxonomyReport:
    """Taxonomy report separating evidence, margin, and performance reversals."""

    initial_decision: DecisionOutcome
    reversed_decision: DecisionOutcome | None
    transition_category: str
    evidence_reversal_distance: float | None
    margin_reversal_distance: float | None
    performance_reversal_distance: float | None
    has_evidence_reversal: bool
    has_margin_reversal: bool
    has_performance_reversal: bool
    true_delta_baseline: float
    true_delta_stratum_0: float
    true_delta_stratum_1: float
    margin: float
    explanation: str


def classify_reversal_taxonomy(
    initial_decision: DecisionOutcome,
    reversed_decision: DecisionOutcome | None,
    evidence_distance: float | None,
    true_delta_0: float,
    true_delta_1: float,
    margin: float,
    *,
    baseline_q0: float = 0.50,
    max_epsilon: float = 0.40,
) -> ReversalTaxonomyReport:
    """Classify and separate evidence, margin, and performance reversals."""
    # True delta under mixture q:
    # Delta*(q) = q delta_0 + (1-q) delta_1 = (delta_0 - delta_1) q + delta_1
    denom = true_delta_0 - true_delta_1
    true_delta_base = baseline_q0 * true_delta_0 + (1.0 - baseline_q0) * true_delta_1

    q_min = max(0.0, baseline_q0 - max_epsilon)
    q_max = min(1.0, baseline_q0 + max_epsilon)

    # 1. Margin crossing: Delta*(q) = margin
    d_margin: float | None = None
    if abs(denom) > 1e-12:
        q_m = (margin - true_delta_1) / denom
        if q_min <= q_m <= q_max:
            d_margin = float(abs(q_m - baseline_q0))

    # 2. Performance crossing: Delta*(q) = 0.0
    d_perf: float | None = None
    if abs(denom) > 1e-12:
        q_p = (0.0 - true_delta_1) / denom
        if q_min <= q_p <= q_max:
            d_perf = float(abs(q_p - baseline_q0))

    # 3. Transition categorization
    if reversed_decision is None:
        cat = "no_reversal"
    elif (
        initial_decision == DecisionOutcome.RECOMMEND
        and reversed_decision == DecisionOutcome.INCONCLUSIVE
    ):
        cat = "recommendation_withdrawal"
    elif (
        initial_decision == DecisionOutcome.RECOMMEND and reversed_decision == DecisionOutcome.HOLD
    ):
        cat = "recommendation_inversion"
    elif (
        initial_decision == DecisionOutcome.HOLD
        and reversed_decision == DecisionOutcome.INCONCLUSIVE
    ):
        cat = "rejection_withdrawal"
    elif (
        initial_decision == DecisionOutcome.HOLD and reversed_decision == DecisionOutcome.RECOMMEND
    ):
        cat = "rejection_inversion"
    elif (
        initial_decision == DecisionOutcome.INCONCLUSIVE
        and reversed_decision == DecisionOutcome.RECOMMEND
    ):
        cat = "recommendation_emergence"
    elif (
        initial_decision == DecisionOutcome.INCONCLUSIVE
        and reversed_decision == DecisionOutcome.HOLD
    ):
        cat = "rejection_emergence"
    else:
        cat = "unclassified_transition"

    # Explanation construction
    has_ev = evidence_distance is not None
    has_mg = d_margin is not None
    has_pf = d_perf is not None

    if cat == "recommendation_withdrawal":
        if has_pf:
            exp = (
                f"Evidence reversal at d={evidence_distance:.4f} withdraws recommendation. "
                f"Under larger shift (d={d_perf:.4f}), policy suffers genuine performance reversal."
            )
        elif has_mg:
            exp = (
                f"Evidence reversal at d={evidence_distance:.4f} withdraws recommendation. "
                f"True improvement drops below margin delta at d={d_margin:.4f}, but policy "
                "remains strictly superior (no performance reversal)."
            )
        else:
            exp = (
                f"Statistical evidence reversal at d={evidence_distance:.4f} withdraws certitude, "
                "but policy remains genuinely superior across all admissible shifts."
            )
    elif cat == "recommendation_emergence":
        exp = (
            f"Under subpopulation shift (d={evidence_distance:.4f}), statistical evidence becomes "
            "sufficient to positively recommend candidate deployment."
        )
    elif cat == "no_reversal":
        exp = "Decision is robust; no evidence reversal occurs within admissible bounds."
    else:
        rev_val = reversed_decision.value if reversed_decision else "None"
        exp = f"Transition: {initial_decision.value} -> {rev_val}."

    return ReversalTaxonomyReport(
        initial_decision=initial_decision,
        reversed_decision=reversed_decision,
        transition_category=cat,
        evidence_reversal_distance=evidence_distance,
        margin_reversal_distance=d_margin,
        performance_reversal_distance=d_perf,
        has_evidence_reversal=has_ev,
        has_margin_reversal=has_mg,
        has_performance_reversal=has_pf,
        true_delta_baseline=true_delta_base,
        true_delta_stratum_0=true_delta_0,
        true_delta_stratum_1=true_delta_1,
        margin=margin,
        explanation=exp,
    )
