r"""Independent analytical reference solver for 1-D decision-transition boundaries.

In the two-stratum setting, the plug-in policy-value difference under mixture proportion
:math:`q \in [0, 1]` for stratum 0 is:
.. math::
    \hat \Delta(q) = q \hat \Delta_0 + (1 - q) \hat \Delta_1
                   = (\hat \Delta_0 - \hat \Delta_1) q + \hat \Delta_1.

Under independent strata, the sample variance of :math:`\hat \Delta(q)` is:
.. math::
    \mathrm{SE}^2(q) = q^2 \widehat{\mathrm{se}}_0^2 + (1 - q)^2 \widehat{\mathrm{se}}_1^2.

Let :math:`\delta` be the minimum meaningful improvement margin and :math:`z = z_{1-\alpha/2}`.
Candidate confidence-bound crossings satisfy:
.. math::
    [\hat \Delta(q) - \delta]^2 = z^2 \mathrm{SE}^2(q).

Expanding both sides yields a quadratic equation in :math:`q`:
.. math::
    c_2 q^2 + c_1 q + c_0 = 0

where:
.. math::
    A = \hat \Delta_0 - \hat \Delta_1, \qquad B = \hat \Delta_1 - \delta, \\
    V_2 = \widehat{\mathrm{se}}_0^2 + \widehat{\mathrm{se}}_1^2, \qquad
    V_1 = -2 \widehat{\mathrm{se}}_1^2, \qquad V_0 = \widehat{\mathrm{se}}_1^2, \\
    c_2 = A^2 - z^2 V_2, \qquad c_1 = 2 A B - z^2 V_1, \qquad c_0 = B^2 - z^2 V_0.

This module provides an exact closed-form algebraic reference that:
1. Solves for all real roots.
2. Checks the original unsquared lower-bound (:math:`L_\Delta(q) = \delta`) and
   upper-bound (:math:`U_\Delta(q) = \delta`) sign conditions.
3. Restricts roots to the admissible perturbation interval :math:`[q_{\min}, q_{\max}]`.
4. Evaluates decision transitions in infinitesimal neighborhoods :math:`q \pm \eta`.
5. Evaluates interval endpoints.
6. Handles linear degeneracy (:math:`c_2 = 0`), tangencies (:math:`\mathcal{D} = 0`),
   and repeated roots.
7. Distinguishes an attained minimum from an open-boundary infimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reversallab.evaluation.estimators import _normal_quantile
from reversallab.reversal.decision import DecisionOutcome
from reversallab.reversal.strata import TwoStratumMixtureModel


@dataclass(frozen=True)
class AnalyticalCandidateRoot:
    """A mathematically valid root of the confidence-bound equation."""

    q: float
    reversal_distance: float
    bound_type: str  # "lower_bound" (L = delta) or "upper_bound" (U = delta)
    is_tangent: bool
    decision_at_root: DecisionOutcome
    decision_left: DecisionOutcome
    decision_right: DecisionOutcome
    is_transition: bool
    is_attained_minimum: bool


@dataclass(frozen=True)
class AnalyticalReversalResult:
    """Exact analytical decision-reversal result."""

    initial_decision: DecisionOutcome
    reversed_decision: DecisionOutcome | None
    exact_reversal_distance: float | None
    exact_reversed_q0: float | None
    is_attained: bool
    status: str
    transition: str
    candidate_roots: list[AnalyticalCandidateRoot]
    baseline_q0: float
    max_epsilon: float
    margin: float
    confidence_level: float

    @property
    def is_reversed(self) -> bool:
        """True if an admissible decision transition exists."""
        return self.exact_reversal_distance is not None


def solve_quadratic_crossing_roots(
    delta_0: float,
    se_0: float,
    delta_1: float,
    se_1: float,
    margin: float,
    z: float,
) -> list[tuple[float, bool]]:
    """Solve c2 q^2 + c1 q + c0 = 0 and return real roots with tangency flag."""
    a = delta_0 - delta_1
    b = delta_1 - margin

    v2 = se_0**2 + se_1**2
    v1 = -2.0 * (se_1**2)
    v0 = se_1**2

    c2 = a**2 - (z**2) * v2
    c1 = 2.0 * a * b - (z**2) * v1
    c0 = b**2 - (z**2) * v0

    tol = 1e-12

    # Linear degeneracy
    if abs(c2) < tol:
        if abs(c1) < tol:
            return []
        root = -c0 / c1
        return [(root, False)]

    discriminant = c1**2 - 4.0 * c2 * c0

    # No real roots
    if discriminant < -tol:
        return []

    # Tangency / repeated root
    if abs(discriminant) <= tol:
        root = -c1 / (2.0 * c2)
        return [(root, True)]

    # Two distinct real roots
    sqrt_d = math.sqrt(discriminant)
    root1 = (-c1 - sqrt_d) / (2.0 * c2)
    root2 = (-c1 + sqrt_d) / (2.0 * c2)
    return [(root1, False), (root2, False)]


def find_analytical_decision_reversal(
    model: TwoStratumMixtureModel,
    margin: float,
    *,
    max_epsilon: float = 0.40,
    confidence_level: float = 0.95,
    target_outcome: DecisionOutcome | None = None,
    diagnostic_passed: bool = True,
    diagnostic_reason: str | None = None,
    neighborhood_eta: float = 1e-6,
) -> AnalyticalReversalResult:
    """Find the exact minimal reversal distance using closed-form algebra."""
    baseline_q0 = model.baseline_q0
    base_res = model.evaluate_mixture_decision(
        baseline_q0,
        margin,
        confidence_level=confidence_level,
        diagnostic_passed=diagnostic_passed,
        diagnostic_reason=diagnostic_reason,
    )
    initial_decision = base_res.outcome

    if not diagnostic_passed:
        return AnalyticalReversalResult(
            initial_decision=initial_decision,
            reversed_decision=None,
            exact_reversal_distance=None,
            exact_reversed_q0=None,
            is_attained=False,
            status="BASELINE_DIAGNOSTIC_FAILURE",
            transition="NO_REVERSAL",
            candidate_roots=[],
            baseline_q0=baseline_q0,
            max_epsilon=max_epsilon,
            margin=margin,
            confidence_level=confidence_level,
        )

    z = _normal_quantile(confidence_level)
    delta_0 = model.stratum_0.mean_diff
    se_0 = model.stratum_0.standard_error
    delta_1 = model.stratum_1.mean_diff
    se_1 = model.stratum_1.standard_error

    q_min = max(0.0, baseline_q0 - max_epsilon)
    q_max = min(1.0, baseline_q0 + max_epsilon)

    raw_roots = solve_quadratic_crossing_roots(delta_0, se_0, delta_1, se_1, margin, z)

    def is_target(d: DecisionOutcome) -> bool:
        if target_outcome is not None:
            return d == target_outcome
        return d != initial_decision

    valid_candidates: list[AnalyticalCandidateRoot] = []

    for q_root, is_tangent in raw_roots:
        # Check feasibility within declared budget interval
        if not (q_min - 1e-9 <= q_root <= q_max + 1e-9):
            continue

        q_clamped = max(q_min, min(q_max, q_root))
        _pt, _se, ci_lower, ci_upper = model.evaluate_mixture(
            q_clamped, confidence_level=confidence_level
        )

        # Check unsquared lower-bound vs upper-bound condition
        diff_lower = abs(ci_lower - margin)
        diff_upper = abs(ci_upper - margin)
        bound_type = "lower_bound" if diff_lower < diff_upper else "upper_bound"

        res_at = model.evaluate_mixture_decision(
            q_clamped, margin, confidence_level=confidence_level
        )

        # Inspect infinitesimal neighborhoods
        q_left = max(0.0, q_clamped - neighborhood_eta)
        q_right = min(1.0, q_clamped + neighborhood_eta)
        res_l = model.evaluate_mixture_decision(q_left, margin, confidence_level=confidence_level)
        res_r = model.evaluate_mixture_decision(q_right, margin, confidence_level=confidence_level)

        # A transition occurs if either the root itself or an adjacent point changes decision
        is_transition = (
            is_target(res_at.outcome) or is_target(res_l.outcome) or is_target(res_r.outcome)
        )

        dist = abs(q_clamped - baseline_q0)
        is_attained = is_target(res_at.outcome)

        valid_candidates.append(
            AnalyticalCandidateRoot(
                q=q_clamped,
                reversal_distance=dist,
                bound_type=bound_type,
                is_tangent=is_tangent,
                decision_at_root=res_at.outcome,
                decision_left=res_l.outcome,
                decision_right=res_r.outcome,
                is_transition=is_transition,
                is_attained_minimum=is_attained,
            )
        )

    # Also evaluate interval endpoints (q_min, q_max) for open boundaries
    for q_edge in [q_min, q_max]:
        if abs(q_edge - baseline_q0) > 1e-9:
            res_edge = model.evaluate_mixture_decision(
                q_edge, margin, confidence_level=confidence_level
            )
            if is_target(res_edge.outcome):
                valid_candidates.append(
                    AnalyticalCandidateRoot(
                        q=q_edge,
                        reversal_distance=abs(q_edge - baseline_q0),
                        bound_type="interval_endpoint",
                        is_tangent=False,
                        decision_at_root=res_edge.outcome,
                        decision_left=res_edge.outcome,
                        decision_right=res_edge.outcome,
                        is_transition=True,
                        is_attained_minimum=True,
                    )
                )

    # Filter to actual transitions
    transitions = [c for c in valid_candidates if c.is_transition]

    if not transitions:
        return AnalyticalReversalResult(
            initial_decision=initial_decision,
            reversed_decision=None,
            exact_reversal_distance=None,
            exact_reversed_q0=None,
            is_attained=False,
            status="NO_REVERSAL_FOUND_WITHIN_BOUNDS",
            transition="NO_REVERSAL",
            candidate_roots=valid_candidates,
            baseline_q0=baseline_q0,
            max_epsilon=max_epsilon,
            margin=margin,
            confidence_level=confidence_level,
        )

    # Sort by minimum perturbation distance
    transitions.sort(key=lambda c: c.reversal_distance)
    best = transitions[0]

    # Target decision outcome
    if is_target(best.decision_at_root):
        reversed_decision = best.decision_at_root
    elif is_target(best.decision_right):
        reversed_decision = best.decision_right
    else:
        reversed_decision = best.decision_left

    transition_str = f"{initial_decision.value} -> {reversed_decision.value}"

    return AnalyticalReversalResult(
        initial_decision=initial_decision,
        reversed_decision=reversed_decision,
        exact_reversal_distance=best.reversal_distance,
        exact_reversed_q0=best.q,
        is_attained=best.is_attained_minimum,
        status="REVERSAL_FOUND",
        transition=transition_str,
        candidate_roots=valid_candidates,
        baseline_q0=baseline_q0,
        max_epsilon=max_epsilon,
        margin=margin,
        confidence_level=confidence_level,
    )
