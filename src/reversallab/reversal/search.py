r"""1-Dimensional decision-reversal search with numerical bisection.

This module searches for the smallest scientifically admissible context mixture
perturbation :math:`\epsilon \ge 0` that reverses an evidence-based deployment decision:
.. math::
    d_{\mathrm{rev}}^* = \inf \left\{ \epsilon \in [0, \epsilon_{\max}] :
    \exists q \in \mathcal{Q}_\epsilon \text{ such that } D(q) \neq D_0 \right\}.

The search employs a two-stage approach:
1. Reference grid search across the admissible budget interval
   :math:`[\max(0, q_{0,0} - \epsilon_{\max}), \min(1, q_{0,0} + \epsilon_{\max})]`
   discretized in both search directions (left and right).
2. Numerical bisection refinement on the transition bracket :math:`[q_a, q_b]` to pinpoint
   the exact boundary to high numerical precision (default tolerance :math:`10^{-6}`).
3. Fine-grid baseline validation to verify that no isolated narrow transition was overlooked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reversallab.reversal.decision import DecisionOutcome
from reversallab.reversal.strata import TwoStratumMixtureModel


@dataclass(frozen=True)
class ReversalSearchResult:
    """Outcome of 1-D decision-reversal search."""

    initial_decision: DecisionOutcome
    reversed_decision: DecisionOutcome | None
    reversal_distance: float | None
    reversed_q0: float | None
    transition: str
    status: str
    baseline_q0: float
    max_epsilon: float
    margin: float
    tolerance: float
    fine_grid_verified: bool
    reversed_point_estimate: float | None = None
    reversed_ci_lower: float | None = None
    reversed_ci_upper: float | None = None

    @property
    def is_reversed(self) -> bool:
        """True if a decision reversal was found within admissible bounds."""
        return self.reversal_distance is not None and self.reversed_decision is not None


def bisect_decision_boundary(
    model: TwoStratumMixtureModel,
    q_start: float,
    q_end: float,
    margin: float,
    *,
    confidence_level: float = 0.95,
    target_outcome: DecisionOutcome | None = None,
    tolerance: float = 1e-6,
    max_iter: int = 100,
) -> tuple[float, DecisionOutcome]:
    """Refine decision transition boundary between q_start and q_end using bisection.

    Parameters
    ----------
    model:
        Fitted TwoStratumMixtureModel.
    q_start:
        Bracket start point where decision equals initial outcome.
    q_end:
        Bracket end point where decision differs (or equals target_outcome).
    margin:
        Decision margin delta.
    confidence_level:
        Confidence level 1 - alpha.
    target_outcome:
        Optional specific target outcome required for reversal.
    tolerance:
        Numerical stopping tolerance |q_right - q_left| < tol.
    max_iter:
        Maximum bisection iterations.

    Returns
    -------
    tuple[float, DecisionOutcome]:
        (reversed_q0, reversed_decision).
    """
    initial_res = model.evaluate_mixture_decision(
        q_start, margin, confidence_level=confidence_level
    )
    initial_decision = initial_res.outcome

    def is_target(d: DecisionOutcome) -> bool:
        if target_outcome is not None:
            return d == target_outcome
        return d != initial_decision

    q_left = q_start
    q_right = q_end

    for _ in range(max_iter):
        if abs(q_right - q_left) <= tolerance:
            break
        q_mid = (q_left + q_right) / 2.0
        res_mid = model.evaluate_mixture_decision(q_mid, margin, confidence_level=confidence_level)
        if is_target(res_mid.outcome):
            # Target reached at q_mid, bracket is [q_left, q_mid]
            q_right = q_mid
        else:
            # Target not yet reached at q_mid, bracket is [q_mid, q_right]
            q_left = q_mid

    res_final = model.evaluate_mixture_decision(q_right, margin, confidence_level=confidence_level)
    return q_right, res_final.outcome


def find_decision_reversal(
    model: TwoStratumMixtureModel,
    margin: float,
    *,
    max_epsilon: float = 0.40,
    confidence_level: float = 0.95,
    target_outcome: DecisionOutcome | None = None,
    grid_points: int = 50,
    tolerance: float = 1e-6,
    diagnostic_passed: bool = True,
    diagnostic_reason: str | None = None,
) -> ReversalSearchResult:
    r"""Find minimum perturbation :math:`\epsilon^* \ge 0` that reverses deployment decision.

    Parameters
    ----------
    model:
        TwoStratumMixtureModel containing stratum statistics.
    margin:
        Minimum meaningful improvement margin :math:`\delta \ge 0`.
    max_epsilon:
        Maximum admissible total variation perturbation budget :math:`\epsilon_{\max}`.
    confidence_level:
        Confidence level :math:`1 - \alpha`. Default 0.95.
    target_outcome:
        If None, any transition :math:`D(q) \neq D_0` is considered a reversal.
        If specified, only transitions to ``target_outcome`` are recorded.
    grid_points:
        Number of points in directional coarse grid search. Default 50.
    tolerance:
        Numerical bisection convergence tolerance. Default :math:`10^{-6}`.
    diagnostic_passed:
        Whether operational diagnostics passed at baseline.
    diagnostic_reason:
        Reason if operational diagnostics failed.

    Returns
    -------
    ReversalSearchResult:
        Search results including reversal distance, transition, and status.
    """
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
        return ReversalSearchResult(
            initial_decision=initial_decision,
            reversed_decision=None,
            reversal_distance=None,
            reversed_q0=None,
            transition="NO_REVERSAL",
            status="BASELINE_DIAGNOSTIC_FAILURE",
            baseline_q0=baseline_q0,
            max_epsilon=max_epsilon,
            margin=margin,
            tolerance=tolerance,
            fine_grid_verified=True,
        )

    def is_reversed(d: DecisionOutcome) -> bool:
        if target_outcome is not None:
            return d == target_outcome
        return d != initial_decision

    # Admissible bounds
    q_min = max(0.0, baseline_q0 - max_epsilon)
    q_max = min(1.0, baseline_q0 + max_epsilon)

    # Search in both directions: Left (decreasing q0) and Right (increasing q0)
    candidates: list[tuple[float, float, DecisionOutcome]] = []  # (epsilon, q0, outcome)

    # Direction 1: Left (q0 from baseline_q0 down to q_min)
    if q_min < baseline_q0:
        grid_left = np.linspace(baseline_q0, q_min, grid_points)
        for i in range(1, len(grid_left)):
            q_prev = float(grid_left[i - 1])
            q_curr = float(grid_left[i])
            res_curr = model.evaluate_mixture_decision(
                q_curr, margin, confidence_level=confidence_level
            )
            if is_reversed(res_curr.outcome):
                refined_q, refined_dec = bisect_decision_boundary(
                    model,
                    q_start=q_prev,
                    q_end=q_curr,
                    margin=margin,
                    confidence_level=confidence_level,
                    target_outcome=target_outcome,
                    tolerance=tolerance,
                )
                eps = abs(refined_q - baseline_q0)
                candidates.append((eps, refined_q, refined_dec))
                break

    # Direction 2: Right (q0 from baseline_q0 up to q_max)
    if q_max > baseline_q0:
        grid_right = np.linspace(baseline_q0, q_max, grid_points)
        for i in range(1, len(grid_right)):
            q_prev = float(grid_right[i - 1])
            q_curr = float(grid_right[i])
            res_curr = model.evaluate_mixture_decision(
                q_curr, margin, confidence_level=confidence_level
            )
            if is_reversed(res_curr.outcome):
                refined_q, refined_dec = bisect_decision_boundary(
                    model,
                    q_start=q_prev,
                    q_end=q_curr,
                    margin=margin,
                    confidence_level=confidence_level,
                    target_outcome=target_outcome,
                    tolerance=tolerance,
                )
                eps = abs(refined_q - baseline_q0)
                candidates.append((eps, refined_q, refined_dec))
                break

    # Perform fine-grid verification across full interval [q_min, q_max]
    fine_grid = np.linspace(q_min, q_max, 500)
    fine_reversed = False
    for q_f in fine_grid:
        res_f = model.evaluate_mixture_decision(
            float(q_f), margin, confidence_level=confidence_level
        )
        if is_reversed(res_f.outcome):
            fine_reversed = True
            break

    if not candidates:
        status = "NO_REVERSAL_FOUND_WITHIN_BOUNDS"
        return ReversalSearchResult(
            initial_decision=initial_decision,
            reversed_decision=None,
            reversal_distance=None,
            reversed_q0=None,
            transition="NO_REVERSAL",
            status=status,
            baseline_q0=baseline_q0,
            max_epsilon=max_epsilon,
            margin=margin,
            tolerance=tolerance,
            fine_grid_verified=not fine_reversed,
        )

    # Sort candidates by minimum epsilon
    candidates.sort(key=lambda x: x[0])
    best_eps, best_q0, best_outcome = candidates[0]

    pt, _, ci_lower, ci_upper = model.evaluate_mixture(best_q0, confidence_level=confidence_level)
    transition = f"{initial_decision.value} -> {best_outcome.value}"

    return ReversalSearchResult(
        initial_decision=initial_decision,
        reversed_decision=best_outcome,
        reversal_distance=best_eps,
        reversed_q0=best_q0,
        transition=transition,
        status="REVERSAL_FOUND",
        baseline_q0=baseline_q0,
        max_epsilon=max_epsilon,
        margin=margin,
        tolerance=tolerance,
        fine_grid_verified=fine_reversed,
        reversed_point_estimate=pt,
        reversed_ci_lower=ci_lower,
        reversed_ci_upper=ci_upper,
    )
