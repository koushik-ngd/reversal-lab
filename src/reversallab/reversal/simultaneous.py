r"""Simultaneous confidence bands and adaptive search inference for decision-reversal.

A pointwise :math:`(1 - \alpha)` confidence interval evaluated at a post-search mixture
weight :math:`q^* \in \mathcal{Q}_\epsilon` does not automatically provide a simultaneous
guarantee over the entire searched family. Searching over :math:`q` to find a boundary
violates standard single-hypothesis testing assumptions.

To provide a mathematically conservative inference guarantee, this module implements a
simultaneous confidence band across all :math:`q \in [0, 1]` via Bonferroni error allocation
over the two independent strata:

1. Let :math:`z_{\text{simult}} = z_{1 - \alpha/4}` (for :math:`\alpha = 0.05`,
   :math:`z_{\text{simult}} = z_{0.9875} \approx 2.2414`).
2. Construct simultaneous intervals for both stratum effects:
   .. math::
       I_0 = [L_0^{\text{simult}}, U_0^{\text{simult}}]
           = [\hat \Delta_0 \mp z_{\text{simult}} \widehat{\mathrm{se}}_0], \\
       I_1 = [L_1^{\text{simult}}, U_1^{\text{simult}}]
           = [\hat \Delta_1 \mp z_{\text{simult}} \widehat{\mathrm{se}}_1].

   By Bonferroni's inequality, :math:`P(\Delta_0 \in I_0 \land \Delta_1 \in I_1) \ge 1 - \alpha`.
3. Because :math:`q \ge 0` and :math:`1 - q \ge 0`, the linear combination satisfies:
   .. math::
       L_\Delta^{\text{simult}}(q) = q L_0^{\text{simult}} + (1 - q) L_1^{\text{simult}}
       \le \Delta(q)
       \le q U_0^{\text{simult}} + (1 - q) U_1^{\text{simult}} = U_\Delta^{\text{simult}}(q)
   simultaneously for all :math:`q \in [0, 1]` with probability at least :math:`1 - \alpha`.

4. Because :math:`L_\Delta^{\text{simult}}(q)` is linear in :math:`q`, its infimum over any
   admissible interval :math:`[q_{\min}, q_{\max}]` occurs at an endpoint:
   .. math::
       \inf_{q \in [q_{\min}, q_{\max}]} L_\Delta^{\text{simult}}(q)
       = \min\big( L_\Delta^{\text{simult}}(q_{\min}), \; L_\Delta^{\text{simult}}(q_{\max}) \big).
"""

from __future__ import annotations

from dataclasses import dataclass

from reversallab.evaluation.estimators import _normal_quantile
from reversallab.reversal.decision import DecisionOutcome, DecisionResult, evaluate_decision
from reversallab.reversal.search import find_decision_reversal
from reversallab.reversal.strata import TwoStratumMixtureModel


@dataclass(frozen=True)
class SimultaneousBandResult:
    """Simultaneous confidence band evaluation at a specific mixture weight q0."""

    q0: float
    point_estimate: float
    simultaneous_ci_lower: float
    simultaneous_ci_upper: float
    pointwise_ci_lower: float
    pointwise_ci_upper: float
    band_expansion_ratio: float
    decision: DecisionResult


@dataclass(frozen=True)
class SimultaneousReversalResult:
    """Outcome of simultaneous confidence-band decision-reversal search."""

    initial_decision: DecisionOutcome
    reversed_decision: DecisionOutcome | None
    simultaneous_reversal_distance: float | None
    simultaneous_reversed_q0: float | None
    pointwise_reversal_distance: float | None
    status: str
    transition: str
    baseline_q0: float
    max_epsilon: float
    margin: float
    confidence_level: float
    min_simultaneous_lower_bound: float
    max_simultaneous_upper_bound: float


class TwoStratumSimultaneousBand:
    """Uniform confidence band over mixture weight q in [0, 1]."""

    def __init__(
        self,
        model: TwoStratumMixtureModel,
        *,
        confidence_level: float = 0.95,
    ) -> None:
        self.model = model
        self.confidence_level = confidence_level

        # Bonferroni alpha / 2 for each of 2 strata => alpha / 4 two-sided
        alpha = 1.0 - confidence_level
        self.z_simult = _normal_quantile(1.0 - alpha / 2.0)
        self.z_pointwise = _normal_quantile(confidence_level)

        s0 = model.stratum_0
        s1 = model.stratum_1

        self.stratum_0_lower = s0.mean_diff - self.z_simult * s0.standard_error
        self.stratum_0_upper = s0.mean_diff + self.z_simult * s0.standard_error

        self.stratum_1_lower = s1.mean_diff - self.z_simult * s1.standard_error
        self.stratum_1_upper = s1.mean_diff + self.z_simult * s1.standard_error

    def evaluate_band(self, q0: float, margin: float) -> SimultaneousBandResult:
        """Evaluate simultaneous and pointwise confidence bounds at mixture weight q0."""
        if not (0.0 <= q0 <= 1.0):
            raise ValueError(f"q0 must be in [0, 1]; got {q0}.")

        q1 = 1.0 - q0
        pt = q0 * self.model.stratum_0.mean_diff + q1 * self.model.stratum_1.mean_diff

        # Simultaneous bounds (linear in q)
        simult_lower = q0 * self.stratum_0_lower + q1 * self.stratum_1_lower
        simult_upper = q0 * self.stratum_0_upper + q1 * self.stratum_1_upper

        # Pointwise bounds (curved in q)
        _, _, pw_lower, pw_upper = self.model.evaluate_mixture(
            q0, confidence_level=self.confidence_level
        )

        pw_width = pw_upper - pw_lower
        simult_width = simult_upper - simult_lower
        ratio = simult_width / pw_width if pw_width > 0 else 1.0

        decision = evaluate_decision(
            ci_lower=simult_lower,
            ci_upper=simult_upper,
            point_estimate=pt,
            margin=margin,
            confidence_level=self.confidence_level,
        )

        return SimultaneousBandResult(
            q0=q0,
            point_estimate=pt,
            simultaneous_ci_lower=simult_lower,
            simultaneous_ci_upper=simult_upper,
            pointwise_ci_lower=pw_lower,
            pointwise_ci_upper=pw_upper,
            band_expansion_ratio=ratio,
            decision=decision,
        )

    def find_simultaneous_reversal(
        self,
        margin: float,
        *,
        max_epsilon: float = 0.40,
        pointwise_distance: float | None = None,
        diagnostic_passed: bool = True,
        diagnostic_reason: str | None = None,
    ) -> SimultaneousReversalResult:
        """Find the reversal distance under the simultaneous confidence band."""
        if pointwise_distance is None:
            pw_res = find_decision_reversal(
                self.model,
                margin=margin,
                max_epsilon=max_epsilon,
                confidence_level=self.confidence_level,
                diagnostic_passed=diagnostic_passed,
                diagnostic_reason=diagnostic_reason,
            )
            pointwise_distance = pw_res.reversal_distance

        baseline_q0 = self.model.baseline_q0
        base_res = self.evaluate_band(baseline_q0, margin)
        initial_decision = base_res.decision.outcome

        if not diagnostic_passed:
            return SimultaneousReversalResult(
                initial_decision=initial_decision,
                reversed_decision=None,
                simultaneous_reversal_distance=None,
                simultaneous_reversed_q0=None,
                pointwise_reversal_distance=pointwise_distance,
                status="BASELINE_DIAGNOSTIC_FAILURE",
                transition="NO_REVERSAL",
                baseline_q0=baseline_q0,
                max_epsilon=max_epsilon,
                margin=margin,
                confidence_level=self.confidence_level,
                min_simultaneous_lower_bound=base_res.simultaneous_ci_lower,
                max_simultaneous_upper_bound=base_res.simultaneous_ci_upper,
            )

        q_min = max(0.0, baseline_q0 - max_epsilon)
        q_max = min(1.0, baseline_q0 + max_epsilon)

        res_min = self.evaluate_band(q_min, margin)
        res_max = self.evaluate_band(q_max, margin)

        min_lower = min(res_min.simultaneous_ci_lower, res_max.simultaneous_ci_lower)
        max_upper = max(res_min.simultaneous_ci_upper, res_max.simultaneous_ci_upper)

        # Because L_simult(q) is linear: L_simult(q) = q L0 + (1-q) L1 = q (L0 - L1) + L1
        # The crossing point with margin delta is:
        denom_lower = self.stratum_0_lower - self.stratum_1_lower
        crossing_lower: float | None = None
        if abs(denom_lower) > 1e-12:
            q_c = (margin - self.stratum_1_lower) / denom_lower
            if q_min <= q_c <= q_max:
                crossing_lower = q_c

        # Upper bound crossing:
        denom_upper = self.stratum_0_upper - self.stratum_1_upper
        crossing_upper: float | None = None
        if abs(denom_upper) > 1e-12:
            q_u = (margin - self.stratum_1_upper) / denom_upper
            if q_min <= q_u <= q_max:
                crossing_upper = q_u

        candidates: list[tuple[float, float, DecisionOutcome]] = []

        for q_cand in [crossing_lower, crossing_upper, q_min, q_max]:
            if q_cand is not None:
                res_c = self.evaluate_band(q_cand, margin)
                if res_c.decision.outcome != initial_decision:
                    dist = abs(q_cand - baseline_q0)
                    candidates.append((dist, q_cand, res_c.decision.outcome))

        if not candidates:
            return SimultaneousReversalResult(
                initial_decision=initial_decision,
                reversed_decision=None,
                simultaneous_reversal_distance=None,
                simultaneous_reversed_q0=None,
                pointwise_reversal_distance=pointwise_distance,
                status="NO_REVERSAL_FOUND_WITHIN_BOUNDS",
                transition="NO_REVERSAL",
                baseline_q0=baseline_q0,
                max_epsilon=max_epsilon,
                margin=margin,
                confidence_level=self.confidence_level,
                min_simultaneous_lower_bound=min_lower,
                max_simultaneous_upper_bound=max_upper,
            )

        candidates.sort(key=lambda x: x[0])
        best_dist, best_q0, best_outcome = candidates[0]
        transition = f"{initial_decision.value} -> {best_outcome.value}"

        return SimultaneousReversalResult(
            initial_decision=initial_decision,
            reversed_decision=best_outcome,
            simultaneous_reversal_distance=best_dist,
            simultaneous_reversed_q0=best_q0,
            pointwise_reversal_distance=pointwise_distance,
            status="REVERSAL_FOUND",
            transition=transition,
            baseline_q0=baseline_q0,
            max_epsilon=max_epsilon,
            margin=margin,
            confidence_level=self.confidence_level,
            min_simultaneous_lower_bound=min_lower,
            max_simultaneous_upper_bound=max_upper,
        )
