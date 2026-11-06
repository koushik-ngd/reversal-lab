r"""Forward robust-value and worst-case mixture baseline evaluation (Method F).

In off-policy decision making, a standard Distributionally Robust Optimization (DRO) or
worst-case mixture baseline evaluates policy superiority over a pre-fixed perturbation set
:math:`\mathcal{Q}_\epsilon = [\max(0, q_0 - \epsilon), \min(1, q_0 + \epsilon)]`.

Rather than solving the inverse problem (finding the critical distance :math:`\hat d_{\mathrm{rev}}`
where evidence breaks down), this forward baseline:
1. Fixes the deployment robustness tolerance :math:`\epsilon_{\text{deploy}} \ge 0`.
2. Computes the worst-case pointwise confidence lower bound over :math:`\mathcal{Q}_\epsilon`:
   .. math::
       L_{\text{robust}}(\epsilon) = \inf_{q \in \mathcal{Q}_\epsilon} L_\Delta(q).
3. Recommends deployment if and only if :math:`L_{\text{robust}}(\epsilon) > \delta`.

Mathematical Concavity Property:
Let :math:`\sigma_0 \ge 0` and :math:`\sigma_1 \ge 0` denote the standard errors of the two strata.
The standard error function :math:`\mathrm{se}(q) = \sqrt{q^2 \sigma_0^2 + (1 - q)^2 \sigma_1^2}`
has second derivative:
.. math::
    \frac{\mathrm{d}^2}{\mathrm{d}q^2} \mathrm{se}(q)
    = \frac{\sigma_0^2 \sigma_1^2}{\big(q^2 \sigma_0^2 + (1 - q)^2 \sigma_1^2\big)^{3/2}} \ge 0.
Thus, :math:`\mathrm{se}(q)` is convex on :math:`[0, 1]` (strictly convex whenever
:math:`\sigma_0 \sigma_1 > 0`, and affine if :math:`\sigma_0 \sigma_1 = 0`).
For any nominal quantile :math:`z \ge 0`, :math:`-z \mathrm{se}(q)` is concave.
Because :math:`\hat \Delta(q) = q \hat \Delta_0 + (1 - q) \hat \Delta_1` is affine, the pointwise
confidence lower bound :math:`L_\Delta(q) = \hat \Delta(q) - z \mathrm{se}(q)` is concave
on :math:`[0, 1]`.

By the minimum principle for concave functions on compact convex sets, the infimum over any
closed sub-interval :math:`[q_{\min}, q_{\max}] \subseteq [0, 1]` is always achieved at an
extreme point (boundary):
.. math::
    \inf_{q \in [q_{\min}, q_{\max}]} L_\Delta(q)
    = \min\big( L_\Delta(q_{\min}), \; L_\Delta(q_{\max}) \big).

Strict Inequality & Boundary Convention:
ReversalLab adopts a conservative boundary convention where strict inequalities are required
for action: deployment recommendation requires :math:`L_{\text{robust}}(\epsilon) > \delta`.
If :math:`L_{\text{robust}}(\epsilon) = \delta`, the evidence exactly touches the margin and
is classified as **INCONCLUSIVE**.

Mathematical Duality with Method D:
Under the two-stratum mixture model with a consistent baseline mixture :math:`q_0`:
.. math::
    \hat d_{\mathrm{rev}}^* > \epsilon_{\text{deploy}}
    \iff L_{\text{robust}}(\epsilon_{\text{deploy}}) > \delta.
For any fixed robustness budget :math:`\epsilon_{\text{deploy}}`, inverse reversal search (Method D)
and forward worst-case confidence evaluation (Method F) yield mathematically equivalent binary
deployment decisions. The proposed practical advantage of ReversalLab is not an algorithmic
divergence in fixed-budget decisions, but rather solving the inverse problem directly: reporting
the continuous critical threshold :math:`\hat d_{\mathrm{rev}}^*` as a proposed conceptual aid
for decision makers without requiring an upfront guess of :math:`\epsilon_{\text{deploy}}`.
"""

from __future__ import annotations

from dataclasses import dataclass

from reversallab.reversal.decision import DecisionOutcome
from reversallab.reversal.strata import TwoStratumMixtureModel


@dataclass(frozen=True)
class RobustMixtureResult:
    """Outcome of forward worst-case robust mixture baseline evaluation (Method F)."""

    outcome: DecisionOutcome
    margin: float
    deploy_epsilon: float
    baseline_q0: float
    q_min: float
    q_max: float
    worst_case_lower_bound: float
    worst_case_lower_q: float
    worst_case_upper_bound: float
    worst_case_upper_q: float
    baseline_lower_bound: float
    baseline_upper_bound: float
    confidence_level: float
    diagnostic_passed: bool
    diagnostic_reason: str | None = None
    reason: str = ""


def evaluate_robust_mixture_baseline(
    model: TwoStratumMixtureModel,
    *,
    margin: float = 0.0,
    deploy_epsilon: float = 0.10,
    confidence_level: float = 0.95,
    diagnostic_passed: bool = True,
    diagnostic_reason: str | None = None,
) -> RobustMixtureResult:
    """Evaluate forward worst-case mixture baseline over Q_epsilon = [q0 - eps, q0 + eps].

    Parameters
    ----------
    model : TwoStratumMixtureModel
        Two-stratum policy value difference model.
    margin : float, default=0.0
        Minimum required deployment margin delta.
    deploy_epsilon : float, default=0.10
        Robustness tolerance ball radius.
    confidence_level : float, default=0.95
        Nominal confidence level (1 - alpha).
    diagnostic_passed : bool, default=True
        Whether operational diagnostic gate passed.
    diagnostic_reason : str | None, default=None
        Rejection reason if diagnostics failed.

    Returns
    -------
    RobustMixtureResult
        Decision outcome, worst-case bounds, and evaluation metadata.
    """
    q0 = model.baseline_q0

    # Admissible mixture interval
    q_min = max(0.0, q0 - deploy_epsilon)
    q_max = min(1.0, q0 + deploy_epsilon)

    def _bounds_at(q: float) -> tuple[float, float]:
        _, _, ci_l, ci_u = model.evaluate_mixture(q, confidence_level=confidence_level)
        return (ci_l, ci_u)

    base_l, base_u = _bounds_at(q0)
    l_min, u_min = _bounds_at(q_min)
    l_max, u_max = _bounds_at(q_max)

    # By strict concavity of L_Delta(q), minimum is strictly at boundary
    if l_min <= l_max:
        worst_lower = l_min
        worst_lower_q = q_min
    else:
        worst_lower = l_max
        worst_lower_q = q_max

    # For upper bound (convex), max is at boundary
    if u_min >= u_max:
        worst_upper = u_min
        worst_upper_q = q_min
    else:
        worst_upper = u_max
        worst_upper_q = q_max

    # Determine decision outcome
    if not diagnostic_passed:
        outcome = DecisionOutcome.INCONCLUSIVE
        reason = f"Diagnostics failed: {diagnostic_reason or 'unknown failure'}"
    elif base_u < margin:
        outcome = DecisionOutcome.HOLD
        reason = f"Baseline upper bound {base_u:.4f} < margin {margin:.4f}"
    elif worst_lower > margin:
        outcome = DecisionOutcome.RECOMMEND
        reason = (
            f"Worst-case lower bound {worst_lower:.4f} > margin {margin:.4f} "
            f"across Q_{deploy_epsilon:.2f}"
        )
    else:
        outcome = DecisionOutcome.INCONCLUSIVE
        reason = (
            f"Worst-case lower bound {worst_lower:.4f} <= margin {margin:.4f} "
            f"over Q_{deploy_epsilon:.2f} (fragile or insufficient evidence)"
        )

    return RobustMixtureResult(
        outcome=outcome,
        margin=margin,
        deploy_epsilon=deploy_epsilon,
        baseline_q0=q0,
        q_min=q_min,
        q_max=q_max,
        worst_case_lower_bound=worst_lower,
        worst_case_lower_q=worst_lower_q,
        worst_case_upper_bound=worst_upper,
        worst_case_upper_q=worst_upper_q,
        baseline_lower_bound=base_l,
        baseline_upper_bound=base_u,
        confidence_level=confidence_level,
        diagnostic_passed=diagnostic_passed,
        diagnostic_reason=diagnostic_reason,
        reason=reason,
    )
