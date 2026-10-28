"""Importance weights and support checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import FloatArray, validate_action_distribution
from reversallab.exceptions import NumericalError, SupportViolationError


@dataclass(frozen=True)
class SupportViolations:
    """Pairs ``(i, a)`` with ``pi_e(a|x_i) > 0`` but ``pi_0(a|x_i) = 0``."""

    n_violating_pairs: int
    n_violating_contexts: int
    mean_unsupported_evaluation_mass: float
    """``(1/N) sum_i sum_{a: pi_0(a|x_i)=0} pi_e(a|x_i)``: the bias of IPS in units of the
    reward range when rewards lie in [0, 1]."""


def validate_evaluation_action_dist(
    data: LoggedBanditData, evaluation_action_dist: npt.ArrayLike
) -> FloatArray:
    """Validate the ``(N, K)`` evaluation-policy distribution against ``data``."""
    return validate_action_distribution(
        evaluation_action_dist,
        n_samples=data.n_samples,
        n_actions=data.n_actions,
        name="evaluation_action_dist",
    )


def find_support_violations(
    data: LoggedBanditData, evaluation_action_dist: FloatArray
) -> SupportViolations | None:
    """Return support violations, or ``None`` if the full logging distribution is unknown.

    With only observed-action propensities, support can never be verified from data:
    an action the logger never takes leaves no trace in the log.
    """
    if data.logging_action_dist is None:
        return None
    mask = (evaluation_action_dist > 0.0) & (data.logging_action_dist <= 0.0)
    unsupported_mass = np.where(mask, evaluation_action_dist, 0.0).sum(axis=1)
    return SupportViolations(
        n_violating_pairs=int(mask.sum()),
        n_violating_contexts=int(mask.any(axis=1).sum()),
        mean_unsupported_evaluation_mass=float(unsupported_mass.mean()),
    )


def importance_weights(
    data: LoggedBanditData,
    evaluation_action_dist: npt.ArrayLike,
    *,
    require_support: bool = True,
) -> FloatArray:
    """Compute ``w_i = pi_e(a_i | x_i) / pi_0(a_i | x_i)``.

    Args:
        data: validated logged data (propensities are guaranteed positive).
        evaluation_action_dist: ``(N, K)`` evaluation-policy distribution on ``data.context``.
        require_support: if True and the full logging distribution is available, raise
            :class:`SupportViolationError` when the evaluation policy puts mass on any
            action the logging policy cannot choose. Diagnostics set this to False so
            that the violation can be *reported* rather than raised.
    """
    eval_dist = validate_evaluation_action_dist(data, evaluation_action_dist)
    if require_support:
        violations = find_support_violations(data, eval_dist)
        if violations is not None and violations.n_violating_pairs > 0:
            raise SupportViolationError(
                f"Evaluation policy assigns positive probability to {violations.n_violating_pairs} "
                f"(context, action) pair(s) in {violations.n_violating_contexts} context(s) where "
                "the logging policy has zero probability (mean unsupported mass "
                f"{violations.mean_unsupported_evaluation_mass:.4g}). Importance-weighted "
                "estimates are not identified; restrict the evaluation policy to the logging "
                "support or collect data with a more exploratory logging policy."
            )
    target = eval_dist[np.arange(data.n_samples), data.action]
    with np.errstate(over="ignore"):
        weights: FloatArray = target / data.logging_propensity
    if not np.all(np.isfinite(weights)):
        n_bad = int(np.size(weights) - np.count_nonzero(np.isfinite(weights)))
        raise NumericalError(
            f"{n_bad} importance weight(s) overflowed to a non-finite value. A logging propensity "
            "is positive but too small for float64 to represent pi_e(a|x) / pi_0(a|x). "
            "The ratio is not clipped. Increase the logging policy's minimum probability, "
            "or analyse the overflowed rows outside this estimator."
        )
    return weights
