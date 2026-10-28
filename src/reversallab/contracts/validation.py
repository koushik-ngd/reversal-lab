"""Low-level array validation shared by the data contract and the estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from reversallab.exceptions import DataValidationError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

PROBABILITY_SUM_ATOL: float = 1e-6
"""Maximum allowed absolute deviation of an action distribution's row sum from 1."""

PROBABILITY_RANGE_ATOL: float = 1e-12
"""Tolerance for probabilities marginally outside [0, 1] due to floating-point rounding."""

PROPENSITY_CONSISTENCY_RTOL: float = 1e-6
PROPENSITY_CONSISTENCY_ATOL: float = 1e-9
"""Tolerances when checking observed propensities against a full logging distribution."""


def as_readonly(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return ``array`` with the writeable flag cleared (a copy is not made)."""
    array.flags.writeable = False
    return array


def to_float_array(values: npt.ArrayLike, *, name: str, ndim: int) -> FloatArray:
    """Convert to a float64 copy, checking dimensionality and finiteness."""
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"`{name}` must be numeric; conversion failed: {exc}") from exc
    if array.ndim != ndim:
        raise DataValidationError(f"`{name}` must be {ndim}-dimensional, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        n_bad = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise DataValidationError(
            f"`{name}` contains {n_bad} non-finite value(s) (NaN or inf). "
            "Remove or impute them explicitly before evaluation."
        )
    return array


def to_action_array(values: npt.ArrayLike, *, n_actions: int, name: str = "action") -> IntArray:
    """Convert to an int64 copy of action indices in ``[0, n_actions)``.

    Non-integer dtypes are rejected rather than rounded, because silently casting
    ``1.7`` to ``1`` would hide a corrupt input.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise DataValidationError(f"`{name}` must be 1-dimensional, got shape {array.shape}.")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise DataValidationError(
            f"`{name}` must have an integer dtype, got {array.dtype}. "
            "Encode actions as integer indices 0..K-1."
        )
    out = array.astype(np.int64, copy=True)
    if out.size:
        lo, hi = int(np.min(out)), int(np.max(out))
        if lo < 0 or hi >= n_actions:
            raise DataValidationError(
                f"`{name}` values must lie in [0, {n_actions - 1}] for n_actions={n_actions}; "
                f"observed range [{lo}, {hi}]."
            )
    return out


def validate_action_distribution(
    values: npt.ArrayLike, *, n_samples: int, n_actions: int, name: str
) -> FloatArray:
    """Validate an ``(N, K)`` matrix of per-context action probabilities.

    Each row must be finite, lie in [0, 1], and sum to 1 within
    :data:`PROBABILITY_SUM_ATOL`. The returned array is a validated float64 copy;
    values are *not* renormalised, so the caller sees exactly what was supplied.
    """
    dist = to_float_array(values, name=name, ndim=2)
    if dist.shape != (n_samples, n_actions):
        raise DataValidationError(
            f"`{name}` must have shape (n_samples, n_actions) = ({n_samples}, {n_actions}), "
            f"got {dist.shape}."
        )
    if np.any(dist < -PROBABILITY_RANGE_ATOL) or np.any(dist > 1.0 + PROBABILITY_RANGE_ATOL):
        raise DataValidationError(f"`{name}` contains probabilities outside [0, 1].")
    row_sums = dist.sum(axis=1)
    deviation = np.abs(row_sums - 1.0)
    if np.any(deviation > PROBABILITY_SUM_ATOL):
        worst = int(np.argmax(deviation))
        raise DataValidationError(
            f"Rows of `{name}` must sum to 1 (atol={PROBABILITY_SUM_ATOL}); "
            f"row {worst} sums to {row_sums[worst]:.10g}."
        )
    return dist
