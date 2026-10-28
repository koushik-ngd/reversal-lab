"""Policy interface and action sampling."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from reversallab.contracts.validation import FloatArray, IntArray, validate_action_distribution
from reversallab.exceptions import DataValidationError


@runtime_checkable
class Policy(Protocol):
    """A stochastic (or deterministic) contextual policy ``pi(a | x)``.

    Implementations must be pure functions of the context: the same context matrix
    must always yield the same distribution. Randomness belongs to
    :func:`sample_actions`, never to the policy itself, so that the probabilities
    used by estimators are exactly those used to generate actions.
    """

    @property
    def n_actions(self) -> int: ...

    def action_distribution(self, context: FloatArray) -> FloatArray:
        """Return an ``(N, K)`` matrix whose row ``i`` is ``pi(. | context[i])``."""
        ...


def sample_actions(action_dist: FloatArray, rng: np.random.Generator) -> IntArray:
    """Draw one action per row of an ``(N, K)`` probability matrix.

    Uses inverse-CDF sampling with one uniform per row. Actions with zero
    probability can never be drawn, even with floating-point rounding in the
    cumulative sum, because each row's CDF is normalised by its own total.
    """
    if np.ndim(action_dist) != 2:
        raise DataValidationError(
            f"`action_dist` must be 2-dimensional (N, K), got shape {np.shape(action_dist)}."
        )
    n_samples, n_actions = (int(s) for s in np.shape(action_dist))
    dist = validate_action_distribution(
        action_dist, n_samples=n_samples, n_actions=n_actions, name="action_dist"
    )
    cdf = np.cumsum(dist, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random(size=n_samples)
    actions = np.sum(u[:, None] >= cdf, axis=1)
    return actions.astype(np.int64)
