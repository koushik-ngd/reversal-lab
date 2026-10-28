"""Reference policies with closed-form action distributions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from reversallab.contracts.validation import FloatArray
from reversallab.exceptions import ConfigurationError, DataValidationError


def _check_context(context: FloatArray, context_dim: int | None) -> FloatArray:
    x = np.asarray(context, dtype=np.float64)
    if x.ndim != 2:
        raise DataValidationError(f"`context` must be 2-dimensional (N, D), got shape {x.shape}.")
    if context_dim is not None and x.shape[1] != context_dim:
        raise DataValidationError(
            f"Policy expects context dimension {context_dim}, got {x.shape[1]}."
        )
    return x


class _LinearScores:
    """Linear action scores ``s(x, a) = x @ weights[:, a] + bias[a]``."""

    def __init__(self, weights: npt.ArrayLike, bias: npt.ArrayLike | None = None) -> None:
        w = np.array(weights, dtype=np.float64, copy=True)
        if w.ndim != 2 or w.shape[1] < 1:
            raise ConfigurationError(
                f"`weights` must have shape (D, K) with K >= 1, got {w.shape}."
            )
        b = np.zeros(w.shape[1]) if bias is None else np.array(bias, dtype=np.float64, copy=True)
        if b.shape != (w.shape[1],):
            raise ConfigurationError(f"`bias` must have shape ({w.shape[1]},), got {b.shape}.")
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(b))):
            raise ConfigurationError("Policy parameters must be finite.")
        w.flags.writeable = False
        b.flags.writeable = False
        self.weights = w
        self.bias = b

    @property
    def n_actions(self) -> int:
        return int(self.weights.shape[1])

    @property
    def context_dim(self) -> int:
        return int(self.weights.shape[0])

    def scores(self, context: FloatArray) -> FloatArray:
        x = _check_context(context, self.context_dim)
        return x @ self.weights + self.bias


class UniformPolicy:
    """``pi(a | x) = 1 / K`` for every context."""

    def __init__(self, n_actions: int) -> None:
        if n_actions < 1:
            raise ConfigurationError(f"`n_actions` must be >= 1, got {n_actions}.")
        self._n_actions = int(n_actions)

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def action_distribution(self, context: FloatArray) -> FloatArray:
        x = _check_context(context, None)
        return np.full((x.shape[0], self._n_actions), 1.0 / self._n_actions)


class LinearSoftmaxPolicy(_LinearScores):
    """``pi(a | x) = softmax_a(beta * s(x, a))`` for linear scores ``s``.

    ``beta = 0`` gives the uniform policy, ``beta -> inf`` approaches the greedy
    policy, and ``beta < 0`` prefers low-scoring actions. Probabilities are
    mathematically positive, but may underflow to exactly 0.0 for extreme scores;
    such underflow is reported as a support problem downstream, never patched.
    """

    def __init__(
        self,
        weights: npt.ArrayLike,
        bias: npt.ArrayLike | None = None,
        inverse_temperature: float = 1.0,
    ) -> None:
        super().__init__(weights, bias)
        if not np.isfinite(inverse_temperature):
            raise ConfigurationError("`inverse_temperature` must be finite.")
        self.inverse_temperature = float(inverse_temperature)

    def action_distribution(self, context: FloatArray) -> FloatArray:
        z = self.inverse_temperature * self.scores(context)
        z -= z.max(axis=1, keepdims=True)
        expz = np.exp(z)
        result: FloatArray = expz / expz.sum(axis=1, keepdims=True)
        return result


class EpsilonGreedyLinearPolicy(_LinearScores):
    """Greedy on linear scores with probability ``1 - epsilon``, uniform otherwise.

    ``pi(a | x) = epsilon / K + (1 - epsilon) * 1[a = argmax_b s(x, b)]``.
    Ties are broken towards the lowest action index (deterministically).
    ``epsilon = 0`` yields a deterministic policy.
    """

    def __init__(
        self, weights: npt.ArrayLike, bias: npt.ArrayLike | None = None, epsilon: float = 0.0
    ) -> None:
        super().__init__(weights, bias)
        if not 0.0 <= epsilon <= 1.0:
            raise ConfigurationError(f"`epsilon` must lie in [0, 1], got {epsilon}.")
        self.epsilon = float(epsilon)

    def action_distribution(self, context: FloatArray) -> FloatArray:
        s = self.scores(context)
        n_samples, n_actions = s.shape
        dist = np.full((n_samples, n_actions), self.epsilon / n_actions)
        dist[np.arange(n_samples), np.argmax(s, axis=1)] += 1.0 - self.epsilon
        return dist
