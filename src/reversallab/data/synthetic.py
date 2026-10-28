"""Synthetic single-position contextual-bandit environment with known ground truth.

Generative model (see ``docs/methodology.md``):

    x ~ N(0, I_D)
    z(x, a) = x @ W[:, a] + b[a]                       (reward logits)
    q(x, a) = sigmoid(z)        if reward_type == "binary"
            = z                 if reward_type == "gaussian"
    a ~ pi_0(. | x)
    r | x, a ~ Bernoulli(q(x, a))      or   N(q(x, a), noise_std^2)

The policy value ``V(pi) = E_x[ sum_a pi(a|x) q(x, a) ]`` has an exact inner sum,
so the only approximation in the reference value is the outer expectation over
``x``, which is estimated from an independent context sample with a reported
Monte Carlo standard error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import FloatArray, IntArray
from reversallab.exceptions import ConfigurationError, DataValidationError
from reversallab.policies.base import Policy, sample_actions


class RewardType(StrEnum):
    BINARY = "binary"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class ReferenceValue:
    """Ground-truth policy value with the Monte Carlo error of its outer expectation."""

    value: float
    monte_carlo_standard_error: float
    n_contexts: int


class SyntheticBanditEnvironment:
    """A fully specified contextual-bandit environment."""

    def __init__(
        self,
        reward_weights: npt.ArrayLike,
        reward_bias: npt.ArrayLike,
        reward_type: RewardType | str = RewardType.BINARY,
        noise_std: float = 1.0,
    ) -> None:
        w = np.array(reward_weights, dtype=np.float64, copy=True)
        b = np.array(reward_bias, dtype=np.float64, copy=True)
        if w.ndim != 2 or w.shape[0] < 1 or w.shape[1] < 2:
            raise ConfigurationError(
                f"`reward_weights` must have shape (D>=1, K>=2), got {w.shape}."
            )
        if b.shape != (w.shape[1],):
            raise ConfigurationError(
                f"`reward_bias` must have shape ({w.shape[1]},), got {b.shape}."
            )
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(b))):
            raise ConfigurationError("Reward parameters must be finite.")
        self.reward_type = RewardType(reward_type)
        if self.reward_type is RewardType.GAUSSIAN and not (
            noise_std >= 0 and math.isfinite(noise_std)
        ):
            raise ConfigurationError(f"`noise_std` must be finite and >= 0, got {noise_std}.")
        w.flags.writeable = False
        b.flags.writeable = False
        self.reward_weights = w
        self.reward_bias = b
        self.noise_std = float(noise_std)

    @classmethod
    def random(
        cls,
        *,
        n_actions: int,
        context_dim: int,
        rng: np.random.Generator,
        reward_type: RewardType | str = RewardType.BINARY,
        reward_scale: float = 1.0,
        bias_scale: float = 0.0,
        noise_std: float = 1.0,
    ) -> SyntheticBanditEnvironment:
        """Draw reward parameters ``W ~ N(0, reward_scale^2 / D)``, ``b ~ N(0, bias_scale^2)``.

        The ``1/D`` scaling keeps the marginal standard deviation of each logit equal
        to ``reward_scale`` regardless of the context dimension.
        """
        if n_actions < 2 or context_dim < 1:
            raise ConfigurationError("Require n_actions >= 2 and context_dim >= 1.")
        if reward_scale < 0 or bias_scale < 0:
            raise ConfigurationError("`reward_scale` and `bias_scale` must be >= 0.")
        weights = rng.normal(
            0.0, reward_scale / math.sqrt(context_dim), size=(context_dim, n_actions)
        )
        bias = (
            rng.normal(0.0, bias_scale, size=n_actions) if bias_scale > 0 else np.zeros(n_actions)
        )
        return cls(weights, bias, reward_type=reward_type, noise_std=noise_std)

    @property
    def n_actions(self) -> int:
        return int(self.reward_weights.shape[1])

    @property
    def context_dim(self) -> int:
        return int(self.reward_weights.shape[0])

    def sample_contexts(self, n_samples: int, rng: np.random.Generator) -> FloatArray:
        if n_samples < 1:
            raise ConfigurationError(f"`n_samples` must be >= 1, got {n_samples}.")
        return rng.standard_normal((n_samples, self.context_dim))

    def reward_logits(self, context: FloatArray) -> FloatArray:
        x = np.asarray(context, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.context_dim:
            raise DataValidationError(
                f"`context` must have shape (N, {self.context_dim}), got {x.shape}."
            )
        logits: FloatArray = x @ self.reward_weights + self.reward_bias
        return logits

    def expected_reward(self, context: FloatArray) -> FloatArray:
        """``(N, K)`` matrix of ``q(x_i, a) = E[r | x_i, a]``."""
        z = self.reward_logits(context)
        if self.reward_type is RewardType.BINARY:
            q: FloatArray = 1.0 / (1.0 + np.exp(-z))
            return q
        return z

    def sample_rewards(
        self, context: FloatArray, action: IntArray, rng: np.random.Generator
    ) -> FloatArray:
        q = self.expected_reward(context)[np.arange(len(action)), action]
        if self.reward_type is RewardType.BINARY:
            return (rng.random(len(action)) < q).astype(np.float64)
        noise: FloatArray = rng.normal(0.0, self.noise_std, size=len(action))
        return q + noise

    def generate_logged_data(
        self,
        n_samples: int,
        logging_policy: Policy,
        rng: np.random.Generator,
        *,
        include_logging_action_dist: bool = True,
    ) -> LoggedBanditData:
        """Simulate ``n_samples`` i.i.d. interactions logged by ``logging_policy``.

        Random draws are consumed in a fixed order (contexts, actions, rewards) so a
        given generator state always produces the same dataset.
        """
        self._check_policy(logging_policy)
        context = self.sample_contexts(n_samples, rng)
        logging_dist = logging_policy.action_distribution(context)
        action = sample_actions(logging_dist, rng)
        reward = self.sample_rewards(context, action, rng)
        return LoggedBanditData(
            context=context,
            action=action,
            reward=reward,
            logging_propensity=logging_dist[np.arange(n_samples), action],
            n_actions=self.n_actions,
            logging_action_dist=logging_dist if include_logging_action_dist else None,
        )

    def conditional_policy_value(self, context: FloatArray, policy: Policy) -> float:
        """Exact ``(1/N) sum_i sum_a pi(a|x_i) q(x_i, a)`` for the given contexts."""
        self._check_policy(policy)
        per_context = np.sum(
            policy.action_distribution(context) * self.expected_reward(context), axis=1
        )
        return float(per_context.mean())

    def policy_value(
        self,
        policy: Policy,
        rng: np.random.Generator,
        *,
        n_contexts: int = 1_000_000,
        chunk_size: int = 100_000,
    ) -> ReferenceValue:
        """Reference ``V(pi)``: exact over actions, Monte Carlo over contexts.

        ``rng`` must be independent of any generator used to produce evaluation data,
        otherwise the reference value and the estimates would share sampling noise.
        """
        self._check_policy(policy)
        if n_contexts < 2:
            raise ConfigurationError("`n_contexts` must be >= 2 to report a standard error.")
        total = 0.0
        total_sq = 0.0
        remaining = n_contexts
        while remaining > 0:
            m = min(chunk_size, remaining)
            x = self.sample_contexts(m, rng)
            f = np.sum(policy.action_distribution(x) * self.expected_reward(x), axis=1)
            total += float(f.sum())
            total_sq += float(np.dot(f, f))
            remaining -= m
        mean = total / n_contexts
        variance = max(total_sq / n_contexts - mean * mean, 0.0) * n_contexts / (n_contexts - 1)
        return ReferenceValue(
            value=mean,
            monte_carlo_standard_error=math.sqrt(variance / n_contexts),
            n_contexts=n_contexts,
        )

    def _check_policy(self, policy: Policy) -> None:
        if policy.n_actions != self.n_actions:
            raise ConfigurationError(
                f"Policy has {policy.n_actions} actions but the environment has {self.n_actions}."
            )
