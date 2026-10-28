"""Validated experiment configuration and policy construction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from reversallab.data.synthetic import RewardType, SyntheticBanditEnvironment
from reversallab.evaluation.estimators import ESTIMATORS
from reversallab.exceptions import ConfigurationError
from reversallab.policies import (
    EpsilonGreedyLinearPolicy,
    LinearSoftmaxPolicy,
    Policy,
    UniformPolicy,
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ScoredPolicySpec(_Strict):
    scores: Literal["reward", "random"] = Field(
        description="'reward': the environment's true reward logits (an informed policy); "
        "'random': logits drawn independently of the reward (an uninformed policy)."
    )
    score_seed: int | None = Field(default=None, description="Required when scores == 'random'.")

    @model_validator(mode="after")
    def _seed_for_random(self) -> _ScoredPolicySpec:
        if self.scores == "random" and self.score_seed is None:
            raise ValueError("`score_seed` is required when scores == 'random'.")
        if self.scores == "reward" and self.score_seed is not None:
            raise ValueError("`score_seed` is only meaningful when scores == 'random'.")
        return self


class UniformPolicySpec(_Strict):
    kind: Literal["uniform"]


class SoftmaxPolicySpec(_ScoredPolicySpec):
    kind: Literal["softmax"]
    inverse_temperature: float = Field(allow_inf_nan=False)


class EpsilonGreedyPolicySpec(_ScoredPolicySpec):
    kind: Literal["epsilon_greedy"]
    epsilon: float = Field(ge=0.0, le=1.0)


PolicySpec = Annotated[
    UniformPolicySpec | SoftmaxPolicySpec | EpsilonGreedyPolicySpec, Field(discriminator="kind")
]


class EnvironmentConfig(_Strict):
    n_actions: int = Field(ge=2)
    context_dim: int = Field(ge=1)
    reward_type: RewardType = RewardType.BINARY
    reward_scale: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    bias_scale: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    noise_std: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)


class ScenarioConfig(_Strict):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    description: str = ""
    n_samples: list[int] = Field(min_length=1)
    logging_policy: PolicySpec
    evaluation_policy: PolicySpec

    @model_validator(mode="after")
    def _positive_sizes(self) -> ScenarioConfig:
        if any(n < 2 for n in self.n_samples):
            raise ValueError("every entry of `n_samples` must be >= 2.")
        if len(set(self.n_samples)) != len(self.n_samples):
            raise ValueError("`n_samples` entries must be unique.")
        return self


class DiagnosticThresholdsConfig(_Strict):
    min_ess_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    max_weight_share: float = Field(default=0.05, gt=0.0, le=1.0)
    mean_weight_z: float = Field(default=4.0, gt=0.0)


class EstimatorValidationConfig(_Strict):
    """Configuration for experiment 001: estimator validation against ground truth."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    description: str = ""
    seed: int = Field(ge=0)
    n_replications: int = Field(ge=2)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    reference_n_contexts: int = Field(default=1_000_000, ge=1000)
    estimators: list[str] = Field(min_length=1)
    include_logging_action_dist: bool = True
    environment: EnvironmentConfig
    diagnostics: DiagnosticThresholdsConfig = DiagnosticThresholdsConfig()
    scenarios: list[ScenarioConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> EstimatorValidationConfig:
        unknown = sorted(set(self.estimators) - set(ESTIMATORS))
        if unknown:
            raise ValueError(f"unknown estimator(s) {unknown}; available: {sorted(ESTIMATORS)}.")
        names = [s.name for s in self.scenarios]
        if len(set(names)) != len(names):
            raise ValueError("scenario names must be unique.")
        return self


def load_config(path: str | Path) -> EstimatorValidationConfig:
    """Load and validate a YAML experiment config, raising :class:`ConfigurationError`."""
    p = Path(path)
    if not p.is_file():
        raise ConfigurationError(f"Config file not found: {p}")
    try:
        raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Could not parse YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{p} must contain a YAML mapping at the top level.")
    try:
        return EstimatorValidationConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid experiment config {p}:\n{exc}") from exc


def build_environment(
    config: EnvironmentConfig, rng: np.random.Generator
) -> SyntheticBanditEnvironment:
    return SyntheticBanditEnvironment.random(
        n_actions=config.n_actions,
        context_dim=config.context_dim,
        rng=rng,
        reward_type=config.reward_type,
        reward_scale=config.reward_scale,
        bias_scale=config.bias_scale,
        noise_std=config.noise_std,
    )


def build_policy(
    spec: UniformPolicySpec | SoftmaxPolicySpec | EpsilonGreedyPolicySpec,
    env: SyntheticBanditEnvironment,
) -> Policy:
    """Instantiate a policy for ``env`` from its spec.

    Score-based policies use either the environment's reward logits (so higher
    inverse temperature / lower epsilon means a better policy) or random logits
    with the same marginal scale, drawn from ``score_seed``.
    """
    if isinstance(spec, UniformPolicySpec):
        return UniformPolicy(env.n_actions)
    if spec.scores == "reward":
        weights, bias = env.reward_weights, env.reward_bias
    else:
        rng = np.random.default_rng(spec.score_seed)
        weights = rng.normal(0.0, 1.0 / math.sqrt(env.context_dim), size=env.reward_weights.shape)
        bias = np.zeros(env.n_actions)
    if isinstance(spec, SoftmaxPolicySpec):
        return LinearSoftmaxPolicy(weights, bias, inverse_temperature=spec.inverse_temperature)
    return EpsilonGreedyLinearPolicy(weights, bias, epsilon=spec.epsilon)
