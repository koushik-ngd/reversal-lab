"""Pydantic v2 schemas for ReversalLab FastAPI service."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str = "reversallab"
    timestamp: str


class VersionResponse(BaseModel):
    """Version information response."""

    version: str
    schema_version: str
    api_version: str = "v1"


class PolicySpec(BaseModel):
    """Specification of an evaluation or baseline policy."""

    type: Literal["uniform", "linear_softmax", "epsilon_greedy"]
    weights: list[list[float]] | None = Field(
        None, description="Weight matrix of shape (D, K) for linear policies."
    )
    inverse_temperature: float = Field(
        1.0, gt=0.0, description="Inverse temperature for softmax policies."
    )
    epsilon: float = Field(
        0.05, ge=0.0, le=1.0, description="Exploration probability for epsilon-greedy."
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is None:
            return None
        if len(v) == 0 or len(v[0]) == 0:
            raise ValueError("Weights matrix cannot be empty.")
        dim = len(v[0])
        for row in v:
            if len(row) != dim:
                raise ValueError("All rows in weights matrix must have identical dimension.")
            for val in row:
                if not math.isfinite(val):
                    raise ValueError("Weights matrix contains non-finite values (NaN/Inf).")
        return v


class StratumPartitionSpec(BaseModel):
    """Context feature splitting partition."""

    split_dim: int = Field(0, ge=0, description="Feature index used for 1-D binary partition.")
    threshold: float = Field(0.0, description="Threshold value for splitting.")

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Partition threshold must be a finite float.")
        return v


class BanditDataPayload(BaseModel):
    """Validated logged contextual bandit dataset."""

    contexts: list[list[float]] = Field(
        ..., description="Observed context matrix (N, D).", min_length=1, max_length=50000
    )
    actions: list[int] = Field(
        ..., description="Observed discrete action indices (N,).", min_length=1, max_length=50000
    )
    rewards: list[float] = Field(
        ..., description="Observed scalar rewards (N,).", min_length=1, max_length=50000
    )
    propensities: list[float] = Field(
        ..., description="Logging propensities P(A_i|X_i) > 0 (N,).", min_length=1, max_length=50000
    )
    n_actions: int = Field(..., ge=2, le=50, description="Total discrete action count K.")

    @model_validator(mode="after")
    def validate_shapes_and_values(self) -> BanditDataPayload:
        n = len(self.contexts)
        if not (len(self.actions) == n and len(self.rewards) == n and len(self.propensities) == n):
            raise ValueError(
                f"Array length mismatch: contexts ({n}), actions ({len(self.actions)}), "
                f"rewards ({len(self.rewards)}), propensities ({len(self.propensities)}) "
                "must all have identical length N."
            )

        if n == 0:
            raise ValueError("Dataset cannot be empty.")

        dim = len(self.contexts[0])
        if dim == 0 or dim > 128:
            raise ValueError(f"Context dimension D={dim} must be between 1 and 128.")

        for row in self.contexts:
            if len(row) != dim:
                raise ValueError("All context rows must have identical feature dimension D.")
            for val in row:
                if not math.isfinite(val):
                    raise ValueError("Context matrix contains non-finite values (NaN/Inf).")

        for a in self.actions:
            if a < 0 or a >= self.n_actions:
                raise ValueError(
                    f"Action index {a} out of bounds for action space [0, {self.n_actions})."
                )

        for r in self.rewards:
            if not math.isfinite(r):
                raise ValueError("Rewards array contains non-finite values (NaN/Inf).")

        for p in self.propensities:
            if not math.isfinite(p):
                raise ValueError("Propensities array contains non-finite values (NaN/Inf).")
            if p <= 0.0 or p > 1.0:
                raise ValueError(f"Propensities must be in (0, 1], got {p}.")

        return self


class PolicyEvaluationRequest(BaseModel):
    """Request payload for single-policy off-policy evaluation."""

    dataset: BanditDataPayload
    policy: PolicySpec
    estimator: Literal["ips", "snips", "dm", "dr"] = "dr"
    cross_fitting_folds: int = Field(5, ge=2, le=10)
    confidence_level: float = Field(0.95, gt=0.50, lt=1.0)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_budget_and_policy(self) -> PolicyEvaluationRequest:
        n = len(self.dataset.contexts)
        d = len(self.dataset.contexts[0])
        k = self.dataset.n_actions

        # Pre-execution computation budget check: N * D * folds * K
        budget = n * d * self.cross_fitting_folds * k
        max_budget = 10_000_000
        if budget > max_budget:
            raise ValueError(
                f"Requested computation budget (N*D*folds*K = {budget}) exceeds "
                f"maximum synchronous limit of {max_budget}."
            )

        # Validate policy parameter shape against dataset
        if self.policy.type in ("linear_softmax", "epsilon_greedy"):
            if self.policy.weights is None:
                raise ValueError(f"Policy type '{self.policy.type}' requires 'weights' parameter.")
            w_d = len(self.policy.weights)
            w_k = len(self.policy.weights[0])
            if w_d != d or w_k != k:
                raise ValueError(
                    f"Policy weights shape ({w_d}, {w_k}) does not match "
                    f"context dimension D={d} and action count K={k}."
                )

        return self


class PolicyComparisonRequest(BaseModel):
    """Request payload for paired policy comparison and decision reversal."""

    dataset: BanditDataPayload
    candidate_policy: PolicySpec
    baseline_policy: PolicySpec
    margin: float = Field(0.0, ge=0.0, description="Deployment superiority margin delta >= 0.")
    deploy_epsilon: float = Field(
        0.10, ge=0.0, le=0.50, description="Operational mixture envelope radius."
    )
    max_epsilon: float = Field(
        0.40, ge=0.0, le=0.50, description="Maximum reversal search boundary."
    )
    stratum_partition: StratumPartitionSpec | None = None
    estimator: Literal["ips", "snips", "dm", "dr"] = "dr"
    cross_fitting_folds: int = Field(5, ge=2, le=10)
    confidence_level: float = Field(0.95, gt=0.50, lt=1.0)
    min_propensity_threshold: float = Field(1e-4, gt=0.0)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_budget_and_policies(self) -> PolicyComparisonRequest:
        n = len(self.dataset.contexts)
        d = len(self.dataset.contexts[0])
        k = self.dataset.n_actions

        budget = n * d * self.cross_fitting_folds * k
        max_budget = 10_000_000
        if budget > max_budget:
            msg = (
                f"Requested computation budget (N*D*folds*K = {budget}) exceeds "
                f"maximum synchronous limit of {max_budget}."
            )
            raise ValueError(msg)

        if self.deploy_epsilon > self.max_epsilon:
            msg = (
                f"deploy_epsilon ({self.deploy_epsilon}) cannot exceed "
                f"max_epsilon ({self.max_epsilon})."
            )
            raise ValueError(msg)

        # Validate stratum partition dimension
        if self.stratum_partition is not None and self.stratum_partition.split_dim >= d:
            raise ValueError(
                f"stratum_partition split_dim={self.stratum_partition.split_dim} "
                f"exceeds context feature dimension D={d}."
            )

        # Validate candidate policy
        if self.candidate_policy.type in ("linear_softmax", "epsilon_greedy"):
            if self.candidate_policy.weights is None:
                raise ValueError(
                    f"Candidate policy '{self.candidate_policy.type}' requires 'weights'."
                )
            w_d = len(self.candidate_policy.weights)
            w_k = len(self.candidate_policy.weights[0])
            if w_d != d or w_k != k:
                raise ValueError(
                    f"Candidate policy weights shape ({w_d}, {w_k}) does not match D={d}, K={k}."
                )

        # Validate baseline policy
        if self.baseline_policy.type in ("linear_softmax", "epsilon_greedy"):
            if self.baseline_policy.weights is None:
                raise ValueError(
                    f"Baseline policy '{self.baseline_policy.type}' requires 'weights'."
                )
            w_d = len(self.baseline_policy.weights)
            w_k = len(self.baseline_policy.weights[0])
            if w_d != d or w_k != k:
                raise ValueError(
                    f"Baseline policy weights shape ({w_d}, {w_k}) does not match D={d}, K={k}."
                )

        return self


class EvaluationResponse(BaseModel):
    """Enveloped response for single policy evaluation."""

    request_id: str
    status: Literal["success", "inconclusive"]
    timestamp: str
    report: dict[str, Any]


class ComparisonResponse(BaseModel):
    """Enveloped response for paired policy comparison and reversal."""

    request_id: str
    status: Literal["success", "inconclusive"]
    timestamp: str
    report: dict[str, Any]


class ErrorDetail(BaseModel):
    """RFC 7807 compliant problem details model."""

    type: str
    title: str
    status: int
    detail: str
    instance: str
    request_id: str
