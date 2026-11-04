r"""Two-stratum context mixture perturbation model (:math:`\mathcal{Q}_\epsilon`).

Real-world distribution shift often appears as demographic or contextual mixture shifts,
where the proportion of distinct subpopulations changes while conditional outcomes
:math:`P(r \mid x, a)` remain stable within each subpopulation.

This module formalizes a 1-dimensional context mixture shift model:
1. Context partition :math:`h(x) = \mathbb{I}(x_{\text{dim}} \ge \tau) \in \{0, 1\}`.
2. Stratum-level policy-value difference statistics :math:`(\hat \Delta_0, \widehat{\mathrm{se}}_0)`
   and :math:`(\hat \Delta_1, \widehat{\mathrm{se}}_1)`.
3. Admissible mixture family :math:`\mathcal{Q}_\epsilon`:
   .. math::
       \mathcal{Q}_\epsilon = \left\{ q \in [0, 1]^2 : |q_0 - q_{0,0}| \le \epsilon \right\}
   where :math:`d_{\mathrm{TV}}(q, q_{\text{base}}) = |q_0 - q_{0,0}| \le \epsilon`.
4. Plug-in difference and uncertainty under mixture :math:`q`:
   .. math::
       \hat \Delta(q) = q_0 \hat \Delta_0 + (1 - q_0) \hat \Delta_1, \\
       \widehat{\mathrm{se}}_\Delta(q) = \sqrt{q_0^2 \widehat{\mathrm{se}}_0^2
       + (1 - q_0)^2 \widehat{\mathrm{se}}_1^2}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from reversallab.contracts.validation import FloatArray, IntArray
from reversallab.evaluation.estimators import _normal_quantile
from reversallab.reversal.decision import DecisionResult, evaluate_decision


@dataclass(frozen=True)
class StratumPartition:
    """Hyperplane split on a single context coordinate."""

    split_dim: int = 0
    split_threshold: float = 0.0

    def assign(self, contexts: FloatArray) -> IntArray:
        """Assign contexts to stratum 0 (x < threshold) or 1 (x >= threshold)."""
        if contexts.ndim != 2:
            raise ValueError(f"Contexts must be 2D; got shape {contexts.shape}.")
        if self.split_dim < 0 or self.split_dim >= contexts.shape[1]:
            raise ValueError(
                f"split_dim {self.split_dim} out of bounds for context dim {contexts.shape[1]}."
            )
        assigned: IntArray = (contexts[:, self.split_dim] >= self.split_threshold).astype(np.int64)
        return assigned


@dataclass(frozen=True)
class StratumStatistics:
    """Empirical statistics of paired differences within a stratum."""

    stratum_id: int
    n_samples: int
    empirical_weight: float
    mean_diff: float
    sample_variance: float
    standard_error: float


@dataclass(frozen=True)
class TwoStratumMixtureModel:
    """Two-stratum context mixture model under total variation perturbation."""

    partition: StratumPartition
    stratum_0: StratumStatistics
    stratum_1: StratumStatistics
    baseline_q0: float
    baseline_q1: float
    is_known_population_mixture: bool = False

    @property
    def inference_mode(self) -> str:
        """Statistical interpretation of the baseline mixture."""
        if self.is_known_population_mixture:
            return "known_population_mixture"
        return "conditional_on_sample_mixture"

    def evaluate_mixture(
        self,
        q0: float,
        *,
        confidence_level: float = 0.95,
    ) -> tuple[float, float, float, float]:
        r"""Compute difference estimate and confidence bounds under mixture :math:`q_0`.

        Parameters
        ----------
        q0:
            Mixture weight for stratum 0 in [0, 1]. The weight for stratum 1 is :math:`1 - q_0`.
        confidence_level:
            Confidence level :math:`1 - \alpha`. Default 0.95.

        Returns
        -------
        tuple[float, float, float, float]:
            ``(point_estimate, standard_error, ci_lower, ci_upper)``.
        """
        if not (0.0 <= q0 <= 1.0):
            raise ValueError(f"Mixture weight q0 must be in [0, 1]; got {q0}.")

        q1 = 1.0 - q0
        point_est = q0 * self.stratum_0.mean_diff + q1 * self.stratum_1.mean_diff
        var_mixture = (q0**2) * (self.stratum_0.standard_error**2) + (q1**2) * (
            self.stratum_1.standard_error**2
        )
        se = float(math.sqrt(var_mixture))

        z = _normal_quantile(confidence_level)
        ci_lower = point_est - z * se
        ci_upper = point_est + z * se
        return point_est, se, ci_lower, ci_upper

    def evaluate_mixture_decision(
        self,
        q0: float,
        margin: float,
        *,
        confidence_level: float = 0.95,
        diagnostic_passed: bool = True,
        diagnostic_reason: str | None = None,
    ) -> DecisionResult:
        """Evaluate deployment decision under mixture weight q0."""
        pt, _, ci_lower, ci_upper = self.evaluate_mixture(q0, confidence_level=confidence_level)
        return evaluate_decision(
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            point_estimate=pt,
            margin=margin,
            confidence_level=confidence_level,
            diagnostic_passed=diagnostic_passed,
            diagnostic_reason=diagnostic_reason,
        )

    def total_variation_distance(self, q0: float) -> float:
        """Compute total variation distance between mixture (q0, 1-q0) and baseline."""
        return abs(q0 - self.baseline_q0)


def create_two_stratum_model(
    contexts: FloatArray,
    diff_scores: FloatArray,
    *,
    partition: StratumPartition | None = None,
    target_mixture: float | None = None,
) -> TwoStratumMixtureModel:
    """Construct TwoStratumMixtureModel from context matrix and paired difference scores.

    Parameters
    ----------
    contexts:
        (N, D) context array.
    diff_scores:
        (N,) paired difference scores.
    partition:
        Hyperplane partition specification.
    target_mixture:
        If provided (e.g. 0.50), the baseline mixture is treated as a known, fixed
        deployment-population mixture. If None, the empirical proportion n_0/N is used
        and inference is conditional on the observed sample allocation.
    """
    if partition is None:
        partition = StratumPartition(split_dim=0, split_threshold=0.0)

    n = len(diff_scores)
    if len(contexts) != n:
        raise ValueError(f"Contexts length ({len(contexts)}) must match diff_scores length ({n}).")

    assignments = partition.assign(contexts)
    idx_0 = np.where(assignments == 0)[0]
    idx_1 = np.where(assignments == 1)[0]

    if len(idx_0) < 2:
        raise ValueError(
            f"Stratum 0 has fewer than 2 observations ({len(idx_0)}); cannot compute variance."
        )
    if len(idx_1) < 2:
        raise ValueError(
            f"Stratum 1 has fewer than 2 observations ({len(idx_1)}); cannot compute variance."
        )

    scores_0 = diff_scores[idx_0]
    scores_1 = diff_scores[idx_1]

    mean_0 = float(np.mean(scores_0))
    var_0 = float(np.var(scores_0, ddof=1))
    se_0 = float(math.sqrt(var_0 / len(idx_0)))

    mean_1 = float(np.mean(scores_1))
    var_1 = float(np.var(scores_1, ddof=1))
    se_1 = float(math.sqrt(var_1 / len(idx_1)))

    empirical_q0 = len(idx_0) / n
    empirical_q1 = len(idx_1) / n

    if target_mixture is not None:
        if not (0.0 <= target_mixture <= 1.0):
            raise ValueError(f"target_mixture must be in [0, 1]; got {target_mixture}.")
        baseline_q0 = float(target_mixture)
        baseline_q1 = 1.0 - baseline_q0
        is_known = True
    else:
        baseline_q0 = empirical_q0
        baseline_q1 = empirical_q1
        is_known = False

    s0 = StratumStatistics(
        stratum_id=0,
        n_samples=len(idx_0),
        empirical_weight=empirical_q0,
        mean_diff=mean_0,
        sample_variance=var_0,
        standard_error=se_0,
    )
    s1 = StratumStatistics(
        stratum_id=1,
        n_samples=len(idx_1),
        empirical_weight=empirical_q1,
        mean_diff=mean_1,
        sample_variance=var_1,
        standard_error=se_1,
    )

    return TwoStratumMixtureModel(
        partition=partition,
        stratum_0=s0,
        stratum_1=s1,
        baseline_q0=baseline_q0,
        baseline_q1=baseline_q1,
        is_known_population_mixture=is_known,
    )
