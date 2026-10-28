r"""Diagnostics for importance-weighted off-policy evaluation.

Diagnostics *report* problems; they never raise for statistical concerns. Each
warning carries a stable machine-readable ``code`` so experiments can count how
often a condition fires.

Key quantities (weights :math:`w_i`, :math:`N` observations):

* Effective sample size (Kish): :math:`\mathrm{ESS} = (\sum w_i)^2 / \sum w_i^2`.
* Max weight share: :math:`\max_i w_i / \sum_i w_i`.
* Mean weight: under full support :math:`E_{\pi_0}[w] = 1`; a mean significantly
  below 1 indicates evaluation mass on actions the logger does not take, and a
  significant deviation in either direction indicates mis-specified propensities.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import FloatArray
from reversallab.evaluation.estimators import PolicyValueEstimate
from reversallab.evaluation.weights import (
    SupportViolations,
    find_support_violations,
    importance_weights,
    validate_evaluation_action_dist,
)


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Warning thresholds. Defaults are conventional heuristics, not validated cut-offs."""

    min_ess_fraction: float = 0.1
    max_weight_share: float = 0.05
    mean_weight_z: float = 4.0
    """Warn when ``|mean(w) - 1| / se(mean(w))`` exceeds this value."""


@dataclass(frozen=True)
class DiagnosticWarning:
    code: str
    message: str


@dataclass(frozen=True)
class WeightDiagnostics:
    n_samples: int
    mean: float
    std: float
    min: float
    max: float
    quantile_50: float
    quantile_90: float
    quantile_99: float
    effective_sample_size: float
    effective_sample_size_fraction: float
    max_weight_share: float
    zero_weight_fraction: float
    mean_weight_standard_error: float


@dataclass(frozen=True)
class SupportDiagnostics:
    support_verifiable: bool
    violations: SupportViolations | None
    logged_action_counts: list[int]
    evaluation_action_mass: list[float]
    """Average evaluation probability per action over the logged contexts."""
    evaluation_mass_on_unlogged_actions: float
    """Average evaluation probability on actions never observed in this log."""


@dataclass(frozen=True)
class DiagnosticReport:
    weights: WeightDiagnostics
    support: SupportDiagnostics
    warnings: tuple[DiagnosticWarning, ...] = field(default_factory=tuple)

    @property
    def warning_codes(self) -> tuple[str, ...]:
        return tuple(w.code for w in self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def weight_diagnostics(weights: FloatArray) -> WeightDiagnostics:
    w = np.asarray(weights, dtype=np.float64)
    n = int(w.size)
    total = float(w.sum())
    total_sq = float(np.dot(w, w))
    ess = total * total / total_sq if total_sq > 0 else 0.0
    q50, q90, q99 = (float(q) for q in np.quantile(w, [0.5, 0.9, 0.99]))
    std = float(w.std(ddof=1)) if n > 1 else math.nan
    w_max = float(np.max(w))
    return WeightDiagnostics(
        n_samples=n,
        mean=float(w.mean()),
        std=std,
        min=float(np.min(w)),
        max=w_max,
        quantile_50=q50,
        quantile_90=q90,
        quantile_99=q99,
        effective_sample_size=ess,
        effective_sample_size_fraction=ess / n,
        max_weight_share=w_max / total if total > 0 else math.nan,
        zero_weight_fraction=float(np.mean(w == 0.0)),
        mean_weight_standard_error=std / math.sqrt(n) if n > 1 else math.nan,
    )


def diagnose(
    data: LoggedBanditData,
    evaluation_action_dist: npt.ArrayLike,
    thresholds: DiagnosticThresholds | None = None,
) -> DiagnosticReport:
    """Compute weight and support diagnostics for one (data, evaluation policy) pair."""
    t = thresholds or DiagnosticThresholds()
    eval_dist = validate_evaluation_action_dist(data, evaluation_action_dist)
    w = importance_weights(data, eval_dist, require_support=False)
    wd = weight_diagnostics(w)

    counts = np.bincount(data.action, minlength=data.n_actions)
    unlogged = counts == 0
    violations = find_support_violations(data, eval_dist)
    sd = SupportDiagnostics(
        support_verifiable=violations is not None,
        violations=violations,
        logged_action_counts=[int(c) for c in counts],
        evaluation_action_mass=[float(m) for m in eval_dist.mean(axis=0)],
        evaluation_mass_on_unlogged_actions=float(eval_dist[:, unlogged].sum(axis=1).mean()),
    )

    warnings: list[DiagnosticWarning] = []
    if violations is None:
        warnings.append(
            DiagnosticWarning(
                "support_unverifiable",
                "Full logging distribution not provided; support of the evaluation policy can "
                "only be checked for logged actions.",
            )
        )
    elif violations.n_violating_pairs > 0:
        warnings.append(
            DiagnosticWarning(
                "support_violation",
                f"Evaluation policy has mass {violations.mean_unsupported_evaluation_mass:.4g} "
                "on actions with zero logging probability; IPS/SNIPS are not identified.",
            )
        )
    if sd.evaluation_mass_on_unlogged_actions > 0:
        warnings.append(
            DiagnosticWarning(
                "unlogged_action_mass",
                "Evaluation policy places average mass "
                f"{sd.evaluation_mass_on_unlogged_actions:.4g} on {int(unlogged.sum())} "
                "action(s) never observed in the log.",
            )
        )
    if wd.effective_sample_size_fraction < t.min_ess_fraction:
        warnings.append(
            DiagnosticWarning(
                "low_effective_sample_size",
                f"ESS = {wd.effective_sample_size:.1f} ({wd.effective_sample_size_fraction:.2%} "
                f"of N) is below {t.min_ess_fraction:.0%}; estimates rely on few observations.",
            )
        )
    if not math.isnan(wd.max_weight_share) and wd.max_weight_share > t.max_weight_share:
        warnings.append(
            DiagnosticWarning(
                "dominant_weight",
                f"A single observation carries {wd.max_weight_share:.2%} of the total weight.",
            )
        )
    if wd.zero_weight_fraction == 1.0:
        warnings.append(
            DiagnosticWarning(
                "all_weights_zero",
                "Evaluation policy assigns zero probability to every logged action.",
            )
        )
    deviation = wd.mean - 1.0
    se = wd.mean_weight_standard_error
    if (se > 0 and abs(deviation) / se > t.mean_weight_z) or (se == 0 and abs(deviation) > 1e-9):
        warnings.append(
            DiagnosticWarning(
                "mean_weight_deviation",
                f"Mean importance weight {wd.mean:.4f} differs from its expectation 1 under full "
                "support; check propensities and support.",
            )
        )
    return DiagnosticReport(weights=wd, support=sd, warnings=tuple(warnings))


def estimator_disagreement(estimates: list[PolicyValueEstimate]) -> float:
    """Largest absolute pairwise difference between point estimates (``nan`` if < 2)."""
    values = [e.value for e in estimates]
    if len(values) < 2:
        return math.nan
    return float(max(values) - min(values))
