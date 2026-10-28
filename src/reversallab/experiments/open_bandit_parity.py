"""Experiment 002: IPS and SNIPS against Open Bandit Pipeline on the same arrays.

The estimand is the position-level value in ``docs/open_bandit_estimand.md``.
Published full-dataset figures are not a comparison target for this sample.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reversallab.data.open_bandit import (
    OpenBanditData,
    constant_action_probabilities,
    load_open_bandit,
    uniform_evaluation_probabilities,
)
from reversallab.evaluation.estimators import ips, snips
from reversallab.evaluation.obp_reference import obp_point_estimates_many
from reversallab.exceptions import ConfigurationError, DegenerateWeightsError
from reversallab.experiments.config import (
    ConstantActionEvaluationSpec,
    EvaluationPolicySpec,
    OpenBanditParityConfig,
    UniformEvaluationSpec,
)
from reversallab.experiments.metadata import git_revision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParityOutput:
    results: dict[str, Any]
    warnings: list[str]
    runtime: dict[str, Any]
    matched: bool


def _policy_name(spec: EvaluationPolicySpec) -> str:
    if isinstance(spec, UniformEvaluationSpec):
        return "uniform"
    return f"constant_action_{spec.action}"


def _evaluation_probabilities(data: OpenBanditData, spec: EvaluationPolicySpec) -> np.ndarray:
    if isinstance(spec, UniformEvaluationSpec):
        return uniform_evaluation_probabilities(data)
    if isinstance(spec, ConstantActionEvaluationSpec):
        if spec.action >= data.n_actions:
            raise ConfigurationError(
                f"constant action {spec.action} is outside 0..{data.n_actions - 1} "
                f"for {data.provenance.behavior_policy}/{data.provenance.campaign}."
            )
        return constant_action_probabilities(data, spec.action)
    raise ConfigurationError(f"Unknown evaluation policy spec {spec!r}.")


def _within(ours: float, reference: float, *, atol: float, rtol: float) -> bool:
    return abs(ours - reference) <= atol + rtol * abs(reference)


def run_open_bandit_parity(config: OpenBanditParityConfig) -> ParityOutput:
    """Compare ReversalLab and OBP on every configured file, slice, and policy."""
    started = time.perf_counter()
    obd_root = Path(config.obd_root)
    obp_root = Path(config.obp_root)
    cases: list[dict[str, Any]] = []
    rewards: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    pscores: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    dataset_records: list[dict[str, Any]] = []

    for spec in config.datasets:
        data = load_open_bandit(
            obd_root,
            behavior_policy=spec.behavior_policy,
            campaign=spec.campaign,
        )
        provenance = data.provenance
        dataset_records.append(
            {
                "behavior_policy": provenance.behavior_policy,
                "campaign": provenance.campaign,
                "behavior_sha256": provenance.behavior_sha256,
                "item_context_sha256": provenance.item_context_sha256,
                "n_rows": provenance.n_rows,
                "n_actions": provenance.n_actions,
                "logging_distribution": provenance.logging_distribution,
                "n_repeated_timestamps": provenance.n_repeated_timestamps,
                "max_rows_per_timestamp": provenance.max_rows_per_timestamp,
                "n_user_feature_tuples": provenance.n_user_feature_tuples,
                "click_rate": float(data.logged.reward.mean()),
                "position_counts": {
                    str(position): int(np.count_nonzero(data.position == position))
                    for position in (1, 2, 3)
                },
            }
        )
        slices: list[tuple[str, OpenBanditData, list[EvaluationPolicySpec]]] = [
            ("all", data, list(config.evaluation_policies))
        ]
        for position in (1, 2, 3):
            if not config.position_evaluation_policies:
                continue
            sliced = data.select_position(position)
            if sliced.n_samples < 2:
                raise ConfigurationError(
                    f"{spec.behavior_policy}/{spec.campaign} position {position} has "
                    f"{sliced.n_samples} rows. IPS needs at least 2."
                )
            slices.append((str(position), sliced, list(config.position_evaluation_policies)))
        for position_label, subset, policies in slices:
            for policy in policies:
                evaluation = _evaluation_probabilities(subset, policy)
                ours = _our_estimates(subset, evaluation)
                cases.append(
                    {
                        "behavior_policy": spec.behavior_policy,
                        "campaign": spec.campaign,
                        "position": position_label,
                        "evaluation_policy": _policy_name(policy),
                        "n": subset.n_samples,
                        "click_rate": float(subset.logged.reward.mean()),
                        "logging_distribution": subset.provenance.logging_distribution,
                        "reversallab": ours,
                    }
                )
                rewards.append(np.asarray(subset.logged.reward, dtype=np.float64))
                actions.append(np.asarray(subset.logged.action, dtype=np.int64))
                pscores.append(np.asarray(subset.logged.logging_propensity, dtype=np.float64))
                dists.append(np.asarray(evaluation, dtype=np.float64))

    reference = obp_point_estimates_many(
        obp_root,
        reward=rewards,
        action=actions,
        pscore=pscores,
        action_dist=dists,
    )
    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []
    n_mismatches = 0
    max_abs_diff = 0.0
    for case, estimate in zip(cases, reference["estimates"], strict=True):
        obp_values = {"ipw": estimate["ipw"], "snipw": estimate["snipw"]}
        diffs: dict[str, float | None] = {}
        matched = True
        for ours_name, obp_name in (("ips", "ipw"), ("snips", "snipw")):
            ours_value = case["reversallab"][ours_name]
            reference_value = obp_values[obp_name]
            if ours_value is None or not math.isfinite(reference_value):
                pair_ok = ours_value is None and not math.isfinite(reference_value)
                diffs[ours_name] = None
            else:
                gap = abs(float(ours_value) - float(reference_value))
                diffs[ours_name] = gap
                pair_ok = _within(
                    float(ours_value),
                    float(reference_value),
                    atol=config.atol,
                    rtol=config.rtol,
                )
                max_abs_diff = max(max_abs_diff, gap)
            matched = matched and pair_ok
        if not matched:
            n_mismatches += 1
            warnings.append(
                "IPS/SNIPS disagree with OBP IPW/SNIPW for "
                f"{case['behavior_policy']}/{case['campaign']} "
                f"position={case['position']} policy={case['evaluation_policy']} "
                f"diffs={diffs}."
            )
        comparisons.append(
            {
                **{key: case[key] for key in case if key != "reversallab"},
                "reversallab": case["reversallab"],
                "obp": obp_values,
                "abs_diff": diffs,
                "matched": matched,
            }
        )

    obp_git = git_revision(obp_root)
    results = {
        "tolerance": {"atol": config.atol, "rtol": config.rtol},
        "obp_estimators_sha256": reference["estimators_sha256"],
        "obp_lambda": "inf",
        "obp_git": obp_git,
        "datasets": dataset_records,
        "comparisons": comparisons,
        "max_abs_diff": max_abs_diff,
        "n_mismatches": n_mismatches,
        "n_comparisons": len(comparisons),
    }
    if n_mismatches:
        warnings.insert(
            0,
            f"{n_mismatches} of {len(comparisons)} comparisons exceeded "
            f"atol={config.atol} rtol={config.rtol}.",
        )
    elapsed = time.perf_counter() - started
    logger.info(
        "comparisons=%d mismatches=%d max_abs_diff=%s seconds=%.3f",
        len(comparisons),
        n_mismatches,
        max_abs_diff,
        elapsed,
    )
    return ParityOutput(
        results=results,
        warnings=warnings,
        runtime={"seconds": elapsed},
        matched=n_mismatches == 0,
    )


def _our_estimates(data: OpenBanditData, evaluation: np.ndarray) -> dict[str, float | None]:
    ips_value = ips(data.logged, evaluation).value
    try:
        snips_value: float | None = snips(data.logged, evaluation).value
    except DegenerateWeightsError:
        snips_value = None
    return {"ips": ips_value, "snips": snips_value}
