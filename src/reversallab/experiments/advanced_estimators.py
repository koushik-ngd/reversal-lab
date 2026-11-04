"""Experiment 003: Advanced Estimator Validation (IPS, SNIPS, DM, DR).

Compares importance sampling estimators (IPS, SNIPS) and model-based / doubly robust
estimators (DM, DR) across sample sizes, overlap conditions, reward specifications,
and reward types (binary and continuous/gaussian).

K-fold cross-fitting is used for reward modeling to ensure leakage-safe out-of-fold
predictions for DM and DR.
"""

from __future__ import annotations

import logging
import math
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.evaluation.cross_fitting import cross_fit_reward_model
from reversallab.evaluation.diagnostics import DiagnosticThresholds, diagnose
from reversallab.evaluation.estimators import dm, dr, ips, snips
from reversallab.evaluation.reward_models import (
    MisspecifiedLinearRewardModel,
    RewardModel,
    StratifiedLinearRewardModel,
)
from reversallab.exceptions import DegenerateWeightsError
from reversallab.experiments.config import (
    AdvancedEstimatorValidationConfig,
    AdvancedScenarioConfig,
    build_environment,
    build_policy,
)
from reversallab.policies.base import Policy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentOutput:
    results: dict[str, Any]
    warnings: list[str]
    runtime: dict[str, Any]


def _stable_key(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


def _rng(*entropy: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(list(entropy)))


def _summarise_estimator(
    name: str,
    values: list[float],
    ci_lower: list[float],
    ci_upper: list[float],
    ses: list[float],
    true_value: float,
    n_failed: int,
) -> dict[str, Any]:
    n_ok = len(values)
    if n_ok == 0:
        return {"n_successful": 0, "n_failed": n_failed}
    v = np.asarray(values)
    errors = v - true_value
    sd = float(v.std(ddof=1)) if n_ok > 1 else math.nan

    # Wald coverage is reported ONLY for IPS and SNIPS.
    # For DM and DR, confidence interval coverage is NOT reported until appropriate
    # uncertainty estimators (e.g. bootstrap/cross-fitted variance) are implemented and validated.
    coverage: float | None = None
    coverage_mc_se: float | None = None
    if name in ("ips", "snips"):
        covered = (np.asarray(ci_lower) <= true_value) & (true_value <= np.asarray(ci_upper))
        cov_val = float(covered.mean())
        coverage = cov_val
        coverage_mc_se = math.sqrt(cov_val * (1.0 - cov_val) / n_ok)

    return {
        "n_successful": n_ok,
        "n_failed": n_failed,
        "mean_estimate": float(v.mean()),
        "bias": float(errors.mean()),
        "bias_monte_carlo_se": sd / math.sqrt(n_ok) if n_ok > 1 else math.nan,
        "empirical_sd": sd,
        "mean_reported_se": float(np.mean(ses)),
        "rmse": float(math.sqrt(float(np.mean(errors**2)))),
        "ci_coverage": coverage,
        "ci_coverage_monte_carlo_se": coverage_mc_se,
    }


def _run_cell(
    config: AdvancedEstimatorValidationConfig,
    scenario: AdvancedScenarioConfig,
    n_samples: int,
    env: SyntheticBanditEnvironment,
    logging_policy: Policy,
    evaluation_policy: Policy,
    true_value: float,
    thresholds: DiagnosticThresholds,
) -> tuple[dict[str, Any], list[str]]:
    key = _stable_key(scenario.name)
    per_est: dict[str, dict[str, list[float]]] = {
        e: {"value": [], "lo": [], "hi": [], "se": []} for e in config.estimators
    }
    failures: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    ess_fractions: list[float] = []
    max_weights: list[float] = []
    disagreements: list[float] = []

    needs_q_hat = any(e in ("dm", "dr") for e in config.estimators)

    for r in range(config.n_replications):
        rng = _rng(config.seed, 2, key, n_samples, r)
        data = env.generate_logged_data(
            n_samples,
            logging_policy,
            rng,
            include_logging_action_dist=config.include_logging_action_dist,
        )
        eval_dist = evaluation_policy.action_distribution(data.context)
        report = diagnose(data, eval_dist, thresholds)
        warning_counts.update(set(report.warning_codes))
        ess_fractions.append(report.weights.effective_sample_size_fraction)
        max_weights.append(report.weights.max)

        q_hat: np.ndarray | None = None
        if needs_q_hat:
            model_seed = int(_stable_key(f"{scenario.name}_{n_samples}_{r}") % (2**31))

            def model_factory() -> RewardModel:
                if scenario.reward_model.specification == "correct":
                    return StratifiedLinearRewardModel(
                        n_actions=env.n_actions,
                        reward_type=env.reward_type,
                        alpha=scenario.reward_model.alpha,
                        standardize=scenario.reward_model.standardize,
                    )
                return MisspecifiedLinearRewardModel(
                    n_actions=env.n_actions,
                    reward_type=env.reward_type,
                )

            q_hat = cross_fit_reward_model(
                data, model_factory, n_folds=config.n_folds, seed=model_seed
            )

        values_this_rep: list[float] = []
        for name in config.estimators:
            try:
                if name == "ips":
                    est = ips(data, eval_dist, confidence_level=config.confidence_level)
                elif name == "snips":
                    est = snips(data, eval_dist, confidence_level=config.confidence_level)
                elif name == "dm":
                    assert q_hat is not None
                    est = dm(data, eval_dist, q_hat, confidence_level=config.confidence_level)
                elif name == "dr":
                    assert q_hat is not None
                    est = dr(data, eval_dist, q_hat, confidence_level=config.confidence_level)
                else:
                    raise ValueError(f"Unknown estimator: {name}")
            except DegenerateWeightsError:
                failures[name] += 1
                continue
            store = per_est[name]
            store["value"].append(est.value)
            store["lo"].append(est.ci_lower)
            store["hi"].append(est.ci_upper)
            store["se"].append(est.standard_error)
            values_this_rep.append(est.value)

        if len(values_this_rep) >= 2:
            disagreements.append(max(values_this_rep) - min(values_this_rep))

    warnings: list[str] = []
    for name, count in failures.items():
        warnings.append(
            f"{scenario.name} n={n_samples}: {name} undefined (degenerate weights) in "
            f"{count}/{config.n_replications} replications; summaries exclude them."
        )

    cell = {
        "scenario": scenario.name,
        "n_samples": n_samples,
        "estimators": {
            name: _summarise_estimator(
                name, s["value"], s["lo"], s["hi"], s["se"], true_value, failures[name]
            )
            for name, s in per_est.items()
        },
        "diagnostics": {
            "mean_ess_fraction": float(np.mean(ess_fractions)),
            "median_ess_fraction": float(np.median(ess_fractions)),
            "median_max_weight": float(np.median(max_weights)),
            "max_max_weight": float(np.max(max_weights)),
            "warning_rates": {
                code: count / config.n_replications
                for code, count in sorted(warning_counts.items())
            },
        },
        "mean_pairwise_disagreement": float(np.mean(disagreements)) if disagreements else 0.0,
    }
    return cell, warnings


def run_advanced_estimator_validation(
    config: AdvancedEstimatorValidationConfig,
) -> ExperimentOutput:
    """Run experiment 003."""
    t0 = time.perf_counter()
    scenario_results: list[dict[str, Any]] = []
    all_warnings: list[str] = []

    for sc in config.scenarios:
        t_sc = time.perf_counter()
        env_rng = _rng(config.seed, 0, _stable_key(sc.name))
        env = build_environment(sc.environment, env_rng)
        logging_policy = build_policy(sc.logging_policy, env)
        evaluation_policy = build_policy(sc.evaluation_policy, env)

        ref_rng = _rng(config.seed, 1, _stable_key(sc.name))
        ref = env.policy_value(evaluation_policy, ref_rng, n_contexts=config.reference_n_contexts)
        logger.info(
            "scenario=%s V(pi_e)=%.5f±%.5f",
            sc.name,
            ref.value,
            ref.monte_carlo_standard_error,
        )

        thresholds = DiagnosticThresholds(
            min_ess_fraction=sc.diagnostics.min_ess_fraction,
            max_weight_share=sc.diagnostics.max_weight_share,
            mean_weight_z=sc.diagnostics.mean_weight_z,
        )

        cells: list[dict[str, Any]] = []
        for n_samples in sc.n_samples:
            cell, warnings = _run_cell(
                config,
                sc,
                n_samples,
                env,
                logging_policy,
                evaluation_policy,
                ref.value,
                thresholds,
            )
            cells.append(cell)
            all_warnings.extend(warnings)

        scenario_results.append(
            {
                "name": sc.name,
                "description": sc.description,
                "evaluation_policy_value": {
                    "value": ref.value,
                    "monte_carlo_standard_error": ref.monte_carlo_standard_error,
                    "n_contexts": ref.n_contexts,
                },
                "reward_model": sc.reward_model.model_dump(),
                "cells": cells,
                "runtime_seconds": round(time.perf_counter() - t_sc, 3),
            }
        )

    return ExperimentOutput(
        results={"scenarios": scenario_results},
        warnings=all_warnings,
        runtime={"wall_seconds": round(time.perf_counter() - t0, 3)},
    )
