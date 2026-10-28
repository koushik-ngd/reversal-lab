"""Experiment 001: Monte Carlo validation of estimators against synthetic ground truth.

For every scenario and sample size, ``n_replications`` independent logged datasets
are drawn and each estimator is applied. Results summarise bias, RMSE, and
confidence-interval coverage relative to the reference policy value, together with
how often each diagnostic warning fired.

Seed derivation (stable under reordering or adding scenarios):

* environment parameters: ``SeedSequence([seed, 0])``
* reference values:       ``SeedSequence([seed, 1, crc32(scenario)])``
* replicate ``r``:          ``SeedSequence([seed, 2, crc32(scenario), n_samples, r])``
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
from reversallab.evaluation.diagnostics import DiagnosticThresholds, diagnose
from reversallab.evaluation.estimators import ESTIMATORS
from reversallab.exceptions import DegenerateWeightsError
from reversallab.experiments.config import (
    EstimatorValidationConfig,
    ScenarioConfig,
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
    covered = (np.asarray(ci_lower) <= true_value) & (true_value <= np.asarray(ci_upper))
    sd = float(v.std(ddof=1)) if n_ok > 1 else math.nan
    coverage = float(covered.mean())
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
        "ci_coverage_monte_carlo_se": math.sqrt(coverage * (1 - coverage) / n_ok),
    }


def _run_cell(
    config: EstimatorValidationConfig,
    scenario: ScenarioConfig,
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

        values_this_rep: list[float] = []
        for name in config.estimators:
            try:
                est = ESTIMATORS[name](data, eval_dist, confidence_level=config.confidence_level)
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
                s["value"], s["lo"], s["hi"], s["se"], true_value, failures[name]
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
        "mean_abs_estimator_disagreement": float(np.mean(disagreements)) if disagreements else None,
    }
    return cell, warnings


def run_estimator_validation(config: EstimatorValidationConfig) -> ExperimentOutput:
    """Run experiment 001. Pure function of ``config`` apart from wall-clock timings."""
    t_start = time.perf_counter()
    env = build_environment(config.environment, _rng(config.seed, 0))
    thresholds = DiagnosticThresholds(**config.diagnostics.model_dump())

    scenarios_out: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    timings: dict[str, float] = {}

    for scenario in config.scenarios:
        logging_policy = build_policy(scenario.logging_policy, env)
        evaluation_policy = build_policy(scenario.evaluation_policy, env)
        ref_rng = _rng(config.seed, 1, _stable_key(scenario.name))
        eval_ref = env.policy_value(
            evaluation_policy, ref_rng, n_contexts=config.reference_n_contexts
        )
        log_ref = env.policy_value(logging_policy, ref_rng, n_contexts=config.reference_n_contexts)
        logger.info(
            "scenario=%s true_value=%.6f mc_se=%.2e logging_value=%.6f",
            scenario.name,
            eval_ref.value,
            eval_ref.monte_carlo_standard_error,
            log_ref.value,
        )

        cells = []
        for n in scenario.n_samples:
            t0 = time.perf_counter()
            cell, warnings = _run_cell(
                config,
                scenario,
                n,
                env,
                logging_policy,
                evaluation_policy,
                eval_ref.value,
                thresholds,
            )
            timings[f"{scenario.name}/n={n}"] = time.perf_counter() - t0
            logger.info(
                "scenario=%s n=%d done seconds=%.2f",
                scenario.name,
                n,
                timings[f"{scenario.name}/n={n}"],
            )
            cells.append(cell)
            all_warnings.extend(warnings)

        scenarios_out.append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "evaluation_policy_value": {
                    "value": eval_ref.value,
                    "monte_carlo_se": eval_ref.monte_carlo_standard_error,
                    "n_contexts": eval_ref.n_contexts,
                },
                "logging_policy_value": {
                    "value": log_ref.value,
                    "monte_carlo_se": log_ref.monte_carlo_standard_error,
                    "n_contexts": log_ref.n_contexts,
                },
                "cells": cells,
            }
        )

    runtime = {
        "total_seconds": time.perf_counter() - t_start,
        "cell_seconds": timings,
    }
    return ExperimentOutput(
        results={"scenarios": scenarios_out}, warnings=all_warnings, runtime=runtime
    )
