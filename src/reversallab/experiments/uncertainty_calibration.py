"""Experiment 004: Statistical uncertainty, confidence interval calibration, and reliability.

Evaluates empirical coverage, mean interval width, bias, RMSE, and failure rates across
asymptotic Wald and nonparametric bootstrap intervals under good and poor overlap,
correct and misspecified models, and binary and Gaussian reward settings.
"""

from __future__ import annotations

import logging
import math
import time
import zlib
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.evaluation.bootstrap import bootstrap_evaluate
from reversallab.evaluation.cross_fitting import cross_fit_reward_model
from reversallab.evaluation.diagnostics import (
    DiagnosticThresholds,
    diagnose,
    diagnose_interval,
)
from reversallab.evaluation.estimators import (
    dm,
    ips,
    snips,
)
from reversallab.evaluation.influence_scores import evaluate_influence_inference
from reversallab.evaluation.reward_models import (
    MisspecifiedLinearRewardModel,
    RewardModel,
    StratifiedLinearRewardModel,
)
from reversallab.exceptions import DegenerateWeightsError
from reversallab.experiments.config import (
    AdvancedScenarioConfig,
    UncertaintyCalibrationConfig,
    build_environment,
    build_policy,
)
from reversallab.experiments.estimator_validation import ExperimentOutput
from reversallab.policies.base import Policy

logger = logging.getLogger(__name__)


def _stable_key(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


def _rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *parts]))


def _build_model_factory(
    sc: AdvancedScenarioConfig, env: SyntheticBanditEnvironment
) -> Callable[[], RewardModel]:
    spec = sc.reward_model
    if spec.specification == "correct":
        return lambda: StratifiedLinearRewardModel(
            n_actions=env.n_actions,
            reward_type=env.reward_type,
            standardize=spec.standardize,
            alpha=spec.alpha,
        )
    return lambda: MisspecifiedLinearRewardModel(
        n_actions=env.n_actions,
        reward_type=env.reward_type,
    )


def _run_uncertainty_cell(
    config: UncertaintyCalibrationConfig,
    sc: AdvancedScenarioConfig,
    n_samples: int,
    env: SyntheticBanditEnvironment,
    logging_policy: Policy,
    evaluation_policy: Policy,
    true_policy_value: float,
    thresholds: DiagnosticThresholds,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    model_factory = _build_model_factory(sc, env)

    # Collect outcomes per method
    method_results: dict[str, dict[str, list[float]]] = {
        m: {
            "estimates": [],
            "reported_ses": [],
            "ci_lowers": [],
            "ci_uppers": [],
            "widths": [],
            "covered": [],
            "failed": [],
        }
        for m in config.uncertainty_methods
    }

    warning_counts: dict[str, int] = defaultdict(int)
    ess_fractions: list[float] = []
    max_weights: list[float] = []

    for r in range(config.n_replications):
        data_rng = _rng(config.seed, 2, _stable_key(sc.name), n_samples, r)
        data = env.generate_logged_data(
            n_samples,
            logging_policy,
            data_rng,
            include_logging_action_dist=True,
        )
        eval_dist = evaluation_policy.action_distribution(data.context)

        # Run dataset diagnostics
        report = diagnose(data, eval_dist, thresholds)
        ess_fractions.append(report.weights.effective_sample_size_fraction)
        max_weights.append(report.weights.max)
        for code in report.warning_codes:
            warning_counts[code] += 1

        # Pre-compute cross-fitted reward models for Wald DM/DR methods
        needs_model = any(
            m in config.uncertainty_methods
            for m in [
                "dm_naive_wald",
                "dr_influence_wald",
            ]
        )
        q_hat = None
        if needs_model:
            cf_seed = int(
                _rng(config.seed, 3, _stable_key(sc.name), n_samples, r).integers(0, 10**8)
            )
            q_hat = cross_fit_reward_model(
                data, model_factory, n_folds=config.n_folds, seed=cf_seed
            )

        # Reusable cache for bootstrap refit if multiple refit intervals requested
        boot_dm_res = None
        boot_dr_res = None

        for m in config.uncertainty_methods:
            point_est: float = math.nan
            se: float = math.nan
            ci_lower: float = math.nan
            ci_upper: float = math.nan

            try:
                if m == "ips_wald":
                    est = ips(data, eval_dist, confidence_level=config.confidence_level)
                    point_est = est.value
                    se = est.standard_error
                    ci_lower = est.ci_lower
                    ci_upper = est.ci_upper
                elif m == "snips_wald":
                    est = snips(data, eval_dist, confidence_level=config.confidence_level)
                    point_est = est.value
                    se = est.standard_error
                    ci_lower = est.ci_lower
                    ci_upper = est.ci_upper
                elif m == "dm_naive_wald":
                    assert q_hat is not None
                    est = dm(data, eval_dist, q_hat, confidence_level=config.confidence_level)
                    point_est = est.value
                    se = est.standard_error
                    ci_lower = est.ci_lower
                    ci_upper = est.ci_upper
                elif m == "dr_influence_wald":
                    assert q_hat is not None
                    inf_res = evaluate_influence_inference(
                        "dr", data, eval_dist, q_hat, confidence_level=config.confidence_level
                    )
                    point_est = inf_res.point_estimate
                    se = inf_res.standard_error
                    ci_lower = inf_res.ci_lower
                    ci_upper = inf_res.ci_upper
                elif m == "ips_bootstrap_percentile":
                    b_seed = int(
                        _rng(config.seed, 4, _stable_key(sc.name), n_samples, r).integers(0, 10**8)
                    )
                    b_res = bootstrap_evaluate(
                        "ips",
                        data,
                        eval_dist,
                        n_bootstraps=config.n_bootstraps,
                        confidence_level=config.confidence_level,
                        seed=b_seed,
                    )
                    point_est = b_res.point_estimate
                    se = b_res.bootstrap_standard_error
                    ci_lower, ci_upper = b_res.percentile_ci
                elif m == "snips_bootstrap_percentile":
                    b_seed = int(
                        _rng(config.seed, 5, _stable_key(sc.name), n_samples, r).integers(0, 10**8)
                    )
                    b_res = bootstrap_evaluate(
                        "snips",
                        data,
                        eval_dist,
                        n_bootstraps=config.n_bootstraps,
                        confidence_level=config.confidence_level,
                        seed=b_seed,
                    )
                    point_est = b_res.point_estimate
                    se = b_res.bootstrap_standard_error
                    ci_lower, ci_upper = b_res.percentile_ci
                elif m == "dm_bootstrap_refit_percentile":
                    if boot_dm_res is None:
                        b_seed = int(
                            _rng(config.seed, 6, _stable_key(sc.name), n_samples, r).integers(
                                0, 10**8
                            )
                        )
                        boot_dm_res = bootstrap_evaluate(
                            "dm",
                            data,
                            eval_dist,
                            model_factory=model_factory,
                            n_bootstraps=config.n_bootstraps,
                            n_folds=config.n_folds,
                            confidence_level=config.confidence_level,
                            refit=True,
                            seed=b_seed,
                        )
                    point_est = boot_dm_res.point_estimate
                    se = boot_dm_res.bootstrap_standard_error
                    ci_lower, ci_upper = boot_dm_res.percentile_ci
                elif m == "dr_bootstrap_refit_percentile":
                    if boot_dr_res is None:
                        b_seed = int(
                            _rng(config.seed, 7, _stable_key(sc.name), n_samples, r).integers(
                                0, 10**8
                            )
                        )
                        boot_dr_res = bootstrap_evaluate(
                            "dr",
                            data,
                            eval_dist,
                            model_factory=model_factory,
                            n_bootstraps=config.n_bootstraps,
                            n_folds=config.n_folds,
                            confidence_level=config.confidence_level,
                            refit=True,
                            seed=b_seed,
                        )
                    point_est = boot_dr_res.point_estimate
                    se = boot_dr_res.bootstrap_standard_error
                    ci_lower, ci_upper = boot_dr_res.percentile_ci
                elif m == "dr_bootstrap_refit_basic":
                    if boot_dr_res is None:
                        b_seed = int(
                            _rng(config.seed, 7, _stable_key(sc.name), n_samples, r).integers(
                                0, 10**8
                            )
                        )
                        boot_dr_res = bootstrap_evaluate(
                            "dr",
                            data,
                            eval_dist,
                            model_factory=model_factory,
                            n_bootstraps=config.n_bootstraps,
                            n_folds=config.n_folds,
                            confidence_level=config.confidence_level,
                            refit=True,
                            seed=b_seed,
                        )
                    point_est = boot_dr_res.point_estimate
                    se = boot_dr_res.bootstrap_standard_error
                    ci_lower, ci_upper = boot_dr_res.basic_ci
                else:
                    raise ValueError(f"unknown method {m}")

            except (DegenerateWeightsError, ValueError) as exc:
                logger.debug("Method %s failed on replication %d: %s", m, r, exc)

            # Check validity
            is_valid = (
                math.isfinite(point_est)
                and math.isfinite(ci_lower)
                and math.isfinite(ci_upper)
                and ci_lower <= ci_upper
            )

            if is_valid:
                covered = 1.0 if (ci_lower <= true_policy_value <= ci_upper) else 0.0
                width = ci_upper - ci_lower
                method_results[m]["estimates"].append(point_est)
                method_results[m]["reported_ses"].append(se)
                method_results[m]["ci_lowers"].append(ci_lower)
                method_results[m]["ci_uppers"].append(ci_upper)
                method_results[m]["widths"].append(width)
                method_results[m]["covered"].append(covered)
                method_results[m]["failed"].append(0.0)

                # Check interval diagnostics
                int_warns = diagnose_interval((ci_lower, ci_upper), se, thresholds=thresholds)
                for iw in int_warns:
                    warning_counts[iw.code] += 1
            else:
                method_results[m]["failed"].append(1.0)

    # Summarize each method
    summary_methods: dict[str, Any] = {}
    for m in config.uncertainty_methods:
        d = method_results[m]
        n_ok = len(d["estimates"])
        n_fail = int(sum(d["failed"]))

        if n_ok > 0:
            est_arr = np.array(d["estimates"], dtype=np.float64)
            width_arr = np.array(d["widths"], dtype=np.float64)
            cov_arr = np.array(d["covered"], dtype=np.float64)
            se_arr = np.array(d["reported_ses"], dtype=np.float64)

            cov = float(np.mean(cov_arr))
            mc_se_cov = math.sqrt(cov * (1.0 - cov) / n_ok) if n_ok > 1 else math.nan
            bias = float(np.mean(est_arr) - true_policy_value)
            emp_sd = float(np.std(est_arr, ddof=1)) if n_ok > 1 else 0.0
            bias_mc_se = emp_sd / math.sqrt(n_ok) if n_ok > 1 else 0.0
            rmse = float(math.sqrt(np.mean((est_arr - true_policy_value) ** 2)))
            mean_width = float(np.mean(width_arr))
            median_width = float(np.median(width_arr))
            mean_reported_se = float(np.mean(se_arr)) if np.all(np.isfinite(se_arr)) else math.nan

            summary_methods[m] = {
                "n_successful": n_ok,
                "n_failed": n_fail,
                "point_estimate_mean": float(np.mean(est_arr)),
                "bias": bias,
                "bias_monte_carlo_se": bias_mc_se,
                "empirical_sd": emp_sd,
                "mean_reported_se": mean_reported_se,
                "rmse": rmse,
                "empirical_coverage": cov,
                "coverage_monte_carlo_se": mc_se_cov,
                "mean_interval_width": mean_width,
                "median_interval_width": median_width,
                "interval_failure_rate": n_fail / config.n_replications,
            }
        else:
            summary_methods[m] = {
                "n_successful": 0,
                "n_failed": n_fail,
                "point_estimate_mean": math.nan,
                "bias": math.nan,
                "bias_monte_carlo_se": math.nan,
                "empirical_sd": math.nan,
                "mean_reported_se": math.nan,
                "rmse": math.nan,
                "empirical_coverage": math.nan,
                "coverage_monte_carlo_se": math.nan,
                "mean_interval_width": math.nan,
                "median_interval_width": math.nan,
                "interval_failure_rate": 1.0,
            }

    cell = {
        "scenario": sc.name,
        "n_samples": n_samples,
        "methods": summary_methods,
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
    }
    return cell, warnings


def run_uncertainty_calibration(
    config: UncertaintyCalibrationConfig,
) -> ExperimentOutput:
    """Run experiment 004: uncertainty estimation and confidence interval calibration."""
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
            "scenario=%s true V(pi_e)=%.5f±%.5f",
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
            cell, cell_warns = _run_uncertainty_cell(
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
            all_warnings.extend(cell_warns)

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
