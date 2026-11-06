r"""Experiment 007: Independent Reliability Evaluation Benchmark (M4C).

Preregistered synthetic evaluation benchmark comparing six off-policy deployment methods:
- Method A: Point-estimate-only (naive rule: \hat{\Delta} > \delta)
- Method B: Baseline confidence-bound (L_\Delta(q_0) > \delta)
- Method C: Worst-stratum screening (L_\Delta(q_0) > \delta and min(L0, L1) > 0)
- Method D: Pointwise Decision-Reversal search
  (L_\Delta(q_0) > \delta and \hat{d}_rev* >= \epsilon_deploy)
- Method E: Simultaneous-band Decision-Reversal search (L_simult(q) > \delta over Q_\epsilon)
- Method F: Forward robust-value / worst-case mixture baseline (L_robust(\epsilon_deploy) > \delta)

Validates:
1. Mathematical duality between Method D (inverse search) and Method F (forward DRO).
2. Simultaneous confidence band coverage across the continuous mixture interval Q_\epsilon vs
   pointwise coverage and naive family-wise error inflation over multiple grid points.
3. Separation of evidence-reversal, margin-crossing, and performance-crossing distances.
4. Recommendation, hold, inconclusive, false recommendation, fragile deployment, and missed
   beneficial deployment rates with Monte Carlo Standard Errors (MCSE) and Wilson CIs.
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
from reversallab.evaluation.estimators import _normal_quantile
from reversallab.evaluation.reward_models import (
    MisspecifiedLinearRewardModel,
    RewardModel,
    StratifiedLinearRewardModel,
)
from reversallab.experiments.config import (
    IndependentReliabilityConfig,
    ReversalScenarioConfig,
    build_environment,
    build_policy,
)
from reversallab.policies.base import Policy
from reversallab.reversal.analytical import find_analytical_decision_reversal
from reversallab.reversal.decision import (
    DecisionOutcome,
    evaluate_decision_from_paired_result,
)
from reversallab.reversal.diagnostics import (
    DiagnosticGateThresholds,
    evaluate_diagnostic_gate,
)
from reversallab.reversal.paired import evaluate_paired_difference
from reversallab.reversal.robust_mixture import evaluate_robust_mixture_baseline
from reversallab.reversal.search import find_decision_reversal
from reversallab.reversal.simultaneous import TwoStratumSimultaneousBand
from reversallab.reversal.strata import (
    StratumPartition,
    create_two_stratum_model,
)
from reversallab.reversal.taxonomy import classify_reversal_taxonomy

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


def _mcse_rate(rate: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return float(math.sqrt(max(0.0, rate * (1.0 - rate) / n)))


def _wilson_ci(k: int, n: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    z = _normal_quantile(confidence_level)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))) / denom
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))


def compute_ground_truth_reference(
    env: SyntheticBanditEnvironment,
    cand_policy: Policy,
    base_policy: Policy,
    partition: StratumPartition,
    *,
    n_contexts: int = 50_000,
    margin: float = 0.0,
    deploy_epsilon: float = 0.10,
    max_epsilon: float = 0.40,
    target_q0: float = 0.50,
    seed: int = 999,
) -> dict[str, Any]:
    """Compute high-precision ground truth and theoretical reversal distances."""
    rng = np.random.default_rng(seed)
    contexts = env.sample_contexts(n_contexts, rng)
    assignments = partition.assign(contexts)

    idx_0 = np.where(assignments == 0)[0]
    idx_1 = np.where(assignments == 1)[0]

    cand_dist = cand_policy.action_distribution(contexts)
    base_dist = base_policy.action_distribution(contexts)
    q_matrix = env.expected_reward(contexts)

    v_cand = np.sum(cand_dist * q_matrix, axis=1)
    v_base = np.sum(base_dist * q_matrix, axis=1)
    diff = v_cand - v_base

    d0 = float(np.mean(diff[idx_0]))
    d1 = float(np.mean(diff[idx_1]))
    d_total = target_q0 * d0 + (1.0 - target_q0) * d1

    # Worst-case population difference over Q_epsilon = [target_q0 - eps, target_q0 + eps]
    q_min_deploy = max(0.0, target_q0 - deploy_epsilon)
    q_max_deploy = min(1.0, target_q0 + deploy_epsilon)
    d_at_min = q_min_deploy * d0 + (1.0 - q_min_deploy) * d1
    d_at_max = q_max_deploy * d0 + (1.0 - q_max_deploy) * d1
    worst_delta_deploy = min(d_at_min, d_at_max)

    # Theoretical crossings
    denom = d0 - d1
    d_margin: float | None = None
    if abs(denom) > 1e-12:
        q_m = (margin - d1) / denom
        if 0.0 <= q_m <= 1.0 and abs(q_m - target_q0) <= max_epsilon:
            d_margin = float(abs(q_m - target_q0))

    d_perf: float | None = None
    if abs(denom) > 1e-12:
        q_p = (0.0 - d1) / denom
        if 0.0 <= q_p <= 1.0 and abs(q_p - target_q0) <= max_epsilon:
            d_perf = float(abs(q_p - target_q0))

    return {
        "true_delta": d_total,
        "true_delta_0": d0,
        "true_delta_1": d1,
        "worst_true_delta_deploy": worst_delta_deploy,
        "true_margin_reversal_distance": d_margin,
        "true_performance_reversal_distance": d_perf,
        "target_q0": target_q0,
        "deploy_epsilon": deploy_epsilon,
        "n_contexts": n_contexts,
    }


def _run_scenario_benchmark(
    scenario: ReversalScenarioConfig,
    config: IndependentReliabilityConfig,
    run_seed: int,
) -> dict[str, Any]:
    """Run full Monte Carlo reliability benchmark for a single preregistered scenario."""
    env_seed = int(_stable_key(f"env_{scenario.name}_{run_seed}") % (2**31))
    env = build_environment(scenario.environment, rng=_rng(env_seed))

    logging_policy = build_policy(scenario.logging_policy, env)
    baseline_policy = build_policy(scenario.baseline_policy, env)
    candidate_policy = build_policy(scenario.candidate_policy, env)

    partition = StratumPartition(
        split_dim=scenario.split_dim,
        split_threshold=scenario.split_threshold,
    )

    # Ground Truth Reference
    gt = compute_ground_truth_reference(
        env=env,
        cand_policy=candidate_policy,
        base_policy=baseline_policy,
        partition=partition,
        n_contexts=config.reference_n_contexts,
        margin=scenario.margin,
        deploy_epsilon=config.deploy_threshold_epsilon,
        max_epsilon=scenario.max_epsilon,
        target_q0=0.50,
        seed=env_seed + 1,
    )

    true_delta = gt["true_delta"]
    worst_true_delta = gt["worst_true_delta_deploy"]
    true_d0 = gt["true_delta_0"]
    true_d1 = gt["true_delta_1"]
    true_d_margin = gt["true_margin_reversal_distance"]
    true_d_perf = gt["true_performance_reversal_distance"]

    margin = scenario.margin
    eps_deploy = config.deploy_threshold_epsilon
    m = config.n_replications

    # Decision outcome collectors
    decisions_a: list[str] = []
    decisions_b: list[str] = []
    decisions_c: list[str] = []
    decisions_d: list[str] = []
    decisions_e: list[str] = []
    decisions_f: list[str] = []

    # Agreement tracker for Method D vs Method F
    df_agreements = 0

    # Coverage trackers (Part 5)
    pointwise_covers_q0 = 0
    simultaneous_covers_interval = 0
    simultaneous_covers_simplex = 0
    pointwise_grid_joint_covers = 0

    # Reversal search trackers (Part 6)
    reversal_detected_count = 0
    reversal_distances_num: list[float] = []
    reversal_distances_ana: list[float] = []
    analytical_discrepancies: list[float] = []
    margin_discrepancies: list[float] = []
    perf_discrepancies: list[float] = []

    diagnostic_gate_failed_count = 0
    z_pw = _normal_quantile(config.confidence_level)

    q_min_deploy = max(0.0, 0.50 - eps_deploy)
    q_max_deploy = min(1.0, 0.50 + eps_deploy)
    grid_points = np.linspace(q_min_deploy, q_max_deploy, 11)

    for rep in range(m):
        data_seed = int(_stable_key(f"data_{scenario.name}_{run_seed}_{rep}") % (2**31))
        data = env.generate_logged_data(
            scenario.n_samples,
            logging_policy,
            _rng(data_seed),
            include_logging_action_dist=True,
        )

        cand_dist = candidate_policy.action_distribution(data.context)
        base_dist = baseline_policy.action_distribution(data.context)

        # Diagnostic gate check
        gate_thresholds = DiagnosticGateThresholds(
            min_propensity=scenario.min_propensity_threshold,
            min_ess_fraction=0.05,
            max_weight_share=0.20,
        )
        gate_report = evaluate_diagnostic_gate(
            data, cand_dist, base_dist, thresholds=gate_thresholds
        )
        diag_passed = gate_report.passed
        diag_reason = gate_report.rejection_reasons[0] if not diag_passed else None
        if not diag_passed:
            diagnostic_gate_failed_count += 1

        # Model cross-fitting
        model_seed = int(_stable_key(f"model_{scenario.name}_{run_seed}_{rep}") % (2**31))

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

        q_hat = cross_fit_reward_model(data, model_factory, n_folds=config.n_folds, seed=model_seed)

        paired = evaluate_paired_difference(
            "dr",
            data,
            cand_dist,
            base_dist,
            predicted_rewards=q_hat,
            confidence_level=config.confidence_level,
        )

        # -------------------------------------------------------------
        # Method A: Point-estimate only
        # -------------------------------------------------------------
        if paired.point_estimate > margin:
            dec_a = DecisionOutcome.RECOMMEND.value
        elif paired.point_estimate < margin:
            dec_a = DecisionOutcome.HOLD.value
        else:
            dec_a = DecisionOutcome.INCONCLUSIVE.value

        # -------------------------------------------------------------
        # Method B: Baseline confidence bound
        # -------------------------------------------------------------
        res_b = evaluate_decision_from_paired_result(
            paired,
            margin=margin,
            min_propensity=gate_report.min_propensity,
            min_propensity_threshold=scenario.min_propensity_threshold,
        )
        dec_b = res_b.outcome.value

        # Strata Model
        strata_model = create_two_stratum_model(
            data.context,
            paired.diff_scores,
            partition=partition,
            target_mixture=0.50,
        )

        # -------------------------------------------------------------
        # Method C: Worst-stratum screening
        # -------------------------------------------------------------
        l0 = strata_model.stratum_0.mean_diff - z_pw * strata_model.stratum_0.standard_error
        l1 = strata_model.stratum_1.mean_diff - z_pw * strata_model.stratum_1.standard_error
        worst_l = min(l0, l1)
        if dec_b == DecisionOutcome.RECOMMEND.value and worst_l > 0.0:
            dec_c = DecisionOutcome.RECOMMEND.value
        elif dec_b == DecisionOutcome.HOLD.value or (paired.ci_upper < margin):
            dec_c = DecisionOutcome.HOLD.value
        else:
            dec_c = DecisionOutcome.INCONCLUSIVE.value

        # -------------------------------------------------------------
        # Method D: Pointwise decision-reversal search
        # Evaluates baseline from target mixture q0 to ensure duality
        # consistency with forward robust mixture evaluation (Method F).
        # -------------------------------------------------------------
        res_d_base = strata_model.evaluate_mixture_decision(
            strata_model.baseline_q0,
            margin=margin,
            confidence_level=config.confidence_level,
            diagnostic_passed=diag_passed,
            diagnostic_reason=diag_reason,
        )
        dec_d_base = res_d_base.outcome.value

        search_res = find_decision_reversal(
            strata_model,
            margin=margin,
            max_epsilon=scenario.max_epsilon,
            confidence_level=config.confidence_level,
            diagnostic_passed=diag_passed,
            diagnostic_reason=diag_reason,
        )
        ana_res = find_analytical_decision_reversal(
            strata_model,
            margin=margin,
            max_epsilon=scenario.max_epsilon,
            confidence_level=config.confidence_level,
            diagnostic_passed=diag_passed,
            diagnostic_reason=diag_reason,
        )

        if search_res.reversal_distance is not None:
            reversal_detected_count += 1
            d_num = search_res.reversal_distance
            reversal_distances_num.append(d_num)
            if ana_res.exact_reversal_distance is not None:
                d_ana = ana_res.exact_reversal_distance
                reversal_distances_ana.append(d_ana)
                analytical_discrepancies.append(abs(d_num - d_ana))
            if true_d_margin is not None:
                margin_discrepancies.append(abs(d_num - true_d_margin))
            if true_d_perf is not None:
                perf_discrepancies.append(abs(d_num - true_d_perf))

        if dec_d_base == DecisionOutcome.RECOMMEND.value:
            if search_res.status == "NO_REVERSAL_FOUND_WITHIN_BOUNDS" or (
                search_res.reversal_distance is not None
                and search_res.reversal_distance > eps_deploy
            ):
                dec_d = DecisionOutcome.RECOMMEND.value
            else:
                dec_d = DecisionOutcome.INCONCLUSIVE.value
        elif dec_d_base == DecisionOutcome.HOLD.value:
            dec_d = DecisionOutcome.HOLD.value
        else:
            dec_d = DecisionOutcome.INCONCLUSIVE.value

        # -------------------------------------------------------------
        # Method E: Simultaneous Band Reversal
        # -------------------------------------------------------------
        band = TwoStratumSimultaneousBand(strata_model, confidence_level=config.confidence_level)
        res_min_eps = band.evaluate_band(q_min_deploy, margin)
        res_max_eps = band.evaluate_band(q_max_deploy, margin)
        min_simult_l = min(res_min_eps.simultaneous_ci_lower, res_max_eps.simultaneous_ci_lower)

        if diag_passed and min_simult_l > margin:
            dec_e = DecisionOutcome.RECOMMEND.value
        elif (
            diag_passed
            and max(res_min_eps.simultaneous_ci_upper, res_max_eps.simultaneous_ci_upper) < margin
        ):
            dec_e = DecisionOutcome.HOLD.value
        else:
            dec_e = DecisionOutcome.INCONCLUSIVE.value

        # -------------------------------------------------------------
        # Method F: Forward robust-value baseline
        # -------------------------------------------------------------
        res_f = evaluate_robust_mixture_baseline(
            strata_model,
            margin=margin,
            deploy_epsilon=eps_deploy,
            confidence_level=config.confidence_level,
            diagnostic_passed=diag_passed,
            diagnostic_reason=diag_reason,
        )
        dec_f = res_f.outcome.value

        if dec_d == dec_f:
            df_agreements += 1

        decisions_a.append(dec_a)
        decisions_b.append(dec_b)
        decisions_c.append(dec_c)
        decisions_d.append(dec_d)
        decisions_e.append(dec_e)
        decisions_f.append(dec_f)

        # -------------------------------------------------------------
        # Simultaneous Coverage Validation (Part 5)
        # -------------------------------------------------------------
        # Pointwise coverage at q0 = 0.50
        if paired.ci_lower <= true_delta <= paired.ci_upper:
            pointwise_covers_q0 += 1

        # Simultaneous coverage over Q_epsilon = [q_min_deploy, q_max_deploy]
        # Linear in q: holds over entire interval iff it holds at both endpoints
        true_d_min = q_min_deploy * true_d0 + (1.0 - q_min_deploy) * true_d1
        true_d_max = q_max_deploy * true_d0 + (1.0 - q_max_deploy) * true_d1
        cov_min = (
            res_min_eps.simultaneous_ci_lower <= true_d_min <= res_min_eps.simultaneous_ci_upper
        )
        cov_max = (
            res_max_eps.simultaneous_ci_lower <= true_d_max <= res_max_eps.simultaneous_ci_upper
        )
        if cov_min and cov_max:
            simultaneous_covers_interval += 1

        # Simultaneous coverage over full simplex [0, 1]
        cov_s0 = band.stratum_0_lower <= true_d0 <= band.stratum_0_upper
        cov_s1 = band.stratum_1_lower <= true_d1 <= band.stratum_1_upper
        if cov_s0 and cov_s1:
            simultaneous_covers_simplex += 1

        # Naive family-wise test: check if pointwise CIs cover true delta at ALL grid points
        joint_grid_covered = True
        for q_g in grid_points:
            _, _, ci_l_g, ci_u_g = strata_model.evaluate_mixture(
                float(q_g), confidence_level=config.confidence_level
            )
            true_d_g = q_g * true_d0 + (1.0 - q_g) * true_d1
            if not (ci_l_g <= true_d_g <= ci_u_g):
                joint_grid_covered = False
                break
        if joint_grid_covered:
            pointwise_grid_joint_covers += 1

    # Summarize metrics across replications
    def _summarize_decisions(decs: list[str]) -> dict[str, Any]:
        counts = Counter(decs)
        rec_count = counts[DecisionOutcome.RECOMMEND.value]
        hold_count = counts[DecisionOutcome.HOLD.value]
        inc_count = counts[DecisionOutcome.INCONCLUSIVE.value]

        rec_rate = rec_count / m
        hold_rate = hold_count / m
        inc_rate = inc_count / m

        # 1. Baseline False Recommendation: RECOMMEND when true_delta < margin
        false_rec_count = rec_count if true_delta < margin else 0
        false_rec_rate = false_rec_count / m

        # 2. Harmful Deployment: RECOMMEND when true_delta < 0 or worst_true_delta < 0
        harmful_count = rec_count if (true_delta < 0.0 or worst_true_delta < 0.0) else 0
        harmful_rate = harmful_count / m

        # 3. Fragile Deployment: RECOMMEND when worst_true_delta < margin
        fragile_count = rec_count if worst_true_delta < margin else 0
        fragile_rate = fragile_count / m

        # 4. Missed Beneficial Deployment: NOT RECOMMEND when candidate is robustly beneficial
        missed_count = (m - rec_count) if worst_true_delta >= margin else 0
        missed_rate = missed_count / m

        return {
            "recommend_rate": rec_rate,
            "recommend_mcse": _mcse_rate(rec_rate, m),
            "hold_rate": hold_rate,
            "hold_mcse": _mcse_rate(hold_rate, m),
            "inconclusive_rate": inc_rate,
            "inconclusive_mcse": _mcse_rate(inc_rate, m),
            "baseline_false_recommendation_rate": false_rec_rate,
            "baseline_false_recommendation_mcse": _mcse_rate(false_rec_rate, m),
            "harmful_deployment_rate": harmful_rate,
            "harmful_deployment_mcse": _mcse_rate(harmful_rate, m),
            "fragile_deployment_rate": fragile_rate,
            "fragile_deployment_mcse": _mcse_rate(fragile_rate, m),
            "missed_beneficial_rate": missed_rate,
            "missed_beneficial_mcse": _mcse_rate(missed_rate, m),
        }

    summary_methods = {
        "method_a_point_estimate": _summarize_decisions(decisions_a),
        "method_b_baseline_ci": _summarize_decisions(decisions_b),
        "method_c_worst_stratum": _summarize_decisions(decisions_c),
        "method_d_pointwise_reversal": _summarize_decisions(decisions_d),
        "method_e_simultaneous_band": _summarize_decisions(decisions_e),
        "method_f_robust_mixture": _summarize_decisions(decisions_f),
    }

    mean_d_num = float(np.mean(reversal_distances_num)) if reversal_distances_num else None
    mean_d_ana = float(np.mean(reversal_distances_ana)) if reversal_distances_ana else None
    max_ana_disc = float(np.max(analytical_discrepancies)) if analytical_discrepancies else 0.0
    mean_margin_disc = float(np.mean(margin_discrepancies)) if margin_discrepancies else None
    mean_perf_disc = float(np.mean(perf_discrepancies)) if perf_discrepancies else None

    # Simultaneous Coverage Evaluation
    pw_cov_rate = pointwise_covers_q0 / m
    simult_cov_rate = simultaneous_covers_interval / m
    simult_simplex_cov_rate = simultaneous_covers_simplex / m
    joint_grid_cov_rate = pointwise_grid_joint_covers / m

    simult_ci_low, simult_ci_high = _wilson_ci(
        simultaneous_covers_interval, m, config.confidence_level
    )

    # Classification taxonomy
    is_rec = summary_methods["method_b_baseline_ci"]["recommend_rate"] > 0.5
    initial_dec = DecisionOutcome.RECOMMEND if is_rec else DecisionOutcome.INCONCLUSIVE
    taxonomy = classify_reversal_taxonomy(
        initial_decision=initial_dec,
        reversed_decision=DecisionOutcome.INCONCLUSIVE if reversal_detected_count > 0 else None,
        evidence_distance=mean_d_num,
        true_delta_0=true_d0,
        true_delta_1=true_d1,
        margin=margin,
        baseline_q0=0.50,
        max_epsilon=scenario.max_epsilon,
    )

    return {
        "scenario": scenario.name,
        "n_samples": scenario.n_samples,
        "margin": scenario.margin,
        "replications": m,
        "ground_truth": gt,
        "taxonomy": {
            "transition_category": taxonomy.transition_category,
            "has_evidence_reversal": taxonomy.has_evidence_reversal,
            "has_margin_reversal": taxonomy.has_margin_reversal,
            "has_performance_reversal": taxonomy.has_performance_reversal,
            "explanation": taxonomy.explanation,
        },
        "methods": summary_methods,
        "method_d_f_duality_validation": {
            "agreement_rate": df_agreements / m,
            "exact_decision_equivalence": df_agreements == m,
        },
        "simultaneous_inference_validation": {
            "pointwise_coverage_at_q0": pw_cov_rate,
            "pointwise_coverage_mcse": _mcse_rate(pw_cov_rate, m),
            "simultaneous_interval_coverage": simult_cov_rate,
            "simultaneous_interval_coverage_mcse": _mcse_rate(simult_cov_rate, m),
            "simultaneous_interval_wilson_ci_95": [simult_ci_low, simult_ci_high],
            "simultaneous_full_simplex_coverage": simult_simplex_cov_rate,
            "pointwise_grid_joint_coverage": joint_grid_cov_rate,
            "familywise_error_inflation": float(pw_cov_rate - joint_grid_cov_rate),
        },
        "reversal_distance_evaluation": {
            "reversal_detected_rate": reversal_detected_count / m,
            "mean_reversal_distance_numerical": mean_d_num,
            "mean_reversal_distance_analytical": mean_d_ana,
            "max_analytical_discrepancy": max_ana_disc,
            "discrepancy_within_tolerance": max_ana_disc <= 1e-4,
            "mean_margin_crossing_discrepancy": mean_margin_disc,
            "mean_perf_crossing_discrepancy": mean_perf_disc,
        },
        "diagnostics": {
            "diagnostic_gate_failed_rate": diagnostic_gate_failed_count / m,
        },
    }


def run_independent_reliability(config: IndependentReliabilityConfig) -> ExperimentOutput:
    """Run Experiment 007 across all preregistered scenarios."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    scenarios_out: list[dict[str, Any]] = []

    for sc in config.scenarios:
        logger.info(
            "Executing independent reliability scenario: %s (N=%d, M=%d)",
            sc.name,
            sc.n_samples,
            config.n_replications,
        )
        out = _run_scenario_benchmark(sc, config, run_seed=config.seed)
        scenarios_out.append(out)

    runtime_s = time.perf_counter() - t0
    results: dict[str, Any] = {
        "experiment": config.name,
        "seed": config.seed,
        "confidence_level": config.confidence_level,
        "n_replications": config.n_replications,
        "n_folds": config.n_folds,
        "deploy_threshold_epsilon": config.deploy_threshold_epsilon,
        "scenarios": scenarios_out,
    }

    return ExperimentOutput(
        results=results,
        warnings=warnings,
        runtime={"wall_clock_seconds": round(runtime_s, 3)},
    )
