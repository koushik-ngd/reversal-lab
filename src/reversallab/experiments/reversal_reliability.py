r"""Experiment 006: Decision-Reversal Reliability Benchmark.

Rigorous Monte Carlo validation across multiple independent random seeds comparing:
Method A: Point-estimate only (naive rule: \hat{\Delta} > \delta)
Method B: Baseline confidence-bound (L_\Delta > \delta)
Method C: Worst-stratum check (L_\Delta > \delta and min(L0, L1) > 0)
Method D: Pointwise decision-reversal search (L_\Delta > \delta and d_rev* >= \epsilon_deploy)
Method E: Simultaneous-band decision-reversal search
(L_simult(q) > \delta over \mathcal{Q}_\epsilon)

Evaluates:
- False confident recommendations (recommending when \Delta^* < \delta or \Delta^* < 0)
- Unnecessary withholding of truly favorable policies (withholding when \Delta^* > \delta)
- Agreement between numerical bisection and exact closed-form analytical reference
- Robustness across sample sizes and diagnostic failure rates
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
    ReversalReliabilityConfig,
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


def compute_ground_truth_reference(
    env: SyntheticBanditEnvironment,
    cand_policy: Policy,
    base_policy: Policy,
    partition: StratumPartition,
    *,
    n_contexts: int = 50_000,
    margin: float = 0.0,
    max_epsilon: float = 0.40,
    target_q0: float = 0.50,
    seed: int = 999,
) -> dict[str, Any]:
    """Compute high-precision reference ground truth and theoretical reversal distances."""
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
        "true_margin_reversal_distance": d_margin,
        "true_performance_reversal_distance": d_perf,
        "target_q0": target_q0,
        "n_contexts": n_contexts,
    }


def _run_reliability_scenario(
    scenario: ReversalScenarioConfig,
    config: ReversalReliabilityConfig,
    run_seed: int,
) -> dict[str, Any]:
    """Run multi-replication Monte Carlo reliability benchmark for a single scenario."""
    env_seed = int(_stable_key(f"env_{scenario.name}_{run_seed}") % (2**31))
    env = build_environment(scenario.environment, rng=_rng(env_seed))

    logging_policy = build_policy(scenario.logging_policy, env)
    baseline_policy = build_policy(scenario.baseline_policy, env)
    candidate_policy = build_policy(scenario.candidate_policy, env)

    partition = StratumPartition(
        split_dim=scenario.split_dim,
        split_threshold=scenario.split_threshold,
    )

    # 1. Ground Truth
    gt = compute_ground_truth_reference(
        env=env,
        cand_policy=candidate_policy,
        base_policy=baseline_policy,
        partition=partition,
        n_contexts=config.reference_n_contexts,
        margin=scenario.margin,
        max_epsilon=scenario.max_epsilon,
        target_q0=0.50,
        seed=env_seed + 1,
    )

    true_delta = gt["true_delta"]
    margin = scenario.margin
    eps_deploy = config.deploy_threshold_epsilon
    m = config.n_replications

    # Collectors for the 5 methods
    # Decisions: "RECOMMEND", "HOLD", "INCONCLUSIVE"
    decisions_a: list[str] = []
    decisions_b: list[str] = []
    decisions_c: list[str] = []
    decisions_d: list[str] = []
    decisions_e: list[str] = []

    reversal_distances_num: list[float] = []
    reversal_distances_ana: list[float] = []
    analytical_discrepancies: list[float] = []
    reversal_detected_count = 0
    diagnostic_gate_failed_count = 0

    point_estimates: list[float] = []
    standard_errors: list[float] = []
    ci_lower_bounds: list[float] = []
    ci_upper_bounds: list[float] = []

    z_pw = _normal_quantile(config.confidence_level)

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
        if not gate_report.passed:
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

        point_estimates.append(paired.point_estimate)
        standard_errors.append(paired.standard_error)
        ci_lower_bounds.append(paired.ci_lower)
        ci_upper_bounds.append(paired.ci_upper)

        # Baseline decision (Method B)
        diag_reason = gate_report.rejection_reasons[0] if not gate_report.passed else None
        res_b = evaluate_decision_from_paired_result(
            paired,
            margin=margin,
            min_propensity=gate_report.min_propensity,
            min_propensity_threshold=scenario.min_propensity_threshold,
        )
        dec_b = res_b.outcome.value

        # Method A: Point-estimate only
        if paired.point_estimate > margin:
            dec_a = DecisionOutcome.RECOMMEND.value
        elif paired.point_estimate < margin:
            dec_a = DecisionOutcome.HOLD.value
        else:
            dec_a = DecisionOutcome.INCONCLUSIVE.value

        # Strata Model with known target mixture 0.50
        strata_model = create_two_stratum_model(
            data.context,
            paired.diff_scores,
            partition=partition,
            target_mixture=0.50,
        )

        # Method C: Baseline confidence bound + Worst-Stratum check
        # Stratum lower bounds:
        l0 = strata_model.stratum_0.mean_diff - z_pw * strata_model.stratum_0.standard_error
        l1 = strata_model.stratum_1.mean_diff - z_pw * strata_model.stratum_1.standard_error
        worst_l = min(l0, l1)
        if dec_b == DecisionOutcome.RECOMMEND.value and worst_l > 0.0:
            dec_c = DecisionOutcome.RECOMMEND.value
        elif dec_b == DecisionOutcome.HOLD.value or (paired.ci_upper < margin):
            dec_c = DecisionOutcome.HOLD.value
        else:
            dec_c = DecisionOutcome.INCONCLUSIVE.value

        # Method D: Pointwise decision-reversal search
        search_res = find_decision_reversal(
            strata_model,
            margin=margin,
            max_epsilon=scenario.max_epsilon,
            confidence_level=config.confidence_level,
            diagnostic_passed=gate_report.passed,
            diagnostic_reason=diag_reason,
        )
        ana_res = find_analytical_decision_reversal(
            strata_model,
            margin=margin,
            max_epsilon=scenario.max_epsilon,
            confidence_level=config.confidence_level,
            diagnostic_passed=gate_report.passed,
            diagnostic_reason=diag_reason,
        )

        if search_res.reversal_distance is not None:
            reversal_detected_count += 1
            reversal_distances_num.append(search_res.reversal_distance)
            if ana_res.exact_reversal_distance is not None:
                reversal_distances_ana.append(ana_res.exact_reversal_distance)
                analytical_discrepancies.append(
                    abs(search_res.reversal_distance - ana_res.exact_reversal_distance)
                )

        # Robust deployment decision under Method D:
        # Require baseline RECOMMEND AND stability margin >= eps_deploy
        if dec_b == DecisionOutcome.RECOMMEND.value:
            if search_res.status == "NO_REVERSAL_FOUND_WITHIN_BOUNDS" or (
                search_res.reversal_distance is not None
                and search_res.reversal_distance >= eps_deploy
            ):
                dec_d = DecisionOutcome.RECOMMEND.value
            else:
                dec_d = DecisionOutcome.INCONCLUSIVE.value  # fragile!
        elif dec_b == DecisionOutcome.HOLD.value:
            dec_d = DecisionOutcome.HOLD.value
        else:
            dec_d = DecisionOutcome.INCONCLUSIVE.value

        # Method E: Simultaneous Band Reversal
        band = TwoStratumSimultaneousBand(strata_model, confidence_level=config.confidence_level)
        _simult_res = band.find_simultaneous_reversal(
            margin=margin,
            max_epsilon=scenario.max_epsilon,
            diagnostic_passed=gate_report.passed,
            diagnostic_reason=diag_reason,
        )
        # Deploy under Method E if simultaneous lower bound over Q_eps strictly exceeds margin
        res_min_eps = band.evaluate_band(max(0.0, 0.50 - eps_deploy), margin)
        res_max_eps = band.evaluate_band(min(1.0, 0.50 + eps_deploy), margin)
        min_simult_l = min(res_min_eps.simultaneous_ci_lower, res_max_eps.simultaneous_ci_lower)

        if gate_report.passed and min_simult_l > margin:
            dec_e = DecisionOutcome.RECOMMEND.value
        elif (
            gate_report.passed
            and max(res_min_eps.simultaneous_ci_upper, res_max_eps.simultaneous_ci_upper) < margin
        ):
            dec_e = DecisionOutcome.HOLD.value
        else:
            dec_e = DecisionOutcome.INCONCLUSIVE.value

        decisions_a.append(dec_a)
        decisions_b.append(dec_b)
        decisions_c.append(dec_c)
        decisions_d.append(dec_d)
        decisions_e.append(dec_e)

    # Summarize metrics across replications
    def _summarize_decisions(decs: list[str]) -> dict[str, Any]:
        counts = Counter(decs)
        rec_rate = counts[DecisionOutcome.RECOMMEND.value] / m
        hold_rate = counts[DecisionOutcome.HOLD.value] / m
        inc_rate = counts[DecisionOutcome.INCONCLUSIVE.value] / m

        # False confident recommendation: RECOMMEND when true_delta < margin
        false_rec_rate = rec_rate if true_delta < margin else 0.0

        # Unnecessary withholding: NOT RECOMMEND when true_delta > margin
        withholding_rate = (1.0 - rec_rate) if true_delta > margin else 0.0

        return {
            "recommend_rate": rec_rate,
            "recommend_mcse": _mcse_rate(rec_rate, m),
            "hold_rate": hold_rate,
            "hold_mcse": _mcse_rate(hold_rate, m),
            "inconclusive_rate": inc_rate,
            "inconclusive_mcse": _mcse_rate(inc_rate, m),
            "false_recommendation_rate": false_rec_rate,
            "unnecessary_withholding_rate": withholding_rate,
        }

    summary_methods = {
        "method_a_point_estimate": _summarize_decisions(decisions_a),
        "method_b_baseline_ci": _summarize_decisions(decisions_b),
        "method_c_worst_stratum": _summarize_decisions(decisions_c),
        "method_d_pointwise_reversal": _summarize_decisions(decisions_d),
        "method_e_simultaneous_band": _summarize_decisions(decisions_e),
    }

    mean_d_num = float(np.mean(reversal_distances_num)) if reversal_distances_num else None
    mean_d_ana = float(np.mean(reversal_distances_ana)) if reversal_distances_ana else None
    max_discrepancy = float(np.max(analytical_discrepancies)) if analytical_discrepancies else 0.0

    # Taxonomy report based on ground truth and empirical averages
    is_rec = summary_methods["method_b_baseline_ci"]["recommend_rate"] > 0.5
    initial_dec = DecisionOutcome.RECOMMEND if is_rec else DecisionOutcome.INCONCLUSIVE
    taxonomy = classify_reversal_taxonomy(
        initial_decision=initial_dec,
        reversed_decision=DecisionOutcome.INCONCLUSIVE if reversal_detected_count > 0 else None,
        evidence_distance=mean_d_num,
        true_delta_0=gt["true_delta_0"],
        true_delta_1=gt["true_delta_1"],
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
        "reversal_search_validation": {
            "reversal_detected_rate": reversal_detected_count / m,
            "mean_reversal_distance_numerical": mean_d_num,
            "mean_reversal_distance_analytical": mean_d_ana,
            "max_analytical_discrepancy": max_discrepancy,
            "discrepancy_within_tolerance": max_discrepancy <= 1e-4,
        },
        "diagnostics": {
            "diagnostic_gate_failed_rate": diagnostic_gate_failed_count / m,
        },
    }


def run_reversal_reliability(config: ReversalReliabilityConfig) -> ExperimentOutput:
    """Run Experiment 006 across all configured scenarios."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    scenarios_out: list[dict[str, Any]] = []

    for sc in config.scenarios:
        logger.info(
            "Executing reliability scenario: %s (N=%d, M=%d)",
            sc.name,
            sc.n_samples,
            config.n_replications,
        )
        out = _run_reliability_scenario(sc, config, run_seed=config.seed)
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
