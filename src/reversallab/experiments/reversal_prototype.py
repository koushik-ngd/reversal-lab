"""Experiment 005: Decision-Reversal Prototype.

Benchmarks evidence-based deployment decisions and 1-D context mixture perturbation
search across four controlled synthetic scenarios:
1. Robust Superiority: Candidate beats baseline in both strata; no reversal within bounds.
2. Fragile Superiority: Candidate beats baseline in aggregate, but is inferior in stratum 1;
   modest shift reverses RECOMMEND to INCONCLUSIVE / HOLD.
3. High Uncertainty: Candidate has positive margin but small N leaves CI spanning margin;
   baseline is INCONCLUSIVE while naive point-estimate falsely recommends deployment.
4. Overlap Violation: Severe propensity deficiency trips diagnostic gate; INCONCLUSIVE.
"""

from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.evaluation.cross_fitting import cross_fit_reward_model
from reversallab.evaluation.reward_models import (
    MisspecifiedLinearRewardModel,
    RewardModel,
    StratifiedLinearRewardModel,
)
from reversallab.experiments.config import (
    ReversalPrototypeConfig,
    ReversalScenarioConfig,
    build_environment,
    build_policy,
)
from reversallab.policies.base import Policy
from reversallab.reversal.decision import evaluate_decision_from_paired_result
from reversallab.reversal.paired import evaluate_paired_difference
from reversallab.reversal.search import find_decision_reversal
from reversallab.reversal.strata import (
    StratumPartition,
    create_two_stratum_model,
)

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


def compute_ground_truth_differences(
    env: SyntheticBanditEnvironment,
    cand_policy: Policy,
    base_policy: Policy,
    partition: StratumPartition,
    *,
    n_contexts: int = 200_000,
    margin: float = 0.0,
    max_epsilon: float = 0.40,
    seed: int = 999,
) -> dict[str, Any]:
    """Compute Monte Carlo ground-truth policy differences and reversal distance."""
    rng = np.random.default_rng(seed)
    contexts = env.sample_contexts(n_contexts, rng)
    assignments = partition.assign(contexts)

    idx_0 = np.where(assignments == 0)[0]
    idx_1 = np.where(assignments == 1)[0]
    q0_true = len(idx_0) / n_contexts

    cand_dist = cand_policy.action_distribution(contexts)
    base_dist = base_policy.action_distribution(contexts)
    q_matrix = env.expected_reward(contexts)

    v_cand_per_obs = np.sum(cand_dist * q_matrix, axis=1)
    v_base_per_obs = np.sum(base_dist * q_matrix, axis=1)
    diff_per_obs = v_cand_per_obs - v_base_per_obs

    diff_0 = float(np.mean(diff_per_obs[idx_0]))
    diff_1 = float(np.mean(diff_per_obs[idx_1]))
    diff_total = float(np.mean(diff_per_obs))

    # Analytical population reversal distance for:
    # Delta*(q0) = q0 * diff_0 + (1 - q0) * diff_1 = margin
    true_rev_dist: float | None = None
    true_rev_q0: float | None = None
    if abs(diff_0 - diff_1) > 1e-9:
        q0_star = (margin - diff_1) / (diff_0 - diff_1)
        if 0.0 <= q0_star <= 1.0:
            dist = abs(q0_star - q0_true)
            if dist <= max_epsilon:
                true_rev_dist = float(dist)
                true_rev_q0 = float(q0_star)

    return {
        "true_delta": diff_total,
        "true_delta_0": diff_0,
        "true_delta_1": diff_1,
        "true_q0": q0_true,
        "true_reversal_distance": true_rev_dist,
        "true_reversal_q0": true_rev_q0,
        "n_reference_contexts": n_contexts,
    }


def _run_single_scenario(
    scenario: ReversalScenarioConfig,
    config: ReversalPrototypeConfig,
    run_seed: int,
) -> dict[str, Any]:
    """Run decision-reversal evaluation for a single synthetic scenario."""
    env_seed = int(_stable_key(f"env_{scenario.name}_{run_seed}") % (2**31))
    env = build_environment(scenario.environment, rng=_rng(env_seed))

    logging_policy = build_policy(scenario.logging_policy, env)
    baseline_policy = build_policy(scenario.baseline_policy, env)
    candidate_policy = build_policy(scenario.candidate_policy, env)

    partition = StratumPartition(
        split_dim=scenario.split_dim,
        split_threshold=scenario.split_threshold,
    )

    # 1. Ground truth
    gt = compute_ground_truth_differences(
        env=env,
        cand_policy=candidate_policy,
        base_policy=baseline_policy,
        partition=partition,
        n_contexts=config.reference_n_contexts,
        margin=scenario.margin,
        max_epsilon=scenario.max_epsilon,
        seed=env_seed + 1,
    )

    # 2. Generate logged dataset
    data_seed = int(_stable_key(f"data_{scenario.name}_{run_seed}") % (2**31))
    data = env.generate_logged_data(
        scenario.n_samples,
        logging_policy,
        _rng(data_seed),
        include_logging_action_dist=True,
    )

    min_propensity = float(np.min(data.logging_propensity))

    # 3. Cross-fit reward model
    model_seed = int(_stable_key(f"model_{scenario.name}_{run_seed}") % (2**31))

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

    # 4. Action distributions
    cand_dist = candidate_policy.action_distribution(data.context)
    base_dist = baseline_policy.action_distribution(data.context)

    # 5. Paired difference evaluation
    paired = evaluate_paired_difference(
        "dr",
        data,
        cand_dist,
        base_dist,
        predicted_rewards=q_hat,
        confidence_level=config.confidence_level,
    )

    # 6. Baseline decision
    decision = evaluate_decision_from_paired_result(
        paired,
        margin=scenario.margin,
        min_propensity=min_propensity,
        min_propensity_threshold=scenario.min_propensity_threshold,
    )

    # 7. Stratum mixture and reversal search
    strata_model = create_two_stratum_model(data.context, paired.diff_scores, partition=partition)
    search_res = find_decision_reversal(
        strata_model,
        margin=scenario.margin,
        max_epsilon=scenario.max_epsilon,
        confidence_level=config.confidence_level,
        diagnostic_passed=decision.diagnostic_passed,
        diagnostic_reason=decision.reason,
    )

    return {
        "scenario": scenario.name,
        "n_samples": scenario.n_samples,
        "margin": scenario.margin,
        "max_epsilon": scenario.max_epsilon,
        "min_propensity": min_propensity,
        "ground_truth": gt,
        "paired_estimation": {
            "point_estimate": paired.point_estimate,
            "standard_error": paired.standard_error,
            "ci_lower": paired.ci_lower,
            "ci_upper": paired.ci_upper,
            "independent_standard_error": paired.independent_standard_error,
            "variance_reduction_pct": paired.variance_reduction_pct,
            "covariance": paired.covariance,
        },
        "stratum_estimates": {
            "stratum_0": {
                "n_samples": strata_model.stratum_0.n_samples,
                "weight": strata_model.stratum_0.empirical_weight,
                "mean_diff": strata_model.stratum_0.mean_diff,
                "standard_error": strata_model.stratum_0.standard_error,
            },
            "stratum_1": {
                "n_samples": strata_model.stratum_1.n_samples,
                "weight": strata_model.stratum_1.empirical_weight,
                "mean_diff": strata_model.stratum_1.mean_diff,
                "standard_error": strata_model.stratum_1.standard_error,
            },
        },
        "decisions": {
            "naive_point_estimate": decision.naive_outcome.value,
            "baseline_confidence_bound": decision.outcome.value,
            "reason": decision.reason,
            "diagnostic_passed": decision.diagnostic_passed,
        },
        "reversal_search": {
            "status": search_res.status,
            "initial_decision": search_res.initial_decision.value,
            "reversed_decision": (
                search_res.reversed_decision.value if search_res.reversed_decision else None
            ),
            "reversal_distance": search_res.reversal_distance,
            "reversed_q0": search_res.reversed_q0,
            "transition": search_res.transition,
            "fine_grid_verified": search_res.fine_grid_verified,
            "reversed_ci_lower": search_res.reversed_ci_lower,
            "reversed_ci_upper": search_res.reversed_ci_upper,
        },
    }


def run_reversal_prototype(config: ReversalPrototypeConfig) -> ExperimentOutput:
    """Run Experiment 005 across configured synthetic scenarios."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    scenarios_out: list[dict[str, Any]] = []

    for sc in config.scenarios:
        logger.info("Executing decision-reversal scenario: %s (N=%d)", sc.name, sc.n_samples)
        out = _run_single_scenario(sc, config, run_seed=config.seed)
        scenarios_out.append(out)

    runtime_s = time.perf_counter() - t0
    results: dict[str, Any] = {
        "experiment": config.name,
        "seed": config.seed,
        "confidence_level": config.confidence_level,
        "n_folds": config.n_folds,
        "scenarios": scenarios_out,
    }

    return ExperimentOutput(
        results=results,
        warnings=warnings,
        runtime={"wall_clock_seconds": round(runtime_s, 3)},
    )
