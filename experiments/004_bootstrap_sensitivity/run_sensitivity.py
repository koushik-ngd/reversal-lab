"""Targeted bootstrap sensitivity benchmark comparing B in {50, 200, 1000}."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from reversallab.data.synthetic import RewardType, SyntheticBanditEnvironment
from reversallab.evaluation.bootstrap import bootstrap_evaluate
from reversallab.evaluation.influence_scores import evaluate_influence_inference
from reversallab.evaluation.reward_models import StratifiedLinearRewardModel
from reversallab.policies.linear import LinearSoftmaxPolicy


def run_benchmark() -> dict[str, Any]:
    seed = 20261104
    rng_master = np.random.default_rng(seed)

    # 1. Setup Environments and Policies
    # Scenario 1: Good Overlap
    env_good = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=rng_master, reward_type=RewardType.BINARY
    )
    log_good = LinearSoftmaxPolicy(env_good.reward_weights, env_good.reward_bias, inverse_temperature=0.5)
    eval_good = LinearSoftmaxPolicy(env_good.reward_weights, env_good.reward_bias, inverse_temperature=1.0)
    ref_rng1 = np.random.default_rng(1001)
    ref_good = env_good.policy_value(eval_good, ref_rng1, n_contexts=1_000_000)

    # Scenario 2: Poor Overlap
    env_poor = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=rng_master, reward_type=RewardType.BINARY
    )
    log_poor = LinearSoftmaxPolicy(env_poor.reward_weights, env_poor.reward_bias, inverse_temperature=-0.5)
    eval_poor = LinearSoftmaxPolicy(env_poor.reward_weights, env_poor.reward_bias, inverse_temperature=1.5)
    ref_rng2 = np.random.default_rng(1002)
    ref_poor = env_poor.policy_value(eval_poor, ref_rng2, n_contexts=1_000_000)

    scenarios = [
        ("good_overlap", env_good, log_good, eval_good, ref_good),
        ("poor_overlap", env_poor, log_poor, eval_poor, ref_poor),
    ]

    all_results: dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "b_values": [50, 200, 1000],
            "n_samples": 500,
            "n_folds": 3,
            "nominal_coverage": 0.95,
        },
        "scenarios": {},
    }

    n_samples = 500
    n_folds = 3
    b_values = [50, 200, 1000]

    def factory() -> StratifiedLinearRewardModel:
        return StratifiedLinearRewardModel(3, RewardType.BINARY)

    for sc_name, env, log_p, eval_p, ref_val in scenarios:
        print(f"\n=======================================================")
        print(f"Running scenario: {sc_name} (True V = {ref_val.value:.5f} +/- {ref_val.monte_carlo_standard_error:.5f})")
        print(f"=======================================================")

        sc_res: dict[str, Any] = {
            "true_value": ref_val.value,
            "true_mcse": ref_val.monte_carlo_standard_error,
            "ips_snips_m500": {},
            "dr_refit_endpoint_study": {},
            "dr_refit_m50": {},
        }

        # ----------------------------------------------------
        # Part 1: IPS and SNIPS across B in {50, 200, 1000} at M = 500
        # ----------------------------------------------------
        m_ips = 500
        print(f"\n--- Part 1: IPS & SNIPS Bootstrap Sensitivity (M = {m_ips} replications) ---")
        t0 = time.perf_counter()

        ips_data = {B: {"covered": [], "widths": [], "lowers": [], "uppers": []} for B in b_values}
        snips_data = {B: {"covered": [], "widths": [], "lowers": [], "uppers": []} for B in b_values}
        ips_wald_cov = []
        ips_wald_width = []

        for r in range(m_ips):
            r_rng = np.random.default_rng(np.random.SeedSequence([seed, 10, r]))
            data = env.generate_logged_data(n_samples, log_p, r_rng, include_logging_action_dist=True)
            eval_dist = eval_p.action_distribution(data.context)

            res_wald = evaluate_influence_inference("ips", data, eval_dist)
            ips_wald_cov.append(1.0 if res_wald.ci_lower <= ref_val.value <= res_wald.ci_upper else 0.0)
            ips_wald_width.append(res_wald.ci_upper - res_wald.ci_lower)

            for B in b_values:
                b_res_ips = bootstrap_evaluate("ips", data, eval_dist, n_bootstraps=B, seed=r * 100 + B)
                l_i, u_i = b_res_ips.percentile_ci
                ips_data[B]["covered"].append(1.0 if l_i <= ref_val.value <= u_i else 0.0)
                ips_data[B]["widths"].append(u_i - l_i)
                ips_data[B]["lowers"].append(l_i)
                ips_data[B]["uppers"].append(u_i)

                b_res_sn = bootstrap_evaluate("snips", data, eval_dist, n_bootstraps=B, seed=r * 100 + B)
                l_s, u_s = b_res_sn.percentile_ci
                snips_data[B]["covered"].append(1.0 if l_s <= ref_val.value <= u_s else 0.0)
                snips_data[B]["widths"].append(u_s - l_s)
                snips_data[B]["lowers"].append(l_s)
                snips_data[B]["uppers"].append(u_s)

            if (r + 1) % 100 == 0:
                print(f"  Completed {r + 1}/{m_ips} replications in {time.perf_counter() - t0:.1f}s")

        ips_wald_c = float(np.mean(ips_wald_cov))
        ips_wald_mcse = math.sqrt(ips_wald_c * (1 - ips_wald_c) / m_ips)
        sc_res["ips_wald"] = {
            "coverage": ips_wald_c,
            "mcse": ips_wald_mcse,
            "mean_width": float(np.mean(ips_wald_width)),
        }
        print(f"  IPS Wald (M=500): coverage={ips_wald_c:.4f} +/- {ips_wald_mcse:.4f}, mean width={np.mean(ips_wald_width):.4f}")

        for B in b_values:
            c_ips = float(np.mean(ips_data[B]["covered"]))
            mcse_ips = math.sqrt(c_ips * (1 - c_ips) / m_ips)
            w_ips = float(np.mean(ips_data[B]["widths"]))
            sc_res["ips_snips_m500"][f"ips_b{B}"] = {
                "coverage": c_ips,
                "mcse": mcse_ips,
                "mean_width": w_ips,
                "mean_lower": float(np.mean(ips_data[B]["lowers"])),
                "mean_upper": float(np.mean(ips_data[B]["uppers"])),
            }
            print(f"  IPS B={B:4d} (M=500): coverage={c_ips:.4f} +/- {mcse_ips:.4f}, mean width={w_ips:.4f}")

            c_sn = float(np.mean(snips_data[B]["covered"]))
            mcse_sn = math.sqrt(c_sn * (1 - c_sn) / m_ips)
            w_sn = float(np.mean(snips_data[B]["widths"]))
            sc_res["ips_snips_m500"][f"snips_b{B}"] = {
                "coverage": c_sn,
                "mcse": mcse_sn,
                "mean_width": w_sn,
                "mean_lower": float(np.mean(snips_data[B]["lowers"])),
                "mean_upper": float(np.mean(snips_data[B]["uppers"])),
            }
            print(f"  SNIPS B={B:4d} (M=500): coverage={c_sn:.4f} +/- {mcse_sn:.4f}, mean width={w_sn:.4f}")

        # ----------------------------------------------------
        # Part 2: DR Refit Bootstrap Endpoint Study at M = 10 for B in {50, 200, 1000}
        # ----------------------------------------------------
        m_dr10 = 10
        print(f"\n--- Part 2: DR Refit Endpoint Discrepancy Study (M = {m_dr10} replications across B in {{50, 200, 1000}}) ---")
        t0_dr = time.perf_counter()

        dr_data10 = {B: {"covered": [], "widths": [], "lowers": [], "uppers": [], "times": []} for B in b_values}
        diffs_lower_50_1000 = []
        diffs_upper_50_1000 = []
        diffs_lower_200_1000 = []
        diffs_upper_200_1000 = []

        for r in range(m_dr10):
            t_rep = time.perf_counter()
            r_rng = np.random.default_rng(np.random.SeedSequence([seed, 20, r]))
            data = env.generate_logged_data(n_samples, log_p, r_rng, include_logging_action_dist=True)
            eval_dist = eval_p.action_distribution(data.context)

            res_by_b = {}
            for B in b_values:
                t_b0 = time.perf_counter()
                b_res = bootstrap_evaluate(
                    "dr",
                    data,
                    eval_dist,
                    model_factory=factory,
                    n_bootstraps=B,
                    n_folds=n_folds,
                    refit=True,
                    seed=r * 1000 + B,
                )
                dt_b = time.perf_counter() - t_b0
                l_b, u_b = b_res.percentile_ci
                res_by_b[B] = (l_b, u_b)
                dr_data10[B]["covered"].append(1.0 if l_b <= ref_val.value <= u_b else 0.0)
                dr_data10[B]["widths"].append(u_b - l_b)
                dr_data10[B]["lowers"].append(l_b)
                dr_data10[B]["uppers"].append(u_b)
                dr_data10[B]["times"].append(dt_b)

            diffs_lower_50_1000.append(abs(res_by_b[50][0] - res_by_b[1000][0]))
            diffs_upper_50_1000.append(abs(res_by_b[50][1] - res_by_b[1000][1]))
            diffs_lower_200_1000.append(abs(res_by_b[200][0] - res_by_b[1000][0]))
            diffs_upper_200_1000.append(abs(res_by_b[200][1] - res_by_b[1000][1]))

            print(
                f"  DR Rep {r + 1}/{m_dr10} in {time.perf_counter() - t_rep:.1f}s | "
                f"B=50: [{res_by_b[50][0]:.4f}, {res_by_b[50][1]:.4f}] | "
                f"B=200: [{res_by_b[200][0]:.4f}, {res_by_b[200][1]:.4f}] | "
                f"B=1000: [{res_by_b[1000][0]:.4f}, {res_by_b[1000][1]:.4f}]"
            )

        for B in b_values:
            c = float(np.mean(dr_data10[B]["covered"]))
            mcse = math.sqrt(c * (1 - c) / m_dr10)
            w = float(np.mean(dr_data10[B]["widths"]))
            avg_t = float(np.mean(dr_data10[B]["times"]))
            sc_res["dr_refit_endpoint_study"][f"b{B}"] = {
                "coverage": c,
                "mcse": mcse,
                "mean_width": w,
                "mean_lower": float(np.mean(dr_data10[B]["lowers"])),
                "mean_upper": float(np.mean(dr_data10[B]["uppers"])),
                "mean_runtime_seconds": avg_t,
            }
            print(f"  DR Refit B={B:4d} (M={m_dr10}): cov={c:.4f} +/- {mcse:.4f}, width={w:.4f}, runtime={avg_t:.2f}s")

        sc_res["dr_refit_endpoint_study"]["endpoint_discrepancy"] = {
            "mean_abs_diff_lower_50_vs_1000": float(np.mean(diffs_lower_50_1000)),
            "mean_abs_diff_upper_50_vs_1000": float(np.mean(diffs_upper_50_1000)),
            "mean_abs_diff_lower_200_vs_1000": float(np.mean(diffs_lower_200_1000)),
            "mean_abs_diff_upper_200_vs_1000": float(np.mean(diffs_upper_200_1000)),
        }
        print(
            f"  Endpoint discrepancy (B=50 vs B=1000): "
            f"mean |dL| = {np.mean(diffs_lower_50_1000):.4f}, mean |dU| = {np.mean(diffs_upper_50_1000):.4f}"
        )
        print(
            f"  Endpoint discrepancy (B=200 vs B=1000): "
            f"mean |dL| = {np.mean(diffs_lower_200_1000):.4f}, mean |dU| = {np.mean(diffs_upper_200_1000):.4f}"
        )

        # ----------------------------------------------------
        # Part 3: DR Refit Bootstrap at M = 50 for B in {50, 200}
        # ----------------------------------------------------
        m_dr50 = 50
        print(f"\n--- Part 3: DR Refit Coverage Sensitivity (M = {m_dr50} replications for B in {{50, 200}}) ---")
        t0_dr50 = time.perf_counter()
        dr_data50 = {B: {"covered": [], "widths": []} for B in [50, 200]}

        for r in range(m_dr50):
            r_rng = np.random.default_rng(np.random.SeedSequence([seed, 30, r]))
            data = env.generate_logged_data(n_samples, log_p, r_rng, include_logging_action_dist=True)
            eval_dist = eval_p.action_distribution(data.context)

            for B in [50, 200]:
                b_res = bootstrap_evaluate(
                    "dr",
                    data,
                    eval_dist,
                    model_factory=factory,
                    n_bootstraps=B,
                    n_folds=n_folds,
                    refit=True,
                    seed=r * 1000 + B,
                )
                l_b, u_b = b_res.percentile_ci
                dr_data50[B]["covered"].append(1.0 if l_b <= ref_val.value <= u_b else 0.0)
                dr_data50[B]["widths"].append(u_b - l_b)

            if (r + 1) % 10 == 0:
                print(f"  Completed {r + 1}/{m_dr50} in {time.perf_counter() - t0_dr50:.1f}s")

        for B in [50, 200]:
            c = float(np.mean(dr_data50[B]["covered"]))
            mcse = math.sqrt(c * (1 - c) / m_dr50)
            w = float(np.mean(dr_data50[B]["widths"]))
            sc_res["dr_refit_m50"][f"b{B}"] = {
                "coverage": c,
                "mcse": mcse,
                "mean_width": w,
            }
            print(f"  DR Refit B={B:4d} (M={m_dr50}): coverage={c:.4f} +/- {mcse:.4f}, mean width={w:.4f}")

        all_results["scenarios"][sc_name] = sc_res

    out_file = Path("experiments/004_bootstrap_sensitivity/results.json")
    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote full benchmark results to {out_file}")
    return all_results


if __name__ == "__main__":
    run_benchmark()
