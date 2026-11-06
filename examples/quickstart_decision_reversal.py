"""ReversalLab End-to-End Quickstart Example.

Demonstrates the core ReversalLab workflow:
1. Synthesize logged contextual bandit data under a known logging policy.
2. Define a deployed baseline policy and a candidate improved policy.
3. Fit a reward model using leakage-safe K-fold cross-fitting.
4. Estimate the paired policy advantage Delta = V(pi_e) - V(pi_b) with Doubly Robust estimation.
5. Inspect operational diagnostic gates (min propensity, effective sample size, max weight share).
6. Execute bounded 1-D context mixture decision-reversal search.
7. Produce and export a structured, reproducible decision audit report.

Usage::

    uv run python examples/quickstart_decision_reversal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import reversallab as rl
from reversallab.data.synthetic import SyntheticBanditEnvironment


def run_quickstart(output_path: str | Path | None = None) -> rl.PolicyComparisonReport:
    """Execute complete end-to-end policy evaluation and decision-reversal workflow."""
    print("=" * 78)
    print("ReversalLab End-to-End Decision-Reversal Workflow")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Environment and Logged Bandit Data Generation
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing synthetic environment and logging data...")
    rng = np.random.default_rng(42)
    env = SyntheticBanditEnvironment.random(
        n_actions=3,
        context_dim=2,
        rng=rng,
        reward_scale=1.5,
        bias_scale=0.0,
    )

    logging_policy = rl.UniformPolicy(n_actions=env.n_actions)
    dataset = rl.BanditDataset.from_synthetic(
        env,
        logging_policy=logging_policy,
        n_samples=2500,
        seed=101,
    )
    print(
        f"  Generated N = {dataset.n_samples} observations across {dataset.n_actions} actions "
        f"(context dim = {dataset.context_dim})."
    )

    # -------------------------------------------------------------------------
    # 2. Define Baseline and Candidate Policies
    # -------------------------------------------------------------------------
    print("\n[Step 2] Defining baseline (pi_b) and candidate (pi_e) policies...")
    # Baseline: uniform exploratory policy
    baseline_policy = rl.UniformPolicy(n_actions=env.n_actions)

    # Candidate: softmax policy targeting true environment reward logits
    candidate_policy = rl.LinearSoftmaxPolicy(
        weights=env.reward_weights,
        bias=env.reward_bias,
        inverse_temperature=1.5,
    )
    print("  Baseline:  Uniform exploration over 3 actions.")
    print("  Candidate: Linear softmax policy informed by environment reward logits.")

    # -------------------------------------------------------------------------
    # 3. Paired Difference Estimation & Decision-Reversal Analysis
    # -------------------------------------------------------------------------
    margin = 0.02  # Minimum 2% business improvement required for deployment
    deploy_eps = 0.10  # Certified deployment robustness envelope Q_0.10 = [0.40, 0.60]
    partition = rl.StratumPartition(split_dim=0, split_threshold=0.0)

    print("\n[Step 3] Running leakage-safe DR evaluation and reversal search...")
    print(f"  Operational margin delta:    {margin:.4f}")
    print(f"  Deployment tolerance ball:   epsilon = {deploy_eps:.2f}")
    print("  Context partition:           x[0] >= 0.0")

    report: rl.PolicyComparisonReport = rl.PolicyComparison.evaluate(
        dataset,
        candidate_policy=candidate_policy,
        baseline_policy=baseline_policy,
        margin=margin,
        deploy_epsilon=deploy_eps,
        max_epsilon=0.40,
        confidence_level=0.95,
        stratum_partition=partition,
        estimator="dr",
        cross_fitting_folds=3,
        min_propensity_threshold=1e-4,
        seed=42,
    )

    # -------------------------------------------------------------------------
    # 4. Display Ergonomic Human-Readable Summary
    # -------------------------------------------------------------------------
    print("\n[Step 4] Reviewing structured decision audit report:")
    print(report.summary())

    # -------------------------------------------------------------------------
    # 5. Export Reproducible Machine-Readable Report
    # -------------------------------------------------------------------------
    save_file = Path(output_path) if output_path else Path("runs/quickstart_report.json")
    save_file.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(save_file)
    print(f"\n[Step 5] Exported machine-readable JSON report to: {save_file}")

    return report


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/quickstart_report.json"
    run_quickstart(out)
