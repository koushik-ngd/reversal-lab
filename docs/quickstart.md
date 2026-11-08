# ReversalLab Quickstart Guide

ReversalLab is a Python library for off-policy evaluation (OPE) of contextual bandits with rigorous decision-reversal and robustness analysis.

This guide walks you through using ReversalLab both via the command line and through the public Python SDK (`reversallab`).

---

## 1. Quickest Path: Run the Built-In Example

To run the complete end-to-end decision-reversal workflow in one command:

```bash
uv run reversallab run-example --output runs/quickstart_report.json
```

Or run the script directly:

```bash
uv run python examples/quickstart_decision_reversal.py
```

This will:
1. Synthesize $N = 2500$ logged contextual bandit interactions with known ground truth.
2. Define a uniform exploration baseline $\pi_b$ and an improved softmax candidate $\pi_e$.
3. Run leakage-safe 3-fold cross-fitting and paired Doubly Robust (DR) estimation.
4. Verify diagnostic gates (logging propensity support, effective sample size, weight concentration).
5. Search for the minimum evidence-reversing perturbation distance $d^*$ under a 1-D demographic mixture model.
6. Evaluate the forward worst-case robust mixture baseline and simultaneous confidence bands.
7. Output a human-readable ASCII audit report and a schema-versioned JSON report (`v1.0.0`).

---

## 2. Using the Python SDK

### Step 1: Ingest and Validate Logged Bandit Data

ReversalLab enforces strict mathematical contracts on logged data (bounded actions, positive propensities, matching dimensions). Data is wrapped in an immutable `BanditDataset`:

```python
import numpy as np
import reversallab as rl

# Ingest numpy arrays from your logged logs
contexts = np.random.randn(1000, 4)  # (N, D) context features
actions = np.random.randint(0, 3, 1000)  # (N,) discrete actions in [0, K)
rewards = np.random.randn(1000)  # (N,) observed scalar rewards
propensities = np.full(1000, 1.0 / 3.0)  # (N,) logging policy probabilities P(a_i | x_i) > 0

dataset = rl.BanditDataset.from_arrays(
    contexts=contexts,
    actions=actions,
    rewards=rewards,
    propensities=propensities,
    n_actions=3,
)

print(f"Dataset: N={dataset.n_samples}, K={dataset.n_actions}, D={dataset.context_dim}")
```

### Step 2: Evaluate a Single Policy

Estimate the expected reward $V(\pi) = \mathbb{E}[r(\pi)]$ using IPS, SNIPS, Direct Method (DM), or Doubly Robust (DR):

```python
# Define candidate policy (e.g. uniform or linear softmax)
policy = rl.UniformPolicy(n_actions=3)

# Evaluate via Doubly Robust with 3-fold cross-fitting
report = rl.PolicyValueEvaluator.evaluate(
    dataset=dataset,
    policy=policy,
    estimator="dr",
    cross_fitting_folds=3,
    confidence_level=0.95,
)

print(report.summary())
```

### Step 3: Paired Policy Comparison & Decision Reversal

In real deployment decisions, you do not deploy a candidate policy in a vacuum. You compare candidate $\pi_e$ against incumbent $\pi_b$ to determine whether candidate advantage exceeds an operational margin $\delta > 0$:

```python
# Incumbent baseline policy
baseline_policy = rl.UniformPolicy(n_actions=3)

# Candidate policy (e.g. linear softmax policy)
weights = np.random.randn(4, 3)
candidate_policy = rl.LinearSoftmaxPolicy(weights=weights, inverse_temperature=1.0)

# Evaluate paired comparison, diagnostic gates, and decision reversal
comparison_report = rl.PolicyComparison.evaluate(
    dataset=dataset,
    candidate_policy=candidate_policy,
    baseline_policy=baseline_policy,
    estimator="dr",
    margin=0.02,  # Minimum deployment margin required (e.g. +2% reward)
    deploy_epsilon=0.10,  # Operational mixture tolerance envelope [q0 - eps, q0 + eps]
    max_epsilon=0.40,  # Maximum search bound for reversal distance
    confidence_level=0.95,
    cross_fitting_folds=3,
)

# Print human-readable report
print(comparison_report.summary())

# Export machine-readable JSON
comparison_report.save("runs/audit_report.json")
```

---

## 3. Understanding the Decision-Reversal Audit Report

A sample audit report looks like this:

```text
==============================================================================
REVERSALLAB DECISION-REVERSAL AUDIT REPORT (v1.0.0)
==============================================================================
Generated:       2026-11-06T11:21:01.783882+00:00
Dataset:         N = 2500 observations, 3 actions, dim = 2
Evaluation:      Estimator = DR (K = 3-fold cross-fitting)
Policy Margin:   delta = 0.0200 (Confidence level: 95%)

1. PAIRED POLICY ADVANTAGE
   Point Estimate (Delta):   +0.2489 (SE: 0.0084)
   95% Confidence Bound:    [+0.2325, +0.2653]
   Baseline Action:          RECOMMEND (Statistically significant superiority)

2. RELIABILITY DIAGNOSTICS
   Gate Passed:              YES
   Min Logging Propensity:   3.33e-01 (Threshold: 1.00e-04)
   Effective Sample Size:    1148.2 (45.9% of N)
   Max Single-Weight Share:  0.1%

3. DECISION-REVERSAL ROBUSTNESS (1-D Context Mixture Model)
   Stratum Partition:        x[0] >= 0.00
   Reversal Distance (d*):   None detected (Robust across bounds)
   Search Status:            NO_REVERSAL_FOUND_WITHIN_BOUNDS
   Taxonomy Category:        no_reversal
   Forward Worst-Case Bound: +0.2316 (across Q_0.10)
   Robust Deployment Rule:   RECOMMEND
   Simultaneous Band Bound:  +0.2217 (Simult Outcome: RECOMMEND)

4. STATISTICAL ASSUMPTIONS & CLAIMS
   Supported:   Asymptotically valid normal confidence intervals for paired difference
                via Doubly Robust estimation with K-fold cross-fitting.
   Limitations: Does not establish safety against unobserved confounding or
                multidimensional demographic shifts beyond the evaluated partition.
==============================================================================
```

### The Three Decision Rules
- **RECOMMEND**: The lower 95% confidence bound strictly exceeds the operational margin $\delta$. Candidate deployment is statistically justified.
- **HOLD**: The upper 95% confidence bound is strictly below $\delta$. The candidate reliably fails to beat the margin.
- **INCONCLUSIVE**: The confidence interval covers $\delta$, or the multi-factor diagnostic gate failed (e.g., active logging propensity deficit $0 < p_i < \tau$ on an actively selected action, or ESS collapse). A small positive logging propensity $p_i > 0$ is not a mathematical support violation, nor does it inherently imply huge importance weights ($w_i = \pi(a_i \mid x_i) / p_i$ is 0 if $\pi(a_i \mid x_i) = 0$, and remains moderate if $\pi(a_i \mid x_i) \approx p_i$). However, when an evaluation policy actively assigns non-negligible probability mass to a rarely logged action ($\pi(a_i \mid x_i) \gg p_i$), importance weights explode, inducing severe finite-sample variance that invalidates asymptotic normal Wald confidence intervals. Conservative deferral to `INCONCLUSIVE` protects against premature deployment under unstable variance.

### Inverse Reversal Distance vs. Forward Worst-Case
- **Inverse Reversal Distance ($d^*$):** The minimum shift in stratum proportion $|\Delta q| = |q - q_0|$ required to overturn the decision (e.g. from RECOMMEND to INCONCLUSIVE or HOLD). Larger $d^*$ indicates higher robustness.
- **Forward Worst-Case Bound:** The worst-case lower confidence bound across the certified operational envelope $\mathcal{Q}_{\epsilon} = [q_0 - \epsilon, q_0 + \epsilon]$. If the worst-case lower bound exceeds $\delta$, the policy is certifiably robust within $\mathcal{Q}_\epsilon$.
- *Equivalence Note:* In the single-dimensional two-stratum setting, both formulations yield identical binary deployment decisions at budget $\epsilon_{\text{deploy}}$.

---

## 4. CLI Commands Reference

| Command | Purpose | Example |
|---|---|---|
| `run-example` | Run complete end-to-end synthetic demo | `uv run reversallab run-example --output runs/report.json` |
| `validate-estimators` | Validate single-policy OPE estimators | `uv run reversallab validate-estimators --config configs/experiments/001_estimator_validation.yaml` |
| `compare-reference` | Check parity against Open Bandit Pipeline | `uv run reversallab compare-reference --config configs/experiments/002_open_bandit_parity.yaml` |
| `run-reversal` | Run batch synthetic reversal experiment | `uv run reversallab run-reversal --config configs/experiments/005_reversal_prototype.yaml` |
| `report` | Format and display saved `run.json` artifact | `uv run reversallab report --run-dir runs/<run_id>` |

---

## 5. Standalone Wheel Installation with `uv`

ReversalLab can be built and installed as a standalone Python wheel across Linux, macOS, and Windows using `uv`'s native package management:

```bash
# 1. Build the standalone pure Python wheel
uv build --wheel

# 2. Create an isolated virtual environment
uv venv .smoke_env

# 3. Install the built wheel into the isolated virtual environment via uv
uv pip install --python .smoke_env dist/reversallab-0.1.0rc1-py3-none-any.whl

# 4. Verify public SDK import and CLI execution
uv run --no-project --python .smoke_env python -c "import reversallab as rl; print(rl.__version__)"
uv run --no-project --python .smoke_env python -m reversallab.cli run-example --output .smoke_env/report.json
```
