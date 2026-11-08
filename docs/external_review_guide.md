# ReversalLab — External Review & Reproduction Guide (v0.1.0-rc1)

This guide provides an independent machine learning engineer, researcher, or open-source reviewer with everything needed to evaluate, reproduce, and audit ReversalLab.

---

## 1. Project Overview

ReversalLab is an open-source Python library and evaluation service for statistically grounded off-policy evaluation (OPE) of contextual bandits, paired policy superiority testing, and deployment-decision robustness analysis under covariate shift.

Rather than relying solely on point estimates or standard confidence intervals that collapse under distribution shift, ReversalLab asks:
> **What is the smallest admissible change in population demographics or evaluation assumptions that reverses an evidence-based deployment decision?**

---

## 2. Five-Minute Quickstart

### Step 1: Environment Setup
Ensure Python 3.11+ is installed. Clone the repository and sync dependencies:

```bash
git clone https://github.com/koushik-ngd/reversal-lab.git
cd reversal-lab
uv sync --frozen --all-groups
```

### Step 2: Run End-to-End CLI Demonstration
```bash
uv run reversallab run-example --output runs/quickstart_report.json
```

Expected terminal output:
```text
==============================================================================
REVERSALLAB DECISION-REVERSAL AUDIT REPORT (v1.0.0)
==============================================================================
Generated:       2026-11-08T...
Dataset:         N = 2500 observations, 3 actions, dim = 2
Evaluation:      Estimator = DR (3-fold cross-fitting)
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
==============================================================================
```

### Step 3: Run Interactive Jupyter Notebook Tutorial
```bash
uv run jupyter lab notebooks/quickstart_decision_reversal.ipynb
```

---

## 3. Scientific Assumptions & Theoretical Grounding

ReversalLab relies on standard causal and off-policy assumptions:
1. **Unconfoundedness (Conditional Ignorability)**: $Y(a) \perp A \mid X$. Action choices in historical logged data depended only on observed contexts $X$.
2. **Propensity Overlap (Positivity)**: For all actions evaluated by the target policies, logging probabilities satisfy $\pi_0(a \mid x) \ge \tau > 0$.
3. **Correct Logging Probabilities**: Recorded propensities $p_i$ reflect true logging probabilities $\pi_0(a_i \mid x_i)$.
4. **Leakage-Safe Nuisance Estimation**: Regression reward models $\hat{\mu}(x, a)$ are trained strictly out-of-fold using $K$-fold cross-fitting ($K \ge 2$).

### Mathematical Relationship: Inverse Reversal vs. Forward Robust Mixture
Under a 1-D demographic mixture perturbation between two strata $\mathcal{S}_0$ and $\mathcal{S}_1$, the perturbed advantage is affine:
$$\Delta(q) = (1 - q)\Delta_0 + q\Delta_1, \quad q \in [0, 1]$$
- **Inverse Reversal Search**: Solves $\Delta(q^*) = \delta \implies q^* = \frac{\delta - \Delta_0}{\Delta_1 - \Delta_0}$, reporting perturbation distance $d^* = |q^* - q_0|$ and reversal threshold $\epsilon^* = d^*$.
- **Forward Robust Mixture**: Evaluates the worst-case advantage within operational tolerance ball $\mathcal{Q}_\epsilon = [q_0 - \epsilon_{\text{deploy}}, q_0 + \epsilon_{\text{deploy}}] \cap [0, 1]$:
  $$\Delta_{\text{worst}}(\epsilon_{\text{deploy}}) = \min_{q \in \mathcal{Q}_\epsilon} \Delta(q)$$
- **Equivalence Note**: In 1-D two-stratum mixture shifts, inverse reversal thresholding ($\epsilon^* \ge \epsilon_{\text{deploy}}$) and forward worst-case advantage checking ($\Delta_{\text{worst}}(\epsilon_{\text{deploy}}) \ge \delta$) produce mathematically identical binary deployment decisions. ReversalLab reports distance-to-failure as an intuitive metric, not as an independent physical law.
- **Simultaneous Confidence Bands**: Simultaneous coverage across $q \in [0, 1]$ via Bonferroni-corrected stratum endpoint intervals guarantees uniform validity. Because $\text{LCB}(q)$ is strictly concave under mixture shifts, the minimum always occurs at the boundary of $\mathcal{Q}_\epsilon$.

---

## 4. Reproducing the Seven Research Milestones

ReversalLab was developed through seven audited empirical experiments:

```bash
# Experiment 001: Estimator validation (bias/variance across sample sizes)
uv run reversallab validate-estimators --config configs/experiments/001_estimator_validation.yaml

# Experiment 002: Numerical parity against Open Bandit Pipeline (OBP)
uv run reversallab compare-reference --config configs/experiments/002_open_bandit_parity.yaml

# Experiment 003: Advanced estimators (DM, DR with K-fold cross-fitting)
uv run reversallab validate-estimators --config configs/experiments/003_advanced_estimators.yaml

# Experiment 004: Uncertainty calibration and Wald coverage
uv run reversallab validate-estimators --config configs/experiments/004_uncertainty_calibration.yaml

# Experiment 005: Prototype 1-D reversal search
uv run reversallab run-reversal --config configs/experiments/005_reversal_prototype.yaml

# Experiment 006: Analytical root-finding & simultaneous band validation
uv run reversallab run-reversal --config configs/experiments/006_reversal_validation.yaml

# Experiment 007: Independent reliability (5,000 Monte Carlo replications)
uv run reversallab run-reversal --config configs/experiments/007_independent_reliability.yaml
```

---

## 5. Deployment Model A & Security Boundary

- **Trusted Single-User Boundary**: ReversalLab assumes clients are trusted entities on localhost or inside a private container network.
- **No Built-in Authentication**: `job_id` and `payload_hash` are operational routing tokens, not access credentials.
- **Production Exposure Rules**: Public exposure requires an external reverse proxy (Envoy, NGINX) providing TLS termination, client authentication, and rate limiting.
- **Storage Persistence**: Docker containers must mount a persistent volume to `/home/reversallab/storage/jobs` to preserve SQLite metadata across restarts.
- Details in [`SECURITY.md`](../SECURITY.md).

---

## 6. Known Scientific & Engineering Limitations

1. **Unobserved Confounding**: Cannot detect or correct for unobserved confounders in historical data.
2. **Overlap Violations**: If $p_i \to 0$ for target policy actions, importance weights diverge. Diagnostic gates flag this, but cannot correct for structural support lack.
3. **1-D Mixture Scope**: The current analytical reversal solver operates on 1-D context partitions. It does not evaluate continuous non-parametric density drift or multidimensional adversarial Wasserstein perturbations.
4. **Single-Node Scale**: The background job service is strictly single-node.

---

## 7. Questions for External Reviewers

Reviewers are encouraged to scrutinize:
1. Is the out-of-fold $K$-fold cross-fitting implementation in `src/reversallab/estimators/cross_fitting.py` free of target leakage?
2. Are the diagnostic gate thresholds ($\tau = 10^{-4}$, $\text{ESS}/N \ge 0.05$, max single weight $\le 0.20$) practical across varied industry datasets?
3. Is the mathematical equivalence between inverse reversal search and forward robust mixture evaluation clearly presented without overclaiming?
4. Does the decoupled state machine (`status: COMPLETED` vs `artifact_available: bool`) adequately preserve auditability during storage failures?
