# Decision-Reversal Research Specification (M4A)

**Document Status:** Formal Specification  
**Milestone:** M4A (Decision-Reversal Engine, Research Prototype)  
**Package:** `reversallab.reversal`  
**Authors:** ReversalLab Research Team  
**Date:** 2026-11-04  

---

## 1. Executive Summary & Core Research Question

Off-policy evaluation (OPE) typically produces a scalar policy-value point estimate $\hat{V}(\pi)$ accompanied by an asymptotic or bootstrap confidence interval $[L, U]$. However, in real-world ML deployment, these point estimates and confidence intervals rest on fragile identifying assumptions:
1. The target context distribution matches the logging distribution ($P_{\text{deploy}}(x) \equiv P_{\text{eval}}(x)$).
2. The logging policy's propensities $p_i = \pi_0(a_i \mid x_i)$ are known exactly and well-conditioned.
3. The evaluation sample is representative across all relevant sub-populations and feature strata.

When these assumptions are violated even slightly, an evidence-based recommendation to deploy a candidate policy can reverse, leading to negative business impact or user harm.

The fundamental research question of Milestone M4 is:

> **What is the smallest scientifically admissible change in evaluation assumptions that reverses an evidence-based decision about deploying a candidate ML policy?**

Milestone M4A establishes the mathematical foundations, formal decision rules, a 1-dimensional admissible context mixture perturbation model, and numerical bisection search algorithms for this question.

> [!IMPORTANT]
> **M4A Scope Boundary:** M4A is a research prototype. It does not claim that decision-reversal constitutes a fully validated or novel production methodology yet. High-dimensional adversarial search, unconstrained distribution shifts, automated propensity manipulation, reward relabeling, and learned policy selection on evaluation logs are explicitly out of scope for M4A.

---

## 2. The Deployment Comparison Problem

### 2.1 Candidate and Baseline Policies
We consider a deployment decision between two pre-specified, fixed policies evaluated on the same logged bandit dataset $\mathcal{D} = \{(x_i, a_i, r_i, p_i)\}_{i=1}^N$:
- **Baseline Policy ($\pi_b$):** The status-quo production policy with action distribution $\pi_b(a \mid x)$.
- **Candidate Policy ($\pi_e$):** The proposed challenger policy with action distribution $\pi_e(a \mid x)$.

We define the true population policy-value difference as:
$$\Delta = V(\pi_e) - V(\pi_b) = \mathbb{E}_{x \sim P(x), a \sim \pi_e(\cdot \mid x)}[r(x, a)] - \mathbb{E}_{x \sim P(x), a \sim \pi_b(\cdot \mid x)}[r(x, a)]$$

A positive $\Delta > 0$ indicates that the candidate policy outperforms the baseline policy in expectation.

### 2.2 Paired Per-Observation Difference Scores
Evaluating $\pi_e$ and $\pi_b$ on the *same* logged observations $\mathcal{D}$ induces strong positive sample covariance between their estimated values. Treating the estimates $\hat{V}(\pi_e)$ and $\hat{V}(\pi_b)$ as independent:
$$\mathrm{Var}_{\text{indep}}(\hat{\Delta}) = \mathrm{Var}(\hat{V}(\pi_e)) + \mathrm{Var}(\hat{V}(\pi_b))$$
is statistically incorrect and needlessly conservative.

Instead, we compute **paired per-observation difference scores**:
$$\psi_{\Delta, i} = \psi_{e, i} - \psi_{b, i}, \quad i = 1, \dots, N$$

where $\psi_{e, i}$ and $\psi_{b, i}$ are the influence scores (or pseudo-outcomes) of $\pi_e$ and $\pi_b$ on observation $i$.

#### Supported Estimators
1. **Doubly Robust (DR):**
   $$\psi_{\Delta, i}^{\mathrm{DR}} = \sum_{a=1}^K \big(\pi_e(a \mid x_i) - \pi_b(a \mid x_i)\big) \hat{q}(x_i, a) + \frac{\pi_e(a_i \mid x_i) - \pi_b(a_i \mid x_i)}{p_i} \big(r_i - \hat{q}(x_i, a_i)\big)$$
2. **Inverse Probability Scoring (IPS):**
   $$\psi_{\Delta, i}^{\mathrm{IPS}} = \frac{\pi_e(a_i \mid x_i) - \pi_b(a_i \mid x_i)}{p_i} \, r_i$$
3. **Direct Method (DM):**
   $$\psi_{\Delta, i}^{\mathrm{DM}} = \sum_{a=1}^K \big(\pi_e(a \mid x_i) - \pi_b(a \mid x_i)\big) \hat{q}(x_i, a)$$
4. **Self-Normalized IPS (SNIPS):**
   Using delta-method linearized influence scores centered at the difference point estimate $\hat{\Delta}_{\text{SNIPS}} = \hat{V}_{\text{SNIPS}}(\pi_e) - \hat{V}_{\text{SNIPS}}(\pi_b)$:
   $$\psi_{\Delta, i}^{\mathrm{SNIPS}} = \hat{\Delta}_{\text{SNIPS}} + \frac{w_{e, i}}{\bar{w}_e} \big(r_i - \hat{V}_{\text{SNIPS}}(\pi_e)\big) - \frac{w_{b, i}}{\bar{w}_b} \big(r_i - \hat{V}_{\text{SNIPS}}(\pi_b)\big)$$

### 2.3 Paired Inference & Covariance Accounting
The sample mean of the paired scores is the point estimate:
$$\hat{\Delta} = \frac{1}{N} \sum_{i=1}^N \psi_{\Delta, i} = \hat{V}(\pi_e) - \hat{V}(\pi_b)$$

The empirical sample variance is:
$$s_\Delta^2 = \frac{1}{N - 1} \sum_{i=1}^N (\psi_{\Delta, i} - \hat{\Delta})^2 = s_e^2 + s_b^2 - 2 s_{e, b}$$
where $s_{e, b} = \frac{1}{N - 1} \sum_{i=1}^N (\psi_{e, i} - \hat{V}_e)(\psi_{b, i} - \hat{V}_b)$ is the empirical covariance between the policy scores.

When $\pi_e$ and $\pi_b$ share positive correlation ($s_{e, b} > 0$), the paired variance is strictly smaller than the independent variance:
$$s_\Delta^2 < s_e^2 + s_b^2$$

The paired standard error and $(1 - \alpha)$ two-sided Wald confidence interval are:
$$\widehat{\mathrm{se}}_\Delta = \sqrt{\frac{s_\Delta^2}{N}}, \qquad [L_\Delta, U_\Delta] = [\hat{\Delta} - z_{1-\alpha/2}\widehat{\mathrm{se}}_\Delta, \; \hat{\Delta} + z_{1-\alpha/2}\widehat{\mathrm{se}}_\Delta]$$

### 2.4 Pre-specification & Selective Inference Restrictions
In M4A, candidate and baseline policies must be fixed prior to evaluation. Selecting a candidate policy from a pool of models evaluated on the same logged rewards introduces the *winner's curse* (selection bias). Accounting for post-selection inference is deferred to future work.

---

## 3. Formal Decision Framework

### 3.1 Decision Outcomes
A deployment decision rule maps logged evidence $\mathcal{D}$ and a minimum meaningful improvement margin $\delta \ge 0$ to an outcome $D \in \{\text{RECOMMEND}, \text{HOLD}, \text{INCONCLUSIVE}\}$:

| Outcome | Meaning | Statistical Criterion |
|---|---|---|
| **`RECOMMEND`** | Sufficient statistical evidence that candidate improvement strictly exceeds margin $\delta$. | $L_\Delta > \delta$ |
| **`HOLD`** | Sufficient statistical evidence that candidate improvement is strictly below margin $\delta$. | $U_\Delta < \delta$ |
| **`INCONCLUSIVE`** | Evidence is insufficient to determine whether $\Delta > \delta$ or $\Delta < \delta$, OR diagnostic checks fail. | $L_\Delta \le \delta \le U_\Delta$ or $\text{diagnostic\_failure}$ |

### 3.2 Exact Boundary Behavior
- Strict inequalities ($L_\Delta > \delta$ and $U_\Delta < \delta$) are required for positive action (`RECOMMEND` or `HOLD`).
- Boundary equalities ($L_\Delta = \delta$ or $U_\Delta = \delta$) are strictly classified as `INCONCLUSIVE`.
- This conservative boundary convention prevents false recommendations when the confidence limit exactly touches the operational margin.

### 3.3 Diagnostic Gating
Before applying interval thresholds, the decision engine evaluates operational diagnostics:
1. **Minimum Propensity Check:** If $\min_{i} p_i < p_{\min}^{\text{thresh}}$ (default $10^{-4}$), inference is flagged as unreliable $\implies \text{INCONCLUSIVE}$.
2. **Effective Sample Size (ESS) Check:** If $n_{\text{eff}} < n_{\min}^{\text{thresh}}$, high-variance weights invalidate asymptotic normality $\implies \text{INCONCLUSIVE}$.
3. **Weight Degeneracy / Finite Checks:** If non-finite scores or weights occur $\implies \text{INCONCLUSIVE}$.

### 3.4 The Hazard of Naive Point-Estimate Decisions
A naive decision rule deploys whenever $\hat{\Delta} > \delta$, ignoring sampling variance $\widehat{\mathrm{se}}_\Delta$.
- If $\Delta = \delta + 0.001$ with large $\widehat{\mathrm{se}}_\Delta = 0.050$, a naive rule recommends deployment 50.8% of the time, while the true probability of underperforming $\delta$ is nearly 50%.
- The interval-based rule classifies this regime as `INCONCLUSIVE`, protecting production systems from unpowered gambles.

---

## 4. Admissible Context Mixture Perturbation Model ($\mathcal{Q}_\epsilon$)

### 4.1 Motivation
Real-world distribution shift frequently manifests as demographic, temporal, or geographic mixture changes: the conditional reward distributions within sub-populations $P(r \mid x, a)$ remain relatively stable, but the proportions of distinct user segments shift.

### 4.2 Stratum Partition
Let $h: \mathcal{X} \to \{0, 1\}$ partition the context space into two disjoint strata:
$$S_0 = \{x \in \mathcal{X} : h(x) = 0\}, \qquad S_1 = \{x \in \mathcal{X} : h(x) = 1\}$$
For continuous contexts, the canonical partition is the coordinate hyperplane $h(x) = \mathbb{I}(x_0 \ge 0)$.

Let $I_0 = \{i : h(x_i) = 0\}$ and $I_1 = \{i : h(x_i) = 1\}$ index the logged observations falling in each stratum, with sample sizes $N_0 = |I_0|$ and $N_1 = |I_1|$ ($N = N_0 + N_1$).

The baseline empirical mixture weights are:
$$q_{0, 0} = \frac{N_0}{N}, \qquad q_{1, 0} = \frac{N_1}{N} = 1 - q_{0, 0}$$

### 4.3 Stratum-Level Estimates
In each stratum $s \in \{0, 1\}$, using the paired scores $\psi_{\Delta, i}$ for $i \in I_s$:
$$\hat{\Delta}_s = \frac{1}{N_s} \sum_{i \in I_s} \psi_{\Delta, i}, \qquad s_s^2 = \frac{1}{N_s - 1} \sum_{i \in I_s} (\psi_{\Delta, i} - \hat{\Delta}_s)^2, \qquad \widehat{\mathrm{se}}_s = \sqrt{\frac{s_s^2}{N_s}}$$

### 4.4 The Perturbation Family $\mathcal{Q}_\epsilon$
Let $q = (q_0, 1 - q_0) \in [0, 1]^2$ denote an alternative mixture distribution over the two strata.
The Total Variation (TV) distance between mixture $q$ and baseline $q_{\text{base}} = (q_{0,0}, 1-q_{0,0})$ is:
$$d_{\mathrm{TV}}(q, q_{\text{base}}) = \frac{1}{2} \big(|q_0 - q_{0,0}| + |(1 - q_0) - (1 - q_{0,0})|\big) = |q_0 - q_{0,0}|$$

For an admissible perturbation budget $\epsilon \ge 0$, the set of admissible distributions is:
$$\mathcal{Q}_\epsilon = \big\{ q = (q_0, 1 - q_0) : |q_0 - q_{0,0}| \le \epsilon, \; 0 \le q_0 \le 1 \big\}$$

This restricts $q_0$ to the compact interval:
$$q_0 \in \big[\max(0, q_{0,0} - \epsilon), \; \min(1, q_{0,0} + \epsilon)\big]$$

### 4.5 Perturbed Policy-Value Difference & Uncertainty
Under mixture $q$, the true policy-value difference is:
$$\Delta(q) = q_0 \Delta_0 + (1 - q_0) \Delta_1$$

The sample plug-in estimator under mixture $q$ is:
$$\hat{\Delta}(q) = q_0 \hat{\Delta}_0 + (1 - q_0) \hat{\Delta}_1$$

Because observations in $S_0$ and $S_1$ are disjoint independent samples, $\mathrm{Cov}(\hat{\Delta}_0, \hat{\Delta}_1) = 0$. Hence:
$$\mathrm{Var}\big(\hat{\Delta}(q)\big) = q_0^2 \mathrm{Var}(\hat{\Delta}_0) + (1 - q_0)^2 \mathrm{Var}(\hat{\Delta}_1) = q_0^2 \frac{s_0^2}{N_0} + (1 - q_0)^2 \frac{s_1^2}{N_1}$$

The standard error under mixture $q$ is:
$$\widehat{\mathrm{se}}_\Delta(q) = \sqrt{q_0^2 \widehat{\mathrm{se}}_0^2 + (1 - q_0)^2 \widehat{\mathrm{se}}_1^2}$$

The $(1 - \alpha)$ Wald confidence limits under mixture $q$ are:
$$L_\Delta(q) = \hat{\Delta}(q) - z_{1-\alpha/2} \widehat{\mathrm{se}}_\Delta(q), \qquad U_\Delta(q) = \hat{\Delta}(q) + z_{1-\alpha/2} \widehat{\mathrm{se}}_\Delta(q)$$

The decision rule under mixture $q$ is $D(q) = \text{evaluate\_decision}(L_\Delta(q), U_\Delta(q), \delta)$.

---

## 5. 1-Dimensional Decision-Reversal Search

### 5.1 Reversal Distance Definition
Let $D_0 = D(q_{\text{base}})$ be the baseline deployment decision.
The minimum decision-reversal perturbation distance is:
$$d_{\mathrm{rev}}^* = \inf \big\{ \epsilon \ge 0 : \exists q \in \mathcal{Q}_\epsilon \text{ such that } D(q) \neq D_0 \big\}$$

If $D(q) = D_0$ for all admissible $q \in \mathcal{Q}_{\epsilon_{\max}}$, the search reports:
$$d_{\mathrm{rev}}^* = \text{None}, \qquad \text{status} = \text{"NO\_REVERSAL\_FOUND\_WITHIN\_BOUNDS"}$$

### 5.2 Transition Types
We categorize transitions explicitly:
1. `RECOMMEND -> INCONCLUSIVE`: Loss of statistical certitude (fragility).
2. `RECOMMEND -> HOLD`: Complete policy rejection (inversion).
3. `HOLD -> INCONCLUSIVE`: Loss of negative certitude.
4. `HOLD -> RECOMMEND`: Inversion to adoption.
5. `INCONCLUSIVE -> RECOMMEND`: Acquisition of positive significance.
6. `INCONCLUSIVE -> HOLD`: Acquisition of negative significance.

### 5.3 Two-Stage Search Algorithm: Grid Search + Bisection Refinement

```
Algorithm 1: 1D Reversal Search with Bisection Refinement
Input: Strata statistics (Δ̂₀, sê₀, Δ̂₁, sê₁), baseline q₀,₀, margin δ, budget ε_max, grid resolution M, tolerance tol
Output: ReversalResult (d_rev*, reversed_decision, reversed_q, transition, status)

1. Evaluate baseline decision D₀ = evaluate_decision(q₀,₀)
2. If D₀ is INCONCLUSIVE due to diagnostic failure:
     Return ReversalResult(d_rev=0.0, status="BASELINE_DIAGNOSTIC_FAILURE")
3. Construct search ranges:
     Left direction:  q₀ ∈ [max(0, q₀,₀ - ε_max), q₀,₀]
     Right direction: q₀ ∈ [q₀,₀, min(1, q₀,₀ + ε_max)]
4. For each direction (left, right):
     Evaluate coarse grid with M points.
     Identify first adjacent pair (q_prev, q_curr) where D(q_curr) ≠ D₀.
     If transition found:
       Bracket root between q_a = q_prev and q_b = q_curr.
       Perform bisection on f(q₀) = (L_Δ(q₀) - δ) or (U_Δ(q₀) - δ) until |q_b - q_a| < tol.
       Record candidate reversal: q* = (q_a + q_b) / 2, ε = |q* - q₀,₀|.
5. Select minimum ε among valid transitions:
     If no transitions found:
       Verify on fine grid (500 points) across [q₀,₀ - ε_max, q₀,₀ + ε_max].
       Return ReversalResult(d_rev=None, status="NO_REVERSAL_FOUND_WITHIN_BOUNDS")
     Else:
       Return ReversalResult(d_rev=ε*, reversed_decision=D(q*), reversed_q=q*, transition=D₀ → D(q*))
```

### 5.4 Search Invariants & Guarantees
1. **Monotonicity along rays:** Because $\hat{\Delta}(q)$ is linear in $q_0$ and $\widehat{\mathrm{se}}_\Delta(q)$ is strictly convex in $q_0$, the bounds $L_\Delta(q_0)$ and $U_\Delta(q_0)$ vary smoothly. Crossing points with $\delta$ are isolated roots.
2. **Fine-Grid Validation:** Every bisection root is checked against fine-grid evaluation to confirm consistency.

---

## 6. Synthetic Experimental Design (Experiment 005)

To evaluate whether the decision-reversal engine identifies genuine fragility rather than arbitrary artifacts, we benchmark four synthetic environments:

### Scenario Matrix
1. **Scenario A (Robust Superiority):**
   - Candidate strongly beats baseline in *both* strata ($\Delta_0^* > \delta$ and $\Delta_1^* > \delta$).
   - Expected behavior: Baseline decision is `RECOMMEND`. Reversal search finds no admissible shift within $\epsilon \le 0.40$ ($d_{\mathrm{rev}}^* = \text{None}$).
2. **Scenario B (Fragile Superiority):**
   - Candidate beats baseline in aggregate ($\Delta^* > \delta$) and in stratum 0 ($\Delta_0^* \gg \delta$), but is *inferior* in stratum 1 ($\Delta_1^* < 0$).
   - Expected behavior: Baseline decision is `RECOMMEND`. A modest context shift towards stratum 1 ($\epsilon \approx 0.08$–$0.15$) reverses the decision to `INCONCLUSIVE` or `HOLD`.
3. **Scenario C (High Uncertainty / Underpowered):**
   - Candidate has positive expected margin, but sample size is small or variance is high so the baseline CI spans $\delta$.
   - Expected behavior: Baseline decision is `INCONCLUSIVE`. Naive point estimate falsely says `RECOMMEND`, demonstrating the safety value of the confidence-bound decision rule.
4. **Scenario D (Overlap Violation):**
   - Logging policy suffers from severe propensity deficiency ($\min p_i < 10^{-4}$).
   - Expected behavior: Diagnostic gate trips, yielding baseline `INCONCLUSIVE` and halting invalid perturbation inference.

---

## 7. Mathematical Summary Table

| Quantity | Mathematical Formulation | Code Symbol |
|---|---|---|
| Paired Score | $\psi_{\Delta, i} = \psi_{e, i} - \psi_{b, i}$ | `diff_scores` |
| Point Estimate | $\hat{\Delta} = \frac{1}{N}\sum_{i=1}^N \psi_{\Delta, i}$ | `point_estimate` |
| Paired Sample Variance | $s_\Delta^2 = s_e^2 + s_b^2 - 2 s_{e, b}$ | `empirical_variance` |
| Standard Error | $\widehat{\mathrm{se}}_\Delta = \sqrt{s_\Delta^2 / N}$ | `standard_error` |
| Confidence Interval | $[L_\Delta, U_\Delta] = [\hat{\Delta} \mp z_{1-\alpha/2}\widehat{\mathrm{se}}_\Delta]$ | `(ci_lower, ci_upper)` |
| Total Variation Distance | $d_{\mathrm{TV}}(q, q_0) = \|q_0 - q_{0,0}\|$ | `tv_distance` |
| Mixture Difference | $\hat{\Delta}(q) = q_0 \hat{\Delta}_0 + (1 - q_0) \hat{\Delta}_1$ | `mixture_diff` |
| Mixture Standard Error | $\widehat{\mathrm{se}}_\Delta(q) = \sqrt{q_0^2 \widehat{\mathrm{se}}_0^2 + (1 - q_0)^2 \widehat{\mathrm{se}}_1^2}$ | `mixture_se` |
| Reversal Distance | $d_{\mathrm{rev}}^* = \inf \{ \epsilon \ge 0 : D(q) \neq D_0 \}$ | `reversal_distance` |
