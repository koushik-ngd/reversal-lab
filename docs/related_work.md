# Related Work: Distributional Robustness, Covariate Shift & Decision Fragility

**Document Status:** Research Literature Synthesis  
**Milestone:** M4B-1 (Decision-Reversal Scientific Validation)  
**Package:** `reversallab.reversal`  
**Date:** 2026-11-04  

---

## 1. Context and Problem Positioning

Off-policy evaluation (OPE) traditionally assumes that the deployment environment follows the identical context distribution as the logged data:
$$P_{\text{deploy}}(x) \equiv P_{\text{eval}}(x).$$
When this assumption fails—due to demographic shifts, non-stationary user bases, or regional rollouts—standard OPE point estimates and confidence intervals can lead to catastrophic deployment mistakes.

In Milestone M4B-1, we contextualize ReversalLab against established literature in:
1. **Distributionally Robust Off-Policy Evaluation (DRO-OPE)**
2. **Covariate Shift Adaptation**
3. **Sensitivity Analysis in Causal Inference**
4. **Subgroup Disparity and Worst-Case Baselines**

---

## 2. Literature Review & Methodological Taxonomy

### 2.1 Distributionally Robust Off-Policy Evaluation (DRO-OPE)
The primary paradigm for distributionally robust reinforcement learning and bandit evaluation formulates a **forward minimax problem**:
$$\max_{\pi} \min_{P \in \mathcal{U}_\epsilon(P_0)} \mathbb{E}_{x \sim P, a \sim \pi}[r(x, a)]$$
where $\mathcal{U}_\epsilon(P_0)$ is a pre-specified uncertainty set of radius $\epsilon$ around the nominal logging distribution $P_0$.

Key foundations in the literature include:
- **Si et al. (ICML 2020), *Distributionally Robust Policy Evaluation and Learning in Offline Contextual Bandits*:**
  Constructs uncertainty sets based on $f$-divergences (e.g. KL divergence or $\chi^2$-divergence) between the logging context-action distribution and candidate perturbed distributions. Solves a dual convex optimization problem to compute the worst-case policy value.
- **Kallus et al. (ICML 2022), *Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning*:**
  Develops Neyman-orthogonal, doubly robust estimators for worst-case policy values under distribution shift. Demonstrates that naive plug-in DRO estimates suffer from first-order nuisance estimation bias, which is mitigated via semiparametric orthogonal influence scores.
- **Guo et al. (2021), *Distributionally Robust Policy Evaluation under General Covariate Shift in Contextual Bandits*:**
  Specifically isolates covariate shift $P(x)$ while keeping conditional reward mechanisms $P(r \mid x, a)$ invariant. Applies density ratio estimation to bound worst-case performance under Wasserstein and Total Variation neighborhoods.

### 2.2 Sensitivity Analysis in Causal Inference
- **Rosenbaum (2002), *Observational Studies*:**
  Introduces sensitivity bounds (e.g. the Rosenbaum $\Gamma$ parameter) that quantify how strong an unmeasured confounder must be to overturn a statistically significant treatment effect finding.
- **Dorn & Guo (2022), *Doubly Robust Sensitivity Analysis for Causal Inference*:**
  Bridges sensitivity parameters with doubly robust influence functions to calculate bounds on average treatment effects under unobserved confounding.

---

## 3. How ReversalLab Differs from Existing Work

| Dimension | Forward DRO (Si et al. 2020, Kallus et al. 2022) | Sensitivity Analysis (Rosenbaum 2002) | ReversalLab Decision-Reversal (M4) |
|---|---|---|---|
| **Core Question** | "Given uncertainty radius $\epsilon$, what is the worst-case policy value $V_{\text{worst}}(\epsilon)$?" | "How strong must an unmeasured confounder $\Gamma$ be to explain away treatment effect?" | "What is the minimum admissible shift $\epsilon^*$ that reverses an evidence-based deployment decision?" |
| **Mathematical Formulation** | **Forward Minimax:** $\min_{P \in \mathcal{U}_\epsilon} V(\pi; P)$ | **Dual Bound:** $T_{\min}(\Gamma) \le \tau \le T_{\max}(\Gamma)$ | **Inverse Decision Boundary:** $\inf \{ \epsilon \ge 0 : D(P_\epsilon) \neq D_0 \}$ |
| **Input Requirement** | Requires pre-specifying radius $\epsilon$ a priori (notoriously difficult in practice) | Requires tuning confounding sensitivity $\Gamma$ | Inputs operational margin $\delta$; outputs actionable perturbation distance $\epsilon^*$ |
| **Target Decision Metric** | Absolute policy value or worst-case value | Sign of average treatment effect ($\tau = 0$) | Policy advantage exceeding operational margin ($\Delta = V(\pi_e) - V(\pi_b) > \delta$) |
| **Statistical Certification** | Asymptotic worst-case point estimate | Nonparametric permutation bounds | Pointwise and simultaneous confidence bands over admissible perturbation family |

### The Inverse Problem Advantage
In real-world ML deployment, machine learning teams rarely know the appropriate radius $\epsilon$ for a Wasserstein or Total Variation ball. Pre-specifying $\epsilon$ too small yields false confidence; pre-specifying $\epsilon$ too large causes DRO to output excessively pessimistic, uninformative bounds that prevent deploying beneficial policies.

ReversalLab turns the problem inside out: by solving for the **decision-reversal distance** $d_{\mathrm{rev}}^*$, it reports the exact stability margin of the evidence. Decision-makers can directly interpret $d_{\mathrm{rev}}^*$ (e.g. "our recommendation to deploy survives up to an 8% demographic drift, but reverses if the shift exceeds 8%").

---

## 4. Subgroup Baselines and the "Worst-Stratum" Check

A critical scientific question is:
> **Does continuous decision-reversal search add genuine empirical value, or does a simple discrete subgroup check (e.g. checking whether the lower bound in every stratum exceeds zero) provide the identical safety benefit?**

### The Simple Worst-Stratum Baseline (Method C)
Consider a candidate policy evaluated on two monitored strata $S_0$ and $S_1$.
A simple production rule:
$$\text{Deploy if } L_\Delta > \delta \quad \text{AND} \quad \min\big(L_{\Delta, 0}, L_{\Delta, 1}\big) > 0.$$

### When Worst-Stratum Suffices vs When Reversal Search Is Needed
1. **Severe Harm Regimes:** When candidate policy severely harms a subpopulation ($\Delta_1^* < 0$), both the worst-stratum check and the decision-reversal search successfully withhold deployment. In this regime, the worst-stratum check provides a fast, simple safeguard.
2. **Unequal Beneficial Regimes (Near Margin):** Suppose candidate policy is beneficial in both strata ($\Delta_0^* = +0.08, \Delta_1^* = +0.01$), and both lower bounds are positive ($L_{\Delta, 0} > 0, L_{\Delta, 1} > 0$).
   - The worst-stratum rule passes!
   - However, if the operational deployment margin is $\delta = 0.03$ (e.g. required to justify infrastructure migration costs), a demographic shift towards Stratum 1 will drop the aggregate policy advantage below $\delta$.
   - The decision-reversal engine detects this margin fragility ($d_{\mathrm{rev}}^* = 0.08$), whereas the worst-stratum baseline fails to flag the risk because neither stratum is harmful in absolute terms.
3. **Quantifying Safety Margins:** The worst-stratum check outputs a binary Yes/No flag. It cannot answer *how much* demographic drift a deployment can withstand. The reversal distance $d_{\mathrm{rev}}^*$ provides a continuous, interpretable stability budget.

---

## 5. Scope Limits and Scientific Integrity

- ReversalLab does not claim that 1-D two-stratum mixture shift models all real-world distribution shifts.
- It provides a tractable, exact mathematical foundation for the inverse decision-boundary problem.
- In Experiment 006, ReversalLab is directly benchmarked against both point-estimate rules and the discrete worst-stratum baseline across hundreds of independent Monte Carlo replications.
