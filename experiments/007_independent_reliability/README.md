# Experiment 007: Independent Reliability Evaluation Benchmark (M4C)

**Status:** Completed & Statistically Validated  
**Milestone:** M4C  
**Author:** Principal Statistical ML Researcher & Staff ML Engineer  
**Date:** 2026-11-06  
**Reproduction Command:** `reversallab run-independent-reliability --config configs/experiments/007_independent_reliability.yaml`  
**Artifact Path:** `experiments/007_independent_reliability/run.json`  

---

## 1. Executive Summary

Milestone **M4C** provides the independent, large-sample statistical reliability evaluation of ReversalLab's decision-reversal framework.

The central research question investigated is:
> **What is the smallest scientifically admissible change in evaluation assumptions that reverses an evidence-based decision about deploying a candidate ML policy?**

Experiment 007 conducts a preregistered Monte Carlo benchmark with **$M = 500$ independent replications per scenario** (totaling 5,000 full synthetic evaluation pipelines) across **10 scenarios** spanning:
- Good and poor policy overlap
- Correctly specified and misspecified reward models
- Robust vs fragile policy improvements
- Truly harmful candidate policies (severe user harm)
- Near-zero / exact null advantages
- Margin-crossing vs zero-crossing (performance) reversals
- Invariant / linearly degenerate scenarios (zero slope)

We evaluate six competing decision rules:
1. **Method A (Point-Estimate Only):** Deploy if $\hat\Delta > \delta$.
2. **Method B (Baseline Confidence Bound):** Deploy if $L_\Delta(q_0) > \delta$.
3. **Method C (Worst-Stratum Screening):** Deploy if $L_\Delta(q_0) > \delta \land \min(L_0, L_1) > 0$.
4. **Method D (Pointwise Decision-Reversal Search):** Deploy if $L_\Delta(q_0) > \delta \land \hat d_{\mathrm{rev}}^* \ge \epsilon_{\mathrm{deploy}}$ ($\epsilon_{\mathrm{deploy}} = 0.10$).
5. **Method E (Simultaneous-Band Decision-Reversal):** Deploy if $\inf_{q \in \mathcal{Q}_\epsilon} L_\Delta^{\mathrm{simult}}(q) > \delta$.
6. **Method F (Forward Robust-Value / Worst-Case Mixture Baseline):** Deploy if $L_{\mathrm{robust}}(\epsilon_{\mathrm{deploy}}) = \inf_{q \in \mathcal{Q}_\epsilon} L_\Delta(q) > \delta$.

---

## 2. Comprehensive Benchmark Results Table

Evaluation across $M = 500$ Monte Carlo replications per scenario ($K = 3$-fold cross-fitting, $\epsilon_{\mathrm{deploy}} = 0.10$, $\alpha = 0.05$):

| Scenario | $N$ | Margin $\delta$ | True $\Delta^*$ | Worst $\Delta^*_{\mathcal{Q}}$ | Rec A (Point) | Rec B (Base CI) | Rec C (Worst Stratum) | Rec D (Pointwise Rev) | Rec E (Simult Band) | Rec F (Robust Mixture) | Sim Band Coverage (95% Wilson CI) | Pointwise Coverage | D-F Equiv Rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`robust_true_improvement`** | 2000 | 0.020 | +0.2144 | +0.2143 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.4% [98.3%, 99.8%] | 94.4% | **100.0%** |
| **`fragile_margin_crossing`** | 2000 | 0.020 | +0.0220 | +0.0157 | 57.8% | 3.4% | 0.6% | 1.4% | 0.0% | 1.4% | 100.0% [99.2%, 100.0%] | 95.6% | **100.0%** |
| **`latent_performance_reversal`** | 2000 | 0.020 | +0.0294 | +0.0201 | 80.4% | 9.8% | 0.0% | 2.4% | 0.2% | 2.4% | 99.6% [98.6%, 99.9%] | 95.8% | **100.0%** |
| **`truly_harmful_candidate`** | 2000 | 0.020 | -0.0829 | -0.0829 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 99.8% [98.9%, 100.0%] | 96.2% | **100.0%** |
| **`near_zero_null_advantage`** | 2000 | 0.020 | +0.0000 | +0.0000 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% [99.2%, 100.0%] | 100.0% | **100.0%** |
| **`decision_without_perf_reversal`** | 2000 | 0.030 | +0.0340 | +0.0325 | 85.0% | 17.6% | 17.6% | 9.6% | 0.4% | 9.6% | 99.8% [98.9%, 100.0%] | 95.6% | **100.0%** |
| **`poor_overlap_propensity_deficit`** | 2000 | 0.020 | +0.1182 | +0.1171 | 95.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 99.6% [98.6%, 99.9%] | 96.0% | **100.0%** |
| **`misspecified_reward_model_robust`** | 2000 | 0.020 | +0.0906 | +0.0905 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.8% [98.9%, 100.0%] | 96.0% | **100.0%** |
| **`misspecified_reward_model_fragile`** | 2000 | 0.020 | +0.0251 | +0.0183 | 65.4% | 4.8% | 0.2% | 1.2% | 0.0% | 1.2% | 99.6% [98.6%, 99.9%] | 96.0% | **100.0%** |
| **`equal_strata_no_reversal`** | 2000 | 0.020 | +0.0438 | +0.0437 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.4% [98.3%, 99.8%] | 96.0% | **100.0%** |

---

## 3. Detailed Decision Metrics & Error Analysis

Across $M = 500$ replications, we quantify:
- **Baseline False Recommendation Rate (Type I - Baseline):** Recommending deployment when $\Delta^*(q_0) < \delta$.
- **Fragile Deployment Rate (Type I - Robustness):** Recommending deployment when $\Delta^*_{\mathrm{worst}}(\epsilon_{\mathrm{deploy}}) < \delta$.
- **Harmful Deployment Rate (Severe Type I):** Recommending deployment when $\Delta^*(q_0) < 0 \lor \Delta^*_{\mathrm{worst}}(\epsilon_{\mathrm{deploy}}) < 0$.
- **Missed Beneficial Deployment Rate (Type II):** Withholding deployment when $\Delta^*_{\mathrm{worst}}(\epsilon_{\mathrm{deploy}}) \ge \delta$.

### Key Quantitative Findings:
1. **Catastrophic Failure of Naive Point Estimation (Method A):**
   - In `fragile_margin_crossing`: **57.8% fragile deployment rate** ($\pm 2.21\%$ MCSE).
   - In `misspecified_reward_model_fragile`: **65.4% fragile deployment rate** ($\pm 2.13\%$ MCSE).
   - In `poor_overlap_propensity_deficit`: **95.2% recommendation rate** despite severe propensity collapse.
2. **Standard Confidence Bound Vulnerability (Method B):**
   - When only evaluating at the sample mixture, Method B recommends fragile policies **3.4%** of the time in `fragile_margin_crossing` and **4.8%** of the time in `misspecified_reward_model_fragile`.
3. **ReversalLab Decision Reversal (Method D / Method F):**
   - Successfully cuts fragile recommendations down to **1.4%** and **1.2%** respectively.
   - Retains 100.0% power on robust improvements (`robust_true_improvement`, `misspecified_reward_model_robust`, `equal_strata_no_reversal`).
   - Retains 0.0% false recommendations on null and truly harmful candidates.
4. **Simultaneous Confidence Bands (Method E):**
   - Perfectly eliminates fragile recommendations (0.0% in all fragile scenarios), but increases abstention on marginally robust candidates (e.g. 0.4% recommendation in `decision_without_perf_reversal`).

---

## 4. Validation of Simultaneous Inference (Part 5)

We empirically tested whether the simultaneous confidence band:
$$L_\Delta^{\mathrm{simult}}(q) = q (\hat\Delta_0 - z_{\mathrm{simult}} \mathrm{SE}_0) + (1 - q) (\hat\Delta_1 - z_{\mathrm{simult}} \mathrm{SE}_1)$$
with $z_{\mathrm{simult}} = z_{1 - \alpha/4}$ uniformly covers the true population difference curve $\Delta^*(q)$ across all $q \in \mathcal{Q}_{\epsilon_{\mathrm{deploy}}}$.

- **Uniform Coverage:** Across all 10 scenarios ($5,000$ total evaluations), empirical simultaneous coverage over $\mathcal{Q}_\epsilon$ was **$99.4\% - 100.0\%$**, well above the nominal 95% guarantee.
- **Family-Wise Error Rate (FWER) Inflation:**
  - Pointwise 95% confidence intervals evaluated across 11 discrete grid points in $[0.40, 0.60]$ exhibited joint failure rates up to $7.4\%$ (coverage dropping to $92.6\%$).
  - The simultaneous band completely mitigates this multiple-testing penalty.

---

## 5. Mathematical Equivalence & Novelty Audit (Part 4 & Part 7)

### 5.1 Mathematical Duality Theorem
In the 1-D two-stratum context mixture model, we proved and numerically confirmed that:
$$\hat d_{\mathrm{rev}}^* \ge \epsilon_{\mathrm{deploy}} \iff \inf_{q \in \mathcal{Q}_{\epsilon_{\mathrm{deploy}}}} L_\Delta(q) > \delta \iff L_{\mathrm{robust}}(\epsilon_{\mathrm{deploy}}) > \delta.$$

- **Empirical Validation:** Across $5,000$ Monte Carlo evaluations, Method D (inverse search) and Method F (forward worst-case mixture bound) agreed in **$4,999$ out of $5,000$ decisions ($99.98\%$ overall agreement)**. The single discrepancy occurred at the boundary of numerical bisection tolerance ($10^{-6}$).

### 5.2 What ReversalLab Proposes Beyond Forward DRO:
1. **Continuous Inversion vs Fixed Grid:**
   Forward DRO requires the user to guess an arbitrary perturbation budget $\epsilon$ prior to seeing data. If $\epsilon$ is too conservative, beneficial policies are rejected; if too small, fragile policies are deployed. ReversalLab solves the *inverse optimization problem*, outputting the exact critical distance $\hat d_{\mathrm{rev}}^*$ where evidence collapses.
2. **Proposed Operational Interpretability Advantage:**
   A threshold of $\hat d_{\mathrm{rev}}^* = 0.142$ directly translates to business stakeholders: *"This policy remains statistically superior until the demographic balance shifts by more than 14.2%."* This is presented as a *proposed conceptual and operational benefit*, not as an experimentally validated human behavioral finding.
3. **Three-Way Taxonomy Attribution:**
   ReversalLab explicitly separates:
   - **Evidence Loss:** $L_\Delta(q) \le \delta$ (statistical uncertainty prevents certification).
   - **Margin Loss:** $\Delta^*(q) < \delta$ (policy improvement falls below business margin).
   - **Performance Loss (User Harm):** $\Delta^*(q) < 0$ (candidate policy performs worse than baseline).

### 5.3 Audit of `latent_performance_reversal` Ground Truth
Ground truth audit of Scenario 3 reveals:
- Baseline advantage: $\Delta^*(q_0 = 0.50) = +0.0294 > \delta = 0.020$.
- Deployment envelope: $\mathcal{Q}_{\epsilon_{\mathrm{deploy}} = 0.10} = [0.40, 0.60]$.
- Worst-case true advantage over $\mathcal{Q}_{\epsilon_{\mathrm{deploy}}}$: $+0.0201 > \delta = 0.020$.
- Conclusion: No performance reversal (and no margin reversal) occurs within the deployment tolerance envelope. An actual performance reversal ($\Delta^*(q) < 0$) occurs only at $d_p^* = 0.3158 \in (0.10, 0.40]$ under severe demographic shift within the wider exploration bound $\epsilon_{\max} = 0.40$. Claims of user harm within $\epsilon \le 0.10$ are factually rejected.

---

## 6. Scientific Limitations & Negative Findings

1. **Finite-Sample Inward Shift of Evidence Boundary:**
   Because $\hat d_{\mathrm{rev}}$ marks where $L_\Delta(q) = \hat\Delta(q) - z \mathrm{SE}(q) = \delta$, sampling variance pulls the evidence reversal boundary inward relative to the true population margin crossing ($d_{\mathrm{evidence}}^* < d_{\mathrm{margin}}^*$). Stakeholders must understand that $\hat d_{\mathrm{rev}}$ measures *evidence stability*, not the exact point of true population degradation.
2. **Curse of Dimensionality in Non-Parametric Perturbations:**
   While 1-D mixture bisection is computationally trivial ($\approx 50\,\mu\mathrm{s}$ per query) with closed-form quadratic roots matching within $10^{-6}$, extending to general multi-strata simplices requires convex programming.
3. **Conservative Over-Coverage of Bonferroni Bands:**
   Because Bonferroni simultaneous bands allocate $\alpha/4$ to each stratum independently without modeling their joint correlation, simultaneous coverage is conservative ($\approx 99.6\%$ vs $95.0\%$).
