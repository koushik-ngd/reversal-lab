# Scientific Usefulness Study Protocol: Decision-Reversal Sensitivity Metrics

## 1. Executive Summary & Objective

Standard off-policy evaluation (OPE) provides a nominal point estimate and confidence interval for policy advantage $\Delta = V(\pi_e) - V(\pi_b)$. Under distribution shift, robust optimization methods typically report a **forward worst-case robust lower bound**:
$$\Delta_{\text{worst}}(\epsilon_{\text{deploy}}) = \min_{q \in \mathcal{Q}_\epsilon} \text{LCB}(q)$$

ReversalLab provides an alternative, inverse sensitivity formulation: the **distance-to-reversal** $d^* = |q^* - q_0|$ (the minimal demographic shift required to overturn a positive deployment certification $\Delta \ge \delta$).

In 1-D linear demographic mixture models, both formulations yield mathematically identical binary decisions:
$$\min_{q \in \mathcal{Q}_\epsilon} \Delta(q) \ge \delta \iff \epsilon^* \ge \epsilon_{\text{deploy}}$$

### Objective
This empirical study evaluates the **incremental value of presenting Distance-to-Reversal Explanations alongside Forward Worst-Case Bounds** compared to presenting forward worst-case bounds alone. Specifically, it tests whether augmenting standard robust bounds with distance-to-reversal metrics improves human practitioners' comprehension of offline evaluation evidence, reduces erroneous deployment recommendations, and accelerates triage.

> [!IMPORTANT]
> **Status: Pre-Registered Experimental Protocol**.  
> This document specifies an unexecuted experimental protocol. ReversalLab does NOT assert an a priori claim of statistical superiority or human decision advantage for distance-to-reversal reporting over forward robust bounds. Any claims of interpretability advantage or decision alignment must be empirically supported by observed experimental data collected under this protocol.

---

## 2. Theoretical Scope & Key Distinctions

### Evidence Comprehension vs. Post-Deployment Outcomes
A critical conceptual distinction must be maintained throughout this evaluation:
- **Offline Evidence Comprehension (Evaluated in this Study)**: Does the practitioner correctly interpret what the logged data, propensity diagnostics, and distribution-shift models imply about the policy's robustness envelope?
- **Actual Real-World Deployment Outcome (External to this Study)**: Real-world operational outcomes depend on unobserved real-world factors (unmeasured confounders, non-covariate concept drift, unforeseen production bugs) that cannot be guaranteed by any offline sensitivity metric alone. This study strictly measures **evidence comprehension and decision triage accuracy**, not real-world business outcomes.

---

## 3. Experimental Hypotheses

- **Primary Hypothesis ($H_1$ — Incremental Triage Accuracy)**:
  Providing the **incremental Distance-to-Reversal explanation** alongside the Forward Robust Bound (Condition B) will significantly improve practitioner decision triage accuracy compared to providing the Forward Robust Bound alone (Condition A), specifically in correctly identifying fragile deployment scenarios (cases where nominal advantage is positive and statistically significant, but demographic tolerance is smaller than anticipated production drift: $\epsilon^* < \epsilon_{\text{deploy}}$).

- **Secondary Hypothesis ($H_2$ — Cognitive Workload & Decision Latency)**:
  Participants in Condition B will complete evaluation triage with lower subjective cognitive workload (NASA-TLX index) and shorter median human decision response latencies (measured in seconds from scenario presentation to decision submission) than participants in Condition A.

- **Secondary Hypothesis ($H_3$ — Confidence-Accuracy Calibration)**:
  Participants in Condition B will exhibit superior calibration (lower Brier score) between subjective confidence and actual decision correctness, reducing overconfidence on borderline fragile policies.

---

## 4. Study Design & Methodology

### 4.1 Design Structure & Treatment Definition
- **Design**: Between-subjects randomized controlled trial (A/B testing) with repeated measures across 8 standardized scenarios.
- **Arms Definition (Incremental Explanation Contrast)**:
  - **Control Arm (Condition A — Forward Robust Bound)**: Practitioners receive standard off-policy point estimates, 95% Wald confidence intervals, diagnostic overlap/ESS metrics, and the **Forward Worst-Case Robust Bound** $\Delta_{\text{worst}}(\epsilon_{\text{deploy}})$.
  - **Treatment Arm (Condition B — Incremental Distance-to-Reversal Explanation)**: Practitioners receive identical standard OPE estimates, diagnostic metrics, and the Forward Worst-Case Robust Bound, plus the **incremental Distance-to-Reversal explanation** ($d^* = |q^* - q_0|$, demographic inflection boundary $q^*$, and margin sensitivity interpretation).
- **Target Population**: $N = 40$–$60$ machine learning practitioners, data scientists, and ML engineering leads with practical experience in offline evaluation or A/B testing.
- **Randomization Procedure**: Permuted block randomization within prior experience strata (Novice vs. Experienced in causal inference/OPE) using a cryptographically secure random sequence.

### 4.2 Sample Size Target & Simulation-Based Power Analysis Requirements
- **Provisional Recruitment Target ($N = 50$)**:
  A total enrollment of $N = 50$ participants ($n = 25$ per arm, generating $50 \times 8 = 400$ scenario evaluations) is established as a **provisional recruitment target**, not an analytically proven adequate sample size.
- **Why Closed-Form Power Formulas Are Inadequate**:
  Standard closed-form power calculations (such as Cohen's $d = 0.50$ for paired or repeated $t$-tests) assume continuous, normally distributed responses. In this protocol, the primary outcome is a clustered binary triage decision ($Y_{ij} \in \{0, 1\}$) modeled via a Generalized Linear Mixed Model (GLMM) with a logit link function and participant random effects. In logistic mixed models, statistical power depends non-linearly on:
  1. Baseline scenario difficulty $p_j$ across the 8 evaluation tasks.
  2. Between-participant heterogeneity in skill and risk tolerance (participant random-intercept variance $\sigma_u^2$).
  3. The true odds ratio effect size $\text{OR} = \exp(\beta_1)$ of the incremental explanation.
- **Simulation-Based Power Protocol Requirements**:
  Prior to final participant enrollment lock, statistical power must be verified via Monte Carlo simulation:
  1. **Simulation Grid**:
     - Candidate participant cohorts: $N \in \{30, 40, 50, 60, 75, 100\}$.
     - Candidate treatment odds ratios: $\text{OR} \in [1.50, 2.50]$ (representing an incremental triage accuracy improvement from $\approx 65\%$ to $74\%\text{--}82\%$).
     - Participant random-intercept variance: $\sigma_u^2 \in [0.20, 0.80]$ (corresponding to intra-class correlations $\rho \approx \sigma_u^2 / (\sigma_u^2 + \pi^2/3) \in [0.06, 0.20]$).
     - Scenario baseline difficulties: $p_j \in [0.40, 0.85]$ across the 8 standardized scenarios.
  2. **Iteration Count**:
     - Generate $M \ge 1{,}000$ synthetic datasets for each parameter combination in the simulation grid.
  3. **Model Fitting & Power Evaluation**:
     - Fit the pre-registered logistic mixed-effects model:
       $$\text{logit}(P(Y_{ij} = 1)) = \beta_0 + \beta_1 \text{Condition}_i + \beta_2 \text{Scenario}_j + u_i, \quad u_i \sim \mathcal{N}(0, \sigma_u^2)$$
     - Empirical power is computed as the proportion of the $M$ simulated cohorts where the two-sided Wald test on $\beta_1$ achieves $p < 0.05$.
  4. **Pilot Calibration**:
     - Execute a preliminary internal pilot cohort ($N_{\text{pilot}} = 10$) to estimate empirical $\hat{\sigma}_u^2$ and verify scenario difficulty distributions. The final enrollment target $N$ will be adjusted if the pilot-calibrated simulation requires $N > 50$ to achieve $80\%$ power.

### 4.3 Information-Equivalence & Complexity Controls
To prevent confounding the presentation format with mere information volume or visual design differences:
1. **Information-Equivalence**: Both conditions receive identical mathematical information:
   - Same dataset context dimension ($D = 2$), sample size ($N = 2500$), and action count ($K = 3$).
   - Same point estimate $\hat{\Delta}$, standard error, and nominal 95% Wald CI.
   - Same operational margin $\delta = 0.020$ and deployment tolerance ball $\epsilon_{\text{deploy}} = 0.10$.
   - Same diagnostic gate results (overlap, ESS, weight concentration).
   - In Condition A, the worst-case bound value at $\epsilon_{\text{deploy}}$ is explicitly stated in words and numbers (e.g., "At drift radius $\epsilon = 0.10$, the worst-case lower confidence bound is $+0.016$, which falls below margin $\delta = 0.020$").
   - In Condition B, the distance-to-reversal is reported alongside the bound (e.g., "Distance to reversal $d^* = 0.042$; reversal occurs within the $\pm 0.10$ tolerance ball at $q^* = 0.542$").
2. **Interface Complexity Control**: Both conditions use identical monochrome typography, identical card layouts, and identical visual hierarchy. Total word count per scenario is equalized within $\pm 10$ words.
3. **Participant Experience**: Total experiment duration is calibrated to 15–20 minutes to prevent participant cognitive fatigue. A practice calibration scenario is provided during onboarding.

---

## 5. Evaluation Scenarios

Participants review **8 randomized evaluation scenarios** drawn from synthetic recommender logs:

| Scenario ID | Nominal Advantage ($\hat{\Delta}$) | Nominal 95% CI | Required Margin ($\delta$) | Ground Truth Robustness | Fragility Root Cause | Expected Decision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | $+0.248$ | $[+0.232, +0.265]$ | $+0.020$ | **Robust** ($\epsilon^* > 0.40$) | None (Homogeneous gain) | `DEPLOY` |
| **S2** | $+0.042$ | $[+0.028, +0.056]$ | $+0.020$ | **Robust** ($\epsilon^* = 0.22 > 0.10$) | None (Mild heterogeneity) | `DEPLOY` |
| **S3** | $+0.038$ | $[+0.024, +0.052]$ | $+0.020$ | **Fragile** ($\epsilon^* = 0.04 < 0.10$) | Severe stratum divergence | `DO NOT DEPLOY` |
| **S4** | $+0.029$ | $[+0.021, +0.037]$ | $+0.020$ | **Fragile** ($\epsilon^* = 0.02 < 0.10$) | Subpopulation harm | `DO NOT DEPLOY` |
| **S5** | $+0.015$ | $[+0.002, +0.028]$ | $+0.020$ | **Non-Superior** | Fails nominal margin | `DO NOT DEPLOY` |
| **S6** | $+0.120$ | $[+0.080, +0.160]$ | $+0.020$ | **Inconclusive** | Overlap gate failure ($\min p_i < 10^{-4}$) | `AUDIT / COLLECT DATA` |
| **S7** | $+0.095$ | $[+0.045, +0.145]$ | $+0.020$ | **Inconclusive** | Kish ESS failure ($\text{ESS}/N = 2.1\%$) | `AUDIT / COLLECT DATA` |
| **S8** | $+0.055$ | $[+0.031, +0.079]$ | $+0.020$ | **Fragile** ($\epsilon^* = 0.06 < 0.15$) | Context mixture boundary | `DO NOT DEPLOY` |

For each scenario, participants answer three standardized questions:
1. **Action Choice**: "Which operational decision do you recommend based on this evidence?"
   - `[A]` Deploy Candidate Policy $\pi_e$ to production.
   - `[B]` Do Not Deploy (Keep Baseline Policy $\pi_b$).
   - `[C]` Reject Deployment / Collect More Data (Reliability Audit Required).
2. **Drift Estimate**: "What is the maximum shift in target subpopulation demographics this policy can tolerate before its advantage drops below $\delta$?" (Numerical slider: $0\%$ to $50\%$).
3. **Subjective Confidence**: "How confident are you in this recommendation?" (Likert scale: 1 = Completely guessing, 7 = Absolutely confident).

---

## 6. Statistical Analysis Plan

### 6.1 Primary Outcome Analysis
- **Model**: Generalized Linear Mixed Model (GLMM) with a logit link function:
  $$\text{logit}(P(Y_{ij} = 1)) = \beta_0 + \beta_1 \text{Condition}_i + \beta_2 \text{Scenario}_j + \beta_3 \text{Experience}_i + u_i$$
  where $Y_{ij} \in \{0, 1\}$ indicates correct decision triage for participant $i$ on scenario $j$, $\text{Condition}_i \in \{0, 1\}$, and $u_i \sim \mathcal{N}(0, \sigma_u^2)$ is the participant random effect.
- **Significance Criterion**: Two-tailed Wald test on $\beta_1$ with $\alpha = 0.05$. Odds ratio ($\text{OR} = \exp(\beta_1)$) reported with 95% profile likelihood confidence intervals.

### 6.2 Secondary Outcomes Analysis
- **Fragility Recall**: Subgroup GLMM fitted strictly on fragile scenarios (S3, S4, S8).
- **Drift Estimation Error**: Linear Mixed Model (LMM) on absolute error $|\text{Slider} - \epsilon^*|$.
- **Decision Latency**: Log-transformed human participant response times (seconds per scenario) analyzed via LMM.
- **Cognitive Workload**: Two-sample $t$-test (or Wilcoxon rank-sum test) comparing NASA-TLX composite scores.
- **Calibration**: Brier score comparison between conditions:
  $$B = \frac{1}{N_{\text{total}}} \sum_{i, j} (p_{ij} - Y_{ij})^2$$

---

## 7. Confounder Controls & Protocol Safeguards

1. **Order Effects**: Latin square rotation counterbalances scenario presentation order across participants.
2. **Experimenter Expectancy (Hawthorne Effect)**: Double-blind delivery. Participants are recruited for an "Evaluation Dashboard Usability Study" without naming ReversalLab or disclosing the researcher's hypotheses.
3. **Pre-Registration**: The complete analysis script and synthetic dataset will be pre-registered on OSF prior to launching participant recruitment.
