# Scientific Usefulness Study Design: Decision-Reversal Sensitivity Metrics

## 1. Executive Summary & Objective

Standard off-policy evaluation (OPE) provides a nominal point estimate and confidence interval for policy advantage $\Delta = V(\pi_e) - V(\pi_b)$. Under distribution shift, robust optimization methods typically report a **forward worst-case robust lower bound**:
$$\Delta_{\text{worst}}(\epsilon_{\text{deploy}}) = \min_{q \in \mathcal{Q}_\epsilon} \text{LCB}(q)$$

ReversalLab provides an alternative, inverse sensitivity formulation: the **distance-to-reversal** $d^* = |q^* - q_0|$ (the minimal population demographic shift required to overturn a positive deployment certification $\Delta \ge \delta$).

In 1-D linear demographic mixture models, both formulations yield mathematically identical binary decisions:
$$\min_{q \in \mathcal{Q}_\epsilon} \Delta(q) \ge \delta \iff \epsilon^* \ge \epsilon_{\text{deploy}}$$

### Objective
This empirical study evaluates whether presenting **Distance-to-Reversal Explanations alongside Forward Worst-Case Bounds** improves human practitioners' comprehension of policy fragility, reduces erroneous deployment decisions, and accelerates triage compared to presenting forward worst-case bounds alone.

> [!IMPORTANT]
> **Scientific Integrity Guardrail**: ReversalLab does NOT assert an a priori superiority claim for distance-to-reversal reporting over forward robust bounds. Any claims of interpretability advantage or decision alignment must be empirically supported by observed experimental data collected under this protocol.

---

## 2. Experimental Hypotheses

- **Primary Hypothesis ($H_1$ — Decision Accuracy & Fragility Detection)**:
  Participants presented with both the Forward Robust Bound and Distance-to-Reversal metrics (Condition B) will identify fragile deployment scenarios (cases where nominal advantage is positive and statistically significant, but demographic tolerance is smaller than anticipated production drift: $\epsilon^* < \epsilon_{\text{deploy}}$) with significantly higher accuracy than participants receiving only the Forward Robust Bound (Condition A).

- **Secondary Hypothesis ($H_2$ — Cognitive Workload & Decision Latency)**:
  Participants in Condition B will complete evaluation triage with lower subjective cognitive workload (NASA-TLX index) and shorter median decision latencies than participants in Condition A.

- **Secondary Hypothesis ($H_3$ — Confidence-Accuracy Calibration)**:
  Participants in Condition B will exhibit superior calibration (lower Brier score) between subjective confidence and actual decision correctness, reducing overconfidence on borderline fragile policies.

---

## 3. Study Design & Methodology

### 3.1 Design
- **Structure**: Between-subjects randomized controlled trial (A/B testing).
- **Target Population**: $N = 40$–$60$ machine learning practitioners, data scientists, and ML engineering leads with practical experience in offline evaluation or A/B testing.
- **Randomization**: Stratified block randomization based on prior causal inference experience (Novice vs. Experienced).

### 3.2 Experimental Conditions

#### Condition A (Control: Forward Robust Bound Only)
Participants receive standard nominal OPE results plus the forward worst-case bound:
1. Point Estimate: $\hat{\Delta} = +0.035$ (SE: $0.006$)
2. 95% Nominal Wald CI: $[+0.023, +0.047]$
3. Operational Margin: $\delta = 0.020$ (Nominal decision: Statistically superior)
4. Forward Worst-Case Lower Bound: $\Delta_{\text{worst}}(\epsilon_{\text{deploy}} = 0.15) = +0.016$
5. Static note: "At demographic shift radius $\epsilon = 0.15$, the worst-case lower confidence bound is $+0.016$."

#### Condition B (Treatment: Forward Robust Bound + Distance-to-Reversal)
Participants receive the identical inputs as Condition A, augmented with ReversalLab's distance-to-reversal explanation and ASCII diagnostic breakdown:
1. Point Estimate: $\hat{\Delta} = +0.035$ (SE: $0.006$)
2. 95% Nominal Wald CI: $[+0.023, +0.047]$
3. Operational Margin: $\delta = 0.020$
4. Forward Worst-Case Lower Bound: $\Delta_{\text{worst}}(\epsilon = 0.15) = +0.016$
5. **Reversal Distance ($d^*$ / $\epsilon^*$)**: $\epsilon^* = 0.082$ (Reversal threshold: Population shift of $8.2\%$ across stratum $X_0 \ge 0.0$ reverses deployment superiority).
6. **Diagnostic Gating**:
   ```text
   Baseline Demographics (q0): 50.0% stratum S1
   Reversal Demographics (q*):  58.2% stratum S1 (Shift: +8.2%)
   Deployment Tolerance Ball:   +/- 15.0% [35.0%, 65.0%]
   FRAGILITY WARNING: Reversal occurs WITHIN operational tolerance ball (d* < epsilon_deploy).
   ```

---

## 4. Participant Tasks & Scenarios

Participants review **8 randomized evaluation scenarios** drawn from synthetic recommender logs:

| Scenario ID | Nominal Advantage ($\hat{\Delta}$) | Nominal 95% CI | Required Margin ($\delta$) | Ground Truth Robustness | Fragility Root Cause | Expected Decision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | $+0.248$ | $[+0.232, +0.265]$ | $+0.020$ | **Robust** ($\epsilon^* > 0.40$) | None (Homogeneous gain) | `DEPLOY` |
| **S2** | $+0.042$ | $[+0.028, +0.056]$ | $+0.020$ | **Robust** ($\epsilon^* = 0.22 > 0.10$) | None (Mild heterogeneity) | `DEPLOY` |
| **S3** | $+0.038$ | $[+0.024, +0.052]$ | $+0.020$ | **Fragile** ($\epsilon^* = 0.04 < 0.10$) | Severe stratum divergence | `DO NOT DEPLOY` |
| **S4** | $+0.029$ | $[+0.021, +0.037]$ | $+0.020$ | **Fragile** ($\epsilon^* = 0.02 < 0.10$) | Subpopulation harm | `DO NOT DEPLOY` |
| **S5** | $+0.015$ | $[+0.002, +0.028]$ | $+0.020$ | **Non-Superior** | Fails nominal margin | `DO NOT DEPLOY` |
| **S6** | $+0.120$ | $[+0.080, +0.160]$ | $+0.020$ | **Inconclusive** | Overlap gate failure ($\min p_i < 10^{-4}$) | `AUDIT / COLLECT DATA` |
| **S7** | $+0.095$ | $[+0.045, +0.145]$ | $+0.020$ | **Inconclusive** | Kish ESS failure ($\text{ESS}/N = 2.1\%$) | `AUDIT / COLLECT DATA` |
| **S8** | $+0.055$ | $[+0.031, +0.079]$ | $+0.020$ | **Fragile** ($\epsilon^* = 0.06 < 0.15$) | Context mixture boundary | `DO NOT DEPLOY` |

For each scenario, participants answer three standardized questions:
1. **Action Choice**: "Which operational decision do you recommend?"
   - `[A]` Deploy Candidate Policy $\pi_e$ to production.
   - `[B]` Do Not Deploy (Keep Baseline Policy $\pi_b$).
   - `[C]` Reject Deployment / Collect More Data (Reliability Audit Required).
2. **Drift Estimate**: "What is the maximum shift in target subpopulation demographics this policy can tolerate before its advantage drops below $\delta$?" (Numerical slider: $0\%$ to $50\%$).
3. **Subjective Confidence**: "How confident are you in this recommendation?" (Likert scale: 1 = Completely guessing, 7 = Absolutely confident).

---

## 5. Metrics & Statistical Analysis Plan

### 5.1 Primary Metric
- **Deployment Decision Accuracy ($A_{\text{deploy}}$)**:
  Percentage of scenarios where the participant's action matches the ground-truth optimal action.
  - *Analysis*: Two-tailed independent samples $t$-test (or Mann-Whitney $U$ test if non-normal) comparing $A_{\text{deploy}}^{(B)}$ vs. $A_{\text{deploy}}^{(A)}$.
  - *Significance threshold*: $\alpha = 0.05$, with Cohen's $d$ effect size reporting.

### 5.2 Secondary Metrics
- **Fragility Detection Recall ($R_{\text{fragile}}$)**:
  Percentage of fragile scenarios (S3, S4, S8) where deployment was correctly rejected.
- **Drift Estimation Mean Absolute Error ($\text{MAE}_{\text{drift}}$)**:
  $| \text{Estimated Drift Tolerance} - \epsilon^* |$.
- **Decision Latency ($T_{\text{decision}}$)**:
  Median time spent per scenario in seconds.
- **Cognitive Workload Index ($W_{\text{TLX}}$)**:
  Post-experiment Raw NASA-TLX score (Mental Demand, Temporal Demand, Performance, Effort, Frustration).
- **Confidence-Accuracy Calibration ($B_{\text{calibration}}$)**:
  Brier score measuring mean squared error between subjective confidence probabilities and binary decision correctness.

---

## 6. Threats to Validity & Confounder Controls

1. **Prior Statistical Literacy Confounder**:
   - *Threat*: Participants with advanced econometrics/biostatistics training may interpret forward bounds faster regardless of condition.
   - *Control*: Administer pre-study screening questionnaire assessing familiarity with OPE, importance sampling, and Wald intervals. Stratify participants across conditions by experience level.
2. **Anchoring & Order Effects**:
   - *Threat*: Presenting strong positive scenarios first may bias subsequent fragility detection.
   - *Control*: Counterbalance scenario presentation order across participants using Latin square rotation.
3. **Information Equivalence Control**:
   - *Threat*: Providing more information to Condition B could confound the effect of metric type with mere information volume.
   - *Control*: Condition A contains the exact mathematical implications in prose (e.g. stating the worst-case bound value at the tested envelope). The difference is strictly the structural presentation of distance-to-failure vs. worst-case bound value.
4. **Hawthorne / Experimenter Bias**:
   - *Threat*: Participants guessing the study hypothesis and favoring ReversalLab's novel metric.
   - *Control*: Double-blind study platform interface; participants are informed they are evaluating "offline ML policy comparison dashboard designs", without naming ReversalLab or indicating the preferred condition.

---

## 7. Pilot Execution & Reporting Checklist

- [ ] Implement browser-based survey instrument using Qualtrics or self-hosted form.
- [ ] Conduct pilot run with $N = 4$ independent engineers to calibrate scenario clarity and time limits.
- [ ] Pre-register study protocol and analysis script on Open Science Framework (OSF) before formal data collection.
- [ ] Execute formal trial ($N \ge 40$).
- [ ] Report descriptive statistics, effect sizes, and confidence intervals without selective reporting.
