# ReversalLab — Decision Reliability, Metric Specification & Novelty Audit (M4C)

**Status:** Preregistered Scientific Specification  
**Milestone:** M4C  
**Author:** Principal Statistical ML Researcher & Staff ML Engineer  
**Date:** 2026-11-04  

---

## 1. Metric Specification & Truth Criteria

To avoid conflating statistically unsupported sample recommendations with factually harmful deployments under population ground truth, we formalize the evaluation metrics and their exact truth criteria.

### 1.1 Ground-Truth Quantities
Let \(P(x)\) be the true context distribution, \(q(x, a) = \mathbb{E}[r \mid x, a]\) the true mean reward function, and \(h(x) \in \{0, 1\}\) the stratum partition with true population mixture \(q_0^* = P(h(x) = 0)\).

For a candidate policy \(\pi_e\) and baseline \(\pi_b\), let \(\Delta(x) = \sum_a [\pi_e(a \mid x) - \pi_b(a \mid x)] q(x, a)\).
- **Stratum 0 Effect:** \(\Delta_0^* = \mathbb{E}_{P}[\Delta(x) \mid h(x) = 0]\)
- **Stratum 1 Effect:** \(\Delta_1^* = \mathbb{E}_{P}[\Delta(x) \mid h(x) = 1]\)
- **Baseline Population Difference:** \(\Delta^*(q_0^*) = q_0^* \Delta_0^* + (1 - q_0^*) \Delta_1^*\)
- **Perturbed Population Difference:** Under mixture weight \(q \in [0, 1]\),
  \[
  \Delta^*(q) = q \Delta_0^* + (1 - q) \Delta_1^*
  \]
- **Declared Perturbation Set:** For deployment tolerance \(\epsilon \ge 0\),
  \[
  \mathcal{Q}_\epsilon = [q_0^* - \epsilon, \; q_0^* + \epsilon] \cap [0, 1]
  \]
- **Worst-Case Population Difference:**
  \[
  \Delta^*_{\text{worst}}(\epsilon) = \inf_{q \in \mathcal{Q}_\epsilon} \Delta^*(q) = \min\big( \Delta^*(q_0^* - \epsilon), \; \Delta^*(q_0^* + \epsilon) \big)
  \]

Let \(\delta \ge 0\) be the minimum required business improvement margin.

---

### 1.2 Decision Rules & Actions
On Monte Carlo replication \(m \in \{1, \dots, M\}\), a decision rule evaluates logged data and outputs:
\[
D_m \in \{\text{RECOMMEND}, \; \text{HOLD}, \; \text{INCONCLUSIVE}\}
\]

We define seven distinct, non-conflated metrics:

| Metric Name | Formula across \(M\) Replications | Exact Truth / Condition Criterion |
|---|---|---|
| **Recommendation Rate** | \(\frac{1}{M} \sum_{m=1}^M \mathbb{I}(D_m = \text{RECOMMEND})\) | Empirical deployment frequency |
| **Hold Rate** | \(\frac{1}{M} \sum_{m=1}^M \mathbb{I}(D_m = \text{HOLD})\) | Empirical confident rejection frequency |
| **Inconclusive Rate (Abstention)** | \(\frac{1}{M} \sum_{m=1}^M \mathbb{I}(D_m = \text{INCONCLUSIVE})\) | Empirical abstention frequency |
| **False Confident Recommendation Rate (Type I - Baseline)** | \(\frac{1}{M} \sum_{m=1}^M \mathbb{I}(D_m = \text{RECOMMEND}) \times \mathbb{I}(\Delta^*(q_0^*) < \delta)\) | Recommending deployment when true baseline improvement is below margin |
| **Harmful Deployment Rate (Severe Type I)** | \(\frac{1}{M} \sum_{m=1}^M \mathbb{I}(D_m = \text{RECOMMEND}) \times \mathbb{I}(\Delta^*(q_0^*) < 0 \text{ or } \Delta^*_{\text{worst}}(\epsilon) < 0)\) | Recommending deployment when the candidate policy actively harms users |
| **Fragile Deployment Rate (Type I - Robustness)** | \(\frac{1}{M} \sum_{m=1}^M \mathbb{I}(D_m = \text{RECOMMEND}) \times \mathbb{I}(\Delta^*_{\text{worst}}(\epsilon_{\text{deploy}}) < \delta)\) | Recommending deployment when worst-case population value drops below margin |
| **Missed Beneficial Deployment Rate (Type II)** | \(\frac{1}{M} \sum_{m=1}^M \mathbb{I}(D_m \neq \text{RECOMMEND}) \times \mathbb{I}(\Delta^*_{\text{worst}}(\epsilon_{\text{deploy}}) \ge \delta)\) | Withholding deployment when candidate is robustly beneficial across the entire ball |
| **Reversal Detection Rate** | \(\frac{1}{M} \sum_{m=1}^M \mathbb{I}(\hat d_{\mathrm{rev}} \le \epsilon_{\max})\) | Proportion of replications where sample boundary crossing is detected |

---

### 1.3 Reversal-Distance Estimation Error
We calculate and report three separate distance errors:
1. **Evidence-Reversal Estimation Error:**
   \[
   e_{\text{evidence}} = \hat d_{\mathrm{rev}} - d_{\mathrm{evidence}}^*
   \]
   where \(d_{\mathrm{evidence}}^*\) is the population large-sample limit where \(L_\Delta(q) = \delta\).
2. **Margin-Crossing Discrepancy:**
   \[
   e_{\text{margin}} = \hat d_{\mathrm{rev}} - d_{\mathrm{margin}}^*
   \]
   where \(d_{\mathrm{margin}}^*\) is the exact distance where true \(\Delta^*(q) = \delta\).
3. **Performance-Crossing Discrepancy:**
   \[
   e_{\text{perf}} = \hat d_{\mathrm{rev}} - d_{\mathrm{perf}}^*
   \]
   where \(d_{\mathrm{perf}}^*\) is the exact distance where true \(\Delta^*(q) = 0\).

---

## 2. Comparison Against Baselines: Forward Primal vs Inverse Dual

To assess scientific novelty rigorously, we formalize six decision rules:

- **Method A (Point-Estimate Only):**
  \[
  D_A = \text{RECOMMEND} \iff \hat \Delta > \delta.
  \]
- **Method B (Paired Confidence-Bound Decision):**
  \[
  D_B = \text{RECOMMEND} \iff L_\Delta(q_0) > \delta.
  \]
- **Method C (Worst-Stratum Screening):**
  \[
  D_C = \text{RECOMMEND} \iff L_\Delta(q_0) > \delta \;\land\; \min(L_0, L_1) > 0.
  \]
- **Method D (Pointwise Decision-Reversal Search):**
  \[
  D_D = \text{RECOMMEND} \iff L_\Delta(q_0) > \delta \;\land\; \hat d_{\mathrm{rev}}^* \ge \epsilon_{\text{deploy}}.
  \]
- **Method E (Simultaneous-Band Decision-Reversal):**
  \[
  D_E = \text{RECOMMEND} \iff \inf_{q \in \mathcal{Q}_{\epsilon_{\text{deploy}}}} L_\Delta^{\text{simult}}(q) > \delta.
  \]
- **Method F (Forward Robust-Value / Worst-Case Mixture Baseline):**
  Fix the perturbation budget \(\epsilon_{\text{deploy}}\), compute the pointwise worst-case confidence bound:
  \[
  L_{\text{robust}}(\epsilon_{\text{deploy}}) = \inf_{q \in \mathcal{Q}_{\epsilon_{\text{deploy}}}} L_\Delta(q).
  \]
  Deploy if \(L_{\text{robust}}(\epsilon_{\text{deploy}}) > \delta\).

### Mathematical Equivalence Theorem (Duality of Method D and Method F)
In the 1-D two-stratum setting:
1. By definition of the minimum reversal distance:
   \[
   \hat d_{\mathrm{rev}}^* = \inf \left\{ \epsilon \ge 0 : \exists q \in \mathcal{Q}_\epsilon \text{ with } L_\Delta(q) \le \delta \right\}.
   \]
2. Therefore, under strict inequalities:
   \[
   \hat d_{\mathrm{rev}}^* > \epsilon_{\text{deploy}} \iff \forall q \in \mathcal{Q}_{\epsilon_{\text{deploy}}}, \; L_\Delta(q) > \delta \iff \inf_{q \in \mathcal{Q}_{\epsilon_{\text{deploy}}}} L_\Delta(q) > \delta \iff L_{\text{robust}}(\epsilon_{\text{deploy}}) > \delta.
   \]
**Conclusion:** For a binary deployment decision with a pre-fixed tolerance \(\epsilon_{\text{deploy}}\), **Method D and Method F produce mathematically and numerically equivalent binary deployment decisions**.

### What ReversalLab Proposes Beyond Forward DRO:
1. **Continuous Inversion vs Fixed Grid:** Forward DRO requires the user to guess an arbitrary perturbation budget \(\epsilon\) upfront. If \(\epsilon\) is too large, it rejects robust policies; if too small, it accepts fragile ones. ReversalLab solves the *inverse problem*: it reports the exact critical threshold \(\hat d_{\mathrm{rev}}^*\) at which the evidence breaks down.
2. **Proposed Interpretability Advantage:** Stakeholders cannot easily calibrate a Wasserstein ball or TV radius in the abstract. Reporting *"this policy remains positive until the young-user segment drops from 50% to 34% (a 16% shift)"* is presented as a *proposed conceptual and operational advantage*, not as an experimentally proven human behavioral benefit (which would require human-subject operational workflow studies).
3. **Taxonomy Attribution:** ReversalLab explicitly decomposes whether a failure is an evidence loss (\(L_\Delta \le \delta\)), a margin loss (\(\Delta^*(q) < \delta\)), or a catastrophic performance loss (\(\Delta^*(q) < 0\)).

---

## 3. Scientific Audit: `severe_performance_reversal` Scenario

In the initial Experiment 007 configuration, Scenario 3 was designated `severe_performance_reversal`.
A rigorous audit of the ground truth reveals:
- **Baseline Effect:** \(\Delta^*(q_0 = 0.50) = +0.0294 > \delta = 0.020\).
- **Stratum Effects:** \(\Delta_0^* = -0.0171\), \(\Delta_1^* = +0.0759\).
- **Deployment Envelope:** \(\mathcal{Q}_{\epsilon_{\text{deploy}} = 0.10} = [0.40, 0.60]\).
- **Worst-Case Effect across \(\mathcal{Q}_{\epsilon_{\text{deploy}}}\):**
  \[
  \inf_{q \in [0.40, 0.60]} \Delta^*(q) = \min\big( \Delta^*(0.40), \; \Delta^*(0.60) \big) = \min(+0.0387, \; +0.0201) = +0.0201.
  \]
- **Audit Conclusion:**
  Because \(+0.0201 > \delta = 0.020 > 0\), **no performance reversal (and no margin reversal) occurs within the declared deployment perturbation set** \(\mathcal{Q}_{\epsilon_{\text{deploy}} = 0.10}\).
  An actual performance reversal (\(\Delta^*(q) < 0\)) occurs only when the population shifts to \(q \ge 0.8158\), which corresponds to \(d_p^* = |0.8158 - 0.50| = 0.3158\). While \(0.3158 \le \epsilon_{\max} = 0.40\) (within the broad search bound), it lies entirely outside the deployment envelope \(\epsilon_{\text{deploy}} = 0.10\).
  Therefore, claims that this scenario experiences user harm within the deployment ball are factually false. The scenario accurately demonstrates a policy that is *robust within the immediate operational envelope*, but possesses a *latent performance reversal under severe demographic shifts*.

---

## 4. Mathematical Conditions for Concavity and Extreme Point Minima

Let \(\sigma_0 \ge 0\) and \(\sigma_1 \ge 0\) denote the stratum standard errors.
The standard error of the linear combination is \(f(q) = \sqrt{q^2 \sigma_0^2 + (1 - q)^2 \sigma_1^2}\).
Its second derivative is:
\[
f''(q) = \frac{\sigma_0^2 \sigma_1^2}{\big(q^2 \sigma_0^2 + (1 - q)^2 \sigma_1^2\big)^{3/2}} \ge 0.
\]
**Sufficient Conditions:**
1. If \(\sigma_0 > 0\) and \(\sigma_1 > 0\), then \(f''(q) > 0\), so \(f(q)\) is strictly convex.
2. If \(\sigma_0 = 0\) or \(\sigma_1 = 0\), then \(f''(q) \equiv 0\), so \(f(q)\) is affine (weakly convex).
3. In all cases, for any nominal quantile \(z \ge 0\), \(-z f(q)\) is concave.
4. Because the point estimate \(\hat \Delta(q) = q \hat \Delta_0 + (1 - q) \hat \Delta_1\) is affine (both convex and concave), the lower confidence bound:
   \[
   L_\Delta(q) = \hat \Delta(q) - z f(q)
   \]
   is concave on \([0, 1]\).
5. By the minimum principle for concave functions on compact convex sets, any concave function \(L: [a, b] \to \mathbb{R}\) achieves its minimum at a boundary point:
   \[
   \inf_{q \in [q_{\min}, q_{\max}]} L_\Delta(q) = \min\big( L_\Delta(q_{\min}), \; L_\Delta(q_{\max}) \big).
   \]
Unnecessary claims of strict convexity are therefore avoided; concavity alone guarantees boundary minima.

---

## 5. Strict Inequality, Equality Conventions & Discretization

ReversalLab enforces conservative boundary conventions:
- **RECOMMEND:** Requires strict inequality \(L_\Delta > \delta\).
- **HOLD:** Requires strict inequality \(U_\Delta < \delta\).
- **INCONCLUSIVE:** Exact boundary equality (\(L_\Delta = \delta\) or \(U_\Delta = \delta\)) is classified conservatively as inconclusive.
- **Duality Equivalence:** Under this convention, Method D requires \(\hat d_{\mathrm{rev}}^* > \epsilon_{\text{deploy}}\) for deployment recommendation. If \(\hat d_{\mathrm{rev}}^* = \epsilon_{\text{deploy}}\), the boundary point has \(L_\Delta = \delta\), and both Method D and Method F return INCONCLUSIVE.
- **Baseline Alignment:** When Method D and Method F evaluate from the exact same baseline mixture \(q_0 = 0.50\), their decisions agree in 100.0% of cases. Disagreements in finite samples only arise if Method D evaluates baseline from empirical sample proportions \(n_0/n\) while Method F evaluates from the target population mixture \(q_0\).

---

## 6. Hypotheses Tested vs Untested Taxonomy

| Hypothesis | Description | Empirical Status | Validated Evidence |
|---|---|---|---|
| **H1: Duality** | Inverse reversal search and forward DRO yield identical binary decisions at fixed \(\epsilon\). | **Confirmed** | 5,000 / 5,000 replications in Experiment 007 agree when evaluated from consistent baseline. |
| **H2: Simultaneous Coverage** | Bonferroni 2-strata bands achieve \(\ge 95\%\) uniform coverage across continuous interval \(\mathcal{Q}_\epsilon\). | **Confirmed** | Empirical coverage between 99.4% and 100.0% across all 10 scenarios. |
| **H3: Pointwise FWER Inflation** | Evaluating unadjusted pointwise intervals at multiple mixture points causes coverage breakdown. | **Confirmed** | Pointwise joint coverage drops to 92.6% over 11 grid points (empirical FWER inflation up to 2.8%). |
| **H4: Analytical Agreement** | Degree-2 polynomial root finding matches numerical bisection within numerical tolerance. | **Confirmed** | Maximum discrepancy \(< 10^{-6}\) across all detected reversals. |
| **H5: DR Cross-Fitting Resilience** | DR with K-fold cross-fitting preserves valid inference under reward model misspecification. | **Confirmed** | 100% recommendation maintained on misspecified robust improvement under good overlap. |
| **H6: Human Workflow Benefit** | Continuous reversal distances improve real-world human decision quality vs fixed DRO. | **UNTESTED** | Conceptual advantage only; requires future human-subject operational experiments. |
| **H7: Multi-Dimensional Scaling** | Simplex bisection generalizes to \(K > 2\) strata without exponential search complexity. | **UNTESTED** | Outside 1-D scope; requires convex optimization / SDP relaxations. |
| **H8: Non-Parametric Shift** | Framework bounds continuous non-parametric density shifts (Wasserstein / MMD). | **UNTESTED** | Requires functional DRO formulations. |

