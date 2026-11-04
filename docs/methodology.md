# Methodology

## Setting

Single-position contextual bandit with \(K\) discrete actions. Each logged round:

1. A context \(x \sim P(x)\) is observed.
2. The logging policy \(\pi_0\) draws an action \(a \sim \pi_0(\cdot \mid x)\).
3. A reward \(r \sim P(r \mid x, a)\) is observed for that action only.

The log is \(\mathcal{D} = \{(x_i, a_i, r_i, p_i)\}_{i=1}^N\) with \(p_i = \pi_0(a_i \mid x_i)\).
The target quantity is the value of an evaluation (candidate) policy \(\pi_e\):

\[
V(\pi_e) = \mathbb{E}_{x}\Big[\sum_a \pi_e(a \mid x)\, q(x, a)\Big], \qquad q(x,a) = \mathbb{E}[r \mid x, a].
\]

## Estimators (implemented)

Importance weight: \(w_i = \pi_e(a_i \mid x_i) / p_i\).

**IPS**: \(\hat V_{\text{IPS}} = \frac{1}{N}\sum_i w_i r_i\).
It is unbiased when (i) observations are i.i.d., (ii) the propensities are correct, and
(iii) support holds, meaning \(\pi_e(a \mid x) > 0 \Rightarrow \pi_0(a \mid x) > 0\) for all \(x, a\).
Its standard error is \(\widehat{\mathrm{sd}}(w_i r_i)/\sqrt{N}\) (ddof = 1).

**SNIPS**: \(\hat V_{\text{SN}} = \sum_i w_i r_i / \sum_i w_i\).
It is a ratio estimator: biased with bias \(O(1/N)\), consistent under the same conditions, and
invariant to rescaling of the weights. It is undefined when \(\sum_i w_i = 0\), and
the code raises `DegenerateWeightsError` instead of returning a value in that case.
Its standard error comes from the delta method, using the \(1/N\) second-moment form
(not the Bessel \(1/(N-1)\) form used by the IPS standard error):
\(\sqrt{\sum_i w_i^2 (r_i - \hat V_{\text{SN}})^2} \,/\, \sum_i w_i\).

**Direct Method (DM)**:
\[
\hat V_{\text{DM}} = \frac{1}{N} \sum_{i=1}^N \sum_{a=1}^K \pi_e(a \mid x_i) \, \hat{q}(x_i, a).
\]
DM plugs in an estimated reward function \(\hat{q}(x, a) \approx \mathbb{E}[r \mid x, a]\) and integrates
over the candidate policy's action distribution. It does not divide by logging propensities, so it has
zero propensity variance. However, if \(\hat{q}\) is misspecified, DM suffers from irreducible asymptotic bias.
Its standard error is computed as the sample standard error of the per-observation expectations:
\(\widehat{\mathrm{sd}}\Big(\sum_a \pi_e(a \mid x_i) \hat{q}(x_i, a)\Big) / \sqrt{N}\). This treats \(\hat{q}\) as fixed
and omits reward-model estimation uncertainty; nominal Wald coverage is uncalibrated and therefore omitted from benchmarks.

**Doubly Robust (DR)**:
\[
\hat V_{\text{DR}} = \frac{1}{N} \sum_{i=1}^N \left( \sum_{a=1}^K \pi_e(a \mid x_i) \, \hat{q}(x_i, a) + w_i \big(r_i - \hat{q}(x_i, a_i)\big) \right).
\]
DR augments the Direct Method prediction with an importance-weighted residual correction on the factual reward.
It satisfies double robustness: it is consistent if *either* the logging propensities \(\pi_0(a \mid x)\) are correct
and support holds, *or* the reward model \(\hat{q}(x, a)\) is correctly specified. When \(\hat{q}(x, a) = 0\), DR
reduces identically to IPS. When factual residuals \(r_i - \hat{q}(x_i, a_i) = 0\), DR reduces identically to DM.
Standard error: \(\widehat{\mathrm{sd}}(\psi_i) / \sqrt{N}\) where \(\psi_i = \sum_a \pi_e(a \mid x_i) \hat{q}(x_i, a) + w_i (r_i - \hat{q}(x_i, a_i))\).
Like DM, this Wald standard error omits uncertainty from fitting \(\hat{q}\).

**Confidence intervals**: Wald intervals \(\hat V \pm z_{1-\alpha/2}\,\widehat{\mathrm{se}}\).
They are asymptotically valid under i.i.d. sampling and finite \(\mathrm{Var}(w r)\). They are
*not* reliable when weights are heavy-tailed (see Experiment 001). For DM, naive Wald intervals
omit reward-model estimation variance and severely undercover. For DR, influence-score Wald intervals
derive from the empirical variance of the uncentered influence function and are valid under
appropriate Neyman orthogonality and cross-fitting rate conditions.

## Influence-Score-Based Uncertainty and Double Machine Learning (DML)

For Doubly Robust, each observation \(i\) yields an uncentered influence score:
\[
\psi_i = \sum_{a=1}^K \pi_e(a \mid x_i) \hat{q}(x_i, a) + w_i \big(r_i - \hat{q}(x_i, a_i)\big).
\]
The DR point estimate is the sample average: \(\hat V_{\text{DR}} = \frac{1}{N} \sum_{i=1}^N \psi_i\).
Under the asymptotic theory of semi-parametric inference and Double Machine Learning (Chernozhukov et al., 2018):
1. **Neyman Orthogonality**: The Gateaux derivative of the DR score with respect to the nuisance parameter \(q\) vanishes at the true \(q_0\) when the logging propensities \(\pi_0\) are known and overlap holds:
   \[
   \partial_q \mathbb{E}[\psi(x, a, r; q)]\Big|_{q = q_0} = 0.
   \]
   Consequently, first-order errors from estimating \(\hat{q}\) do not affect the asymptotic distribution of \(\sqrt{N}(\hat V_{\text{DR}} - V(\pi_e))\).
2. **DML Rate Requirements and Known Propensities**:
   In the general Double Machine Learning framework where both the outcome regression \(\hat{q}\) and propensity model \(\hat{e}\) are estimated nonparametrically, valid asymptotic inference requires the product rate condition \(\|\hat{q} - q_0\|_{L_2} \times \|\hat{e} - e_0\|_{L_2} = o_P(N^{-1/2})\) (commonly achieved when each nuisance converges at \(o_P(N^{-1/4})\)).
   In ReversalLab's current synthetic benchmark, logging propensities are **known exactly and recorded without estimation error** (\(\|\hat{e} - e_0\| = 0\)). Consequently, Neyman orthogonality holds exactly, and the influence Wald interval remains asymptotically valid even under reward-model misspecification, provided importance weights have finite variance (\(\mathbb{E}[w^2] < \infty\)) and support holds.
3. **Cross-Fitting**: Fitting \(\hat{q}\) out-of-fold breaks empirical process dependencies between \(\hat{q}\) and the evaluation observations \((x_i, a_i, r_i)\), obviating restrictive Donsker-class assumptions.
4. **Direct Method Additive Error Decomposition**:
   The Direct Method prediction \(\hat v_i = \sum_a \pi_e(a \mid x_i) \hat{q}(x_i, a)\) does *not* satisfy Neyman orthogonality. The error decomposes additively into:
   \[
   \hat V_{\text{DM}} - V(\pi_e) = \frac{1}{N} \sum_{i=1}^N \big(v_0(x_i) - V(\pi_e)\big) + \frac{1}{N} \sum_{i=1}^N \sum_a \pi_e(a \mid x_i) \big(\hat{q}(x_i, a) - q_0(x_i, a)\big).
   \]
   These components form an additive error decomposition rather than an orthogonal variance decomposition (their finite-sample covariance is not assumed to vanish unconditionally). The sample variance of \(\hat v_i\) captures only the first term (context sampling variability), completely ignoring the second term (reward-model estimation uncertainty of order \(O(1/\sqrt{N_{\text{train}}})\)). In Experiment 004, this omission leads naive DM Wald intervals to undercover drastically (covering only 28%–35% at nominal 95%).

## Nonparametric Bootstrap for Off-Policy Inference

To evaluate finite-sample uncertainty and provide inference without relying on asymptotic normal approximations:
1. **Resampling Unit**: For i.i.d. contextual bandits, the resampling unit is the full observation tuple \((x_i, a_i, r_i, p_i)\). Sampling with replacement preserves the joint distribution between contexts, actions, rewards, and propensities.
2. **Identity-Aware Cross-Fitting in Refit Bootstrap**:
   When resampled observations contain duplicate copies of original interactions (multiplicities \(M_i \ge 1\)), naive random partitioning would place duplicate copies into both training and validation folds. To prevent this leakage:
   - Unique original observation identities are partitioned across folds.
   - All copies of an original observation remain in the same evaluation fold, while all training copies of other observations retain their full multiplicities.
   - Out-of-fold models never observe any copy of the evaluation interactions during training.
3. **Fixed-Model vs. Refit Cross-Fitting Bootstrap**:
   - *Fixed-Model Bootstrap* (`refit=False`): Computes bootstrap resamples of the estimator using a pre-computed \(\hat{Q}\) matrix. For DM, this suffers from the same limitation as the naive Wald interval by treating \(\hat{q}\) as fixed.
   - *Refit Cross-Fitting Bootstrap* (`refit=True`): Each bootstrap draw re-partitions the resampled observations into \(K\) folds and re-runs the entire model-fitting and out-of-fold prediction pipeline with identity-aware fold assignment. This properly incorporates the sampling variability of \(\hat{q}\).
4. **Interval Types**:
   - *Percentile Bootstrap*: \([q^*_{\alpha/2}, \; q^*_{1-\alpha/2}]\), where \(q^*_p\) is the \(p\)-th empirical quantile of \(\{\hat V^{*b}\}_{b=1}^B\).
   - *Basic (Empirical) Bootstrap*: \([2\hat V - q^*_{1-\alpha/2}, \; 2\hat V - q^*_{\alpha/2}]\).
   - *Normal Bootstrap*: \(\hat V \pm z_{1-\alpha/2}\,\widehat{\mathrm{se}}_{\text{boot}}\), where \(\widehat{\mathrm{se}}_{\text{boot}}\) is the sample standard deviation of bootstrap estimates.
5. **Heavy-Tail Behavior and Bootstrap Inconsistency**:
   Under poor overlap, importance weights \(w_i = \pi_e(a_i \mid x_i) / p_i\) exhibit heavy tails. As shown by Bickel & Freedman (1981), the classical nonparametric bootstrap sample mean is asymptotically inconsistent when the underlying distribution has an **infinite second moment** (\(\mathbb{E}[w^2] = \infty\)). In our synthetic contextual bandit benchmark, propensities are bounded away from zero on compact regions, so theoretical moments are finite. However, extreme finite-sample kurtosis causes severe quantile instability: extreme weights appear sporadically across resamples, causing bootstrap intervals to undercover under poor overlap (e.g., 81%–85% for IPS and 83%–88% for DR in Experiment 004). Bootstrapping does not resolve poor-overlap undercoverage in finite samples.

## Reward modeling and K-fold cross-fitting

Estimation is strictly separated from model training:
1. `dm()` and `dr()` accept pre-computed predicted-reward matrices \(\hat{Q} \in \mathbb{R}^{N \times K}\).
2. To avoid in-sample overfitting bias and optimistic performance claims, \(\hat{Q}\) is generated via \(K\)-fold cross-fitting:
   - The log \(\mathcal{D} = \{1, \dots, N\}\) is partitioned into \(K\) disjoint folds \(F_1, \dots, F_K\) using a deterministic seed.
   - For each fold \(k\), a reward model \(\hat{q}^{(-k)}\) is trained exclusively on the remaining observations \(\mathcal{D} \setminus F_k\).
   - Preprocessing steps (e.g., context standardization) are fitted strictly on the training partition \(\mathcal{D} \setminus F_k\).
   - Out-of-fold predictions \(\hat{q}^{(-k)}(x_i, a)\) are evaluated for all actions \(a \in \{1, \dots, K\}\) for each observation \(i \in F_k\).
   - The resulting rows are re-assembled in original order into the full matrix \(\hat{Q}\).
3. If an action \(a\) has zero observations in a training fold, the stratified model falls back gracefully to the global training fold mean reward.

## Finite-sample properties and limitations of DR

While Doubly Robust policy evaluation guarantees asymptotic unbiasedness under correct logging propensities,
**it does not guarantee lower finite-sample variance than DM or SNIPS**:
- In the presence of poor overlap (rare logging propensities \(\pi_0(a_i \mid x_i) \ll \pi_e(a_i \mid x_i)\)),
  the importance weights \(w_i\) become heavy-tailed.
- In finite samples, the factual residual \(r_i - \hat{q}(x_i, a_i)\) is non-zero due to sampling noise, even when \(\mathbb{E}[r - \hat{q}] = 0\).
- Multiplying this residual by extreme weights \(w_i\) inflates the variance of \(\hat V_{\text{DR}}\), causing its RMSE to be substantially worse than DM (as demonstrated empirically in Experiment 003).

## Diagnostics (implemented)

| Quantity | Definition | Interpretation |
|---|---|---|
| ESS | \((\sum w)^2 / \sum w^2\) (Kish) | Rough number of "effective" observations; heuristic, not an exact variance identity for IPS |
| ESS/N | ESS divided by N | Below `min_ess_fraction` (default 0.1) → warning |
| Max weight share | \(\max w / \sum w\) | One observation dominating the estimate |
| Mean weight | \(\bar w\); \(\mathbb{E}_{\pi_0}[w] = 1 - \text{unsupported mass}\) | A significant deviation suggests support or propensity problems; also fires under heavy tails (false alarms, see limitations) |
| Support | full \(\pi_0(\cdot\mid x_i)\) required | Verifiable only when the full logging distribution is logged |
| Unlogged-action mass | mean \(\pi_e\) mass on actions with zero count in the log | Empirical coverage gap |

The default thresholds are conventional heuristics and have not been validated.
M4 will treat them as baselines to compare against, not as ground truth.

## Synthetic ground truth

Contexts are \(x \sim \mathcal{N}(0, I_D)\). Reward logits are \(z(x,a) = x^\top W_a + b_a\), with
\(W \sim \mathcal{N}(0, s^2/D)\) and \(b \sim \mathcal{N}(0, s_b^2)\). Binary rewards are
\(r \sim \text{Bernoulli}(\sigma(z))\); Gaussian rewards are \(r \sim \mathcal{N}(z, \sigma_\epsilon^2)\).

The reference value \(V(\pi)\) is computed **exactly over actions** and by Monte Carlo
over an independent draw of \(M\) contexts (default \(10^6\)), with its Monte Carlo
standard error reported. The reference value uses a separate random stream from the
evaluation data. Where a closed form exists (Gaussian rewards, uniform policy:
\(V = \bar b\)), the Monte Carlo reference is tested against it.

Policies used in experiments are defined relative to the reward logits (an informed
policy) or to independent random logits (an uninformed policy):

* softmax with inverse temperature \(\beta\): \(\beta = 0\) is uniform, \(\beta < 0\) is anti-informed;
* \(\epsilon\)-greedy; \(\epsilon = 0\) is deterministic.

This gives policies with known performance ordering and controllable overlap.
The generator is not tuned towards any outcome; scenario parameters are recorded in
the config and hashed into the run identity.

## Open Bandit sample (position-level value)

The target, the logging policy, the action set, and the position slice are
defined in `docs/open_bandit_estimand.md` before any estimate. The value is
the expected click of one row. It is a different functional from the expected
number of clicks on a three-item slate and from the probability of at least
one slate click. Experiment 002 compares IPS and SNIPS with OBP's IPW and
SNIPW on that position-level estimand.

## Decision-Reversal Validation and Scientific Framework (M4A / M4B-1)

The decision-reversal framework identifies the smallest scientifically admissible change in evaluation assumptions that reverses a deployment decision:
\[
d_{\mathrm{rev}}^* = \inf \left\{ \epsilon \in [0, \epsilon_{\max}] : \exists q \in \mathcal{Q}_\epsilon \text{ such that } D(q) \neq D_0 \right\}.
\]

### 1. Closed-Form Analytical Reference
For two independent strata under total variation perturbation \(d_{\mathrm{TV}}(q, q_0) = |q - q_0|\), the decision boundary crossing condition satisfies:
\[
[\hat \Delta(q) - \delta]^2 = z^2 \mathrm{SE}^2(q).
\]
Expanding both sides yields the quadratic equation \(c_2 q^2 + c_1 q + c_0 = 0\) with:
\[
c_2 = (\hat \Delta_0 - \hat \Delta_1)^2 - z^2 (\widehat{\mathrm{se}}_0^2 + \widehat{\mathrm{se}}_1^2), \quad
c_1 = 2(\hat \Delta_0 - \hat \Delta_1)(\hat \Delta_1 - \delta) + 2 z^2 \widehat{\mathrm{se}}_1^2, \quad
c_0 = (\hat \Delta_1 - \delta)^2 - z^2 \widehat{\mathrm{se}}_1^2.
\]
The roots are solved in closed form, filtered by unsquared sign conditions (\(L_\Delta(q) = \delta\) or \(U_\Delta(q) = \delta\)), and verified against numerical bisection. Experiment 006 proves numerical and analytical agreement within \(9.77 \times 10^{-7}\).

### 2. Simultaneous Confidence Bands
To address the post-search inference challenge over the continuum \(\mathcal{Q}_\epsilon\), Bonferroni error allocation over the two strata provides uniform validity:
\[
z_{\mathrm{simult}} = z_{1 - \alpha/4}, \qquad
I_k^{\mathrm{simult}} = [\hat \Delta_k \mp z_{\mathrm{simult}} \widehat{\mathrm{se}}_k].
\]
Because \(q \ge 0\) and \(1-q \ge 0\), the bounds propagate linearly:
\[
L_\Delta^{\mathrm{simult}}(q) = q L_0^{\mathrm{simult}} + (1 - q) L_1^{\mathrm{simult}}.
\]
The infimum over \([q_{\min}, q_{\max}]\) is exactly attained at one of the endpoints, avoiding iterative search while providing a conservative uniform guarantee.

### 3. Reversal Taxonomy
Reversals are classified into three distinct categories:
- **Evidence Reversal:** The sample confidence bound crosses \(\delta\) (loss of deployment certitude).
- **Margin Reversal:** The true population difference \(\Delta^*(q)\) crosses \(\delta\) (policy falls below the required business margin).
- **Performance Reversal:** The true population difference \(\Delta^*(q)\) crosses \(0\) (candidate actively inflicts net harm compared to baseline).

### 4. Multi-Factor Diagnostic Gate
- **Active vs Inactive Propensity Deficit:** When \(\min_i p_i < \tau\) occurs on an action with \(\pi_e(a \mid x) = 0\), the deficit is inactive; it creates zero variance inflation and does not trip the gate. When \(\pi_e(a \mid x) > 0\), it trips the gate.
- **ESS Collapse & Weight Concentration:** Flags effective sample size collapse (\(\mathrm{ESS}/N < 0.05\)) and single-observation weight share concentration (\(\max_i w_i / \sum w_i > 0.20\)).
- **Unverified Support vs Proven Violation:** Distinguishes unverified support (when full logging action distribution is omitted) from proven mathematical support violations.

## Related work

* Horvitz & Thompson (1952): inverse probability weighting.
* Dudík, Langford & Li (2011): doubly robust policy evaluation.
* Dudík, Erhan, Langford & Li (2014): doubly robust policy evaluation and optimization.
* Swaminathan & Joachims (2015): self-normalised estimator for counterfactual learning.
* Su, Dimakopoulou, Krishnamurthy & Dudík (2020): doubly robust off-policy evaluation with shrinkage.
* Saito et al. (2021): Open Bandit Dataset and Open Bandit Pipeline (`zr-obp`).
* Si et al. (2020), Kallus et al. (2022): Distributionally robust off-policy evaluation (forward DRO).
* Rosenbaum (2002): Observational sensitivity analysis for treatment effects.
* Detailed synthesis and comparison against forward DRO and Rosenbaum sensitivity analysis is in `docs/related_work.md`.
