# Limitations

The empirical coverage and error rates below come from the study scenarios and sample sizes stated for each experiment.

## Estimators and Uncertainty Quantification

* IPS, SNIPS, Direct Method (DM), and Doubly Robust (DR) are implemented.
* **Direct Method misspecification bias and uncalibrated naive Wald intervals:**
  - DM depends entirely on the accuracy of the reward model \(\hat{q}(x, a)\). If \(\hat{q}\) is misspecified, DM suffers from irreducible asymptotic bias that does not decay with sample size \(N\). In Experiment 004 under misspecification at \(N = 2000\), DM coverage collapses to 0.00 across all confidence interval methods.
  - Naive DM Wald intervals treat \(\hat{q}\) as fixed, completely omitting reward-model estimation variance. In Experiment 004, naive DM Wald undercovers severely even under a correctly specified linear model (28% coverage at \(N = 500\), 35% at \(N = 2000\) for nominal 95%).
  - Full-refit cross-fitting bootstrap remedies this omission for well-specified models (achieving 86%–92% empirical coverage), but cannot overcome asymptotic bias under misspecification.
* **Doubly Robust finite-sample variance under poor overlap:**
  - Although DR is asymptotically consistent whenever propensities are correct, double robustness does **not** guarantee low finite-sample variance.
  - Under poor overlap (heavy-tailed importance weights \(w_i\)), the factual residual \(w_i (r_i - \hat{q}(x_i, a_i))\) introduces massive sampling variance, leading to RMSE substantially worse than DM (e.g., in Experiment 003, DR RMSE was 0.1411 vs DM RMSE of 0.0166 at \(N = 2000\) under poor overlap).
* **Influence-score Wald calibration vs. poor-overlap undercoverage:**
  - DR influence-score Wald intervals (`dr_influence_wald`) achieve calibrated coverage (94%–98%) under good overlap, even when the reward model is misspecified (thanks to Neyman orthogonality and known logging propensities).
  - However, under poor overlap (mean ESS fraction \(\approx 0.053\)), DR influence Wald undercovers at 88%–91% (and IPS Wald undercovers at 82%–84%) because extreme weights violate finite-sample normality.
* **Nonparametric bootstrap fails to repair heavy-tail undercoverage:**
  - Under poor overlap, nonparametric bootstrap intervals (`ips_bootstrap_percentile`, `dr_bootstrap_refit_percentile`, `dr_bootstrap_refit_basic`) do not improve coverage over Wald intervals (e.g., IPS bootstrap covers 81%–85%, DR refit bootstrap covers 83%–88%).
  - This is consistent with Bickel & Freedman (1981): when importance weight variance is large or infinite, bootstrap empirical distributions do not converge uniformly, and resamples are dominated by rare extreme weights.
* **Diagnostics reliably detect uncalibrated regimes:**
  - Extended diagnostics (`concentrated_weights`, `low_effective_sample_size`, `invalid_inference_assumptions`) fired on 100% of poor-overlap replicates in Experiment 004, providing actionable warnings without invalidating point estimates.
* IPS is unbiased only under the assumptions in `assumptions.md`. In a finite
  sample it is unbounded above for nonnegative weights. An estimate above any
  bound on the reward is possible and is not truncated.
* If every importance weight is zero, IPS returns 0 and SNIPS raises
  `DegenerateWeightsError`. The IPS value 0 is the average of an empty
  contribution. It is not evidence that the policy value is 0. Diagnostics emit
  `all_weights_zero` in that case.
* SNIPS is a ratio estimator. It has bias of order \(1/N\) under standard
  conditions. Self-normalisation often reduces variance when rewards are bounded
  and weights are heavy-tailed. That is a known property of the estimator, not
  a finding of this project, and it is not true for every reward distribution.
* No cluster-robust interval or dependent-observation adjustment is
  implemented. `group_id` is stored and ignored. Real-data cluster bootstrap is deferred
  until genuine grouping identifiers and sampling assumptions are established.
* Kish effective sample size summarises weight imbalance. It is not the exact
  variance of IPS, which depends on \(w r\), not on \(w\) alone.
* The default diagnostic thresholds (ESS fraction 0.1, max weight share 0.05,
  mean-weight z-score 4) are conventional cut-offs. They have not been shown to
  predict model-selection error. Under heavy tails the mean-weight z-test
  produces false alarms, because the standard error of the mean weight is itself
  unstable and because \(\mathbb{E}[w] = 1\) only after averaging over a tail
  the sample may not contain.
* Softmax policies are positive in exact arithmetic. In float64 a very negative
  logit underflows to probability 0. That zero is left in place. If an
  evaluation policy then puts mass there, the run raises `SupportViolationError`
  rather than smoothing the probability.

## Data and ground truth

* One data source is a synthetic single-position contextual bandit.
  Contexts are isotropic Gaussian. Reward means are linear in the context, passed
  through a sigmoid for binary rewards or used directly for Gaussian rewards.
  This is a correctness testbed. It is not a model of a production recommender.
* The ground-truth policy value is Monte Carlo over contexts, with a reported
  standard error. For Gaussian rewards and a uniform policy it has the closed
  form \(\bar b\), and that case is tested. Other policies do not have a closed
  form in this repository.
* The other data source is the 10,000-row Open Bandit sample. Its estimand is
  the expected click of one position-level row. That is not a three-item slate
  value. The files are not committed. The full log was not downloaded, and its
  license at https://research.zozo.com/data.html has not been read.

## Open Bandit Dataset (small sample only)

Verified against the files on 2026-10-28. The write-up is
`docs/open_bandit_data_audit.md` and `docs/open_bandit_estimand.md`.

* The probability column in the files is `propensity_score`. The README in the
  same commit still says `action_prob`. The adapter does not alias them.
* Behavior-file item ids are `0–79`, `0–33`, and `0–45`. The item tables have
  the same ids. The README's item-context ranges `0–80` and `0–46` do not match
  these files.
* Random-policy propensities are exactly `1/K`, and the adapter stores a uniform
  logging distribution in that case. BTS propensities are only the displayed
  action's probability. Support is not verifiable from those values.
* User-feature hashes repeat. Estimators still assume independent rows, so the
  Wald intervals would be anti-conservative if those hashes are a dependence unit.
  Experiments 002 and 003 do not assume repeated hashes identify actual users.
* About 40–70 clicks per file. That sample is not evidence about interval coverage.
* Reference parity testing executes isolated upstream methods from Open Bandit Pipeline
  via a subprocess script (`src/reversallab/evaluation/obp_reference.py`) rather than
  importing the full unmaintained upstream package.
* The small files ship inside Apache-2.0 `zr-obp`. Cite arXiv:2008.07146 when
  using them. The full-data license is still unread.

## Decision reversal

* The search varies the proportion of two prespecified strata and assumes stable
  outcomes within each stratum. Its distance measures sensitivity to that one
  mixture change. A diagnostic failure prevents deployment certification.
* The simultaneous band combines two stratum-level Wald intervals. Its coverage
  depends on the quality of those intervals, especially under poor overlap.

## Reproducibility

* OBP reference checks run isolated estimator functions in an isolated subprocess.
  Full OBP is not an installed dependency.
