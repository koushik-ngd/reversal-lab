# Experiment protocol

This document specifies the studies before their results are interpreted.
Experiments 1, 3, 003, and 004 are implemented and run. Experiments 2, 5–7 are specified and not run.
The decision-reversal method is not implemented.

A local 500-replication run of Experiment 1 already existed in this working
tree when this protocol was written down. The expectations in Experiment 1
follow from the estimator definitions (unbiasedness of IPS under support,
variance reduction of SNIPS for bounded rewards with variable weights, and
failure of Wald intervals under heavy tails). They are not an independent
preregistration, and they are not acceptance gates for the descriptive study.

## Central question

Can automated, statistically grounded sensitivity analysis identify deployment
decisions that look favorable under ordinary offline evaluation and become
unsupported under stated changes to the evaluation assumptions, the data
distribution, or operational constraints?

M2 and M3 do not collect evidence on this question. They check whether the value
estimators that a later sensitivity analysis would call are the estimators
they claim to be.

## Decision rule (design only)

Not implemented. The following is the rule M4 has to implement and then
compare with simpler baselines. It is not a validated policy.

Inputs, all pre-specified:

* an estimator (IPS, SNIPS, DM, or DR);
* a candidate policy and a baseline policy, evaluated on the same log;
* a minimum improvement \(\delta \ge 0\) in reward units;
* a confidence level;
* a support requirement;
* optional measured cost and latency limits.

Let \(\widehat{\Delta}\) be the estimated improvement and \([L, U]\) its
interval at the chosen level, on the same assumptions as the estimator.

* `RECOMMEND` when support holds, every declared operational limit is met,
  and \(L > \delta\).
* `HOLD` when support fails, or an operational limit fails, or \(U < \delta\).
* `INCONCLUSIVE` otherwise, including a missing full logging distribution
  when the protocol requires support to be verified.

These labels are summaries of the logged evidence. They are not a statement
that the candidate will win in production.

## Fragility (not defined yet)

M4 has to define, before any search code is written:

* the original decision under the rule above;
* the perturbation parameter and its unit;
* the admissible set, with bounds that have a sampling or operational meaning;
* the magnitude of a perturbation;
* the decision-change condition;
* the search procedure.

A reversal is a change of label inside that set. It is not, by itself, a
demonstration that the candidate is unsafe. The first search should be a
direct evaluation on a grid or a bisection in one dimension. The quantity to
study is the smallest admissible magnitude that changes the label, together
with the statement that no such magnitude exists inside the set.

That quantity has to be compared with three baselines on data that were not
used to choose it: a lower-confidence-bound rule, an effective-sample-size
threshold, and leaving the estimate unchanged. The comparison has to include
cases where the reversal flag is wrong. Until that comparison exists, the
repository will not describe the method as novel or superior.

Perturbations that are in scope for the design, and only where the required
data exist:

* reweighting of a pre-declared cohort coordinate;
* changing the sample composition inside a pre-declared bound;
* restricting the time window, only if a timestamp exists and the window is
  declared before looking at the estimate;
* down-weighting or dropping the largest importance weights inside a
  pre-declared cap, reported as a sensitivity curve and not as a corrected
  estimate;
* replacing the point estimate by its interval endpoint;
* a cost or latency constraint, only with a measured cost or latency.

Arbitrarily editing propensities or rewards to force a label change is not
an admissible perturbation.

## Leakage controls

* Policy parameters in Experiment 1 are either the environment's reward
  parameters or logits drawn from `score_seed`. No policy is fit on the
  evaluation log.
* The reference value uses `SeedSequence([seed, 1, crc32(scenario)])`.
  Replicate `r` at size `n` uses
  `SeedSequence([seed, 2, crc32(scenario), n, r])`.
  Adding a scenario does not move the stream of another scenario.
* In Experiment 003, cross-fitting uses deterministic disjoint fold partitions:
  training folds never observe held-out evaluation observations, and feature
  preprocessors are fit exclusively on training folds.
* Reversal tuning and confirmatory evaluation do not exist yet. When they do,
  they get separate splits, and the split is recorded in the run config.
  Discovery results will not be reported as confirmation.

## Multiple comparisons

Experiments 1, 2, and 003 have fixed scenario lists and report every cell.
They do not declare a discovery. Later experiments that search over
perturbations will state how many comparisons were made and will not treat
the most extreme reversal as a pre-specified test.

## Experiment 1 — Estimator validation (implemented)

**Question.** Do IPS and SNIPS recover the known policy value of the synthetic
environment, and how do bias, RMSE, and Wald coverage change with sample size
and overlap?

**Config.** `configs/experiments/001_estimator_validation.yaml`

* seed `20261028`, 500 replications, 95% Wald intervals;
* reference value from \(10^6\) independent contexts;
* binary rewards, \(K = 10\), \(D = 5\);
* sample sizes 500, 2000, 8000;
* scenarios: identical policies, uniform logging with a stochastic candidate,
  moderate mismatch, deterministic candidate, a candidate that prefers
  low-reward actions, and poor overlap (confident logging policy with scores
  independent of reward, near-greedy informed candidate).

**Metrics recorded per cell.** Mean estimate, bias, Monte Carlo standard error
of the bias, RMSE, empirical standard deviation across replications, mean
reported standard error, interval coverage and its Monte Carlo standard error,
mean ESS fraction, max weight, diagnostic warning rates, and mean absolute
IPS–SNIPS disagreement. Failed replicates (degenerate SNIPS weights) are
counted and excluded from the moments.

The numbers from the run executed for this revision are in
`experiments/001_estimator_validation/README.md`.

## Experiment 2 — Uncertainty calibration (not run)

Extend Experiment 1 with a grid of sample sizes and overlap strengths.
Report coverage of the current Wald intervals and, once implemented, of a
bootstrap interval. For synthetic data the intervals are independent across
rows; the coverage target is the nominal level. A grouped design will be added
only together with a variance estimator that respects the group. Success is a
measured coverage curve with Monte Carlo error, including the region where
coverage fails.

## Experiment 3 — Open Bandit reference parity (small sample, run)

The estimand is in `docs/open_bandit_estimand.md`. The file audit is in
`docs/open_bandit_data_audit.md`. The comparison target is Open Bandit
Pipeline's IPW and SNIPW on the same arrays, with clipping off. A mismatch
is a bug investigation, not a new estimator.

The small-sample run is experiment 002. It compared 30 policy/slice pairs.
The largest absolute difference was `1.73e-18`, inside absolute and relative
tolerance `1e-12`. Numbers and provenance are in
`experiments/002_open_bandit_parity/README.md`.

Still required before any full-log estimate:

1. Read the license at https://research.zozo.com/data.html.
2. Re-check column names on a file from that distribution. Do not assume the
   small-sample header is the full-log header.
3. Keep the position-level estimand, or write a different one if a real slate
   key is present. Do not compare a 10,000-row estimate with a published
   full-data figure.

## Experiment 003 — Advanced Estimator Validation: DM, DR, and Cross-Fitting (implemented)

**Question.** Do Direct Method (DM) and Doubly Robust (DR) estimators with \(K\)-fold
cross-fitting behave as predicted by statistical theory under good and poor overlap,
correct and misspecified reward models, and binary and continuous rewards?

**Config.** `configs/experiments/003_advanced_estimators.yaml`

* seed `20261030`, 100 replications, 5-fold cross-fitting;
* sample sizes 500 and 2000;
* reference values from 500,000 independent contexts with exact action summation;
* scenarios:
  1. `good_overlap_correct_model`: binary rewards, well-specified logistic regression;
  2. `good_overlap_misspecified_model`: binary rewards, constant mean prediction model;
  3. `poor_overlap_correct_model`: binary rewards, confident logging policy, well-specified model;
  4. `poor_overlap_misspecified_model`: binary rewards, confident logging policy, constant model;
  5. `continuous_reward_good_overlap`: Gaussian rewards, well-specified Ridge regression;
  6. `continuous_reward_misspecified`: Gaussian rewards, constant mean prediction model.

**Metrics recorded per cell.** Bias, Monte Carlo standard error of bias, RMSE,
empirical standard deviation across replications, mean reported standard error,
Kish effective sample size fraction, and max importance weight.
For DM and DR, CI coverage is explicitly omitted (`N/A`) because naive Wald intervals
that treat \(\hat{q}\) as fixed omit reward-model estimation variance.

**Key findings:**
* **Double Robustness verified:** Under model misspecification, DM has large asymptotic bias
  (\(-0.0259\) at \(N = 2000\)), while DR remains completely unbiased (\(-0.0009 \pm 0.0014\)).
* **Finite-sample variance under poor overlap:** When overlap is poor (mean ESS fraction 0.060),
  DR RMSE (0.1411 at \(N = 2000\)) is substantially worse than DM RMSE (0.0166) due to the
  interaction of non-zero sample residuals with extreme importance weights.
* Full numbers and provenance are in `experiments/003_advanced_estimators/README.md`.

## Experiment 004 — Confidence Interval Calibration and Statistical Uncertainty (run)

Formal benchmark evaluating confidence interval coverage, interval width, bias, RMSE, and failure rates across 9 uncertainty methods.

**Design:**
- 6 scenarios: Good overlap (correct and misspecified model), Poor overlap (correct and misspecified model), Continuous rewards (Gaussian with correct and misspecified model).
- Sample sizes: \(N \in \{500, 2000\}\).
- 100 independent Monte Carlo replications per cell.
- 50 bootstrap resamples per replication.
- 3-fold cross-fitting for reward model training.
- Nominal confidence level: \(95\%\) (\(\alpha = 0.05\)).
- Independent ground truth computed via \(10^6\) Monte Carlo contexts.

**Evaluated Methods:**
1. `ips_wald`: Asymptotic normal Wald interval for IPS using empirical sample variance.
2. `snips_wald`: Asymptotic normal Wald interval for SNIPS using delta-method variance.
3. `dm_naive_wald`: Sample standard error of DM pseudo-outcomes (treating \(\hat{q}\) as fixed).
4. `dr_influence_wald`: Empirical variance of uncentered DR influence scores \(\psi_i\).
5. `ips_bootstrap_percentile`: Nonparametric percentile bootstrap for IPS.
6. `snips_bootstrap_percentile`: Nonparametric percentile bootstrap for SNIPS.
7. `dm_bootstrap_refit_percentile`: Nonparametric percentile bootstrap for DM with full pipeline refitting.
8. `dr_bootstrap_refit_percentile`: Nonparametric percentile bootstrap for DR with full pipeline refitting.
9. `dr_bootstrap_refit_basic`: Nonparametric empirical (basic) bootstrap for DR with full pipeline refitting.

**Key empirical findings:**
1. **Direct Method Naive Wald Breakdown**: Naive DM Wald undercovers severely (28% at \(N = 500\), 35% at \(N = 2000\)) because it ignores reward-model estimation variance. Full-refit bootstrap restores near-nominal coverage (86%–92%) when the model is well-specified. Under misspecification, DM coverage drops to 0% across all methods due to irreducible bias.
2. **Doubly Robust Calibration via Influence Scores**: Under good overlap, `dr_influence_wald` achieves nominal 94%–98% coverage with correct models and 94%–96% with misspecified models, verifying the Neyman orthogonality property of the efficient influence function.
3. **Poor Overlap & Heavy Tails (Bootstrap Inconsistency)**: Under poor overlap (mean ESS fraction \(\approx 0.053\)), both IPS Wald (82%–84%) and DR influence Wald (88%–91%) undercover. The nonparametric bootstrap fails to repair this undercoverage (`ips_bootstrap_percentile` covers 81%–85%, `dr_bootstrap_refit_percentile` covers 83%–88%), confirming the Bickel & Freedman (1981) heavy-tail bootstrap breakdown.
4. Full run artifact and provenance: `experiments/004_uncertainty_calibration/README.md`.

## Experiment 4 — Decision reversal (not run)

Requires Experiments 1 and 2, plus a frozen written definition of the
perturbation set. The study estimates how often the label changes inside that
set, and whether the change agrees with the policy value on an independent
draw from the same environment. On real data there is no independent draw from
the target policy, so the real-data arm can measure stability only.

## Experiment 5 — Baseline comparison (not run)

Same splits as Experiment 4. Compare the reversal flag with a
lower-confidence-bound rule, an ESS threshold, and the unadjusted estimate.
Report false reversals and missed failures separately. Do not collapse them
into one accuracy number without the base rate.

## Experiment 6 — Ablations (not run)

Remove one declared component of the perturbation set at a time. A component
that does not change agreement with the independent evaluation is reported as
unused. Components are not dropped because they make the method look better.

## Experiment 7 — Efficiency (not run)

Measure wall time, peak memory, and rows per second against \(N\) for the
implemented estimators. No service, container, or cluster is in this design.

## Artifacts

Every run writes one `run.json` with schema version, experiment name, full
config, `config_sha256`, results, `results_sha256`, warnings, runtime, Python
and NumPy versions, platform, git commit and dirty flag when git can provide
them, and the command that reproduces the logical result. `results_sha256`
covers `results` only.

Generated runs stay out of version control (`runs/` is gitignored).
