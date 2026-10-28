# Experiment protocol

This document specifies the studies before their results are interpreted.
Experiment 1 is implemented. Experiments 2–7 are specified and not run.
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

M2 does not collect evidence on this question. It checks whether the value
estimators that a later sensitivity analysis would call are the estimators
they claim to be.

## Decision rule (design only)

Not implemented. The following is the rule M4 has to implement and then
compare with simpler baselines. It is not a validated policy.

Inputs, all pre-specified:

* an estimator (IPS or SNIPS now; DM or DR once they exist);
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
* Reward-model fitting, reversal tuning, and confirmatory evaluation do not
  exist yet. When they do, they get separate splits, and the split is recorded
  in the run config. Discovery results will not be reported as confirmation.

## Multiple comparisons

Experiment 1 has a fixed scenario list and reports every cell. It does not
declare a discovery. Later experiments that search over perturbations will
state how many comparisons were made and will not treat the most extreme
reversal as a pre-specified test.

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

**Implementation gates, enforced by tests.**

* Hand-computed IPS and SNIPS on a four-row example.
* On-policy weights reduce both estimators to the sample mean of the reward.
* Invalid probabilities, shapes, non-finite values, unverifiable dtypes, and
  non-positive propensities raise `DataValidationError`.
* Verified support violations raise `SupportViolationError`.
* SNIPS with all weights zero raises `DegenerateWeightsError`.
* Overflow of a weight to infinity raises `NumericalError`.
* For the fixed cases in `tests/statistical/test_ground_truth.py`, the Monte
  Carlo mean of the estimate lies within four combined standard errors of the
  reference value.
* With uniform logging and the informed softmax target, the IPS 95% interval
  at \(N = 2000\) covers within four binomial standard errors of 0.95.
* The same config produces the same `results_sha256`. Results for a scenario
  do not depend on which other scenarios are in the file.

**Descriptive expectations, not test gates.**

* On policy, IPS and SNIPS match.
* Where weights are bounded by \(K\) (uniform logging), coverage should sit
  near 95% and RMSE should fall as \(N\) grows.
* Under poor overlap, IPS should have larger RMSE than SNIPS, reported
  standard errors should fall short of the cross-replication standard
  deviation, and coverage may sit below 95%. ESS as a fraction of \(N\) may
  fall as \(N\) grows, because larger samples include rarer and heavier weights.

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
