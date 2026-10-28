# Limitations

Status: M2. Items below are properties of this version, not a backlog disguised
as completed work.

## Estimators

* Only IPS and SNIPS are implemented. Direct Method and Doubly Robust are not.
  There is no reward model and no cross-fitting.
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
* Wald intervals undercover when weights are heavy-tailed. The per-dataset
  standard error can be much smaller than the standard deviation of the estimate
  across repeated samples, because one sample often misses the tail that
  dominates the sampling distribution. Experiment 001 measures this. It does
  not correct it.
* No bootstrap, cluster-robust, or effective-sample-size-adjusted interval is
  implemented. `group_id` is stored and ignored.
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
  through a sigmoid for binary rewards. This is a correctness testbed. It is not
  a model of a production recommender.
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
  Experiment 002 does not report intervals.
* About 40–70 clicks per file. That sample is not evidence about interval coverage.
* IPS and SNIPS match OBP's IPW and SNIPW on the same arrays (experiment 002).
  That match is not a published full-dataset benchmark.
* The small files ship inside Apache-2.0 `zr-obp`. Cite arXiv:2008.07146 when
  using them. The full-data license is still unread.

## Decision reversal

* There is no decision rule, perturbation search, or fragility metric in the
  code. The central research question is unanswered. A sketch of the future
  decision rule is in `experiment_protocol.md` so that it can be reviewed
  before it is implemented.

## Reproducibility gaps

* The M2 baseline commit is `2a83f9e184f51f1b9356f8d32ac7601266b22f2e`.
  Later edits stay dirty in `run.json` until they are committed. `uv.lock`
  is hashed into provenance. Runtime and platform stay outside `results_sha256`.
* The Wald interval undercovers under poor overlap. The formula audit is
  `docs/standard_error_audit.md`. No replacement interval is implemented.
* OBP is executed from a local source checkout, not installed as a dependency.
  Its package init does not import on this Python. The bridge is described in
  experiment 002.
