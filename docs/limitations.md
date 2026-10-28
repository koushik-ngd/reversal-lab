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

* The only data source is a synthetic single-position contextual bandit.
  Contexts are isotropic Gaussian. Reward means are linear in the context, passed
  through a sigmoid for binary rewards. This is a correctness testbed. It is not
  a model of a production recommender.
* The ground-truth policy value is Monte Carlo over contexts, with a reported
  standard error. For Gaussian rewards and a uniform policy it has the closed
  form \(\bar b\), and that case is tested. Other policies do not have a closed
  form in this repository.
* Real logged data are not loaded. No number in this repository is an Open
  Bandit Dataset result.

## Open Bandit Dataset (not integrated)

Checked on 2026-10-28, without downloading the data:

* Open Bandit Pipeline (`st-tech/zr-obp`) is Apache-2.0 and is the reference
  library for a later numerical comparison. It is not a dependency yet.
* The dataset is a 7-day ZOZOTOWN experiment with two logging policies
  (uniform random, Bernoulli Thompson sampling) and three campaigns
  (all, men, women). The small sample ships with the library (10,000 rows per
  policy-campaign pair). The full log is about 26 million position-level rows
  and is distributed from the ZOZO research site and as
  `zozonext/open-bandit` on Hugging Face. It was not downloaded.
* Each row is one recommended item in one of three interface positions, with a
  click indicator and the probability of that item at that position. Timestamps
  and hashed user features are present in both the library README and the
  Hugging Face card.
* The two public field lists disagree, so neither is treated as authoritative
  until a file is read:
  * `obd/README.md` in `zr-obp` names the probability column `action_prob` and
    describes item indices `0-79` (all), `0-33` (men), `0-45` (women). The same
    file describes item-context indices `0-80` for the all campaign.
  * The Hugging Face card names the probability column `propensity_score` and
    describes item indices `0-80`, `0-33`, and `0-46`. Its viewer also displays
    two different row counts. Those counts are not used here.
* The logged probability is the propensity of the displayed item at that
  position. The full action distribution is not a column. Whether positions are
  filled without replacement, and whether rows that share a user and a timestamp
  are one slate, has to be taken from the paper and the loader, not guessed.
* The data license is not stated in `obd/README.md`. The files will not be
  copied into this repository until that license is read at the distribution
  site. Citation for the accompanying paper is arXiv:2008.07146. The README and
  the dataset card use different titles for that preprint; the PDF is the
  citation to use in a later research report.
* A single-position IPS applied to these rows would answer a question this
  schema does not state. M3 has to name the estimand first.

## Decision reversal

* There is no decision rule, perturbation search, or fragility metric in the
  code. The central research question is unanswered. A sketch of the future
  decision rule is in `experiment_protocol.md` so that it can be reviewed
  before it is implemented.

## Reproducibility gaps

* Run artifacts record `git commit: null` until this tree has a commit.
  Config and result SHA-256 hashes still identify the logical experiment.
* Runtime and platform are recorded and are excluded from `results_sha256`.
* This version has not been compared numerically with Open Bandit Pipeline.
  Hand calculation and the synthetic Monte Carlo checks are the evidence that
  IPS and SNIPS are implemented as defined. They are not a comparison with
  prior software.
