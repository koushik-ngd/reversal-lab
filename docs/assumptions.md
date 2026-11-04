# Assumptions

These are the conditions under which the implemented estimators have their usual
guarantees. When a condition fails, the code either raises or emits a diagnostic.
A diagnostic is not a correction.

## Sampling

1. Rows are independent and identically distributed draws from one logging policy.
   `group_id` may be stored, but no estimator in this version resamples by group.
   User- or session-level dependence makes the reported standard errors too small.
2. The context was observed before the action. Features that depend on the action
   or the reward are leakage.
3. The reward is a scalar observed only for the logged action. Rewards of the
   other actions are missing, not zero.
4. There is one action per row. A slate, a ranked list, or a position-dependent
   recommendation is a different estimand. See `limitations.md` for the Open
   Bandit Dataset, which is position-level data.

## Propensities and support

5. `logging_propensity[i]` equals \(\pi_0(a_i \mid x_i)\), the probability the
   logging policy assigned to the action it actually took, computed at logging
   time from the information available then. Estimated or post-hoc propensities
   are a different method and are not what IPS implements here.
6. That probability is strictly positive. A zero is a contract violation. It is
   rejected. It is not replaced by an epsilon.
7. **Support.** For every context in the population and every action,
   \(\pi_e(a \mid x) > 0\) implies \(\pi_0(a \mid x) > 0\).
   Under that condition,
   \[
   \mathbb{E}[w_i r_i] = V(\pi_e), \qquad w_i = \frac{\pi_e(a_i \mid x_i)}{\pi_0(a_i \mid x_i)}.
   \]
   If the full logging distribution is supplied and support fails on the logged
   contexts, estimation raises `SupportViolationError`. If the full distribution
   is absent, support is not identified from the log, and diagnostics report
   `support_unverifiable`. The estimate is still returned, and it can be biased.
   Observed-action propensities cannot supply the missing check: an action that
   was never logged leaves no probability behind. On the Open Bandit random
   files every propensity equals `1/K`, and the adapter stores the uniform
   matrix that column is consistent with. On the BTS files the matrix stays
   missing. The poor-overlap gap between the Wald SE and the empirical SD is
   audited in `docs/standard_error_audit.md`.
8. Action probabilities are finite, nonnegative, and each row sums to 1 within
   absolute tolerance `1e-6`. Rows are not renormalised.

## Reward modeling and cross-fitting

9. **Direct Method identification.** DM assumes that the estimated reward function
   \(\hat{q}(x, a)\) converges to the true conditional expectation \(\mathbb{E}[r \mid x, a]\).
   If \(\hat{q}\) is misspecified, DM is asymptotically biased regardless of sample size.
10. **Doubly Robust identification.** DR requires that *either* (a) the logging propensities
    and support condition hold, *or* (b) \(\hat{q}(x, a)\) converges to \(\mathbb{E}[r \mid x, a]\).
    If propensities are correct, DR is consistent even under severe reward-model misspecification.
11. **Cross-fitting independence.** To prevent in-sample overfitting bias, reward models
    must be trained out-of-fold: observations in fold \(k\) must never be used to fit
    the model or any preprocessing transformations that predict on fold \(k\).
12. **Action support in reward training.** Model fitting assumes that each action has enough
    observations across training folds to estimate parameters. If an action has zero
    observations in a training fold, `StratifiedLinearRewardModel` falls back to the global
    training mean reward for that fold.

## Uncertainty

13. Interval endpoints are Wald intervals from a normal approximation.
    IPS uses the unbiased sample standard deviation of \(w_i r_i\).
    SNIPS uses the delta-method standard error
    \(\sqrt{\sum_i w_i^2 (r_i - \hat V)^2} / \sum_i w_i\),
    which is the \(1/N\) form of the ratio variance.
14. For DM and DR, the naive sample variance treats \(\hat{q}\) as a fixed deterministic function,
    ignoring the estimation variance of \(\hat{q}\). Consequently, naive Wald intervals for DM and DR
    undercover and are omitted from benchmark reporting until calibrated intervals (influence functions
    or bootstrap) are implemented.
15. The normal approximation needs a finite variance of \(w r\) and a sample large
    enough for that variance to be estimated. Heavy-tailed weights violate the
    practical requirement even when the variance is finite. Coverage below the
    nominal level in that regime is an expected limitation, not a reason to clip
    the weights.
16. The reference policy value used in synthetic checks is exact in the action
    and Monte Carlo in the context. Its Monte Carlo standard error is part of
    any comparison. It is not an oracle available for real logs.

## What is not assumed

* Rewards are not assumed to lie in \([0, 1]\), except where a specific diagnostic
  comment says so (unsupported evaluation mass, as an upper bound on IPS bias
  for rewards in that range). The synthetic binary scenario does use Bernoulli
  rewards. Gaussian rewards are unbounded.
* The logging and evaluation policies are not assumed to be optimal, linear, or
  softmax. Those families are generators for the synthetic study.
* **Double robustness does NOT assume or guarantee low finite-sample variance.** Under poor
  overlap, DR can have substantially higher RMSE than DM due to the interaction of
  noisy factual residuals with extreme importance weights.
* A confidence interval is not a deployment guarantee.
* Agreement between IPS, SNIPS, DM, and DR is not validation. Disagreement is a diagnostic.
