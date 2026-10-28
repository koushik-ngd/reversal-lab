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
8. Action probabilities are finite, nonnegative, and each row sums to 1 within
   absolute tolerance `1e-6`. Rows are not renormalised.

## Uncertainty

9. Interval endpoints are Wald intervals from a normal approximation.
   IPS uses the unbiased sample standard deviation of \(w_i r_i\).
   SNIPS uses the delta-method standard error
   \(\sqrt{\sum_i w_i^2 (r_i - \hat V)^2} / \sum_i w_i\),
   which is the \(1/N\) form of the ratio variance. The two standard errors are
   therefore not the same finite-sample correction.
10. The normal approximation needs a finite variance of \(w r\) and a sample large
    enough for that variance to be estimated. Heavy-tailed weights violate the
    practical requirement even when the variance is finite. Coverage below the
    nominal level in that regime is an expected limitation, not a reason to clip
    the weights.
11. The reference policy value used in synthetic checks is exact in the action
    and Monte Carlo in the context. Its Monte Carlo standard error is part of
    any comparison. It is not an oracle available for real logs.

## What is not assumed

* Rewards are not assumed to lie in \([0, 1]\), except where a specific diagnostic
  comment says so (unsupported evaluation mass, as an upper bound on IPS bias
  for rewards in that range). The synthetic binary scenario does use Bernoulli
  rewards. Gaussian rewards are unbounded.
* The logging and evaluation policies are not assumed to be optimal, linear, or
  softmax. Those families are generators for the synthetic study.
* A confidence interval is not a deployment guarantee.
* Agreement between IPS and SNIPS is not validation. Disagreement is a diagnostic.
