# Standard-error audit: poor overlap

Experiment 001 is the evidence. The IPS and SNIPS formulas were not changed.

## What was measured

Run `001_estimator_validation_20261028T115521877076Z_4b6f33ae`, results SHA-256 `7112fe2c80e886c5738aaf8d49b17cdc76bf48c7d3a654ec6890bff13ebf695c`. The poor-overlap cell uses a confident softmax logging policy on reward-independent scores and a near-greedy candidate on the true reward. Support did not fail. The largest importance weight in the `N = 500` cell is on the order of `10^5`.

| n | estimator | bias | Monte Carlo SE of bias | RMSE | coverage | mean reported SE | empirical SD of estimates | ESS/N |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 500 | ips | +0.6424 | 0.5602 | 12.5297 | 0.522 | 1.1177 | 12.5257 | 0.021 |
| 500 | snips | -0.0148 | 0.0062 | 0.1402 | 0.676 | 0.0807 | 0.1395 | 0.021 |
| 2000 | ips | +0.0318 | 0.0417 | 0.9329 | 0.632 | 0.3799 | 0.9333 | 0.010 |
| 2000 | snips | -0.0096 | 0.0049 | 0.1104 | 0.762 | 0.0667 | 0.1101 | 0.010 |
| 8000 | ips | -0.0076 | 0.0291 | 0.6491 | 0.654 | 0.2486 | 0.6497 | 0.005 |
| 8000 | snips | -0.0070 | 0.0032 | 0.0723 | 0.842 | 0.0509 | 0.0720 | 0.005 |

True value of the candidate in that cell: 0.8382. At `N = 8000` the IPS bias is compatible with zero (`-0.0076 ± 0.029`). Coverage is still 0.654. ESS/N falls as `N` grows because larger samples include rarer, heavier weights.

## Why the two standard errors differ

The reported IPS standard error is the sample standard deviation of the terms `w_i r_i`, with one degree of freedom, divided by `sqrt(N)`. The empirical SD is the standard deviation, across independent replications, of the IPS estimates themselves.

Those two numbers estimate the same dispersion when a typical sample's variance is close to the true variance of `w r`. In the poor-overlap cell a typical sample does not contain the rare terms that dominate the variance. The sample standard deviation is then small, the Wald interval is too short, and the replication that does contain a huge weight lands far outside the intervals reported by the quiet replications. The mean reported SE of 1.12 against an empirical SD of 12.5 at `N = 500` is that gap. It is not a second implementation of the sample-mean formula.

SNIPS shows the same pattern at a smaller scale: its delta-method SE is also below the empirical SD (0.081 against 0.140 at `N = 500`), and coverage rises from 0.676 to 0.842 without reaching the nominal level. The ratio estimator's finite-sample bias at `N = 8000` is about two Monte Carlo standard errors (`-0.0070 ± 0.0032`) and is small next to the logging-policy value.

`tests/unit/test_heavy_tail_se.py` isolates the arithmetic. Ninety-nine samples never take the rewarded action, so their Wald SE is 0 and their interval is `{0}`. One sample contains a single weight of 50. The average reported SE is much smaller than the standard deviation of the hundred estimates. That test does not calibrate coverage. It shows the mechanism with no Monte Carlo noise.

## When support cannot be checked

`find_support_violations` returns `None` when `logging_action_dist` is absent. An action the logger never takes does not appear in a column of observed-action propensities, so those propensities cannot prove that the action had probability zero, and they cannot prove that it did not. The estimator returns a number. Diagnostics add `support_unverifiable`. The number can be biased.

The check fires only when the caller supplies the full logging distribution and the candidate puts positive probability on an action whose logging probability is zero. The random Open Bandit files reach that check because every propensity equals `1/K` and the adapter stores the uniform matrix. The BTS files do not.

## What was not done

Weights were not clipped, truncated, or replaced by an epsilon. The Wald formula was not replaced. A bootstrap interval was not added. The small Open Bandit sample was not used as coverage evidence: it has on the order of 40–70 clicks per file and no ground-truth policy value.
