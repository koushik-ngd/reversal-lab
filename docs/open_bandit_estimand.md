# Estimand for the small Open Bandit sample

This is the policy value that experiment 002 estimates. It is fixed before the estimates. The file facts it relies on are in `docs/open_bandit_data_audit.md`.

## Unit of observation

One row of `{behavior_policy}/{campaign}/{campaign}.csv`. That row is one displayed item in one interface position, with one click indicator. The small sample does not group rows into three-item slates. Timestamps are not used as group ids.

## Action space

`A = {0, 1, ..., K-1}` with `K = 80` (all), `34` (men), or `46` (women). `item_id` is already this index.

## Policies

The logging policy `π_0` is the behavior policy that produced the file, either `random` or `bts`.

For `random`, every logged propensity equals `1/K`. The estimand uses

`π_0(a | row) = 1/K` for every action `a`.

That matrix is what the adapter stores as `logging_action_dist`. The file records only the displayed action's probability. Filling the other `K-1` entries is the uniform random policy this column is consistent with. It is the reason support can be checked on the random files.

For `bts`, the logging probability used by IPS and SNIPS is the logged `propensity_score` of the displayed action. The probabilities of the other actions are not in the file. `logging_action_dist` stays missing. Support is an assumption, not a check. A candidate that puts mass on an action the BTS logger never played is not detected.

The target policies in experiment 002 are functions of the row, not of the logged action:

* Uniform: `π_e(a | row) = 1/K` for every action.
* Constant action 0: `π_e(0 | row) = 1` and `π_e(a | row) = 0` otherwise.

"Probability 1 on whatever action was logged" is not one of these policies. It is not a function of context.

## Policy-value estimand

Let the rows of one file, or of one explicit position slice of that file, be the empirical distribution of contexts. The target is

`V(π_e) = E[ E[click | row, action] evaluated at action ~ π_e(· | row) ]`.

IPS and SNIPS estimate `V(π_e)` from `(action, click, propensity)` and from `π_e`. They do not use the affinity matrix.

On the random logger, the uniform target is the logging policy, the importance weights are 1, and IPS equals SNIPS equals the click rate of the rows in the slice. That is a sanity check. On the BTS logger the uniform target is a different policy, and its IPS estimate is not the BTS click rate.

## Position

A pooled estimate uses every row in the file. Positions 1, 2, and 3 keep the frequencies in the file. The estimand is the expected click of a single slot drawn from that mixture.

A position-specific estimate uses only the rows with that raw position. It is a different estimand. Experiment 002 reports both. The position filter is applied before either estimator runs, and both estimators see the same filtered arrays.

OBP's slate tensor, shape `(n_rounds, n_actions, len_list)`, is not this estimand. We do not place ZOZOTOWN positions 1, 2, and 3 on that third axis. Doing so would treat one position-level row as one slot of a slate whose other slots are missing.

## How this differs from the value of a three-item slate

A slate value is a function of the three positions together. Two examples are the expected number of clicks among the three displayed items, and the probability that at least one of them is clicked. Either quantity needs a joint distribution over the three items and their clicks.

The estimand above is the expected click of one row. Averaging position-level clicks does not produce the expected total of a slate, and it does not produce the probability of at least one click. This sample has no key that reassembles three positions into one impression: shared timestamps have multiplicity at most 2. The adapter does not invent that key.

## Dependence and support

User-feature tuples repeat, so the rows are not known to be independent. The implemented standard errors still treat rows as i.i.d. Those intervals are anti-conservative if the repeated tuples are a real dependence unit. Experiment 002 does not report intervals and does not claim coverage.

IPS is unbiased for `V(π_e)` when the propensities are the true logging probabilities, the rows satisfy the usual identical-distribution condition used to define the expectation, and `π_e(a | row) > 0` implies `π_0(a | row) > 0`. The last condition is checked for the random logger because the uniform matrix is stored. It is not checked for BTS. SNIPS is the self-normalized ratio of the same weights. It is undefined when the total weight is zero.

## What is out of scope

Direct Method, Doubly Robust, bootstrap intervals, a slate estimand, the full 26M-row log, and any published full-data number. A match against OBP on these arrays is a check that both implementations compute this estimand. It is not a reproduction of a published benchmark computed on a different sample.
