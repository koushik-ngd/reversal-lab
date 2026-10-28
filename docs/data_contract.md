# Data contract: `LoggedBanditData`

Single-position contextual-bandit log with \(N\) observations, \(K\) actions, and
\(D\) context features. The class is validated on construction and immutable afterwards:
arrays are copied and marked read-only.

| Field | Shape / dtype | Statistical meaning | Validation |
|---|---|---|---|
| `context` | `(N, D)` float64 | Features observed **before** the action was chosen. Post-action features would leak outcome information. | 2-D, finite, `N >= 1` |
| `action` | `(N,)` int64 | Index of the action the logging policy actually took | Integer dtype (floats/bools rejected, never rounded), values in `[0, K)` |
| `reward` | `(N,)` float64 | Observed outcome of the logged action only; rewards of other actions are unobserved | Finite |
| `logging_propensity` | `(N,)` float64 | \(p_i = \pi_0(a_i \mid x_i)\), the probability of the **observed** action under the logging policy at logging time | Strictly in `(0, 1]`; zeros are rejected, **not** replaced with epsilon |
| `n_actions` | int | Size \(K\) of the action space | `>= 1` |
| `logging_action_dist` | `(N, K)` float64, optional | Full \(\pi_0(\cdot \mid x_i)\). Allows support to be verified for every action | Rows in [0, 1], sum to 1 (atol 1e-6); must agree with `logging_propensity` at the logged action (rtol 1e-6, atol 1e-9) |
| `group_id` | `(N,)` any, optional | Dependence unit (user, session). Not used by M2 estimators, which assume i.i.d. rows | Shape only |

Evaluation-policy input, passed alongside the data:

| Input | Shape | Meaning | Validation |
|---|---|---|---|
| `evaluation_action_dist` | `(N, K)` | \(\pi_e(\cdot \mid x_i)\) evaluated on the logged contexts | Finite, in [0, 1], rows sum to 1 (atol 1e-6); **not renormalised** |

## Observed propensities vs full distributions

* `logging_propensity` is always required. It is the only quantity IPS/SNIPS need.
* Support (\(\pi_e > 0 \Rightarrow \pi_0 > 0\)) **cannot be verified** from observed
  propensities alone, because an action the logger never takes leaves no trace in the log.
  Without `logging_action_dist`, diagnostics emit `support_unverifiable`.
* With `logging_action_dist`, the estimators raise `SupportViolationError` if any
  `(i, a)` has \(\pi_e(a\mid x_i) > 0\) and \(\pi_0(a\mid x_i) = 0\).
* A strictly positive propensity can still be too small for float64. If
  \(\pi_e(a_i \mid x_i) / p_i\) overflows, estimators raise `NumericalError`.
  The ratio is not clipped and not replaced by a large finite cap.

## Not covered

* Slates and multiple positions. A single-position estimator must not be applied to
  slate data. The Open Bandit Dataset has positions, and its handling is an explicit
  M3 design task.
* Time stamps and cohort labels. They will be added only after verifying that they
  exist in the source data.
* Continuous actions.
