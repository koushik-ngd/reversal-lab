# ReversalLab

ReversalLab is a Python library for off-policy evaluation of contextual bandits,
with a planned decision-reversal analysis that is not implemented yet.

This version (0.1.0.dev0) estimates the value of a candidate policy from logged
single-position data using inverse propensity scoring (IPS) and self-normalized
IPS (SNIPS). It checks those estimators against a synthetic environment whose
policy value is known up to a reported Monte Carlo error. IPS and SNIPS are
standard estimators. They are not a research claim.

The question the project is being built to study, and has not yet studied, is
whether a deployment comparison that looks favorable under ordinary offline
evaluation remains supported when the evaluation assumptions, the sample
composition, or declared operational constraints are changed inside a
pre-specified set. The design is in [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer. The pinned
interpreter for this repository is 3.12 (`.python-version`).

```text
uv sync --frozen --all-groups
```

## Test and check

```text
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
```

`pytest` includes the Monte Carlo tests in `tests/statistical/`. They use fixed
seeds. On a recent laptop the full suite takes a few minutes, dominated by
those tests.

`make check` runs the same steps where GNU make is available. The `uv` commands
are the ones that work in any shell, including PowerShell.

## Reproduce the baseline experiment

```text
uv run reversallab validate-estimators --config configs/experiments/001_estimator_validation.yaml
uv run reversallab report --run-dir runs/<run_id>
```

The run directory is printed at the end of the first command. It contains
`run.json` with the config, both SHA-256 hashes, estimates, diagnostic rates,
warnings, runtime, and the Python environment. The measured summary is written
up in [`experiments/001_estimator_validation/README.md`](experiments/001_estimator_validation/README.md)
for the revision that produced it. Re-running the command is the check.

`reversallab --help` lists the two commands that exist: `validate-estimators`
and `report`.

## Layout

```text
src/reversallab/
  contracts/      LoggedBanditData and array validation
  data/           synthetic contextual bandit with a reference policy value
  policies/       uniform, linear softmax, linear epsilon-greedy
  evaluation/     importance weights, IPS, SNIPS, diagnostics
  experiments/    config, Monte Carlo runner, run.json provenance
  cli.py
configs/experiments/001_estimator_validation.yaml
docs/             architecture, methodology, data contract, protocol,
                  assumptions, limitations
tests/            unit, statistical, integration
```

Direct Method, Doubly Robust, the Open Bandit Dataset, the reversal search,
a service, and containers are not in this tree. The reasons and the entry
conditions for the next milestone are in [`docs/limitations.md`](docs/limitations.md).

## Mathematical scope

For logged tuples \((x_i, a_i, r_i, p_i)\) and \(w_i = \pi_e(a_i \mid x_i) / p_i\),

\[
\hat V_{\mathrm{IPS}} = \frac{1}{N}\sum_i w_i r_i, \qquad
\hat V_{\mathrm{SNIPS}} = \frac{\sum_i w_i r_i}{\sum_i w_i}.
\]

IPS is unbiased when the rows are i.i.d., the propensities are the true logging
probabilities, and \(\pi_e(a \mid x) > 0\) implies \(\pi_0(a \mid x) > 0\).
SNIPS is consistent under those conditions and undefined when the total weight
is zero. Intervals are normal approximations. Full definitions, diagnostics,
and the synthetic data law are in [`docs/methodology.md`](docs/methodology.md).
Field-by-field validation is in [`docs/data_contract.md`](docs/data_contract.md).

Zero propensities and overflowing weights are errors. They are not clipped.

## License

MIT. See [LICENSE](LICENSE).
