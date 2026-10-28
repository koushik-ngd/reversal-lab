# Architecture

Status: describes the code as of M3A (`0.1.0.dev0`). Planned components are marked **planned**.

## Layering

```
contracts/     validated containers and array checks          (no dependencies on other layers)
   ↑
policies/      Policy protocol, sampling, reference policies  (depends on contracts)
data/          synthetic environment and the Open Bandit adapter (contracts, policies)
evaluation/    importance weights, IPS, SNIPS, diagnostics, OBP bridge (contracts)
   ↑
experiments/   configs, Monte Carlo runner, provenance         (all of the above)
   ↑
cli.py         argparse front end                              (experiments)
```

Rules enforced by convention and review:

* `evaluation/` never imports `data/` or `policies/`. Estimators see only a
  `LoggedBanditData` and an `(N, K)` evaluation-policy matrix, so the same code
  path serves synthetic and real data.
* Policies are deterministic functions `context -> action distribution`. All
  randomness flows through explicit `numpy.random.Generator` arguments.
* Estimators do not perform I/O. The Open Bandit adapter does: it is a file
  reader with a typed result. It does not estimate a policy value.
* Statistical problems (low ESS, heavy weights) are *reported* by diagnostics.
  Identification failures (support violation, all-zero weights for SNIPS),
  non-finite importance weights, and contract violations *raise* typed exceptions
  from `reversallab.exceptions`. Overflowed weights are not clipped.

## Key interfaces

| Interface | Location | Contract |
|---|---|---|
| `LoggedBanditData` | `contracts/logged_data.py` | Immutable, validated batch; see `data_contract.md` |
| `Policy` (Protocol) | `policies/base.py` | `n_actions`, `action_distribution(context) -> (N, K)` |
| `Estimator` (Protocol) | `evaluation/estimators.py` | `(data, eval_dist, *, confidence_level) -> PolicyValueEstimate` |
| `diagnose` | `evaluation/diagnostics.py` | `(data, eval_dist, thresholds) -> DiagnosticReport` |
| `EstimatorValidationConfig` | `experiments/config.py` | Pydantic model, `extra="forbid"`, frozen |
| `run.json` | `experiments/metadata.py` | Run artifact, schema version `1` |

The evaluation-policy distribution is passed separately from the logged data
rather than stored inside it: one log is typically evaluated against many
candidate policies, and keeping the log immutable avoids accidental reuse of a
stale candidate matrix.

## Design decisions and trade-offs

* **NumPy arrays plus a frozen dataclass, not pandas/pydantic, for data.** Pydantic
  validation of large arrays is slow and adds little over explicit NumPy checks.
  Pydantic is used for configuration, where it gives precise error messages.
* **Functions for estimators, not classes.** IPS/SNIPS have no fitted state. The
  Direct Method and Doubly Robust estimators (M3) need fitted reward models, so
  they will get an explicit fit/estimate split with cross-fitting; the
  `Estimator` protocol will be revisited then rather than designed speculatively now.
* **argparse, not click/typer.** There are three commands, and argparse adds no dependency.
* **JSON run artifacts, not MLflow.** One self-describing `run.json` per run
  contains the config, its hash, results, a results hash, warnings, timings, and
  provenance. MLflow can be added later as an optional sink.
* **Stable seed derivation.** Seeds are derived from `(seed, purpose, crc32(scenario name), n, replicate)`
  rather than from loop position, so adding or reordering scenarios leaves the
  existing results unchanged (this is tested).

## Planned (not implemented)

* `evaluation/`: reward models, Direct Method, Doubly Robust with cross-fitting,
  bootstrap and cluster-robust intervals.
* Columnar access to the full Open Bandit log, after its license is read.
  The small-sample adapter is `data/open_bandit.py`. `evaluation/obp_reference.py`
  runs upstream IPW/SNIPW from a local checkout. OBP is not a package dependency.
* `reversal/`: decision rule, perturbation sets, reversal search, fragility (M4).
* `reporting/`, `service/` (FastAPI), dashboard, containers (M5–M7).
