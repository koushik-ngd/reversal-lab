# Architecture

Status: describes the code as of M3B-2 (`0.1.0.dev0`). Planned components are marked **planned**.

## Layering

```
contracts/     validated containers and array checks          (no dependencies on other layers)
   ↑
policies/      Policy protocol, sampling, reference policies  (depends on contracts)
data/          synthetic environment and the Open Bandit adapter (contracts, policies)
evaluation/    estimators (IPS, SNIPS, DM, DR), influence scores, bootstrap, reward models, cross-fitting, diagnostics, OBP bridge (contracts)
   ↑
experiments/   configs, Monte Carlo runners, uncertainty calibration, provenance (all of the above)
   ↑
cli.py         argparse front end                             (experiments)
```

Rules enforced by convention and review:

* `evaluation/` never imports `data/` or `policies/`. Estimators see only a
  `LoggedBanditData`, an `(N, K)` evaluation-policy matrix, and (for DM/DR) an
  `(N, K)` predicted-reward matrix, so the same code path serves synthetic and real data.
* Policies are deterministic functions `context -> action distribution`. All
  randomness flows through explicit `numpy.random.Generator` arguments.
* Estimators do not perform I/O. The Open Bandit adapter does: it is a file
  reader with a typed result. It does not estimate a policy value.
* Model training is kept separate from estimation. Estimators (`dm`, `dr`) take
  pre-computed predicted-reward matrices. Cross-fitting (`cross_fit_reward_model`)
  manages the out-of-fold partitioning and model fitting.
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
| `ModelBasedEstimator` (Protocol) | `evaluation/estimators.py` | `(data, eval_dist, predicted_rewards, *, confidence_level) -> PolicyValueEstimate` |
| `RewardModel` (Protocol) | `evaluation/reward_models.py` | `fit(context, action, reward)`, `predict(context) -> (N, K)` |
| `cross_fit_reward_model` | `evaluation/cross_fitting.py` | `(data, model_factory, *, n_folds, seed) -> FloatArray` |
| `infer_from_influence_scores` | `evaluation/influence_scores.py` | `(scores, *, confidence_level) -> InfluenceScoreResult` |
| `bootstrap_evaluate` | `evaluation/bootstrap.py` | `(estimator, data, eval_dist, ..., *, refit, seed) -> BootstrapResult` |
| `diagnose` | `evaluation/diagnostics.py` | `(data, eval_dist, thresholds) -> DiagnosticReport` |
| `diagnose_interval` | `evaluation/diagnostics.py` | `(interval, se, thresholds) -> list[DiagnosticWarning]` |
| `EstimatorValidationConfig` | `experiments/config.py` | Pydantic model for Exp 001, `extra="forbid"`, frozen |
| `AdvancedEstimatorValidationConfig` | `experiments/config.py` | Pydantic model for Exp 003 (DM/DR/cross-fitting), frozen |
| `UncertaintyCalibrationConfig` | `experiments/config.py` | Pydantic model for Exp 004 (Uncertainty & Bootstrap), frozen |
| `run.json` | `experiments/metadata.py` | Run artifact, schema version `1` |

The evaluation-policy distribution and predicted-reward matrix are passed separately
from the logged data rather than stored inside it: one log is typically evaluated
against many candidate policies, and keeping the log immutable avoids accidental reuse
of a stale candidate or prediction matrix.

## Design decisions and trade-offs

* **NumPy arrays plus a frozen dataclass, not pandas/pydantic, for data.** Pydantic
  validation of large arrays is slow and adds little over explicit NumPy checks.
  Pydantic is used for configuration, where it gives precise error messages.
* **Functions for estimators, not classes.** Estimators (`ips`, `snips`, `dm`, `dr`)
  are pure functions. State lives in trained `RewardModel` instances or in
  cross-fitting orchestrators.
* **Separation of model fitting and estimation.** `dm` and `dr` do not train models.
  They accept an `(N, K)` float array of predicted rewards. Cross-fitting handles
  partitioning and out-of-fold inference.
* **Action-stratified linear reward models with fallback.** `StratifiedLinearRewardModel`
  fits separate scikit-learn models (`LogisticRegression` for binary rewards, `Ridge` for
  Gaussian/continuous rewards) per action. If an action is unobserved in a training fold,
  it falls back safely to the fold's global mean reward.
* **argparse, not click/typer.** Simple standard library command-line parsing without
  unnecessary dependencies.
* **JSON run artifacts, not MLflow.** One self-describing `run.json` per run
  contains the config, its hash, results, a results hash, warnings, timings, and
  provenance. MLflow can be added later as an optional sink.
* **Stable seed derivation.** Seeds are derived deterministically from
  `(seed, purpose, crc32(scenario name), n, replicate)` rather than from loop position,
  so adding or reordering scenarios leaves existing results unchanged.

## Planned (not implemented)

* `evaluation/`: cluster-robust intervals for real-world grouped data, shrinkage/regularized estimators
  (DR with shrinkage, CAB-DR).
* Columnar access to the full Open Bandit log, after its license is read.
  The small-sample adapter is `data/open_bandit.py`. `evaluation/obp_reference.py`
  runs upstream IPW/SNIPW/DM/DR from a local checkout. OBP is not a package dependency.
* `reversal/`: decision rule, perturbation sets, reversal search, fragility (M4).
* `reporting/`, `service/` (FastAPI), dashboard, containers (M5–M7).
