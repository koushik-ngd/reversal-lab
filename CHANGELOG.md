# Changelog

All notable changes to the ReversalLab project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [PEP 440](https://peps.python.org/pep-0440/).

---

## [0.1.0-rc1] - 2026-11-08

### Added

#### Core Off-Policy Evaluation & Statistical Estimators
- **Standard & Advanced OPE Estimators**:
  - Inverse Propensity Scoring (`ips`) with unbiasedness under unconfoundedness and full support.
  - Self-Normalized Inverse Propensity Scoring (`snips`) for variance reduction under propensity scale misspecification.
  - Direct Method (`dm`) via regression-based reward modeling.
  - Doubly Robust (`dr`) estimation combining reward modeling with propensity weighting.
- **Leakage-Safe Cross-Fitting**:
  - $K$-fold cross-fitting ($K \ge 2$) guaranteeing that reward regression models are fitted strictly out-of-fold, preventing in-sample overfitting and optimism bias.
- **Paired Policy Comparison**:
  - Advantage formulation $\Delta = V(\pi_e) - V(\pi_b)$ estimated via empirical influence score differences.
  - Asymptotic Wald confidence intervals at declared confidence levels ($1 - \alpha$).
  - Operational deployment margin testing ($\delta \ge 0$).

#### Multi-Factor Diagnostic Gates
- **Propensity Overlap Gate**: Verifies $\min_i p_i \ge \tau$ across active target policy evaluations.
- **Effective Sample Size Gate**: Kish's $\text{ESS} / N \ge 0.05$ to guard against sample collapse.
- **Weight Concentration Gate**: Maximum individual importance weight share $\max_i w_i / \sum_i w_i \le 0.20$.
- **Automated Certification**: Policy recommendations are gated to `INCONCLUSIVE` if any diagnostic fails.

#### Decision-Reversal Robustness Engine
- **Two-Stratum Demographic Mixture Model**: Context space partitioning $\mathcal{S}_0 = \{X : X_j < c\}$ and $\mathcal{S}_1 = \{X : X_j \ge c\}$ with baseline proportion $q_0$.
- **Analytical Root-Finding**: Exact computation of the decision-reversal root $q^* = (\delta - \Delta_0) / (\Delta_1 - \Delta_0)$ and minimum perturbation distance $d^* = |q^* - q_0|$.
- **Dual Statistical Certification**:
  - Simultaneous confidence bands via Bonferroni-corrected Wald intervals across stratum endpoints certifying $\min_{q \in \mathcal{Q}_\epsilon} \text{LCB}_{\text{sim}}(q) \ge \delta$.
  - Forward worst-case robust mixture baseline evaluating $\Delta_{\text{worst}}(\epsilon) = \min_{q \in \mathcal{Q}_\epsilon} \Delta(q)$.
- **Strict Mathematical Positioning**: Documented the binary decision equivalence between inverse reversal search and forward robust bounds in 1-D mixture shifts, establishing inverse reversal as an interpretable distance-to-failure metric rather than a novel physical law.

#### Public Python SDK & JSON Contract
- High-level ergonomic interfaces: [`BanditDataset`](docs/api_reference.md#1-data-contract-banditdataset), [`PolicyValueEvaluator`](docs/api_reference.md#2-single-policy-evaluation-policyvalueevaluator), and [`PolicyComparison`](docs/api_reference.md#3-paired-comparison--decision-reversal-policycomparison).
- Policy implementations: `UniformPolicy`, `LinearSoftmaxPolicy`, `EpsilonGreedyLinearPolicy`.
- Frozen machine-readable JSON report schema (`v1.0.0`) with ASCII summary formatting.
- Explicit distinction between package version (`0.1.0.dev0` / `0.1.0rc1`) and JSON report schema version (`1.0.0`).

#### Command-Line Interface & Tutorials
- Unified CLI (`reversallab`) supporting research reproduction (`validate-estimators`, `run-reversal`, `report`, `run-example`, `serve`).
- Standalone service entrypoint (`reversallab-service`).
- Fully executable Jupyter tutorial notebook ([`notebooks/quickstart_decision_reversal.ipynb`](notebooks/quickstart_decision_reversal.ipynb)) verified via headless kernel regression tests.

#### Persistent Background Evaluation Service
- **FastAPI HTTP Pipeline**: Asynchronous endpoints (`POST /v1/jobs/comparison`, `GET /v1/jobs/{job_id}`, `DELETE /v1/jobs/{job_id}`).
- **SQLite Metadata Persistence**: Transactional state tracking in WAL mode with busy timeout handling.
- **Single-Supervisor Leadership**: Non-blocking advisory file locking (`supervisor.lock`) electing exactly one active dispatcher across multiple Uvicorn worker processes.
- **Dedicated Process Isolation**: CPU-intensive evaluations run in isolated child worker processes.
- **Ephemeral Generation Tokens**: Ephemeral UUID4 `execution_token` fencing against stale worker commits across supervisor restarts or leader changes.
- **Decoupled State Machine**: Permanent terminal execution outcome (`status: COMPLETED`) decoupled from physical artifact state (`artifact_available: bool`) with explicit diagnostic error codes (`ARTIFACT_EXPIRED`, `ARTIFACT_CORRUPTED`).
- **Targeted Cancellation**: Two-phase worker fencing allowing immediate child process termination without interrupting sibling evaluations.

#### Security & Quality Assurance
- **Deployment Model A**: Formally specified trusted-local / single-tenant security boundary with loopback defaults.
- **Vulnerability Policy**: Documented private disclosure procedure in [`SECURITY.md`](SECURITY.md).
- **Test Coverage**: 289 automated tests (255 fast integration/unit + 34 statistical Monte Carlo checks).
- **Remote CI Matrix**: GitHub Actions pipeline validating Python 3.11 and 3.12 with Ruff, strict Mypy, and isolated wheel installations.
- **Containerization**: Secure, non-root Docker packaging (`reversallab:reversallab`, UID 10001) with persistent volume contracts.

#### Audited Empirical Experiments
- Full reproduction configurations for Experiments 001–007:
  - Experiment 001: Estimator validation (bias/variance across sample sizes).
  - Experiment 002: Numerical parity against Open Bandit Pipeline (OBP).
  - Experiment 003: Advanced estimators (DM, DR with cross-fitting).
  - Experiment 004: Uncertainty calibration and coverage.
  - Experiment 005: Prototype reversal search.
  - Experiment 006: Analytical root-finding & simultaneous band validation.
  - Experiment 007: 5,000 Monte Carlo replications verifying exact decision agreement between inverse reversal search and forward robust mixture evaluation.
