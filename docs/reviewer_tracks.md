# ReversalLab — Role-Specific External Review Protocols (v0.1.0-rc1)

This document provides structured, actionable review protocols tailored for three independent reviewer profiles:
- **Track A**: Statistical Machine Learning Researcher
- **Track B**: Python Package & Systems Engineer
- **Track C**: Backend & ML Infrastructure Engineer

All reviewers should reference the [External Review Guide](external_review_guide.md) for environment setup and reproduction commands.

---

## Track A: Statistical ML Researcher Review Protocol

### Focus Areas
- Mathematical soundness of estimator implementations (IPS, SNIPS, DM, DR).
- Validity of causal assumptions (unconfoundedness, positivity, pure covariate shift).
- Out-of-fold $K$-fold cross-fitting mechanics and target leakage prevention.
- Dual certification: Analytical 1-D mixture root-finding vs. forward worst-case robust mixture bounds.
- Confidence interval coverage, asymptotic properties, and honest positioning.

### Review Steps & Verification Tasks

1. **Estimator Implementation Audit**:
   - Inspect [`src/reversallab/estimators/`](../src/reversallab/estimators/):
     - `ips.py`: Verify Horvitz-Thompson unbiasedness under unconfoundedness and full propensity overlap.
     - `snips.py`: Verify Hajek self-normalized ratio estimation and asymptotic consistency.
     - `direct_method.py`: Verify reward model training and out-of-sample prediction.
     - `doubly_robust.py`: Verify that DR formulation combines conditional mean estimation with importance weighting:
       $$\hat{V}_{\text{DR}} = \frac{1}{N} \sum_{i=1}^N \left( \hat{\mu}(X_i, \pi) + \frac{\mathbb{I}(A_i = \pi(X_i))}{\pi_0(A_i \mid X_i)} (Y_i - \hat{\mu}(X_i, A_i)) \right)$$
   - Verify double-robustness consistency: If propensities are correct, DR is consistent even if $\hat{\mu}$ is misspecified.

2. **Out-of-Fold Cross-Fitting Verification**:
   - Inspect [`src/reversallab/estimators/cross_fitting.py`](../src/reversallab/estimators/cross_fitting.py):
     - Confirm that for each fold $k \in \{1, \dots, K\}$, the reward regression model $\hat{\mu}^{(-k)}$ is fitted exclusively on $\{i : i \notin \text{fold}_k\}$.
     - Confirm that predictions $\hat{\mu}^{(-k)}(X_i, a)$ for $i \in \text{fold}_k$ are strictly evaluated out-of-fold.
     - Verify test coverage in `tests/unit/test_cross_fitting.py`.

3. **Demographic Mixture Sensitivity & Equivalence**:
   - Inspect [`src/reversallab/reversal/analytical.py`](../src/reversallab/reversal/analytical.py):
     - Check the two-stratum mixture model: context space partitioned into $\mathcal{S}_0 = \{X : X_j < c\}$ and $\mathcal{S}_1 = \{X : X_j \ge c\}$ with baseline logging frequency $q_0$.
     - Verify the affine advantage function: $\Delta(q) = (1 - q)\Delta_0 + q\Delta_1$.
     - Verify exact closed-form root computation: $q^* = \frac{\delta - \Delta_0}{\Delta_1 - \Delta_0}$ and distance $d^* = |q^* - q_0|$.
     - Verify that in 1-D linear mixture shifts, checking $\min_{q \in \mathcal{Q}_\epsilon} \Delta(q) \ge \delta$ and checking $\epsilon^* \ge \epsilon_{\text{deploy}}$ produce mathematically identical binary decisions.
     - Verify that Bonferroni-corrected simultaneous confidence bands $\text{LCB}_{\text{sim}}(q)$ have minima strictly at interval endpoints due to concavity.

4. **Empirical Monte Carlo Reproduction**:
   - Execute the 5,000 Monte Carlo replication benchmark:
     ```bash
     uv run reversallab run-independent-reliability --config configs/experiments/007_independent_reliability.yaml
     ```
   - Verify that decision agreement between inverse reversal search and forward robust mixture evaluation is 100% across all replications.

5. **Scientific Claims & Limitations Audit**:
   - Inspect [`docs/assumptions.md`](assumptions.md) and [`docs/release_notes_v0.1.0.md`](release_notes_v0.1.0.md):
     - Confirm that asymptotic Wald intervals are clearly documented as normal approximations rather than finite-sample concentration bounds.
     - Confirm that transportability requires pure covariate shift (invariance of $\mathbb{P}(Y \mid X, A)$ across strata).
     - Confirm that ReversalLab makes no claim of novel physical laws or universal confidence bounds.

---

## Track B: Python Package Engineer Review Protocol

### Focus Areas
- Wheel and sdist installation in clean, unpolluted Python 3.11 and 3.12 environments.
- Packaging metadata, classifiers, dependencies, and optional extras isolation.
- Ergonomics of public SDK classes (`BanditDataset`, `PolicyComparison`, policies).
- Contract enforcement, input validation, and exception clarity.
- Command-line interface usability, entrypoints, and JSON schema version stability.

### Review Steps & Verification Tasks

1. **Clean Installation & Environment Isolation**:
   - Download the published release assets from GitHub:
     - `reversallab-0.1.0rc1-py3-none-any.whl`
     - `reversallab-0.1.0rc1.tar.gz`
     - `SHA256SUMS.txt`
   - Verify SHA-256 checksums:
     ```bash
     sha256sum -c SHA256SUMS.txt
     ```
   - In a completely fresh virtual environment (without cloning source):
     ```bash
     uv venv .review_env
     uv pip install --python .review_env reversallab-0.1.0rc1-py3-none-any.whl
     ```
   - Verify minimal base dependencies: Only `numpy`, `scikit-learn`, `pydantic`, `pyyaml` installed. No FastAPI or Uvicorn in core installation.

2. **Public SDK Ergonomics**:
   - Run a minimal evaluation script in the isolated environment:
     ```python
     import numpy as np
     import reversallab as rl

     # Verify version contracts
     assert rl.__version__ == "0.1.0rc1"
     assert rl.SDK_SCHEMA_VERSION == "1.0.0"

     # Verify dataset creation
     dataset = rl.BanditDataset.from_arrays(
         contexts=np.random.randn(1000, 2),
         actions=np.random.randint(0, 3, size=1000),
         rewards=np.random.randn(1000),
         propensities=np.full(1000, 1.0 / 3.0),
         n_actions=3,
     )

     # Verify policy comparison
     baseline = rl.UniformPolicy(n_actions=3)
     candidate = rl.LinearSoftmaxPolicy(weights=np.random.randn(2, 3), inverse_temperature=1.0)
     report = rl.PolicyComparison.evaluate(
         dataset=dataset,
         candidate_policy=candidate,
         baseline_policy=baseline,
         estimator="dr",
         margin=0.02,
         deploy_epsilon=0.10,
         seed=42,
     )
     print(report.summary())
     ```

3. **Input Validation & Exception Safety**:
   - Test invalid inputs and inspect error types:
     - Mismatched context/action lengths $\implies$ `ValueError` or `DataContractError`.
     - Non-finite values (`NaN`, `Inf`) in rewards or contexts $\implies$ rejected.
     - Zero or negative propensities ($p_i \le 0$) $\implies$ rejected immediately without silent clipping.
     - Empty dataset ($N = 0$) $\implies$ rejected.

4. **CLI Entrypoints & Script Wiring**:
   - Verify both console scripts:
     ```bash
     reversallab --version
     reversallab run-example --output report.json
     ```
   - Install service extras:
     ```bash
     uv pip install --python .review_env "reversallab-0.1.0rc1-py3-none-any.whl[service]"
     reversallab-service --version
     reversallab-service --help
     ```

5. **Type Safety & Metadata Integrity**:
   - Inspect wheel archive: Confirm presence of `reversallab/py.typed`.
   - Run `mypy` against code importing `reversallab` in strict mode to confirm zero type leakages.
   - Confirm frozen report contract schema `SDK_SCHEMA_VERSION == "1.0.0"` in exported JSON.

---

## Track C: Backend & ML Infrastructure Engineer Review Protocol

### Focus Areas
- Service lifecycle, startup, shutdown, and resource footprint.
- Persistent job storage using SQLite WAL mode and atomic JSON artifact persistence.
- Single-supervisor leader election using non-blocking advisory file locking (`supervisor.lock`).
- Process-isolated evaluation execution (`reversallab.service.worker`).
- Ephemeral generation token fencing (`execution_token`) and crash recovery.
- Targeted cancellation mechanics without sibling task interruption.
- Trusted-local single-tenant security boundary (Deployment Model A).

### Review Steps & Verification Tasks

1. **Local Service Startup & Health Probes**:
   - Install service extras and launch local daemon:
     ```bash
     uv run reversallab-service --host 127.0.0.1 --port 8000 --storage-dir ./var/reversallab
     ```
   - Probe standard endpoints:
     - `GET /health` $\implies$ `{"status": "healthy", "service": "reversallab"}`
     - `GET /version` $\implies$ `{"version": "0.1.0rc1", "schema_version": "1.0.0", "api_version": "v1"}`
     - `GET /metrics` $\implies$ Prometheus-style or JSON operational metrics snapshot.

2. **Synchronous vs. Persistent Asynchronous Pipeline**:
   - Submit synchronous evaluation to `POST /v1/evaluate/comparison`:
     - Verify response time for $N = 2500$ observations ($\approx 25\text{ ms}$).
   - Submit persistent asynchronous evaluation to `POST /v1/jobs/comparison`:
     - Verify immediate HTTP 202 response with assigned `job_id`, `status: PENDING`, and operational `payload_hash`.
     - Poll `GET /v1/jobs/{job_id}`: verify transition `PENDING` $\to$ `RUNNING` $\to$ `COMPLETED`.
     - Verify persistent storage: SQLite metadata in `./var/reversallab/jobs.db` and output artifact in `./var/reversallab/{job_id}/result.json`.

3. **Targeted Job Cancellation Audit**:
   - Submit a compute-intensive job ($N = 25{,}000$, $K$-fold cross-fitting, grid search).
   - Issue cancellation request: `DELETE /v1/jobs/{job_id}`.
   - Verify cancellation confirmation:
     - Worker process terminated promptly (measured reference baseline: p50 = $84.89\text{ ms}$, p95 = $105.81\text{ ms}$ on test hardware; actual cancellation confirmation latency depends on OS process management and scheduler dynamics).
     - SQLite status transitioned to `CANCELLED`.
     - No orphan child processes remaining in OS process table.
     - Sibling evaluations continue uninterrupted.

4. **Multi-Worker Advisory Locking & Crash Recovery**:
   - Start Uvicorn with multiple worker processes:
     ```bash
     uvicorn reversallab.service.app:app --host 127.0.0.1 --port 8000 --workers 4
     ```
   - Inspect lock acquisition: exactly one worker holds the advisory lock (`supervisor.lock`) and runs the job dispatcher. Sibling workers serve HTTP traffic normally.
   - Simulate supervisor crash: terminate the active leader worker (`SIGKILL` or crash injection).
   - Verify that another worker acquires `supervisor.lock` and startup crash recovery scans for interrupted `RUNNING` jobs, marking them `FAILED` with diagnostic recovery codes.

5. **Security Model A & Boundary Controls**:
   - Verify that service binds strictly to loopback (`127.0.0.1`) by default.
   - Test streaming request size limit: Attempt sending payload $> 10\text{ MB} \implies$ rejected with HTTP 413.
   - Test compute budget ceiling: Attempt submitting budget $> 50{,}000{,}000 \implies$ rejected with HTTP 422.
   - Test path traversal protection: Attempt `GET /v1/jobs/../../etc/passwd` or `..\..\windows` $\implies$ rejected with HTTP 422 / 400.
   - Confirm lack of multi-tenant authentication: Verify that documentation explicitly mandates reverse proxy (TLS, auth, rate limiting) for non-loopback exposure.

---

## 4. Role-Specific Onboarding Messages (Draft Invitations) & Task Checklists

> [!CAUTION]
> **Maintainer Instruction**: These draft invitation messages and task checklists are prepared for authorized dispatch only. Do NOT send invitations or invite GitHub collaborators without explicit project authorization. Completed external reviews are cataloged in [docs/review_intake_log.md](review_intake_log.md).

### Track A: Statistical ML Researcher Onboarding Message
```text
Subject: ReversalLab v0.1.0-rc1 External Research Review Invitation

Dear Colleague,

We would appreciate your independent expert review of ReversalLab (v0.1.0-rc1), an open-source Python library for statistically grounded off-policy evaluation (OPE) of contextual bandits and decision-reversal robustness analysis.

Repository & Prerelease:
https://github.com/koushik-ngd/reversal-lab/releases/tag/v0.1.0-rc1

Reviewer Protocol:
https://github.com/koushik-ngd/reversal-lab/blob/master/docs/reviewer_tracks.md#track-a-statistical-ml-researcher-review-protocol

We specifically invite your critical scrutiny of:
1. Causal identification and support assumptions for IPS, SNIPS, DM, and DR.
2. Out-of-fold K-fold cross-fitting mechanics to confirm elimination of target leakage.
3. Dual certification: verifying that inverse reversal search (distance-to-failure d*) and forward worst-case robust mixture evaluation yield identical binary decisions in 1-D linear mixtures.
4. Ensuring Wald confidence intervals and transportability under demographic shift are accurately qualified without overclaiming.

Please submit findings using our standardized review template (docs/reviewer_feedback_template.md). Thank you for your time and research rigor!
```

#### Track A Actionable Task Checklist
- [ ] **Step 1: Environment Setup**: Clone repository or unpack release tarball; ensure Python 3.11+ and `uv` are available. Run `uv sync`.
- [ ] **Step 2: Identification & Assumption Audit**: Read `docs/inference_assumptions.md` and `docs/assumptions.md`. Verify positivity checks, overlap diagnostics, and out-of-fold cross-fitting in `src/reversallab/advanced_estimators.py`.
- [ ] **Step 3: Dual Equivalence Verification**: Audit Theorem 1 in `docs/reviewer_tracks.md`. Confirm that inverse distance-to-reversal $d^*$ and forward robust bound $\Delta_{\text{worst}}(\epsilon)$ yield identical binary decisions on 1-D demographic mixtures.
- [ ] **Step 4: Statistical Benchmark Reproduction**:
  - `uv run reversallab validate-estimators --config configs/experiments/001_estimator_validation.yaml` (Exp 001 — basic estimators)
  - `uv run reversallab compare-reference --config configs/experiments/002_open_bandit_parity.yaml` (Exp 002 — reference baseline)
  - `uv run reversallab validate-advanced-estimators --config configs/experiments/003_advanced_estimators.yaml` (Exp 003 — cross-fitting)
  - `uv run reversallab calibrate-uncertainty --config configs/experiments/004_uncertainty_calibration.yaml` (Exp 004 — coverage calibration)
  - `uv run reversallab run-reversal --config configs/experiments/005_reversal_prototype.yaml` (Exp 005 — reversal search)
  - `uv run reversallab run-reliability --config configs/experiments/006_reversal_reliability.yaml` (Exp 006 — reliability)
  - `uv run reversallab run-independent-reliability --config configs/experiments/007_independent_reliability.yaml` (Exp 007 — independent seed)
  Confirm all benchmarks pass and numerical results match published tables within tolerance ($\le 10^{-4}$ rel / $\le 10^{-5}$ abs).
- [ ] **Step 5: Scope & Limitation Check**: Verify that docs accurately state asymptotic Wald bounds rather than claiming finite-sample concentration guarantees.
- [ ] **Step 6: Submit Review**: Complete sections in `docs/reviewer_feedback_template.md`.

### Track B: Python Packaging Engineer Onboarding Message
```text
Subject: ReversalLab v0.1.0-rc1 Packaging & SDK Usability Review Invitation

Dear Colleague,

We invite you to review the packaging, dependency hygiene, and public SDK usability of ReversalLab (v0.1.0-rc1).

Repository & Prerelease Assets:
https://github.com/koushik-ngd/reversal-lab/releases/tag/v0.1.0-rc1

Reviewer Protocol:
https://github.com/koushik-ngd/reversal-lab/blob/master/docs/reviewer_tracks.md#track-b-python-package-engineer-review-protocol

Key areas for evaluation:
1. Clean-room wheel installation in isolated Python 3.11/3.12 environments.
2. Dependency footprint (core dependencies vs optional service extras).
3. Public SDK ergonomics (BanditDataset, PolicyComparison, exception clarity).
4. Frozen JSON report contract schema (v1.0.0) and CLI entrypoint wiring.

Please document your evaluation in docs/reviewer_feedback_template.md. Thank you for helping ensure high-quality open-source software standards!
```

#### Track B Actionable Task Checklist
- [ ] **Step 1: Clean-Room Environment**: In an empty directory, create an isolated virtualenv (`python -m venv .clean_env && source .clean_env/bin/activate` or Windows equivalent).
- [ ] **Step 2: Asset Verification**: Download wheel and sdist from GitHub prerelease `v0.1.0-rc1`. Verify SHA-256 checksums match `SHA256SUMS.txt`.
- [ ] **Step 3: Installation**: Run `pip install reversallab-0.1.0rc1-py3-none-any.whl`. Confirm zero extraneous dependencies installed for core package.
- [ ] **Step 4: Public SDK API Check**:
  - `python -c "import reversallab; print(reversallab.__version__)"`
  - `python -c "from reversallab import BanditDataset, PolicyComparison, PolicyValueEvaluator, run_quickstart; print('Imports OK')"`
  - Verify helpful error messages when invalid dataset inputs are provided.
- [ ] **Step 5: CLI Entrypoint Verification**: Run `reversallab --help`. Verify subcommands and help documentation are clear and accurate.
- [ ] **Step 6: Schema Contract Verification**: Generate JSON report via `reversallab validate-estimators --output report.json` and verify compliance with schema `src/reversallab/service/schemas/evaluation_v1.0.0.json`.
- [ ] **Step 7: Submit Review**: Complete sections in `docs/reviewer_feedback_template.md`.

### Track C: Backend & ML Infrastructure Engineer Onboarding Message
```text
Subject: ReversalLab v0.1.0-rc1 Service Reliability & Infrastructure Review Invitation

Dear Colleague,

We invite your technical evaluation of ReversalLab's persistent asynchronous background job engine and FastAPI service architecture (v0.1.0-rc1).

Repository & Prerelease:
https://github.com/koushik-ngd/reversal-lab/releases/tag/v0.1.0-rc1

Reviewer Protocol:
https://github.com/koushik-ngd/reversal-lab/blob/master/docs/reviewer_tracks.md#track-c-backend--ml-infrastructure-engineer-review-protocol

Key areas for evaluation:
1. SQLite WAL metadata persistence, busy-timeout handling, and atomic artifact persistence.
2. Single-supervisor leader election using non-blocking advisory file locks across multiple Uvicorn workers.
3. Process-isolated child worker execution and targeted job cancellation.
4. Crash recovery mechanics and Deployment Model A (trusted-local / single-tenant loopback boundary).

Please record any anomalies or feedback in docs/reviewer_feedback_template.md. We appreciate your infrastructure expertise!
```

#### Track C Actionable Task Checklist
- [ ] **Step 1: Service Installation & Health**: Install with service extras (`pip install ".[service]"` or `uv sync --extra service`). Launch `reversallab-service --host 127.0.0.1 --port 8000`. Probe `GET /health`, `GET /version`, `GET /metrics`.
- [ ] **Step 2: Synchronous Evaluation**: Submit evaluation payload to `POST /v1/evaluate/comparison`. Confirm immediate response and schema validation.
- [ ] **Step 3: Asynchronous Persistent Job Flow**:
  - Submit compute payload to `POST /v1/jobs/comparison`. Verify HTTP 202, job ID, and `payload_hash`.
  - Poll `GET /v1/jobs/{job_id}` through `PENDING` $\to$ `RUNNING` $\to$ `COMPLETED`.
  - Inspect SQLite metadata in `./var/reversallab/jobs.db` and output artifact in `./var/reversallab/{job_id}/result.json`.
- [ ] **Step 4: Targeted Cancellation**:
  - Submit long-running job and issue `DELETE /v1/jobs/{job_id}`.
  - Verify prompt process termination (reference baseline: p50 = $84.89\text{ ms}$, p95 = $105.81\text{ ms}$ on test hardware).
  - Verify status transition to `CANCELLED` and absence of orphan worker processes.
- [ ] **Step 5: Multi-Worker Supervisor Lock & Crash Recovery**:
  - Launch with multiple workers (`uvicorn reversallab.service.app:app --workers 4`). Verify single supervisor lock holder.
  - Terminate leader worker and verify advisory lock transfer and recovery of interrupted jobs.
- [ ] **Step 6: Boundary & Security Auditing**:
  - Confirm default loopback binding (`127.0.0.1`).
  - Verify payload size limits ($> 10\text{ MB} \implies 413$) and budget limits ($> 50{,}000{,}000 \implies 422$).
  - Verify path traversal rejection on `/v1/jobs/..%2F..`.
- [ ] **Step 7: Submit Review**: Complete sections in `docs/reviewer_feedback_template.md`.
