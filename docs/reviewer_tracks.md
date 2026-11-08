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
     uv run reversallab run-reversal --config configs/experiments/007_independent_reliability.yaml
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
     - Worker process terminated promptly (target p50 latency $\le 100\text{ ms}$).
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
