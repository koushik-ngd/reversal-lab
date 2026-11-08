# ReversalLab

ReversalLab is an open-source Python library and evaluation service for statistically grounded off-policy evaluation (OPE) of contextual bandits, paired policy superiority testing, and decision-reversal robustness analysis under covariate shift.

Rather than relying solely on point estimates or standard confidence intervals that collapse under distribution shift, ReversalLab asks:

> **What is the smallest admissible change in population demographics or evaluation assumptions that reverses an evidence-based deployment decision?**

---

## 1. Scientific Purpose & Core Intuition

Off-policy evaluation enables safe policy comparison using historical logged bandit data without costly online A/B testing. However, real-world deployment decisions frequently fail because production user distributions drift from historical logging distributions.

ReversalLab formalizes **Decision-Reversal Auditing**:
- Instead of computing an advantage $\Delta = V(\pi_e) - V(\pi_b)$ at only the observed sample distribution $q_0$, ReversalLab assesses the stability of the deployment decision $\Delta(q) \ge \delta$ across demographic shifts.
- It calculates the exact **decision-reversal threshold** $\epsilon^*$ (minimum perturbation budget required to overturn deployment certification) and evaluates the worst-case advantage within a certified operational envelope $\mathcal{Q}_\epsilon$.
- If $\epsilon^* < \epsilon_{\text{deploy}}$, the policy pair is flagged for high reversal risk even if standard nominal tests claim statistical significance.

---

## 2. Mathematical Formulation

Let contextual bandit data $\mathcal{D} = \{(X_i, A_i, Y_i, p_i)\}_{i=1}^N$ be partitioned into two demographic strata $\mathcal{S}_0$ and $\mathcal{S}_1$ via a verified covariate split, with baseline logging frequency $q_0 = \mathbb{P}(X \in \mathcal{S}_1)$.

### Paired Policy Advantage
For candidate policy $\pi_e$ and baseline policy $\pi_b$, the stratum-specific policy advantage is estimated using out-of-fold Doubly Robust (DR) estimation with leakage-safe $K$-fold cross-fitting:
$$\Delta_s = \hat{V}_{\text{DR}}(\pi_e \mid \mathcal{S}_s) - \hat{V}_{\text{DR}}(\pi_b \mid \mathcal{S}_s), \quad s \in \{0, 1\}$$

Under demographic mixture perturbation $q = \mathbb{P}(X \in \mathcal{S}_1) \in [0, 1]$, the perturbed advantage is affine:
$$\Delta(q) = (1 - q)\Delta_0 + q\Delta_1$$

### Decision-Reversal Threshold $\epsilon^*$
Given an operational deployment margin $\delta \ge 0$, the decision-reversal root $q^*$ solves:
$$\Delta(q^*) = \delta \implies q^* = \frac{\delta - \Delta_0}{\Delta_1 - \Delta_0}$$
The minimum perturbation distance is:
$$d^* = |q^* - q_0|$$
A certified operational envelope $\mathcal{Q}_\epsilon = [q_0 - \epsilon_{\text{deploy}}, q_0 + \epsilon_{\text{deploy}}] \cap [0, 1]$ certifies deployment if and only if:
$$\min_{q \in \mathcal{Q}_\epsilon} \text{LCB}_{1-\alpha}(q) \ge \delta$$

### Simultaneous Confidence Bands
To ensure uniform statistical validity across the entire perturbation set, ReversalLab computes simultaneous confidence bands via Bonferroni-corrected Wald intervals across stratum endpoints:
$$\mathbb{P}\left(\forall q \in [0, 1], \quad \Delta(q) \in \left[\text{LCB}_{\text{sim}}(q), \text{UCB}_{\text{sim}}(q)\right]\right) \ge 1 - \alpha$$
Because the lower confidence bound $\text{LCB}(q)$ is strictly concave under mixture shifts, its minimum across any closed interval $\mathcal{Q}_\epsilon$ occurs at the interval boundary.

---

## 3. Installation

ReversalLab requires Python 3.11+ (recommended: Python 3.12 pinned via [uv](https://docs.astral.sh/uv/)):

```bash
# Clone the repository
git clone https://github.com/reversallab/reversal-lab.git
cd reversal-lab

# Install dependencies using uv
uv sync --frozen --all-groups
```

### Standalone Wheel Installation

To build and verify the standalone wheel in an isolated environment without source dependencies:

```bash
# Build pure Python wheel
uv build --wheel

# Create an isolated environment and install wheel via uv
uv venv .smoke_env
uv pip install --python .smoke_env dist/reversallab-0.1.0.dev0-py3-none-any.whl

# Verify execution in the isolated environment
uv run --no-project --python .smoke_env python -m reversallab.cli run-example --output .smoke_env/report.json
```

---

## 4. Python SDK Quickstart

The high-level Python SDK exposes `BanditDataset`, `PolicyComparison`, and policy definitions:

```python
import numpy as np
import reversallab as rl

# 1. Load or synthesize validated logged bandit data
dataset = rl.BanditDataset.from_arrays(
    contexts=np.random.randn(1000, 2),
    actions=np.random.randint(0, 3, size=1000),
    rewards=np.random.randn(1000),
    propensities=np.full(1000, 1.0 / 3.0),
    n_actions=3,
)

# 2. Define baseline and candidate policies
baseline = rl.UniformPolicy(n_actions=3)
candidate = rl.LinearSoftmaxPolicy(
    weights=np.random.randn(2, 3),
    inverse_temperature=1.0,
)

# 3. Evaluate paired advantage and decision robustness
report = rl.PolicyComparison.evaluate(
    dataset=dataset,
    candidate_policy=candidate,
    baseline_policy=baseline,
    estimator="dr",
    margin=0.02,  # Minimum deployment advantage required (+2%)
    deploy_epsilon=0.10,  # Certified demographic envelope [q0 - 0.10, q0 + 0.10]
    max_epsilon=0.40,  # Maximum search radius
    cross_fitting_folds=3,
    seed=42,
)

# 4. Inspect ASCII audit report and export JSON
print(report.summary())
report.save("runs/decision_audit.json")
```

---

## 5. CLI Quickstart

ReversalLab includes an end-to-end command-line interface for reproducible evaluations:

```bash
# Run end-to-end synthetic decision-reversal demonstration
uv run reversallab run-example --output runs/quickstart_report.json

# Launch the FastAPI REST service
uv run reversallab-service --host 127.0.0.1 --port 8000 --storage-dir ./var/reversallab

# Execute research experiment reproduction
uv run reversallab run-reversal --config configs/experiments/007_independent_reliability.yaml
```

---

## 6. Interactive Jupyter Notebook Guide

A fully reproducible, executable Jupyter tutorial is located at:
[`notebooks/quickstart_decision_reversal.ipynb`](notebooks/quickstart_decision_reversal.ipynb)

The notebook demonstrates:
1. Synthetic environment initialization and logged bandit data generation.
2. Leakage-safe Doubly Robust cross-fitting.
3. Diagnostic gate evaluation (propensity overlap, effective sample size, weight concentration).
4. Analytical decision-reversal search and margin-crossing calculation.
5. Dual certification via simultaneous confidence bands and forward worst-case robust mixture evaluation.

To launch the notebook interactively:

```bash
uv run jupyter lab notebooks/quickstart_decision_reversal.ipynb
```

The notebook is also automatically verified in CI via a genuine Jupyter kernel execution test (`tests/integration/test_notebook_kernel.py`).

---

## 7. Persistent Background Job Service Architecture

For large datasets ($N > 25{,}000$) and CPU-intensive grid scans, ReversalLab provides a persistent, bounded asynchronous evaluation engine (`reversallab.service`):

- **FastAPI HTTP Layer**: Asynchronous non-blocking endpoints (`POST /v1/jobs/comparison`, `GET /v1/jobs/{job_id}`, `DELETE /v1/jobs/{job_id}`).
- **SQLite Metadata Persistence**: Transactional state transitions in WAL mode with busy timeout handling.
- **Single-Supervisor Advisory Lock**: Cross-worker leader election using advisory file locks (`supervisor.lock`), ensuring exactly one supervisor manages child processes regardless of Uvicorn worker count.
- **Dedicated Child Worker Processes**: Evaluators execute in isolated child processes (`reversallab.service.worker`), preventing CPU starvation of the HTTP event loop.
- **Ephemeral Generation Tokens**: Every claimed job receives an ephemeral UUID4 `execution_token`, fencing against stale worker commits across supervisor restarts or leader changes.
- **Decoupled Execution vs. Artifact State**: Completed compute remains permanently immutable (`status: COMPLETED`), while artifact availability is independently reported via `artifact_available: bool` with diagnostic codes (`ARTIFACT_EXPIRED`, `ARTIFACT_CORRUPTED`).
- **Targeted Cancellation**: Two-phase worker fencing enables immediate, verified process termination without interrupting sibling jobs.

For detailed architecture diagrams and data flows, see [`docs/architecture_jobs.md`](docs/architecture_jobs.md).

---

## 8. Trust Boundary & Security Model (Model A)

> [!IMPORTANT]
> **Deployment Model A: Trusted-Local / Single-Tenant Boundary**
>
> ReversalLab is designed and certified for single-tenant, trusted execution environments:
> - **Default Loopback Binding**: The service binds exclusively to `127.0.0.1` or runs inside private, isolated container networks.
> - **No Built-in Authentication**: The service does not implement multi-tenant authentication or authorization. `job_id` and `payload_hash` are operational routing tokens, not access credentials.
> - **Public Exposure Requirements**: If exposed beyond loopback or private networks, operators **MUST** place the service behind an authenticating reverse proxy or API gateway providing TLS termination, client authentication, and rate limiting.
> - **Persistent Volume Required**: Docker containers must mount a persistent volume to `/home/reversallab/storage/jobs` to preserve SQLite metadata and report artifacts across container restarts.

See [`SECURITY.md`](SECURITY.md) for full vulnerability reporting policies and threat model disclosures.

---

## 9. Verified Operational Performance & Capacity Envelope

ReversalLab's operational capacity has been verified under load and operational stress benchmarks:

| Metric / Parameter | Synchronous Pipeline | Persistent Background Job Engine |
| :--- | :--- | :--- |
| **Max Compute Budget ($N \cdot D \cdot K \cdot \text{folds}$)** | $10{,}000{,}000$ | $50{,}000{,}000$ |
| **Request Size Ceiling** | 10 MB (Streaming check) | 10 MB (Streaming check) |
| **Worker Concurrency Limit** | Dynamic threadpool | Strictly Bounded ($W = 2$) |
| **Admission Queue Capacity** | Socket backlog | Strict 50 pending jobs |
| **Execution Timeout** | Client socket timeout | 300.0 seconds (Reaped automatically) |
| **Storage Hard Quota** | In-memory | 200 MB (Automatic 24h retention purge) |
| **Submission Latency ($\text{p50}$ / $\text{p95}$)** | $64\text{ ms} / 140\text{ ms}$ | $114.98\text{ ms} / 158.05\text{ ms}$ |
| **Pure Evaluation Time ($N=2500$)** | $\approx 22.5\text{ ms}$ | $\approx 22.5\text{ ms}$ |
| **Process Startup Baseline** | 0 ms | $\approx 1{,}690\text{ ms}$ (Windows process spawn) |
| **Cancellation Confirmation Latency** | N/A | $\text{p50} = 84.89\text{ ms}, \text{p95} = 105.81\text{ ms}$ |
| **Process Tree RSS Footprint** | $\approx 85\text{ MB}$ | Peak $423.16\text{ MB}$ under burst |

---

## 10. Reproducible Research Experiments

ReversalLab was developed through seven audited empirical milestones:

| Experiment | Title | Description | Reproduction Command |
|---|---|---|---|
| **001** | Estimator Validation | Finite-sample bias, variance, and MSE of IPS and SNIPS across sample sizes. | `uv run reversallab validate-estimators --config configs/experiments/001_estimator_validation.yaml` |
| **002** | Reference Parity | Numerical parity comparison against Open Bandit Pipeline (OBP) reference estimators. | `uv run reversallab compare-reference --config configs/experiments/002_open_bandit_parity.yaml` |
| **003** | Advanced Estimators | DM and DR evaluation with leakage-safe K-fold cross-fitting and misspecification tests. | `uv run reversallab validate-estimators --config configs/experiments/003_advanced_estimators.yaml` |
| **004** | Uncertainty Calibration | Coverage and reliability audit of Wald intervals, influence scores, and bootstrap resamples. | `uv run reversallab validate-estimators --config configs/experiments/004_uncertainty_calibration.yaml` |
| **005** | Decision-Reversal Prototype | Prototype 1-D context mixture perturbation search and margin-crossing detection. | `uv run reversallab run-reversal --config configs/experiments/005_reversal_prototype.yaml` |
| **006** | Mathematical Validation | Analytical reversal root-finding and simultaneous confidence band certification. | `uv run reversallab run-reversal --config configs/experiments/006_reversal_validation.yaml` |
| **007** | Independent Reliability | 5,000 Monte Carlo replications comparing inverse reversal distance and forward robust baselines. | `uv run reversallab run-reversal --config configs/experiments/007_independent_reliability.yaml` |

---

## 11. Documentation Directory

- [`docs/quickstart.md`](docs/quickstart.md): Practitioner guide and tutorial.
- [`docs/api_reference.md`](docs/api_reference.md): Public SDK API contracts.
- [`docs/architecture_jobs.md`](docs/architecture_jobs.md): Background job engine specification.
- [`docs/service.md`](docs/service.md): FastAPI REST service endpoints and load testing.
- [`docs/decision_reliability_audit.md`](docs/decision_reliability_audit.md): Complete mathematical and empirical closeout audit of Experiment 007.
- [`docs/methodology.md`](docs/methodology.md): Formal definitions of estimators and mixture perturbations.
- [`docs/data_contract.md`](docs/data_contract.md): Field-by-field array contracts and validation rules.
- [`docs/assumptions.md`](docs/assumptions.md): Supported statistical claims vs. deliberate limitations.
- [`SECURITY.md`](SECURITY.md): Vulnerability reporting policy and Model A deployment boundary.

---

## 12. Scientific Scope & Limitations

- **Supported:** Asymptotically valid paired difference confidence intervals via Doubly Robust estimation with K-fold cross-fitting, under unconfoundedness, positive propensity support, and correctly specified logging probabilities.
- **Supported:** Exact boundary minima for forward worst-case bounds under 1-D mixture perturbations due to concavity of the lower confidence bound.
- **Equivalence Note:** In 1-D two-stratum mixture shifts, inverse reversal thresholding and forward robust-value checks yield identical binary deployment decisions at budget $\epsilon_{\text{deploy}}$. Interpretability advantages of distance reporting are proposed conceptual benefits, not experimentally established user advantages.
- **Limitations:** Does not establish safety against unobserved confounding, continuous non-parametric density drift, or multidimensional demographic shifts beyond the evaluated partition.

---

## License

MIT. See [LICENSE](LICENSE).
