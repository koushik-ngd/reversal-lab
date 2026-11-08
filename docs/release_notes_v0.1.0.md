# ReversalLab v0.1.0-rc1 Release Notes

> **Version**: `0.1.0rc1` (Prerelease Candidate 1)  
> **Report Schema Version**: `1.0.0`  
> **Date**: November 8, 2026  
> **Repository**: [https://github.com/koushik-ngd/reversal-lab](https://github.com/koushik-ngd/reversal-lab)

ReversalLab is an open-source Python library and evaluation service for statistically grounded off-policy evaluation (OPE) of contextual bandits, paired policy superiority testing, and decision-reversal robustness analysis under covariate shift.

---

## 1. Executive Summary

Standard off-policy evaluation methods estimate policy value or paired policy advantage under the historical logging distribution. However, deployment decisions made from these point estimates frequently fail in production when population demographics or user contexts drift away from the logging distribution.

ReversalLab answers the question:
> **What is the smallest admissible change in population demographics or evaluation assumptions that reverses an evidence-based deployment decision?**

By framing robust policy comparison through **decision-reversal auditing**, ReversalLab provides practitioners with an interpretable distance-to-failure metric ($d^* = |q^* - q_0|$) alongside traditional forward worst-case robust bounds and simultaneous confidence bands.

---

## 2. Scientific Architecture & Theoretical Grounding

### Core Estimators
- **Inverse Propensity Scoring (IPS)**: Asymptotically unbiased under unconfoundedness and full propensity overlap; high variance when propensities are small.
- **Self-Normalized IPS (SNIPS)**: Ratio estimator trading finite-sample bias for reduced variance under propensity scale misspecification.
- **Direct Method (DM)**: Fits a reward regression model $\hat{\mu}(x, a)$; consistent only when the regression model is correctly specified.
- **Doubly Robust (DR)**: Combines regression reward models with propensity weighting. Possesses the double-robustness property: consistent if *either* the propensity model or the reward regression model is correctly specified.

### Leakage-Safe K-Fold Cross-Fitting
- Out-of-fold training ($K \ge 2$) guarantees that reward regression models $\hat{\mu}^{(-k)}$ are trained strictly on data excluding fold $k$. This prevents optimism and target leakage bias during evaluation.

### Paired Comparison & Uncertainty
- Evaluates the policy advantage $\Delta = V(\pi_e) - V(\pi_b)$ directly from influence score differences $\psi_i(\pi_e) - \psi_i(\pi_b)$.
- Computes standard Wald confidence intervals and tests against an operational deployment margin $\delta \ge 0$.

### Decision-Reversal Root Finding & Robust Mixture Baseline
- Under a 1-D demographic mixture perturbation between strata $\mathcal{S}_0$ and $\mathcal{S}_1$, the perturbed advantage is affine:
  $$\Delta(q) = (1 - q)\Delta_0 + q\Delta_1$$
- Given an operational deployment margin $\delta$, the reversal root $q^*$ solves $\Delta(q^*) = \delta$. The distance to reversal is $d^* = |q^* - q_0|$.
- **Equivalence Note**: In 1-D two-stratum mixture shifts, inverse reversal thresholding ($\epsilon^* \ge \epsilon_{\text{deploy}}$) and forward worst-case advantage checking ($\min_{q \in \mathcal{Q}_\epsilon} \Delta(q) \ge \delta$) yield mathematically identical binary deployment decisions. ReversalLab reports distance-to-failure as an intuitive metric, not as an independent physical law.
- **Simultaneous Confidence Bands**: Computes Bonferroni-adjusted Wald intervals across stratum boundaries certifying $\min_{q \in \mathcal{Q}_\epsilon} \text{LCB}(q) \ge \delta$. Strict concavity of the lower bound ensures that the minimum always lies on the boundary of the certified interval $\mathcal{Q}_\epsilon$.

---

## 3. Library & Service Capabilities

1. **Public Python SDK**:
   - High-level classes: `BanditDataset`, `PolicyValueEvaluator`, `PolicyComparison`.
   - Policy primitives: `UniformPolicy`, `LinearSoftmaxPolicy`, `EpsilonGreedyLinearPolicy`.
   - Schema-versioned JSON report export (`1.0.0`) and clean ASCII summaries.
2. **Command-Line Interface**:
   - `reversallab run-example`: Complete synthetic workflow demonstration.
   - `reversallab serve` / `reversallab-service`: Local FastAPI evaluation service runner.
   - `reversallab validate-estimators`, `run-reversal`: Full research experiment reproductions.
3. **Interactive Jupyter Tutorial**:
   - [`notebooks/quickstart_decision_reversal.ipynb`](notebooks/quickstart_decision_reversal.ipynb): Step-by-step walkthrough covering data ingestion, cross-fitting, diagnostic gates, reversal search, and simultaneous bands.
4. **Persistent Asynchronous Evaluation Engine**:
   - Minimal single-node architecture with zero external broker dependencies (no Redis, Celery, or RabbitMQ).
   - SQLite WAL metadata persistence with busy timeout handling.
   - Advisory file locking (`supervisor.lock`) electing a single supervisor across multiple Uvicorn workers.
   - Process-isolated child execution (`reversallab.service.worker`).
   - Ephemeral generation tokens (`execution_token`) fencing against stale worker commits across restarts or leader transitions.
   - Targeted cancellation terminating active child processes without interrupting sibling tasks.
   - Decoupled terminal execution status (`status: COMPLETED`) from physical storage availability (`artifact_available: bool`).

---

## 4. Packaging & Versioning

- **Package Version**: `0.1.0rc1` (PEP 440 compliant prerelease candidate).
- **Report Schema Version**: `1.0.0` (frozen, versioned contract for audit artifacts).
- **Build Backend**: Hatchling (`hatchling>=1.26`).
- **Python Support**: Python 3.11 and 3.12.
- **Standalone Wheel**: Verified clean installation into isolated virtual environments with zero source dependencies.

---

## 5. Deployment Model A & Trust Boundary

ReversalLab is explicitly designed and certified for **Deployment Model A: Trusted-Local Single-User / Single-Tenant Execution**:
- Default binding to loopback (`127.0.0.1`).
- No built-in authentication or multi-tenant authorization. `job_id` and `payload_hash` are operational routing tokens, not access credentials.
- Any public or multi-tenant deployment **MUST** place the service behind an authenticating reverse proxy or API gateway providing TLS termination, client authentication, and rate limiting.
- Docker containers must mount a persistent volume to `/home/reversallab/storage/jobs` to preserve SQLite metadata and report artifacts across container restarts.

Full vulnerability reporting policies are documented in [`SECURITY.md`](SECURITY.md).

---

## 6. Verified Operational Performance

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

## 7. Scientific Grounding, Scope & Distinctions

### Established Foundations vs. Engineering Contributions
- **Established Off-Policy Estimators**: Inverse Propensity Scoring (Horvitz & Thompson, 1952), Self-Normalized IPS (Hajek, 1964), Direct Method regression modeling, Doubly Robust estimation (Cassy & Robins, 1994; Bang & Robins, 2005), and out-of-fold cross-fitting (Chernozhukov et al., 2018) are foundational techniques from the causal inference and econometrics literature.
- **ReversalLab Engineering Contributions**: ReversalLab integrates these estimators into an automated, auditable evaluation framework featuring multi-factor diagnostic gating (propensity overlap, Kish ESS, weight concentration), closed-form 1-D demographic mixture reversal search, Bonferroni-corrected simultaneous confidence bands, and a bounded persistent execution engine.

### Empirical Findings vs. Open Hypotheses
- **Empirical Simulation Results**: In synthetic benchmarks (Experiments 001–007), $K$-fold cross-fitting prevented optimism bias from in-sample reward overfit, and analytical root-finding reached 100% binary decision agreement with forward robust mixture evaluation across 5,000 Monte Carlo replications in 1-D linear mixtures.
- **Open Scientific Hypotheses**: Whether distance-to-failure metrics ($d^* = |q^* - q_0|$) improve practitioner decision alignment or reduce erroneous ML deployments compared to standard worst-case bounds in live industrial A/B environments is an open human-factors research question. Generalization of closed-form reversal search to non-convex, high-dimensional uncertainty sets remains an open theoretical problem.

### Confidence Interval Limitations & Transportability
- **Asymptotic Limitations**: All reported confidence intervals are asymptotic Wald intervals based on the Central Limit Theorem and empirical influence scores. They do not constitute finite-sample concentration guarantees; in the presence of heavy-tailed importance weights or extreme overlap violations, empirical coverage may fall below nominal $1 - \alpha$.
- **Transportability Requirements**: Policy advantage guarantees under demographic mixture shift require **pure covariate shift**: the conditional outcome distribution $\mathbb{P}(Y \mid X, A)$ must remain strictly invariant across strata, and propensity positivity must hold in the target population. If unobserved confounding or concept drift occurs, guarantees do not apply.

### Cross-Platform Numerical Tolerances
- Point estimates and interval bounds are deterministic under fixed random seeds on identical hardware, OS, and Python environments. Across different operating systems (Windows, Linux, macOS) or linear algebra libraries (OpenBLAS, Intel MKL, Apple Accelerate), floating-point accumulation orders may cause slight numerical discrepancies. ReversalLab guarantees agreement within standard numerical tolerances ($\le 10^{-4}$ relative tolerance or $\le 10^{-5}$ absolute tolerance), rather than universal bit-for-bit identity across disparate platforms.

### Service Security & Deployment Model A
- ReversalLab is explicitly designed and certified for **Deployment Model A: Trusted-Local / Single-Tenant Execution**. It is NOT internet-ready or multi-tenant. The service binds exclusively to loopback (`127.0.0.1`) by default. Any public or multi-tenant deployment must operate behind an authenticating reverse proxy providing TLS termination, authorization, and rate limiting.

---

## 8. Known Scientific & Engineering Limitations

1. **Unobserved Confounding**: ReversalLab assumes unconfoundedness (conditional ignorability) in historical logged data. If unobserved variables influenced action selection, all estimators remain asymptotically biased.
2. **Overlap Violations**: If target policies select actions where logging propensity $p_i \to 0$, importance weights diverge. The multi-factor diagnostic gates flag these conditions, but cannot correct for structural lack of support.
3. **Dimensionality of Perturbation**: The current analytical reversal engine evaluates 1-D demographic mixture shifts across a two-stratum partition. It does not evaluate continuous non-parametric density drift or high-dimensional adversarial Wasserstein perturbations.
4. **Single-Node Ceiling**: The background job service is strictly single-node. Scaling beyond a single machine requires an external queue or distributed scheduler.

---

## 9. Reviewer Evaluation Prompts

External reviewers evaluating ReversalLab v0.1.0-rc1 are encouraged to inspect:
1. Does the out-of-fold $K$-fold cross-fitting implementation in `src/reversallab/estimators/cross_fitting.py` strictly eliminate target leakage under synthetic misspecification?
2. Are the diagnostic gate thresholds ($\tau = 10^{-4}$, $\text{ESS}/N \ge 0.05$, weight share $\le 0.20$) appropriate across diverse real-world recommender settings?
3. Is the mathematical equivalence between inverse reversal search and forward robust mixture evaluation clearly presented without overclaiming theoretical novelty?
4. Does the decoupled state machine in `src/reversallab/service/` adequately prevent state corruption under crash and restart scenarios?
