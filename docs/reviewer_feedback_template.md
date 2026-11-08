# ReversalLab — External Reviewer Feedback Template (v0.1.0-rc1)

Please use this standardized template to document your evaluation of ReversalLab v0.1.0-rc1. Submit feedback via GitHub Issues or as a pull request against the `docs/reviews/` directory.

> [!NOTE]
> This template is for genuine external reviewers. Per project validation rules, no reviewer feedback should be fabricated or claimed without actual participant evaluation.

---

## 1. Reviewer Information

- **Reviewer Identifier / GitHub Handle**:
- **Evaluation Track**: [ ] Track A (Statistical ML) | [ ] Track B (Python Packaging) | [ ] Track C (ML Infrastructure)
- **Review Date**:
- **Affiliation / Role**: (Optional)

---

## 2. Environment & Installation

- **Operating System**: (e.g., Ubuntu 22.04 LTS, macOS Sonoma 14.5, Windows 11 23H2)
- **CPU Architecture**: (e.g., x86_64, ARM64 / Apple Silicon)
- **Python Version**: (e.g., 3.11.9, 3.12.3)
- **BLAS / Linear Algebra Backend**: (e.g., OpenBLAS, Intel MKL, Apple Accelerate)
- **Installation Method**:
  - [ ] Standalone Wheel (`reversallab-0.1.0rc1-py3-none-any.whl`)
  - [ ] Source Distribution (`reversallab-0.1.0rc1.tar.gz`)
  - [ ] Git Checkout (`git clone https://github.com/koushik-ngd/reversal-lab.git`)
- **Package Manager**: (e.g., `uv`, `pip`, `conda`, `poetry`)

---

## 3. Review Tasks Attempted & Reproducibility Matrix

| Review Task | Attempted? | Result (Pass / Fail / Partial) | Observed Numerical Outputs / Tolerances |
| :--- | :---: | :---: | :--- |
| **Clean Wheel Installation** | [ ] | | Installed cleanly in isolated env without source deps. |
| **SDK Quickstart Execution** | [ ] | | Point estimate $\Delta$: `+0.2489`, 95% CI: `[+0.2325, +0.2653]`. |
| **CLI Quickstart (`run-example`)** | [ ] | | JSON report generated with schema `1.0.0`. |
| **Jupyter Notebook Tutorial** | [ ] | | All 16 cells executed cleanly in fresh kernel. |
| **Estimator Validation (Exp 001)** | [ ] | | Bias/variance matches reference properties. |
| **OBP Parity Benchmark (Exp 002)** | [ ] | | Parity within numerical tolerance ($< 10^{-6}$). |
| **Reliability MC Study (Exp 007)** | [ ] | | 100% decision agreement across replications. |
| **Service Execution & Probes** | [ ] | | `/health`, `/version`, `/metrics` functional. |
| **Asynchronous Job Lifecycle** | [ ] | | `PENDING` $\to$ `RUNNING` $\to$ `COMPLETED` transitions. |
| **Targeted Job Cancellation** | [ ] | | Child process killed promptly without sibling impact. |
| **Advisory Lock Election** | [ ] | | Single supervisor across multiple Uvicorn workers. |

---

## 4. Issues & Anomalies Encountered

Please record any unexpected behavior, exceptions, numerical discrepancies, or documentation gaps.

### Issue 1
- **Severity**: [ ] P0 Blocker | [ ] P1 Correctness / Security | [ ] P2 Ergonomics / Docs | [ ] P3 Polish
- **Component**: (e.g., `estimators`, `reversal`, `service`, `packaging`, `cli`)
- **Description**:
- **Steps to Reproduce**:
- **Observed Behavior**:
- **Expected Behavior**:
- **Error Traceback / Output**: (If applicable)
- **Suggested Resolution**:

### Issue 2
- **Severity**: [ ] P0 Blocker | [ ] P1 Correctness / Security | [ ] P2 Ergonomics / Docs | [ ] P3 Polish
- **Component**:
- **Description**:
- **Steps to Reproduce**:
- **Observed Behavior**:
- **Expected Behavior**:
- **Error Traceback / Output**:
- **Suggested Resolution**:

---

## 5. Scientific & Theoretical Feedback (Track A)

- **Causal Assumptions**: (Are unconfoundedness, overlap positivity, and pure covariate shift clearly communicated?)
- **Estimator Properties**: (Did IPS, SNIPS, DM, and DR behave in accordance with published causal theory?)
- **Out-of-Fold Cross-Fitting**: (Did the $K$-fold implementation strictly prevent target leakage in your inspection?)
- **Decision-Reversal Sensitivity**: (Is the equivalence between inverse reversal search and forward robust mixture bounds clear?)
- **Confidence Intervals**: (Are asymptotic limitations of Wald intervals adequately emphasized?)
- **Unsupported Novelty**: (Did you identify any unjustified claims of theoretical novelty or physical universality?)

---

## 6. Engineering, Packaging & Service Feedback (Tracks B & C)

- **Dependency Footprint**: (Were core dependencies sufficiently lean? Were service extras appropriately decoupled?)
- **SDK & CLI Usability**: (Were class interfaces, method signatures, and exception types clear and intuitive?)
- **Job Engine Reliability**: (Did SQLite WAL metadata persistence and process-isolated child execution operate reliably?)
- **Security Trust Boundary**: (Is the single-tenant, trusted-local loopback boundary clearly understood?)
- **Platform Portability**: (Did you observe any OS-specific or architecture-specific failures?)

---

## 7. Documentation Clarity

- **README & Quickstart**: [ ] Excellent | [ ] Good | [ ] Adequate | [ ] Needs Work
- **API Reference**: [ ] Excellent | [ ] Good | [ ] Adequate | [ ] Needs Work
- **Tutorial Notebook**: [ ] Excellent | [ ] Good | [ ] Adequate | [ ] Needs Work
- **Service Architecture**: [ ] Excellent | [ ] Good | [ ] Adequate | [ ] Needs Work
- **Specific Documentation Gaps**:

---

## 8. Overall Readiness Verdict

- **Release Recommendation**:
  - [ ] **READY**: Approve for open-source v0.1.0 release as-is.
  - [ ] **READY WITH MINOR REVISIONS**: Approve once non-blocking P2/P3 items are resolved.
  - [ ] **NOT READY**: Requires resolution of critical P0/P1 items before release.
- **Summary Assessment**:
