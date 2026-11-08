# ReversalLab — Issue Intake, Classification & Triage Policy

This document establishes the official issue intake, classification taxonomy, and triage workflow for findings reported by external reviewers, researchers, and open-source contributors.

---

## 1. Core Principles

1. **Scientific Integrity First**: Mathematical flaws, incorrect estimator properties, target leakage, and false superiority recommendations take absolute priority over cosmetic polish or non-critical features.
2. **Reproducibility Guarantee**: Every valid bug report must be accompanied by minimal reproduction code or seed configurations.
3. **No Fabricated Evidence**: Review findings are logged and prioritized only when submitted by real human or automated validation agents.

---

## 2. Issue Classification Taxonomy

Every incoming issue must be categorized under one of the five canonical domains:

### 1. Scientific Correctness (`domain:science`)
- **Definition**: Flaws in mathematical formulation, causal identification assumptions, nuisance model cross-fitting, confidence interval estimation, or empirical simulation reporting.
- **Examples**:
  - Out-of-fold cross-fitting leakage in `src/reversallab/estimators/cross_fitting.py`.
  - Inconsistent importance weight calculation in IPS/SNIPS.
  - Flawed Bonferroni interval boundary calculation for simultaneous lower bounds.
  - Mathematical failure of the 1-D reversal root-finding solver.

### 2. Software Defect (`domain:software`)
- **Definition**: Implementation bugs, unhandled exceptions, type errors, or crashes occurring on valid inputs.
- **Examples**:
  - `IndexError` or `ValueError` during valid `BanditDataset` construction.
  - CLI argument parsing regression.
  - Memory leak during high-repetition Monte Carlo execution.
  - Non-deterministic output under a pinned seed on identical hardware.

### 3. Documentation & Usability (`domain:docs`)
- **Definition**: Inaccuracies, broken links, unclear parameter descriptions, misleading comments, or friction in developer ergonomics.
- **Examples**:
  - Outdated CLI commands in quickstarts or tutorial notebooks.
  - Unclear error message tracebacks that obscure invalid input causes.
  - Missing parameter docstrings or typos in public API reference.

### 4. Security & Reliability (`domain:security`)
- **Definition**: Violations of the Deployment Model A trust boundary, resource exhaustion vulnerabilities, unhandled race conditions, or storage state corruption.
- **Examples**:
  - Default network binding to non-loopback addresses (`0.0.0.0`).
  - Path traversal vulnerability in `job_id` routing.
  - Crash recovery leaving orphan child worker processes.
  - SQLite WAL lock contention deadlocks.

### 5. Feature Suggestion (`domain:enhancement`)
- **Definition**: Proposals for new capabilities, additional estimators, alternative metrics, or performance micro-optimizations that extend beyond current project scope.
- **Examples**:
  - Support for multi-dimensional or Wasserstein perturbation sets.
  - Integration with distributed schedulers (Ray, Celery, Kubernetes).
  - New policy classes or estimator variants (e.g., CAB, DR-DR).

---

## 3. Severity & Prioritization Matrix

| Priority Level | Classification | Impact & Criteria | Action & SLA |
| :---: | :--- | :--- | :--- |
| **P0 (Blocker)** | Critical Correctness / Security | Invalidates core statistical claims, causes false policy recommendations, leaks sensitive memory, or bypasses loopback boundary. | Immediate maintainer attention. Fix required prior to any public v0.1.0 release. |
| **P1 (High)** | Major Defect / Reliability | Reproducible crashes on valid data, supervisor lock failure, diagnostic gate false negatives, or critical documentation errors leading to misuse. | Prioritized for resolution within the active release candidate cycle. |
| **P2 (Medium)** | Usability / Minor Defect | Unclear error messages, non-blocking test flakes on specific OS versions, minor documentation typos, non-critical logging verbosity. | Scheduled for standard release iterations. |
| **P3 (Low)** | Enhancement / Polish | Code refactoring, minor performance optimizations, feature requests for future roadmap. | Backlogged for future milestones (v0.2.0+). |

---

## 4. Triage Workflow

```
[Issue Received]
       │
       ▼
[Initial Screen] ──▶ Invalid / Incomplete? ──▶ Request reproduction / Close
       │
       ▼
[Classify Domain] ──▶ Science / Software / Docs / Security / Enhancement
       │
       ▼
[Assign Priority] ──▶ P0 (Blocker) ──▶ Immediate patch & release candidate hold
       │
       ├──▶ P1 (High) ──────▶ Active sprint backlog
       │
       ├──▶ P2 (Medium) ────▶ Next scheduled release cycle
       │
       └──▶ P3 (Low) ───────▶ Future roadmap discussion
```

---

## 5. Maintenance & Support Policy

ReversalLab is an open-source academic research project developed and maintained on a **sustainable best-effort basis**:
- Maintainers prioritize issues impacting scientific validity and reproducibility over feature requests.
- Commercial SLAs (e.g., guaranteed hourly response times) are not offered.
- Contributions and patches from external researchers and engineers are welcomed via standard GitHub Pull Requests.
