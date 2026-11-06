# ReversalLab

ReversalLab is an open-source Python library for statistically grounded off-policy evaluation (OPE) of contextual bandits, paired policy superiority testing, and decision-reversal robustness analysis.

Rather than relying solely on point estimates or standard confidence intervals that collapse under distribution shift, ReversalLab asks:

> **What is the smallest admissible change in population demographics or evaluation assumptions that reverses an evidence-based deployment decision?**

---

## Key Capabilities

- **Standard & Advanced OPE Estimators:** Inverse Propensity Scoring (IPS), Self-Normalized IPS (SNIPS), Direct Method (DM), and Doubly Robust (DR) estimation.
- **Leakage-Safe K-Fold Cross-Fitting:** Strict out-of-fold reward-model training preventing optimism and target-leakage bias.
- **Paired Policy Difference Estimation:** Influence-score-based paired comparison $\Delta = V(\pi_e) - V(\pi_b)$ with Wald confidence intervals and operational deployment margins $\delta$.
- **Multi-Factor Diagnostic Gates:** Evaluates logging propensity overlap ($p_i \ge \tau$ on actively evaluated actions), effective sample size ($\text{ESS}/N \ge 0.05$), and weight concentration ($\max w_i / \sum w_i \le 0.20$) before certifying deployment.
- **Two-Stratum Decision-Reversal Engine:** Computes the exact minimum demographic shift distance $d^* = |q^* - q_0|$ required to overturn a deployment decision, using analytical root-finding and validated bisection search.
- **Simultaneous Confidence Bands & Forward Worst-Case Bounds:** Dual validation via $2 \times 95\%$ simultaneous confidence bands and forward worst-case robust mixture evaluation across certified operational envelopes $\mathcal{Q}_\epsilon$.
- **Ergonomic Python SDK & CLI:** High-level API (`BanditDataset`, `PolicyValueEvaluator`, `PolicyComparison`), schema-versioned JSON report exports (`v1.0.0`), and reproducible CLI commands.

---

## Installation

ReversalLab requires Python 3.11+ (recommended: Python 3.12 pinned via [uv](https://docs.astral.sh/uv/)).

```bash
# Clone the repository
git clone https://github.com/reversallab/reversal-lab.git
cd reversal-lab

# Install dependencies using uv
uv sync --frozen --all-groups
```

To build and install the standalone wheel into an isolated environment using `uv`:

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

## Quickstart

Run the complete end-to-end decision-reversal workflow in one command:

```bash
uv run reversallab run-example --output runs/quickstart_report.json
```

Or use the Python SDK directly:

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
)

# 4. Inspect ASCII audit report and export JSON
print(report.summary())
report.save("runs/decision_audit.json")
```

For a comprehensive guide, see [`docs/quickstart.md`](docs/quickstart.md).

---

## Testing & Quality Verification

ReversalLab maintains strict type-checking, code styling, and statistical test coverage:

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy src tests examples
```

---

## Reproducible Research Experiments

ReversalLab was developed through seven rigorously audited empirical milestones:

| Experiment | Title | Description | Reproduction Command |
|---|---|---|---|
| **001** | Estimator Validation | Finite-sample bias, variance, and MSE of IPS and SNIPS across sample sizes. | `uv run reversallab validate-estimators --config configs/experiments/001_estimator_validation.yaml` |
| **002** | Reference Parity | Numerical parity comparison against Open Bandit Pipeline (OBP) reference estimators. | `uv run reversallab compare-reference --config configs/experiments/002_open_bandit_parity.yaml` |
| **003** | Advanced Estimators | DM and DR evaluation with leakage-safe K-fold cross-fitting and misspecification tests. | `uv run reversallab validate-estimators --config configs/experiments/003_advanced_estimators.yaml` |
| **004** | Uncertainty Calibration | Coverage and reliability audit of Wald intervals, influence scores, and bootstrap resamples. | `uv run reversallab validate-estimators --config configs/experiments/004_uncertainty_calibration.yaml` |
| **005** | Decision-Reversal Prototype | Prototype 1-D context mixture perturbation search and margin-crossing detection. | `uv run reversallab run-reversal --config configs/experiments/005_reversal_prototype.yaml` |
| **006** | Mathematical Validation | Analytical reversal root-finding and simultaneous confidence band certification. | `uv run reversallab run-reversal --config configs/experiments/006_reversal_validation.yaml` |
| **007** | Independent Reliability | 5,000 Monte Carlo replications comparing inverse reversal distance and forward robust baselines. | `uv run reversallab run-reversal --config configs/experiments/007_independent_reliability.yaml` |

To inspect any generated run artifact:

```bash
uv run reversallab report --run-dir runs/<run_id>
```

---

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md): Step-by-step tutorial for practitioners and engineers.
- [`docs/api_reference.md`](docs/api_reference.md): Public SDK API reference (`BanditDataset`, `PolicyValueEvaluator`, `PolicyComparison`).
- [`notebooks/quickstart_decision_reversal.ipynb`](notebooks/quickstart_decision_reversal.ipynb): Executable Jupyter tutorial with actual computed results.
- [`docs/architecture.md`](docs/architecture.md): System layout, public SDK contracts, and data-flow diagrams.
- [`docs/decision_reliability_audit.md`](docs/decision_reliability_audit.md): Complete mathematical and empirical closeout audit of Experiment 007.
- [`docs/methodology.md`](docs/methodology.md): Formal definitions of estimators, influence scores, and mixture perturbations.
- [`docs/data_contract.md`](docs/data_contract.md): Field-by-field array contracts and validation rules.
- [`docs/assumptions.md`](docs/assumptions.md): Supported statistical claims vs. deliberate limitations.

---

## Scientific Scope & Limitations

ReversalLab provides mathematical guarantees under declared assumptions:
- **Supported:** Asymptotically valid paired difference confidence intervals via Doubly Robust estimation with K-fold cross-fitting, under unconfoundedness, positive propensity support, and correctly specified logging probabilities.
- **Supported:** Exact boundary minima for forward worst-case bounds under 1-D mixture perturbations due to concavity of the lower confidence bound.
- **Equivalence Note:** In 1-D two-stratum mixture shifts, inverse reversal thresholding and forward robust-value checks yield identical binary deployment decisions at budget $\epsilon_{\text{deploy}}$. Interpretability advantages of distance reporting are proposed conceptual benefits, not experimentally established user advantages.
- **Limitations:** Does not establish safety against unobserved confounding, continuous non-parametric density drift, or multidimensional demographic shifts beyond the evaluated partition.

---

## License

MIT. See [LICENSE](LICENSE).
