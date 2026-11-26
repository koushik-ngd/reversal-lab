# ReversalLab Public SDK Reference

This document describes the public Python API exposed by ReversalLab (`reversallab` and `reversallab.api`).

---

## Package Versioning vs. Schema Versioning

- **Python Package Version (`__version__`):** `0.1.0rc1` (prerelease candidate).
- **Report Schema Version (`SDK_SCHEMA_VERSION`):** `1.0.0` (frozen, versioned JSON contract for audit artifacts).

---

## 1. Data Contract: `BanditDataset`

[`BanditDataset`](../src/reversallab/api.py#L89) is an immutable container for contextual-bandit logged data, enforcing array dimensionality, finiteness, and positivity contracts.

### Constructors

#### `BanditDataset.from_arrays`
```python
@classmethod
def from_arrays(
    cls,
    contexts: FloatArray,
    actions: IntArray,
    rewards: FloatArray,
    propensities: FloatArray,
    *,
    n_actions: int | None = None,
    logging_action_dist: FloatArray | None = None,
) -> BanditDataset
```
- `contexts`: 2D array of shape `(N, D)` representing context features $x_i \in \mathbb{R}^D$. Must be finite.
- `actions`: 1D array of shape `(N,)` with discrete action indices $a_i \in \{0, \dots, K-1\}$.
- `rewards`: 1D array of shape `(N,)` with observed scalar rewards $r_i$.
- `propensities`: 1D array of shape `(N,)` with logging probabilities $p_i = \pi_0(a_i \mid x_i)$. Must be strictly positive ($p_i > 0$).
- `n_actions`: Total discrete actions $K$. Defaults to $\max(a_i) + 1$.
- `logging_action_dist`: Optional full logging distribution of shape `(N, K)`. When present, full overlap can be verified across all actions.

#### `BanditDataset.from_logged_data`
```python
@classmethod
def from_logged_data(cls, data: LoggedBanditData) -> BanditDataset
```
Wraps an existing validated low-level [`LoggedBanditData`](../src/reversallab/contracts/logged_data.py) object.

#### `BanditDataset.from_synthetic`
```python
@classmethod
def from_synthetic(
    cls,
    env: SyntheticBanditEnvironment,
    logging_policy: Policy,
    n_samples: int = 2000,
    *,
    seed: int = 42,
) -> BanditDataset
```
Generates synthetic logged data from a specified environment and logging policy.

### Properties
- `dataset.n_samples`: Integer $N$.
- `dataset.n_actions`: Integer $K$.
- `dataset.context_dim`: Integer $D$.
- `dataset.context`: Read-only `(N, D)` float array.
- `dataset.action`: Read-only `(N,)` int array.
- `dataset.reward`: Read-only `(N,)` float array.
- `dataset.logging_propensity`: Read-only `(N,)` float array (aliased as `dataset.propensity`).

---

## 2. Single-Policy Evaluation: `PolicyValueEvaluator`

[`PolicyValueEvaluator.evaluate`](../src/reversallab/api.py#L243) estimates the expected reward $V(\pi) = \mathbb{E}[r(\pi)]$ for a single target policy.

```python
report = PolicyValueEvaluator.evaluate(
    dataset: BanditDataset,
    policy: Policy,
    *,
    estimator: Literal["ips", "snips", "dm", "dr"] | str = "dr",
    confidence_level: float = 0.95,
    policy_name: str = "evaluated_policy",
    reward_model: RewardModel | None = None,
    cross_fitting_folds: int = 3,
    seed: int = 42,
) -> PolicyValueReport
```

### Supported Estimators
- `"ips"`: Standard Inverse Propensity Scoring $\hat V_{\text{IPS}} = \frac{1}{N} \sum_i w_i r_i$. Unbiased under unconfoundedness and full support, but exhibits high finite-sample variance under poor overlap.
- `"snips"`: Self-Normalized IPS $\hat V_{\text{SNIPS}} = \frac{\sum_i w_i r_i}{\sum_i w_i}$. Reduces variance under propensity misspecification; consistent.
- `"dm"`: Direct Method $\hat V_{\text{DM}} = \frac{1}{N} \sum_i \sum_a \pi(a \mid x_i) \hat q(x_i, a)$. Relies strictly on the fitted reward model; biased if the reward model is misspecified.
- `"dr"`: Doubly Robust $\hat V_{\text{DR}} = \hat V_{\text{DM}} + \frac{1}{N} \sum_i w_i (r_i - \hat q(x_i, a_i))$. Unbiased if *either* the logging propensities or the reward model is correctly specified. Uses K-fold cross-fitting to eliminate sample reuse optimism.

### Returned Report: `PolicyValueReport`
Dataclass containing:
- `point_estimate`: Float $\hat V$.
- `standard_error`: Asymptotic standard error via empirical influence scores.
- `ci_lower`, `ci_upper`: Normal Wald confidence interval $[L, U]$.
- `min_propensity`: Minimum observed logging probability $\min_i p_i$.
- `effective_sample_size`: Kish's ESS $= (\sum w_i)^2 / \sum w_i^2$.
- `ess_fraction`: $\text{ESS} / N$.
- `max_weight_share`: $\max_i w_i / \sum w_i$.
- `summary()`: Human-readable string summary.
- `to_dict()`: JSON-serializable dictionary.

---

## 3. Paired Comparison & Decision Reversal: `PolicyComparison`

[`PolicyComparison.evaluate`](../src/reversallab/api.py#L455) executes complete paired superiority testing and decision robustness analysis.

```python
report = PolicyComparison.evaluate(
    dataset: BanditDataset,
    candidate_policy: Policy,
    baseline_policy: Policy,
    *,
    margin: float = 0.0,
    deploy_epsilon: float = 0.10,
    max_epsilon: float = 0.40,
    confidence_level: float = 0.95,
    stratum_partition: StratumPartition | None = None,
    estimator: Literal["dr", "ips", "snips", "dm"] | str = "dr",
    reward_model_factory: Any | None = None,
    cross_fitting_folds: int = 3,
    min_propensity_threshold: float = 1e-4,
    seed: int = 42,
) -> PolicyComparisonReport
```

### Parameters
- `margin` ($\delta$): Operational margin required to justify switching to candidate policy (e.g. $+0.02$).
- `deploy_epsilon` ($\epsilon_{\text{deploy}}$): Certified operational perturbation ball radius around baseline mixture $q_0$: $\mathcal{Q}_{\epsilon} = [\max(0, q_0 - \epsilon), \min(1, q_0 + \epsilon)]$.
- `max_epsilon`: Maximum search radius for decision reversal (default $0.40$).
- `stratum_partition`: Defines the 1-D demographic context split. Default is [`StratumPartition(split_dim=0, split_threshold=0.0)`](../src/reversallab/reversal/strata.py) which divides observations into $\mathcal{X}_0 = \{x : x_0 \ge 0\}$ and $\mathcal{X}_1 = \{x : x_0 < 0\}$.
- `min_propensity_threshold`: Minimum logging propensity threshold $\tau$ for diagnostic gating (default $10^{-4}$).

### Decision Rules
- **RECOMMEND:** Validated lower confidence bound strictly exceeds margin: $L > \delta$.
- **HOLD:** Validated upper confidence bound strictly below margin: $U < \delta$.
- **INCONCLUSIVE:** Confidence interval covers $\delta$, or the diagnostic gate failed. (Strict boundary equality $L = \delta$ or $U = \delta$ is classified conservatively as `INCONCLUSIVE`).

### Statistical Support, Propensity Deficits, and Heuristic Operational Thresholds

Statistical validity requires carefully distinguishing between four distinct concepts:

1. **Proven Mathematical Support Violations:**
   Occurs when the logging policy distribution is fully recorded and assigns zero probability $\pi_0(a \mid x) = 0$ to an action where the evaluation policy places positive mass $\pi_e(a \mid x) > 0$. In this case, off-policy identification mathematically fails without structural extrapolation assumptions.

2. **Unverified Support:**
   In observational data where the full action distribution $\pi_0(\cdot \mid x)$ is not recorded (only the observed tuple $(x_i, a_i, r_i, p_i)$ is preserved), support cannot be verified across counterfactual actions. ReversalLab reports this as unverified support rather than conflating it with a proven violation.

3. **Low but Positive Logging Propensities (Active Deficit):**
   Occurs when $0 < p_i < \tau$ on an observation. **This is not a mathematical support violation**, because $p_i > 0$ and the Radon-Nikodym derivative exists. Crucially, a small positive logging propensity does **not** inherently imply a large importance weight: if the evaluation policy places zero mass on that action ($\pi(a_i \mid x_i) = 0$), the importance weight is exactly zero ($w_i = 0$), and if $\pi(a_i \mid x_i) \approx p_i$, the weight remains modest ($O(1)$). An explosive weight condition ($w_i \gg 1$) occurs specifically when the evaluation policy actively assigns substantial probability mass to that action while the logging policy took it only rarely ($\pi(a_i \mid x_i) \gg p_i$). When this active deficit occurs, finite-sample variance inflates dramatically, causing asymptotic normal Wald confidence intervals to undercover severely.

4. **Heuristic Operational Risk Thresholds:**
   Engineered safeguard parameters (default: $\tau = 10^{-4}$, Kish's $\text{ESS}/N < 0.05$, max single-weight share $> 0.20$) designed to detect finite-sample variance destabilization.

#### Why Diagnostic Gate Rejection Forces `INCONCLUSIVE`
When an active propensity deficit or ESS collapse is detected, the diagnostic gate fails. The engine responsibly overrides the nominal confidence bound and sets the decision outcome to `INCONCLUSIVE`. This conservative behavior prevents spurious `RECOMMEND` decisions: when importance weights are destabilized, the empirical confidence interval can severely undercover, giving a false illusion of precision. Deferring to `INCONCLUSIVE` protects deployment decisions until additional exploratory data or variance-regularized estimators can be obtained.

### Reversal Distance vs. Forward Robust Mixture
- **Inverse Reversal Distance ($d^*$):** The minimum change in stratum mixture $|q - q_0|$ that changes the deployment decision. Computed via validated numerical bisection and checked against analytical quadratic roots.
- **Forward Robust Mixture Baseline (Method F):** Evaluates the worst-case lower bound $L_{\text{worst}} = \min_{q \in \mathcal{Q}_\epsilon} L_\Delta(q)$. By concavity of $L_\Delta(q)$, the worst-case within any closed interval is always attained at a boundary endpoint.
- **Equivalence:** In 1-D two-stratum mixture shifts, inverse thresholding ($\hat d^* > \epsilon_{\text{deploy}}$) and forward robust bounds ($L_{\text{worst}} > \delta$) yield identical binary deployment decisions.

---

## 4. Report Contract: `PolicyComparisonReport`

The `paired_advantage` section summarizes the observed sample. The `decision` section includes the reference stratum-0 weight (`baseline_q0`, currently 0.5), its `point_estimate`, and its interval (`ci_lower`, `ci_upper`). These quantities can differ when the observed sample has unequal stratum sizes.

[`PolicyComparisonReport`](../src/reversallab/api.py#L319) provides both human-readable visualization and versioned machine-readable JSON exports.

### Methods
- `report.summary() -> str`: ASCII formatted summary.
- `report.to_dict() -> dict[str, Any]`: Structured dictionary matching schema `1.0.0`.
- `report.to_json(path: Path | str | None = None) -> str`: Serializes to JSON string or writes to disk.
- `report.save(path: Path | str)`: Convenience alias for saving to disk.

### Schema Fields (`v1.0.0`)
```json
{
  "schema_version": "1.0.0",
  "timestamp": "ISO-8601 UTC timestamp",
  "metadata": {
    "n_samples": 2500,
    "n_actions": 3,
    "context_dim": 2,
    "split_dim": 0,
    "split_threshold": 0.0,
    "cross_fitting_folds": 3,
    "seed": 42
  },
  "diagnostics": {
    "gate_passed": true,
    "min_propensity": 0.3333,
    "min_propensity_threshold": 0.0001,
    "effective_sample_size": 1148.2,
    "ess_fraction": 0.459,
    "max_weight_share": 0.001,
    "rejection_reasons": []
  },
  "paired_advantage": {
    "estimator": "dr",
    "point_estimate": 0.2489,
    "standard_error": 0.0084,
    "ci_lower": 0.2325,
    "ci_upper": 0.2653,
    "confidence_level": 0.95
  },
  "decision": {
    "outcome": "RECOMMEND",
    "margin": 0.02,
    "reason": "Statistically significant superiority: Lower confidence bound exceeds margin.",
    "is_actionable": true
  },
  "reversal_search": {
    "status": "NO_REVERSAL_FOUND_WITHIN_BOUNDS",
    "transition": "NO_REVERSAL",
    "reversal_distance": null,
    "exact_reversal_distance": null,
    "analytical_discrepancy": 0.0,
    "reversed_q0": null,
    "max_epsilon": 0.40,
    "taxonomy_category": "no_reversal"
  },
  "simultaneous_band": {
    "deploy_epsilon": 0.10,
    "min_simultaneous_lower": 0.2217,
    "simultaneous_decision": "RECOMMEND"
  },
  "robust_mixture": {
    "deploy_epsilon": 0.10,
    "worst_case_lower_bound": 0.2316,
    "worst_case_upper_bound": 0.2651,
    "outcome": "RECOMMEND",
    "reason": "Robust superiority certified: Worst-case lower bound across Q_epsilon exceeds margin."
  },
  "assumptions_and_guarantees": {
    "supported_claims": "Asymptotically valid normal confidence intervals for paired difference...",
    "unsupported_claims": "Does not establish safety against unobserved confounding..."
  },
  "warnings": []
}
```

---

## 5. Deliberate Boundaries & Unsupported Claims

ReversalLab explicitly does **not** support or claim:
1. **Unobserved Confounding:** All estimators require unconfoundedness $r(a) \perp a \mid x$. If unobserved variables influence logging actions and rewards, OPE point estimates and bounds are biased.
2. **Continuous Density Drift:** Perturbations are defined over discrete stratum weights $q \in [0, 1]$, not non-parametric infinite-dimensional density shifts.
3. **Multidimensional Reversal Optimization:** Multi-stratum optimization across many dimensions is prone to non-convex local minima and is out of scope.
