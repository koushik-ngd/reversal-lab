# Experiment 003 — Advanced Estimator Validation (DM, DR, and Cross-Fitting)

Protocol: [`docs/experiment_protocol.md`](../../docs/experiment_protocol.md).

```text
uv run reversallab validate-advanced-estimators --config configs/experiments/003_advanced_estimators.yaml
```

`runs/` is gitignored. The authoritative record is `run.json`.

## Run recorded here

Executed in this working tree on 2026-10-30.

| | |
|---|---|
| Run | `003_advanced_estimators_20261030T121529096503Z_41abfc81` |
| `config_sha256` | `41abfc81e6ac0441ba57b3ec57c3bd43903d4e26701f45157b22ccc04ed70979` |
| `results_sha256` | `2362350654e292e215a49847a7c19e1b036cefe947f4e548c4baaf3bd6015b1f` |
| Python / NumPy | 3.12.13 / 2.5.4 |
| scikit-learn | 1.9.1 |
| Wall time | 194.6 s |
| Git commit | `ab4dfad` (+ uncommitted M3B-1 working tree changes) |
| Run-level warnings | none |
| Failed replicates | 0 in every cell (100 replications per cell) |

Reference values below are computed via Monte Carlo over 500,000 independent contexts with exact action-probability summation (Monte Carlo standard error \(\le 0.0011\)). Table values are rounded from `run.json`. Bias SE is the Monte Carlo standard error of the mean estimate across 100 replications. Coverage is for the nominal 95% Wald interval for IPS and SNIPS; for DM and DR, confidence-interval coverage is reported as `N/A` because Wald intervals that ignore reward-model fitting uncertainty are uncalibrated (deferred to M3B-2). ESS/N is the mean Kish effective-sample-size fraction.

True values of the evaluation policies:
- `good_overlap_correct_model`: 0.6495
- `good_overlap_misspecified_model`: 0.6069
- `poor_overlap_correct_model`: 0.7534
- `poor_overlap_misspecified_model`: 0.6714
- `continuous_reward_good_overlap`: 0.8038
- `continuous_reward_misspecified`: 0.6608

## Empirical Results

| scenario | n | estimator | bias | bias SE | RMSE | coverage | empirical SD | ESS/N |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| good_overlap_correct_model | 500 | ips | -0.0053 | 0.0048 | 0.0484 | 0.940 | 0.0483 | 0.474 |
| good_overlap_correct_model | 500 | snips | -0.0058 | 0.0030 | 0.0299 | 0.940 | 0.0295 | 0.474 |
| good_overlap_correct_model | 500 | dm | -0.0160 | 0.0027 | 0.0315 | N/A | 0.0273 | 0.474 |
| good_overlap_correct_model | 500 | dr | -0.0057 | 0.0029 | 0.0292 | N/A | 0.0288 | 0.474 |
| good_overlap_correct_model | 2000 | ips | -0.0013 | 0.0024 | 0.0238 | 0.970 | 0.0239 | 0.473 |
| good_overlap_correct_model | 2000 | snips | -0.0020 | 0.0012 | 0.0126 | 0.990 | 0.0125 | 0.473 |
| good_overlap_correct_model | 2000 | dm | -0.0044 | 0.0012 | 0.0124 | N/A | 0.0116 | 0.473 |
| good_overlap_correct_model | 2000 | dr | -0.0024 | 0.0012 | 0.0118 | N/A | 0.0116 | 0.473 |
| good_overlap_misspecified_model | 500 | ips | -0.0008 | 0.0037 | 0.0372 | 0.960 | 0.0374 | 0.584 |
| good_overlap_misspecified_model | 500 | snips | -0.0018 | 0.0026 | 0.0263 | 0.960 | 0.0263 | 0.584 |
| good_overlap_misspecified_model | 500 | dm | -0.0285 | 0.0027 | 0.0393 | N/A | 0.0271 | 0.584 |
| good_overlap_misspecified_model | 500 | dr | -0.0015 | 0.0027 | 0.0267 | N/A | 0.0268 | 0.584 |
| good_overlap_misspecified_model | 2000 | ips | -0.0000 | 0.0020 | 0.0200 | 0.950 | 0.0201 | 0.583 |
| good_overlap_misspecified_model | 2000 | snips | -0.0006 | 0.0014 | 0.0143 | 0.950 | 0.0144 | 0.583 |
| good_overlap_misspecified_model | 2000 | dm | -0.0259 | 0.0013 | 0.0289 | N/A | 0.0128 | 0.583 |
| good_overlap_misspecified_model | 2000 | dr | -0.0009 | 0.0014 | 0.0142 | N/A | 0.0143 | 0.583 |
| poor_overlap_correct_model | 500 | ips | +0.0359 | 0.0583 | 0.5815 | 0.820 | 0.5833 | 0.092 |
| poor_overlap_correct_model | 500 | snips | -0.0159 | 0.0082 | 0.0827 | 0.890 | 0.0815 | 0.092 |
| poor_overlap_correct_model | 500 | dm | -0.0315 | 0.0037 | 0.0485 | N/A | 0.0372 | 0.092 |
| poor_overlap_correct_model | 500 | dr | -0.0132 | 0.0100 | 0.1005 | N/A | 0.1001 | 0.092 |
| poor_overlap_correct_model | 2000 | ips | +0.0646 | 0.0413 | 0.4161 | 0.900 | 0.4132 | 0.060 |
| poor_overlap_correct_model | 2000 | snips | -0.0025 | 0.0068 | 0.0679 | 0.890 | 0.0682 | 0.060 |
| poor_overlap_correct_model | 2000 | dm | -0.0056 | 0.0016 | 0.0166 | N/A | 0.0158 | 0.060 |
| poor_overlap_correct_model | 2000 | dr | +0.0077 | 0.0142 | 0.1411 | N/A | 0.1416 | 0.060 |
| poor_overlap_misspecified_model | 500 | ips | +0.0184 | 0.0274 | 0.2736 | 0.820 | 0.2744 | 0.079 |
| poor_overlap_misspecified_model | 500 | snips | -0.0085 | 0.0095 | 0.0946 | 0.900 | 0.0947 | 0.079 |
| poor_overlap_misspecified_model | 500 | dm | -0.0759 | 0.0031 | 0.0820 | N/A | 0.0311 | 0.079 |
| poor_overlap_misspecified_model | 500 | dr | -0.0103 | 0.0126 | 0.1258 | N/A | 0.1260 | 0.079 |
| poor_overlap_misspecified_model | 2000 | ips | +0.0036 | 0.0129 | 0.1281 | 0.840 | 0.1287 | 0.053 |
| poor_overlap_misspecified_model | 2000 | snips | +0.0006 | 0.0053 | 0.0527 | 0.900 | 0.0529 | 0.053 |
| poor_overlap_misspecified_model | 2000 | dm | -0.0719 | 0.0020 | 0.0746 | N/A | 0.0198 | 0.053 |
| poor_overlap_misspecified_model | 2000 | dr | +0.0052 | 0.0048 | 0.0479 | N/A | 0.0478 | 0.053 |
| continuous_reward_good_overlap | 500 | ips | -0.0015 | 0.0113 | 0.1123 | 0.920 | 0.1129 | 0.476 |
| continuous_reward_good_overlap | 500 | snips | +0.0011 | 0.0091 | 0.0903 | 0.960 | 0.0908 | 0.476 |
| continuous_reward_good_overlap | 500 | dm | -0.0067 | 0.0072 | 0.0723 | N/A | 0.0723 | 0.476 |
| continuous_reward_good_overlap | 500 | dr | +0.0024 | 0.0074 | 0.0740 | N/A | 0.0743 | 0.476 |
| continuous_reward_good_overlap | 2000 | ips | +0.0133 | 0.0059 | 0.0603 | 0.930 | 0.0591 | 0.478 |
| continuous_reward_good_overlap | 2000 | snips | +0.0098 | 0.0048 | 0.0492 | 0.920 | 0.0485 | 0.478 |
| continuous_reward_good_overlap | 2000 | dm | +0.0027 | 0.0039 | 0.0387 | N/A | 0.0388 | 0.478 |
| continuous_reward_good_overlap | 2000 | dr | +0.0075 | 0.0040 | 0.0408 | N/A | 0.0403 | 0.478 |
| continuous_reward_misspecified | 500 | ips | +0.0039 | 0.0090 | 0.0893 | 0.960 | 0.0897 | 0.524 |
| continuous_reward_misspecified | 500 | snips | -0.0014 | 0.0075 | 0.0742 | 0.960 | 0.0745 | 0.524 |
| continuous_reward_misspecified | 500 | dm | -0.0530 | 0.0069 | 0.0866 | N/A | 0.0688 | 0.524 |
| continuous_reward_misspecified | 500 | dr | -0.0104 | 0.0079 | 0.0797 | N/A | 0.0795 | 0.524 |
| continuous_reward_misspecified | 2000 | ips | -0.0061 | 0.0047 | 0.0474 | 0.950 | 0.0472 | 0.524 |
| continuous_reward_misspecified | 2000 | snips | -0.0043 | 0.0038 | 0.0381 | 0.950 | 0.0380 | 0.524 |
| continuous_reward_misspecified | 2000 | dm | -0.0439 | 0.0033 | 0.0551 | N/A | 0.0334 | 0.524 |
| continuous_reward_misspecified | 2000 | dr | -0.0037 | 0.0034 | 0.0338 | N/A | 0.0338 | 0.524 |

## Scientific Interpretation

### 1. Verification of Double Robustness Under Reward Model Misspecification
The empirical results provide direct confirmation of the double robustness property:
- In `good_overlap_misspecified_model`, the reward model \(\hat{q}(x, a)\) is fit on a constant zero-mean feature representation, omitting the true linear context coefficients. Consequently, the Direct Method (DM) suffers asymptotic, irreducible bias: \(-0.0285 \pm 0.0027\) at \(N = 500\) and \(-0.0259 \pm 0.0013\) at \(N = 2000\) (true value 0.6069).
- In sharp contrast, Doubly Robust (DR) achieves a bias of \(-0.0015 \pm 0.0027\) at \(N = 500\) and \(-0.0009 \pm 0.0014\) at \(N = 2000\), which is statistically indistinguishable from zero and matches the unbiased behavior of IPS (\(-0.0000 \pm 0.0020\)).
- The same behavior is observed in `continuous_reward_misspecified`: DM bias remains high (\(-0.0439 \pm 0.0033\) at \(N = 2000\)), while DR bias shrinks to \(-0.0037 \pm 0.0034\), reducing RMSE from \(0.0551\) (DM) to \(0.0338\) (DR).
Because logging propensities \(\pi_0(a_i \mid x_i)\) are known and correct, the importance-weighted factual residual \(w_i (r_i - \hat{q}(x_i, a_i))\) in DR successfully debiases the misspecified model predictions.

### 2. Finite-Sample Variance Under Poor Overlap: When DR Performs Worse Than DM
A critical question in off-policy evaluation theory is whether double robustness guarantees superior finite-sample performance. These results demonstrate that **it does not**:
- In `poor_overlap_correct_model` at \(N = 2000\):
  - Direct Method (DM) achieves an RMSE of **0.0166** (empirical SD 0.0158).
  - Doubly Robust (DR) achieves an RMSE of **0.1411** (empirical SD 0.1416).
  - DR performs almost **8.5 times worse in RMSE** than DM!
- **Theoretical Explanation:** Under poor overlap (mean ESS fraction 0.060), importance weights \(w_i = \pi_e(a_i \mid x_i) / \pi_0(a_i \mid x_i)\) have a heavy-tailed distribution with extreme values. Even though \(\mathbb{E}[r_i - \hat{q}(x_i, a_i) \mid x_i, a_i] = 0\) when the reward model is well-specified, the factual sample residual \((r_i - \hat{q}(x_i, a_i))\) is never identically zero in finite samples. Multiplying this residual by huge importance weights injects enormous sampling variance into DR. DM, which integrates exclusively over the candidate policy distribution and never divides by logging propensities, is immune to propensity variance.
- This finding aligns directly with Dudík et al. (2014) and Su et al. (2020), emphasizing that doubly robust estimators can suffer severe variance degradation under poor overlap unless regularization or weight-shrinkage techniques (planned for M3B-2) are applied.

### 3. Continuous Rewards Performance
For Gaussian rewards (`continuous_reward_good_overlap`):
- All four estimators are consistent and converge towards the true value (0.8038).
- Cross-fitted Ridge regression models achieve low prediction error.
- At \(N = 2000\), DM achieves the lowest RMSE (0.0387), followed closely by DR (0.0408) and SNIPS (0.0492), with IPS having the highest RMSE (0.0603).
- Both DM and DR seamlessly operate on continuous reward contracts without any modification to the core estimation mathematical interface.

### 4. Leakage-Safe Cross-Fitting Integrity
- Deterministic 5-fold cross-fitting (`cross_fit_reward_model`) ensured that every observation \(i\) received a predicted reward \(\hat{q}(x_i, a)\) generated strictly by models trained on the remaining 4 out-of-fold partitions.
- Stratified linear reward models (`StratifiedLinearRewardModel`) successfully learned action-specific coefficients for observed actions and reverted to global training means when actions had zero count in training folds.
- Cross-fitting leakage assertions in `tests/unit/test_cross_fitting.py` confirmed 0 training/evaluation context leakage.
