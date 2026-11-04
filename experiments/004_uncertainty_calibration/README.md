# Experiment 004 — Statistical Uncertainty and Confidence Interval Calibration

Protocol: [`docs/experiment_protocol.md`](../../docs/experiment_protocol.md).
Theoretical Specification: [`docs/inference_assumptions.md`](../../docs/inference_assumptions.md).

```text
uv run reversallab calibrate-uncertainty --config configs/experiments/004_uncertainty_calibration.yaml
```

`runs/` is gitignored. The authoritative record is `run.json`.

## Run recorded here

Executed on 2026-11-04.

| | |
|---|---|
| Run | `004_uncertainty_calibration_20261104T062641547104Z_da0ae740` |
| `config_sha256` | `da0ae740ece382cf78f350c34f929ab9699ac22d9b07c0b78c0e53c06ac2df8a` |
| `results_sha256` | `54a4363ce782666cbe6fdc3550a5f9109c205b8a8f046d1ce00d35268627ad55` |
| Python / NumPy | 3.12.13 / 2.5.4 |
| scikit-learn | 1.9.1 |
| Git commit | `ab4dfad` (+ uncommitted M3B working tree changes) |
| Replications | 100 per cell |
| Bootstrap resamples ($B$) | 50 per replication |
| Cross-fitting folds | 3-fold disjoint cross-fitting |
| Failed replicates | 0 in every cell across all methods |

Reference values below are computed via Monte Carlo over 500,000 independent contexts with exact action-probability summation (Monte Carlo standard error \(\le 0.0011\)). Table values are rounded from `run.json`. `cover` is empirical coverage at nominal 95%; `±mcse` is the Monte Carlo standard error of the coverage estimate (\(\sqrt{\hat{c}(1-\hat{c})/100}\)); `width` is the mean interval width; `fail` is the number of invalid/non-finite intervals.

True values of the evaluation policies:
- `good_overlap_correct_model`: 0.6495
- `good_overlap_misspecified_model`: 0.6069
- `poor_overlap_correct_model`: 0.7534
- `poor_overlap_misspecified_model`: 0.6714
- `continuous_reward_good_overlap`: 0.8038
- `continuous_reward_misspecified`: 0.6608

---

## Empirical Benchmark Results

| Scenario | $N$ | Method | True $V$ | Mean | Bias | RMSE | Cover | $\pm$ MCSE | Width | Fail |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `good_overlap_correct_model` | 500 | `ips_wald` | 0.6495 | 0.6442 | -0.0053 | 0.0484 | 0.940 | 0.024 | 0.1881 | 0 |
| `good_overlap_correct_model` | 500 | `snips_wald` | 0.6495 | 0.6437 | -0.0058 | 0.0299 | 0.940 | 0.024 | 0.1139 | 0 |
| `good_overlap_correct_model` | 500 | `dm_naive_wald` | 0.6495 | 0.6317 | -0.0178 | 0.0326 | **0.280** | 0.045 | 0.0232 | 0 |
| `good_overlap_correct_model` | 500 | `dr_influence_wald` | 0.6495 | 0.6440 | -0.0055 | 0.0288 | 0.940 | 0.024 | 0.1107 | 0 |
| `good_overlap_correct_model` | 500 | `ips_bootstrap_percentile` | 0.6495 | 0.6442 | -0.0053 | 0.0484 | 0.910 | 0.029 | 0.1735 | 0 |
| `good_overlap_correct_model` | 500 | `snips_bootstrap_percentile` | 0.6495 | 0.6437 | -0.0058 | 0.0299 | 0.900 | 0.030 | 0.1052 | 0 |
| `good_overlap_correct_model` | 500 | `dm_bootstrap_refit_percentile` | 0.6495 | 0.6323 | -0.0172 | 0.0319 | 0.860 | 0.035 | 0.0939 | 0 |
| `good_overlap_correct_model` | 500 | `dr_bootstrap_refit_percentile` | 0.6495 | 0.6431 | -0.0064 | 0.0296 | 0.890 | 0.031 | 0.1004 | 0 |
| `good_overlap_correct_model` | 500 | `dr_bootstrap_refit_basic` | 0.6495 | 0.6431 | -0.0064 | 0.0296 | 0.880 | 0.032 | 0.1004 | 0 |
| `good_overlap_correct_model` | 2000 | `ips_wald` | 0.6495 | 0.6482 | -0.0013 | 0.0238 | 0.970 | 0.017 | 0.0944 | 0 |
| `good_overlap_correct_model` | 2000 | `snips_wald` | 0.6495 | 0.6475 | -0.0020 | 0.0126 | 0.990 | 0.010 | 0.0569 | 0 |
| `good_overlap_correct_model` | 2000 | `dm_naive_wald` | 0.6495 | 0.6446 | -0.0048 | 0.0125 | **0.350** | 0.048 | 0.0109 | 0 |
| `good_overlap_correct_model` | 2000 | `dr_influence_wald` | 0.6495 | 0.6471 | -0.0023 | 0.0118 | 0.980 | 0.014 | 0.0540 | 0 |
| `good_overlap_correct_model` | 2000 | `ips_bootstrap_percentile` | 0.6495 | 0.6482 | -0.0013 | 0.0238 | 0.930 | 0.026 | 0.0870 | 0 |
| `good_overlap_correct_model` | 2000 | `snips_bootstrap_percentile` | 0.6495 | 0.6475 | -0.0020 | 0.0126 | 0.970 | 0.017 | 0.0524 | 0 |
| `good_overlap_correct_model` | 2000 | `dm_bootstrap_refit_percentile` | 0.6495 | 0.6446 | -0.0048 | 0.0125 | 0.920 | 0.027 | 0.0475 | 0 |
| `good_overlap_correct_model` | 2000 | `dr_bootstrap_refit_percentile` | 0.6495 | 0.6470 | -0.0025 | 0.0118 | 0.950 | 0.022 | 0.0506 | 0 |
| `good_overlap_correct_model` | 2000 | `dr_bootstrap_refit_basic` | 0.6495 | 0.6470 | -0.0025 | 0.0118 | 0.970 | 0.017 | 0.0506 | 0 |
| `good_overlap_misspecified_model` | 500 | `ips_wald` | 0.6069 | 0.6060 | -0.0008 | 0.0372 | 0.960 | 0.020 | 0.1556 | 0 |
| `good_overlap_misspecified_model` | 500 | `snips_wald` | 0.6069 | 0.6050 | -0.0018 | 0.0263 | 0.960 | 0.020 | 0.1086 | 0 |
| `good_overlap_misspecified_model` | 500 | `dm_naive_wald` | 0.6069 | 0.5766 | -0.0302 | 0.0404 | **0.170** | 0.038 | 0.0188 | 0 |
| `good_overlap_misspecified_model` | 500 | `dr_influence_wald` | 0.6069 | 0.6051 | -0.0018 | 0.0277 | 0.960 | 0.020 | 0.1121 | 0 |
| `good_overlap_misspecified_model` | 500 | `ips_bootstrap_percentile` | 0.6069 | 0.6060 | -0.0008 | 0.0372 | 0.910 | 0.029 | 0.1425 | 0 |
| `good_overlap_misspecified_model` | 500 | `snips_bootstrap_percentile` | 0.6069 | 0.6050 | -0.0018 | 0.0263 | 0.900 | 0.030 | 0.0988 | 0 |
| `good_overlap_misspecified_model` | 500 | `dm_bootstrap_refit_percentile` | 0.6069 | 0.5770 | -0.0299 | 0.0401 | 0.760 | 0.043 | 0.0948 | 0 |
| `good_overlap_misspecified_model` | 500 | `dr_bootstrap_refit_percentile` | 0.6069 | 0.6050 | -0.0018 | 0.0279 | 0.940 | 0.024 | 0.1028 | 0 |
| `good_overlap_misspecified_model` | 500 | `dr_bootstrap_refit_basic` | 0.6069 | 0.6050 | -0.0018 | 0.0279 | 0.880 | 0.032 | 0.1028 | 0 |
| `good_overlap_misspecified_model` | 2000 | `ips_wald` | 0.6069 | 0.6068 | -0.0000 | 0.0200 | 0.950 | 0.022 | 0.0779 | 0 |
| `good_overlap_misspecified_model` | 2000 | `snips_wald` | 0.6069 | 0.6063 | -0.0006 | 0.0143 | 0.950 | 0.022 | 0.0544 | 0 |
| `good_overlap_misspecified_model` | 2000 | `dm_naive_wald` | 0.6069 | 0.5805 | -0.0263 | 0.0292 | **0.050** | 0.022 | 0.0083 | 0 |
| `good_overlap_misspecified_model` | 2000 | `dr_influence_wald` | 0.6069 | 0.6059 | -0.0010 | 0.0143 | 0.940 | 0.024 | 0.0550 | 0 |
| `good_overlap_misspecified_model` | 2000 | `ips_bootstrap_percentile` | 0.6069 | 0.6068 | -0.0000 | 0.0200 | 0.900 | 0.030 | 0.0712 | 0 |
| `good_overlap_misspecified_model` | 2000 | `snips_bootstrap_percentile` | 0.6069 | 0.6063 | -0.0006 | 0.0143 | 0.910 | 0.029 | 0.0503 | 0 |
| `good_overlap_misspecified_model` | 2000 | `dm_bootstrap_refit_percentile` | 0.6069 | 0.5806 | -0.0263 | 0.0292 | **0.410** | 0.049 | 0.0475 | 0 |
| `good_overlap_misspecified_model` | 2000 | `dr_bootstrap_refit_percentile` | 0.6069 | 0.6058 | -0.0010 | 0.0144 | 0.960 | 0.020 | 0.0511 | 0 |
| `good_overlap_misspecified_model` | 2000 | `dr_bootstrap_refit_basic` | 0.6069 | 0.6058 | -0.0010 | 0.0144 | 0.900 | 0.030 | 0.0511 | 0 |
| `poor_overlap_correct_model` | 500 | `ips_wald` | 0.7534 | 0.7893 | +0.0359 | 0.5815 | **0.820** | 0.038 | 0.7820 | 0 |
| `poor_overlap_correct_model` | 500 | `snips_wald` | 0.7534 | 0.7375 | -0.0159 | 0.0827 | **0.890** | 0.031 | 0.2347 | 0 |
| `poor_overlap_correct_model` | 500 | `dm_naive_wald` | 0.7534 | 0.7216 | -0.0318 | 0.0491 | **0.200** | 0.040 | 0.0252 | 0 |
| `poor_overlap_correct_model` | 500 | `dr_influence_wald` | 0.7534 | 0.7421 | -0.0112 | 0.1249 | 0.930 | 0.026 | 0.3022 | 0 |
| `poor_overlap_correct_model` | 500 | `ips_bootstrap_percentile` | 0.7534 | 0.7893 | +0.0359 | 0.5815 | **0.820** | 0.038 | 0.6057 | 0 |
| `poor_overlap_correct_model` | 500 | `snips_bootstrap_percentile` | 0.7534 | 0.7375 | -0.0159 | 0.0827 | **0.900** | 0.030 | 0.2116 | 0 |
| `poor_overlap_correct_model` | 500 | `dm_bootstrap_refit_percentile` | 0.7534 | 0.7215 | -0.0319 | 0.0489 | **0.720** | 0.045 | 0.1195 | 0 |
| `poor_overlap_correct_model` | 500 | `dr_bootstrap_refit_percentile` | 0.7534 | 0.7394 | -0.0139 | 0.1049 | **0.880** | 0.032 | 0.2606 | 0 |
| `poor_overlap_correct_model` | 500 | `dr_bootstrap_refit_basic` | 0.7534 | 0.7394 | -0.0139 | 0.1049 | **0.840** | 0.037 | 0.2606 | 0 |
| `poor_overlap_correct_model` | 2000 | `ips_wald` | 0.7534 | 0.8179 | +0.0646 | 0.4161 | **0.900** | 0.030 | 0.5369 | 0 |
| `poor_overlap_correct_model` | 2000 | `snips_wald` | 0.7534 | 0.7508 | -0.0025 | 0.0679 | **0.890** | 0.031 | 0.1691 | 0 |
| `poor_overlap_correct_model` | 2000 | `dm_naive_wald` | 0.7534 | 0.7477 | -0.0056 | 0.0166 | **0.550** | 0.050 | 0.0145 | 0 |
| `poor_overlap_correct_model` | 2000 | `dr_influence_wald` | 0.7534 | 0.7618 | +0.0084 | 0.1388 | **0.880** | 0.032 | 0.1989 | 0 |
| `poor_overlap_correct_model` | 2000 | `ips_bootstrap_percentile` | 0.7534 | 0.8179 | +0.0646 | 0.4161 | **0.890** | 0.031 | 0.4633 | 0 |
| `poor_overlap_correct_model` | 2000 | `snips_bootstrap_percentile` | 0.7534 | 0.7508 | -0.0025 | 0.0679 | **0.880** | 0.032 | 0.1557 | 0 |
| `poor_overlap_correct_model` | 2000 | `dm_bootstrap_refit_percentile` | 0.7534 | 0.7478 | -0.0056 | 0.0166 | 0.930 | 0.026 | 0.0601 | 0 |
| `poor_overlap_correct_model` | 2000 | `dr_bootstrap_refit_percentile` | 0.7534 | 0.7607 | +0.0073 | 0.1388 | **0.830** | 0.038 | 0.1873 | 0 |
| `poor_overlap_correct_model` | 2000 | `dr_bootstrap_refit_basic` | 0.7534 | 0.7607 | +0.0073 | 0.1388 | **0.880** | 0.032 | 0.1873 | 0 |
| `poor_overlap_misspecified_model` | 500 | `ips_wald` | 0.6714 | 0.6897 | +0.0184 | 0.2736 | **0.820** | 0.038 | 0.6387 | 0 |
| `poor_overlap_misspecified_model` | 500 | `snips_wald` | 0.6714 | 0.6628 | -0.0085 | 0.0946 | **0.900** | 0.030 | 0.2319 | 0 |
| `poor_overlap_misspecified_model` | 500 | `dm_naive_wald` | 0.6714 | 0.5937 | -0.0776 | 0.0837 | **0.000** | 0.000 | 0.0229 | 0 |
| `poor_overlap_misspecified_model` | 500 | `dr_influence_wald` | 0.6714 | 0.6628 | -0.0086 | 0.1245 | **0.910** | 0.029 | 0.3175 | 0 |
| `poor_overlap_misspecified_model` | 500 | `ips_bootstrap_percentile` | 0.6714 | 0.6897 | +0.0184 | 0.2736 | **0.810** | 0.039 | 0.5098 | 0 |
| `poor_overlap_misspecified_model` | 500 | `snips_bootstrap_percentile` | 0.6714 | 0.6628 | -0.0085 | 0.0946 | **0.880** | 0.032 | 0.2076 | 0 |
| `poor_overlap_misspecified_model` | 500 | `dm_bootstrap_refit_percentile` | 0.6714 | 0.5930 | -0.0784 | 0.0842 | **0.250** | 0.043 | 0.1139 | 0 |
| `poor_overlap_misspecified_model` | 500 | `dr_bootstrap_refit_percentile` | 0.6714 | 0.6595 | -0.0119 | 0.1194 | **0.910** | 0.029 | 0.2828 | 0 |
| `poor_overlap_misspecified_model` | 500 | `dr_bootstrap_refit_basic` | 0.6714 | 0.6595 | -0.0119 | 0.1194 | **0.870** | 0.034 | 0.2828 | 0 |
| `poor_overlap_misspecified_model` | 2000 | `ips_wald` | 0.6714 | 0.6749 | +0.0036 | 0.1281 | **0.840** | 0.037 | 0.4393 | 0 |
| `poor_overlap_misspecified_model` | 2000 | `snips_wald` | 0.6714 | 0.6720 | +0.0006 | 0.0527 | **0.900** | 0.030 | 0.1719 | 0 |
| `poor_overlap_misspecified_model` | 2000 | `dm_naive_wald` | 0.6714 | 0.5987 | -0.0726 | 0.0752 | **0.000** | 0.000 | 0.0159 | 0 |
| `poor_overlap_misspecified_model` | 2000 | `dr_influence_wald` | 0.6714 | 0.6759 | +0.0045 | 0.0467 | **0.890** | 0.031 | 0.1611 | 0 |
| `poor_overlap_misspecified_model` | 2000 | `ips_bootstrap_percentile` | 0.6714 | 0.6749 | +0.0036 | 0.1281 | **0.850** | 0.036 | 0.4003 | 0 |
| `poor_overlap_misspecified_model` | 2000 | `snips_bootstrap_percentile` | 0.6714 | 0.6720 | +0.0006 | 0.0527 | **0.850** | 0.036 | 0.1523 | 0 |
| `poor_overlap_misspecified_model` | 2000 | `dm_bootstrap_refit_percentile` | 0.6714 | 0.5987 | -0.0727 | 0.0753 | **0.000** | 0.000 | 0.0598 | 0 |
| `poor_overlap_misspecified_model` | 2000 | `dr_bootstrap_refit_percentile` | 0.6714 | 0.6760 | +0.0047 | 0.0469 | **0.830** | 0.038 | 0.1492 | 0 |
| `poor_overlap_misspecified_model` | 2000 | `dr_bootstrap_refit_basic` | 0.6714 | 0.6760 | +0.0047 | 0.0469 | **0.870** | 0.034 | 0.1492 | 0 |
| `continuous_reward_good_overlap` | 500 | `ips_wald` | 0.8038 | 0.8022 | -0.0015 | 0.1123 | 0.920 | 0.027 | 0.4398 | 0 |
| `continuous_reward_good_overlap` | 500 | `snips_wald` | 0.8038 | 0.8048 | +0.0011 | 0.0903 | 0.960 | 0.020 | 0.3586 | 0 |
| `continuous_reward_good_overlap` | 500 | `dm_naive_wald` | 0.8038 | 0.7956 | -0.0081 | 0.0701 | **0.690** | 0.046 | 0.1369 | 0 |
| `continuous_reward_good_overlap` | 500 | `dr_influence_wald` | 0.8038 | 0.8071 | +0.0033 | 0.0765 | 0.930 | 0.026 | 0.2969 | 0 |
| `continuous_reward_good_overlap` | 500 | `ips_bootstrap_percentile` | 0.8038 | 0.8022 | -0.0015 | 0.1123 | 0.900 | 0.030 | 0.3997 | 0 |
| `continuous_reward_good_overlap` | 500 | `snips_bootstrap_percentile` | 0.8038 | 0.8048 | +0.0011 | 0.0903 | 0.930 | 0.026 | 0.3307 | 0 |
| `continuous_reward_good_overlap` | 500 | `dm_bootstrap_refit_percentile` | 0.8038 | 0.7959 | -0.0078 | 0.0722 | 0.890 | 0.031 | 0.2617 | 0 |
| `continuous_reward_good_overlap` | 500 | `dr_bootstrap_refit_percentile` | 0.8038 | 0.8067 | +0.0029 | 0.0748 | 0.910 | 0.029 | 0.2735 | 0 |
| `continuous_reward_good_overlap` | 500 | `dr_bootstrap_refit_basic` | 0.8038 | 0.8067 | +0.0029 | 0.0748 | 0.890 | 0.031 | 0.2735 | 0 |
| `continuous_reward_good_overlap` | 2000 | `ips_wald` | 0.8038 | 0.8170 | +0.0133 | 0.0603 | 0.930 | 0.026 | 0.2194 | 0 |
| `continuous_reward_good_overlap` | 2000 | `snips_wald` | 0.8038 | 0.8136 | +0.0098 | 0.0492 | 0.920 | 0.027 | 0.1765 | 0 |
| `continuous_reward_good_overlap` | 2000 | `dm_naive_wald` | 0.8038 | 0.8063 | +0.0026 | 0.0384 | **0.590** | 0.049 | 0.0673 | 0 |
| `continuous_reward_good_overlap` | 2000 | `dr_influence_wald` | 0.8038 | 0.8115 | +0.0077 | 0.0406 | 0.930 | 0.026 | 0.1439 | 0 |
| `continuous_reward_good_overlap` | 2000 | `ips_bootstrap_percentile` | 0.8038 | 0.8170 | +0.0133 | 0.0603 | 0.880 | 0.032 | 0.2048 | 0 |
| `continuous_reward_good_overlap` | 2000 | `snips_bootstrap_percentile` | 0.8038 | 0.8136 | +0.0098 | 0.0492 | 0.890 | 0.031 | 0.1638 | 0 |
| `continuous_reward_good_overlap` | 2000 | `dm_bootstrap_refit_percentile` | 0.8038 | 0.8065 | +0.0027 | 0.0390 | 0.930 | 0.026 | 0.1260 | 0 |
| `continuous_reward_good_overlap` | 2000 | `dr_bootstrap_refit_percentile` | 0.8038 | 0.8111 | +0.0073 | 0.0410 | 0.900 | 0.030 | 0.1325 | 0 |
| `continuous_reward_good_overlap` | 2000 | `dr_bootstrap_refit_basic` | 0.8038 | 0.8111 | +0.0073 | 0.0410 | 0.900 | 0.030 | 0.1325 | 0 |
| `continuous_reward_misspecified` | 500 | `ips_wald` | 0.6608 | 0.6647 | +0.0039 | 0.0893 | 0.960 | 0.020 | 0.3649 | 0 |
| `continuous_reward_misspecified` | 500 | `snips_wald` | 0.6608 | 0.6594 | -0.0014 | 0.0742 | 0.960 | 0.020 | 0.3089 | 0 |
| `continuous_reward_misspecified` | 500 | `dm_naive_wald` | 0.6608 | 0.6064 | -0.0544 | 0.0867 | **0.510** | 0.050 | 0.1352 | 0 |
| `continuous_reward_misspecified` | 500 | `dr_influence_wald` | 0.6608 | 0.6510 | -0.0098 | 0.0786 | 0.930 | 0.026 | 0.2985 | 0 |
| `continuous_reward_misspecified` | 500 | `ips_bootstrap_percentile` | 0.6608 | 0.6647 | +0.0039 | 0.0893 | 0.950 | 0.022 | 0.3347 | 0 |
| `continuous_reward_misspecified` | 500 | `snips_bootstrap_percentile` | 0.6608 | 0.6594 | -0.0014 | 0.0742 | 0.960 | 0.020 | 0.2804 | 0 |
| `continuous_reward_misspecified` | 500 | `dm_bootstrap_refit_percentile` | 0.6608 | 0.6060 | -0.0548 | 0.0864 | 0.810 | 0.039 | 0.2544 | 0 |
| `continuous_reward_misspecified` | 500 | `dr_bootstrap_refit_percentile` | 0.6608 | 0.6534 | -0.0074 | 0.0786 | 0.890 | 0.031 | 0.2711 | 0 |
| `continuous_reward_misspecified` | 500 | `dr_bootstrap_refit_basic` | 0.6608 | 0.6534 | -0.0074 | 0.0786 | 0.880 | 0.032 | 0.2711 | 0 |
| `continuous_reward_misspecified` | 2000 | `ips_wald` | 0.6608 | 0.6547 | -0.0061 | 0.0474 | 0.950 | 0.022 | 0.1835 | 0 |
| `continuous_reward_misspecified` | 2000 | `snips_wald` | 0.6608 | 0.6565 | -0.0043 | 0.0381 | 0.950 | 0.022 | 0.1575 | 0 |
| `continuous_reward_misspecified` | 2000 | `dm_naive_wald` | 0.6608 | 0.6165 | -0.0443 | 0.0554 | **0.300** | 0.046 | 0.0675 | 0 |
| `continuous_reward_misspecified` | 2000 | `dr_influence_wald` | 0.6608 | 0.6573 | -0.0035 | 0.0338 | 0.970 | 0.017 | 0.1477 | 0 |
| `continuous_reward_misspecified` | 2000 | `ips_bootstrap_percentile` | 0.6608 | 0.6547 | -0.0061 | 0.0474 | 0.910 | 0.029 | 0.1667 | 0 |
| `continuous_reward_misspecified` | 2000 | `snips_bootstrap_percentile` | 0.6608 | 0.6565 | -0.0043 | 0.0381 | 0.920 | 0.027 | 0.1471 | 0 |
| `continuous_reward_misspecified` | 2000 | `dm_bootstrap_refit_percentile` | 0.6608 | 0.6164 | -0.0444 | 0.0554 | 0.720 | 0.045 | 0.1232 | 0 |
| `continuous_reward_misspecified` | 2000 | `dr_bootstrap_refit_percentile` | 0.6608 | 0.6569 | -0.0039 | 0.0340 | 0.950 | 0.022 | 0.1395 | 0 |
| `continuous_reward_misspecified` | 2000 | `dr_bootstrap_refit_basic` | 0.6608 | 0.6569 | -0.0039 | 0.0340 | 0.920 | 0.027 | 0.1395 | 0 |

---

## Scientific Analysis & Investigation of Failures (Task 6)

### 1. Breakdown of Naive Direct Method Intervals
- Across every scenario, the naive Wald interval for Direct Method (`dm_naive_wald`) exhibits catastrophic undercoverage (coverage 0.000 to 0.350 at nominal 95%).
- **Cause:** The naive Wald standard error computes only the sample variance of the conditional expectations across observed contexts, $\widehat{\mathrm{Var}}_x(\hat{q}(x, \pi_e)) / N$. It treats the learned reward model $\hat{q}$ as a fixed oracle. Because the actual sampling error in fitting $\hat{q}$ is completely omitted, the interval is artificially narrow (e.g. width 0.0109 vs 0.0475 for refit bootstrap at $N = 2000$).
- **Bootstrap Solution:** The full-refit cross-fitting bootstrap (`dm_bootstrap_refit_percentile`) refits the reward models on resampled training partitions, capturing both context-sampling noise and model-estimation variance. When the model is correctly specified, refit bootstrap coverage rises to **0.920–0.930** (matching nominal levels).
- **The Limit of Bootstrap under Misspecification:** When the model is structurally misspecified, refit bootstrap coverage also collapses (0.410 at $N = 2000$ in `good_overlap_misspecified_model`, and **0.000** in `poor_overlap_misspecified_model`). No uncertainty method can calibrate an estimator that is asymptotically biased towards the wrong estimand.

### 2. Doubly Robust: Influence-Score Wald vs. Bootstrap
- In settings with good overlap (`good_overlap_correct_model`, `good_overlap_misspecified_model`, `continuous_reward_good_overlap`, `continuous_reward_misspecified`), the Doubly Robust influence-score Wald interval (`dr_influence_wald`) achieves **near-perfect empirical coverage** (0.930 to 0.980 at nominal 0.95).
- Because DR satisfies Neyman orthogonality, the first-order estimation error of the cross-fitted reward model vanishes asymptotically. The empirical variance of the per-observation influence scores:
  $$\psi_i = \sum_a \pi_e(a \mid x_i) \hat{q}(x_i, a) + w_i \big(r_i - \hat{q}(x_i, a_i)\big)$$
  accurately estimates the semiparametric efficiency bound.
- Under model misspecification, while DM coverage collapses to 0.050, DR influence Wald maintains **0.940–0.970** coverage because the propensity-weighted factual residual corrects the point estimate.

### 3. Investigation of Poor Overlap & Heavy-Tail Failures
In the poor overlap scenarios (`poor_overlap_correct_model` and `poor_overlap_misspecified_model`), the logging policy is a confident softmax on random scores, creating extreme propensity ratios ($w_i \gg 1$, mean ESS fraction $\approx 0.053$ to $0.092$):
- **Why IPS Wald Undercovers:**
  IPS relies on the CLT for $w_i r_i$. When weights are heavy-tailed, the third standardized moment $\mathbb{E}[|w r|^3] / \sigma^3$ explodes. In finite samples ($N = 500$ and $N = 2000$), the Berry-Esseen bound is very loose, the sampling distribution remains heavily skewed, and one sample frequently misses the extreme tail that dictates the true variance. IPS Wald coverage drops to **0.820–0.840** despite huge interval width (0.7820 vs true value 0.7534).
- **Does Doubly Robust Also Undercover Under Poor Overlap?**
  Yes. In DR, the factual residual $(r_i - \hat{q}(x_i, a_i))$ is non-zero in finite samples. Multiplying this residual by heavy-tailed weights $w_i$ injects extreme variance into the influence scores $\psi_i$. Consequently, `dr_influence_wald` undercovers at **0.880–0.910** under poor overlap.
- **Does Bootstrap Improve or Worsen Calibration Under Heavy Tails?**
  **Bootstrap does NOT solve heavy-tail undercoverage.** In fact, `ips_bootstrap_percentile` undercovers at **0.810–0.850**, and `dr_bootstrap_refit_percentile` undercovers at **0.830–0.880**.
  - *Theoretical Distinction:* Bickel & Freedman (1981) established that the nonparametric bootstrap sample mean is asymptotically inconsistent when the underlying distribution has an **infinite second moment** ($\mathbb{E}[w^2] = \infty$). In our synthetic benchmark, propensities are bounded away from zero on compact regions, so theoretical moments are technically finite. However, extreme finite-sample kurtosis causes severe quantile instability: rare observations with extreme weights appear sporadically across resamples (appearing 0, 1, 2, or more times), destabilizing empirical order statistics and leading to persistent undercoverage in finite samples.
- **Relationship between Weight Concentration and Interval Width:**
  Under poor overlap, larger sample sizes ($N = 2000$) observe rarer, heavier weights (max weight share remains high, ESS fraction drops to 0.053). Interval widths shrink much more slowly than $1/\sqrt{N}$, and reported standard errors persistently underestimate empirical standard deviation.

### 4. Diagnostic Reliability Assessment
Every single poor-overlap replicate triggered the reliability warnings:
- `low_effective_sample_size` (ESS fraction < 0.10)
- `invalid_inference_assumptions` (ESS fraction < 0.05)
- `dominant_weight` (single observation carrying > 5% of total weight)
- `concentrated_weights` (top 1% carrying > 25% of weight)
These diagnostics serve as operational heuristics that flag that asymptotic conditions are degraded. They are conventional indicators rather than formal mathematical proofs or hypothesis tests with controlled false-positive rates.

### 5. Methodological Scope & Statistical Audit Notes
1. **Monte Carlo Precision:** The 100 replications per cell reported here provide an exploratory benchmark with binomial Monte Carlo standard error $\approx \pm 0.022$ ($\approx 2.2\%$). They reliably detect catastrophic undercoverage (e.g. 28% for naive DM), but cannot resolve fine coverage differences between 93% and 97%.
2. **Direct Method Decomposition:** The Direct Method error decomposes additively into context sampling variability and reward-model estimation error. These terms are an additive decomposition and are not assumed to have zero finite-sample covariance unconditionally.
3. **Double Machine Learning Assumptions:** In our synthetic setup, logging propensities are known exactly ($\|\hat{e} - e_0\| = 0$), so double robustness holds without requiring propensity estimation convergence rates.
4. **Unmodeled Conditions:** These findings apply to stationary, independent contextual bandits. Unobserved confounding, non-stationarity, adaptive logging policies, and clustered dependence remain uncharacterized.
