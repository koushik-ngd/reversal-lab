# Experiment 001 — Estimator validation

Protocol: [`docs/experiment_protocol.md`](../../docs/experiment_protocol.md).

```text
uv run reversallab validate-estimators --config configs/experiments/001_estimator_validation.yaml
```

`runs/` is gitignored. The authoritative record is `run.json`.

## Run recorded here

Executed in this working tree on 2026-10-28.

| | |
|---|---|
| Run | `001_estimator_validation_20261028T115521877076Z_4b6f33ae` |
| `config_sha256` | `4b6f33ae11870b14b53904293c9ff78152b1d09a74389e54b2bdd243465d92ea` |
| `results_sha256` | `7112fe2c80e886c5738aaf8d49b17cdc76bf48c7d3a654ec6890bff13ebf695c` |
| Python / NumPy | 3.12.13 / 2.5.4 |
| Wall time | 151.2 s |
| Git commit | unavailable (this tree has no commit yet) |
| Run-level warnings | none |
| Failed replicates | 0 in every cell |

An earlier execution of the same config, `001_estimator_validation_20261028T114303606895Z_4b6f33ae`, produced the same `results_sha256`. Runtime is not part of that hash. A second machine is not claimed to match.

Reference values below are the Monte Carlo policy values from that file (standard errors about \(1 \times 10^{-4}\)). Table entries are rounded from `run.json`. Bias SE is the Monte Carlo standard error of the mean estimate across 500 replications. Coverage is for the nominal 95% Wald interval. ESS/N is the mean Kish effective-sample-size fraction.

True values of the evaluation policy: on-policy 0.7365, uniform logging 0.8391, moderate mismatch 0.8182, deterministic candidate 0.8583, inferior candidate 0.2154, poor overlap 0.8382. The logging policy's value is about 0.737 when it is the informed softmax, about 0.457 when it is uniform, and about 0.431 in the poor-overlap scenario. The inferior candidate is the only evaluation policy with a lower true value than its logger.

| scenario | n | estimator | bias | bias SE | RMSE | coverage | reported SE | empirical SD | ESS/N |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| on_policy | 500 | ips | +0.0005 | 0.0009 | 0.0202 | 0.936 | 0.0197 | 0.0202 | 1.000 |
| on_policy | 500 | snips | +0.0005 | 0.0009 | 0.0202 | 0.936 | 0.0197 | 0.0202 | 1.000 |
| on_policy | 2000 | ips | -0.0000 | 0.0004 | 0.0097 | 0.954 | 0.0098 | 0.0097 | 1.000 |
| on_policy | 2000 | snips | -0.0000 | 0.0004 | 0.0097 | 0.954 | 0.0098 | 0.0097 | 1.000 |
| on_policy | 8000 | ips | +0.0000 | 0.0002 | 0.0051 | 0.944 | 0.0049 | 0.0051 | 1.000 |
| on_policy | 8000 | snips | +0.0000 | 0.0002 | 0.0051 | 0.944 | 0.0049 | 0.0051 | 1.000 |
| uniform_logging | 500 | ips | +0.0028 | 0.0047 | 0.1055 | 0.952 | 0.1035 | 0.1055 | 0.146 |
| uniform_logging | 500 | snips | +0.0008 | 0.0016 | 0.0366 | 0.930 | 0.0369 | 0.0366 | 0.146 |
| uniform_logging | 2000 | ips | -0.0001 | 0.0023 | 0.0524 | 0.952 | 0.0518 | 0.0525 | 0.146 |
| uniform_logging | 2000 | snips | -0.0019 | 0.0008 | 0.0190 | 0.932 | 0.0189 | 0.0189 | 0.146 |
| uniform_logging | 8000 | ips | -0.0010 | 0.0012 | 0.0261 | 0.948 | 0.0259 | 0.0261 | 0.146 |
| uniform_logging | 8000 | snips | -0.0004 | 0.0004 | 0.0096 | 0.946 | 0.0094 | 0.0096 | 0.146 |
| moderate_mismatch | 500 | ips | +0.0017 | 0.0022 | 0.0494 | 0.928 | 0.0475 | 0.0495 | 0.439 |
| moderate_mismatch | 500 | snips | -0.0018 | 0.0013 | 0.0297 | 0.926 | 0.0277 | 0.0297 | 0.439 |
| moderate_mismatch | 2000 | ips | -0.0007 | 0.0011 | 0.0255 | 0.928 | 0.0237 | 0.0255 | 0.438 |
| moderate_mismatch | 2000 | snips | -0.0006 | 0.0007 | 0.0151 | 0.942 | 0.0140 | 0.0151 | 0.438 |
| moderate_mismatch | 8000 | ips | -0.0001 | 0.0005 | 0.0117 | 0.952 | 0.0119 | 0.0117 | 0.437 |
| moderate_mismatch | 8000 | snips | -0.0000 | 0.0003 | 0.0070 | 0.944 | 0.0070 | 0.0070 | 0.437 |
| deterministic_candidate | 500 | ips | +0.0054 | 0.0054 | 0.1218 | 0.948 | 0.1254 | 0.1218 | 0.101 |
| deterministic_candidate | 500 | snips | -0.0007 | 0.0022 | 0.0485 | 0.924 | 0.0485 | 0.0485 | 0.101 |
| deterministic_candidate | 2000 | ips | -0.0004 | 0.0028 | 0.0626 | 0.962 | 0.0626 | 0.0626 | 0.100 |
| deterministic_candidate | 2000 | snips | +0.0002 | 0.0011 | 0.0248 | 0.948 | 0.0246 | 0.0248 | 0.100 |
| deterministic_candidate | 8000 | ips | +0.0016 | 0.0014 | 0.0315 | 0.936 | 0.0313 | 0.0314 | 0.100 |
| deterministic_candidate | 8000 | snips | -0.0000 | 0.0005 | 0.0119 | 0.956 | 0.0123 | 0.0120 | 0.100 |
| inferior_candidate | 500 | ips | -0.0088 | 0.0033 | 0.0750 | 0.764 | 0.0585 | 0.0745 | 0.036 |
| inferior_candidate | 500 | snips | +0.0212 | 0.0043 | 0.0973 | 0.856 | 0.0732 | 0.0951 | 0.036 |
| inferior_candidate | 2000 | ips | +0.0014 | 0.0031 | 0.0684 | 0.808 | 0.0416 | 0.0685 | 0.020 |
| inferior_candidate | 2000 | snips | +0.0122 | 0.0030 | 0.0670 | 0.880 | 0.0498 | 0.0660 | 0.020 |
| inferior_candidate | 8000 | ips | -0.0012 | 0.0012 | 0.0263 | 0.876 | 0.0224 | 0.0263 | 0.013 |
| inferior_candidate | 8000 | snips | +0.0012 | 0.0017 | 0.0374 | 0.926 | 0.0300 | 0.0374 | 0.013 |
| poor_overlap | 500 | ips | +0.6424 | 0.5602 | 12.5297 | 0.522 | 1.1177 | 12.5257 | 0.021 |
| poor_overlap | 500 | snips | -0.0148 | 0.0062 | 0.1402 | 0.676 | 0.0807 | 0.1395 | 0.021 |
| poor_overlap | 2000 | ips | +0.0318 | 0.0417 | 0.9329 | 0.632 | 0.3799 | 0.9333 | 0.010 |
| poor_overlap | 2000 | snips | -0.0096 | 0.0049 | 0.1104 | 0.762 | 0.0667 | 0.1101 | 0.010 |
| poor_overlap | 8000 | ips | -0.0076 | 0.0291 | 0.6491 | 0.654 | 0.2486 | 0.6497 | 0.005 |
| poor_overlap | 8000 | snips | -0.0070 | 0.0032 | 0.0723 | 0.842 | 0.0509 | 0.0720 | 0.005 |

## Reading

On policy, IPS and SNIPS are the same estimator: every weight is 1, disagreement is 0, ESS/N is 1, and no diagnostic warning fired. Bias is within one Monte Carlo standard error of zero. Coverage is 0.936 to 0.954. The reported standard error matches the cross-replication standard deviation.

With uniform logging, weights are bounded by \(K = 10\). IPS bias is within one standard error of zero at each sample size, and IPS coverage is 0.948 to 0.952. RMSE falls as \(N\) grows. SNIPS has smaller RMSE in every one of those cells. Its coverage is 0.930 to 0.946.

The poor-overlap logger is a confident softmax on reward-independent scores, and the candidate is near-greedy on the true reward. Support did not fail in any replicate. IPS at \(N = 500\) has mean 1.48 against a true value of 0.84, with a Monte Carlo standard error of 0.56, so 500 replications do not locate the bias. The RMSE is 12.5 because a single importance weight can be enormous (the largest weight observed in that cell is on the order of \(10^5\)). The mean reported standard error is 1.12, against an empirical standard deviation of 12.5. Coverage is 0.522, 0.632, and 0.654 as \(N\) goes from 500 to 8000. By \(N = 8000\) the IPS bias is \(-0.0076 \pm 0.029\), which is compatible with zero, while the interval still undercovers. ESS/N falls from 0.021 to 0.005: larger samples include rarer, heavier weights. SNIPS keeps RMSE near 0.07–0.14. Its bias at \(N = 8000\) is \(-0.0070 \pm 0.0032\), about two standard errors, which is the finite-sample bias of a ratio estimator and is small next to the IPS dispersion. SNIPS coverage rises from 0.676 to 0.842 and is still short of 95%. Every poor-overlap replicate triggered the low-ESS warning, and 93–100% triggered the dominant-weight warning.

The inferior candidate (softmax inverse temperature \(-1\) against a logger at \(+1\)) is the same phenomenon at a milder setting. ESS/N is 0.036, 0.020, and 0.013. IPS coverage is 0.764, 0.808, and 0.876. SNIPS bias at \(N = 500\) is \(+0.021 \pm 0.004\) and is \(+0.0012 \pm 0.0017\) at \(N = 8000\). Low ESS was warned on essentially every replicate at \(N \ge 2000\). The point estimates remain near 0.21, against a logging-policy value near 0.74, so the direction of the comparison is visible in this particular design. The interval is not calibrated. That is a property of these weights, not a second ground-truth failure of the point estimator.

No cell is a comparison with Open Bandit Pipeline, a real log, Direct Method, or Doubly Robust. Nothing here is a result about decision reversal.
