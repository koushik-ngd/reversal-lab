# Experiment 002 — Open Bandit reference parity

Protocol: [`docs/experiment_protocol.md`](../../docs/experiment_protocol.md).
Estimand: [`docs/open_bandit_estimand.md`](../../docs/open_bandit_estimand.md).
File audit: [`docs/open_bandit_data_audit.md`](../../docs/open_bandit_data_audit.md).

```text
uv run reversallab compare-reference --config configs/experiments/002_open_bandit_parity.yaml
```

The command needs a local checkout of `st-tech/zr-obp` at `data/raw/zr-obp` containing `obd/` and `obp/`. It does not download the full dataset. `runs/` is gitignored. The authoritative record is `run.json`.

## Run recorded here

Executed in this working tree on 2026-10-28. The M2 baseline commit is `2a83f9e184f51f1b9356f8d32ac7601266b22f2e`. This run's tree was dirty because the M3A sources were not yet a second commit.

| | |
|---|---|
| Run | `002_open_bandit_parity_20261028T125457097780Z_b4679eee` |
| `config_sha256` | `b4679eee4826b4f9a1bb26feda3dec78a79b780ddc3726c3260ad74a9b91c735` |
| `results_sha256` | `dc6043c8ecc56d251d4762b7168d89287be39538b46fc8805dc6c8a26fc8937b` |
| `uv.lock` SHA-256 | `cebfdbf689715c076ba81ec64def9ad8839107cb5727afa95d44b7a3f181b53f` |
| Python / NumPy | 3.12.13 / 2.5.4 |
| OBP source commit | `8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4` (clean) |
| `estimators.py` SHA-256 | `6e72fa21b81ef731d033ccea7e37bef4ab12347bafccf023e7671f3ca32b16ad` |
| OBP `lambda_` | infinity (no weight clipping) |
| Tolerance | absolute `1e-12`, relative `1e-12` |
| Comparisons | 30 |
| Mismatches | 0 |
| Largest absolute difference | `1.734723475976807e-18` |
| Runtime | 7.9 s |

OBP 0.5.5 declares Python `<3.10` and imports PyTorch, pandas, SciPy, and scikit-learn from its package init. This process is Python 3.12, and SciPy's compiled extension was blocked by Windows Application Control. The comparison therefore loads `InverseProbabilityWeighting.estimate_policy_value` and `SelfNormalizedInverseProbabilityWeighting.estimate_policy_value` from that commit in a subprocess, after substituting empty modules for the imports those two methods do not call. `check_ope_inputs` and the weight arithmetic are the upstream functions. `lambda_` is required to be infinite; a clipped OBP estimator would be refused.

Both sides receive the adapter's reward, action, and propensity arrays and the same `(N, K)` candidate matrix. OBP's tensor is that matrix with a length-1 list axis. OBP's position index is 0. The ZOZOTOWN position is not that index.

## Pooled estimates

`position = all` uses every row. For the random logger and the uniform candidate, IPS and SNIPS equal the click rate. For BTS, the uniform candidate is a different policy, so the estimate is not the click rate.

| Logger | Campaign | Candidate | IPS = OBP IPW | SNIPS = OBP SNIPW |
|---|---|---|---:|---:|
| random | all | uniform | 0.0038 | 0.0038 |
| random | all | always item 0 | 0 | 0 |
| random | men | uniform | 0.0046 | 0.0046 |
| random | men | always item 0 | 0.0136 | 0.0147059 |
| random | women | uniform | 0.0046 | 0.0046 |
| random | women | always item 0 | 0 | 0 |
| bts | all | uniform | 0.00235964 | 0.00233371 |
| bts | all | always item 0 | 0 | 0 |
| bts | men | uniform | 0.00300863 | 0.00318942 |
| bts | men | always item 0 | 0.0106084 | 0.0105246 |
| bts | women | uniform | 0.00743758 | 0.00237305 |
| bts | women | always item 0 | 0 | 0 |

The zeros are the estimate of "always recommend item 0" on a file in which that item's reweighted clicks are zero. They are not a published CTR.

## Position slices

Each raw position was evaluated under the uniform candidate only. All 18 slice comparisons matched within the tolerance. On random/all the three slice values, which equal the slice click rates, are 0.00391331 (left), 0.00410317 (center), and 0.00336803 (right). Those are different estimands from the pooled value 0.0038. The full slice table is in the run's `results.comparisons`.

## What this run does not show

It does not reproduce a published full-dataset benchmark. It does not estimate interval coverage. The BTS rows still have unverifiable support. Repeated user-feature hashes are recorded and unused by the standard error, and this run does not report a standard error.
