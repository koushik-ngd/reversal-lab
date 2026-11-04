# Bootstrap Sensitivity Benchmark: B in {50, 200, 1000}

This document reports the targeted sensitivity experiment evaluating the stability of nonparametric bootstrap confidence intervals as the number of bootstrap replications $B$ increases from 50 to 200 and 1000.

## Benchmark Design
- **Scenarios:**
  - `good_overlap`: 3 actions, context dimension 2, binary rewards, softmax logging ($\beta = 0.5$), softmax evaluation ($\beta = 1.0$), true value $V = 0.55646 \pm 0.00021$.
  - `poor_overlap`: 3 actions, context dimension 2, binary rewards, anti-informed softmax logging ($\beta = -0.5$), concentrated softmax evaluation ($\beta = 1.5$), true value $V = 0.57756 \pm 0.00011$.
- **Sample Size:** $N = 500$.
- **Nominal Level:** 95% ($\alpha = 0.05$).
- **Replications:**
  - $M = 500$ independent Monte Carlo replications for IPS and SNIPS across $B \in \{50, 200, 1000\}$ ($\text{MCSE} \le 0.012$).
  - $M = 10$ paired replications for DR Refit Bootstrap across $B \in \{50, 200, 1000\}$ measuring endpoint discrepancies $|L_B - L_{1000}|$ and $|U_B - U_{1000}|$.
  - $M = 50$ independent replications for DR Refit Bootstrap comparing $B = 50$ vs $B = 200$.

---

## 1. IPS & SNIPS Sensitivity Benchmark ($M = 500$ Replications)

| Scenario | Estimator / Interval | $B$ | Empirical Coverage | $\pm$ MCSE | Mean Width | Mean Lower | Mean Upper |
|---|---|---|---|---|---|---|---|
| **Good Overlap** | `ips_wald` | N/A | 0.9560 | 0.0092 | 0.0957 | 0.5085 | 0.6042 |
| | `ips_bootstrap_percentile` | 50 | 0.9200 | 0.0121 | 0.0878 | 0.5126 | 0.6004 |
| | `ips_bootstrap_percentile` | 200 | 0.9480 | 0.0099 | 0.0934 | 0.5097 | 0.6031 |
| | `ips_bootstrap_percentile` | 1000 | **0.9560** | 0.0092 | 0.0951 | 0.5089 | 0.6040 |
| | `snips_bootstrap_percentile` | 50 | 0.9280 | 0.0116 | 0.0817 | 0.5144 | 0.5961 |
| | `snips_bootstrap_percentile` | 200 | 0.9420 | 0.0105 | 0.0872 | 0.5117 | 0.5989 |
| | `snips_bootstrap_percentile` | 1000 | **0.9540** | 0.0094 | 0.0888 | 0.5109 | 0.5997 |
| **Poor Overlap** | `ips_wald` | N/A | 0.9300 | 0.0114 | 0.1787 | 0.4912 | 0.6699 |
| | `ips_bootstrap_percentile` | 50 | 0.9140 | 0.0125 | 0.1643 | 0.4988 | 0.6631 |
| | `ips_bootstrap_percentile` | 200 | 0.9280 | 0.0116 | 0.1746 | 0.4934 | 0.6680 |
| | `ips_bootstrap_percentile` | 1000 | **0.9340** | 0.0111 | 0.1776 | 0.4919 | 0.6695 |
| | `snips_bootstrap_percentile` | 50 | 0.9180 | 0.0123 | 0.1114 | 0.5195 | 0.6309 |
| | `snips_bootstrap_percentile` | 200 | 0.9420 | 0.0105 | 0.1181 | 0.5161 | 0.6342 |
| | `snips_bootstrap_percentile` | 1000 | **0.9460** | 0.0101 | 0.1202 | 0.5150 | 0.6352 |

---

## 2. Doubly Robust Refit Endpoint Discrepancy ($M = 10$, Paired Datasets)

| Metric | Good Overlap ($B=50$ vs $1000$) | Good Overlap ($B=200$ vs $1000$) | Poor Overlap ($B=50$ vs $1000$) | Poor Overlap ($B=200$ vs $1000$) |
|---|---|---|---|---|
| Mean Absolute Lower Diff $\mathbb{E}[\|L_B - L_{1000}\|]$ | 0.0057 | **0.0034** | 0.0112 | **0.0044** |
| Mean Absolute Upper Diff $\mathbb{E}[\|U_B - U_{1000}\|]$ | 0.0051 | **0.0033** | 0.0082 | **0.0053** |
| Mean Interval Width | $B=50$: 0.0821<br>$B=1000$: 0.0882 | $B=200$: 0.0894<br>$B=1000$: 0.0882 | $B=50$: 0.1108<br>$B=1000$: 0.1188 | $B=200$: 0.1160<br>$B=1000$: 0.1188 |
| Mean Runtime per Dataset | 0.52s | 2.07s | 0.79s | 3.05s |

*(Runtime for $B=1000$: 10.36s on Good Overlap, 15.77s on Poor Overlap).*

---

## 3. Doubly Robust Refit Coverage Sensitivity ($M = 50$ Replications)

| Scenario | $B$ | Empirical Coverage | $\pm$ MCSE | Mean Width |
|---|---|---|---|---|
| **Good Overlap** | 50 | 0.8800 | 0.0460 | 0.0832 |
| | 200 | 0.9000 | 0.0424 | 0.0872 |
| **Poor Overlap** | 50 | 0.9200 | 0.0384 | 0.1098 |
| | 200 | 0.9600 | 0.0277 | 0.1194 |

---

## Key Conclusions
1. **$B = 50$ is Exploratory:** At $B = 50$, the empirical 2.5% quantile interpolates between the 1st and 2nd smallest bootstrap replicates. This produces quantile jitter ($\approx \pm 0.006$ to $0.011$ on endpoints) and artificial width contraction of roughly 7–10%, lowering empirical coverage by 2–4 percentage points compared to larger $B$.
2. **$B = 200$ vs $B = 1000$ Monte Carlo Precision:** Increasing $B$ to 200 reduces quantile discrepancy ($|dL| \approx 0.0034$) and expands interval width to proper scale. Increasing to $B = 1000$ further reduces bootstrap Monte Carlo sampling noise, but does **not** guarantee nominal coverage: coverage depends fundamentally on the sample size $N$, overlap, and estimator bias/variance properties.
3. **Computational Scalability & Context-Dependent Choices:** While IPS/SNIPS bootstrap is millisecond-fast even at $B = 1000$, DR refit cross-fitting scales linearly with $B \times K$ model fits. In our synthetic configuration, $B = 200$ provided a practical computational trade-off (taking $\approx 2$–$3$ seconds per evaluation vs $\approx 10$–$16$ seconds for $B = 1000$), but $B$ must be calibrated to the specific tail behavior and accuracy requirements of the target application rather than assumed to be universally production-ready.
