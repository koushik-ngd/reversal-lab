# Methodology

## Setting

Single-position contextual bandit with \(K\) discrete actions. Each logged round:

1. A context \(x \sim P(x)\) is observed.
2. The logging policy \(\pi_0\) draws an action \(a \sim \pi_0(\cdot \mid x)\).
3. A reward \(r \sim P(r \mid x, a)\) is observed for that action only.

The log is \(\mathcal{D} = \{(x_i, a_i, r_i, p_i)\}_{i=1}^N\) with \(p_i = \pi_0(a_i \mid x_i)\).
The target quantity is the value of an evaluation (candidate) policy \(\pi_e\):

\[
V(\pi_e) = \mathbb{E}_{x}\Big[\sum_a \pi_e(a \mid x)\, q(x, a)\Big], \qquad q(x,a) = \mathbb{E}[r \mid x, a].
\]

## Estimators (implemented)

Importance weight: \(w_i = \pi_e(a_i \mid x_i) / p_i\).

**IPS**: \(\hat V_{\text{IPS}} = \frac{1}{N}\sum_i w_i r_i\).
It is unbiased when (i) observations are i.i.d., (ii) the propensities are correct, and
(iii) support holds, meaning \(\pi_e(a \mid x) > 0 \Rightarrow \pi_0(a \mid x) > 0\) for all \(x, a\).
Its standard error is \(\widehat{\mathrm{sd}}(w_i r_i)/\sqrt{N}\) (ddof = 1).

**SNIPS**: \(\hat V_{\text{SN}} = \sum_i w_i r_i / \sum_i w_i\).
It is a ratio estimator: biased with bias \(O(1/N)\), consistent under the same conditions, and
invariant to rescaling of the weights. It is undefined when \(\sum_i w_i = 0\), and
the code raises `DegenerateWeightsError` instead of returning a value in that case.
Its standard error comes from the delta method, using the \(1/N\) second-moment form
(not the Bessel \(1/(N-1)\) form used by the IPS standard error):
\(\sqrt{\sum_i w_i^2 (r_i - \hat V_{\text{SN}})^2} \,/\, \sum_i w_i\).

**Confidence intervals**: Wald intervals \(\hat V \pm z_{1-\alpha/2}\,\widehat{\mathrm{se}}\).
They are asymptotically valid under i.i.d. sampling and finite \(\mathrm{Var}(w r)\). They are
*not* reliable when weights are heavy-tailed (see Experiment 001).

## Diagnostics (implemented)

| Quantity | Definition | Interpretation |
|---|---|---|
| ESS | \((\sum w)^2 / \sum w^2\) (Kish) | Rough number of "effective" observations; heuristic, not an exact variance identity for IPS |
| ESS/N | ESS divided by N | Below `min_ess_fraction` (default 0.1) → warning |
| Max weight share | \(\max w / \sum w\) | One observation dominating the estimate |
| Mean weight | \(\bar w\); \(\mathbb{E}_{\pi_0}[w] = 1 - \text{unsupported mass}\) | A significant deviation suggests support or propensity problems; also fires under heavy tails (false alarms, see limitations) |
| Support | full \(\pi_0(\cdot\mid x_i)\) required | Verifiable only when the full logging distribution is logged |
| Unlogged-action mass | mean \(\pi_e\) mass on actions with zero count in the log | Empirical coverage gap |

The default thresholds are conventional heuristics and have not been validated.
M4 will treat them as baselines to compare against, not as ground truth.

## Synthetic ground truth

Contexts are \(x \sim \mathcal{N}(0, I_D)\). Reward logits are \(z(x,a) = x^\top W_a + b_a\), with
\(W \sim \mathcal{N}(0, s^2/D)\) and \(b \sim \mathcal{N}(0, s_b^2)\). Binary rewards are
\(r \sim \text{Bernoulli}(\sigma(z))\); Gaussian rewards are \(r \sim \mathcal{N}(z, \sigma_\epsilon^2)\).

The reference value \(V(\pi)\) is computed **exactly over actions** and by Monte Carlo
over an independent draw of \(M\) contexts (default \(10^6\)), with its Monte Carlo
standard error reported. The reference value uses a separate random stream from the
evaluation data. Where a closed form exists (Gaussian rewards, uniform policy:
\(V = \bar b\)), the Monte Carlo reference is tested against it.

Policies used in experiments are defined relative to the reward logits (an informed
policy) or to independent random logits (an uninformed policy):

* softmax with inverse temperature \(\beta\): \(\beta = 0\) is uniform, \(\beta < 0\) is anti-informed;
* \(\epsilon\)-greedy; \(\epsilon = 0\) is deterministic.

This gives policies with known performance ordering and controllable overlap.
The generator is not tuned towards any outcome; scenario parameters are recorded in
the config and hashed into the run identity.

## Planned methodology (M3–M4, not implemented)

* Direct Method and Doubly Robust estimation with cross-fitted reward models
  (out-of-fold predictions only), so that no observation's reward model was fit on it.
* Bootstrap intervals; cluster bootstrap over `group_id` when dependence exists.
* The decision rule for M4 is sketched in `experiment_protocol.md`. The perturbation
  set and the fragility metric are not defined yet. They will be formalised, and
  compared with simpler baselines, before any implementation is described as a result.

## Related work (to be expanded with verified bibliographic detail)

* Horvitz & Thompson (1952): inverse probability weighting.
* Dudík, Langford & Li (2011): doubly robust policy evaluation.
* Swaminathan & Joachims (2015): self-normalised estimator for counterfactual learning.
* Saito et al. (2021): Open Bandit Dataset and Open Bandit Pipeline (`zr-obp`).

IPS and SNIPS here are standard re-implementations, not contributions.
