r"""Nonparametric bootstrap for off-policy contextual-bandit uncertainty estimation.

In this framework, observations are drawn i.i.d. from a stationary contextual-bandit
environment :math:`P(x, a, r)`. The resampling unit is the full observation tuple:

.. math::
    u_i = (x_i, a_i, r_i, p_i) \in \mathbb{R}^D \times \{0, \dots, K-1\} \times \mathbb{R}
    \times (0, 1].

Two bootstrap modes are supported:

1. **Fixed-model bootstrap (`refit=False`)**:
   Resamples observation rows while holding the precomputed reward-model predictions
   :math:`\hat{Q}` fixed. This isolates context- and reward-sampling variation from model
   estimation noise.

2. **Full-refit cross-fitting bootstrap (`refit=True`)**:
   In each bootstrap draw :math:`b \in \{1, \dots, B\}`, the resampled batch
   :math:`\mathcal{D}^{(b)}` is partitioned into :math:`K`-fold disjoint subsets, and the
   reward model is trained from scratch out-of-fold. This captures the complete end-to-end
   variability of the learning and evaluation pipeline.

Supported interval constructions:
- **Percentile interval**: :math:`[q_{\alpha/2}^B, \, q_{1-\alpha/2}^B]`.
- **Basic (empirical quantile / pivotal) interval**:
  :math:`[2\hat V - q_{1-\alpha/2}^B, \, 2\hat V - q_{\alpha/2}^B]`.
- **Normal bootstrap interval**: :math:`\hat V \pm z_{1-\alpha/2} \, \widehat{\mathrm{sd}}_B`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import (
    FloatArray,
    validate_predicted_rewards,
)
from reversallab.evaluation.cross_fitting import cross_fit_reward_model
from reversallab.evaluation.estimators import (
    _normal_quantile,
    _require_two_samples,
    dm,
    dr,
    ips,
    snips,
)
from reversallab.evaluation.reward_models import RewardModel
from reversallab.evaluation.weights import (
    validate_evaluation_action_dist,
)
from reversallab.exceptions import DegenerateWeightsError


@dataclass(frozen=True)
class BootstrapResult:
    """Nonparametric bootstrap results and confidence intervals."""

    estimator: str
    point_estimate: float
    bootstrap_estimates: FloatArray
    bootstrap_mean: float
    bootstrap_standard_error: float
    percentile_ci: tuple[float, float]
    basic_ci: tuple[float, float]
    normal_ci: tuple[float, float]
    confidence_level: float
    n_bootstraps: int
    refit_model: bool

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Default to the percentile interval."""
        return self.percentile_ci


def resample_logged_data(
    data: LoggedBanditData,
    rng: np.random.Generator,
) -> tuple[LoggedBanditData, np.ndarray]:
    """Sample N observation rows uniformly at random with replacement.

    Parameters
    ----------
    data:
        Logged contextual bandit dataset of size N.
    rng:
        NumPy random generator.

    Returns
    -------
    tuple[LoggedBanditData, np.ndarray]:
        Resampled dataset and the 1D integer index array used to draw it.
    """
    n = data.n_samples
    indices = rng.integers(0, n, size=n)

    resampled = LoggedBanditData(
        context=data.context[indices],
        action=data.action[indices],
        reward=data.reward[indices],
        logging_propensity=data.logging_propensity[indices],
        n_actions=data.n_actions,
        logging_action_dist=(
            data.logging_action_dist[indices] if data.logging_action_dist is not None else None
        ),
    )
    return resampled, indices


def construct_bootstrap_intervals(
    point_estimate: float,
    bootstrap_estimates: FloatArray,
    confidence_level: float = 0.95,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute percentile, basic, and normal bootstrap confidence intervals.

    Parameters
    ----------
    point_estimate:
        Estimate computed on the original full sample.
    bootstrap_estimates:
        1D float array of estimates across B bootstrap replications.
    confidence_level:
        Target nominal coverage level in (0, 1). Default 0.95.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        ``(percentile_ci, basic_ci, normal_ci)``.
    """
    z = _normal_quantile(confidence_level)
    alpha = 1.0 - confidence_level

    clean_estimates = bootstrap_estimates[np.isfinite(bootstrap_estimates)]
    if len(clean_estimates) < 2:
        return (
            (float("nan"), float("nan")),
            (float("nan"), float("nan")),
            (float("nan"), float("nan")),
        )

    # 1. Percentile interval: [q_{alpha/2}, q_{1 - alpha/2}]
    q_low = float(np.percentile(clean_estimates, 100.0 * (alpha / 2.0)))
    q_high = float(np.percentile(clean_estimates, 100.0 * (1.0 - alpha / 2.0)))
    percentile_ci = (q_low, q_high)

    # 2. Basic (empirical quantile / pivotal) interval:
    # [2*theta - q_{1 - alpha/2}, 2*theta - q_{alpha/2}]
    basic_ci = (2.0 * point_estimate - q_high, 2.0 * point_estimate - q_low)

    # 3. Normal bootstrap interval: theta +/- z * se_boot
    se_boot = float(np.std(clean_estimates, ddof=1))
    normal_ci = (point_estimate - z * se_boot, point_estimate + z * se_boot)

    return percentile_ci, basic_ci, normal_ci


def bootstrap_evaluate(
    estimator_name: str,
    data: LoggedBanditData,
    evaluation_action_dist: FloatArray,
    *,
    model_factory: Callable[[], RewardModel] | None = None,
    predicted_rewards: FloatArray | None = None,
    n_bootstraps: int = 200,
    n_folds: int = 5,
    confidence_level: float = 0.95,
    refit: bool = True,
    seed: int = 0,
) -> BootstrapResult:
    r"""Run nonparametric bootstrap uncertainty estimation.

    Parameters
    ----------
    estimator_name:
        One of ``'ips'``, ``'snips'``, ``'dm'``, or ``'dr'``.
    data:
        Logged contextual bandit batch.
    evaluation_action_dist:
        Evaluation policy distribution of shape ``(N, K)``.
    model_factory:
        Factory returning a fresh un-fitted :class:`RewardModel`.
        Required for DM/DR if ``refit=True``.
    predicted_rewards:
        Precomputed predicted reward matrix of shape ``(N, K)``. Used for DM/DR if ``refit=False``
        and for computing the point estimate.
    n_bootstraps:
        Number of bootstrap replications :math:`B`. Default 200.
    n_folds:
        Number of folds used in cross-fitting when ``refit=True``. Default 5.
    confidence_level:
        Target nominal coverage in (0, 1). Default 0.95.
    refit:
        If ``True``, completely re-runs cross-fitting on each bootstrap sample for DM and DR.
        If ``False``, holds the provided ``predicted_rewards`` fixed.
    seed:
        Seed for deterministic resampling.
    """
    _require_two_samples(data)
    name = estimator_name.lower().strip()
    eval_dist = validate_evaluation_action_dist(data, evaluation_action_dist)

    # Compute base point estimate
    if name == "ips":
        base_est = ips(data, eval_dist, confidence_level=confidence_level).value
    elif name == "snips":
        base_est = snips(data, eval_dist, confidence_level=confidence_level).value
    elif name == "dm":
        if predicted_rewards is None and model_factory is not None:
            predicted_rewards = cross_fit_reward_model(
                data, model_factory, n_folds=n_folds, seed=seed
            )
        if predicted_rewards is None:
            raise ValueError("either predicted_rewards or model_factory must be supplied for DM")
        base_est = dm(data, eval_dist, predicted_rewards, confidence_level=confidence_level).value
    elif name == "dr":
        if predicted_rewards is None and model_factory is not None:
            predicted_rewards = cross_fit_reward_model(
                data, model_factory, n_folds=n_folds, seed=seed
            )
        if predicted_rewards is None:
            raise ValueError("either predicted_rewards or model_factory must be supplied for DR")
        base_est = dr(data, eval_dist, predicted_rewards, confidence_level=confidence_level).value
    else:
        raise ValueError(f"unknown estimator '{estimator_name}'; expected ips, snips, dm, or dr")

    # If predicted_rewards was validated/generated
    if predicted_rewards is not None:
        q_hat = validate_predicted_rewards(
            predicted_rewards, n_samples=data.n_samples, n_actions=data.n_actions
        )
    else:
        q_hat = None

    if name in {"dm", "dr"} and refit and model_factory is None:
        raise ValueError(f"model_factory is required to run refit bootstrap for '{name}'")

    boot_estimates = np.empty(n_bootstraps, dtype=np.float64)
    parent_ss = np.random.SeedSequence(seed)
    child_seeds = parent_ss.spawn(n_bootstraps)

    for b in range(n_bootstraps):
        rng_b = np.random.default_rng(child_seeds[b])
        boot_data, indices = resample_logged_data(data, rng_b)
        boot_eval_dist = eval_dist[indices]

        try:
            if name == "ips":
                val = ips(boot_data, boot_eval_dist, confidence_level=confidence_level).value
            elif name == "snips":
                val = snips(boot_data, boot_eval_dist, confidence_level=confidence_level).value
            elif name == "dm":
                if refit:
                    assert model_factory is not None
                    # Child seed for cross-fitting inside bootstrap replication
                    cf_seed = int(child_seeds[b].generate_state(1)[0])
                    boot_q = cross_fit_reward_model(
                        boot_data,
                        model_factory,
                        n_folds=n_folds,
                        seed=cf_seed,
                        groups=indices,
                    )
                else:
                    assert q_hat is not None
                    boot_q = q_hat[indices]
                val = dm(boot_data, boot_eval_dist, boot_q, confidence_level=confidence_level).value
            else:  # dr
                if refit:
                    assert model_factory is not None
                    cf_seed = int(child_seeds[b].generate_state(1)[0])
                    boot_q = cross_fit_reward_model(
                        boot_data,
                        model_factory,
                        n_folds=n_folds,
                        seed=cf_seed,
                        groups=indices,
                    )
                else:
                    assert q_hat is not None
                    boot_q = q_hat[indices]
                val = dr(boot_data, boot_eval_dist, boot_q, confidence_level=confidence_level).value
        except (DegenerateWeightsError, ValueError):
            val = float("nan")

        boot_estimates[b] = val

    clean = boot_estimates[np.isfinite(boot_estimates)]
    boot_mean = float(np.mean(clean)) if len(clean) > 0 else float("nan")
    boot_se = float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan")

    perc_ci, basic_ci, norm_ci = construct_bootstrap_intervals(
        base_est, boot_estimates, confidence_level=confidence_level
    )

    return BootstrapResult(
        estimator=name,
        point_estimate=base_est,
        bootstrap_estimates=boot_estimates,
        bootstrap_mean=boot_mean,
        bootstrap_standard_error=boot_se,
        percentile_ci=perc_ci,
        basic_ci=basic_ci,
        normal_ci=norm_ci,
        confidence_level=confidence_level,
        n_bootstraps=n_bootstraps,
        refit_model=(name in {"dm", "dr"} and refit),
    )
