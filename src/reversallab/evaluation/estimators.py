r"""Importance-sampling off-policy value estimators.

For logged data :math:`\{(x_i, a_i, r_i, p_i)\}_{i=1}^N` drawn i.i.d. from the logging
policy :math:`\pi_0` and weights :math:`w_i = \pi_e(a_i|x_i) / p_i`:

* IPS:   :math:`\hat V = \frac{1}{N}\sum_i w_i r_i` (unbiased under support).
* SNIPS: :math:`\hat V = \sum_i w_i r_i / \sum_i w_i` (biased, consistent).

Standard errors assume i.i.d. observations. They are Wald-type normal
approximations and are known to under-cover when weights are heavy-tailed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.evaluation.weights import importance_weights
from reversallab.exceptions import ConfigurationError, DegenerateWeightsError, InsufficientDataError


@dataclass(frozen=True)
class PolicyValueEstimate:
    estimator: str
    value: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normal_quantile(confidence_level: float) -> float:
    if not 0.0 < confidence_level < 1.0:
        raise ConfigurationError(f"`confidence_level` must lie in (0, 1), got {confidence_level}.")
    return NormalDist().inv_cdf(0.5 + confidence_level / 2.0)


def _require_two_samples(data: LoggedBanditData) -> None:
    if data.n_samples < 2:
        raise InsufficientDataError(
            "At least 2 observations are required to estimate a standard error."
        )


def _make_estimate(
    name: str, value: float, se: float, confidence_level: float, n_samples: int
) -> PolicyValueEstimate:
    z = _normal_quantile(confidence_level)
    return PolicyValueEstimate(
        estimator=name,
        value=value,
        standard_error=se,
        ci_lower=value - z * se,
        ci_upper=value + z * se,
        confidence_level=confidence_level,
        n_samples=n_samples,
    )


def ips(
    data: LoggedBanditData,
    evaluation_action_dist: npt.ArrayLike,
    *,
    confidence_level: float = 0.95,
) -> PolicyValueEstimate:
    """Inverse propensity scoring.

    Standard error: sample standard deviation of ``w_i r_i`` (ddof=1) over ``sqrt(N)``.
    """
    _normal_quantile(confidence_level)
    _require_two_samples(data)
    terms = importance_weights(data, evaluation_action_dist) * data.reward
    value = float(terms.mean())
    se = float(terms.std(ddof=1) / math.sqrt(data.n_samples))
    return _make_estimate("ips", value, se, confidence_level, data.n_samples)


def snips(
    data: LoggedBanditData,
    evaluation_action_dist: npt.ArrayLike,
    *,
    confidence_level: float = 0.95,
) -> PolicyValueEstimate:
    """Self-normalised IPS.

    Raises :class:`DegenerateWeightsError` if every importance weight is zero, i.e.
    the evaluation policy never takes any logged action, in which case the logged
    data contain no information about the evaluation policy's value.

    Standard error: delta method (ratio estimator),
    ``sqrt(sum_i w_i^2 (r_i - V)^2) / sum_i w_i``.
    """
    _normal_quantile(confidence_level)
    _require_two_samples(data)
    w = importance_weights(data, evaluation_action_dist)
    total = float(w.sum())
    if total <= 0.0:
        raise DegenerateWeightsError(
            "SNIPS is undefined: all importance weights are zero (the evaluation policy assigns "
            "zero probability to every logged action). Collect more data or use a more "
            "exploratory logging policy."
        )
    value = float(np.dot(w, data.reward) / total)
    residual = data.reward - value
    se = float(math.sqrt(float(np.sum((w * residual) ** 2))) / total)
    return _make_estimate("snips", value, se, confidence_level, data.n_samples)


class Estimator(Protocol):
    def __call__(
        self,
        data: LoggedBanditData,
        evaluation_action_dist: npt.ArrayLike,
        *,
        confidence_level: float = ...,
    ) -> PolicyValueEstimate: ...


ESTIMATORS: dict[str, Estimator] = {"ips": ips, "snips": snips}
