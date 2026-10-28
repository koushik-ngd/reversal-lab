"""The Wald standard error can miss the dispersion of a rare-weight IPS estimator.

Experiment 001 measured this on the poor-overlap scenario. This test is a small
deterministic illustration of the same arithmetic. It is not a coverage calibration.
"""

from __future__ import annotations

import numpy as np

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.evaluation.estimators import ips


def _dataset(n: int, *, hit: bool) -> LoggedBanditData:
    action = np.zeros(n, dtype=np.int64)
    reward = np.zeros(n)
    if hit:
        action[0] = 1
        reward[0] = 1.0
    propensity = np.where(action == 1, 0.02, 0.98).astype(np.float64)
    logging = np.column_stack([np.full(n, 0.98), np.full(n, 0.02)])
    return LoggedBanditData(
        context=np.zeros((n, 1)),
        action=action,
        reward=reward,
        logging_propensity=propensity,
        n_actions=2,
        logging_action_dist=logging,
    )


def test_average_wald_se_is_far_below_the_sd_of_rare_spike_ips() -> None:
    n = 50
    evaluation = np.zeros((n, 2))
    evaluation[:, 1] = 1.0
    quiet = ips(_dataset(n, hit=False), evaluation)
    loud = ips(_dataset(n, hit=True), evaluation)
    assert quiet.value == 0.0
    assert quiet.standard_error == 0.0
    assert loud.value == 1.0
    assert loud.standard_error > 0.0

    values = np.array([0.0] * 99 + [loud.value])
    reported = np.array([0.0] * 99 + [loud.standard_error])
    empirical_sd = float(values.std(ddof=1))
    # Ninety-nine replicates never see the rewarded action, so their sample
    # standard deviation is zero. One replicate contains the spike. The average
    # reported SE is then about one hundredth of that spike's SE, while the
    # standard deviation across replicates is on the order of the spike itself.
    assert float(reported.mean()) < 0.25 * empirical_sd
    assert quiet.ci_lower == 0.0
    assert quiet.ci_upper == 0.0
    assert not (quiet.ci_lower <= float(values.mean()) <= quiet.ci_upper)
