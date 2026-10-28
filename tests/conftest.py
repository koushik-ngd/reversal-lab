from __future__ import annotations

import numpy as np
import pytest

from reversallab.contracts import FloatArray, LoggedBanditData

HAND_LOGGING_DIST: FloatArray = np.array(
    [[0.5, 0.5], [0.8, 0.2], [0.25, 0.75], [0.5, 0.5]], dtype=np.float64
)
HAND_EVAL_DIST: FloatArray = np.array(
    [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0]], dtype=np.float64
)
HAND_ACTIONS = np.array([0, 1, 0, 1], dtype=np.int64)
HAND_REWARDS: FloatArray = np.array([1.0, 0.0, 1.0, 1.0])


@pytest.fixture
def hand_data() -> LoggedBanditData:
    """Four observations, two actions; weights are w = [2, 2.5, 2, 2]."""
    return LoggedBanditData(
        context=np.zeros((4, 1)),
        action=HAND_ACTIONS,
        reward=HAND_REWARDS,
        logging_propensity=HAND_LOGGING_DIST[np.arange(4), HAND_ACTIONS],
        n_actions=2,
        logging_action_dist=HAND_LOGGING_DIST,
    )


@pytest.fixture
def hand_eval_dist() -> FloatArray:
    return HAND_EVAL_DIST.copy()
