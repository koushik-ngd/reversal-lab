from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from reversallab.contracts import LoggedBanditData
from reversallab.exceptions import DataValidationError


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "context": np.zeros((3, 2)),
        "action": np.array([0, 1, 2]),
        "reward": np.array([1.0, 0.0, 1.0]),
        "logging_propensity": np.array([0.2, 0.5, 0.3]),
        "n_actions": 3,
    }
    base.update(overrides)
    return base


def test_valid_batch_is_readonly_copy() -> None:
    action = np.array([0, 1, 2])
    data = LoggedBanditData(**_kwargs(action=action))
    assert data.n_samples == 3
    assert data.context_dim == 2
    assert data.action.dtype == np.int64
    action[0] = 2
    assert data.action[0] == 0
    with pytest.raises(ValueError, match="read-only"):
        data.reward[0] = 5.0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"reward": np.array([1.0, 0.0])}, "same length"),
        ({"logging_propensity": np.array([0.2, 0.5])}, "same length"),
        ({"context": np.zeros(3)}, "2-dimensional"),
        (
            {
                "context": np.zeros((0, 2)),
                "action": np.array([], dtype=int),
                "reward": np.array([]),
                "logging_propensity": np.array([]),
            },
            "at least one",
        ),
        ({"action": np.array([0.0, 1.0, 2.0])}, "integer dtype"),
        ({"action": np.array([True, False, True])}, "integer dtype"),
        ({"action": np.array([0, 1, 3])}, r"\[0, 2\]"),
        ({"action": np.array([-1, 1, 2])}, r"\[0, 2\]"),
        ({"reward": np.array([1.0, np.nan, 0.0])}, "non-finite"),
        ({"context": np.array([[0.0, np.inf], [0, 0], [0, 0]])}, "non-finite"),
        ({"logging_propensity": np.array([0.2, 0.0, 0.3])}, "non-positive"),
        ({"logging_propensity": np.array([0.2, -0.1, 0.3])}, "non-positive"),
        ({"logging_propensity": np.array([0.2, 1.5, 0.3])}, "greater than 1"),
        ({"reward": np.array(["a", "b", "c"])}, "numeric"),
        ({"n_actions": 0}, ">= 1"),
        ({"n_actions": True}, "must be an int"),
        ({"group_id": np.array([1, 2])}, "group_id"),
    ],
)
def test_invalid_inputs_rejected(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(DataValidationError, match=match):
        LoggedBanditData(**_kwargs(**overrides))


def test_logging_dist_must_match_observed_propensity() -> None:
    dist = np.array([[0.2, 0.4, 0.4], [0.25, 0.5, 0.25], [0.3, 0.4, 0.3]])
    LoggedBanditData(**_kwargs(logging_action_dist=dist))
    wrong = dist.copy()
    wrong[1] = [0.5, 0.25, 0.25]
    with pytest.raises(DataValidationError, match="disagrees"):
        LoggedBanditData(**_kwargs(logging_action_dist=wrong))


@pytest.mark.parametrize(
    ("dist", "match"),
    [
        (np.array([[0.2, 0.4, 0.5], [0.25, 0.5, 0.25], [0.3, 0.4, 0.3]]), "sum to 1"),
        (np.array([[0.2, 0.9, -0.1], [0.25, 0.5, 0.25], [0.3, 0.4, 0.3]]), "outside"),
        (np.ones((3, 2)) / 2, "shape"),
    ],
)
def test_invalid_logging_dist_rejected(dist: np.ndarray[Any, Any], match: str) -> None:
    with pytest.raises(DataValidationError, match=match):
        LoggedBanditData(**_kwargs(logging_action_dist=dist))


def test_group_id_accepted() -> None:
    data = LoggedBanditData(**_kwargs(group_id=np.array(["u1", "u1", "u2"])))
    assert data.group_id is not None
    assert list(data.group_id) == ["u1", "u1", "u2"]
