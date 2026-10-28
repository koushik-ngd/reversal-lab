"""Adapter checks that do not need the Open Bandit Pipeline checkout."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from reversallab.data.open_bandit import (
    constant_action_probabilities,
    load_open_bandit,
    uniform_evaluation_probabilities,
)
from reversallab.evaluation.estimators import ips
from reversallab.exceptions import (
    ConfigurationError,
    DataValidationError,
    NumericalError,
    SupportViolationError,
)
from reversallab.experiments.metadata import collect_provenance

_BEHAVIOR = (
    "timestamp",
    "item_id",
    "position",
    "click",
    "propensity_score",
    "user_feature_0",
    "user_feature_1",
    "user_feature_2",
    "user_feature_3",
    "user-item_affinity_0",
    "user-item_affinity_1",
)
_ITEM = ("item_id", "item_feature_0", "item_feature_1", "item_feature_2", "item_feature_3")


def _write(path: Path, header: tuple[str, ...], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _sample(
    tmp_path: Path,
    behavior_rows: list[list[object]] | None = None,
    *,
    behavior_header: tuple[str, ...] = _BEHAVIOR,
    item_rows: list[list[object]] | None = None,
) -> Path:
    root = tmp_path / "obd"
    if behavior_rows is None:
        behavior_rows = [
            ["10", 0, 1, 0, 0.5, "u", "v", "w", "x", 0.1, 0.2],
            ["11", 1, 2, 1, 0.5, "u", "v", "w", "x", 0.3, 0.4],
        ]
    if item_rows is None:
        item_rows = [
            [0, 1.5, "a", "b", "c"],
            [1, -0.25, "d", "e", "f"],
        ]
    _write(root / "random" / "all" / "all.csv", behavior_header, behavior_rows)
    _write(root / "random" / "all" / "item_context.csv", _ITEM, item_rows)
    return root


def test_valid_sample_keeps_file_order_and_uniform_logging(tmp_path: Path) -> None:
    root = _sample(tmp_path)
    loaded = load_open_bandit(root, behavior_policy="random", campaign="all")
    assert loaded.n_samples == 2
    assert loaded.n_actions == 2
    assert loaded.timestamp == ("10", "11")
    assert loaded.position.tolist() == [1, 2]
    assert loaded.logged.reward.tolist() == [0.0, 1.0]
    assert loaded.provenance.logging_distribution == "uniform_identified"
    assert loaded.logged.logging_action_dist is not None
    assert loaded.user_features[0] == ("u", "v", "w", "x")
    assert loaded.logged.group_id is not None
    assert loaded.logged.group_id[0] == "u|v|w|x"
    assert loaded.item_feature_0.tolist() == [1.5, -0.25]
    assert loaded.item_feature_labels[1] == ("d", "e", "f")
    np.testing.assert_allclose(loaded.logged.context, [[0.1, 0.2], [0.3, 0.4]])
    again = load_open_bandit(root, behavior_policy="random", campaign="all")
    assert again.provenance.behavior_sha256 == loaded.provenance.behavior_sha256
    np.testing.assert_array_equal(again.logged.action, loaded.logged.action)
    np.testing.assert_array_equal(again.logged.reward, loaded.logged.reward)
    np.testing.assert_allclose(again.logged.logging_propensity, loaded.logged.logging_propensity)


def test_uniform_policy_on_uniform_logger_equals_click_rate(tmp_path: Path) -> None:
    root = _sample(tmp_path)
    loaded = load_open_bandit(root, behavior_policy="random", campaign="all")
    evaluation = uniform_evaluation_probabilities(loaded)
    estimate = ips(loaded.logged, evaluation)
    assert estimate.value == pytest.approx(float(loaded.logged.reward.mean()))


def test_action_prob_is_not_aliased(tmp_path: Path) -> None:
    header = tuple("action_prob" if name == "propensity_score" else name for name in _BEHAVIOR)
    root = _sample(tmp_path, behavior_header=header)
    with pytest.raises(DataValidationError, match="action_prob"):
        load_open_bandit(root, behavior_policy="random", campaign="all")


def test_gapped_item_ids_are_not_remapped(tmp_path: Path) -> None:
    rows = [
        ["10", 0, 1, 0, 0.5, "u", "v", "w", "x", 0.0, 0.0, 0.0],
        ["11", 2, 2, 1, 0.5, "u", "v", "w", "x", 0.0, 0.0, 0.0],
    ]
    header = (*_BEHAVIOR, "user-item_affinity_2")
    root = _sample(tmp_path, rows, behavior_header=header)
    with pytest.raises(DataValidationError, match="not remapped"):
        load_open_bandit(root, behavior_policy="random", campaign="all")


def test_position_outside_1_2_3_is_rejected(tmp_path: Path) -> None:
    rows = [
        ["10", 0, 0, 0, 0.5, "u", "v", "w", "x", 0.1, 0.2],
        ["11", 1, 2, 1, 0.5, "u", "v", "w", "x", 0.3, 0.4],
    ]
    root = _sample(tmp_path, rows)
    with pytest.raises(DataValidationError, match="position"):
        load_open_bandit(root, behavior_policy="random", campaign="all")


def test_noncanonical_integer_is_rejected(tmp_path: Path) -> None:
    rows = [
        ["10", "01", 1, 0, 0.5, "u", "v", "w", "x", 0.1, 0.2],
        ["11", 1, 2, 1, 0.5, "u", "v", "w", "x", 0.3, 0.4],
    ]
    root = _sample(tmp_path, rows)
    with pytest.raises(DataValidationError, match="canonical integer"):
        load_open_bandit(root, behavior_policy="random", campaign="all")


def test_invalid_click_and_propensity(tmp_path: Path) -> None:
    click_rows = [
        ["10", 0, 1, 2, 0.5, "u", "v", "w", "x", 0.1, 0.2],
        ["11", 1, 2, 1, 0.5, "u", "v", "w", "x", 0.3, 0.4],
    ]
    with pytest.raises(DataValidationError, match="click"):
        load_open_bandit(
            _sample(tmp_path / "click", click_rows), behavior_policy="random", campaign="all"
        )
    zero_rows = [
        ["10", 0, 1, 0, 0.0, "u", "v", "w", "x", 0.1, 0.2],
        ["11", 1, 2, 1, 0.5, "u", "v", "w", "x", 0.3, 0.4],
    ]
    with pytest.raises(DataValidationError, match="non-positive"):
        load_open_bandit(
            _sample(tmp_path / "zero", zero_rows), behavior_policy="random", campaign="all"
        )
    large_rows = [
        ["10", 0, 1, 0, 1.1, "u", "v", "w", "x", 0.1, 0.2],
        ["11", 1, 2, 1, 0.5, "u", "v", "w", "x", 0.3, 0.4],
    ]
    with pytest.raises(DataValidationError, match="greater than 1"):
        load_open_bandit(
            _sample(tmp_path / "large", large_rows), behavior_policy="random", campaign="all"
        )


def test_near_zero_propensity_is_kept_and_can_overflow(tmp_path: Path) -> None:
    rows = [
        ["10", 0, 1, 1, "1e-310", "u", "v", "w", "x", 0.1, 0.2],
        ["11", 1, 2, 0, 0.5, "u", "v", "w", "x", 0.3, 0.4],
    ]
    loaded = load_open_bandit(_sample(tmp_path, rows), behavior_policy="random", campaign="all")
    assert loaded.logged.logging_propensity[0] > 0.0
    assert loaded.provenance.logging_distribution == "observed_action_only"
    with pytest.raises(NumericalError, match="overflowed"):
        ips(loaded.logged, constant_action_probabilities(loaded, 0))


def test_unknown_policy_and_missing_file(tmp_path: Path) -> None:
    root = _sample(tmp_path)
    with pytest.raises(ConfigurationError, match="behavior_policy"):
        load_open_bandit(root, behavior_policy="epsilon", campaign="all")
    with pytest.raises(ConfigurationError, match="campaign"):
        load_open_bandit(root, behavior_policy="random", campaign="kids")
    with pytest.raises(ConfigurationError, match="does not download"):
        load_open_bandit(tmp_path / "missing", behavior_policy="random", campaign="all")


def test_position_slice_keeps_source_provenance(tmp_path: Path) -> None:
    rows = [
        ["10", 0, 1, 0, 0.5, "u", "v", "w", "x", 0.1, 0.2],
        ["11", 1, 3, 1, 0.5, "p", "q", "r", "s", 0.3, 0.4],
        ["12", 0, 1, 0, 0.5, "u", "v", "w", "x", 0.0, 1.0],
    ]
    # Action 1 never appears twice, but both actions appear. Propensity is 0.5 so
    # the full logging distribution is the uniform distribution.
    root = _sample(tmp_path, rows)
    loaded = load_open_bandit(root, behavior_policy="random", campaign="all")
    sliced = loaded.select_position(1)
    assert sliced.n_samples == 2
    assert set(sliced.position.tolist()) == {1}
    assert sliced.provenance.n_rows == 3
    with pytest.raises(ConfigurationError, match="ZOZOTOWN position"):
        loaded.select_position(0)


def test_bts_style_propensities_do_not_invent_support(tmp_path: Path) -> None:
    rows = [
        ["10", 0, 1, 0, 0.2, "u", "v", "w", "x", 0.1, 0.2],
        ["11", 1, 2, 1, 0.7, "u", "v", "w", "x", 0.3, 0.4],
    ]
    loaded = load_open_bandit(_sample(tmp_path, rows), behavior_policy="random", campaign="all")
    assert loaded.logged.logging_action_dist is None
    assert loaded.provenance.logging_distribution == "observed_action_only"
    # A deterministic policy is a function of the row, not of the logged action.
    # Without the full logging distribution this call must not raise: the missing
    # probabilities are not evidence that those actions had probability zero.
    estimate = ips(loaded.logged, constant_action_probabilities(loaded, 0))
    assert estimate.value == pytest.approx(0.0)


def test_full_logging_distribution_can_reject_an_unsupported_action() -> None:
    """Support is checked only when the caller supplies the whole logging distribution."""
    from reversallab.contracts.logged_data import LoggedBanditData

    blocked = LoggedBanditData(
        context=np.zeros((2, 1)),
        action=np.array([1, 1]),
        reward=np.zeros(2),
        logging_propensity=np.ones(2),
        n_actions=2,
        logging_action_dist=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    evaluation = np.array([[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(SupportViolationError, match="not identified"):
        ips(blocked, evaluation)


def test_provenance_records_the_lockfile_hash() -> None:
    info = collect_provenance()
    digest = info["uv_lock_sha256"]
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert info["git"]["commit"]
    assert isinstance(info["git"]["dirty"], bool)
