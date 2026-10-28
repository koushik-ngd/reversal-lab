"""Checks against the small Open Bandit sample and the OBP source checkout.

Skipped when ``data/raw/zr-obp`` is absent. That checkout is not a package
dependency and is not committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reversallab.data.open_bandit import load_open_bandit
from reversallab.evaluation.estimators import ips, snips
from reversallab.evaluation.obp_reference import obp_point_estimates
from reversallab.experiments.config import load_parity_config
from reversallab.experiments.open_bandit_parity import run_open_bandit_parity

ROOT = Path(__file__).resolve().parents[2]
OBP_ROOT = ROOT / "data" / "raw" / "zr-obp"
OBD_ROOT = OBP_ROOT / "obd"

BEHAVIOR_SHA256 = {
    ("random", "all"): "c0c58e3fa085803dd8dabfcfea77cf8d5fc0324e6ba6307528838d59324b7605",
    ("random", "men"): "379fd2efc808b49c6082b81af6e322b69b2a267963e17a18dd27a12ed6e01571",
    ("random", "women"): "12d8a65cb0a90823ea6b3fdb428501f2bd01534dbcea9e0876052d7b0d2ba2c5",
    ("bts", "all"): "f27f426907a567ef658616263949536ff14e99a4c623ea9f343108b6699a06ac",
    ("bts", "men"): "6459dfce78b37cb845b73099f07d2aa4ddb645a47e2332ca6998c0172e6879b8",
    ("bts", "women"): "64697a19cc6519859019204dd3098a987b89f2b50d347f6a83311f8ecc7c036c",
}
N_ACTIONS = {"all": 80, "men": 34, "women": 46}
CLICK_RATE = {
    ("random", "all"): 0.0038,
    ("random", "men"): 0.0046,
    ("random", "women"): 0.0046,
    ("bts", "all"): 0.0042,
    ("bts", "men"): 0.0069,
    ("bts", "women"): 0.0046,
}


def _require_sample() -> None:
    sample = OBD_ROOT / "random" / "all" / "all.csv"
    if not sample.is_file():
        pytest.skip(f"small Open Bandit sample is not at {sample}")


def _require_obp() -> None:
    source = OBP_ROOT / "obp" / "ope" / "estimators.py"
    if not source.is_file():
        pytest.skip(f"Open Bandit Pipeline source is not at {source}")


@pytest.mark.parametrize(("policy", "campaign"), list(BEHAVIOR_SHA256))
def test_small_sample_schema_and_provenance(policy: str, campaign: str) -> None:
    _require_sample()
    loaded = load_open_bandit(OBD_ROOT, behavior_policy=policy, campaign=campaign)
    again = load_open_bandit(OBD_ROOT, behavior_policy=policy, campaign=campaign)
    assert loaded.n_samples == 10_000
    assert loaded.n_actions == N_ACTIONS[campaign]
    assert loaded.logged.context.shape == (10_000, N_ACTIONS[campaign])
    assert loaded.provenance.behavior_sha256 == BEHAVIOR_SHA256[(policy, campaign)]
    assert loaded.provenance.behavior_sha256 == again.provenance.behavior_sha256
    assert loaded.provenance.n_user_feature_tuples == again.provenance.n_user_feature_tuples
    np.testing.assert_array_equal(loaded.logged.action, again.logged.action)
    np.testing.assert_allclose(loaded.logged.logging_propensity, again.logged.logging_propensity)
    assert set(np.unique(loaded.position).tolist()) == {1, 2, 3}
    assert set(np.unique(loaded.logged.action).tolist()) == set(range(loaded.n_actions))
    assert set(np.unique(loaded.logged.reward).tolist()) <= {0.0, 1.0}
    assert float(loaded.logged.reward.mean()) == pytest.approx(CLICK_RATE[(policy, campaign)])
    assert np.all(loaded.logged.logging_propensity > 0.0)
    assert np.all(loaded.logged.logging_propensity <= 1.0)
    if policy == "random":
        assert loaded.provenance.logging_distribution == "uniform_identified"
        assert loaded.logged.logging_action_dist is not None
        np.testing.assert_allclose(loaded.logged.logging_propensity, 1.0 / loaded.n_actions)
    else:
        assert loaded.provenance.logging_distribution == "observed_action_only"
        assert loaded.logged.logging_action_dist is None
        assert not np.allclose(loaded.logged.logging_propensity, 1.0 / loaded.n_actions)


def test_loader_preserves_file_order(tmp_path: Path) -> None:
    _require_sample()
    del tmp_path
    path = OBD_ROOT / "random" / "all" / "all.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        header = handle.readline()
        first = handle.readline().split(",")
    assert header.split(",")[0] == ""
    loaded = load_open_bandit(OBD_ROOT, behavior_policy="random", campaign="all")
    # The behavior file starts with an unnamed index column, then timestamp.
    assert loaded.timestamp[0] == first[1]
    assert int(loaded.logged.action[0]) == int(first[2])
    assert int(loaded.position[0]) == int(first[3])


def test_hand_arrays_match_obp_ipw_and_snipw() -> None:
    _require_obp()
    from reversallab.contracts.logged_data import LoggedBanditData

    action = np.array([0, 1, 0, 1], dtype=np.int64)
    reward = np.array([1.0, 0.0, 1.0, 1.0])
    logging = np.array([[0.5, 0.5], [0.8, 0.2], [0.25, 0.75], [0.5, 0.5]])
    evaluation = np.array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0]])
    data = LoggedBanditData(
        context=np.zeros((4, 1)),
        action=action,
        reward=reward,
        logging_propensity=logging[np.arange(4), action],
        n_actions=2,
        logging_action_dist=logging,
    )
    ours_ips = ips(data, evaluation)
    ours_snips = snips(data, evaluation)
    assert ours_ips.value == pytest.approx(1.5)
    assert ours_snips.value == pytest.approx(6.0 / 8.5)
    reference = obp_point_estimates(
        OBP_ROOT,
        reward=data.reward,
        action=data.action,
        pscore=data.logging_propensity,
        action_dist=evaluation,
    )
    assert reference["lambda"] == float("inf")
    assert reference["ipw"] == pytest.approx(ours_ips.value, rel=1e-12, abs=1e-12)
    assert reference["snipw"] == pytest.approx(ours_snips.value, rel=1e-12, abs=1e-12)


def test_small_sample_matches_obp_within_documented_tolerance() -> None:
    _require_sample()
    _require_obp()
    config = load_parity_config(ROOT / "configs" / "experiments" / "002_open_bandit_parity.yaml")
    config = config.model_copy(update={"obd_root": str(OBD_ROOT), "obp_root": str(OBP_ROOT)})
    output = run_open_bandit_parity(config)
    assert output.matched, output.warnings
    assert output.results["n_mismatches"] == 0
    assert output.results["max_abs_diff"] <= 1e-12
    assert output.results["obp_lambda"] == "inf"
    uniform_all = next(
        row
        for row in output.results["comparisons"]
        if row["behavior_policy"] == "random"
        and row["campaign"] == "all"
        and row["position"] == "all"
        and row["evaluation_policy"] == "uniform"
    )
    loaded = load_open_bandit(OBD_ROOT, behavior_policy="random", campaign="all")
    click_rate = float(loaded.logged.reward.mean())
    assert uniform_all["reversallab"]["ips"] == pytest.approx(click_rate)
    assert uniform_all["reversallab"]["snips"] == pytest.approx(click_rate)
    position_one = loaded.select_position(1)
    assert position_one.n_samples < loaded.n_samples
    sliced = next(
        row
        for row in output.results["comparisons"]
        if row["behavior_policy"] == "random"
        and row["campaign"] == "all"
        and row["position"] == "1"
        and row["evaluation_policy"] == "uniform"
    )
    assert sliced["reversallab"]["ips"] == pytest.approx(float(position_one.logged.reward.mean()))
    assert sliced["n"] == position_one.n_samples
