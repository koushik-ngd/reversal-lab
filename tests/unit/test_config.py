from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reversallab.data import SyntheticBanditEnvironment
from reversallab.exceptions import ConfigurationError
from reversallab.experiments.config import (
    EpsilonGreedyPolicySpec,
    SoftmaxPolicySpec,
    build_policy,
    load_config,
)
from reversallab.policies import EpsilonGreedyLinearPolicy, LinearSoftmaxPolicy

REPO_CONFIG = (
    Path(__file__).parents[2] / "configs" / "experiments" / "001_estimator_validation.yaml"
)

MINIMAL = """
name: t
seed: 1
n_replications: 3
estimators: [ips]
environment: {n_actions: 3, context_dim: 2}
scenarios:
  - name: s
    n_samples: [10]
    logging_policy: {kind: uniform}
    evaluation_policy: {kind: uniform}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repository_config_is_valid() -> None:
    config = load_config(REPO_CONFIG)
    assert config.n_replications == 500
    assert {s.name for s in config.scenarios} >= {"on_policy", "poor_overlap"}


def test_minimal_config(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, MINIMAL)).confidence_level == 0.95


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("estimators: [ips]", "estimators: [dr]", "unknown estimator"),
        ("seed: 1", "seed: 1\nunexpected: 2", "Extra inputs"),
        (
            "{kind: uniform}\n    evaluation",
            "{kind: softmax, inverse_temperature: 1, scores: random}\n    evaluation",
            "score_seed",
        ),
        ("n_samples: [10]", "n_samples: [1]", ">= 2"),
        (
            "{kind: uniform}\n",
            "{kind: epsilon_greedy, epsilon: 2, scores: reward}\n",
            "less than or equal",
        ),
    ],
)
def test_invalid_configs(tmp_path: Path, old: str, new: str, match: str) -> None:
    assert old in MINIMAL
    with pytest.raises(ConfigurationError, match=match):
        load_config(_write(tmp_path, MINIMAL.replace(old, new, 1)))


def test_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_config(_write(tmp_path, "- a\n- b\n"))


def test_build_policy_uses_reward_scores() -> None:
    env = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=np.random.default_rng(0)
    )
    policy = build_policy(
        SoftmaxPolicySpec(kind="softmax", inverse_temperature=2.0, scores="reward"), env
    )
    assert isinstance(policy, LinearSoftmaxPolicy)
    np.testing.assert_array_equal(policy.weights, env.reward_weights)


def test_build_policy_random_scores_are_seeded() -> None:
    env = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=np.random.default_rng(0)
    )
    spec = EpsilonGreedyPolicySpec(
        kind="epsilon_greedy", epsilon=0.1, scores="random", score_seed=4
    )
    a, b = build_policy(spec, env), build_policy(spec, env)
    assert isinstance(a, EpsilonGreedyLinearPolicy)
    assert isinstance(b, EpsilonGreedyLinearPolicy)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert not np.allclose(a.weights, env.reward_weights)
