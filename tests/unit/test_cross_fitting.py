"""Tests for reward models and cross-fitting, including strict leakage checks."""

from __future__ import annotations

import numpy as np
import pytest

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import FloatArray, IntArray
from reversallab.data.synthetic import RewardType, SyntheticBanditEnvironment
from reversallab.evaluation.cross_fitting import cross_fit_reward_model
from reversallab.evaluation.reward_models import (
    ConstantRewardModel,
    MisspecifiedLinearRewardModel,
    StratifiedLinearRewardModel,
)
from reversallab.exceptions import ConfigurationError, InsufficientDataError
from reversallab.policies.linear import UniformPolicy


class LeakageTrackingRewardModel:
    """Mock model that records all training contexts seen during fit()."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self.seen_contexts_: list[FloatArray] = []

    def fit(self, context: FloatArray, action: IntArray, reward: FloatArray) -> None:
        self.seen_contexts_.append(context.copy())

    def predict(self, context: FloatArray) -> FloatArray:
        # Return dummy constant predictions
        n = context.shape[0]
        return np.zeros((n, self.n_actions), dtype=np.float64)


def test_cross_fitting_out_of_fold_leakage_check() -> None:
    """Verify that no training fold ever observes contexts from the held-out validation fold."""
    rng = np.random.default_rng(42)
    n, d, k = 100, 3, 4
    contexts = rng.standard_normal((n, d))
    actions = rng.integers(0, k, size=n)
    rewards = rng.random(n)
    propensities = np.full(n, 1.0 / k)

    data = LoggedBanditData(
        context=contexts,
        action=actions,
        reward=rewards,
        logging_propensity=propensities,
        n_actions=k,
    )

    models_created: list[LeakageTrackingRewardModel] = []

    def model_factory() -> LeakageTrackingRewardModel:
        model = LeakageTrackingRewardModel(n_actions=k)
        models_created.append(model)
        return model

    n_folds = 5
    q_hat = cross_fit_reward_model(data, model_factory, n_folds=n_folds, seed=123)

    assert q_hat.shape == (n, k)
    assert len(models_created) == n_folds

    # Check fold partitions: get deterministic fold assignment
    split_rng = np.random.default_rng(123)
    indices = split_rng.permutation(n)
    fold_splits = np.array_split(indices, n_folds)

    for fold_k, model in enumerate(models_created):
        val_indices = fold_splits[fold_k]
        val_contexts = contexts[val_indices]
        assert len(model.seen_contexts_) == 1
        train_contexts = model.seen_contexts_[0]

        # Ensure NO validation context is present in train_contexts
        for val_row in val_contexts:
            matches = np.all(np.isclose(train_contexts, val_row), axis=1)
            msg = f"Data leakage detected! Context from fold {fold_k} was used in training."
            assert not np.any(matches), msg


def test_cross_fitting_prediction_invariance_to_test_reward() -> None:
    """Changing a held-out observation's reward must not change its out-of-fold prediction."""
    rng = np.random.default_rng(7)
    env = SyntheticBanditEnvironment.random(
        n_actions=4, context_dim=3, rng=rng, reward_type=RewardType.BINARY
    )
    policy = UniformPolicy(n_actions=4)
    data = env.generate_logged_data(n_samples=60, logging_policy=policy, rng=rng)

    def model_factory() -> StratifiedLinearRewardModel:
        return StratifiedLinearRewardModel(n_actions=4, reward_type=RewardType.BINARY, alpha=1.0)

    q_hat_1 = cross_fit_reward_model(data, model_factory, n_folds=4, seed=42)

    # Modify reward of observation 10
    altered_rewards = data.reward.copy()
    altered_rewards[10] = 1.0 - altered_rewards[10]
    data_altered = LoggedBanditData(
        context=data.context,
        action=data.action,
        reward=altered_rewards,
        logging_propensity=data.logging_propensity,
        n_actions=data.n_actions,
    )

    q_hat_2 = cross_fit_reward_model(data_altered, model_factory, n_folds=4, seed=42)

    # The prediction on row 10 itself must NOT change, because row 10 was held-out in its fold!
    msg = "Prediction for row 10 changed when its reward changed! Leakage occurred."
    np.testing.assert_allclose(q_hat_1[10], q_hat_2[10], err_msg=msg)


def test_cross_fitting_deterministic_seeds() -> None:
    rng = np.random.default_rng(1)
    env = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=rng, reward_type=RewardType.GAUSSIAN
    )
    policy = UniformPolicy(n_actions=3)
    data = env.generate_logged_data(n_samples=50, logging_policy=policy, rng=rng)

    def model_factory() -> StratifiedLinearRewardModel:
        return StratifiedLinearRewardModel(n_actions=3, reward_type=RewardType.GAUSSIAN)

    q1 = cross_fit_reward_model(data, model_factory, n_folds=3, seed=999)
    q2 = cross_fit_reward_model(data, model_factory, n_folds=3, seed=999)
    q3 = cross_fit_reward_model(data, model_factory, n_folds=3, seed=1000)

    np.testing.assert_array_equal(q1, q2)
    assert not np.allclose(q1, q3)


def test_cross_fitting_invalid_folds() -> None:
    data = LoggedBanditData(
        context=np.zeros((3, 2)),
        action=np.zeros(3, dtype=np.int64),
        reward=np.zeros(3),
        logging_propensity=np.full(3, 0.5),
        n_actions=2,
    )
    with pytest.raises(ConfigurationError, match="`n_folds` must be >= 2"):
        cross_fit_reward_model(data, lambda: ConstantRewardModel(2), n_folds=1)

    with pytest.raises(InsufficientDataError, match="At least 5 observations are required"):
        cross_fit_reward_model(data, lambda: ConstantRewardModel(2), n_folds=5)


def test_stratified_reward_model_binary() -> None:
    rng = np.random.default_rng(10)
    context = rng.standard_normal((100, 3))
    action = rng.integers(0, 3, size=100)
    reward = rng.integers(0, 2, size=100).astype(np.float64)

    model = StratifiedLinearRewardModel(
        n_actions=3, reward_type=RewardType.BINARY, standardize=True
    )
    model.fit(context, action, reward)
    preds = model.predict(context)

    assert preds.shape == (100, 3)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_stratified_reward_model_unobserved_action_fallback() -> None:
    # Action 2 is never observed in training data
    context = np.array([[1.0], [2.0], [3.0], [4.0]])
    action = np.array([0, 0, 1, 1], dtype=np.int64)
    reward = np.array([1.0, 1.0, 0.0, 0.0])

    model = StratifiedLinearRewardModel(n_actions=3, reward_type=RewardType.BINARY)
    model.fit(context, action, reward)
    preds = model.predict(context)

    assert preds.shape == (4, 3)
    # Action 2 should fall back to empirical global mean reward (0.5)
    np.testing.assert_allclose(preds[:, 2], 0.5)


def test_misspecified_reward_model() -> None:
    context = np.array([[1.0, 2.0], [3.0, 4.0]])
    action = np.array([0, 1], dtype=np.int64)
    reward = np.array([1.0, 0.0])

    model = MisspecifiedLinearRewardModel(n_actions=2, reward_type=RewardType.BINARY)
    model.fit(context, action, reward)
    preds = model.predict(context)

    assert preds.shape == (2, 2)
    assert np.all(np.isfinite(preds))


def test_identity_aware_cross_fitting_prevents_bootstrap_duplicate_leakage() -> None:
    """Verify that multiple bootstrap copies of the same observation never leak across folds."""
    from reversallab.evaluation.bootstrap import resample_logged_data

    rng = np.random.default_rng(101)
    n_orig = 40
    n_actions = 3
    context_dim = 2

    orig_contexts = rng.standard_normal((n_orig, context_dim))
    orig_actions = rng.integers(0, n_actions, size=n_orig)
    orig_rewards = rng.random(n_orig)
    orig_propensities = np.full(n_orig, 1.0 / n_actions)

    orig_data = LoggedBanditData(
        context=orig_contexts,
        action=orig_actions,
        reward=orig_rewards,
        logging_propensity=orig_propensities,
        n_actions=n_actions,
    )

    # Resample with replacement to create duplicate copies
    resampled_data, original_indices = resample_logged_data(orig_data, rng)
    n_samples = resampled_data.n_samples
    _, counts = np.unique(original_indices, return_counts=True)
    assert np.any(counts > 1), "Resample must contain duplicates for this test"

    # 1. Under old split (naive row permutation), verify that leakage occurs
    n_folds = 4
    naive_splits = np.array_split(rng.permutation(n_samples), n_folds)
    naive_leakage = False
    for k in range(n_folds):
        val_rows = naive_splits[k]
        train_rows = np.setdiff1d(np.arange(n_samples), val_rows)
        val_ids = set(original_indices[val_rows])
        train_ids = set(original_indices[train_rows])
        if len(val_ids.intersection(train_ids)) > 0:
            naive_leakage = True
            break
    assert naive_leakage, "Naive row-level splitting should cause duplicate copies to leak"

    # 2. With groups=original_indices, verify ZERO leakage
    models_created: list[LeakageTrackingRewardModel] = []

    def model_factory() -> LeakageTrackingRewardModel:
        model = LeakageTrackingRewardModel(n_actions=n_actions)
        models_created.append(model)
        return model

    q_hat = cross_fit_reward_model(
        resampled_data, model_factory, n_folds=n_folds, seed=42, groups=original_indices
    )
    assert q_hat.shape == (n_samples, n_actions)

    # Verify that for every fold, all copies of an ID are strictly in its validation set
    split_rng = np.random.default_rng(42)
    permuted_unique = split_rng.permutation(np.unique(original_indices))
    group_splits = np.array_split(permuted_unique, n_folds)

    for k in range(n_folds):
        val_groups = set(group_splits[k])
        val_rows = np.where(np.isin(original_indices, group_splits[k]))[0]
        train_rows = np.setdiff1d(np.arange(n_samples), val_rows)
        train_groups = set(original_indices[train_rows])

        assert val_groups.isdisjoint(train_groups), (
            f"Fold {k} leaked original IDs across train and validation!"
        )


def test_identity_aware_cross_fitting_error_handling() -> None:
    """Verify validation of groups parameter shape and unique count."""
    rng = np.random.default_rng(42)
    env = SyntheticBanditEnvironment.random(
        n_actions=2, context_dim=2, rng=rng, reward_type=RewardType.BINARY
    )
    data = env.generate_logged_data(20, UniformPolicy(2), rng)

    def factory() -> StratifiedLinearRewardModel:
        return StratifiedLinearRewardModel(2, RewardType.BINARY)

    # Wrong shape
    with pytest.raises(ConfigurationError, match=r"`groups` must have shape"):
        cross_fit_reward_model(data, factory, n_folds=3, groups=np.array([1, 2]))

    # Fewer unique groups than n_folds
    with pytest.raises(InsufficientDataError, match=r"distinct groups are required"):
        cross_fit_reward_model(data, factory, n_folds=5, groups=np.array([1] * 10 + [2] * 10))
