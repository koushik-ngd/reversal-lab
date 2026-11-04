"""Leakage-safe K-fold reward-model cross-fitting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import FloatArray, validate_predicted_rewards
from reversallab.evaluation.reward_models import RewardModel
from reversallab.exceptions import ConfigurationError, InsufficientDataError


def cross_fit_reward_model(
    data: LoggedBanditData,
    model_factory: Callable[[], RewardModel],
    *,
    n_folds: int = 5,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> FloatArray:
    """Generate out-of-fold predicted reward matrix q_hat(x, a) using K-fold cross-fitting.

    For each fold k in {0, ..., n_folds - 1}:
    1. A fresh reward model is instantiated from ``model_factory()``.
    2. The model is fitted strictly on the training partition (all observations except fold k).
    3. The model predicts expected rewards for every action on the held-out contexts of fold k.

    This ensures that for every observation i, q_hat(x_i, :) is computed from a model that
    never observed (x_i, a_i, r_i, p_i) during training.

    Args:
        data: validated logged bandit interactions.
        model_factory: zero-argument callable returning a fresh, un-fitted RewardModel.
        n_folds: number of cross-fitting folds (must be >= 2 and <= n_samples).
        seed: random seed for deterministic fold partitioning.
        groups: optional 1D array of shape (N,) indicating observation identity or cluster.
            When provided, all rows sharing the same group identity are assigned to the
            same fold, preventing duplicate copies (e.g. from bootstrap resampling) from
            leaking across training and evaluation partitions while preserving multiplicities.

    Returns:
        (N, K) float64 matrix of out-of-fold expected rewards.
    """
    if n_folds < 2:
        raise ConfigurationError(f"`n_folds` must be >= 2, got {n_folds}.")
    if data.n_samples < n_folds:
        raise InsufficientDataError(
            f"At least {n_folds} observations are required for {n_folds}-fold cross-fitting, "
            f"but dataset only contains {data.n_samples}."
        )

    rng = np.random.default_rng(seed)

    if groups is not None:
        groups_arr = np.asarray(groups)
        if groups_arr.shape != (data.n_samples,):
            raise ConfigurationError(
                f"`groups` must have shape ({data.n_samples},), got {groups_arr.shape}."
            )
        unique_groups = np.unique(groups_arr)
        if len(unique_groups) < n_folds:
            raise InsufficientDataError(
                f"At least {n_folds} distinct groups are required for cross-fitting, "
                f"but found only {len(unique_groups)}."
            )
        permuted_groups = rng.permutation(unique_groups)
        group_splits = np.array_split(permuted_groups, n_folds)
        fold_splits = [
            np.sort(np.where(np.isin(groups_arr, g_split))[0]) for g_split in group_splits
        ]
    else:
        indices = rng.permutation(data.n_samples)
        fold_splits = np.array_split(indices, n_folds)

    q_hat = np.empty((data.n_samples, data.n_actions), dtype=np.float64)

    for k in range(n_folds):
        val_idx = fold_splits[k]
        if len(val_idx) == 0:
            continue

        train_mask = np.ones(data.n_samples, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]

        train_context = data.context[train_idx]
        train_action = data.action[train_idx]
        train_reward = data.reward[train_idx]
        val_context = data.context[val_idx]

        model = model_factory()
        model.fit(train_context, train_action, train_reward)
        val_preds = model.predict(val_context)

        # Validate fold prediction shape and values
        validated_preds = validate_predicted_rewards(
            val_preds, n_samples=len(val_idx), n_actions=data.n_actions, name=f"fold_{k}_q_hat"
        )
        q_hat[val_idx] = validated_preds

    return validate_predicted_rewards(
        q_hat, n_samples=data.n_samples, n_actions=data.n_actions, name="q_hat"
    )
