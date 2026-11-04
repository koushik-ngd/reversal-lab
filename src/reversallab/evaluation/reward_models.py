"""Reward models for contextual bandits: action-stratified linear and logistic models."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from reversallab.contracts.validation import (
    FloatArray,
    IntArray,
    validate_predicted_rewards,
)
from reversallab.data.synthetic import RewardType
from reversallab.exceptions import DataValidationError


class RewardModel(Protocol):
    """Protocol for reward models that predict q(x, a) for all actions."""

    def fit(self, context: FloatArray, action: IntArray, reward: FloatArray) -> None:
        """Fit the reward model on logged context, action, and reward."""
        ...

    def predict(self, context: FloatArray) -> FloatArray:
        """Predict expected rewards for all actions on the given contexts.

        Returns an (N, n_actions) float array.
        """
        ...


class StratifiedLinearRewardModel:
    r"""Action-stratified reward model.

    Trains an independent supervised model for each action :math:`a \in \{0, \dots, K-1\}`.
    For binary rewards, fits an L2-regularized logistic regression.
    For gaussian/continuous rewards, fits a Ridge regression.

    Preprocessors (such as StandardScaler) are fitted strictly on the training contexts.
    If an action has 0 samples in the training set, predictions fall back to the empirical
    mean reward of the training fold.
    """

    def __init__(
        self,
        n_actions: int,
        reward_type: RewardType | str = RewardType.BINARY,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        standardize: bool = False,
    ) -> None:
        if n_actions < 1:
            raise DataValidationError(f"`n_actions` must be >= 1, got {n_actions}.")
        self.n_actions = n_actions
        r_type = reward_type if isinstance(reward_type, str) else reward_type.value
        if r_type in ("binary", RewardType.BINARY):
            self.reward_type = RewardType.BINARY
        elif r_type in ("gaussian", "continuous", RewardType.GAUSSIAN):
            self.reward_type = RewardType.GAUSSIAN
        else:
            raise DataValidationError(f"Unsupported reward_type: {reward_type!r}")
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.standardize = standardize

        self.models_: list[Any] = []
        self.scalers_: list[StandardScaler | None] = []
        self.constant_fallbacks_: list[float | None] = []
        self.global_mean_reward_: float = 0.0
        self.is_fitted_: bool = False

    def fit(self, context: FloatArray, action: IntArray, reward: FloatArray) -> None:
        n_samples = context.shape[0]
        if n_samples == 0:
            raise DataValidationError("Cannot fit a reward model on an empty dataset.")

        self.global_mean_reward_ = float(np.mean(reward))
        self.models_ = []
        self.scalers_ = []
        self.constant_fallbacks_ = []

        for a in range(self.n_actions):
            mask = action == a
            n_action_samples = int(np.count_nonzero(mask))
            if n_action_samples == 0:
                self.models_.append(None)
                self.scalers_.append(None)
                self.constant_fallbacks_.append(self.global_mean_reward_)
                continue

            x_a = context[mask]
            y_a = reward[mask]

            # Fit preprocessor strictly on training data
            scaler: StandardScaler | None = None
            if self.standardize:
                scaler = StandardScaler()
                x_a = scaler.fit_transform(x_a)
            self.scalers_.append(scaler)

            if self.reward_type == RewardType.BINARY:
                unique_y = np.unique(y_a)
                if len(unique_y) < 2:
                    # Only one class present for this action in the training fold
                    self.models_.append(None)
                    self.constant_fallbacks_.append(float(unique_y[0]))
                else:
                    clf = LogisticRegression(
                        C=1.0 / max(self.alpha, 1e-9),
                        fit_intercept=self.fit_intercept,
                        solver="lbfgs",
                        max_iter=1000,
                    )
                    clf.fit(x_a, y_a)
                    self.models_.append(clf)
                    self.constant_fallbacks_.append(None)
            else:
                reg = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
                reg.fit(x_a, y_a)
                self.models_.append(reg)
                self.constant_fallbacks_.append(None)

        self.is_fitted_ = True

    def predict(self, context: FloatArray) -> FloatArray:
        if not self.is_fitted_:
            raise RuntimeError("Reward model must be fitted before predict() is called.")

        n_eval = context.shape[0]
        q_hat = np.empty((n_eval, self.n_actions), dtype=np.float64)

        for a in range(self.n_actions):
            fallback = self.constant_fallbacks_[a]
            if fallback is not None:
                q_hat[:, a] = fallback
                continue

            scaler = self.scalers_[a]
            x_eval = scaler.transform(context) if scaler is not None else context
            model = self.models_[a]

            if self.reward_type == RewardType.BINARY:
                probs = model.predict_proba(x_eval)
                classes = list(model.classes_)
                if 1.0 in classes or 1 in classes:
                    col = classes.index(1.0 if 1.0 in classes else 1)
                    q_hat[:, a] = probs[:, col]
                else:
                    q_hat[:, a] = probs[:, -1]
            else:
                q_hat[:, a] = model.predict(x_eval)

        if self.reward_type == RewardType.BINARY:
            np.clip(q_hat, 0.0, 1.0, out=q_hat)

        return validate_predicted_rewards(
            q_hat, n_samples=n_eval, n_actions=self.n_actions, name="q_hat"
        )


class ConstantRewardModel:
    """Baseline / misspecified model that predicts a constant reward for every action."""

    def __init__(self, n_actions: int, constant_value: float = 0.0) -> None:
        self.n_actions = n_actions
        self.constant_value = float(constant_value)

    def fit(self, context: FloatArray, action: IntArray, reward: FloatArray) -> None:
        pass

    def predict(self, context: FloatArray) -> FloatArray:
        n = context.shape[0]
        return np.full((n, self.n_actions), self.constant_value, dtype=np.float64)


class MisspecifiedLinearRewardModel:
    """Deliberately misspecified linear reward model.

    Inverts the relationship between context and reward to test the double robustness
    property of DR under model misspecification.
    """

    def __init__(
        self,
        n_actions: int,
        reward_type: RewardType | str = RewardType.BINARY,
    ) -> None:
        self.base_model = StratifiedLinearRewardModel(
            n_actions=n_actions, reward_type=reward_type, alpha=1.0
        )
        self.n_actions = n_actions
        r_type = reward_type if isinstance(reward_type, str) else reward_type.value
        self.reward_type = (
            RewardType.BINARY if r_type in ("binary", RewardType.BINARY) else RewardType.GAUSSIAN
        )

    def fit(self, context: FloatArray, action: IntArray, reward: FloatArray) -> None:
        corrupted_context = -context
        self.base_model.fit(corrupted_context, action, reward)

    def predict(self, context: FloatArray) -> FloatArray:
        preds = self.base_model.predict(context)
        if self.reward_type == RewardType.BINARY:
            return 1.0 - preds
        return -preds
