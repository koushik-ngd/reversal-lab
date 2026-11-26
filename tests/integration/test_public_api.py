"""Integration and contract tests for the ReversalLab public Python SDK and API.

Validates:
1. BanditDataset ingestion, validation, and immutability.
2. PolicyValueEvaluator across all supported estimators (IPS, SNIPS, DM, DR).
3. PolicyComparison workflow: paired difference, diagnostic gating, two-stratum reversal,
   forward robust baseline, simultaneous bands, JSON serialization, and summary rendering.
4. Diagnostic gate negative test: propensity deficit / support violations tripping the gate
   and preventing spurious RECOMMEND decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import reversallab as rl
from reversallab.api import (
    SDK_SCHEMA_VERSION,
)
from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.exceptions import DataValidationError
from reversallab.policies.linear import LinearSoftmaxPolicy, UniformPolicy


@pytest.fixture
def sample_dataset() -> rl.BanditDataset:
    """Generate a clean synthetic bandit dataset."""
    rng = np.random.default_rng(42)
    env = SyntheticBanditEnvironment.random(
        n_actions=3,
        context_dim=2,
        rng=rng,
        reward_scale=1.5,
        bias_scale=0.0,
    )
    logging_policy = UniformPolicy(n_actions=env.n_actions)
    return rl.BanditDataset.from_synthetic(
        env,
        logging_policy=logging_policy,
        n_samples=200,
        seed=42,
    )


class TestBanditDataset:
    """Contract tests for BanditDataset."""

    def test_from_arrays_valid(self, sample_dataset: rl.BanditDataset) -> None:
        assert sample_dataset.n_samples == 200
        assert sample_dataset.n_actions == 3
        assert sample_dataset.context_dim == 2
        assert sample_dataset.context.shape == (200, 2)
        assert sample_dataset.action.shape == (200,)
        assert sample_dataset.reward.shape == (200,)
        assert sample_dataset.propensity.shape == (200,)

    def test_from_logged_data(self, sample_dataset: rl.BanditDataset) -> None:
        dataset = rl.BanditDataset.from_logged_data(sample_dataset.data)
        assert dataset.n_samples == 200
        assert dataset.data is sample_dataset.data

    def test_validation_error_on_invalid_actions(self) -> None:
        contexts = np.zeros((10, 2))
        actions = np.array([0, 1, 3, 0, 1, 0, 1, 0, 1, 0])  # action 3 >= n_actions=3
        rewards = np.zeros(10)
        propensities = np.full(10, 0.33)

        with pytest.raises(DataValidationError):
            rl.BanditDataset.from_arrays(
                contexts=contexts,
                actions=actions,
                rewards=rewards,
                propensities=propensities,
                n_actions=3,
            )

    def test_validation_error_on_mismatched_lengths(self) -> None:
        contexts = np.zeros((10, 2))
        actions = np.zeros(9, dtype=int)
        rewards = np.zeros(10)
        propensities = np.full(10, 0.33)

        with pytest.raises(DataValidationError):
            rl.BanditDataset.from_arrays(
                contexts=contexts,
                actions=actions,
                rewards=rewards,
                propensities=propensities,
                n_actions=3,
            )

    def test_validation_error_on_invalid_propensities(self) -> None:
        contexts = np.zeros((5, 2))
        actions = np.zeros(5, dtype=int)
        rewards = np.zeros(5)
        propensities = np.array([0.5, 0.5, -0.1, 0.5, 0.5])  # negative propensity

        with pytest.raises(DataValidationError):
            rl.BanditDataset.from_arrays(
                contexts=contexts,
                actions=actions,
                rewards=rewards,
                propensities=propensities,
                n_actions=3,
            )


class TestPolicyValueEvaluator:
    """Contract tests for single policy value evaluation."""

    @pytest.mark.parametrize("estimator", ["ips", "snips", "dm", "dr"])
    def test_evaluate_supported_estimators(
        self, sample_dataset: rl.BanditDataset, estimator: str
    ) -> None:
        policy = UniformPolicy(n_actions=3)
        report = rl.PolicyValueEvaluator.evaluate(
            dataset=sample_dataset,
            policy=policy,
            estimator=estimator,
            cross_fitting_folds=2,
            seed=42,
        )

        assert report.schema_version == SDK_SCHEMA_VERSION
        assert report.estimator == estimator
        assert np.isfinite(report.point_estimate)
        assert report.standard_error >= 0.0
        assert report.ci_lower <= report.ci_upper
        assert report.effective_sample_size > 0.0

        d = report.to_dict()
        assert d["schema_version"] == SDK_SCHEMA_VERSION
        assert d["estimator"] == estimator
        assert "point_estimate" in d

    def test_reject_unsupported_estimator(self, sample_dataset: rl.BanditDataset) -> None:
        policy = UniformPolicy(n_actions=3)
        with pytest.raises(ValueError, match="Unknown estimator"):
            rl.PolicyValueEvaluator.evaluate(
                dataset=sample_dataset,
                policy=policy,
                estimator="unknown_estimator",
            )


class TestPolicyComparison:
    """Contract tests for paired policy comparison and decision reversal."""

    def test_end_to_end_workflow(self, sample_dataset: rl.BanditDataset, tmp_path: Path) -> None:
        base_policy = UniformPolicy(n_actions=3)
        cand_policy = LinearSoftmaxPolicy(
            weights=np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
            inverse_temperature=1.0,
        )

        report = rl.PolicyComparison.evaluate(
            dataset=sample_dataset,
            candidate_policy=cand_policy,
            baseline_policy=base_policy,
            estimator="dr",
            margin=0.01,
            deploy_epsilon=0.10,
            max_epsilon=0.30,
            cross_fitting_folds=2,
            seed=42,
        )

        assert report.schema_version == SDK_SCHEMA_VERSION
        assert "timestamp" in report.to_dict()
        assert report.diagnostics["gate_passed"] is True
        assert report.decision["outcome"] in ("RECOMMEND", "HOLD", "INCONCLUSIVE")
        assert report.reversal_search["status"] in (
            "REVERSAL_FOUND",
            "NO_REVERSAL_FOUND_WITHIN_BOUNDS",
            "REVERSAL_AT_BOUNDARY",
        )
        assert report.robust_mixture["outcome"] in (
            "RECOMMEND",
            "HOLD",
            "INCONCLUSIVE",
        )
        assert "Asymptotically valid" in report.assumptions_and_guarantees["supported_claims"]

        # Test summary string
        summary_text = report.summary()
        assert "REVERSALLAB DECISION-REVERSAL AUDIT REPORT" in summary_text
        assert "1. PAIRED POLICY ADVANTAGE" in summary_text
        assert report.decision["baseline_q0"] == 0.5
        assert "Decision at q0=0.500" in summary_text

        # Test JSON serialization and save
        out_file = tmp_path / "report.json"
        report.save(out_file)
        assert out_file.exists()

        loaded_json = json.loads(out_file.read_text(encoding="utf-8"))
        assert loaded_json["schema_version"] == SDK_SCHEMA_VERSION
        assert loaded_json["decision"]["outcome"] == report.decision["outcome"]

    def test_diagnostic_gate_rejection_blocks_recommend(self) -> None:
        """When logging propensities have severe deficit, the gate must trip."""
        rng = np.random.default_rng(123)
        n = 100
        contexts = rng.normal(size=(n, 2))
        actions = np.zeros(n, dtype=int)
        rewards = rng.normal(size=n)
        # Create severe propensity deficit below default 1e-4 threshold
        propensities = np.full(n, 1e-6)

        dataset = rl.BanditDataset.from_arrays(
            contexts=contexts,
            actions=actions,
            rewards=rewards,
            propensities=propensities,
            n_actions=3,
        )

        base_policy = UniformPolicy(n_actions=3)
        cand_policy = UniformPolicy(n_actions=3)

        report = rl.PolicyComparison.evaluate(
            dataset=dataset,
            candidate_policy=cand_policy,
            baseline_policy=base_policy,
            min_propensity_threshold=1e-4,
        )

        # Gate must have failed
        assert report.diagnostics["gate_passed"] is False
        assert len(report.diagnostics["rejection_reasons"]) > 0

        # Decision MUST be INCONCLUSIVE due to gate failure, even if point estimate was positive
        assert report.decision["outcome"] == "INCONCLUSIVE"
        assert "Unavailable (diagnostic gate failed)" in report.summary()
        assert "Diagnostic gate failure" in report.decision["reason"]
        assert report.reversal_search["status"] == "BASELINE_DIAGNOSTIC_FAILURE"
        assert report.robust_mixture["outcome"] == "INCONCLUSIVE"
