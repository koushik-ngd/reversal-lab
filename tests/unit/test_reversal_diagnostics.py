"""Unit tests for diagnostic gate evaluation.

Tests active/inactive deficits, ESS, weight share, and support violations.
"""

from __future__ import annotations

import numpy as np

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.reversal.diagnostics import (
    DiagnosticGateThresholds,
    evaluate_diagnostic_gate,
)


def _make_logged_data(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    logging_action_dist: np.ndarray | None = None,
    n_actions: int = 2,
) -> LoggedBanditData:
    return LoggedBanditData(
        n_actions=n_actions,
        context=contexts,
        action=actions,
        reward=rewards,
        logging_propensity=propensities,
        logging_action_dist=logging_action_dist,
    )


class TestDiagnosticGateEvaluation:
    """Test multi-criteria diagnostic evaluation against false alarms and missed risks."""

    def test_inactive_propensity_deficit_does_not_trip_gate(self) -> None:
        """When propensity < tau on an action where candidate has 0 probability, gate passes."""
        n = 100
        contexts = np.zeros((n, 2))
        actions = np.zeros(n, dtype=np.int64)
        rewards = np.ones(n)
        propensities = np.full(n, 0.50)

        # Force observation 0 to have tiny propensity, but action 1
        actions[0] = 1
        propensities[0] = 1e-5  # Below default tau=1e-4

        # Candidate policy puts 100% mass on action 0, 0% on action 1
        cand_dist = np.zeros((n, 2))
        cand_dist[:, 0] = 1.0

        base_dist = np.zeros((n, 2))
        base_dist[:, 0] = 1.0

        data = _make_logged_data(contexts, actions, rewards, propensities)
        thresholds = DiagnosticGateThresholds(min_propensity=1e-4)

        report = evaluate_diagnostic_gate(data, cand_dist, base_dist, thresholds)

        # Propensity is below tau, but it is INACTIVE because candidate puts 0 mass on action 1
        assert report.min_propensity == 1e-5
        assert report.is_propensity_active is False
        assert report.passed is True
        assert len(report.warnings) > 0  # Inactive warning recorded

    def test_active_propensity_deficit_trips_gate(self) -> None:
        """When propensity < tau on an action candidate actually takes, gate fails."""
        n = 100
        contexts = np.zeros((n, 2))
        actions = np.zeros(n, dtype=np.int64)
        rewards = np.ones(n)
        propensities = np.full(n, 0.50)

        actions[0] = 1
        propensities[0] = 1e-5  # Below tau=1e-4

        # Candidate policy puts 1.0 mass on action 1 for observation 0
        cand_dist = np.zeros((n, 2))
        cand_dist[:, 0] = 1.0
        cand_dist[0, 0] = 0.0
        cand_dist[0, 1] = 1.0  # Active on the rare action!

        base_dist = np.full((n, 2), 0.50)

        data = _make_logged_data(contexts, actions, rewards, propensities)
        thresholds = DiagnosticGateThresholds(min_propensity=1e-4)

        report = evaluate_diagnostic_gate(data, cand_dist, base_dist, thresholds)

        assert report.min_propensity == 1e-5
        assert report.is_propensity_active is True
        assert report.passed is False
        assert any("Active propensity deficit" in r for r in report.rejection_reasons)

    def test_ess_collapse_trips_gate(self) -> None:
        """When effective sample size falls below 5% of N, gate fails."""
        n = 100
        contexts = np.zeros((n, 2))
        actions = np.zeros(n, dtype=np.int64)
        rewards = np.ones(n)
        propensities = np.full(n, 0.50)

        # Make one observation have enormous weight, collapsing ESS
        propensities[0] = 0.001
        actions[0] = 0

        cand_dist = np.zeros((n, 2))
        cand_dist[:, 0] = 1.0

        base_dist = np.zeros((n, 2))
        base_dist[:, 0] = 1.0

        data = _make_logged_data(contexts, actions, rewards, propensities)
        thresholds = DiagnosticGateThresholds(min_propensity=1e-6, min_ess_fraction=0.05)

        report = evaluate_diagnostic_gate(data, cand_dist, base_dist, thresholds)
        assert report.ess_fraction_candidate < 0.05
        assert report.passed is False
        assert any("ESS fraction collapse" in r for r in report.rejection_reasons)

    def test_max_weight_share_trips_gate(self) -> None:
        """When a single observation accounts for > 20% of total weights, gate fails."""
        n = 50
        contexts = np.zeros((n, 2))
        actions = np.zeros(n, dtype=np.int64)
        rewards = np.ones(n)
        propensities = np.full(n, 0.50)

        # Extreme weight on observation 0
        propensities[0] = 0.005

        cand_dist = np.zeros((n, 2))
        cand_dist[:, 0] = 1.0

        base_dist = np.zeros((n, 2))
        base_dist[:, 0] = 1.0

        data = _make_logged_data(contexts, actions, rewards, propensities)
        thresholds = DiagnosticGateThresholds(
            min_propensity=1e-6, min_ess_fraction=0.01, max_weight_share=0.20
        )

        report = evaluate_diagnostic_gate(data, cand_dist, base_dist, thresholds)
        assert report.max_weight_share_candidate > 0.20
        assert report.passed is False
        assert any("weight share" in r for r in report.rejection_reasons)

    def test_proven_support_violation_trips_gate(self) -> None:
        """When logging policy has zero probability where evaluation policy puts mass."""
        n = 50
        contexts = np.zeros((n, 2))
        actions = np.zeros(n, dtype=np.int64)
        rewards = np.ones(n)
        propensities = np.full(n, 1.0)

        # Logging action dist: action 1 is never possible (probability 0.0)
        logging_action_dist = np.zeros((n, 2))
        logging_action_dist[:, 0] = 1.0
        logging_action_dist[:, 1] = 0.0

        # Candidate puts positive mass on action 1
        cand_dist = np.full((n, 2), 0.50)
        base_dist = np.zeros((n, 2))
        base_dist[:, 0] = 1.0

        data = _make_logged_data(
            contexts, actions, rewards, propensities, logging_action_dist=logging_action_dist
        )
        report = evaluate_diagnostic_gate(data, cand_dist, base_dist)

        assert report.support_status == "proven_support_violation"
        assert report.passed is False
        assert any("Proven support violation" in r for r in report.rejection_reasons)
