r"""ReversalLab Public Python SDK.

A statistically grounded, user-friendly API for off-policy evaluation,
paired policy advantage estimation, and decision-reversal robustness analysis.

Typical usage::

    import reversallab as rl

    # 1. Create or load validated bandit dataset
    dataset = rl.BanditDataset.from_arrays(
        contexts=X,
        actions=a,
        rewards=r,
        propensities=p,
    )

    # 2. Define baseline and candidate policies
    baseline = rl.UniformPolicy(n_actions=3)
    candidate = rl.LinearSoftmaxPolicy(weights=W, bias=b, inverse_temperature=1.5)

    # 3. Evaluate paired deployment advantage and decision reversal
    report = rl.PolicyComparison.evaluate(
        dataset,
        candidate_policy=candidate,
        baseline_policy=baseline,
        margin=0.02,
        deploy_epsilon=0.10,
    )

    # 4. Inspect actionable conclusions
    print(report.summary())
    report.to_json("deployment_report.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import FloatArray, IntArray
from reversallab.data.synthetic import RewardType, SyntheticBanditEnvironment
from reversallab.evaluation.cross_fitting import cross_fit_reward_model
from reversallab.evaluation.diagnostics import diagnose
from reversallab.evaluation.estimators import (
    ALL_ESTIMATORS,
    ESTIMATORS,
    MODEL_BASED_ESTIMATORS,
    PolicyValueEstimate,
)
from reversallab.evaluation.reward_models import (
    RewardModel,
    StratifiedLinearRewardModel,
)
from reversallab.policies.base import Policy
from reversallab.policies.linear import (
    EpsilonGreedyLinearPolicy,
    LinearSoftmaxPolicy,
    UniformPolicy,
)
from reversallab.reversal.analytical import find_analytical_decision_reversal
from reversallab.reversal.decision import DecisionOutcome
from reversallab.reversal.diagnostics import (
    DiagnosticGateReport,
    DiagnosticGateThresholds,
    evaluate_diagnostic_gate,
)
from reversallab.reversal.paired import PairedDifferenceResult, evaluate_paired_difference
from reversallab.reversal.robust_mixture import evaluate_robust_mixture_baseline
from reversallab.reversal.search import find_decision_reversal
from reversallab.reversal.simultaneous import TwoStratumSimultaneousBand
from reversallab.reversal.strata import (
    StratumPartition,
    TwoStratumMixtureModel,
    create_two_stratum_model,
)
from reversallab.reversal.taxonomy import classify_reversal_taxonomy

SDK_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class BanditDataset:
    """Validated, immutable container for off-policy logged contextual bandit data."""

    data: LoggedBanditData

    @classmethod
    def from_logged_data(cls, data: LoggedBanditData) -> BanditDataset:
        """Create a dataset directly from validated LoggedBanditData."""
        return cls(data=data)

    @classmethod
    def from_arrays(
        cls,
        contexts: FloatArray,
        actions: IntArray,
        rewards: FloatArray,
        propensities: FloatArray,
        *,
        n_actions: int | None = None,
        logging_action_dist: FloatArray | None = None,
    ) -> BanditDataset:
        """Create a dataset from numpy arrays, validating all dimensional and positivity contracts.

        Parameters
        ----------
        contexts : ndarray of shape (n_samples, context_dim)
            Context feature vectors.
        actions : ndarray of shape (n_samples,)
            Discrete action taken per observation, 0 <= a < n_actions.
        rewards : ndarray of shape (n_samples,)
            Observed scalar rewards.
        propensities : ndarray of shape (n_samples,)
            Logging policy probability P(a | x) strictly > 0.
        n_actions : int, optional
            Number of discrete actions. Defaults to max(actions) + 1.
        logging_action_dist : ndarray of shape (n_samples, n_actions), optional
            Full action distribution under logging policy, if available.
        """
        acts = np.asarray(actions, dtype=np.int64)
        n_act = int(np.max(acts) + 1) if n_actions is None else int(n_actions)
        logged = LoggedBanditData(
            context=np.asarray(contexts, dtype=np.float64),
            action=acts,
            reward=np.asarray(rewards, dtype=np.float64),
            logging_propensity=np.asarray(propensities, dtype=np.float64),
            n_actions=n_act,
            logging_action_dist=(
                np.asarray(logging_action_dist, dtype=np.float64)
                if logging_action_dist is not None
                else None
            ),
        )
        return cls(data=logged)

    @classmethod
    def from_synthetic(
        cls,
        env: SyntheticBanditEnvironment,
        logging_policy: Policy,
        n_samples: int = 2000,
        *,
        seed: int = 42,
    ) -> BanditDataset:
        """Generate synthetic logged bandit data from an environment and logging policy."""
        rng = np.random.default_rng(seed)
        logged = env.generate_logged_data(
            n_samples=n_samples,
            logging_policy=logging_policy,
            rng=rng,
            include_logging_action_dist=True,
        )
        return cls(data=logged)

    @property
    def n_samples(self) -> int:
        return self.data.n_samples

    @property
    def n_actions(self) -> int:
        return self.data.n_actions

    @property
    def context_dim(self) -> int:
        return self.data.context_dim

    @property
    def context(self) -> FloatArray:
        return self.data.context

    @property
    def action(self) -> IntArray:
        return self.data.action

    @property
    def reward(self) -> FloatArray:
        return self.data.reward

    @property
    def logging_propensity(self) -> FloatArray:
        return self.data.logging_propensity

    @property
    def propensity(self) -> FloatArray:
        """Convenience alias for logging_propensity."""
        return self.data.logging_propensity


@dataclass(frozen=True)
class PolicyValueReport:
    """Report summarizing the off-policy value estimate for a single policy."""

    policy_name: str
    estimator: str
    point_estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    min_propensity: float
    effective_sample_size: float
    ess_fraction: float
    max_weight_share: float
    schema_version: str = SDK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_name": self.policy_name,
            "estimator": self.estimator,
            "point_estimate": self.point_estimate,
            "standard_error": self.standard_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "min_propensity": self.min_propensity,
            "effective_sample_size": self.effective_sample_size,
            "ess_fraction": self.ess_fraction,
            "max_weight_share": self.max_weight_share,
        }

    def summary(self) -> str:
        pct = round(self.confidence_level * 100)
        lines = [
            f"Policy Value Estimation ({self.policy_name})",
            f"  Estimator:        {self.estimator.upper()}",
            f"  Value Estimate:   {self.point_estimate:.4f} (SE: {self.standard_error:.4f})",
            f"  {pct}% Wald CI:      [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"  Min Propensity:   {self.min_propensity:.2e}",
            f"  ESS Fraction:     {self.ess_fraction * 100:.1f}%",
            f"  Max Weight Share: {self.max_weight_share * 100:.1f}%",
        ]
        return "\n".join(lines)


class PolicyValueEvaluator:
    """Evaluates off-policy value of a single policy via standard OPE estimators."""

    @staticmethod
    def evaluate(
        dataset: BanditDataset,
        policy: Policy,
        *,
        estimator: Literal["ips", "snips", "dm", "dr"] | str = "dr",
        confidence_level: float = 0.95,
        policy_name: str = "evaluated_policy",
        reward_model: RewardModel | None = None,
        cross_fitting_folds: int = 3,
        seed: int = 42,
    ) -> PolicyValueReport:
        """Estimate the expected reward of a target policy."""
        est_lower = estimator.lower()
        if est_lower not in ALL_ESTIMATORS:
            raise ValueError(f"Unknown estimator '{estimator}'; choose from {ALL_ESTIMATORS}")

        data = dataset.data
        action_dist = policy.action_distribution(data.context)
        diag = diagnose(data, action_dist)

        predicted_rewards: FloatArray | None = None
        if est_lower in ("dm", "dr"):
            if reward_model is not None:
                reward_model.fit(data.context, data.action, data.reward)
                predicted_rewards = reward_model.predict(data.context)
            else:
                is_binary = bool(np.all(np.isin(data.reward, [0.0, 1.0])))
                inferred_type = RewardType.BINARY if is_binary else RewardType.GAUSSIAN

                def factory() -> RewardModel:
                    return StratifiedLinearRewardModel(
                        n_actions=data.n_actions,
                        reward_type=inferred_type,
                    )

                predicted_rewards = cross_fit_reward_model(
                    data, factory, n_folds=cross_fitting_folds, seed=seed
                )

        res: PolicyValueEstimate
        if est_lower in MODEL_BASED_ESTIMATORS:
            if predicted_rewards is None:
                raise ValueError(f"Estimator '{est_lower}' requires predicted rewards.")
            res = MODEL_BASED_ESTIMATORS[est_lower](
                data,
                action_dist,
                predicted_rewards,
                confidence_level=confidence_level,
            )
        else:
            res = ESTIMATORS[est_lower](
                data,
                action_dist,
                confidence_level=confidence_level,
            )

        return PolicyValueReport(
            policy_name=policy_name,
            estimator=est_lower,
            point_estimate=res.value,
            standard_error=res.standard_error,
            ci_lower=res.ci_lower,
            ci_upper=res.ci_upper,
            confidence_level=confidence_level,
            min_propensity=float(np.min(data.logging_propensity)),
            effective_sample_size=diag.weights.effective_sample_size,
            ess_fraction=diag.weights.effective_sample_size_fraction,
            max_weight_share=diag.weights.max_weight_share,
        )


@dataclass(frozen=True)
class PolicyComparisonReport:
    """Comprehensive, machine-readable report for paired policy advantage and decision-reversal."""

    schema_version: str
    timestamp: str
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]
    paired_advantage: dict[str, Any]
    decision: dict[str, Any]
    reversal_search: dict[str, Any]
    simultaneous_band: dict[str, Any]
    robust_mixture: dict[str, Any]
    assumptions_and_guarantees: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def outcome(self) -> str:
        return str(self.decision.get("outcome", "INCONCLUSIVE"))

    @property
    def is_actionable(self) -> bool:
        return bool(self.decision.get("is_actionable", False))

    @property
    def reversal_distance(self) -> float | None:
        val = self.reversal_search.get("reversal_distance")
        return float(val) if val is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Export the report as a JSON string or save to a file."""
        serialized = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(serialized, encoding="utf-8")
        return serialized

    def save(self, path: str | Path, indent: int = 2) -> None:
        """Save report to a JSON file."""
        self.to_json(path=path, indent=indent)

    def summary(self) -> str:
        """Render an ergonomic, human-readable ASCII summary of the analysis."""
        adv = self.paired_advantage
        dec = self.decision
        diag = self.diagnostics
        rev = self.reversal_search
        sim = self.simultaneous_band
        rob = self.robust_mixture

        pct = round(adv["confidence_level"] * 100)
        d_rev_str = (
            f"{rev['reversal_distance']:.4f} (analytical: {rev['exact_reversal_distance']:.4f})"
            if rev.get("reversal_distance") is not None
            else "None detected (Robust across bounds)"
        )

        n_s = self.metadata["n_samples"]
        n_a = self.metadata["n_actions"]
        c_d = self.metadata["context_dim"]
        k_f = self.metadata["cross_fitting_folds"]
        s_d = self.metadata["split_dim"]
        s_t = self.metadata["split_threshold"]

        lines = [
            "=" * 78,
            f"REVERSALLAB DECISION-REVERSAL AUDIT REPORT (v{self.schema_version})",
            "=" * 78,
            f"Generated:       {self.timestamp}",
            f"Dataset:         N = {n_s} observations, {n_a} actions, dim = {c_d}",
            f"Evaluation:      Estimator = {adv['estimator'].upper()} ({k_f}-fold cross-fitting)",
            f"Policy Margin:   delta = {dec['margin']:.4f} (Confidence level: {pct}%)",
            "",
            "1. PAIRED POLICY ADVANTAGE",
            (
                f"   Point Estimate (Delta):   {adv['point_estimate']:+.4f} "
                f"(SE: {adv['standard_error']:.4f})"
            ),
            f"   {pct}% Confidence Bound:    [{adv['ci_lower']:+.4f}, {adv['ci_upper']:+.4f}]",
            f"   Baseline Action:          {dec['outcome']} ({dec['reason']})",
            "",
            "2. RELIABILITY DIAGNOSTICS",
            f"   Gate Passed:              {'YES' if diag['gate_passed'] else 'NO (FAILURE)'}",
            (
                f"   Min Logging Propensity:   {diag['min_propensity']:.2e} "
                f"(Threshold: {diag['min_propensity_threshold']:.2e})"
            ),
            (
                f"   Effective Sample Size:    {diag['effective_sample_size']:.1f} "
                f"({diag['ess_fraction'] * 100:.1f}% of N)"
            ),
            f"   Max Single-Weight Share:  {diag['max_weight_share'] * 100:.1f}%",
            "",
            "3. DECISION-REVERSAL ROBUSTNESS (1-D Context Mixture Model)",
            f"   Stratum Partition:        x[{s_d}] >= {s_t:.2f}",
            f"   Reversal Distance (d*):   {d_rev_str}",
            f"   Search Status:            {rev['status']} ({rev['transition']})",
            f"   Taxonomy Category:        {rev['taxonomy_category']}",
            (
                f"   Forward Worst-Case Bound: {rob['worst_case_lower_bound']:+.4f} "
                f"(across Q_{rob['deploy_epsilon']:.2f})"
            ),
            f"   Robust Deployment Rule:   {rob['outcome']}",
            (
                f"   Simultaneous Band Bound:  {sim['min_simultaneous_lower']:+.4f} "
                f"(Simult Outcome: {sim['simultaneous_decision']})"
            ),
            "",
            "4. STATISTICAL ASSUMPTIONS & CLAIMS",
            f"   Supported:   {self.assumptions_and_guarantees['supported_claims']}",
            f"   Limitations: {self.assumptions_and_guarantees['unsupported_claims']}",
        ]

        if self.warnings:
            lines += ["", "WARNINGS:"] + [f"   ! {w}" for w in self.warnings]

        lines.append("=" * 78)
        return "\n".join(lines)


class PolicyComparison:
    """High-level paired policy advantage and decision-reversal analyzer."""

    @staticmethod
    def evaluate(
        dataset: BanditDataset,
        candidate_policy: Policy,
        baseline_policy: Policy,
        *,
        margin: float = 0.0,
        deploy_epsilon: float = 0.10,
        max_epsilon: float = 0.40,
        confidence_level: float = 0.95,
        stratum_partition: StratumPartition | None = None,
        estimator: Literal["dr", "ips", "snips", "dm"] | str = "dr",
        reward_model_factory: Any | None = None,
        cross_fitting_folds: int = 3,
        min_propensity_threshold: float = 1e-4,
        seed: int = 42,
    ) -> PolicyComparisonReport:
        """Execute complete, leakage-safe paired policy evaluation and decision reversal.

        Parameters
        ----------
        dataset : BanditDataset
            Logged bandit data with verified propensity support.
        candidate_policy : Policy
            New target candidate policy pi_e.
        baseline_policy : Policy
            Existing deployed baseline policy pi_b.
        margin : float, default=0.0
            Minimum operational improvement delta required for deployment.
        deploy_epsilon : float, default=0.10
            Robustness tolerance ball radius Q_eps around sample mixture.
        max_epsilon : float, default=0.40
            Maximum search radius for decision-reversal boundary.
        confidence_level : float, default=0.95
            Nominal coverage probability (1 - alpha).
        stratum_partition : StratumPartition, optional
            Hyperplane partition for two-stratum context mixture shift.
            Defaults to coordinate split on context dimension 0 at threshold 0.0.
        estimator : {"dr", "ips", "snips", "dm"}, default="dr"
            Off-policy estimator for paired difference estimation.
        reward_model_factory : callable, optional
            Zero-argument factory returning a RewardModel instance for cross-fitting.
            Defaults to StratifiedLinearRewardModel.
        cross_fitting_folds : int, default=3
            Number of K-fold cross-fitting partitions for reward models.
        min_propensity_threshold : float, default=1e-4
            Minimum allowable propensity before tripping diagnostic gate.
        seed : int, default=42
            Reproducible pseudorandom seed.
        """
        data = dataset.data
        partition = stratum_partition or StratumPartition(split_dim=0, split_threshold=0.0)
        cand_dist = candidate_policy.action_distribution(data.context)
        base_dist = baseline_policy.action_distribution(data.context)

        # 1. Evaluate operational diagnostic gates
        gate_thresholds = DiagnosticGateThresholds(
            min_propensity=min_propensity_threshold,
            min_ess_fraction=0.05,
            max_weight_share=0.20,
        )
        gate_report: DiagnosticGateReport = evaluate_diagnostic_gate(
            data, cand_dist, base_dist, thresholds=gate_thresholds
        )
        diag_passed = gate_report.passed
        diag_reason = gate_report.rejection_reasons[0] if not diag_passed else None

        warnings: list[str] = []
        if not diag_passed:
            warnings.extend(gate_report.rejection_reasons)

        # 2. Cross-fit reward models if required
        predicted_rewards: FloatArray | None = None
        if estimator.lower() in ("dm", "dr"):
            if reward_model_factory is None:
                is_binary = bool(np.all(np.isin(data.reward, [0.0, 1.0])))
                inferred_type = RewardType.BINARY if is_binary else RewardType.GAUSSIAN

                def factory() -> RewardModel:
                    return StratifiedLinearRewardModel(
                        n_actions=data.n_actions,
                        reward_type=inferred_type,
                        standardize=True,
                    )

                reward_model_factory = factory

            predicted_rewards = cross_fit_reward_model(
                data,
                reward_model_factory,
                n_folds=cross_fitting_folds,
                seed=seed,
            )

        # 3. Estimate paired difference
        paired: PairedDifferenceResult = evaluate_paired_difference(
            estimator,
            data,
            cand_dist,
            base_dist,
            predicted_rewards=predicted_rewards,
            confidence_level=confidence_level,
        )

        # 4. Create two-stratum mixture model
        strata_model: TwoStratumMixtureModel = create_two_stratum_model(
            data.context,
            paired.diff_scores,
            partition=partition,
            target_mixture=0.50,
        )

        # 5. Evaluate baseline deployment decision from target mixture q0
        base_dec = strata_model.evaluate_mixture_decision(
            strata_model.baseline_q0,
            margin=margin,
            confidence_level=confidence_level,
            diagnostic_passed=diag_passed,
            diagnostic_reason=diag_reason,
        )

        # 6. Execute reversal search (numerical bisection + analytical reference)
        search_res = find_decision_reversal(
            strata_model,
            margin=margin,
            max_epsilon=max_epsilon,
            confidence_level=confidence_level,
            diagnostic_passed=diag_passed,
            diagnostic_reason=diag_reason,
        )
        ana_res = find_analytical_decision_reversal(
            strata_model,
            margin=margin,
            max_epsilon=max_epsilon,
            confidence_level=confidence_level,
            diagnostic_passed=diag_passed,
            diagnostic_reason=diag_reason,
        )

        # 7. Evaluate simultaneous confidence band
        band = TwoStratumSimultaneousBand(strata_model, confidence_level=confidence_level)
        q_min = max(0.0, strata_model.baseline_q0 - deploy_epsilon)
        q_max = min(1.0, strata_model.baseline_q0 + deploy_epsilon)
        res_min = band.evaluate_band(q_min, margin)
        res_max = band.evaluate_band(q_max, margin)
        min_simult_l = min(res_min.simultaneous_ci_lower, res_max.simultaneous_ci_lower)

        if diag_passed and min_simult_l > margin:
            simult_outcome = DecisionOutcome.RECOMMEND.value
        elif (
            diag_passed
            and max(res_min.simultaneous_ci_upper, res_max.simultaneous_ci_upper) < margin
        ):
            simult_outcome = DecisionOutcome.HOLD.value
        else:
            simult_outcome = DecisionOutcome.INCONCLUSIVE.value

        # 8. Evaluate forward worst-case robust mixture baseline (Method F)
        robust_res = evaluate_robust_mixture_baseline(
            strata_model,
            margin=margin,
            deploy_epsilon=deploy_epsilon,
            confidence_level=confidence_level,
            diagnostic_passed=diag_passed,
            diagnostic_reason=diag_reason,
        )

        # 9. Classify reversal taxonomy
        taxonomy = classify_reversal_taxonomy(
            initial_decision=base_dec.outcome,
            reversed_decision=search_res.reversed_decision,
            evidence_distance=search_res.reversal_distance,
            true_delta_0=strata_model.stratum_0.mean_diff,
            true_delta_1=strata_model.stratum_1.mean_diff,
            margin=margin,
            baseline_q0=strata_model.baseline_q0,
            max_epsilon=max_epsilon,
        )

        disc = (
            abs(search_res.reversal_distance - ana_res.exact_reversal_distance)
            if (
                search_res.reversal_distance is not None
                and ana_res.exact_reversal_distance is not None
            )
            else 0.0
        )

        return PolicyComparisonReport(
            schema_version=SDK_SCHEMA_VERSION,
            timestamp=datetime.now(UTC).isoformat(),
            metadata={
                "n_samples": data.n_samples,
                "n_actions": data.n_actions,
                "context_dim": data.context_dim,
                "split_dim": partition.split_dim,
                "split_threshold": partition.split_threshold,
                "cross_fitting_folds": cross_fitting_folds,
                "seed": seed,
            },
            diagnostics={
                "gate_passed": diag_passed,
                "min_propensity": gate_report.min_propensity,
                "min_propensity_threshold": min_propensity_threshold,
                "effective_sample_size": gate_report.ess_candidate,
                "ess_fraction": gate_report.ess_fraction_candidate,
                "max_weight_share": gate_report.max_weight_share_candidate,
                "rejection_reasons": gate_report.rejection_reasons,
            },
            paired_advantage={
                "estimator": estimator.lower(),
                "point_estimate": paired.point_estimate,
                "standard_error": paired.standard_error,
                "ci_lower": paired.ci_lower,
                "ci_upper": paired.ci_upper,
                "confidence_level": confidence_level,
            },
            decision={
                "outcome": base_dec.outcome.value,
                "margin": margin,
                "reason": base_dec.reason,
                "is_actionable": base_dec.is_actionable,
            },
            reversal_search={
                "status": search_res.status,
                "transition": search_res.transition,
                "reversal_distance": search_res.reversal_distance,
                "exact_reversal_distance": ana_res.exact_reversal_distance,
                "analytical_discrepancy": disc,
                "reversed_q0": search_res.reversed_q0,
                "max_epsilon": max_epsilon,
                "taxonomy_category": taxonomy.transition_category,
                "has_margin_reversal": taxonomy.has_margin_reversal,
                "has_performance_reversal": taxonomy.has_performance_reversal,
            },
            simultaneous_band={
                "deploy_epsilon": deploy_epsilon,
                "min_simultaneous_lower": min_simult_l,
                "simultaneous_decision": simult_outcome,
            },
            robust_mixture={
                "deploy_epsilon": deploy_epsilon,
                "worst_case_lower_bound": robust_res.worst_case_lower_bound,
                "worst_case_upper_bound": robust_res.worst_case_upper_bound,
                "outcome": robust_res.outcome.value,
                "reason": robust_res.reason,
            },
            assumptions_and_guarantees={
                "supported_claims": (
                    "Asymptotically valid normal confidence intervals for paired difference "
                    "via Doubly Robust estimation with K-fold cross-fitting, under "
                    "unconfoundedness and positive propensity support."
                ),
                "unsupported_claims": (
                    "Does not establish safety against unobserved confounding, continuous "
                    "non-parametric density drift, or multidimensional demographic shifts "
                    "beyond the evaluated partition."
                ),
            },
            warnings=warnings,
        )


def run_quickstart(output_path: str | Path | None = None) -> PolicyComparisonReport:
    """Execute complete end-to-end policy evaluation and decision-reversal workflow."""
    print("=" * 78)
    print("ReversalLab End-to-End Decision-Reversal Workflow")
    print("=" * 78)

    # 1. Environment and Logged Bandit Data Generation
    print("\n[Step 1] Initializing synthetic environment and logging data...")
    rng = np.random.default_rng(42)
    env = SyntheticBanditEnvironment.random(
        n_actions=3,
        context_dim=2,
        rng=rng,
        reward_scale=1.5,
        bias_scale=0.0,
    )

    logging_policy = UniformPolicy(n_actions=env.n_actions)
    dataset = BanditDataset.from_synthetic(
        env,
        logging_policy=logging_policy,
        n_samples=2500,
        seed=101,
    )
    print(
        f"  Generated N = {dataset.n_samples} observations across {dataset.n_actions} actions "
        f"(context dim = {dataset.context_dim})."
    )

    # 2. Define Baseline and Candidate Policies
    print("\n[Step 2] Defining baseline (pi_b) and candidate (pi_e) policies...")
    baseline_policy = UniformPolicy(n_actions=env.n_actions)
    candidate_policy = LinearSoftmaxPolicy(
        weights=env.reward_weights,
        bias=env.reward_bias,
        inverse_temperature=1.5,
    )
    print("  Baseline:  Uniform exploration over 3 actions.")
    print("  Candidate: Linear softmax policy informed by environment reward logits.")

    # 3. Paired Difference Estimation & Decision-Reversal Analysis
    margin = 0.02
    deploy_eps = 0.10
    partition = StratumPartition(split_dim=0, split_threshold=0.0)

    print("\n[Step 3] Running leakage-safe DR evaluation and reversal search...")
    print(f"  Operational margin delta:    {margin:.4f}")
    print(f"  Deployment tolerance ball:   epsilon = {deploy_eps:.2f}")
    print("  Context partition:           x[0] >= 0.0")

    report: PolicyComparisonReport = PolicyComparison.evaluate(
        dataset,
        candidate_policy=candidate_policy,
        baseline_policy=baseline_policy,
        margin=margin,
        deploy_epsilon=deploy_eps,
        max_epsilon=0.40,
        confidence_level=0.95,
        stratum_partition=partition,
        estimator="dr",
        cross_fitting_folds=3,
        min_propensity_threshold=1e-4,
        seed=42,
    )

    # 4. Display Ergonomic Human-Readable Summary
    print("\n[Step 4] Reviewing structured decision audit report:")
    print(report.summary())

    # 5. Export Reproducible Machine-Readable Report
    save_file = Path(output_path) if output_path else Path("runs/quickstart_report.json")
    save_file.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(save_file)
    print(f"\n[Step 5] Exported machine-readable JSON report to: {save_file}")

    return report


__all__ = [
    "SDK_SCHEMA_VERSION",
    "BanditDataset",
    "EpsilonGreedyLinearPolicy",
    "LinearSoftmaxPolicy",
    "LoggedBanditData",
    "Policy",
    "PolicyComparison",
    "PolicyComparisonReport",
    "PolicyValueEvaluator",
    "PolicyValueReport",
    "StratumPartition",
    "UniformPolicy",
    "run_quickstart",
]
