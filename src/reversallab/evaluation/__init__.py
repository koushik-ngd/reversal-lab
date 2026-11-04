"""Off-policy evaluation: weights, estimators, diagnostics, reward models, and cross-fitting."""

from reversallab.evaluation.bootstrap import (
    BootstrapResult,
    bootstrap_evaluate,
    construct_bootstrap_intervals,
    resample_logged_data,
)
from reversallab.evaluation.cross_fitting import cross_fit_reward_model
from reversallab.evaluation.diagnostics import (
    DiagnosticReport,
    DiagnosticThresholds,
    DiagnosticWarning,
    diagnose,
    diagnose_interval,
    estimator_disagreement,
    weight_diagnostics,
)
from reversallab.evaluation.estimators import (
    ALL_ESTIMATORS,
    ESTIMATORS,
    MODEL_BASED_ESTIMATORS,
    ModelBasedEstimator,
    PolicyValueEstimate,
    dm,
    dr,
    ips,
    snips,
)
from reversallab.evaluation.influence_scores import (
    InfluenceScoreResult,
    compute_dm_pseudo_outcomes,
    compute_dr_influence_scores,
    compute_ips_influence_scores,
    compute_snips_influence_scores,
    evaluate_influence_inference,
)
from reversallab.evaluation.reward_models import (
    ConstantRewardModel,
    MisspecifiedLinearRewardModel,
    RewardModel,
    StratifiedLinearRewardModel,
)
from reversallab.evaluation.weights import find_support_violations, importance_weights

__all__ = [
    "ALL_ESTIMATORS",
    "ESTIMATORS",
    "MODEL_BASED_ESTIMATORS",
    "BootstrapResult",
    "ConstantRewardModel",
    "DiagnosticReport",
    "DiagnosticThresholds",
    "DiagnosticWarning",
    "InfluenceScoreResult",
    "MisspecifiedLinearRewardModel",
    "ModelBasedEstimator",
    "PolicyValueEstimate",
    "RewardModel",
    "StratifiedLinearRewardModel",
    "bootstrap_evaluate",
    "compute_dm_pseudo_outcomes",
    "compute_dr_influence_scores",
    "compute_ips_influence_scores",
    "compute_snips_influence_scores",
    "construct_bootstrap_intervals",
    "cross_fit_reward_model",
    "diagnose",
    "diagnose_interval",
    "dm",
    "dr",
    "estimator_disagreement",
    "evaluate_influence_inference",
    "find_support_violations",
    "importance_weights",
    "ips",
    "resample_logged_data",
    "snips",
    "weight_diagnostics",
]
