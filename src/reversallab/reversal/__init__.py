"""Decision-Reversal analysis subpackage for ReversalLab."""

from __future__ import annotations

from reversallab.reversal.analytical import (
    AnalyticalCandidateRoot,
    AnalyticalReversalResult,
    find_analytical_decision_reversal,
    solve_quadratic_crossing_roots,
)
from reversallab.reversal.decision import (
    DecisionOutcome,
    DecisionResult,
    evaluate_decision,
    evaluate_decision_from_paired_result,
)
from reversallab.reversal.diagnostics import (
    DiagnosticGateReport,
    DiagnosticGateThresholds,
    evaluate_diagnostic_gate,
)
from reversallab.reversal.paired import (
    PairedDifferenceResult,
    compute_paired_difference_scores,
    evaluate_paired_difference,
)
from reversallab.reversal.robust_mixture import (
    RobustMixtureResult,
    evaluate_robust_mixture_baseline,
)
from reversallab.reversal.search import (
    ReversalSearchResult,
    bisect_decision_boundary,
    find_decision_reversal,
)
from reversallab.reversal.simultaneous import (
    SimultaneousBandResult,
    SimultaneousReversalResult,
    TwoStratumSimultaneousBand,
)
from reversallab.reversal.strata import (
    StratumPartition,
    StratumStatistics,
    TwoStratumMixtureModel,
    create_two_stratum_model,
)
from reversallab.reversal.taxonomy import (
    ReversalTaxonomyReport,
    classify_reversal_taxonomy,
)

__all__ = [
    "AnalyticalCandidateRoot",
    "AnalyticalReversalResult",
    "DecisionOutcome",
    "DecisionResult",
    "DiagnosticGateReport",
    "DiagnosticGateThresholds",
    "PairedDifferenceResult",
    "ReversalSearchResult",
    "ReversalTaxonomyReport",
    "RobustMixtureResult",
    "SimultaneousBandResult",
    "SimultaneousReversalResult",
    "StratumPartition",
    "StratumStatistics",
    "TwoStratumMixtureModel",
    "TwoStratumSimultaneousBand",
    "bisect_decision_boundary",
    "classify_reversal_taxonomy",
    "compute_paired_difference_scores",
    "create_two_stratum_model",
    "evaluate_decision",
    "evaluate_decision_from_paired_result",
    "evaluate_diagnostic_gate",
    "evaluate_paired_difference",
    "evaluate_robust_mixture_baseline",
    "find_analytical_decision_reversal",
    "find_decision_reversal",
    "solve_quadratic_crossing_roots",
]
