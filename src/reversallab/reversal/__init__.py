"""Decision-reversal subpackage.

Provides paired evaluation, decision rules, strata, and perturbation search.
"""

from __future__ import annotations

from reversallab.reversal.decision import (
    DecisionOutcome,
    DecisionResult,
    evaluate_decision,
    evaluate_decision_from_paired_result,
)
from reversallab.reversal.paired import (
    PairedDifferenceResult,
    compute_paired_difference_scores,
    evaluate_paired_difference,
)
from reversallab.reversal.search import (
    ReversalSearchResult,
    bisect_decision_boundary,
    find_decision_reversal,
)
from reversallab.reversal.strata import (
    StratumPartition,
    StratumStatistics,
    TwoStratumMixtureModel,
    create_two_stratum_model,
)

__all__ = [
    "DecisionOutcome",
    "DecisionResult",
    "PairedDifferenceResult",
    "ReversalSearchResult",
    "StratumPartition",
    "StratumStatistics",
    "TwoStratumMixtureModel",
    "bisect_decision_boundary",
    "compute_paired_difference_scores",
    "create_two_stratum_model",
    "evaluate_decision",
    "evaluate_decision_from_paired_result",
    "evaluate_paired_difference",
    "find_decision_reversal",
]
