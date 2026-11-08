"""ReversalLab: off-policy evaluation and deployment-decision robustness analysis."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reversallab")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.1.0rc1"

from reversallab.api import (
    SDK_SCHEMA_VERSION,
    BanditDataset,
    EpsilonGreedyLinearPolicy,
    LinearSoftmaxPolicy,
    LoggedBanditData,
    Policy,
    PolicyComparison,
    PolicyComparisonReport,
    PolicyValueEvaluator,
    PolicyValueReport,
    StratumPartition,
    UniformPolicy,
    run_quickstart,
)

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
    "__version__",
    "run_quickstart",
]
