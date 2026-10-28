"""Exception hierarchy.

Every error raised deliberately by ReversalLab derives from :class:`ReversalLabError`
so that callers can distinguish library-detected problems from unexpected failures.
"""

from __future__ import annotations


class ReversalLabError(Exception):
    """Base class for all ReversalLab errors."""


class DataValidationError(ReversalLabError, ValueError):
    """Input arrays violate the logged-data contract (shape, dtype, range, finiteness)."""


class SupportViolationError(ReversalLabError, ValueError):
    """The evaluation policy places mass on actions the logging policy cannot take.

    Importance-weighted estimators are not identified in this case. ReversalLab never
    clips or replaces zero propensities with an arbitrary epsilon to hide the problem.
    """


class DegenerateWeightsError(ReversalLabError, ArithmeticError):
    """Importance weights make an estimator undefined (e.g. SNIPS with zero total weight)."""


class NumericalError(ReversalLabError, ArithmeticError):
    """A finite input produced a non-finite intermediate value (for example an overflowing weight).

    ReversalLab does not clip, winsorise, or replace that value.
    """


class InsufficientDataError(ReversalLabError, ValueError):
    """Too few observations to compute the requested quantity."""


class ConfigurationError(ReversalLabError, ValueError):
    """An experiment or simulation configuration is invalid."""
