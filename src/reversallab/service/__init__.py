"""ReversalLab FastAPI Evaluation Service."""

from __future__ import annotations

from reversallab.service.app import app, create_app
from reversallab.service.metrics import metrics

__all__ = ["app", "create_app", "metrics"]
