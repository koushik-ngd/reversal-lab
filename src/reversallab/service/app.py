"""FastAPI evaluation service for ReversalLab."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from reversallab import __version__
from reversallab.api import (
    SDK_SCHEMA_VERSION,
    BanditDataset,
    EpsilonGreedyLinearPolicy,
    LinearSoftmaxPolicy,
    Policy,
    PolicyComparison,
    PolicyValueEvaluator,
    StratumPartition,
    UniformPolicy,
)
from reversallab.exceptions import DataValidationError, ReversalLabError
from reversallab.service.metrics import metrics
from reversallab.service.schemas import (
    ComparisonResponse,
    EvaluationResponse,
    HealthResponse,
    PolicyComparisonRequest,
    PolicyEvaluationRequest,
    PolicySpec,
    VersionResponse,
)

# Configure service-specific logger without polluting root logger
logger = logging.getLogger("reversallab.service")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

MAX_REQUEST_BYTES = 10 * 1024 * 1024  # 10 MB limit


def _build_policy(spec: PolicySpec, n_actions: int) -> Policy:
    """Instantiate a policy object from a validated specification."""
    if spec.type == "uniform":
        return UniformPolicy(n_actions=n_actions)
    if spec.type == "linear_softmax":
        assert spec.weights is not None
        w = np.asarray(spec.weights, dtype=np.float64)
        return LinearSoftmaxPolicy(weights=w, inverse_temperature=spec.inverse_temperature)
    if spec.type == "epsilon_greedy":
        assert spec.weights is not None
        w = np.asarray(spec.weights, dtype=np.float64)
        return EpsilonGreedyLinearPolicy(weights=w, epsilon=spec.epsilon)
    raise DataValidationError(f"Unsupported policy type '{spec.type}'.")


class RequestSizeAndTracingMiddleware(BaseHTTPMiddleware):
    """Enforces request payload size limits and injects request IDs with structured timing."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        # 1. Establish Request ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id

        # 2. Check Content-Length if present
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                cl_int = int(content_length)
                if cl_int > self.max_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "type": "https://reversallab.org/errors/payload-too-large",
                            "title": "Payload Too Large",
                            "status": status.HTTP_413_CONTENT_TOO_LARGE,
                            "detail": (
                                f"Request body length ({cl_int} bytes) exceeds maximum "
                                f"allowed limit of {self.max_bytes} bytes."
                            ),
                            "instance": request.url.path,
                            "request_id": request_id,
                        },
                        headers={"x-request-id": request_id},
                    )
            except ValueError:
                pass

        # 3. Stream body incrementally with byte counter to protect against
        # chunked transfer encoding or missing Content-Length header
        chunks: list[bytes] = []
        total_bytes = 0
        async for chunk in request.stream():
            total_bytes += len(chunk)
            if total_bytes > self.max_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={
                        "type": "https://reversallab.org/errors/payload-too-large",
                        "title": "Payload Too Large",
                        "status": status.HTTP_413_CONTENT_TOO_LARGE,
                        "detail": (
                            f"Request body size exceeded maximum allowed limit of "
                            f"{self.max_bytes} bytes."
                        ),
                        "instance": request.url.path,
                        "request_id": request_id,
                    },
                    headers={"x-request-id": request_id},
                )
            chunks.append(chunk)

        body = b"".join(chunks)
        # Store body on request so downstream handlers can access it repeatedly
        request._body = body

        # 4. Check for client disconnection before execution
        if await request.is_disconnected():
            return Response(status_code=499, headers={"x-request-id": request_id})

        # 5. Measure request execution time and record metrics
        metrics.record_request_start()
        start_time = time.perf_counter()
        status_code = 500
        try:
            response = cast(Response, await call_next(request))
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            metrics.record_request_end(request.url.path, status_code, duration_ms)

            # Emit structured JSON log without logging raw arrays
            log_payload = {
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            }
            logger.info(json.dumps(log_payload))


def create_app() -> FastAPI:
    """Application factory for the ReversalLab evaluation service."""
    app = FastAPI(
        title="ReversalLab Off-Policy Evaluation & Decision-Reversal Service",
        description=(
            "Statistically grounded contextual-bandit off-policy evaluation, paired policy "
            "difference estimation, and population-mixture decision-reversal robustness analysis."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestSizeAndTracingMiddleware, max_bytes=MAX_REQUEST_BYTES)

    # Error handling middleware
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        errors = exc.errors()
        error_details = []
        for err in errors:
            loc = " -> ".join(str(item) for item in err.get("loc", []))
            msg = err.get("msg", "")
            error_details.append(f"{loc}: {msg}")
        detail_msg = "; ".join(error_details) if error_details else "Invalid request body."

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "type": "https://reversallab.org/errors/validation-error",
                "title": "Unprocessable Entity",
                "status": status.HTTP_422_UNPROCESSABLE_CONTENT,
                "detail": detail_msg,
                "instance": request.url.path,
                "request_id": request_id,
            },
            headers={"x-request-id": request_id},
        )

    @app.exception_handler(DataValidationError)
    async def data_validation_exception_handler(
        request: Request, exc: DataValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "type": "https://reversallab.org/errors/data-validation-error",
                "title": "Data Contract Violation",
                "status": status.HTTP_422_UNPROCESSABLE_CONTENT,
                "detail": str(exc),
                "instance": request.url.path,
                "request_id": request_id,
            },
            headers={"x-request-id": request_id},
        )

    @app.exception_handler(ReversalLabError)
    async def reversallab_exception_handler(
        request: Request, exc: ReversalLabError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "type": "https://reversallab.org/errors/client-error",
                "title": "Invalid Evaluation Request",
                "status": status.HTTP_400_BAD_REQUEST,
                "detail": str(exc),
                "instance": request.url.path,
                "request_id": request_id,
            },
            headers={"x-request-id": request_id},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        # Internal log contains traceback for debugging; response is sanitized
        logger.error(
            json.dumps(
                {
                    "request_id": request_id,
                    "error": exc.__class__.__name__,
                    "detail": str(exc),
                }
            )
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "type": "https://reversallab.org/errors/internal-server-error",
                "title": "Internal Server Error",
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "detail": "An internal unexpected error occurred. Quote request ID for tracking.",
                "instance": request.url.path,
                "request_id": request_id,
            },
            headers={"x-request-id": request_id},
        )

    # Routes
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """System health and liveness probe."""
        return HealthResponse(
            status="healthy",
            service="reversallab",
            timestamp=datetime.now(UTC).isoformat(),
        )

    @app.get("/version", response_model=VersionResponse)
    def version() -> VersionResponse:
        """Version identifiers for service, package, and JSON report schema."""
        return VersionResponse(
            version=__version__,
            schema_version=SDK_SCHEMA_VERSION,
            api_version="v1",
        )

    @app.get("/metrics")
    def get_metrics() -> dict[str, Any]:
        """Internal operational metrics snapshot."""
        return metrics.snapshot()

    @app.post(
        "/v1/evaluate/policy",
        response_model=EvaluationResponse,
        summary="Evaluate single policy value via off-policy estimation",
    )
    def evaluate_policy(req: PolicyEvaluationRequest, request: Request) -> EvaluationResponse:
        """Evaluate expected reward using IPS, SNIPS, DM, or DR estimation."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Convert arrays
        contexts = np.asarray(req.dataset.contexts, dtype=np.float64)
        actions = np.asarray(req.dataset.actions, dtype=np.int64)
        rewards = np.asarray(req.dataset.rewards, dtype=np.float64)
        propensities = np.asarray(req.dataset.propensities, dtype=np.float64)

        dataset = BanditDataset.from_arrays(
            contexts=contexts,
            actions=actions,
            rewards=rewards,
            propensities=propensities,
            n_actions=req.dataset.n_actions,
        )

        policy = _build_policy(req.policy, req.dataset.n_actions)

        seed = req.seed if req.seed is not None else 42
        report = PolicyValueEvaluator.evaluate(
            dataset=dataset,
            policy=policy,
            estimator=req.estimator,
            cross_fitting_folds=req.cross_fitting_folds,
            confidence_level=req.confidence_level,
            seed=seed,
        )

        report_dict = report.to_dict()
        metrics.record_evaluation(req.estimator, "EVALUATED")

        return EvaluationResponse(
            request_id=request_id,
            status="success",
            timestamp=datetime.now(UTC).isoformat(),
            report=report_dict,
        )

    @app.post(
        "/v1/evaluate/comparison",
        response_model=ComparisonResponse,
        summary="Evaluate paired policy difference, diagnostic gates, and decision reversal",
    )
    def evaluate_comparison(req: PolicyComparisonRequest, request: Request) -> ComparisonResponse:
        """Run paired policy superiority testing, diagnostic gates, and mixture reversal search."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        contexts = np.asarray(req.dataset.contexts, dtype=np.float64)
        actions = np.asarray(req.dataset.actions, dtype=np.int64)
        rewards = np.asarray(req.dataset.rewards, dtype=np.float64)
        propensities = np.asarray(req.dataset.propensities, dtype=np.float64)

        dataset = BanditDataset.from_arrays(
            contexts=contexts,
            actions=actions,
            rewards=rewards,
            propensities=propensities,
            n_actions=req.dataset.n_actions,
        )

        candidate_policy = _build_policy(req.candidate_policy, req.dataset.n_actions)
        baseline_policy = _build_policy(req.baseline_policy, req.dataset.n_actions)

        stratum_partition = (
            StratumPartition(
                split_dim=req.stratum_partition.split_dim,
                split_threshold=req.stratum_partition.threshold,
            )
            if req.stratum_partition is not None
            else None
        )

        seed = req.seed if req.seed is not None else 42
        comp_report = PolicyComparison.evaluate(
            dataset=dataset,
            candidate_policy=candidate_policy,
            baseline_policy=baseline_policy,
            estimator=req.estimator,
            margin=req.margin,
            deploy_epsilon=req.deploy_epsilon,
            max_epsilon=req.max_epsilon,
            stratum_partition=stratum_partition,
            confidence_level=req.confidence_level,
            cross_fitting_folds=req.cross_fitting_folds,
            min_propensity_threshold=req.min_propensity_threshold,
            seed=seed,
        )

        outcome = comp_report.decision["outcome"]
        gate_passed = comp_report.diagnostics["gate_passed"]

        metrics.record_evaluation(req.estimator, outcome)
        if not gate_passed:
            for reason in comp_report.diagnostics["rejection_reasons"]:
                metrics.record_gate_failure(reason)

        return ComparisonResponse(
            request_id=request_id,
            status="inconclusive" if outcome == "INCONCLUSIVE" else "success",
            timestamp=datetime.now(UTC).isoformat(),
            report=comp_report.to_dict(),
        )

    return app


# Default app instance
app = create_app()
