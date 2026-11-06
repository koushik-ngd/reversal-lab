"""Integration tests for ReversalLab FastAPI service."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from reversallab import __version__
from reversallab.api import (
    BanditDataset,
    LinearSoftmaxPolicy,
    PolicyComparison,
    PolicyValueEvaluator,
    UniformPolicy,
)
from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.service.app import MAX_REQUEST_BYTES, create_app
from reversallab.service.metrics import metrics
from reversallab.service.schemas import PolicyEvaluationRequest


@pytest.fixture
def client() -> TestClient:
    """Create test client with clean metrics."""
    metrics.reset()
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_dataset_and_policies() -> dict[str, Any]:
    """Generate reproducible sample dataset and policies for parity tests."""
    rng = np.random.default_rng(42)
    env = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=rng, reward_scale=1.0, bias_scale=0.0
    )
    logging_policy = UniformPolicy(n_actions=3)
    dataset = BanditDataset.from_synthetic(
        env, logging_policy=logging_policy, n_samples=300, seed=7
    )

    w_cand = rng.normal(size=(2, 3)).tolist()
    w_base = rng.normal(size=(2, 3)).tolist()

    return {
        "dataset_payload": {
            "contexts": dataset.context.tolist(),
            "actions": dataset.action.tolist(),
            "rewards": dataset.reward.tolist(),
            "propensities": dataset.logging_propensity.tolist(),
            "n_actions": 3,
        },
        "candidate_spec": {
            "type": "linear_softmax",
            "weights": w_cand,
            "inverse_temperature": 1.0,
        },
        "baseline_spec": {
            "type": "linear_softmax",
            "weights": w_base,
            "inverse_temperature": 1.0,
        },
        "sdk_dataset": dataset,
        "sdk_candidate": LinearSoftmaxPolicy(weights=np.array(w_cand), inverse_temperature=1.0),
        "sdk_baseline": LinearSoftmaxPolicy(weights=np.array(w_base), inverse_temperature=1.0),
    }


def test_health_endpoint(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "reversallab"
    assert "timestamp" in data
    assert "x-request-id" in res.headers


def test_version_endpoint(client: TestClient) -> None:
    res = client.get("/version")
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == __version__
    assert data["schema_version"] == "1.0.0"
    assert data["api_version"] == "v1"


def test_openapi_schema_generation(client: TestClient) -> None:
    res = client.get("/openapi.json")
    assert res.status_code == 200
    data = res.json()
    assert "paths" in data
    assert "/v1/evaluate/policy" in data["paths"]
    assert "/v1/evaluate/comparison" in data["paths"]


def test_sdk_numerical_parity_single_policy(
    client: TestClient, sample_dataset_and_policies: dict[str, Any]
) -> None:
    payload = sample_dataset_and_policies["dataset_payload"]
    policy_spec = sample_dataset_and_policies["candidate_spec"]
    sdk_dataset = sample_dataset_and_policies["sdk_dataset"]
    sdk_policy = sample_dataset_and_policies["sdk_candidate"]

    for estimator in ["ips", "snips", "dm", "dr"]:
        # SDK Evaluation
        sdk_rep = PolicyValueEvaluator.evaluate(
            dataset=sdk_dataset,
            policy=sdk_policy,
            estimator=estimator,
            cross_fitting_folds=3,
            confidence_level=0.95,
            seed=42,
        )

        # Service Evaluation
        req_body = {
            "dataset": payload,
            "policy": policy_spec,
            "estimator": estimator,
            "cross_fitting_folds": 3,
            "confidence_level": 0.95,
            "seed": 42,
        }
        res = client.post("/v1/evaluate/policy", json=req_body)
        assert res.status_code == 200, f"Failed for {estimator}: {res.text}"
        res_data = res.json()
        assert res_data["status"] == "success"

        api_rep = res_data["report"]
        np.testing.assert_allclose(
            api_rep["point_estimate"],
            sdk_rep.point_estimate,
            rtol=1e-12,
            err_msg=f"Estimator {estimator} point estimate mismatch",
        )
        np.testing.assert_allclose(
            api_rep["standard_error"],
            sdk_rep.standard_error,
            rtol=1e-12,
            err_msg=f"Estimator {estimator} standard error mismatch",
        )
        np.testing.assert_allclose(
            api_rep["ci_lower"],
            sdk_rep.ci_lower,
            rtol=1e-12,
            err_msg=f"Estimator {estimator} ci_lower mismatch",
        )
        np.testing.assert_allclose(
            api_rep["ci_upper"],
            sdk_rep.ci_upper,
            rtol=1e-12,
            err_msg=f"Estimator {estimator} ci_upper mismatch",
        )


def test_sdk_numerical_parity_paired_comparison(
    client: TestClient, sample_dataset_and_policies: dict[str, Any]
) -> None:
    payload = sample_dataset_and_policies["dataset_payload"]
    cand_spec = sample_dataset_and_policies["candidate_spec"]
    base_spec = sample_dataset_and_policies["baseline_spec"]
    sdk_dataset = sample_dataset_and_policies["sdk_dataset"]
    sdk_cand = sample_dataset_and_policies["sdk_candidate"]
    sdk_base = sample_dataset_and_policies["sdk_baseline"]

    # SDK Evaluation
    sdk_comp = PolicyComparison.evaluate(
        dataset=sdk_dataset,
        candidate_policy=sdk_cand,
        baseline_policy=sdk_base,
        estimator="dr",
        margin=0.01,
        deploy_epsilon=0.10,
        max_epsilon=0.30,
        confidence_level=0.95,
        cross_fitting_folds=3,
        seed=42,
    )

    # API Evaluation
    req_body = {
        "dataset": payload,
        "candidate_policy": cand_spec,
        "baseline_policy": base_spec,
        "margin": 0.01,
        "deploy_epsilon": 0.10,
        "max_epsilon": 0.30,
        "estimator": "dr",
        "cross_fitting_folds": 3,
        "confidence_level": 0.95,
        "seed": 42,
    }
    res = client.post("/v1/evaluate/comparison", json=req_body)
    assert res.status_code == 200, res.text
    res_data = res.json()
    assert res_data["status"] == "success"
    api_comp = res_data["report"]

    assert api_comp["schema_version"] == "1.0.0"
    assert api_comp["decision"]["outcome"] == sdk_comp.decision["outcome"]

    np.testing.assert_allclose(
        api_comp["paired_advantage"]["point_estimate"],
        sdk_comp.paired_advantage["point_estimate"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        api_comp["paired_advantage"]["standard_error"],
        sdk_comp.paired_advantage["standard_error"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        api_comp["paired_advantage"]["ci_lower"],
        sdk_comp.paired_advantage["ci_lower"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        api_comp["paired_advantage"]["ci_upper"],
        sdk_comp.paired_advantage["ci_upper"],
        rtol=1e-12,
    )


def test_reliability_gate_returns_inconclusive(
    client: TestClient, sample_dataset_and_policies: dict[str, Any]
) -> None:
    payload = dict(sample_dataset_and_policies["dataset_payload"])
    # Induce an active logging propensity deficit below threshold 1e-4
    props = list(payload["propensities"])
    props[0] = 1e-6
    payload["propensities"] = props

    req_body = {
        "dataset": payload,
        "candidate_policy": sample_dataset_and_policies["candidate_spec"],
        "baseline_policy": sample_dataset_and_policies["baseline_spec"],
        "estimator": "dr",
        "min_propensity_threshold": 1e-4,
        "cross_fitting_folds": 3,
        "seed": 42,
    }

    res = client.post("/v1/evaluate/comparison", json=req_body)
    # Must NOT return HTTP 422! Gate failure returns 200 with INCONCLUSIVE decision
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "inconclusive"
    report = data["report"]
    assert report["decision"]["outcome"] == "INCONCLUSIVE"
    assert report["diagnostics"]["gate_passed"] is False
    assert any(
        "Active propensity deficit" in reason
        for reason in report["diagnostics"]["rejection_reasons"]
    )


def test_invalid_inputs_return_http_422(
    client: TestClient, sample_dataset_and_policies: dict[str, Any]
) -> None:
    # 1. Non-positive propensity
    bad_payload = dict(sample_dataset_and_policies["dataset_payload"])
    bad_props = list(bad_payload["propensities"])
    bad_props[0] = 0.0
    bad_payload["propensities"] = bad_props
    res = client.post(
        "/v1/evaluate/policy",
        json={"dataset": bad_payload, "policy": sample_dataset_and_policies["candidate_spec"]},
    )
    assert res.status_code == 422
    data = res.json()
    assert data["title"] == "Unprocessable Entity"
    assert "Propensities must be in (0, 1]" in data["detail"]

    # 2. Mismatched array length
    bad_payload2 = dict(sample_dataset_and_policies["dataset_payload"])
    bad_payload2["actions"] = bad_payload2["actions"][:-5]
    res2 = client.post(
        "/v1/evaluate/policy",
        json={"dataset": bad_payload2, "policy": sample_dataset_and_policies["candidate_spec"]},
    )
    assert res2.status_code == 422
    assert "Array length mismatch" in res2.json()["detail"]

    # 3. Action out of bounds
    bad_payload3 = dict(sample_dataset_and_policies["dataset_payload"])
    bad_acts = list(bad_payload3["actions"])
    bad_acts[0] = 99
    bad_payload3["actions"] = bad_acts
    res3 = client.post(
        "/v1/evaluate/policy",
        json={"dataset": bad_payload3, "policy": sample_dataset_and_policies["candidate_spec"]},
    )
    assert res3.status_code == 422
    assert "Action index 99 out of bounds" in res3.json()["detail"]

    # 4. Non-finite values
    n_act = len(sample_dataset_and_policies["dataset_payload"]["actions"])
    raw_bad_json = json.dumps(
        {
            "dataset": {
                **sample_dataset_and_policies["dataset_payload"],
                "rewards": ["invalid_nan"] * n_act,
            },
            "policy": sample_dataset_and_policies["candidate_spec"],
        }
    )
    res4 = client.post(
        "/v1/evaluate/policy",
        content=raw_bad_json,
        headers={"Content-Type": "application/json"},
    )
    assert res4.status_code == 422
    assert "Unprocessable Entity" in res4.json()["title"]


def test_computation_budget_limit_rejection() -> None:
    from pydantic import ValidationError

    from reversallab.service.schemas import PolicyEvaluationRequest

    # Validating that model rejects budget > 1,000,000 (N=25000, K=50, folds=10 -> 12.5M)
    oversized_dataset = {
        "contexts": [[0.0, 1.0]] * 25000,
        "actions": [0] * 25000,
        "rewards": [1.0] * 25000,
        "propensities": [0.5] * 25000,
        "n_actions": 50,
    }
    with pytest.raises(ValidationError, match="Requested computation budget"):
        PolicyEvaluationRequest.model_validate(
            {
                "dataset": oversized_dataset,
                "policy": {"type": "uniform"},
                "cross_fitting_folds": 10,
            }
        )


def test_request_body_size_limit_http_413(
    client: TestClient, sample_dataset_and_policies: dict[str, Any]
) -> None:
    # Send request with simulated oversized content-length
    headers = {"Content-Length": str(MAX_REQUEST_BYTES + 1000)}
    res = client.post(
        "/v1/evaluate/policy",
        json={"dataset": sample_dataset_and_policies["dataset_payload"]},
        headers=headers,
    )
    assert res.status_code == 413
    data = res.json()
    assert data["title"] == "Payload Too Large"
    assert "exceeds maximum allowed limit" in data["detail"]


def test_request_id_propagation_and_tracing(
    client: TestClient, sample_dataset_and_policies: dict[str, Any]
) -> None:
    custom_id = "custom-trace-uuid-9999"
    res = client.get("/health", headers={"x-request-id": custom_id})
    assert res.status_code == 200
    assert res.headers["x-request-id"] == custom_id

    req_body = {
        "dataset": sample_dataset_and_policies["dataset_payload"],
        "policy": sample_dataset_and_policies["candidate_spec"],
    }
    res2 = client.post(
        "/v1/evaluate/policy",
        json=req_body,
        headers={"x-request-id": custom_id},
    )
    assert res2.status_code == 200
    assert res2.headers["x-request-id"] == custom_id
    assert res2.json()["request_id"] == custom_id


def test_logging_redaction_and_structure(
    client: TestClient,
    sample_dataset_and_policies: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="reversallab.service")
    req_body = {
        "dataset": sample_dataset_and_policies["dataset_payload"],
        "policy": sample_dataset_and_policies["candidate_spec"],
    }
    res = client.post("/v1/evaluate/policy", json=req_body)
    assert res.status_code == 200

    captured = caplog.text
    assert "request_id" in captured
    assert "duration_ms" in captured
    assert "/v1/evaluate/policy" in captured

    # Crucial security check: Ensure raw contexts, rewards, or actions are NEVER logged
    raw_ctx_sample = json.dumps(sample_dataset_and_policies["dataset_payload"]["contexts"][0])
    assert raw_ctx_sample not in captured, "Raw context leaked into logs!"


def test_metrics_endpoint(client: TestClient, sample_dataset_and_policies: dict[str, Any]) -> None:
    client.get("/health")
    req_body = {
        "dataset": sample_dataset_and_policies["dataset_payload"],
        "policy": sample_dataset_and_policies["candidate_spec"],
    }
    client.post("/v1/evaluate/policy", json=req_body)

    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "/health:200" in data["requests_total"]
    assert "/v1/evaluate/policy:200" in data["requests_total"]
    assert "latencies_ms" in data
    assert "/v1/evaluate/policy" in data["latencies_ms"]
    assert "memory_rss_mb" in data


def test_streaming_body_size_limit_without_content_length(client: TestClient) -> None:
    # Generator yielding stream chunks exceeding MAX_REQUEST_BYTES without Content-Length
    def oversized_chunk_stream() -> Any:
        chunk = b"X" * (1024 * 1024)  # 1MB
        for _ in range(11):  # 11MB total
            yield chunk

    res = client.post(
        "/v1/evaluate/policy",
        content=oversized_chunk_stream(),
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 413
    assert res.json()["title"] == "Payload Too Large"


def test_dimension_aware_compute_budget_validation() -> None:
    # High dimension D=50 with N=5000, K=10, folds=5 -> 12,500,000 budget
    high_dim_dataset = {
        "contexts": [[0.0] * 50] * 5000,
        "actions": [0] * 5000,
        "rewards": [1.0] * 5000,
        "propensities": [0.1] * 5000,
        "n_actions": 10,
    }
    with pytest.raises(ValidationError, match="Requested computation budget"):
        PolicyEvaluationRequest.model_validate(
            {
                "dataset": high_dim_dataset,
                "policy": {"type": "uniform"},
                "cross_fitting_folds": 5,
            }
        )
