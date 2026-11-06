"""Automated container verification script for ReversalLab Docker image."""

from __future__ import annotations

import subprocess
import time
from typing import Any

import httpx
import numpy as np

from reversallab.api import BanditDataset, LinearSoftmaxPolicy, PolicyComparison, UniformPolicy
from reversallab.data.synthetic import SyntheticBanditEnvironment

PORT = 8088
CONTAINER_NAME = "reversallab-container-test"
IMAGE_TAG = "reversallab-service:test"


def start_container() -> None:
    # Cleanup any previous run
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "-p",
        f"{PORT}:8000",
        IMAGE_TAG,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"[Container] Started with container ID: {res.stdout.strip()[:12]}")


def wait_for_ready(max_wait_s: float = 10.0) -> None:
    url = f"http://localhost:{PORT}/health"
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                print(f"[Health] Container is ready in {round(time.time() - t0, 2)}s.")
                return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("Container failed to become healthy within timeout.")


def verify_endpoints_and_parity() -> dict[str, Any]:
    base_url = f"http://localhost:{PORT}"

    # 1. Health & Version
    h_res = httpx.get(f"{base_url}/health").json()
    assert h_res["status"] == "healthy"

    v_res = httpx.get(f"{base_url}/version").json()
    assert v_res["schema_version"] == "1.0.0"
    print(f"[Version] API Version: {v_res['api_version']}, Package Version: {v_res['version']}")

    # 2. SDK Parity Fixture
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

    sdk_cand = LinearSoftmaxPolicy(weights=np.array(w_cand), inverse_temperature=1.0)
    sdk_base = LinearSoftmaxPolicy(weights=np.array(w_base), inverse_temperature=1.0)

    sdk_comp = PolicyComparison.evaluate(
        dataset=dataset,
        candidate_policy=sdk_cand,
        baseline_policy=sdk_base,
        estimator="dr",
        margin=0.01,
        deploy_epsilon=0.10,
        max_epsilon=0.30,
        cross_fitting_folds=3,
        seed=42,
    )

    req_body: dict[str, Any] = {
        "dataset": {
            "contexts": dataset.context.tolist(),
            "actions": dataset.action.tolist(),
            "rewards": dataset.reward.tolist(),
            "propensities": dataset.logging_propensity.tolist(),
            "n_actions": 3,
        },
        "candidate_policy": {"type": "linear_softmax", "weights": w_cand},
        "baseline_policy": {"type": "linear_softmax", "weights": w_base},
        "margin": 0.01,
        "deploy_epsilon": 0.10,
        "max_epsilon": 0.30,
        "estimator": "dr",
        "cross_fitting_folds": 3,
        "seed": 42,
    }

    c_res = httpx.post(f"{base_url}/v1/evaluate/comparison", json=req_body, timeout=10.0)
    assert c_res.status_code == 200, c_res.text
    api_rep = c_res.json()["report"]

    np.testing.assert_allclose(
        api_rep["paired_advantage"]["point_estimate"],
        sdk_comp.paired_advantage["point_estimate"],
        rtol=1e-12,
    )
    cand_point = api_rep["paired_advantage"]["point_estimate"]
    sdk_point = sdk_comp.paired_advantage["point_estimate"]
    print(f"[Parity] Matched SDK Paired Point Estimate: {cand_point:.6f} vs SDK {sdk_point:.6f}")

    # 3. Invalid Request -> HTTP 422
    bad_req: dict[str, Any] = dict(req_body)
    bad_ds: dict[str, Any] = dict(bad_req["dataset"])
    bad_ds["actions"] = bad_ds["actions"][:-1]  # length mismatch
    bad_req["dataset"] = bad_ds
    err_res = httpx.post(f"{base_url}/v1/evaluate/comparison", json=bad_req, timeout=5.0)
    assert err_res.status_code == 422
    print(f"[Validation] Successfully caught length mismatch: HTTP {err_res.status_code}")

    # 4. Oversized Request -> HTTP 413
    oversized_body = b" " * (10 * 1024 * 1024 + 1024)
    large_res = httpx.post(
        f"{base_url}/v1/evaluate/policy",
        content=oversized_body,
        headers={"Content-Type": "application/json"},
        timeout=10.0,
    )
    assert large_res.status_code == 413
    print(f"[Payload Limit] Intercepted oversized body: HTTP {large_res.status_code}")

    # 5. Load Test: 10 concurrent requests
    t_load_start = time.perf_counter()
    latencies = []
    for _ in range(10):
        t_req = time.perf_counter()
        r = httpx.post(f"{base_url}/v1/evaluate/comparison", json=req_body, timeout=5.0)
        assert r.status_code == 200
        latencies.append((time.perf_counter() - t_req) * 1000.0)
    total_load_time = time.perf_counter() - t_load_start

    mean_lat = round(float(np.mean(latencies)), 2)
    print(f"[Load Test] 10 requests finished in {round(total_load_time, 2)}s (mean: {mean_lat}ms)")

    # 6. Metrics Check
    m_res = httpx.get(f"{base_url}/metrics").json()
    assert "/health:200" in m_res["requests_total"]
    assert "/v1/evaluate/comparison:200" in m_res["requests_total"]
    reqs = m_res["requests_total"]
    rss = m_res.get("memory_rss_mb")
    print(f"[Metrics] Recorded total requests: {reqs}, Container RSS: {rss} MB")

    return {
        "status": "PASSED",
        "container_rss_mb": rss,
        "load_latencies_ms": latencies,
        "mean_latency_ms": mean_lat,
    }


def teardown() -> None:
    subprocess.run(
        ["docker", "stop", CONTAINER_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["docker", "rm", CONTAINER_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("[Container] Stopped and removed cleanly.")


if __name__ == "__main__":
    try:
        start_container()
        wait_for_ready()
        verify_endpoints_and_parity()
    finally:
        teardown()
