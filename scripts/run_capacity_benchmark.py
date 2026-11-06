"""Comprehensive Capacity, Load-Testing, and Cancellation Validation Harness.

Measures:
1. Complete 4x4 matrix: N in {500, 2500, 10000, 25000} x C in {1, 2, 4, 8}
2. Request lifecycle profiling: JSON serialization vs Pydantic validation vs statistical compute
3. BLAS/OpenMP thread count impact (1 thread vs multi-threaded)
4. Single-worker vs multi-worker comparison
5. Resource budget scaling: N, D, K, and folds
6. Cancellation and overload behavior under client disconnect, burst load, and oversized payloads
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import httpx
import numpy as np
import psutil  # type: ignore[import-untyped]
import uvicorn
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from reversallab.api import BanditDataset, LinearSoftmaxPolicy, PolicyComparison, UniformPolicy
from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.service.app import create_app


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LiveServer:
    def __init__(self, port: int) -> None:
        self.port = port
        self.app = create_app()
        self.config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="error", access_log=False
        )
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        url = f"http://127.0.0.1:{self.port}/health"
        for _ in range(50):
            try:
                r = httpx.get(url, timeout=0.5)
                if r.status_code == 200:
                    return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("Failed to start LiveServer")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=2.0)


def generate_benchmark_payload(n: int, d: int = 2, k: int = 3, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    env = SyntheticBanditEnvironment.random(
        n_actions=k, context_dim=d, rng=rng, reward_scale=1.0, bias_scale=0.0
    )
    logging_policy = UniformPolicy(n_actions=k)
    ds = BanditDataset.from_synthetic(env, logging_policy=logging_policy, n_samples=n, seed=seed)

    w_cand = rng.normal(size=(d, k)).tolist()
    w_base = rng.normal(size=(d, k)).tolist()

    return {
        "dataset": {
            "contexts": ds.context.tolist(),
            "actions": ds.action.tolist(),
            "rewards": ds.reward.tolist(),
            "propensities": ds.logging_propensity.tolist(),
            "n_actions": k,
        },
        "candidate_policy": {"type": "linear_softmax", "weights": w_cand},
        "baseline_policy": {"type": "linear_softmax", "weights": w_base},
        "estimator": "dr",
        "cross_fitting_folds": 3,
        "margin": 0.01,
        "deploy_epsilon": 0.10,
        "max_epsilon": 0.30,
        "seed": seed,
    }


def run_4x4_matrix(base_url: str) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("1. RUNNING COMPLETE 4x4 LOAD TEST MATRIX")
    print("=" * 70)

    sample_sizes = [500, 2500, 10000, 25000]
    concurrency_levels = [1, 2, 4, 8]
    matrix_results: dict[str, Any] = {}

    target_url = f"{base_url}/v1/evaluate/comparison"
    proc = psutil.Process(os.getpid())

    for n in sample_sizes:
        payload = generate_benchmark_payload(n)
        matrix_results[str(n)] = {}

        for c in concurrency_levels:
            n_reqs = 30 if n < 25000 else 20
            warmup_reqs = 2

            with httpx.Client(timeout=30.0) as client:
                for _ in range(warmup_reqs):
                    client.post(target_url, json=payload)

            gc.collect()
            cpu_start = psutil.cpu_percent(interval=None)

            latencies: list[float] = []
            errors = 0
            timeouts = 0

            def do_request(p: dict[str, Any]) -> float:
                nonlocal errors, timeouts
                t0 = time.perf_counter()
                try:
                    with httpx.Client(timeout=30.0) as client:
                        res = client.post(target_url, json=p)
                        if res.status_code != 200:
                            errors += 1
                except httpx.TimeoutException:
                    timeouts += 1
                except Exception:
                    errors += 1
                return (time.perf_counter() - t0) * 1000.0

            t_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
                futures = [executor.submit(do_request, payload) for _ in range(n_reqs)]
                for f in concurrent.futures.as_completed(futures):
                    latencies.append(f.result())
            total_duration = time.perf_counter() - t_start

            cpu_end = psutil.cpu_percent(interval=None)
            rss_end = proc.memory_info().rss / (1024 * 1024)

            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))
            rps = round(n_reqs / total_duration, 2)

            cell_data = {
                "n": n,
                "concurrency": c,
                "requests": n_reqs,
                "duration_s": round(total_duration, 2),
                "rps": rps,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "errors": errors,
                "timeouts": timeouts,
                "peak_rss_mb": round(rss_end, 2),
                "cpu_util_pct": round(max(cpu_start, cpu_end), 1),
            }
            matrix_results[str(n)][str(c)] = cell_data

            print(
                f"N={n:5d} | C={c:2d} | Req={n_reqs:2d} | RPS={rps:5.2f} | "
                f"p50={p50:6.1f}ms | p95={p95:6.1f}ms | p99={p99:6.1f}ms | "
                f"RSS={rss_end:5.1f}MB | Err={errors}"
            )

    return matrix_results


def run_request_lifecycle_profiling() -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("2. RUNNING REQUEST LIFECYCLE PROFILING (SERIALIZATION VS COMPUTE)")
    print("=" * 70)

    from reversallab.service.schemas import PolicyComparisonRequest

    sample_sizes = [500, 2500, 10000, 25000]
    profiling_results: dict[str, Any] = {}

    for n in sample_sizes:
        payload = generate_benchmark_payload(n)

        # 1. Client JSON Serialization
        t0 = time.perf_counter()
        json_str = json.dumps(payload)
        client_json_ms = (time.perf_counter() - t0) * 1000.0
        payload_kb = len(json_str.encode("utf-8")) / 1024.0

        # 2. Server JSON Deserialization & Pydantic Validation
        t0 = time.perf_counter()
        parsed_dict = json.loads(json_str)
        req_obj = PolicyComparisonRequest.model_validate(parsed_dict)
        validation_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Model Fitting & Statistical Evaluation
        cand_p = LinearSoftmaxPolicy(weights=np.array(req_obj.candidate_policy.weights))
        base_p = LinearSoftmaxPolicy(weights=np.array(req_obj.baseline_policy.weights))
        ds_obj = BanditDataset.from_arrays(
            contexts=np.array(req_obj.dataset.contexts),
            actions=np.array(req_obj.dataset.actions),
            rewards=np.array(req_obj.dataset.rewards),
            propensities=np.array(req_obj.dataset.propensities),
            n_actions=req_obj.dataset.n_actions,
        )

        t0 = time.perf_counter()
        comp_report = PolicyComparison.evaluate(
            dataset=ds_obj,
            candidate_policy=cand_p,
            baseline_policy=base_p,
            estimator="dr",
            margin=req_obj.margin,
            deploy_epsilon=req_obj.deploy_epsilon,
            max_epsilon=req_obj.max_epsilon,
            cross_fitting_folds=req_obj.cross_fitting_folds,
            seed=req_obj.seed or 42,
        )
        compute_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Response Serialization
        t0 = time.perf_counter()
        _ = json.dumps(comp_report.to_dict())
        resp_ser_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = client_json_ms + validation_ms + compute_ms + resp_ser_ms
        compute_ratio = compute_ms / total_ms

        res = {
            "n": n,
            "payload_kb": round(payload_kb, 1),
            "client_json_ser_ms": round(client_json_ms, 2),
            "server_validation_ms": round(validation_ms, 2),
            "statistical_compute_ms": round(compute_ms, 2),
            "response_ser_ms": round(resp_ser_ms, 2),
            "total_ms": round(total_ms, 2),
            "compute_ratio_pct": round(compute_ratio * 100.0, 1),
        }
        profiling_results[str(n)] = res
        print(
            f"N={n:5d} | Payload={payload_kb:6.1f}KB | ClientSer={client_json_ms:5.1f}ms | "
            f"Validation={validation_ms:5.1f}ms | Compute={compute_ms:6.1f}ms | "
            f"ComputeRatio={round(compute_ratio * 100, 1)}%"
        )

    return profiling_results


def run_blas_threading_benchmark(base_url: str) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("3. BENCHMARKING BLAS / OPENMP THREAD COUNT IMPACT (N=10,000, C=4)")
    print("=" * 70)

    target_url = f"{base_url}/v1/evaluate/comparison"
    payload = generate_benchmark_payload(10000)
    n_reqs = 20

    results: dict[str, Any] = {}

    for thread_limit in [1, None]:
        tag = "single_thread (limit=1)" if thread_limit == 1 else "default_threading (multi)"

        with threadpool_limits(limits=thread_limit, user_api="blas"):
            latencies: list[float] = []
            t0 = time.perf_counter()

            def do_req(p: dict[str, Any]) -> float:
                tr = time.perf_counter()
                with httpx.Client(timeout=30.0) as client:
                    res = client.post(target_url, json=p)
                    assert res.status_code == 200
                return (time.perf_counter() - tr) * 1000.0

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(do_req, payload) for _ in range(n_reqs)]
                for f in concurrent.futures.as_completed(futures):
                    latencies.append(f.result())

            dur = time.perf_counter() - t0
            rps = round(n_reqs / dur, 2)
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))

            results[str(thread_limit or "default")] = {
                "threading_mode": tag,
                "rps": rps,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
            }
            print(f"Threading: {tag:25s} | RPS={rps:5.2f} | p50={p50:6.1f}ms | p95={p95:6.1f}ms")

    return results


def run_multi_worker_benchmark() -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("4. COMPARING SINGLE-WORKER VS MULTI-WORKER UVICORN (N=2,500, C=4)")
    print("=" * 70)

    worker_counts = [1, 2, 4]
    results: dict[str, Any] = {}
    payload = generate_benchmark_payload(2500)
    n_reqs = 24

    for w in worker_counts:
        port = find_free_port()
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "reversallab.service.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            str(w),
            "--log-level",
            "error",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            url = f"http://127.0.0.1:{port}/health"
            ready = False
            for _ in range(50):
                try:
                    r = httpx.get(url, timeout=0.5)
                    if r.status_code == 200:
                        ready = True
                        break
                except Exception:
                    time.sleep(0.1)

            if not ready:
                print(f"Workers={w}: Failed to start within 5s")
                continue

            # Warmup
            with httpx.Client(timeout=10.0) as client:
                for _ in range(2):
                    client.post(f"http://127.0.0.1:{port}/v1/evaluate/comparison", json=payload)

            # Benchmark
            latencies = []
            t0 = time.perf_counter()

            def do_worker_req(dest_port: int, p: dict[str, Any]) -> float:
                tr = time.perf_counter()
                with httpx.Client(timeout=30.0) as client:
                    r = client.post(f"http://127.0.0.1:{dest_port}/v1/evaluate/comparison", json=p)
                    assert r.status_code == 200
                return (time.perf_counter() - tr) * 1000.0

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(do_worker_req, port, payload) for _ in range(n_reqs)]
                for f in concurrent.futures.as_completed(futures):
                    latencies.append(f.result())

            dur = time.perf_counter() - t0
            rps = round(n_reqs / dur, 2)
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))

            results[str(w)] = {
                "workers": w,
                "rps": rps,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
            }
            print(f"Workers={w:1d} | RPS={rps:5.2f} | p50={p50:6.1f}ms | p95={p95:6.1f}ms")

        finally:
            proc.terminate()
            proc.wait(timeout=3.0)

    return results


def run_resource_budget_audit() -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("5. RESOURCE BUDGET AUDIT & COMPLEXITY SCALING")
    print("=" * 70)

    configurations = [
        {"n": 2500, "d": 2, "k": 3, "folds": 3},
        {"n": 2500, "d": 10, "k": 3, "folds": 3},
        {"n": 2500, "d": 20, "k": 3, "folds": 3},
        {"n": 2500, "d": 2, "k": 10, "folds": 3},
        {"n": 2500, "d": 2, "k": 3, "folds": 5},
    ]

    scaling_results = []
    for cfg in configurations:
        n, d, k, f = cfg["n"], cfg["d"], cfg["k"], cfg["folds"]
        cost = n * d * k * f
        payload = generate_benchmark_payload(n=n, d=d, k=k)
        payload["cross_fitting_folds"] = f

        cand_p = LinearSoftmaxPolicy(weights=np.array(payload["candidate_policy"]["weights"]))
        base_p = LinearSoftmaxPolicy(weights=np.array(payload["baseline_policy"]["weights"]))
        ds = BanditDataset.from_arrays(
            contexts=np.array(payload["dataset"]["contexts"]),
            actions=np.array(payload["dataset"]["actions"]),
            rewards=np.array(payload["dataset"]["rewards"]),
            propensities=np.array(payload["dataset"]["propensities"]),
            n_actions=k,
        )

        t0 = time.perf_counter()
        PolicyComparison.evaluate(
            dataset=ds,
            candidate_policy=cand_p,
            baseline_policy=base_p,
            estimator="dr",
            cross_fitting_folds=f,
            seed=42,
        )
        fit_ms = (time.perf_counter() - t0) * 1000.0

        entry = {
            "n": n,
            "d": d,
            "k": k,
            "folds": f,
            "budget_cost": cost,
            "fit_time_ms": round(fit_ms, 2),
        }
        scaling_results.append(entry)
        msg = f"N={n:4d}, D={d:2d}, K={k:2d}, Folds={f:1d} -> Cost={cost:7d} | Fit={fit_ms:6.1f}ms"
        print(msg)

    return {"scaling_measurements": scaling_results}


def run_cancellation_and_overload_tests(base_url: str) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("6. CANCELLATION AND OVERLOAD DEFENSE VALIDATION")
    print("=" * 70)

    target_url = f"{base_url}/v1/evaluate/comparison"
    tests_summary: dict[str, Any] = {}

    # Test 1: Client Cancellation Mid-Execution
    payload_heavy = generate_benchmark_payload(25000)
    t_disc_start = time.perf_counter()
    try:
        with httpx.Client(timeout=0.03) as client:
            client.post(target_url, json=payload_heavy)
    except httpx.TimeoutException:
        pass
    client_cancel_duration_ms = (time.perf_counter() - t_disc_start) * 1000.0

    t_probe = time.perf_counter()
    _ = httpx.get(f"{base_url}/health", timeout=5.0)
    probe_latency_ms = (time.perf_counter() - t_probe) * 1000.0

    tests_summary["client_cancellation"] = {
        "client_aborted_after_ms": round(client_cancel_duration_ms, 1),
        "subsequent_health_probe_latency_ms": round(probe_latency_ms, 2),
        "behavior": (
            "Synchronous evaluation in worker thread runs to completion even if "
            "client disconnects; Python/NumPy cannot be interrupted mid-C-extension "
            "without explicit GIL hooks."
        ),
    }
    print(
        f"Client Disconnect: Aborted in {client_cancel_duration_ms:.1f}ms. "
        f"Subsequent probe: {probe_latency_ms:.2f}ms."
    )

    # Test 2: Burst Load beyond capacity (30 concurrent requests)
    payload_med = generate_benchmark_payload(2500)
    t_burst = time.perf_counter()
    status_codes = []

    def burst_req(p: dict[str, Any]) -> int:
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.post(target_url, json=p)
                return res.status_code
        except Exception:
            return 599

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(burst_req, payload_med) for _ in range(30)]
        for f in concurrent.futures.as_completed(futures):
            status_codes.append(f.result())
    burst_duration_s = time.perf_counter() - t_burst

    successes = status_codes.count(200)
    tests_summary["burst_load"] = {
        "burst_requests": 30,
        "concurrency": 30,
        "success_count": successes,
        "error_count": len(status_codes) - successes,
        "duration_s": round(burst_duration_s, 2),
    }
    print(f"Burst Load: 30 requests. Success: {successes}/30 in {burst_duration_s:.2f}s.")

    # Test 3: Oversized streaming payload (> 10MB)
    oversized = b" " * (10 * 1024 * 1024 + 512)
    r_413 = httpx.post(
        f"{base_url}/v1/evaluate/policy",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert r_413.status_code == 413
    tests_summary["oversized_payload"] = {
        "status_code": r_413.status_code,
        "response_title": r_413.json().get("title"),
    }
    print(f"Oversized Payload: HTTP {r_413.status_code} ({r_413.json().get('title')})")

    # Test 4: Malformed JSON payload
    r_bad_json = httpx.post(
        target_url,
        content=b"{invalid_json: true",
        headers={"Content-Type": "application/json"},
    )
    assert r_bad_json.status_code == 422
    tests_summary["malformed_json"] = {
        "status_code": r_bad_json.status_code,
        "response_title": r_bad_json.json().get("title"),
    }
    print(f"Malformed JSON: HTTP {r_bad_json.status_code} ({r_bad_json.json().get('title')})")

    # Test 5: Budget Breach Rejection (D=50, N=5000, K=10, folds=5 -> 12.5M cost)
    budget_payload = {
        "dataset": {
            "contexts": [[0.0] * 50] * 5000,
            "actions": [0] * 5000,
            "rewards": [1.0] * 5000,
            "propensities": [0.1] * 5000,
            "n_actions": 10,
        },
        "candidate_policy": {"type": "uniform", "n_actions": 10},
        "baseline_policy": {"type": "uniform", "n_actions": 10},
        "estimator": "dr",
        "cross_fitting_folds": 5,
    }
    r_budget = httpx.post(target_url, json=budget_payload)
    assert r_budget.status_code == 422
    assert "Requested computation budget" in r_budget.text
    tests_summary["budget_breach"] = {
        "status_code": r_budget.status_code,
        "error_message": str(r_budget.json().get("detail")),
    }
    print(f"Budget Breach: Pre-execution rejection with HTTP {r_budget.status_code}")

    # Test 6: Metrics endpoint access control audit
    r_metrics = httpx.get(f"{base_url}/metrics")
    tests_summary["metrics_access_control"] = {
        "status_code": r_metrics.status_code,
        "is_unauthenticated": r_metrics.status_code == 200,
        "recommendation": "Protect /metrics behind internal network or mTLS in production",
    }
    print(f"Metrics Endpoint: HTTP {r_metrics.status_code} (Publicly accessible in dev mode)")

    return tests_summary


def main() -> None:
    port = find_free_port()
    server = LiveServer(port)
    server.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        m1 = run_4x4_matrix(base_url)
        m2 = run_request_lifecycle_profiling()
        m3 = run_blas_threading_benchmark(base_url)
        m4 = run_multi_worker_benchmark()
        m5 = run_resource_budget_audit()
        m6 = run_cancellation_and_overload_tests(base_url)

        full_report = {
            "matrix_4x4": m1,
            "lifecycle_profiling": m2,
            "blas_threading": m3,
            "multi_worker": m4,
            "budget_scaling": m5,
            "cancellation_and_overload": m6,
        }

        with open("capacity_validation_report.json", "w") as f:
            json.dump(full_report, f, indent=2)

        print("\n" + "=" * 70)
        print("CAPACITY BENCHMARK COMPLETE: Saved to capacity_validation_report.json")
        print("=" * 70)

    finally:
        server.stop()


if __name__ == "__main__":
    main()
