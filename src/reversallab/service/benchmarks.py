"""Empirical latency, memory, and concurrency benchmarks for ReversalLab FastAPI service."""

from __future__ import annotations

import concurrent.futures
import gc
import os
import platform
import socket
import sys
import threading
import time
from typing import Any

import httpx
import numpy as np
import psutil  # type: ignore[import-untyped]
import uvicorn
from starlette.testclient import TestClient

from reversallab import __version__
from reversallab.api import BanditDataset, UniformPolicy
from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.service.app import create_app


def get_current_memory_mb() -> float:
    """Return process resident set size (RSS) in megabytes."""
    gc.collect()
    process = psutil.Process(os.getpid())
    return float(round(process.memory_info().rss / (1024 * 1024), 2))


def get_environment_metadata() -> dict[str, Any]:
    """Capture hardware and software runtime environment metadata."""
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": sys.version.split()[0],
        "reversallab_version": __version__,
        "numpy_version": np.__version__,
        "fastapi_version": uvicorn.__version__,
    }


def find_free_port() -> int:
    """Find a free local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class UvicornServerThread:
    """Runs a live Uvicorn HTTP server in a daemon background thread."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.app = create_app()
        self.config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="warning", access_log=False
        )
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        # Wait for server to become responsive
        url = f"http://{self.host}:{self.port}/health"
        for _ in range(50):
            try:
                res = httpx.get(url, timeout=1.0)
                if res.status_code == 200:
                    return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("Failed to start Uvicorn benchmark server.")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=3.0)


def run_comprehensive_benchmarks() -> dict[str, Any]:
    """Execute complete empirical benchmarks across sample sizes and concurrency levels."""
    env_meta = get_environment_metadata()
    in_process_results: dict[str, Any] = {}
    http_concurrency_results: dict[str, Any] = {}

    rng = np.random.default_rng(42)
    env = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=rng, reward_scale=1.0, bias_scale=0.0
    )
    logging_policy = UniformPolicy(n_actions=3)

    w_cand = rng.normal(size=(2, 3)).tolist()
    w_base = rng.normal(size=(2, 3)).tolist()

    app = create_app()
    in_proc_client = TestClient(app)

    # 1. In-Process Warmup & Latency Tests across sample sizes
    sample_sizes = [500, 2500, 10000, 25000]

    # Warmup
    for _ in range(3):
        ds_warm = BanditDataset.from_synthetic(
            env, logging_policy=logging_policy, n_samples=200, seed=1
        )
        in_proc_client.post(
            "/v1/evaluate/policy",
            json={
                "dataset": {
                    "contexts": ds_warm.context.tolist(),
                    "actions": ds_warm.action.tolist(),
                    "rewards": ds_warm.reward.tolist(),
                    "propensities": ds_warm.logging_propensity.tolist(),
                    "n_actions": 3,
                },
                "policy": {"type": "linear_softmax", "weights": w_cand},
                "estimator": "dr",
            },
        )

    for n in sample_sizes:
        ds = BanditDataset.from_synthetic(env, logging_policy=logging_policy, n_samples=n, seed=42)
        payload = {
            "contexts": ds.context.tolist(),
            "actions": ds.action.tolist(),
            "rewards": ds.reward.tolist(),
            "propensities": ds.logging_propensity.tolist(),
            "n_actions": 3,
        }

        n_res: dict[str, Any] = {}

        for est, folds in [("ips", 3), ("snips", 3), ("dr", 3), ("dr", 5)]:
            config_key = f"{est}_k{folds}" if "dr" in est else est
            latencies = []
            for _ in range(3):
                t0 = time.perf_counter()
                res = in_proc_client.post(
                    "/v1/evaluate/policy",
                    json={
                        "dataset": payload,
                        "policy": {"type": "linear_softmax", "weights": w_cand},
                        "estimator": est,
                        "cross_fitting_folds": folds,
                    },
                )
                latencies.append((time.perf_counter() - t0) * 1000.0)
                assert res.status_code == 200

            n_res[config_key] = {
                "mean_ms": round(float(np.mean(latencies)), 2),
                "std_ms": round(float(np.std(latencies)), 2),
                "min_ms": round(float(np.min(latencies)), 2),
                "max_ms": round(float(np.max(latencies)), 2),
            }

        # Paired comparison & reversal search
        comp_latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            res = in_proc_client.post(
                "/v1/evaluate/comparison",
                json={
                    "dataset": payload,
                    "candidate_policy": {"type": "linear_softmax", "weights": w_cand},
                    "baseline_policy": {"type": "linear_softmax", "weights": w_base},
                    "estimator": "dr",
                    "cross_fitting_folds": 3,
                    "margin": 0.02,
                    "deploy_epsilon": 0.10,
                    "max_epsilon": 0.30,
                },
            )
            comp_latencies.append((time.perf_counter() - t0) * 1000.0)
            assert res.status_code == 200

        n_res["paired_comparison"] = {
            "mean_ms": round(float(np.mean(comp_latencies)), 2),
            "std_ms": round(float(np.std(comp_latencies)), 2),
            "min_ms": round(float(np.min(comp_latencies)), 2),
            "max_ms": round(float(np.max(comp_latencies)), 2),
            "decision": res.json()["report"]["decision"]["outcome"],
        }
        in_process_results[str(n)] = n_res

    # 2. Live HTTP Server Concurrency Benchmark over TCP
    port = find_free_port()
    server = UvicornServerThread("127.0.0.1", port)
    server.start()

    base_url = f"http://127.0.0.1:{port}"
    target_url = f"{base_url}/v1/evaluate/comparison"

    ds_http = BanditDataset.from_synthetic(
        env, logging_policy=logging_policy, n_samples=2500, seed=42
    )
    http_payload = {
        "dataset": {
            "contexts": ds_http.context.tolist(),
            "actions": ds_http.action.tolist(),
            "rewards": ds_http.reward.tolist(),
            "propensities": ds_http.logging_propensity.tolist(),
            "n_actions": 3,
        },
        "candidate_policy": {"type": "linear_softmax", "weights": w_cand},
        "baseline_policy": {"type": "linear_softmax", "weights": w_base},
        "estimator": "dr",
        "cross_fitting_folds": 3,
    }

    concurrency_levels = [1, 2, 4, 8]
    n_total_requests = 20

    try:
        for c in concurrency_levels:
            latencies_http: list[float] = []
            failures = 0
            timeouts = 0

            mem_start = get_current_memory_mb()
            psutil.cpu_percent(interval=None)  # reset cpu counter
            wall_start = time.perf_counter()

            def _send_request() -> float:
                nonlocal failures, timeouts
                t_req = time.perf_counter()
                try:
                    with httpx.Client(timeout=10.0) as http_client:
                        resp = http_client.post(target_url, json=http_payload)
                        if resp.status_code == 200:
                            return (time.perf_counter() - t_req) * 1000.0
                        failures += 1
                        return -1.0
                except httpx.TimeoutException:
                    timeouts += 1
                    return -1.0
                except Exception:
                    failures += 1
                    return -1.0

            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
                futures = [executor.submit(_send_request) for _ in range(n_total_requests)]
                for fut in concurrent.futures.as_completed(futures):
                    dur = fut.result()
                    if dur > 0:
                        latencies_http.append(dur)

            wall_total_s = time.perf_counter() - wall_start
            cpu_usage = psutil.cpu_percent(interval=None)
            mem_peak = get_current_memory_mb()

            rps = round(len(latencies_http) / wall_total_s, 2) if wall_total_s > 0 else 0.0

            http_concurrency_results[f"concurrency_{c}"] = {
                "concurrency": c,
                "total_requests": n_total_requests,
                "successful_requests": len(latencies_http),
                "failures": failures,
                "timeouts": timeouts,
                "throughput_rps": rps,
                "latency_p50_ms": round(float(np.percentile(latencies_http, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies_http, 95)), 2),
                "latency_p99_ms": round(float(np.percentile(latencies_http, 99)), 2),
                "latency_mean_ms": round(float(np.mean(latencies_http)), 2),
                "latency_min_ms": round(float(np.min(latencies_http)), 2),
                "latency_max_ms": round(float(np.max(latencies_http)), 2),
                "cpu_utilization_pct": cpu_usage,
                "rss_start_mb": mem_start,
                "rss_peak_mb": mem_peak,
            }
    finally:
        server.stop()

    return {
        "environment": env_meta,
        "in_process_benchmarks": in_process_results,
        "live_http_concurrency_benchmarks": http_concurrency_results,
        "final_rss_mb": get_current_memory_mb(),
    }


if __name__ == "__main__":
    import json

    res = run_comprehensive_benchmarks()
    print(json.dumps(res, indent=2))
