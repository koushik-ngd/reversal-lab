"""Empirical performance benchmark harness for ReversalLab asynchronous job execution."""

from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]
from starlette.testclient import TestClient

from reversallab.api import BanditDataset, UniformPolicy
from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.service.app import create_app


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


def run_job_benchmarks() -> dict[str, Any]:
    print("=" * 70)
    print("REVERSALLAB ASYNCHRONOUS JOB ENGINE CAPACITY & LATENCY BENCHMARK")
    print("=" * 70)

    sample_sizes = [500, 2500, 10000, 25000]
    bench_results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir)
        app = create_app(storage_dir=storage_path, enable_supervisor=True)

        with TestClient(app) as client:
            proc = psutil.Process(os.getpid())

            for n in sample_sizes:
                payload = generate_benchmark_payload(n)
                sub_latencies: list[float] = []
                poll_latencies: list[float] = []
                queue_durations: list[float] = []
                total_turnarounds: list[float] = []

                n_iterations = 10 if n <= 2500 else 5

                for i in range(n_iterations):
                    p = dict(payload)
                    p["seed"] = 42 + i

                    # 1. Measure Submission Latency (POST 202)
                    t0 = time.perf_counter()
                    res = client.post("/v1/jobs/comparison", json=p)
                    t_sub = (time.perf_counter() - t0) * 1000.0
                    assert res.status_code == 202
                    job_id = res.json()["job_id"]
                    sub_latencies.append(t_sub)

                    # 2. Poll until completion
                    t_poll_start = time.perf_counter()
                    while True:
                        t_p0 = time.perf_counter()
                        r = client.get(f"/v1/jobs/{job_id}")
                        t_p = (time.perf_counter() - t_p0) * 1000.0
                        poll_latencies.append(t_p)
                        data = r.json()
                        if data["status"] in ("COMPLETED", "FAILED"):
                            break
                        time.sleep(0.05)

                    total_turnaround_s = time.perf_counter() - t_poll_start
                    total_turnarounds.append(total_turnaround_s * 1000.0)
                    assert data["status"] == "COMPLETED"

                    # Compute queue time vs execution time from timestamps
                    if data.get("started_at") and data.get("created_at"):
                        t_created = time.mktime(
                            time.strptime(data["created_at"][:19], "%Y-%m-%dT%H:%M:%S")
                        )
                        t_started = time.mktime(
                            time.strptime(data["started_at"][:19], "%Y-%m-%dT%H:%M:%S")
                        )
                        queue_durations.append(max(0.0, (t_started - t_created) * 1000.0))

                gc.collect()
                rss_mb = proc.memory_info().rss / (1024 * 1024)

                sub_p50 = float(np.percentile(sub_latencies, 50))
                sub_p95 = float(np.percentile(sub_latencies, 95))
                poll_p50 = float(np.percentile(poll_latencies, 50))
                poll_p95 = float(np.percentile(poll_latencies, 95))
                turnaround_p50 = float(np.percentile(total_turnarounds, 50))

                entry = {
                    "n": n,
                    "iterations": n_iterations,
                    "submission_p50_ms": round(sub_p50, 2),
                    "submission_p95_ms": round(sub_p95, 2),
                    "polling_p50_ms": round(poll_p50, 2),
                    "polling_p95_ms": round(poll_p95, 2),
                    "e2e_turnaround_p50_ms": round(turnaround_p50, 2),
                    "peak_rss_mb": round(rss_mb, 2),
                }
                bench_results[str(n)] = entry

                print(
                    f"N={n:5d} | Sub p50={sub_p50:5.1f}ms | Poll p50={poll_p50:4.1f}ms | "
                    f"Turnaround p50={turnaround_p50:6.1f}ms | RSS={rss_mb:5.1f}MB"
                )

    # Save benchmark report
    with open("job_benchmarks_report.json", "w") as f:
        json.dump(bench_results, f, indent=2)

    return bench_results


if __name__ == "__main__":
    run_job_benchmarks()
