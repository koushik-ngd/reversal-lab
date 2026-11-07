"""Operational stress benchmark for ReversalLab background job engine (M6C Phase 7)."""

from __future__ import annotations

import concurrent.futures
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]
from starlette.testclient import TestClient

from reversallab.api import BanditDataset, LinearSoftmaxPolicy, PolicyComparison, UniformPolicy
from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.service.app import create_app


def generate_payload(n: int, d: int = 4, k: int = 3, seed: int = 42) -> dict[str, Any]:
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


def get_tree_rss_mb() -> float:
    try:
        proc = psutil.Process(os.getpid())
        rss = float(proc.memory_info().rss)
        for child in proc.children(recursive=True):
            with child.oneshot():
                rss += float(child.memory_info().rss)
        return float(round(rss / (1024 * 1024), 2))
    except Exception:
        return 0.0


def measure_dir_size_kb(dir_path: Path) -> float:
    total = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return float(round(total / 1024.0, 2))


def run_operational_benchmark() -> dict[str, Any]:
    print("=" * 75)
    print("REVERSALLAB OPERATIONAL ENGINE BENCHMARK (M6C Phase 7)")
    print("=" * 75)

    report: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        app = create_app(storage_dir=tmp_dir, enable_supervisor=True)

        with TestClient(app) as client:
            # 1. Distinguish pure Python evaluation time from worker process startup
            print("\n[Stage 1] Measuring isolated statistical evaluation vs subprocess overhead...")
            pure_rng = np.random.default_rng(42)
            pure_ds = BanditDataset.from_synthetic(
                SyntheticBanditEnvironment.random(
                    n_actions=3, context_dim=4, rng=pure_rng, reward_scale=1.0, bias_scale=0.0
                ),
                UniformPolicy(3),
                n_samples=2500,
                seed=42,
            )
            t_math_0 = time.perf_counter()
            PolicyComparison.evaluate(
                dataset=pure_ds,
                candidate_policy=LinearSoftmaxPolicy(np.zeros((4, 3))),
                baseline_policy=LinearSoftmaxPolicy(np.zeros((4, 3))),
                estimator="dr",
                cross_fitting_folds=3,
                seed=42,
            )
            pure_math_duration_ms = (time.perf_counter() - t_math_0) * 1000.0
            print(f"  -> Pure in-process PolicyComparison.evaluate: {pure_math_duration_ms:.2f} ms")

            # 2. Concurrent submission burst (8 submissions at once)
            print("\n[Stage 2] Concurrent submission burst (8 concurrent distinct payloads)...")
            payloads = [generate_payload(n=1000, seed=100 + i) for i in range(8)]
            submission_latencies_ms: list[float] = []

            def submit_single(payload: dict[str, Any]) -> tuple[str, float]:
                t0 = time.perf_counter()
                r = client.post("/v1/jobs/comparison", json=payload)
                lat = (time.perf_counter() - t0) * 1000.0
                assert r.status_code == 202
                return r.json()["job_id"], lat

            t_burst_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(submit_single, payloads))

            burst_duration_ms = (time.perf_counter() - t_burst_start) * 1000.0
            job_ids = [res[0] for res in results]
            submission_latencies_ms = [res[1] for res in results]

            sub_p50 = float(np.percentile(submission_latencies_ms, 50))
            sub_p95 = float(np.percentile(submission_latencies_ms, 95))
            print(f"  -> Burst of 8 submissions completed in {burst_duration_ms:.2f} ms")
            print(f"  -> Submission latency p50: {sub_p50:.2f} ms, p95: {sub_p95:.2f} ms")

            # 3. Queue wait times and execution times under 2-worker concurrency
            print("\n[Stage 3] Processing queue under 2-worker concurrency...")
            queue_wait_times_ms: list[float] = []
            exec_times_ms: list[float] = []
            peak_rss_mb = get_tree_rss_mb()

            completed_count = 0
            t_poll_start = time.time()
            while time.time() - t_poll_start < 45.0:
                current_rss = get_tree_rss_mb()
                if current_rss > peak_rss_mb:
                    peak_rss_mb = current_rss

                all_done = True
                for jid in job_ids:
                    r = client.get(f"/v1/jobs/{jid}")
                    st = r.json()
                    if st["status"] not in ("COMPLETED", "FAILED", "CANCELED"):
                        all_done = False
                if all_done:
                    break
                time.sleep(0.1)

            for jid in job_ids:
                r = client.get(f"/v1/jobs/{jid}")
                st = r.json()
                assert st["status"] == "COMPLETED"
                completed_count += 1

                # Calculate queue wait and execution time from ISO timestamps
                t_created = time.mktime(time.strptime(st["created_at"][:19], "%Y-%m-%dT%H:%M:%S"))
                t_started = time.mktime(time.strptime(st["started_at"][:19], "%Y-%m-%dT%H:%M:%S"))
                t_completed = time.mktime(
                    time.strptime(st["completed_at"][:19], "%Y-%m-%dT%H:%M:%S")
                )

                queue_wait_times_ms.append((t_started - t_created) * 1000.0)
                exec_times_ms.append((t_completed - t_started) * 1000.0)

            queue_p50 = float(np.percentile(queue_wait_times_ms, 50))
            queue_p95 = float(np.percentile(queue_wait_times_ms, 95))
            exec_p50 = float(np.percentile(exec_times_ms, 50))
            exec_p95 = float(np.percentile(exec_times_ms, 95))

            print(f"  -> All {completed_count} jobs completed successfully.")
            print(f"  -> Queue wait time p50: {queue_p50:.2f} ms, p95: {queue_p95:.2f} ms")
            print(f"  -> Execution time p50: {exec_p50:.2f} ms, p95: {exec_p95:.2f} ms")
            print(f"  -> Subprocess startup baseline: ~{exec_p50 - pure_math_duration_ms:.2f} ms")

            # 4. Cancellation confirmation latency
            print("\n[Stage 4] Measuring cancellation confirmation latency...")
            cancel_latencies_ms: list[float] = []
            canceled_count = 0

            for i in range(5):
                # Submit heavy job
                heavy_p = generate_payload(n=2000, seed=500 + i)
                heavy_p["cross_fitting_folds"] = 10
                r_sub = client.post("/v1/jobs/comparison", json=heavy_p)
                c_jid = r_sub.json()["job_id"]

                # Wait until job is RUNNING
                t_w = time.time()
                while time.time() - t_w < 5.0:
                    if client.get(f"/v1/jobs/{c_jid}").json()["status"] == "RUNNING":
                        break
                    time.sleep(0.05)

                # Time cancellation request
                t_c0 = time.perf_counter()
                r_del = client.delete(f"/v1/jobs/{c_jid}")
                c_lat = (time.perf_counter() - t_c0) * 1000.0
                assert r_del.status_code == 200
                assert r_del.json()["status"] == "CANCELED"
                cancel_latencies_ms.append(c_lat)
                canceled_count += 1

            cancel_p50 = float(np.percentile(cancel_latencies_ms, 50))
            cancel_p95 = float(np.percentile(cancel_latencies_ms, 95))
            print(f"  -> Canceled {canceled_count} running jobs cleanly.")
            print(f"  -> Cancellation latency p50: {cancel_p50:.2f} ms, p95: {cancel_p95:.2f} ms")

            # 5. Disk storage measurement
            disk_kb = measure_dir_size_kb(tmp_dir)
            print(f"\n[Stage 5] Storage footprints: {disk_kb:.2f} KB across reports and db.")
            print(f"  -> Peak process tree RSS: {peak_rss_mb:.2f} MB")

            report = {
                "pure_math_eval_ms": round(pure_math_duration_ms, 2),
                "submission_p50_ms": round(sub_p50, 2),
                "submission_p95_ms": round(sub_p95, 2),
                "queue_wait_p50_ms": round(queue_p50, 2),
                "queue_wait_p95_ms": round(queue_p95, 2),
                "execution_p50_ms": round(exec_p50, 2),
                "execution_p95_ms": round(exec_p95, 2),
                "subprocess_startup_overhead_ms": round(exec_p50 - pure_math_duration_ms, 2),
                "cancellation_confirmation_p50_ms": round(cancel_p50, 2),
                "cancellation_confirmation_p95_ms": round(cancel_p95, 2),
                "peak_tree_rss_mb": round(peak_rss_mb, 2),
                "storage_footprint_kb": round(disk_kb, 2),
                "counts": {
                    "completed": completed_count,
                    "failed": 0,
                    "canceled": canceled_count,
                },
            }

    with open("operational_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nBenchmark completed. Report written to operational_benchmark_report.json")
    return report


if __name__ == "__main__":
    run_operational_benchmark()
