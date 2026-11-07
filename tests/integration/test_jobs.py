"""Integration and fault-injection test suite for ReversalLab asynchronous job system."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from starlette.testclient import TestClient

from reversallab.api import BanditDataset, LinearSoftmaxPolicy, PolicyComparison, UniformPolicy
from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.service.app import create_app
from reversallab.service.job_models import JobRecord, JobStatus
from reversallab.service.job_repository import JobRepository
from reversallab.service.job_storage import JobStorageManager
from reversallab.service.supervisor import JobSupervisor


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    """Generate reproducible sample payload for policy comparison job."""
    rng = np.random.default_rng(42)
    env = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=rng, reward_scale=1.0, bias_scale=0.0
    )
    logging_policy = UniformPolicy(n_actions=3)
    dataset = BanditDataset.from_synthetic(
        env, logging_policy=logging_policy, n_samples=250, seed=7
    )

    w_cand = rng.normal(size=(2, 3)).tolist()
    w_base = rng.normal(size=(2, 3)).tolist()

    return {
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


def wait_for_job_terminal(
    client: TestClient, job_id: str, timeout_s: float = 15.0
) -> dict[str, Any]:
    """Poll job endpoint until terminal state is reached."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        res = client.get(f"/v1/jobs/{job_id}")
        assert res.status_code == 200
        data = res.json()
        if data["status"] in ("COMPLETED", "FAILED", "CANCELED"):
            return cast(dict[str, Any], data)
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not reach terminal state within {timeout_s}s.")


def test_job_submission_lifecycle_and_numerical_parity(
    tmp_path: Path, sample_payload: dict[str, Any]
) -> None:
    """Verify 202 Accepted submission, polling state transitions, and SDK parity."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        # 1. Submission
        res = client.post("/v1/jobs/comparison", json=sample_payload)
        assert res.status_code == 202
        sub_data = res.json()
        assert sub_data["status"] == "PENDING"
        job_id = sub_data["job_id"]
        assert "Location" in res.headers
        assert res.headers["Location"] == f"/v1/jobs/{job_id}"

        # 2. Polling until COMPLETED
        final_data = wait_for_job_terminal(client, job_id)
        assert final_data["status"] == "COMPLETED"
        assert final_data["result"] is not None
        assert final_data["error"] is None

        # 3. SDK Numerical Parity Check
        cand_p = LinearSoftmaxPolicy(
            weights=np.array(sample_payload["candidate_policy"]["weights"])
        )
        base_p = LinearSoftmaxPolicy(weights=np.array(sample_payload["baseline_policy"]["weights"]))
        ds = BanditDataset.from_arrays(
            contexts=np.array(sample_payload["dataset"]["contexts"]),
            actions=np.array(sample_payload["dataset"]["actions"]),
            rewards=np.array(sample_payload["dataset"]["rewards"]),
            propensities=np.array(sample_payload["dataset"]["propensities"]),
            n_actions=3,
        )
        sdk_rep = PolicyComparison.evaluate(
            dataset=ds,
            candidate_policy=cand_p,
            baseline_policy=base_p,
            estimator="dr",
            margin=0.01,
            deploy_epsilon=0.10,
            max_epsilon=0.30,
            cross_fitting_folds=3,
            seed=42,
        )

        job_pt = final_data["result"]["paired_advantage"]["point_estimate"]
        sdk_pt = sdk_rep.paired_advantage["point_estimate"]
        np.testing.assert_allclose(job_pt, sdk_pt, rtol=1e-12)


def test_job_invalid_request_rejected_before_persistence(tmp_path: Path) -> None:
    """Invalid requests must be rejected with 422 without writing to disk."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:
        bad_req = {
            "dataset": {
                "contexts": [[0.1, 0.2]],
                "actions": [0, 1],  # length mismatch
                "rewards": [1.0],
                "propensities": [0.5],
                "n_actions": 2,
            },
            "candidate_policy": {"type": "uniform"},
            "baseline_policy": {"type": "uniform"},
        }
        res = client.post("/v1/jobs/comparison", json=bad_req)
        assert res.status_code == 422

        storage = JobStorageManager(tmp_path)
        assert len(list(storage.inputs_dir.glob("*.json"))) == 0


def test_job_idempotency_duplicate_submission(
    tmp_path: Path, sample_payload: dict[str, Any]
) -> None:
    """Duplicate submissions with identical payload hash return existing job_id."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:
        res1 = client.post("/v1/jobs/comparison", json=sample_payload)
        assert res1.status_code == 202
        job_id1 = res1.json()["job_id"]

        res2 = client.post("/v1/jobs/comparison", json=sample_payload)
        assert res2.status_code == 202
        job_id2 = res2.json()["job_id"]

        assert job_id1 == job_id2

        repo = JobRepository(tmp_path / "jobs.db")
        assert repo.count_pending_jobs() == 1


def test_job_cancellation_pending_state(tmp_path: Path, sample_payload: dict[str, Any]) -> None:
    """Canceling a PENDING job immediately transitions it to CANCELED."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:
        res = client.post("/v1/jobs/comparison", json=sample_payload)
        assert res.status_code == 202
        job_id = res.json()["job_id"]

        del_res = client.delete(f"/v1/jobs/{job_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "CANCELED"

        get_res = client.get(f"/v1/jobs/{job_id}")
        assert get_res.status_code == 200
        assert get_res.json()["status"] == "CANCELED"


def test_job_cannot_cancel_completed_job(tmp_path: Path, sample_payload: dict[str, Any]) -> None:
    """Attempting to cancel a COMPLETED job returns HTTP 409 Conflict."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        res = client.post("/v1/jobs/comparison", json=sample_payload)
        job_id = res.json()["job_id"]

        final_data = wait_for_job_terminal(client, job_id)
        assert final_data["status"] == "COMPLETED"

        del_res = client.delete(f"/v1/jobs/{job_id}")
        assert del_res.status_code == 409
        assert "Cannot cancel job in terminal state" in del_res.json()["detail"]


def test_job_results_survive_api_restart(tmp_path: Path, sample_payload: dict[str, Any]) -> None:
    """Completed job metadata and result file survive application restart."""
    # Run in first application instance
    app1 = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app1) as client1:
        res = client1.post("/v1/jobs/comparison", json=sample_payload)
        job_id = res.json()["job_id"]
        final_data = wait_for_job_terminal(client1, job_id)
        assert final_data["status"] == "COMPLETED"
        report1 = final_data["result"]

    # Start second application instance with new supervisor pointing to same storage
    app2 = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app2) as client2:
        res2 = client2.get(f"/v1/jobs/{job_id}")
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["status"] == "COMPLETED"
        assert data2["result"] == report1


def test_job_crash_recovery_on_startup(tmp_path: Path) -> None:
    """Orphaned RUNNING jobs from a previous crash are recovered to FAILED status."""
    repo = JobRepository(tmp_path / "jobs.db")
    storage = JobStorageManager(tmp_path)

    # Manually insert an orphaned RUNNING job
    job_id = "00000000-0000-4000-8000-000000000001"
    in_path = storage.write_input(job_id, {"dummy": 1})
    out_path = storage.get_result_path(job_id)

    rec = JobRecord(
        job_id=job_id,
        created_at="2026-11-07T00:00:00Z",
        started_at="2026-11-07T00:01:00Z",
        completed_at=None,
        status=JobStatus.RUNNING,
        attempt_count=1,
        payload_hash="dummy_hash",
        input_path=str(in_path),
        result_path=str(out_path),
        worker_pid=999999,
    )
    repo.insert_job(rec)

    # Start app with supervisor: startup recovery must trigger
    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        res = client.get(f"/v1/jobs/{job_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "FAILED"
        assert data["error"]["code"] == "CRASH_OR_RESTART_DETECTED"


def test_job_queue_capacity_limit(tmp_path: Path, sample_payload: dict[str, Any]) -> None:
    """Submissions beyond MAX_QUEUED_JOBS (50) return HTTP 429 Too Many Requests."""
    repo = JobRepository(tmp_path / "jobs.db")
    storage = JobStorageManager(tmp_path)

    # Fill queue with 50 pending jobs
    for i in range(50):
        jid = f"00000000-0000-4000-8000-{i:012d}"
        in_p = storage.write_input(jid, {"i": i})
        out_p = storage.get_result_path(jid)
        repo.insert_job(
            JobRecord(
                job_id=jid,
                created_at=f"2026-11-07T00:00:{i:02d}Z",
                started_at=None,
                completed_at=None,
                status=JobStatus.PENDING,
                attempt_count=1,
                payload_hash=f"hash_{i}",
                input_path=str(in_p),
                result_path=str(out_p),
            )
        )

    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:
        res = client.post("/v1/jobs/comparison", json=sample_payload)
        assert res.status_code == 429
        assert "Job admission queue is full" in res.json()["detail"]
        assert res.headers.get("Retry-After") == "5"


def test_job_storage_quota_enforcement(tmp_path: Path, sample_payload: dict[str, Any]) -> None:
    """Exceeding storage byte quota returns HTTP 507 Insufficient Storage."""
    storage = JobStorageManager(tmp_path)
    # Create large dummy file to exceed quota
    big_file = storage.inputs_dir / "big_dummy.bin"
    big_file.write_bytes(b"X" * (201 * 1024 * 1024))  # 201 MB

    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:
        res = client.post("/v1/jobs/comparison", json=sample_payload)
        assert res.status_code == 507
        assert "storage quota exceeded" in res.json()["detail"].lower()


def test_job_path_traversal_validation(tmp_path: Path) -> None:
    """Malformed or non-UUID job IDs are rejected with HTTP 422."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:
        res = client.get("/v1/jobs/../../etc/passwd")
        # Starlette router normalizes or rejects
        assert res.status_code in (404, 422)

        res2 = client.get("/v1/jobs/not-a-valid-uuid")
        assert res2.status_code == 422
        assert "must be a valid UUID4" in res2.json()["detail"]


def test_supervisor_single_ownership_lock(tmp_path: Path) -> None:
    """Only one supervisor can acquire leadership; secondary instances yield gracefully."""
    storage = JobStorageManager(tmp_path)
    repo = JobRepository(tmp_path / "jobs.db")

    s1 = JobSupervisor(storage, repo)
    s2 = JobSupervisor(storage, repo)

    try:
        assert s1.start() is True
        assert s1.is_leader is True

        # Second supervisor cannot acquire lock
        assert s2.start() is False
        assert s2.is_leader is False

    finally:
        s1.stop()
        s2.stop()

    # Now s2 can acquire leadership
    try:
        assert s2.start() is True
        assert s2.is_leader is True
    finally:
        s2.stop()


def test_raw_data_never_logged(
    tmp_path: Path, sample_payload: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure raw dataset arrays and contexts are never logged in service output."""
    caplog.set_level(logging.INFO)
    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        res = client.post("/v1/jobs/comparison", json=sample_payload)
        assert res.status_code == 202
        job_id = res.json()["job_id"]
        wait_for_job_terminal(client, job_id)

    # Check all captured log text
    full_logs = "\n".join(record.message for record in caplog.records)
    # The distinct context value from sample_payload should not be in logs
    first_ctx_val = str(sample_payload["dataset"]["contexts"][0][0])
    assert first_ctx_val not in full_logs


def test_job_cancellation_running_while_another_job_runs(
    tmp_path: Path, sample_payload: dict[str, Any]
) -> None:
    """Canceling a running job terminates only that child process, leaving concurrent jobs fine."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        p1 = dict(sample_payload)
        p1["seed"] = 101
        p2 = dict(sample_payload)
        p2["seed"] = 202

        res1 = client.post("/v1/jobs/comparison", json=p1)
        res2 = client.post("/v1/jobs/comparison", json=p2)
        assert res1.status_code == 202
        assert res2.status_code == 202
        j1 = res1.json()["job_id"]
        j2 = res2.json()["job_id"]

        # Cancel job 1
        del_res = client.delete(f"/v1/jobs/{j1}")
        assert del_res.status_code == 200

        # Repeated cancellation request returns 200 CANCELED
        del_repeat = client.delete(f"/v1/jobs/{j1}")
        assert del_repeat.status_code == 200
        assert del_repeat.json()["status"] == "CANCELED"

        # Verify job 1 ends in CANCELED
        d1 = wait_for_job_terminal(client, j1)
        assert d1["status"] == "CANCELED"

        # Verify job 2 runs to completion unaffected
        d2 = wait_for_job_terminal(client, j2)
        assert d2["status"] == "COMPLETED"
        assert d2["result"] is not None


def test_worker_crash_fault_injection(tmp_path: Path) -> None:
    """If a child worker process crashes or exits non-zero, supervisor marks job FAILED."""
    storage = JobStorageManager(tmp_path)
    repo = JobRepository(tmp_path / "jobs.db")

    job_id = "00000000-0000-4000-8000-000000000099"
    # Write invalid JSON into input file so worker fails on parse
    in_path = storage.inputs_dir / f"{job_id}.json"
    in_path.write_text("NOT_JSON", encoding="utf-8")
    out_path = storage.get_result_path(job_id)

    repo.insert_job(
        JobRecord(
            job_id=job_id,
            created_at="2026-11-07T00:00:00Z",
            started_at=None,
            completed_at=None,
            status=JobStatus.PENDING,
            attempt_count=1,
            payload_hash="crash_test_hash",
            input_path=str(in_path),
            result_path=str(out_path),
        )
    )

    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        final_data = wait_for_job_terminal(client, job_id)
        assert final_data["status"] == "FAILED"
        assert final_data["error"] is not None
