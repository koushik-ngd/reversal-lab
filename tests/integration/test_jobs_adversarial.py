"""Adversarial fault-injection and concurrency test suite for ReversalLab engine (M6C)."""

from __future__ import annotations

import concurrent.futures
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from starlette.testclient import TestClient

from reversallab.api import BanditDataset, UniformPolicy
from reversallab.data.synthetic import SyntheticBanditEnvironment
from reversallab.service.app import create_app
from reversallab.service.job_models import JobRecord, JobStatus
from reversallab.service.job_repository import JobRepository
from reversallab.service.job_storage import JobStorageManager
from reversallab.service.supervisor import initialize_supervisor


@pytest.fixture
def adversarial_payload() -> dict[str, Any]:
    """Generate reproducible sample payload for adversarial tests."""
    rng = np.random.default_rng(123)
    env = SyntheticBanditEnvironment.random(
        n_actions=3, context_dim=2, rng=rng, reward_scale=1.0, bias_scale=0.0
    )
    logging_policy = UniformPolicy(n_actions=3)
    dataset = BanditDataset.from_synthetic(
        env, logging_policy=logging_policy, n_samples=300, seed=17
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
        "seed": 123,
    }


def wait_for_terminal(client: TestClient, job_id: str, timeout_s: float = 15.0) -> dict[str, Any]:
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


def test_model_a_security_boundary_no_tenant_isolation(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Audit Model A: Verify that job_id is an operational identifier, not an auth token."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        # Client 1 submits a job
        res1 = client.post("/v1/jobs/comparison", json=adversarial_payload)
        assert res1.status_code == 202
        job_id = res1.json()["job_id"]

        # Client 2 (unauthenticated, different client headers) can poll the same job
        res2 = client.get(
            f"/v1/jobs/{job_id}",
            headers={"x-client-id": "client-2", "user-agent": "AuditAgent/1.0"},
        )
        assert res2.status_code == 200
        assert res2.json()["job_id"] == job_id

        # Client 2 can also cancel Client 1's job under Model A
        res_cancel = client.delete(
            f"/v1/jobs/{job_id}",
            headers={"x-client-id": "client-2"},
        )
        assert res_cancel.status_code == 200
        assert res_cancel.json()["status"] == "CANCELED"


def test_canonical_payload_hashing_consistency(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Verify that differently ordered JSON keys resolve to identical canonical SHA-256 hash."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:
        # Submit original payload
        res1 = client.post("/v1/jobs/comparison", json=adversarial_payload)
        assert res1.status_code == 202
        job_id_1 = res1.json()["job_id"]

        # Construct payload with reversed dictionary keys
        reversed_payload = {
            k: adversarial_payload[k] for k in reversed(list(adversarial_payload.keys()))
        }

        # Submit reordered payload
        res2 = client.post("/v1/jobs/comparison", json=reversed_payload)
        assert res2.status_code == 202
        job_id_2 = res2.json()["job_id"]

        # Deduplication must identify them as the exact same job
        assert job_id_1 == job_id_2


def test_idempotency_allows_rerun_after_cancellation(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Verify that a previously CANCELED job does not permanently block resubmissions."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:
        # 1. Submit job
        res1 = client.post("/v1/jobs/comparison", json=adversarial_payload)
        assert res1.status_code == 202
        job_id_1 = res1.json()["job_id"]

        # 2. Cancel the pending job
        res_cancel = client.delete(f"/v1/jobs/{job_id_1}")
        assert res_cancel.status_code == 200
        assert res_cancel.json()["status"] == "CANCELED"

        # 3. Resubmit identical payload: must admit as a new job, NOT deduplicate to CANCELED
        res2 = client.post("/v1/jobs/comparison", json=adversarial_payload)
        assert res2.status_code == 202
        job_id_2 = res2.json()["job_id"]
        assert job_id_2 != job_id_1
        assert res2.json()["status"] == "PENDING"


def test_idempotency_allows_rerun_when_completed_result_missing(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Verify that if a completed result file is deleted from disk, fresh evaluation is admitted."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        # 1. Submit and wait for completion
        res1 = client.post("/v1/jobs/comparison", json=adversarial_payload)
        assert res1.status_code == 202
        job_id_1 = res1.json()["job_id"]

        final_1 = wait_for_terminal(client, job_id_1)
        assert final_1["status"] == "COMPLETED"

        # 2. Simulate file retention purge: delete result file from disk
        storage = JobStorageManager(tmp_path)
        res_file = storage.get_result_path(job_id_1)
        assert res_file.exists()
        res_file.unlink()

        # 3. Resubmit identical payload: must detect missing result file and admit fresh job
        res2 = client.post("/v1/jobs/comparison", json=adversarial_payload)
        assert res2.status_code == 202
        job_id_2 = res2.json()["job_id"]
        assert job_id_2 != job_id_1


def test_concurrent_identical_submissions_atomic_deduplication(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Adversarially submit 10 identical requests concurrently; verify no duplicate jobs in DB."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:

        def submit_one() -> str:
            res = client.post("/v1/jobs/comparison", json=adversarial_payload)
            assert res.status_code == 202
            return str(res.json()["job_id"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(submit_one) for _ in range(10)]
            job_ids = [f.result() for f in futures]

        # All concurrent submissions must resolve to exactly the same single job ID
        assert len(set(job_ids)) == 1
        winning_id = job_ids[0]

        # Verify only 1 row exists in SQLite
        repo = JobRepository(tmp_path / "jobs.db")
        assert repo.count_pending_jobs() == 1
        rec = repo.get_job(winning_id)
        assert rec is not None
        assert rec.status == JobStatus.PENDING


def test_cancel_running_job_pre_commit_fence(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Verify that canceling a running job kills the process and prevents result file write."""
    # Use larger folds to ensure job runs for at least ~1.5s
    payload = dict(adversarial_payload)
    payload["cross_fitting_folds"] = 10

    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        res = client.post("/v1/jobs/comparison", json=payload)
        assert res.status_code == 202
        job_id = res.json()["job_id"]

        # Wait until job starts running
        t0 = time.time()
        while time.time() - t0 < 5.0:
            st = client.get(f"/v1/jobs/{job_id}").json()
            if st["status"] == "RUNNING":
                break
            time.sleep(0.05)

        # Issue cancellation while running
        res_del = client.delete(f"/v1/jobs/{job_id}")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "CANCELED"

        # Verify result file does not exist on disk
        storage = JobStorageManager(tmp_path)
        assert not storage.get_result_path(job_id).exists()

        # Polling confirmed canceled
        final_st = client.get(f"/v1/jobs/{job_id}").json()
        assert final_st["status"] == "CANCELED"


def test_cancel_terminal_state_immutability(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Verify that once a job is COMPLETED, cancellation requests fail with 409 Conflict."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        res = client.post("/v1/jobs/comparison", json=adversarial_payload)
        job_id = res.json()["job_id"]

        final_st = wait_for_terminal(client, job_id)
        assert final_st["status"] == "COMPLETED"

        # Attempt cancellation on completed job
        res_del = client.delete(f"/v1/jobs/{job_id}")
        assert res_del.status_code == 409
        err = res_del.json()
        assert "Cannot cancel job in terminal state" in err["detail"]

        # Verify status in DB remains COMPLETED
        st_after = client.get(f"/v1/jobs/{job_id}").json()
        assert st_after["status"] == "COMPLETED"


def test_worker_fence_prevents_stale_completion_after_crash_recovery(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Verify that a worker cannot mark a job COMPLETED if the supervisor recovered it to FAILED."""
    storage = JobStorageManager(tmp_path)
    repo = JobRepository(tmp_path / "jobs.db")

    job_id = "11111111-2222-4333-8444-555555555555"
    in_path = storage.write_input(job_id, adversarial_payload, 10_000_000)
    out_path = storage.get_result_path(job_id)

    # Insert job as RUNNING
    rec = JobRecord(
        job_id=job_id,
        created_at="2026-11-07T00:00:00Z",
        started_at="2026-11-07T00:00:01Z",
        completed_at=None,
        status=JobStatus.RUNNING,
        attempt_count=1,
        payload_hash="dummy_hash",
        input_path=str(in_path),
        result_path=str(out_path),
    )
    repo.insert_job(rec)

    # Simulate supervisor crash recovery before worker finishes
    repo.recover_orphaned_jobs()
    job_after_recovery = repo.get_job(job_id)
    assert job_after_recovery is not None
    assert job_after_recovery.status == JobStatus.FAILED

    # Now run worker CLI on this job
    cmd = [
        sys.executable,
        "-m",
        "reversallab.service.worker",
        "--job-id",
        job_id,
        "--input-path",
        str(in_path),
        "--result-path",
        str(out_path),
        "--db-path",
        str(repo.db_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    # Worker must exit with code 2 (fenced) and NOT overwrite FAILED status with COMPLETED
    assert proc.returncode == 2
    final_job = repo.get_job(job_id)
    assert final_job is not None
    assert final_job.status == JobStatus.FAILED
    assert not out_path.exists()


def test_orphaned_zombie_worker_terminated_on_startup(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Verify supervisor terminates running zombie worker processes from prior crash on startup."""
    storage = JobStorageManager(tmp_path)
    repo = JobRepository(tmp_path / "jobs.db")

    # High fold count so process stays alive
    payload = dict(adversarial_payload)
    payload["cross_fitting_folds"] = 15

    job_id = "22222222-3333-4444-8555-666666666666"
    in_path = storage.write_input(job_id, payload, 10_000_000)
    out_path = storage.get_result_path(job_id)

    # Spawn worker directly in background
    cmd = [
        sys.executable,
        "-m",
        "reversallab.service.worker",
        "--job-id",
        job_id,
        "--input-path",
        str(in_path),
        "--result-path",
        str(out_path),
        "--db-path",
        str(repo.db_path),
    ]
    proc = subprocess.Popen(cmd)
    try:
        # Insert job as RUNNING with child process PID
        rec = JobRecord(
            job_id=job_id,
            created_at="2026-11-07T00:00:00Z",
            started_at="2026-11-07T00:00:01Z",
            completed_at=None,
            status=JobStatus.RUNNING,
            attempt_count=1,
            payload_hash="dummy_hash_2",
            input_path=str(in_path),
            result_path=str(out_path),
            worker_pid=proc.pid,
        )
        repo.insert_job(rec)

        # Now start supervisor, which must detect the zombie worker PID and terminate it
        sup = initialize_supervisor(storage, repo, force=True)
        sup.start()
        try:
            # Process must be terminated within 3 seconds
            t0 = time.time()
            is_alive = True
            while time.time() - t0 < 3.0:
                if proc.poll() is not None:
                    is_alive = False
                    break
                time.sleep(0.1)
            assert not is_alive, (
                f"Zombie worker process {proc.pid} was not terminated by supervisor."
            )

            # Job status must be FAILED with CRASH_OR_RESTART_DETECTED
            job_rec = repo.get_job(job_id)
            assert job_rec is not None
            assert job_rec.status == JobStatus.FAILED
            assert job_rec.error_code == "CRASH_OR_RESTART_DETECTED"
        finally:
            sup.stop()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_input_file_rollback_on_db_insert_error(
    tmp_path: Path, adversarial_payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that if SQLite insertion fails, the staged input file is cleaned up immediately."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:
        # Mock insert_or_get_active to simulate disk full or database error
        def mock_insert(*args: Any, **kwargs: Any) -> Any:
            raise sqlite3.OperationalError("disk I/O error")

        storage = cast(JobStorageManager, app.state.storage)
        monkeypatch.setattr(app.state.repo, "insert_or_get_active", mock_insert)

        res = client.post("/v1/jobs/comparison", json=adversarial_payload)
        # Should return 500 error
        assert res.status_code == 500

        # Assert no input files were left in storage
        input_files = list(storage.inputs_dir.glob("*.json"))
        assert len(input_files) == 0


def test_corrupt_result_file_transitions_to_failed(
    tmp_path: Path, adversarial_payload: dict[str, Any]
) -> None:
    """Verify corrupted result JSON returns RESULT_READ_FAILED and updates DB to FAILED."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=True)
    with TestClient(app) as client:
        res = client.post("/v1/jobs/comparison", json=adversarial_payload)
        assert res.status_code == 202
        job_id = res.json()["job_id"]

        final_st = wait_for_terminal(client, job_id)
        assert final_st["status"] == "COMPLETED"

        # Corrupt the result JSON file on disk
        storage = JobStorageManager(tmp_path)
        res_file = storage.get_result_path(job_id)
        assert res_file.exists()
        res_file.write_text("{corrupt: [json", encoding="utf-8")

        # Now poll the job again
        poll_res = client.get(f"/v1/jobs/{job_id}")
        assert poll_res.status_code == 200
        poll_data = poll_res.json()
        assert poll_data["status"] == "FAILED"
        assert poll_data["error"]["code"] == "RESULT_READ_FAILED"

        # Verify DB row was also updated to FAILED
        repo = JobRepository(tmp_path / "jobs.db")
        job_rec = repo.get_job(job_id)
        assert job_rec is not None
        assert job_rec.status == JobStatus.FAILED
        assert job_rec.error_code == "RESULT_READ_FAILED"


def test_sqlite_locked_returns_503(
    tmp_path: Path, adversarial_payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify sqlite3.OperationalError with 'locked' returns HTTP 503 with Retry-After header."""
    app = create_app(storage_dir=tmp_path, enable_supervisor=False)
    with TestClient(app) as client:

        def mock_insert(*args: Any, **kwargs: Any) -> Any:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(app.state.repo, "insert_or_get_active", mock_insert)

        res = client.post("/v1/jobs/comparison", json=adversarial_payload)
        assert res.status_code == 503
        data = res.json()
        assert data["title"] == "Service Unavailable"
        assert "Retry-After" in res.headers
        assert res.headers["Retry-After"] == "1"
