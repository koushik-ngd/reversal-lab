"""SQLite metadata repository for ReversalLab asynchronous evaluation jobs."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from reversallab.service.job_models import JobRecord, JobStatus


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class JobRepository:
    """Thread-safe SQLite repository with WAL mode, busy timeout, and atomic transitions."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a configured SQLite connection with row factory and WAL mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            isolation_level="DEFERRED",
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create database tables and indices if not already present."""
        with self._connection() as conn, conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        status TEXT NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 1,
                        payload_hash TEXT NOT NULL,
                        input_path TEXT NOT NULL,
                        result_path TEXT NOT NULL,
                        error_code TEXT,
                        error_message TEXT,
                        cancel_requested INTEGER NOT NULL DEFAULT 0,
                        worker_pid INTEGER
                    );
                    """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_payload_hash ON jobs(payload_hash);")

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        """Convert a SQLite Row into a typed JobRecord."""
        return JobRecord(
            job_id=str(row["job_id"]),
            created_at=str(row["created_at"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            status=JobStatus(row["status"]),
            attempt_count=int(row["attempt_count"]),
            payload_hash=str(row["payload_hash"]),
            input_path=str(row["input_path"]),
            result_path=str(row["result_path"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            cancel_requested=bool(row["cancel_requested"]),
            worker_pid=row["worker_pid"],
        )

    def insert_job(self, job: JobRecord) -> None:
        """Insert a newly submitted job into the repository."""
        with self._connection() as conn, conn:
            conn.execute(
                """
                    INSERT INTO jobs (
                        job_id, created_at, started_at, completed_at, status,
                        attempt_count, payload_hash, input_path, result_path,
                        error_code, error_message, cancel_requested, worker_pid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                (
                    job.job_id,
                    job.created_at,
                    job.started_at,
                    job.completed_at,
                    job.status.value,
                    job.attempt_count,
                    job.payload_hash,
                    job.input_path,
                    job.result_path,
                    job.error_code,
                    job.error_message,
                    1 if job.cancel_requested else 0,
                    job.worker_pid,
                ),
            )

    def insert_or_get_active(self, job: JobRecord) -> tuple[JobRecord, bool]:
        """Atomically insert job if no active job exists with the same payload_hash.

        Returns (record, inserted). If an active job already exists, returns (existing, False).
        Uses an immediate transaction to eliminate race conditions between concurrent submissions.
        """
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE payload_hash = ? AND status IN ('PENDING', 'RUNNING', 'COMPLETED')
                    ORDER BY created_at DESC
                    LIMIT 1;
                    """,
                    (job.payload_hash,),
                )
                row = cursor.fetchone()
                if row is not None:
                    conn.commit()
                    return self._row_to_record(row), False

                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, created_at, started_at, completed_at, status,
                        attempt_count, payload_hash, input_path, result_path,
                        error_code, error_message, cancel_requested, worker_pid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        job.job_id,
                        job.created_at,
                        job.started_at,
                        job.completed_at,
                        job.status.value,
                        job.attempt_count,
                        job.payload_hash,
                        job.input_path,
                        job.result_path,
                        job.error_code,
                        job.error_message,
                        1 if job.cancel_requested else 0,
                        job.worker_pid,
                    ),
                )
                conn.commit()
                return job, True
            except Exception:
                conn.rollback()
                raise

    def get_job(self, job_id: str) -> JobRecord | None:
        """Retrieve a job by its unique identifier."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?;",
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def find_active_by_payload_hash(self, payload_hash: str) -> JobRecord | None:
        """Find an existing active or recently completed job with matching payload hash."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM jobs
                WHERE payload_hash = ? AND status IN ('PENDING', 'RUNNING', 'COMPLETED')
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (payload_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def count_pending_jobs(self) -> int:
        """Return the number of jobs currently waiting in PENDING status."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) AS cnt FROM jobs WHERE status = 'PENDING';")
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0

    def claim_next_pending_job(self, worker_pid: int) -> JobRecord | None:
        """Atomically claim the oldest non-canceled PENDING job and transition to RUNNING."""
        now = _utc_now_iso()
        with self._connection() as conn:
            with conn:
                # Find oldest candidate
                cursor = conn.execute(
                    """
                    SELECT job_id FROM jobs
                    WHERE status = 'PENDING' AND cancel_requested = 0
                    ORDER BY created_at ASC
                    LIMIT 1;
                    """
                )
                row = cursor.fetchone()
                if row is None:
                    return None

                candidate_id = str(row["job_id"])
                update_cur = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'RUNNING', started_at = ?, worker_pid = ?
                    WHERE job_id = ? AND status = 'PENDING';
                    """,
                    (now, worker_pid, candidate_id),
                )
                if update_cur.rowcount == 0:
                    return None

            # Fetch the updated row
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (candidate_id,))
            updated_row = cursor.fetchone()
            if updated_row is None:
                return None
            return self._row_to_record(updated_row)

    def update_worker_pid(self, job_id: str, worker_pid: int) -> bool:
        """Update the OS worker_pid associated with a RUNNING job."""
        with self._connection() as conn, conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET worker_pid = ?
                WHERE job_id = ? AND status = 'RUNNING';
                """,
                (worker_pid, job_id),
            )
            return cur.rowcount > 0

    def get_orphaned_running_jobs(self) -> list[JobRecord]:
        """Retrieve all job records currently marked as RUNNING."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE status = 'RUNNING';")
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def mark_completed(self, job_id: str) -> bool:
        """Mark a RUNNING job as COMPLETED upon successful output write.

        Fails atomically if cancellation has been requested, preventing stale completion.
        """
        now = _utc_now_iso()
        with self._connection() as conn, conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'COMPLETED', completed_at = ?
                WHERE job_id = ? AND status = 'RUNNING' AND cancel_requested = 0;
                """,
                (now, job_id),
            )
            return cur.rowcount > 0

    def mark_failed(self, job_id: str, error_code: str, error_message: str) -> bool:
        """Mark a job as FAILED with diagnostic details.

        Only transitions from PENDING or RUNNING, unless error_code is RESULT_READ_FAILED
        or RESULT_EXPIRED (recovering corrupted/purged output files).
        """
        now = _utc_now_iso()
        with self._connection() as conn, conn:
            if error_code in ("RESULT_READ_FAILED", "RESULT_EXPIRED"):
                status_filter = "status IN ('PENDING', 'RUNNING', 'COMPLETED')"
            else:
                status_filter = "status IN ('PENDING', 'RUNNING')"

            cur = conn.execute(
                f"""
                UPDATE jobs
                SET status = 'FAILED', completed_at = ?, error_code = ?, error_message = ?
                WHERE job_id = ? AND {status_filter};
                """,
                (now, error_code, error_message, job_id),
            )
            return cur.rowcount > 0

    def request_cancellation(self, job_id: str) -> tuple[bool, JobRecord | None]:
        """Request cancellation. If PENDING, cancel immediately. If RUNNING, set flag."""
        now = _utc_now_iso()
        with self._connection() as conn, conn:
            # Check current status
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
            row = cursor.fetchone()
            if row is None:
                return False, None

            rec = self._row_to_record(row)
            if rec.status == JobStatus.PENDING:
                conn.execute(
                    """
                        UPDATE jobs
                        SET status = 'CANCELED', completed_at = ?, cancel_requested = 1
                        WHERE job_id = ? AND status = 'PENDING';
                        """,
                    (now, job_id),
                )
                # Return updated
                updated_cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
                updated_row = updated_cur.fetchone()
                return True, self._row_to_record(updated_row) if updated_row is not None else None

            if rec.status == JobStatus.RUNNING:
                conn.execute(
                    """
                        UPDATE jobs
                        SET cancel_requested = 1
                        WHERE job_id = ? AND status = 'RUNNING';
                        """,
                    (job_id,),
                )
                return True, rec

            # Already in terminal state
            return False, rec

    def mark_canceled(self, job_id: str) -> bool:
        """Transition a job to CANCELED status after verified worker process termination.

        Terminal states (COMPLETED, FAILED) are immutable and cannot be overwritten.
        """
        now = _utc_now_iso()
        with self._connection() as conn, conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'CANCELED', completed_at = ?
                WHERE job_id = ? AND status IN ('PENDING', 'RUNNING');
                """,
                (now, job_id),
            )
            return cur.rowcount > 0

    def recover_orphaned_jobs(self) -> int:
        """Find any jobs stuck in RUNNING on supervisor startup and mark them FAILED."""
        now = _utc_now_iso()
        with self._connection() as conn, conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'FAILED',
                    completed_at = ?,
                    error_code = 'CRASH_OR_RESTART_DETECTED',
                    error_message = 'Job execution was interrupted by server crash or restart.'
                WHERE status = 'RUNNING';
                """,
                (now,),
            )
            return cur.rowcount

    def purge_expired_records(self, max_age_seconds: float = 86400.0) -> int:
        """Purge terminal job rows older than max_age_seconds."""
        cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - max_age_seconds, tz=UTC
        ).isoformat()
        with self._connection() as conn, conn:
            cur = conn.execute(
                """
                    DELETE FROM jobs
                    WHERE status IN ('COMPLETED', 'FAILED', 'CANCELED')
                      AND completed_at IS NOT NULL
                      AND completed_at < ?;
                    """,
                (cutoff,),
            )
            return cur.rowcount
