"""Background worker supervisor managing single-node child process execution."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from reversallab.service.job_models import JobStatus
from reversallab.service.job_repository import JobRepository
from reversallab.service.job_storage import JobStorageManager

logger = logging.getLogger("reversallab.supervisor")


class FileLock:
    """Advisory file lock for supervisor leader election on Windows and POSIX."""

    def __init__(self, lock_path: str | Path) -> None:
        self.lock_path = str(lock_path)
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Attempt non-blocking acquisition of the lock file."""
        try:
            if sys.platform == "win32":
                import msvcrt

                fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    self._fd = fd
                    return True
                except OSError:
                    os.close(fd)
                    return False
            else:
                import fcntl

                fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._fd = fd
                    return True
                except (OSError, BlockingIOError):
                    os.close(fd)
                    return False
        except Exception as exc:
            logger.debug("Failed to acquire supervisor lock: %s", exc)
            return False

    def release(self) -> None:
        """Release the held lock file."""
        if self._fd is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except Exception:
                pass
            finally:
                self._fd = None


class JobSupervisor:
    """Singleton supervisor coordinating persistent SQLite jobs and bounded child processes."""

    def __init__(
        self,
        storage: JobStorageManager,
        repo: JobRepository,
        max_active_workers: int = 2,
        job_timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        self.storage = storage
        self.repo = repo
        self.max_active_workers = max_active_workers
        self.job_timeout_seconds = job_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

        self.lock = FileLock(self.storage.root_dir / "supervisor.lock")
        self.is_leader = False

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_processes: dict[str, tuple[subprocess.Popen[bytes], float, str]] = {}
        self._last_purge_time = 0.0

    def _safe_terminate_orphan_pid(self, pid: int, job_id: str) -> None:
        """Safely verify and terminate an orphaned worker process from a supervisor crash."""
        if pid <= 0:
            return
        try:
            is_worker = False
            if sys.platform != "win32":
                proc_cmd = Path(f"/proc/{pid}/cmdline")
                if proc_cmd.exists():
                    cmd = proc_cmd.read_text(encoding="utf-8", errors="ignore")
                    if "reversallab.service.worker" in cmd and job_id in cmd:
                        is_worker = True
            else:
                try:
                    import psutil  # type: ignore[import-untyped]

                    p = psutil.Process(pid)
                    cmd_str = " ".join(p.cmdline())
                    if "reversallab.service.worker" in cmd_str and job_id in cmd_str:
                        is_worker = True
                except Exception:
                    pass

            if is_worker:
                logger.info("Terminating orphaned worker process PID %d for job %s", pid, job_id)
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True,
                    )
                else:
                    os.kill(pid, 15)  # SIGTERM
        except Exception as exc:
            logger.debug("Could not terminate orphaned PID %d: %s", pid, exc)

    def start(self) -> bool:
        """Attempt to acquire supervisor leadership and launch background polling thread."""
        if not self.lock.acquire():
            logger.info("Supervisor lock held by peer worker; running in API-only mode.")
            self.is_leader = False
            return False

        self.is_leader = True
        logger.info("Supervisor leader lock acquired. Recovering orphaned jobs...")

        # Terminate any zombie child worker processes left running by prior leader crash
        try:
            orphans = self.repo.get_orphaned_running_jobs()
            for orphan in orphans:
                if orphan.worker_pid and orphan.worker_pid > 0:
                    self._safe_terminate_orphan_pid(orphan.worker_pid, orphan.job_id)
        except Exception as exc:
            logger.debug("Error inspecting orphaned worker processes: %s", exc)

        # Recover any jobs left in RUNNING by previous crash/restart
        recovered = self.repo.recover_orphaned_jobs()
        if recovered > 0:
            logger.warning("Recovered %d orphaned RUNNING jobs to FAILED status.", recovered)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="JobSupervisor")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Terminate all active child processes, release leadership, and stop loop."""
        if not self.is_leader:
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        # Terminate any remaining child processes
        for job_id, (proc, _, exec_token) in list(self._active_processes.items()):
            self._terminate_process(proc)
            rec = self.repo.get_job(job_id)
            if rec and rec.status != JobStatus.COMPLETED:
                self.repo.mark_failed(
                    job_id,
                    "SUPERVISOR_SHUTDOWN",
                    "Job was terminated due to server shutdown.",
                    execution_token=exec_token,
                )

        self._active_processes.clear()
        self.lock.release()
        self.is_leader = False
        global _supervisor_instance
        if _supervisor_instance is self:
            _supervisor_instance = None
        logger.info("Supervisor stopped and leadership released.")

    def cancel_active_job(self, job_id: str) -> bool:
        """Immediately terminate a specific child process if currently running."""
        # Record cancellation request in DB
        success, rec = self.repo.request_cancellation(job_id)
        if not success or rec is None:
            return False

        if rec.status == JobStatus.CANCELED:
            # Was PENDING, already canceled directly
            self.storage.delete_job_files(job_id)
            return True

        # If it is running locally in our active processes
        if job_id in self._active_processes:
            proc, _, _ = self._active_processes.pop(job_id)
            self._terminate_process(proc)
            self.repo.mark_canceled(job_id)
            self.storage.delete_job_files(job_id)
            return True

        if rec.worker_pid and rec.worker_pid > 0:
            self._safe_terminate_orphan_pid(rec.worker_pid, job_id)
        self.repo.mark_canceled(job_id)
        self.storage.delete_job_files(job_id)
        return True

    def _terminate_process(self, proc: subprocess.Popen[bytes]) -> None:
        """Terminate a child process safely with escalation to kill."""
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
        except Exception:
            pass

    def _run_loop(self) -> None:
        """Main supervisor monitoring and dispatch loop."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.error("Unexpected error in supervisor loop tick: %s", exc)

            time.sleep(self.poll_interval_seconds)

    def _tick(self) -> None:
        now = time.time()

        # 1. Inspect active processes
        for job_id, (proc, start_time, exec_token) in list(self._active_processes.items()):
            ret = proc.poll()
            if ret is not None:
                # Process exited
                del self._active_processes[job_id]
                rec = self.repo.get_job(job_id)

                if rec and rec.cancel_requested:
                    self.repo.mark_canceled(job_id)
                    self.storage.delete_job_files(job_id)
                elif ret == 2:
                    # Worker exited with code 2 due to pre/post-commit cancellation fence
                    self.repo.mark_canceled(job_id)
                    self.storage.delete_job_files(job_id)
                elif ret != 0:
                    if rec and rec.status == JobStatus.RUNNING:
                        self.repo.mark_failed(
                            job_id,
                            "WORKER_PROCESS_CRASHED",
                            f"Worker process exited unexpectedly with code {ret}",
                            execution_token=exec_token,
                        )
                else:
                    # Successful exit: verify result file
                    res_path = self.storage.get_result_path(job_id)
                    if not res_path.exists():
                        self.repo.mark_failed(
                            job_id,
                            "RESULT_FILE_MISSING",
                            "Worker completed with code 0 but result file was not found on disk.",
                            execution_token=exec_token,
                        )
                    else:
                        self.repo.mark_completed(job_id, execution_token=exec_token)

            elif now - start_time > self.job_timeout_seconds:
                # Timeout exceeded
                del self._active_processes[job_id]
                self._terminate_process(proc)
                self.repo.mark_failed(
                    job_id,
                    "JOB_TIMEOUT_EXCEEDED",
                    f"Job exceeded maximum runtime limit of {self.job_timeout_seconds} seconds.",
                    execution_token=exec_token,
                )

        # 2. Check for cancellation requests on active jobs
        for job_id, (proc, _, _) in list(self._active_processes.items()):
            rec = self.repo.get_job(job_id)
            if rec and rec.cancel_requested:
                del self._active_processes[job_id]
                self._terminate_process(proc)
                self.repo.mark_canceled(job_id)
                self.storage.delete_job_files(job_id)

        # 3. Dispatch new jobs up to capacity
        while len(self._active_processes) < self.max_active_workers:
            # Temporary dummy PID for reservation, updated once process spawns
            dummy_pid = os.getpid()
            exec_token = uuid.uuid4().hex
            claimed = self.repo.claim_next_pending_job(dummy_pid, execution_token=exec_token)
            if claimed is None:
                break

            job_id = claimed.job_id
            in_path = self.storage.get_input_path(job_id)
            out_path = self.storage.get_result_path(job_id)
            db_path = self.repo.db_path

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
                str(db_path),
                "--execution-token",
                exec_token,
            ]

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.repo.update_worker_pid(job_id, proc.pid)
                self._active_processes[job_id] = (proc, time.time(), exec_token)
            except Exception as exc:
                self.repo.mark_failed(
                    job_id,
                    "WORKER_SPAWN_FAILED",
                    f"Failed to spawn worker process: {exc}",
                    execution_token=exec_token,
                )

        # 4. Periodic storage purge (every 5 minutes)
        if now - self._last_purge_time > 300.0:
            self._last_purge_time = now
            try:
                self.storage.purge_expired_files(max_age_seconds=86400.0)
                self.repo.purge_expired_records(max_age_seconds=86400.0)
            except Exception as exc:
                logger.debug("Failed periodic purge: %s", exc)


# Global singleton holder for FastAPI lifespan management
_supervisor_instance: JobSupervisor | None = None


def get_supervisor() -> JobSupervisor | None:
    return _supervisor_instance


def initialize_supervisor(
    storage: JobStorageManager,
    repo: JobRepository,
    max_active_workers: int = 2,
    job_timeout_seconds: float = 300.0,
    force: bool = False,
) -> JobSupervisor:
    global _supervisor_instance
    if (
        _supervisor_instance is None
        or force
        or _supervisor_instance.storage.root_dir != storage.root_dir
    ):
        if _supervisor_instance is not None and _supervisor_instance.is_leader:
            _supervisor_instance.stop()
        _supervisor_instance = JobSupervisor(
            storage=storage,
            repo=repo,
            max_active_workers=max_active_workers,
            job_timeout_seconds=job_timeout_seconds,
        )
    return _supervisor_instance
