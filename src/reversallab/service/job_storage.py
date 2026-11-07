"""Persistent local filesystem storage for asynchronous job inputs and results."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class StorageQuotaExceededError(Exception):
    """Raised when persistent storage directory exceeds configured byte quota."""


class InvalidJobIdError(ValueError):
    """Raised when job_id fails strict UUID4 validation, preventing path traversal."""


class JobStorageManager:
    """Manages persistent job files with atomic writing, quota checks, and traversal protection."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.inputs_dir = self.root_dir / "inputs"
        self.results_dir = self.root_dir / "results"
        self.tmp_dir = self.root_dir / "tmp"

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage directories with restricted permissions."""
        for d in (self.inputs_dir, self.results_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(d, 0o700)

    def validate_job_id(self, job_id: str) -> None:
        """Validate job_id format to prevent path traversal attacks."""
        if not isinstance(job_id, str) or not UUID4_PATTERN.match(job_id):
            raise InvalidJobIdError(f"Invalid job_id format: {job_id!r}")

    def get_input_path(self, job_id: str) -> Path:
        """Return safe path to persisted input file."""
        self.validate_job_id(job_id)
        return self.inputs_dir / f"{job_id}.json"

    def get_result_path(self, job_id: str) -> Path:
        """Return safe path to persisted result file."""
        self.validate_job_id(job_id)
        return self.results_dir / f"{job_id}.json"

    def get_storage_usage_bytes(self) -> int:
        """Compute total byte size of files in storage directory."""
        total = 0
        for p in self.root_dir.rglob("*"):
            if p.is_file():
                with contextlib.suppress(OSError):
                    total += p.stat().st_size
        return total

    def check_quota(self, max_bytes: int = 200 * 1024 * 1024) -> None:
        """Check whether storage quota is exceeded."""
        current = self.get_storage_usage_bytes()
        if current >= max_bytes:
            raise StorageQuotaExceededError(
                f"Job storage quota exceeded: {current} bytes used, limit is {max_bytes} bytes."
            )

    def write_input(
        self, job_id: str, payload: dict[str, Any], max_storage_bytes: int = 200 * 1024 * 1024
    ) -> Path:
        """Write input payload atomically to inputs directory."""
        self.validate_job_id(job_id)
        self.check_quota(max_storage_bytes)

        target_path = self.get_input_path(job_id)
        self._atomic_write_json(target_path, payload)
        return target_path

    def write_result(
        self, job_id: str, result: dict[str, Any], max_storage_bytes: int = 200 * 1024 * 1024
    ) -> Path:
        """Write evaluation result atomically to results directory."""
        self.validate_job_id(job_id)
        self.check_quota(max_storage_bytes)

        target_path = self.get_result_path(job_id)
        self._atomic_write_json(target_path, result)
        return target_path

    def read_input(self, job_id: str) -> dict[str, Any]:
        """Read and parse input payload from disk."""
        target_path = self.get_input_path(job_id)
        if not target_path.exists():
            raise FileNotFoundError(f"Job input file missing: {target_path}")
        with open(target_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data

    def read_result(self, job_id: str) -> dict[str, Any]:
        """Read and parse completed evaluation report from disk."""
        target_path = self.get_result_path(job_id)
        if not target_path.exists():
            raise FileNotFoundError(f"Job result file missing: {target_path}")
        with open(target_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data

    def _atomic_write_json(self, target_path: Path, data: dict[str, Any]) -> None:
        """Write JSON data to a staging file in tmp_dir, flush to disk, and replace target."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(self.tmp_dir),
            delete=False,
            encoding="utf-8",
            prefix=f"{target_path.stem}_",
            suffix=".tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(data, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())

        with contextlib.suppress(OSError):
            os.chmod(tmp_path, 0o600)

        tmp_path.replace(target_path)

    def delete_job_files(self, job_id: str) -> None:
        """Safely delete input and result files associated with a job."""
        self.validate_job_id(job_id)
        in_path = self.get_input_path(job_id)
        out_path = self.get_result_path(job_id)

        for p in (in_path, out_path):
            if p.exists():
                with contextlib.suppress(OSError):
                    p.unlink()

    def purge_expired_files(self, max_age_seconds: float = 86400.0) -> int:
        """Delete files older than max_age_seconds from inputs and results directories."""
        now = time.time()
        purged = 0
        for d in (self.inputs_dir, self.results_dir, self.tmp_dir):
            for p in d.glob("*"):
                if p.is_file() and now - p.stat().st_mtime > max_age_seconds:
                    with contextlib.suppress(OSError):
                        p.unlink()
                        purged += 1
        return purged
