"""Run provenance and artifact writing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import reversallab

SCHEMA_VERSION = "1"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert to strict-JSON types; non-finite floats become ``None``."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def git_revision(cwd: Path | None = None) -> dict[str, Any]:
    """Return the current commit and dirty flag, or ``None`` values if unavailable."""

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit is not None else None
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def sha256_file(path: Path) -> str:
    """SHA-256 of file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_lockfile() -> Path | None:
    """Find ``uv.lock`` by walking up from the working directory and this file."""
    starts = [Path.cwd(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            if directory in seen:
                continue
            seen.add(directory)
            candidate = directory / "uv.lock"
            if candidate.is_file():
                return candidate
    return None


def collect_provenance() -> dict[str, Any]:
    lock = find_lockfile()
    return {
        "reversallab_version": reversallab.__version__,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "git": git_revision(),
        "uv_lock_sha256": None if lock is None else sha256_file(lock),
    }


def write_run(
    output_dir: Path,
    *,
    name: str,
    config: dict[str, Any],
    results: dict[str, Any],
    warnings: list[str],
    runtime: dict[str, Any],
    reproduction_command: str,
) -> Path:
    """Write ``run.json`` to a new run directory and return that directory.

    ``results_sha256`` covers only the ``results`` section, which is a deterministic
    function of the config and the code; timing and environment details are excluded
    so that identical configurations can be compared across machines.
    """
    started = datetime.now(UTC)
    config_hash = sha256_of(config)
    run_id = f"{name}_{started:%Y%m%dT%H%M%S%fZ}_{config_hash[:8]}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_name": name,
        "created_utc": started.isoformat(),
        "config": config,
        "config_sha256": config_hash,
        "results": results,
        "results_sha256": sha256_of(results),
        "warnings": warnings,
        "runtime": runtime,
        "provenance": collect_provenance(),
        "reproduction_command": reproduction_command,
    }
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, run_dir / "run.json")
    return run_dir


def load_run(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"No run.json in {run_dir}; pass the directory created by a run.")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
