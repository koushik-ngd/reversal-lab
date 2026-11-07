"""Standalone child worker process for CPU-isolated ReversalLab evaluation jobs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from reversallab.api import BanditDataset, PolicyComparison, StratumPartition
from reversallab.service.app import _build_policy
from reversallab.service.job_repository import JobRepository
from reversallab.service.schemas import PolicyComparisonRequest


def execute_job(job_id: str, input_path: Path, result_path: Path, db_path: Path) -> int:
    """Execute evaluation job isolated in this dedicated child process."""
    repo = JobRepository(db_path)

    try:
        if not input_path.exists():
            repo.mark_failed(job_id, "INPUT_FILE_NOT_FOUND", f"Input file missing: {input_path}")
            return 1

        with open(input_path, encoding="utf-8") as f:
            raw_payload = json.load(f)

        req_obj = PolicyComparisonRequest.model_validate(raw_payload)

        # Build policies
        cand_p = _build_policy(req_obj.candidate_policy, req_obj.dataset.n_actions)
        base_p = _build_policy(req_obj.baseline_policy, req_obj.dataset.n_actions)

        # Build dataset
        dataset = BanditDataset.from_arrays(
            contexts=np.asarray(req_obj.dataset.contexts, dtype=np.float64),
            actions=np.asarray(req_obj.dataset.actions, dtype=np.int64),
            rewards=np.asarray(req_obj.dataset.rewards, dtype=np.float64),
            propensities=np.asarray(req_obj.dataset.propensities, dtype=np.float64),
            n_actions=req_obj.dataset.n_actions,
        )

        stratum_partition = (
            StratumPartition(
                split_dim=req_obj.stratum_partition.split_dim,
                split_threshold=req_obj.stratum_partition.threshold,
            )
            if req_obj.stratum_partition is not None
            else None
        )

        seed = req_obj.seed if req_obj.seed is not None else 42

        # Execute statistical evaluation
        report = PolicyComparison.evaluate(
            dataset=dataset,
            candidate_policy=cand_p,
            baseline_policy=base_p,
            estimator=req_obj.estimator,
            margin=req_obj.margin,
            deploy_epsilon=req_obj.deploy_epsilon,
            max_epsilon=req_obj.max_epsilon,
            stratum_partition=stratum_partition,
            confidence_level=req_obj.confidence_level,
            cross_fitting_folds=req_obj.cross_fitting_folds,
            min_propensity_threshold=req_obj.min_propensity_threshold,
            seed=seed,
        )

        result_data = report.to_dict()

        # Atomic write of result JSON
        tmp_dir = result_path.parent.parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(tmp_dir),
            delete=False,
            encoding="utf-8",
            prefix=f"{job_id}_",
            suffix=".tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(result_data, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())

        with contextlib.suppress(OSError):
            os.chmod(tmp_path, 0o600)

        tmp_path.replace(result_path)

        # Mark completed in DB
        repo.mark_completed(job_id)
        return 0

    except Exception as exc:
        repo.mark_failed(job_id, type(exc).__name__, str(exc))
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="ReversalLab isolated child evaluation worker")
    parser.add_argument("--job-id", required=True, help="Job UUID4 identifier")
    parser.add_argument("--input-path", required=True, type=Path, help="Path to input JSON")
    parser.add_argument("--result-path", required=True, type=Path, help="Path to output JSON")
    parser.add_argument("--db-path", required=True, type=Path, help="Path to SQLite database")

    args = parser.parse_args()
    code = execute_job(args.job_id, args.input_path, args.result_path, args.db_path)
    sys.exit(code)


if __name__ == "__main__":
    main()
