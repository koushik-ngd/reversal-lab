"""Data models and schemas for ReversalLab asynchronous job processing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Lifecycle states for asynchronous evaluation jobs."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


@dataclass(frozen=True)
class JobRecord:
    """Internal SQLite database representation of an evaluation job."""

    job_id: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    status: JobStatus
    attempt_count: int
    payload_hash: str
    input_path: str
    result_path: str
    error_code: str | None = None
    error_message: str | None = None
    cancel_requested: bool = False
    worker_pid: int | None = None
    execution_token: str | None = None


class JobLinks(BaseModel):
    """HATEOAS navigational links for polling and managing jobs."""

    self: str = Field(description="URL to poll job status and result.")
    cancel: str = Field(description="URL to request job cancellation.")


class JobSubmissionResponse(BaseModel):
    """Response returned when an evaluation job is accepted for execution."""

    job_id: str = Field(description="Unique UUID4 identifier for the submitted job.")
    status: JobStatus = Field(description="Initial job state (always PENDING).")
    created_at: str = Field(description="ISO 8601 UTC timestamp of submission.")
    links: JobLinks = Field(description="Actionable hypermedia links.")


class JobErrorDetails(BaseModel):
    """Sanitized failure diagnostic details for a FAILED job."""

    code: str = Field(description="Structured error category code.")
    message: str = Field(description="Safe, non-sensitive diagnostic message.")


class JobStatusResponse(BaseModel):
    """Response returned when polling job state and results."""

    job_id: str = Field(description="Job identifier.")
    status: JobStatus = Field(description="Current lifecycle status.")
    artifact_available: bool = Field(
        default=True, description="Whether the result artifact is currently available on storage."
    )
    created_at: str = Field(description="Submission timestamp.")
    started_at: str | None = Field(default=None, description="Execution start timestamp.")
    completed_at: str | None = Field(default=None, description="Completion timestamp.")
    attempt_count: int = Field(default=1, description="Number of execution attempts.")
    result: dict[str, Any] | None = Field(
        default=None, description="Full evaluation report if COMPLETED."
    )
    error: JobErrorDetails | None = Field(
        default=None, description="Error details if FAILED or CANCELED."
    )
