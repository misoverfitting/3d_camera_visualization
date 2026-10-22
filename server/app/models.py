"""Pydantic request/response schemas."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ReconstructionMode = Literal["accurate", "compelling"]
JobStatus = Literal["new", "queued", "running", "done", "error"]


class SessionCreate(BaseModel):
    name: str = Field(default="Untitled capture", max_length=200)


class SessionInfo(BaseModel):
    id: str
    name: str
    created_at: str
    photo_count: int
    status: str
    source: str = 'rgb'


class PhotoUploadResult(BaseModel):
    filename: str
    photo_count: int


class ReconstructRequest(BaseModel):
    mode: ReconstructionMode = "accurate"
    allow_sparse_fallback: bool = False


class JobProgress(BaseModel):
    status: JobStatus
    stage: str
    message: str
    percent: int = 0
    # `percent` is retained for older clients; it is a pipeline milestone, not
    # measured whole-job completion. New clients use stage_percent or activity.
    stage_percent: Optional[float] = None
    completed_units: Optional[int] = None
    total_units: Optional[int] = None
    unit: Optional[str] = None
    work_detail: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    stage_elapsed_seconds: Optional[float] = None
    last_output_seconds_ago: Optional[float] = None
    result_files: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class ScaleCalibration(BaseModel):
    # Two points in the exported mesh's local coordinate space (from the
    # viewer's picking raycast), plus the real-world distance between them
    # in meters, as measured on the physical reference object.
    point_a: tuple[float, float, float]
    point_b: tuple[float, float, float]
    real_world_distance_m: float = Field(gt=0)
