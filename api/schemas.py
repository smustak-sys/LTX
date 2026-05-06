"""Pydantic schemas for the LTX-2 video-generation API.

These models also drive the Swagger UI rendered at /docs.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ImageConditioning(BaseModel):
    path: str = Field(..., description="Absolute or repo-relative path to a conditioning image.")
    frame_idx: int = Field(0, description="Frame index this image conditions. 0 replaces; others guide.")
    strength: float = Field(1.0, ge=0.0, le=1.0, description="Conditioning strength in [0, 1].")
    crf: int = Field(35, description="Image-encoding CRF used for the conditioning latent.")


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Cinematographic, single-paragraph description of the shot. Keep <= ~200 words.",
        examples=[
            "A red fox trots across a snowy meadow at dawn, breath visible in the cold air, "
            "low golden sunlight raking across the snow, slow tracking shot from the side."
        ],
    )
    negative_prompt: str = Field(
        "worst quality, blurry, jittery, distorted",
        description="Things to discourage in the output.",
    )
    seed: int = Field(0, description="RNG seed for reproducibility.")
    height: int = Field(704, ge=64, description="Output height in pixels (must satisfy pipeline constraints).")
    width: int = Field(1216, ge=64, description="Output width in pixels (must satisfy pipeline constraints).")
    num_frames: int = Field(121, ge=1, description="Total frames to generate.")
    frame_rate: float = Field(24.0, gt=0, description="Output frame rate (fps).")
    num_inference_steps: int = Field(30, ge=1, le=200, description="Denoising steps for stage 1.")
    video_cfg_scale: float = Field(3.0, description="Video classifier-free guidance scale.")
    audio_cfg_scale: float = Field(7.0, description="Audio classifier-free guidance scale.")
    enhance_prompt: bool = Field(False, description="Run automatic prompt enhancement.")
    images: list[ImageConditioning] = Field(
        default_factory=list, description="Optional image conditioning inputs."
    )
    max_batch_size: int = Field(1, ge=1, description="Internal batch-split cap; raise if you have spare VRAM.")


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    request: GenerateRequest
    error: Optional[str] = None
    output_path: Optional[str] = None


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus
    poll_url: str
    download_url: str
