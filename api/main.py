"""FastAPI entry point for LTX-2 video generation.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Swagger UI:    http://localhost:8000/docs
ReDoc:         http://localhost:8000/redoc
OpenAPI JSON:  http://localhost:8000/openapi.json
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from . import jobs
from .schemas import GenerateRequest, JobAccepted, JobInfo, JobStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="LTX-2 Video Generation API",
    version="1.0.0",
    description=(
        "HTTP wrapper around LTX-2 pipelines. Submit a generation job, poll for its "
        "status, and download the resulting MP4. Interactive docs at /docs."
    ),
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 once the process is up; does not load the model."""
    return {"status": "ok"}


@app.post(
    "/v1/generate",
    response_model=JobAccepted,
    status_code=202,
    tags=["generate"],
    summary="Submit a video-generation job",
)
def generate(req: GenerateRequest, request: Request) -> JobAccepted:
    info = jobs.submit(req)
    base = str(request.base_url).rstrip("/")
    return JobAccepted(
        job_id=info.job_id,
        status=info.status,
        poll_url=f"{base}/v1/jobs/{info.job_id}",
        download_url=f"{base}/v1/jobs/{info.job_id}/video",
    )


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobInfo,
    tags=["generate"],
    summary="Get job status",
)
def get_job(job_id: str) -> JobInfo:
    info = jobs.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return info


@app.get(
    "/v1/jobs/{job_id}/video",
    tags=["generate"],
    summary="Download the generated MP4",
    responses={
        200: {"content": {"video/mp4": {}}, "description": "MP4 file."},
        404: {"description": "Job unknown."},
        409: {"description": "Job not yet finished or failed."},
    },
)
def get_video(job_id: str) -> FileResponse:
    info = jobs.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if info.status != JobStatus.succeeded:
        raise HTTPException(status_code=409, detail=f"Job is {info.status.value}")
    path = jobs.output_path(job_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Output file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
