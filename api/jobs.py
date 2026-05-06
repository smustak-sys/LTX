"""In-process job store with a single background worker.

Generation is GPU-bound and not safely concurrent on one device, so jobs run
serially on a single worker thread. Swap in Redis/RQ/Celery if you need more.
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback
import uuid
from pathlib import Path

from .config import settings
from .runner import run_generation
from .schemas import GenerateRequest, JobInfo, JobStatus

logger = logging.getLogger(__name__)

_jobs: dict[str, JobInfo] = {}
_jobs_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None:
                    continue
                job.status = JobStatus.running

            output_path = settings.output_dir / f"{job_id}.mp4"
            run_generation(job.request, output_path)

            with _jobs_lock:
                job.status = JobStatus.succeeded
                job.output_path = str(output_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is not None:
                    job.status = JobStatus.failed
                    job.error = f"{exc}\n{traceback.format_exc()}"
        finally:
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=_worker_loop, name="ltx2-worker", daemon=True)
            t.start()
            _worker_started = True


def submit(req: GenerateRequest) -> JobInfo:
    _ensure_worker()
    job_id = uuid.uuid4().hex
    info = JobInfo(job_id=job_id, status=JobStatus.queued, request=req)
    with _jobs_lock:
        _jobs[job_id] = info
    _queue.put(job_id)
    return info


def get(job_id: str) -> JobInfo | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def output_path(job_id: str) -> Path | None:
    info = get(job_id)
    if info is None or info.output_path is None:
        return None
    return Path(info.output_path)
