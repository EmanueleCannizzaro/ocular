"""
Batch OCR Processing API endpoints
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from pydantic import BaseModel

from ocular.routers import templates
from ocular.services.batch_processing_service import (
    get_batch_processing_service,
    JobStatus,
    JobPriority,
    ScheduleType,
    BatchJob,
    ScheduledJob,
)


router = APIRouter()


# ==================== Request/Response Models ====================

class BatchJobCreateRequest(BaseModel):
    """Create batch job request."""
    job_name: str
    file_paths: List[str]
    provider: str = "mistral"
    strategy: str = "fallback"
    priority: JobPriority = JobPriority.NORMAL
    batch_size: int = 5
    retry_failed: bool = True
    max_retries: int = 3


class ScheduledJobCreateRequest(BaseModel):
    """Create scheduled job request."""
    schedule_name: str
    source_path: str
    schedule_type: ScheduleType
    schedule_config: dict
    provider: str = "mistral"
    strategy: str = "fallback"
    destination_path: Optional[str] = None
    enabled: bool = True


# ==================== Batch Job Endpoints ====================

@router.post("/jobs", response_model=BatchJob)
async def create_batch_job(request: BatchJobCreateRequest):
    """
    Create a new batch OCR job.

    The job will be queued and processed based on priority.
    """
    service = get_batch_processing_service()

    job = service.create_batch_job(
        job_name=request.job_name,
        file_paths=request.file_paths,
        provider=request.provider,
        strategy=request.strategy,
        priority=request.priority,
        batch_size=request.batch_size,
    )

    # Start processing asynchronously
    import asyncio
    asyncio.create_task(service.process_batch_job(job.job_id))

    return job


@router.get("/jobs/{job_id}", response_model=BatchJob)
async def get_batch_job(job_id: str):
    """Get batch job details."""
    service = get_batch_processing_service()
    job = service.get_batch_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.get("/jobs", response_model=List[BatchJob])
async def list_batch_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 100,
):
    """List batch jobs with optional filters."""
    service = get_batch_processing_service()
    jobs = service.list_batch_jobs(status=status, limit=limit)
    return jobs


@router.post("/jobs/{job_id}/cancel")
async def cancel_batch_job(job_id: str):
    """Cancel a batch job."""
    service = get_batch_processing_service()

    try:
        job = await service.cancel_batch_job(job_id)
        return {"success": True, "job": job}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/jobs/{job_id}/pause")
async def pause_batch_job(job_id: str):
    """Pause a running batch job."""
    service = get_batch_processing_service()

    try:
        job = await service.pause_batch_job(job_id)
        return {"success": True, "job": job}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/resume")
async def resume_batch_job(job_id: str):
    """Resume a paused batch job."""
    service = get_batch_processing_service()

    try:
        job = await service.resume_batch_job(job_id)
        return {"success": True, "job": job}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/results")
async def get_batch_job_results(job_id: str):
    """Get batch job results file."""
    from fastapi.responses import FileResponse

    service = get_batch_processing_service()
    job = service.get_batch_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.results_path:
        raise HTTPException(status_code=404, detail="Results not available")

    return FileResponse(
        job.results_path,
        media_type="application/json",
        filename=f"{job.job_name}_results.json"
    )


# ==================== Scheduled Jobs Endpoints ====================

@router.post("/schedules", response_model=ScheduledJob)
async def create_scheduled_job(request: ScheduledJobCreateRequest):
    """Create a scheduled batch job."""
    service = get_batch_processing_service()

    scheduled = service.create_scheduled_job(
        schedule_name=request.schedule_name,
        source_path=request.source_path,
        schedule_type=request.schedule_type,
        schedule_config=request.schedule_config,
        provider=request.provider,
        strategy=request.strategy,
        destination_path=request.destination_path,
    )

    return scheduled


@router.get("/schedules", response_model=List[ScheduledJob])
async def list_scheduled_jobs():
    """List all scheduled jobs."""
    service = get_batch_processing_service()
    return list(service.scheduled_jobs.values())


@router.get("/schedules/{schedule_id}", response_model=ScheduledJob)
async def get_scheduled_job(schedule_id: str):
    """Get scheduled job details."""
    service = get_batch_processing_service()
    scheduled = service.scheduled_jobs.get(schedule_id)

    if not scheduled:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    return scheduled


@router.post("/schedules/{schedule_id}/execute")
async def execute_scheduled_job(schedule_id: str):
    """Manually execute a scheduled job."""
    service = get_batch_processing_service()

    try:
        job = await service.execute_scheduled_job(schedule_id)
        return {"success": True, "job": job}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/schedules/{schedule_id}")
async def delete_scheduled_job(schedule_id: str):
    """Delete a scheduled job."""
    service = get_batch_processing_service()

    if schedule_id not in service.scheduled_jobs:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    del service.scheduled_jobs[schedule_id]

    return {"success": True, "message": "Scheduled job deleted"}


@router.post("/schedules/{schedule_id}/enable")
async def enable_scheduled_job(schedule_id: str):
    """Enable a scheduled job."""
    service = get_batch_processing_service()
    scheduled = service.scheduled_jobs.get(schedule_id)

    if not scheduled:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    scheduled.enabled = True
    return {"success": True, "scheduled": scheduled}


@router.post("/schedules/{schedule_id}/disable")
async def disable_scheduled_job(schedule_id: str):
    """Disable a scheduled job."""
    service = get_batch_processing_service()
    scheduled = service.scheduled_jobs.get(schedule_id)

    if not scheduled:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    scheduled.enabled = False
    return {"success": True, "scheduled": scheduled}


# ==================== Batch Upload Endpoint ====================

@router.post("/upload-batch")
async def upload_batch_files(
    files: List[UploadFile] = File(...),
    job_name: str = Form(...),
    provider: str = Form("mistral"),
    strategy: str = Form("fallback"),
    priority: str = Form("normal"),
):
    """
    Upload multiple files for batch OCR processing.

    This endpoint accepts multiple files, saves them temporarily,
    and creates a batch job to process them.
    """
    import tempfile
    from pathlib import Path

    # Save uploaded files
    temp_dir = Path(tempfile.mkdtemp(prefix="batch_ocr_"))
    file_paths = []

    for file in files:
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        file_paths.append(str(file_path))

    # Create batch job
    service = get_batch_processing_service()

    job = service.create_batch_job(
        job_name=job_name,
        file_paths=file_paths,
        provider=provider,
        strategy=strategy,
        priority=JobPriority(priority),
    )

    # Start processing
    import asyncio
    asyncio.create_task(service.process_batch_job(job.job_id))

    return {
        "success": True,
        "job_id": job.job_id,
        "total_files": len(file_paths),
        "message": f"Batch job created with {len(file_paths)} files"
    }
