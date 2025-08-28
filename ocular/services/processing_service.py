"""
Processing coordination service for Ocular OCR system.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime

from ..core.models import ProcessingRequest, ProcessingResult, BatchProcessingRequest
from ..core.enums import ProcessingStatus, ProcessingStrategy
from ..core.exceptions import OCRError, ValidationError
from ..providers.settings import OcularSettings
from .ocr_service import OCRService

logger = logging.getLogger(__name__)


class ProcessingService:
    """Service for managing OCR processing workflows."""
    
    def __init__(self, settings: OcularSettings):
        self.settings = settings
        self.ocr_service = OCRService(settings)
        self._active_jobs: Dict[str, ProcessingJob] = {}
        self._job_counter = 0
        
    async def submit_processing_job(
        self, 
        request: ProcessingRequest,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit a processing job and return job ID."""
        
        job_id = self._generate_job_id()
        job = ProcessingJob(
            job_id=job_id,
            request=request,
            callback=callback,
            submitted_at=datetime.now()
        )
        
        self._active_jobs[job_id] = job
        
        # Start processing in background
        asyncio.create_task(self._process_job(job))
        
        logger.info(f"Submitted processing job {job_id} for {request.file_path}")
        return job_id
    
    async def submit_batch_job(
        self,
        batch_request: BatchProcessingRequest,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit a batch processing job and return job ID."""
        
        job_id = self._generate_job_id()
        
        # Convert batch request to individual requests
        individual_requests = []
        for file_path in batch_request.file_paths:
            individual_request = ProcessingRequest(
                file_path=file_path,
                strategy=batch_request.strategy,
                providers=batch_request.providers,
                prompt=batch_request.prompt,
                options=batch_request.options
            )
            individual_requests.append(individual_request)
        
        job = BatchProcessingJob(
            job_id=job_id,
            requests=individual_requests,
            max_concurrent=batch_request.max_concurrent,
            callback=callback,
            submitted_at=datetime.now()
        )
        
        self._active_jobs[job_id] = job
        
        # Start batch processing in background
        asyncio.create_task(self._process_batch_job(job))
        
        logger.info(f"Submitted batch job {job_id} with {len(individual_requests)} files")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a processing job."""
        
        job = self._active_jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "submitted_at": job.submitted_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "progress": job.get_progress(),
            "error": job.error,
            "result_count": len(job.results) if hasattr(job, 'results') else (1 if job.result else 0)
        }
    
    async def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get result of a completed processing job."""
        
        job = self._active_jobs.get(job_id)
        if not job or job.status != ProcessingStatus.COMPLETED:
            return None
        
        if isinstance(job, BatchProcessingJob):
            return job.results
        else:
            return job.result
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job."""
        
        job = self._active_jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            return False
        
        job.status = ProcessingStatus.CANCELLED
        logger.info(f"Cancelled processing job {job_id}")
        return True
    
    async def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all active jobs."""
        
        active_jobs = []
        for job_id, job in self._active_jobs.items():
            if job.status in [ProcessingStatus.PENDING, ProcessingStatus.IN_PROGRESS]:
                status = await self.get_job_status(job_id)
                if status:
                    active_jobs.append(status)
        
        return active_jobs
    
    async def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old completed jobs."""
        
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        jobs_to_remove = []
        
        for job_id, job in self._active_jobs.items():
            if (job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED] and
                job.completed_at and job.completed_at.timestamp() < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self._active_jobs[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
        return len(jobs_to_remove)
    
    async def _process_job(self, job: 'ProcessingJob') -> None:
        """Process a single job."""
        
        try:
            job.status = ProcessingStatus.IN_PROGRESS
            job.started_at = datetime.now()
            
            # Process the request
            result = await self.ocr_service.process_document(job.request)
            
            job.result = result
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Call callback if provided
            if job.callback:
                try:
                    await job.callback(job.job_id, result)
                except Exception as e:
                    logger.error(f"Error in job callback: {e}")
            
            logger.info(f"Completed processing job {job.job_id}")
            
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            
            logger.error(f"Processing job {job.job_id} failed: {e}")
    
    async def _process_batch_job(self, job: 'BatchProcessingJob') -> None:
        """Process a batch job."""
        
        try:
            job.status = ProcessingStatus.IN_PROGRESS
            job.started_at = datetime.now()
            
            # Process batch with concurrency control
            results = await self.ocr_service.process_batch(job.requests)
            
            job.results = results
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Call callback if provided
            if job.callback:
                try:
                    await job.callback(job.job_id, results)
                except Exception as e:
                    logger.error(f"Error in batch job callback: {e}")
            
            logger.info(f"Completed batch job {job.job_id}")
            
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            
            logger.error(f"Batch job {job.job_id} failed: {e}")
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        self._job_counter += 1
        timestamp = int(datetime.now().timestamp())
        return f"job_{timestamp}_{self._job_counter:04d}"
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        
        stats = {
            "total_jobs": len(self._active_jobs),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }
        
        for job in self._active_jobs.values():
            if job.status == ProcessingStatus.PENDING:
                stats["pending"] += 1
            elif job.status == ProcessingStatus.IN_PROGRESS:
                stats["in_progress"] += 1
            elif job.status == ProcessingStatus.COMPLETED:
                stats["completed"] += 1
            elif job.status == ProcessingStatus.FAILED:
                stats["failed"] += 1
            elif job.status == ProcessingStatus.CANCELLED:
                stats["cancelled"] += 1
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown processing service."""
        
        # Cancel all pending jobs
        for job in self._active_jobs.values():
            if job.status in [ProcessingStatus.PENDING, ProcessingStatus.IN_PROGRESS]:
                job.status = ProcessingStatus.CANCELLED
        
        # Cleanup OCR service
        await self.ocr_service.cleanup()
        
        logger.info("Processing service shutdown completed")


class ProcessingJob:
    """Represents a single processing job."""
    
    def __init__(
        self,
        job_id: str,
        request: ProcessingRequest,
        callback: Optional[Callable] = None,
        submitted_at: Optional[datetime] = None
    ):
        self.job_id = job_id
        self.request = request
        self.callback = callback
        self.submitted_at = submitted_at or datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status = ProcessingStatus.PENDING
        self.result: Optional[ProcessingResult] = None
        self.error: Optional[str] = None
    
    def get_progress(self) -> Dict[str, Any]:
        """Get job progress information."""
        
        progress = {"percentage": 0}
        
        if self.status == ProcessingStatus.PENDING:
            progress["percentage"] = 0
        elif self.status == ProcessingStatus.IN_PROGRESS:
            progress["percentage"] = 50  # Halfway through
        elif self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]:
            progress["percentage"] = 100
        
        if self.started_at:
            elapsed = (datetime.now() - self.started_at).total_seconds()
            progress["elapsed_seconds"] = elapsed
        
        return progress


class BatchProcessingJob:
    """Represents a batch processing job."""
    
    def __init__(
        self,
        job_id: str,
        requests: List[ProcessingRequest],
        max_concurrent: int = 3,
        callback: Optional[Callable] = None,
        submitted_at: Optional[datetime] = None
    ):
        self.job_id = job_id
        self.requests = requests
        self.max_concurrent = max_concurrent
        self.callback = callback
        self.submitted_at = submitted_at or datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status = ProcessingStatus.PENDING
        self.results: List[ProcessingResult] = []
        self.error: Optional[str] = None
    
    def get_progress(self) -> Dict[str, Any]:
        """Get batch job progress information."""
        
        total_files = len(self.requests)
        completed_files = len(self.results)
        
        progress = {
            "total_files": total_files,
            "completed_files": completed_files,
            "percentage": (completed_files / total_files * 100) if total_files > 0 else 0
        }
        
        if self.started_at:
            elapsed = (datetime.now() - self.started_at).total_seconds()
            progress["elapsed_seconds"] = elapsed
            
            if completed_files > 0:
                avg_time_per_file = elapsed / completed_files
                estimated_remaining = (total_files - completed_files) * avg_time_per_file
                progress["estimated_remaining_seconds"] = estimated_remaining
        
        return progress