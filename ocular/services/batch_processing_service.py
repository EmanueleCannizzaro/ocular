"""
Batch OCR Processing Service with Job Queue and Scheduler

Provides batch file processing capabilities with job queue management,
progress tracking, and scheduled batch jobs.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from collections import deque
import json
import hashlib

from pydantic import BaseModel


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ScheduleType(str, Enum):
    """Schedule type enumeration."""
    ONCE = "once"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CRON = "cron"


class FileItem(BaseModel):
    """Individual file in a batch."""
    file_id: str
    file_path: str
    file_name: str
    file_size: int
    status: JobStatus = JobStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = None


class BatchJob(BaseModel):
    """Batch OCR processing job."""
    job_id: str
    job_name: str
    created_by: Optional[int] = None  # user_id
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: JobStatus
    priority: JobPriority = JobPriority.NORMAL

    # Files
    total_files: int
    files: List[FileItem]

    # Progress
    completed_count: int = 0
    failed_count: int = 0
    progress_percent: float = 0.0

    # Configuration
    provider: str = "mistral"
    strategy: str = "fallback"
    batch_size: int = 5  # Process N files concurrently
    retry_failed: bool = True
    max_retries: int = 3

    # Results
    results_path: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ScheduledJob(BaseModel):
    """Scheduled batch job."""
    schedule_id: str
    schedule_name: str
    created_by: Optional[int] = None
    created_at: datetime

    # Schedule configuration
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]  # cron expression, time, day, etc.
    enabled: bool = True

    # Job configuration
    source_path: str  # Directory or file pattern to process
    destination_path: Optional[str] = None
    provider: str = "mistral"
    strategy: str = "fallback"

    # Execution tracking
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0

    metadata: Dict[str, Any] = {}


class BatchProcessingService:
    """
    Batch OCR processing service with job queue and scheduling.
    """

    def __init__(self, max_concurrent_jobs: int = 3):
        """Initialize batch processing service."""
        self.max_concurrent_jobs = max_concurrent_jobs

        # Job storage (use database in production)
        self.batch_jobs: Dict[str, BatchJob] = {}
        self.scheduled_jobs: Dict[str, ScheduledJob] = {}
        self.job_queue: deque = deque()

        # Active jobs tracking
        self.active_jobs: Dict[str, asyncio.Task] = {}

        # Callbacks
        self.job_callbacks: Dict[str, List[Callable]] = {}

    # ==================== Batch Job Management ====================

    def create_batch_job(
        self,
        job_name: str,
        file_paths: List[str],
        provider: str = "mistral",
        strategy: str = "fallback",
        priority: JobPriority = JobPriority.NORMAL,
        batch_size: int = 5,
        created_by: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """
        Create a new batch OCR job.

        Args:
            job_name: Name for the job
            file_paths: List of file paths to process
            provider: OCR provider to use
            strategy: Processing strategy
            priority: Job priority
            batch_size: Number of files to process concurrently
            created_by: User ID who created the job
            metadata: Additional metadata

        Returns:
            Created batch job
        """
        job_id = str(uuid.uuid4())

        # Create file items
        files = []
        for file_path in file_paths:
            path = Path(file_path)
            file_id = hashlib.md5(file_path.encode()).hexdigest()

            files.append(FileItem(
                file_id=file_id,
                file_path=file_path,
                file_name=path.name,
                file_size=path.stat().st_size if path.exists() else 0,
            ))

        job = BatchJob(
            job_id=job_id,
            job_name=job_name,
            created_by=created_by,
            created_at=datetime.utcnow(),
            status=JobStatus.PENDING,
            priority=priority,
            total_files=len(files),
            files=files,
            provider=provider,
            strategy=strategy,
            batch_size=batch_size,
            metadata=metadata or {},
        )

        self.batch_jobs[job_id] = job

        # Add to queue
        self._add_to_queue(job)

        return job

    def _add_to_queue(self, job: BatchJob):
        """Add job to processing queue based on priority."""
        # Priority order: URGENT -> HIGH -> NORMAL -> LOW
        priority_order = {
            JobPriority.URGENT: 0,
            JobPriority.HIGH: 1,
            JobPriority.NORMAL: 2,
            JobPriority.LOW: 3,
        }

        # Find insertion point based on priority
        job_priority = priority_order[job.priority]
        insert_index = len(self.job_queue)

        for i, queued_job_id in enumerate(self.job_queue):
            queued_job = self.batch_jobs[queued_job_id]
            if priority_order[queued_job.priority] > job_priority:
                insert_index = i
                break

        self.job_queue.insert(insert_index, job.job_id)
        job.status = JobStatus.QUEUED

    async def process_batch_job(self, job_id: str) -> BatchJob:
        """
        Process a batch job.

        Args:
            job_id: Job ID to process

        Returns:
            Updated batch job
        """
        job = self.batch_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()

        try:
            # Process files in batches
            for i in range(0, len(job.files), job.batch_size):
                batch = job.files[i:i + job.batch_size]

                # Process batch concurrently
                tasks = [
                    self._process_single_file(job, file_item)
                    for file_item in batch
                    if file_item.status == JobStatus.PENDING
                ]

                await asyncio.gather(*tasks, return_exceptions=True)

                # Update progress
                job.completed_count = sum(
                    1 for f in job.files
                    if f.status == JobStatus.COMPLETED
                )
                job.failed_count = sum(
                    1 for f in job.files
                    if f.status == JobStatus.FAILED
                )
                job.progress_percent = (
                    (job.completed_count + job.failed_count) / job.total_files * 100
                )

                # Call progress callbacks
                await self._call_callbacks(job, "progress")

            # Job complete
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()

            # Generate results file
            await self._generate_results_file(job)

            # Call completion callbacks
            await self._call_callbacks(job, "complete")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.metadata["error"] = str(e)

            # Call error callbacks
            await self._call_callbacks(job, "error")

        return job

    async def _process_single_file(
        self,
        job: BatchJob,
        file_item: FileItem
    ) -> FileItem:
        """Process a single file in the batch."""
        file_item.status = JobStatus.PROCESSING
        file_item.started_at = datetime.utcnow()

        try:
            # Import here to avoid circular imports
            from ocular.unified_processor import UnifiedDocumentProcessor
            from ocular.settings import get_settings

            settings = get_settings()
            processor = UnifiedDocumentProcessor(settings)

            # Process file
            start_time = datetime.utcnow()

            result = await processor.process_file_async(
                file_path=file_item.file_path,
                provider=job.provider,
                strategy=job.strategy,
            )

            end_time = datetime.utcnow()

            file_item.status = JobStatus.COMPLETED
            file_item.result = {
                "text": result.text,
                "confidence": result.confidence,
                "provider": result.provider,
                "metadata": result.metadata,
            }
            file_item.completed_at = end_time
            file_item.processing_time_ms = (
                (end_time - start_time).total_seconds() * 1000
            )

        except Exception as e:
            file_item.status = JobStatus.FAILED
            file_item.error = str(e)
            file_item.completed_at = datetime.utcnow()

            # Retry logic
            if job.retry_failed and file_item.result is None:
                retries = file_item.metadata.get("retries", 0)
                if retries < job.max_retries:
                    file_item.status = JobStatus.PENDING
                    file_item.metadata["retries"] = retries + 1

        return file_item

    async def _generate_results_file(self, job: BatchJob):
        """Generate results JSON file for the batch job."""
        results = {
            "job_id": job.job_id,
            "job_name": job.job_name,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "total_files": job.total_files,
            "completed_count": job.completed_count,
            "failed_count": job.failed_count,
            "files": [
                {
                    "file_name": f.file_name,
                    "file_path": f.file_path,
                    "status": f.status,
                    "result": f.result,
                    "error": f.error,
                    "processing_time_ms": f.processing_time_ms,
                }
                for f in job.files
            ],
        }

        # Save to file
        results_dir = Path("./batch_results")
        results_dir.mkdir(exist_ok=True)

        results_path = results_dir / f"{job.job_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        job.results_path = str(results_path)

    def get_batch_job(self, job_id: str) -> Optional[BatchJob]:
        """Get batch job by ID."""
        return self.batch_jobs.get(job_id)

    def list_batch_jobs(
        self,
        status: Optional[JobStatus] = None,
        created_by: Optional[int] = None,
        limit: int = 100,
    ) -> List[BatchJob]:
        """List batch jobs with filters."""
        jobs = list(self.batch_jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        if created_by is not None:
            jobs = [j for j in jobs if j.created_by == created_by]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def cancel_batch_job(self, job_id: str) -> BatchJob:
        """Cancel a batch job."""
        job = self.batch_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Cancel active task if running
        if job_id in self.active_jobs:
            self.active_jobs[job_id].cancel()
            del self.active_jobs[job_id]

        # Remove from queue if pending
        if job_id in self.job_queue:
            self.job_queue.remove(job_id)

        job.status = JobStatus.CANCELLED
        return job

    async def pause_batch_job(self, job_id: str) -> BatchJob:
        """Pause a batch job."""
        job = self.batch_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status != JobStatus.PROCESSING:
            raise ValueError("Can only pause processing jobs")

        job.status = JobStatus.PAUSED
        return job

    async def resume_batch_job(self, job_id: str) -> BatchJob:
        """Resume a paused batch job."""
        job = self.batch_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status != JobStatus.PAUSED:
            raise ValueError("Can only resume paused jobs")

        # Reset pending files
        for file_item in job.files:
            if file_item.status == JobStatus.PROCESSING:
                file_item.status = JobStatus.PENDING

        self._add_to_queue(job)
        return job

    # ==================== Job Scheduling ====================

    def create_scheduled_job(
        self,
        schedule_name: str,
        source_path: str,
        schedule_type: ScheduleType,
        schedule_config: Dict[str, Any],
        provider: str = "mistral",
        strategy: str = "fallback",
        destination_path: Optional[str] = None,
        created_by: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledJob:
        """
        Create a scheduled batch job.

        Args:
            schedule_name: Name for the schedule
            source_path: Directory or file pattern to process
            schedule_type: Type of schedule (hourly, daily, etc.)
            schedule_config: Schedule configuration
            provider: OCR provider
            strategy: Processing strategy
            destination_path: Where to save results
            created_by: User ID
            metadata: Additional metadata

        Returns:
            Created scheduled job
        """
        schedule_id = str(uuid.uuid4())

        scheduled_job = ScheduledJob(
            schedule_id=schedule_id,
            schedule_name=schedule_name,
            created_by=created_by,
            created_at=datetime.utcnow(),
            schedule_type=schedule_type,
            schedule_config=schedule_config,
            source_path=source_path,
            destination_path=destination_path,
            provider=provider,
            strategy=strategy,
            metadata=metadata or {},
        )

        # Calculate next run
        scheduled_job.next_run = self._calculate_next_run(
            schedule_type,
            schedule_config
        )

        self.scheduled_jobs[schedule_id] = scheduled_job

        return scheduled_job

    def _calculate_next_run(
        self,
        schedule_type: ScheduleType,
        schedule_config: Dict[str, Any]
    ) -> datetime:
        """Calculate next run time based on schedule."""
        now = datetime.utcnow()

        if schedule_type == ScheduleType.ONCE:
            # Run at specific time
            run_at = datetime.fromisoformat(schedule_config["run_at"])
            return run_at if run_at > now else now

        elif schedule_type == ScheduleType.HOURLY:
            # Run every N hours
            interval_hours = schedule_config.get("interval_hours", 1)
            return now + timedelta(hours=interval_hours)

        elif schedule_type == ScheduleType.DAILY:
            # Run daily at specific time
            hour = schedule_config.get("hour", 0)
            minute = schedule_config.get("minute", 0)

            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)

            return next_run

        elif schedule_type == ScheduleType.WEEKLY:
            # Run weekly on specific day and time
            day_of_week = schedule_config.get("day_of_week", 0)  # 0=Monday
            hour = schedule_config.get("hour", 0)
            minute = schedule_config.get("minute", 0)

            days_ahead = day_of_week - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7

            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )

            return next_run

        elif schedule_type == ScheduleType.MONTHLY:
            # Run monthly on specific day and time
            day_of_month = schedule_config.get("day_of_month", 1)
            hour = schedule_config.get("hour", 0)
            minute = schedule_config.get("minute", 0)

            next_run = now.replace(
                day=day_of_month, hour=hour, minute=minute,
                second=0, microsecond=0
            )

            if next_run <= now:
                # Next month
                if next_run.month == 12:
                    next_run = next_run.replace(year=next_run.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=next_run.month + 1)

            return next_run

        elif schedule_type == ScheduleType.CRON:
            # Parse cron expression (simplified)
            # Format: "minute hour day month day_of_week"
            # For full cron support, use croniter library
            cron_expr = schedule_config.get("cron_expression", "0 0 * * *")
            # For now, return next hour (implement full cron parsing)
            return now + timedelta(hours=1)

        return now

    async def execute_scheduled_job(self, schedule_id: str) -> BatchJob:
        """Execute a scheduled job."""
        scheduled = self.scheduled_jobs.get(schedule_id)
        if not scheduled:
            raise ValueError(f"Scheduled job {schedule_id} not found")

        # Find files to process
        source_path = Path(scheduled.source_path)

        if source_path.is_file():
            file_paths = [str(source_path)]
        elif source_path.is_dir():
            # Process all files in directory
            file_paths = [
                str(f) for f in source_path.rglob("*")
                if f.is_file() and f.suffix in [".pdf", ".png", ".jpg", ".jpeg"]
            ]
        else:
            # Glob pattern
            file_paths = [str(f) for f in Path().glob(scheduled.source_path)]

        # Create batch job
        job = self.create_batch_job(
            job_name=f"{scheduled.schedule_name} - {datetime.utcnow().isoformat()}",
            file_paths=file_paths,
            provider=scheduled.provider,
            strategy=scheduled.strategy,
            created_by=scheduled.created_by,
            metadata={
                "schedule_id": schedule_id,
                "scheduled": True,
            },
        )

        # Update scheduled job tracking
        scheduled.last_run = datetime.utcnow()
        scheduled.total_runs += 1

        # Calculate next run
        scheduled.next_run = self._calculate_next_run(
            scheduled.schedule_type,
            scheduled.schedule_config
        )

        return job

    async def check_scheduled_jobs(self):
        """Check and execute due scheduled jobs."""
        now = datetime.utcnow()

        for scheduled in self.scheduled_jobs.values():
            if not scheduled.enabled:
                continue

            if scheduled.next_run and scheduled.next_run <= now:
                try:
                    await self.execute_scheduled_job(scheduled.schedule_id)
                    scheduled.successful_runs += 1
                except Exception as e:
                    scheduled.failed_runs += 1
                    scheduled.metadata["last_error"] = str(e)

    # ==================== Callback Management ====================

    def add_callback(
        self,
        job_id: str,
        event: str,
        callback: Callable
    ):
        """Add callback for job events (progress, complete, error)."""
        key = f"{job_id}:{event}"
        if key not in self.job_callbacks:
            self.job_callbacks[key] = []
        self.job_callbacks[key].append(callback)

    async def _call_callbacks(self, job: BatchJob, event: str):
        """Call registered callbacks for job event."""
        key = f"{job.job_id}:{event}"
        callbacks = self.job_callbacks.get(key, [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(job)
                else:
                    callback(job)
            except Exception as e:
                print(f"Callback error: {e}")


# Global batch processing service instance
_batch_service: Optional[BatchProcessingService] = None


def get_batch_processing_service() -> BatchProcessingService:
    """Get or create the global batch processing service instance."""
    global _batch_service
    if _batch_service is None:
        _batch_service = BatchProcessingService()
    return _batch_service
