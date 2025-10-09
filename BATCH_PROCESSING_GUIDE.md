# Batch Processing & Job Queue Guide

**Version**: 3.0
**Feature Status**: ✅ Fully Implemented

---

## 📦 Overview

The Ocular OCR Platform includes a comprehensive batch processing system with job queue management and scheduling capabilities. This guide shows you how to access and use these features.

---

## 🚀 Quick Start

### 1. Start the Application

```bash
# From the project root
cd /mnt/developer/git/aecs4u.it/ocular

# Activate virtual environment
source .venv/bin/activate

# Start the web application
uvicorn ocular.app:app --host 0.0.0.0 --port 8000 --reload
```

The batch processing API will be available at: **http://localhost:8000/api/batch**

### 2. Access the API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Look for the **"batch-processing"** tag to see all 15 batch endpoints.

---

## 📋 Available Endpoints

### Batch Job Management (8 Endpoints)

#### 1. Create Batch Job
```http
POST /api/batch/jobs
Content-Type: application/json

{
  "job_name": "Invoice Processing Batch",
  "file_paths": [
    "/path/to/file1.pdf",
    "/path/to/file2.pdf",
    "/path/to/file3.pdf"
  ],
  "provider": "mistral",
  "strategy": "fallback",
  "priority": "NORMAL",
  "batch_size": 5
}
```

**Response**:
```json
{
  "job_id": "batch_abc123",
  "job_name": "Invoice Processing Batch",
  "status": "PENDING",
  "total_files": 3,
  "created_at": "2025-10-06T10:00:00Z",
  "priority": "NORMAL"
}
```

#### 2. List All Jobs
```http
GET /api/batch/jobs
```

#### 3. Get Job Details
```http
GET /api/batch/jobs/{job_id}
```

#### 4. Start Job Processing
```http
POST /api/batch/jobs/{job_id}/start
```

#### 5. Cancel Job
```http
POST /api/batch/jobs/{job_id}/cancel
```

#### 6. Pause Job
```http
POST /api/batch/jobs/{job_id}/pause
```

#### 7. Resume Job
```http
POST /api/batch/jobs/{job_id}/resume
```

#### 8. Upload Files for Batch Processing
```http
POST /api/batch/upload-batch
Content-Type: multipart/form-data

files: [file1.pdf, file2.pdf, ...]
job_name: "My Batch Job"
provider: "mistral"
strategy: "fallback"
priority: "HIGH"
```

### Scheduled Jobs (7 Endpoints)

#### 1. Create Scheduled Job
```http
POST /api/batch/schedules
Content-Type: application/json

{
  "schedule_name": "Daily Invoice Processing",
  "schedule_type": "DAILY",
  "schedule_config": {
    "time": "09:00",
    "timezone": "UTC"
  },
  "source_path": "/data/invoices/*.pdf",
  "provider": "mistral",
  "strategy": "fallback"
}
```

**Schedule Types**:
- `ONCE` - Run once at specified time
- `HOURLY` - Run every N hours
- `DAILY` - Run daily at specified time
- `WEEKLY` - Run weekly on specified day
- `MONTHLY` - Run monthly on specified date
- `CRON` - Custom cron expression

#### 2. List Scheduled Jobs
```http
GET /api/batch/schedules
```

#### 3. Get Schedule Details
```http
GET /api/batch/schedules/{schedule_id}
```

#### 4. Update Schedule
```http
PUT /api/batch/schedules/{schedule_id}
Content-Type: application/json

{
  "enabled": true,
  "schedule_config": {
    "time": "10:00"
  }
}
```

#### 5. Enable Schedule
```http
POST /api/batch/schedules/{schedule_id}/enable
```

#### 6. Disable Schedule
```http
POST /api/batch/schedules/{schedule_id}/disable
```

#### 7. Delete Schedule
```http
DELETE /api/batch/schedules/{schedule_id}
```

---

## 💡 Usage Examples

### Example 1: Process a Batch of PDFs

```python
import requests

# Create a batch job
response = requests.post('http://localhost:8000/api/batch/jobs', json={
    "job_name": "Q4 Invoices",
    "file_paths": [
        "/data/invoice_001.pdf",
        "/data/invoice_002.pdf",
        "/data/invoice_003.pdf",
        "/data/invoice_004.pdf",
        "/data/invoice_005.pdf"
    ],
    "provider": "mistral",
    "strategy": "fallback",
    "priority": "HIGH",
    "batch_size": 3  # Process 3 files concurrently
})

job = response.json()
job_id = job['job_id']

# Start the job
requests.post(f'http://localhost:8000/api/batch/jobs/{job_id}/start')

# Check job status
status_response = requests.get(f'http://localhost:8000/api/batch/jobs/{job_id}')
job_status = status_response.json()

print(f"Status: {job_status['status']}")
print(f"Progress: {job_status['progress_percent']}%")
print(f"Completed: {job_status['completed_count']}/{job_status['total_files']}")
```

### Example 2: Schedule Daily OCR Processing

```python
import requests

# Create a daily scheduled job
response = requests.post('http://localhost:8000/api/batch/schedules', json={
    "schedule_name": "Daily Document Processing",
    "schedule_type": "DAILY",
    "schedule_config": {
        "time": "02:00",  # Run at 2 AM
        "timezone": "UTC"
    },
    "source_path": "/data/incoming/*.pdf",
    "destination_path": "/data/processed/",
    "provider": "google_vision",
    "strategy": "single",
    "enabled": True
})

schedule = response.json()
print(f"Scheduled job created: {schedule['schedule_id']}")
print(f"Next run: {schedule['next_run']}")
```

### Example 3: Upload Files for Batch Processing

```python
import requests

# Prepare files
files = [
    ('files', open('doc1.pdf', 'rb')),
    ('files', open('doc2.pdf', 'rb')),
    ('files', open('doc3.pdf', 'rb'))
]

data = {
    'job_name': 'Uploaded Documents',
    'provider': 'mistral',
    'strategy': 'fallback',
    'priority': 'NORMAL'
}

# Upload and create batch job
response = requests.post(
    'http://localhost:8000/api/batch/upload-batch',
    files=files,
    data=data
)

result = response.json()
print(f"Batch job created: {result['job_id']}")
print(f"Processing {result['total_files']} files")
```

### Example 4: Monitor Job Progress

```python
import requests
import time

def monitor_job(job_id):
    """Monitor job progress until completion."""
    while True:
        response = requests.get(f'http://localhost:8000/api/batch/jobs/{job_id}')
        job = response.json()

        status = job['status']
        progress = job['progress_percent']

        print(f"Status: {status} | Progress: {progress:.1f}%")

        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            break

        time.sleep(5)  # Check every 5 seconds

    return job

# Monitor a job
job_id = "batch_abc123"
final_status = monitor_job(job_id)

print(f"\nFinal Status: {final_status['status']}")
print(f"Completed Files: {final_status['completed_count']}")
print(f"Failed Files: {final_status['failed_count']}")
```

---

## 🎯 Job Priority System

Jobs are processed based on priority:

1. **URGENT** - Processed immediately, highest priority
2. **HIGH** - Processed before normal jobs
3. **NORMAL** - Standard processing priority (default)
4. **LOW** - Processed when system is idle

```python
# Create high-priority job
response = requests.post('http://localhost:8000/api/batch/jobs', json={
    "job_name": "Critical Documents",
    "file_paths": [...],
    "provider": "mistral",
    "priority": "URGENT"  # Will be processed first
})
```

---

## 📅 Schedule Configuration Examples

### Hourly Schedule
```json
{
  "schedule_type": "HOURLY",
  "schedule_config": {
    "interval_hours": 2  // Every 2 hours
  }
}
```

### Daily Schedule
```json
{
  "schedule_type": "DAILY",
  "schedule_config": {
    "time": "14:30",  // 2:30 PM
    "timezone": "America/New_York"
  }
}
```

### Weekly Schedule
```json
{
  "schedule_type": "WEEKLY",
  "schedule_config": {
    "day_of_week": "monday",
    "time": "09:00"
  }
}
```

### Monthly Schedule
```json
{
  "schedule_type": "MONTHLY",
  "schedule_config": {
    "day_of_month": 1,  // First day of month
    "time": "00:00"
  }
}
```

### CRON Schedule
```json
{
  "schedule_type": "CRON",
  "schedule_config": {
    "cron_expression": "0 */6 * * *"  // Every 6 hours
  }
}
```

---

## 🔍 Job Status Flow

```
PENDING → QUEUED → PROCESSING → COMPLETED
                              ↘ FAILED
                              ↘ CANCELLED
                              ↘ PAUSED
```

### Status Descriptions

- **PENDING**: Job created, waiting to be queued
- **QUEUED**: Job in queue, waiting for processing
- **PROCESSING**: Job currently being processed
- **COMPLETED**: Job finished successfully
- **FAILED**: Job failed with errors
- **CANCELLED**: Job cancelled by user
- **PAUSED**: Job paused by user (can be resumed)

---

## 📊 Progress Tracking

Each job tracks:
- **Total files**: Number of files to process
- **Completed count**: Successfully processed files
- **Failed count**: Files that failed processing
- **Progress percent**: Completion percentage (0-100%)
- **Processing time**: Time elapsed since start
- **Estimated completion**: Estimated time to complete (if available)

```python
# Get detailed progress
response = requests.get(f'http://localhost:8000/api/batch/jobs/{job_id}')
job = response.json()

print(f"Total: {job['total_files']}")
print(f"Completed: {job['completed_count']}")
print(f"Failed: {job['failed_count']}")
print(f"Progress: {job['progress_percent']:.1f}%")

# List individual file statuses
for file_item in job['files']:
    print(f"{file_item['file_path']}: {file_item['status']}")
```

---

## ⚡ Performance Tips

### 1. Optimize Batch Size
```python
# For small files (< 1MB): batch_size = 10-20
# For medium files (1-5MB): batch_size = 5-10
# For large files (> 5MB): batch_size = 2-5

{
  "batch_size": 10,  # Process 10 files concurrently
  "file_paths": [...]
}
```

### 2. Choose Appropriate Strategy
- `single`: Fastest, single provider
- `fallback`: Reliable, tries backup providers
- `ensemble`: Most accurate, combines multiple providers
- `best`: Highest quality, selects best result

### 3. Use Priority for Critical Jobs
```python
# Critical jobs
"priority": "URGENT"

# Regular jobs
"priority": "NORMAL"

# Background jobs
"priority": "LOW"
```

### 4. Enable Retry for Reliability
```python
{
  "retry_failed": True,
  "max_retries": 3  # Retry failed files up to 3 times
}
```

---

## 🛠️ Troubleshooting

### Job Stuck in QUEUED Status
- Check if job was started: `POST /api/batch/jobs/{job_id}/start`
- Verify system resources are available
- Check logs for errors

### Job Fails Immediately
- Verify file paths exist and are accessible
- Check provider API keys are configured
- Ensure files are in supported formats

### Schedule Not Running
- Verify schedule is enabled: `schedule['enabled'] == True`
- Check `next_run` time is in the future
- Verify source_path pattern is correct

### High Failure Rate
- Check provider availability and API limits
- Verify file quality (not corrupted)
- Try different provider or strategy
- Enable retry: `retry_failed: True`

---

## 📈 Best Practices

1. **Use Schedules for Recurring Tasks**
   - Daily document processing
   - Periodic batch imports
   - Regular data extraction

2. **Monitor Job Progress**
   - Implement progress callbacks
   - Track completion rates
   - Alert on failures

3. **Optimize for Cost**
   - Use appropriate batch sizes
   - Choose cost-effective providers
   - Schedule during off-peak hours

4. **Handle Errors Gracefully**
   - Enable retry for transient failures
   - Implement error notifications
   - Log failed files for manual review

5. **Clean Up Completed Jobs**
   - Archive old job records
   - Export results regularly
   - Remove temporary files

---

## 🔗 Related Features

### Token Optimization
Reduce costs by optimizing images and prompts before batch processing:
- See: `ocular/services/optimization_service.py`
- Endpoint: `/api/optimize` (if implemented)

### Analytics
Track batch processing metrics:
- Visit: http://localhost:8000/analytics
- API: `/api/analytics/batch-stats` (if implemented)

### Cost Tracking
Monitor batch processing costs:
- Visit: http://localhost:8000/billing
- API: `/api/billing/batch-costs` (if implemented)

---

## 📚 Additional Resources

- **Implementation**: `ocular/services/batch_processing_service.py`
- **API Routes**: `ocular/routers/batch.py`
- **Documentation**: `FINAL_COMPLETION_REPORT.md`
- **Examples**: `examples/batch_processing_example.py` (if available)

---

## 🎉 Summary

The Batch Processing & Job Queue system provides:

✅ **15 API Endpoints** for complete batch management
✅ **Priority Queue** with 4 priority levels
✅ **6 Schedule Types** including CRON support
✅ **Progress Tracking** with detailed status
✅ **Automatic Retry** for failed files
✅ **Concurrent Processing** with configurable batch size
✅ **Job Lifecycle Management** (pause, resume, cancel)

**Start using batch processing today to process thousands of documents efficiently!**

---

**Last Updated**: 2025-10-06
**Version**: 3.0
**Status**: Production Ready ✅
