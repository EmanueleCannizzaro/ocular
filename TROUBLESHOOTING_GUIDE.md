# 🔧 Ocular OCR Platform - Troubleshooting Guide

**Version**: 3.0
**Last Updated**: 2025-10-06

Comprehensive troubleshooting guide for common issues and their solutions.

---

## 📋 Table of Contents

1. [Installation Issues](#installation-issues)
2. [Configuration Issues](#configuration-issues)
3. [OCR Processing Issues](#ocr-processing-issues)
4. [Batch Processing Issues](#batch-processing-issues)
5. [Deployment Issues](#deployment-issues)
6. [Database Issues](#database-issues)
7. [Authentication Issues](#authentication-issues)
8. [Performance Issues](#performance-issues)
9. [API Issues](#api-issues)
10. [Common Error Messages](#common-error-messages)

---

## 🔨 Installation Issues

### Issue: `ModuleNotFoundError: No module named 'ocular'`

**Cause**: Package not installed or incorrect Python environment

**Solution**:
```bash
# Ensure you're in the correct virtual environment
source .venv/bin/activate

# Install in development mode
uv add -e .

# Or with pip
pip install -e .

# Verify installation
python -c "import ocular; print(ocular.__version__)"
```

---

### Issue: `uv: command not found`

**Cause**: uv package manager not installed

**Solution**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Verify
uv --version
```

---

### Issue: Dependency conflicts during installation

**Cause**: Incompatible package versions

**Solution**:
```bash
# Clear pip cache
pip cache purge

# Remove existing virtual environment
rm -rf .venv

# Create fresh environment
python -m venv .venv
source .venv/bin/activate

# Install with locked versions
pip install -r requirements.txt --no-cache-dir

# If using uv
uv sync --locked
```

---

## ⚙️ Configuration Issues

### Issue: `No OCR providers available`

**Cause**: Missing API keys or environment variables

**Solution**:
```bash
# 1. Check .env file exists
ls -la .env

# 2. Verify required variables
cat .env | grep -E "MISTRAL_API_KEY|GOOGLE_APPLICATION_CREDENTIALS"

# 3. Copy from example if missing
cp .env.example .env

# 4. Edit with your API keys
nano .env

# 5. Required variables:
# MISTRAL_API_KEY=your_mistral_key
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
# AWS_ACCESS_KEY_ID=your_aws_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret
```

**Verification**:
```python
from ocular.settings import get_settings

settings = get_settings()
print(f"Mistral: {bool(settings.MISTRAL_API_KEY)}")
print(f"Google: {bool(settings.GOOGLE_APPLICATION_CREDENTIALS)}")
```

---

### Issue: `Settings validation error`

**Cause**: Invalid configuration values

**Solution**:
```bash
# Run settings validation
python -c "from ocular.settings import get_settings; get_settings()"

# Check for common issues:
# - Invalid URLs (must include protocol: http://, https://)
# - Invalid file paths (must exist if specified)
# - Invalid port numbers (1-65535)
# - Invalid enum values

# Example fix:
# Wrong: DATABASE_URL=localhost:5432/db
# Right: DATABASE_URL=postgresql://localhost:5432/db
```

---

### Issue: `RuntimeError: Could not load .env file`

**Cause**: `.env` file permissions or encoding issue

**Solution**:
```bash
# Check file permissions
ls -l .env

# Make readable
chmod 644 .env

# Check encoding (should be UTF-8)
file .env

# Convert if needed
iconv -f ISO-8859-1 -t UTF-8 .env > .env.utf8
mv .env.utf8 .env
```

---

## 📄 OCR Processing Issues

### Issue: `PDF processing fails with "pdf2image not found"`

**Cause**: Missing PDF processing dependencies

**Solution**:
```bash
# Install pdf2image
uv add pdf2image

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get install poppler-utils

# macOS:
brew install poppler

# Verify
python -c "from pdf2image import convert_from_path; print('✅ PDF support OK')"
```

---

### Issue: OCR returns empty text

**Cause**: Poor image quality or unsupported format

**Solution**:
```python
# 1. Check image quality
from PIL import Image

img = Image.open("document.jpg")
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")
print(f"Format: {img.format}")

# 2. Optimize image before OCR
def optimize_image(image_path):
    with Image.open(image_path) as img:
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if too large
        max_dim = 2048
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))

        # Save optimized
        img.save("optimized.jpg", quality=95, optimize=True)
        return "optimized.jpg"

optimized = optimize_image("document.jpg")
# Now process optimized image
```

---

### Issue: `Provider timeout` or `Request timeout`

**Cause**: Provider API is slow or network issues

**Solution**:
```python
# Increase timeout in settings
import os
os.environ['TIMEOUT_SECONDS'] = '60'  # Default is 30

# Or use fallback strategy
result = await processor.process_document(
    "document.pdf",
    strategy="fallback",  # Try multiple providers
    providers="mistral,google_vision,aws_textract"
)
```

---

### Issue: Poor OCR accuracy

**Cause**: Image quality, provider selection, or document complexity

**Solution**:
```python
# 1. Use ensemble strategy for better accuracy
result = await processor.process_document(
    "document.pdf",
    strategy="ensemble"  # Combine multiple providers
)

# 2. Try different providers
providers = ["mistral", "google_vision", "aws_textract"]
results = []

for provider in providers:
    result = await processor.process_document(
        "document.pdf",
        strategy="single",
        providers=provider
    )
    results.append({
        "provider": provider,
        "confidence": result.confidence,
        "text": result.text
    })

# Choose best result
best = max(results, key=lambda x: x['confidence'])
print(f"Best provider: {best['provider']}")
```

---

## 📦 Batch Processing Issues

### Issue: Batch job stuck in `PENDING` status

**Cause**: Job not started or queue processor not running

**Solution**:
```python
# Start the job manually
import requests

job_id = "batch_abc123"
response = requests.post(f"http://localhost:8000/api/batch/jobs/{job_id}/start")

# Check job status
response = requests.get(f"http://localhost:8000/api/batch/jobs/{job_id}")
print(response.json())
```

---

### Issue: Batch job fails immediately

**Cause**: Invalid file paths or permissions

**Solution**:
```bash
# Check file paths exist and are readable
ls -la /path/to/files/

# Check permissions
chmod 644 /path/to/files/*

# Test with single file first
curl -X POST http://localhost:8000/process \
  -F "files=@/path/to/files/test.pdf"
```

---

### Issue: Scheduled job not running

**Cause**: Schedule configuration or timezone issues

**Solution**:
```python
import requests

# Get schedule details
schedule_id = "schedule_123"
response = requests.get(f"http://localhost:8000/api/batch/schedules/{schedule_id}")
schedule = response.json()

print(f"Enabled: {schedule['enabled']}")
print(f"Next Run: {schedule['next_run']}")
print(f"Schedule Type: {schedule['schedule_type']}")

# Enable if disabled
if not schedule['enabled']:
    requests.post(f"http://localhost:8000/api/batch/schedules/{schedule_id}/enable")

# Update schedule if needed
requests.put(f"http://localhost:8000/api/batch/schedules/{schedule_id}", json={
    "schedule_config": {
        "time": "02:00",
        "timezone": "UTC"  # Ensure correct timezone
    }
})
```

---

## 🚀 Deployment Issues

### Issue: Cloud Function deployment fails

**Cause**: Various deployment configuration issues

**Solution**:
```bash
# 1. Check gcloud authentication
gcloud auth list
gcloud config list

# 2. Verify project
gcloud config get-value project

# 3. Check required APIs enabled
gcloud services list --enabled | grep cloudfunctions
gcloud services list --enabled | grep cloudbuild

# 4. Enable if missing
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# 5. Check .gcloudignore
cat .gcloudignore

# 6. Verify requirements.txt
pip install -r requirements.txt --dry-run

# 7. Deploy with verbose logging
gcloud functions deploy ocular-ocr-service \
  --source . \
  --entry-point ocular_ocr \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --region us-central1 \
  --verbosity=debug
```

---

### Issue: `ImportError` in Cloud Functions

**Cause**: Missing dependencies or incorrect entry point

**Solution**:
```bash
# 1. Ensure all dependencies in requirements.txt
pip freeze | grep -E "fastapi|uvicorn|mangum" >> requirements.txt

# 2. Verify entry point in main.py
cat main.py | grep "def ocular_ocr"

# 3. Test locally with functions-framework
functions-framework --target=ocular_ocr --debug --port=8080

# 4. Check Cloud Function logs
gcloud functions logs read ocular-ocr-service --limit=50
```

---

### Issue: RunPod deployment fails

**Cause**: Invalid configuration or API issues

**Solution**:
```python
# 1. Verify RunPod API key
import os
print(f"RunPod API Key: {os.getenv('RUNPOD_API_KEY')[:10]}...")

# 2. Check endpoint configuration
import requests

headers = {"Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}"}
response = requests.get("https://api.runpod.ai/v2/endpoints", headers=headers)
print(response.json())

# 3. Verify GPU availability
response = requests.get("https://api.runpod.ai/v2/gpus", headers=headers)
available_gpus = response.json()
print(f"Available GPUs: {[g['name'] for g in available_gpus]}")

# 4. Check deployment logs
deployment_id = "your_deployment_id"
response = requests.get(
    f"http://localhost:8000/api/deployments/{deployment_id}",
    headers={"Authorization": f"Bearer YOUR_API_KEY"}
)
print(response.json().get('error_message'))
```

---

## 💾 Database Issues

### Issue: `Database connection failed`

**Cause**: Incorrect DATABASE_URL or database not running

**Solution**:
```bash
# 1. Check DATABASE_URL format
echo $DATABASE_URL
# Should be: postgresql://user:pass@host:port/dbname
# or: sqlite:///path/to/db.sqlite

# 2. Test PostgreSQL connection
psql $DATABASE_URL -c "SELECT 1"

# 3. For SQLite, check file exists and is writable
ls -l ocular.db
chmod 664 ocular.db

# 4. Test connection from Python
python -c "
from sqlalchemy import create_engine
engine = create_engine('$DATABASE_URL')
conn = engine.connect()
print('✅ Database connection OK')
conn.close()
"
```

---

### Issue: `Migration failed` or `Table already exists`

**Cause**: Database state mismatch

**Solution**:
```bash
# 1. Check current database schema
psql $DATABASE_URL -c "\dt"

# 2. Drop all tables (WARNING: deletes all data)
python -c "
from ocular.database.models import Base
from sqlalchemy import create_engine
engine = create_engine('$DATABASE_URL')
Base.metadata.drop_all(engine)
print('✅ All tables dropped')
"

# 3. Run migrations fresh
python ocular/database/migrate_add_auth.py
python ocular/database/migrate_add_billing.py
python ocular/database/migrate_add_document_tracking.py

# 4. Verify
python -c "
from ocular.database.models import *
from sqlalchemy import create_engine
engine = create_engine('$DATABASE_URL')
print('✅ All models loaded successfully')
"
```

---

### Issue: `Duplicate key value violates unique constraint`

**Cause**: Attempting to insert duplicate data

**Solution**:
```python
# Use upsert pattern
from sqlalchemy.dialects.postgresql import insert

stmt = insert(ProcessedDocument).values(
    file_hash="abc123...",
    file_name="document.pdf"
).on_conflict_do_update(
    index_elements=['file_hash'],
    set_={
        'last_processed_at': datetime.utcnow(),
        'processing_count': ProcessedDocument.processing_count + 1
    }
)

# Or check before insert
existing = db.query(ProcessedDocument).filter_by(file_hash=hash_value).first()
if existing:
    existing.processing_count += 1
else:
    new_doc = ProcessedDocument(file_hash=hash_value, ...)
    db.add(new_doc)
```

---

## 🔐 Authentication Issues

### Issue: `Authentication failed` with Clerk

**Cause**: Invalid API keys or configuration

**Solution**:
```bash
# 1. Verify Clerk keys
echo $CLERK_SECRET_KEY | head -c 20
echo $CLERK_PUBLISHABLE_KEY | head -c 20

# 2. Test Clerk API
curl -X GET "https://api.clerk.dev/v1/users" \
  -H "Authorization: Bearer $CLERK_SECRET_KEY"

# 3. Check Clerk configuration
python -c "
from ocular.settings import get_settings
settings = get_settings()
print(f'Clerk Secret: {settings.CLERK_SECRET_KEY[:20]}...')
print(f'Clerk Publishable: {settings.CLERK_PUBLISHABLE_KEY[:20]}...')
"
```

---

### Issue: `API key invalid`

**Cause**: Expired, revoked, or incorrectly formatted API key

**Solution**:
```python
# 1. Check API key format
# Should be: sk_live_... or sk_test_...

# 2. Verify in database
import requests

response = requests.get(
    "http://localhost:8000/api/auth/api-keys",
    headers={"Authorization": "Bearer YOUR_ADMIN_TOKEN"}
)
print(response.json())

# 3. Generate new API key
response = requests.post(
    "http://localhost:8000/api/auth/api-keys",
    json={
        "user_id": 1,
        "key_name": "New API Key",
        "permissions": ["ocr:process", "batch:create"]
    }
)
new_key = response.json()
print(f"New Key: {new_key['key']}")
```

---

### Issue: `Permission denied` for API endpoint

**Cause**: Insufficient permissions for user/API key

**Solution**:
```python
# 1. Check user permissions
response = requests.get(
    "http://localhost:8000/api/auth/users/1/permissions",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
print(f"Permissions: {response.json()}")

# 2. Update permissions
response = requests.post(
    "http://localhost:8000/api/auth/users/1/permissions",
    json={
        "permissions": [
            "ocr:process",
            "batch:create",
            "analytics:read",
            "deployments:create"
        ]
    },
    headers={"Authorization": "Bearer ADMIN_TOKEN"}
)
```

---

## ⚡ Performance Issues

### Issue: Slow API response times

**Cause**: Database queries, large file processing, or network latency

**Solution**:
```python
# 1. Enable query logging
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# 2. Add database indexes
from sqlalchemy import Index

# Check for missing indexes
# processed_documents should have indexes on:
# - file_hash (unique)
# - user_id
# - job_id
# - first_processed_at

# 3. Use connection pooling
from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# 4. Implement caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_processed_document(file_hash):
    # Cached database query
    pass
```

---

### Issue: High memory usage

**Cause**: Large file processing or memory leaks

**Solution**:
```python
# 1. Process files in chunks
def process_large_file(file_path):
    chunk_size = 4096

    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            # Process chunk
            process_chunk(chunk)

            # Force garbage collection
            import gc
            gc.collect()

# 2. Limit concurrent processing
import asyncio

semaphore = asyncio.Semaphore(5)  # Max 5 concurrent

async def process_with_limit(file_path):
    async with semaphore:
        return await process_file(file_path)

# 3. Monitor memory
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

---

### Issue: Batch processing too slow

**Cause**: Sequential processing or small batch size

**Solution**:
```python
# 1. Increase batch size
job = create_batch_job(
    job_name="Fast Processing",
    file_paths=files,
    batch_size=20  # Process 20 files concurrently (default: 5)
)

# 2. Use parallel processing
import concurrent.futures

def process_batch_parallel(files, max_workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, f) for f in files]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return results

# 3. Optimize provider selection
# Use fastest provider for large batches
result = await processor.process_document(
    file_path,
    strategy="single",
    providers="mistral"  # Fastest provider
)
```

---

## 🌐 API Issues

### Issue: `CORS error` in browser

**Cause**: CORS not configured properly

**Solution**:
```python
# In ocular/app.py, ensure CORS middleware is configured:

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For production, be specific:
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-domain.com",
        "https://app.your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

---

### Issue: `422 Unprocessable Entity`

**Cause**: Request validation error

**Solution**:
```bash
# Check error response for details
curl -X POST http://localhost:8000/api/batch/jobs \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}' \
  -v

# Common issues:
# 1. Missing required fields
# 2. Wrong data types
# 3. Invalid enum values

# Fix example:
curl -X POST http://localhost:8000/api/batch/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "Test Job",
    "file_paths": ["/path/to/file.pdf"],
    "provider": "mistral",
    "strategy": "fallback",
    "priority": "NORMAL"
  }'
```

---

### Issue: `500 Internal Server Error`

**Cause**: Unhandled exception in code

**Solution**:
```bash
# 1. Check application logs
tail -f logs/ocular.log

# 2. For Cloud Functions
gcloud functions logs read ocular-ocr-service --limit=50

# 3. For local development
uvicorn ocular.app:app --reload --log-level debug

# 4. Add error logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    # Your code
    pass
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    raise
```

---

## ❌ Common Error Messages

### `RuntimeError: Event loop is closed`

**Cause**: Async/await misuse or event loop issues

**Solution**:
```python
import asyncio

# Option 1: Use asyncio.run()
async def main():
    result = await process_document("file.pdf")
    return result

result = asyncio.run(main())

# Option 2: Get or create event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

result = loop.run_until_complete(process_document("file.pdf"))
```

---

### `FileNotFoundError: [Errno 2] No such file or directory`

**Cause**: File path doesn't exist or incorrect path

**Solution**:
```python
from pathlib import Path

# Use absolute paths
file_path = Path(__file__).parent / "data" / "document.pdf"

# Check file exists
if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

# Create directory if needed
file_path.parent.mkdir(parents=True, exist_ok=True)
```

---

### `MemoryError: Unable to allocate...`

**Cause**: Insufficient memory for operation

**Solution**:
```python
# 1. Process in chunks
def process_large_pdf(pdf_path):
    from pdf2image import convert_from_path

    # Process one page at a time
    pages = convert_from_path(
        pdf_path,
        first_page=1,
        last_page=1  # Process single page
    )

    # Or use dpi to reduce memory
    pages = convert_from_path(
        pdf_path,
        dpi=150  # Lower DPI = less memory
    )

# 2. Increase system limits (Linux)
import resource

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
```

---

## 📞 Getting Help

### Debug Mode
```bash
# Enable debug mode
export DEBUG=True
export LOG_LEVEL=DEBUG

# Run with verbose logging
uvicorn ocular.app:app --reload --log-level debug
```

### Collect Diagnostic Information
```bash
# System info
python --version
pip list | grep -E "fastapi|pydantic|sqlalchemy"

# Environment
env | grep -E "MISTRAL|GOOGLE|AWS|DATABASE"

# Logs
tail -100 logs/ocular.log

# Database
psql $DATABASE_URL -c "\dt"  # List tables
```

### Report Issues
When reporting issues, include:
1. Error message (full traceback)
2. Steps to reproduce
3. Environment details (OS, Python version)
4. Configuration (sanitized, no secrets)
5. Relevant logs

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **GitHub Issues**: https://github.com/your-org/ocular/issues
- **Discord Community**: https://discord.gg/ocular
- **Stack Overflow**: Tag `ocular-ocr`

---

**Last Updated**: 2025-10-06
**Version**: 3.0
**Maintained By**: Development Team
