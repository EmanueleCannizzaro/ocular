# Ocular OCR Platform - API Usage Examples

**Version**: 3.0
**Last Updated**: 2025-10-06

Complete API usage examples with code snippets in Python, JavaScript, and cURL.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Basic OCR Processing](#basic-ocr-processing)
3. [Batch Processing](#batch-processing)
4. [Document Tracking](#document-tracking)
5. [Model Deployment](#model-deployment)
6. [Analytics & Monitoring](#analytics--monitoring)
7. [Authentication & API Keys](#authentication--api-keys)
8. [Cost Optimization](#cost-optimization)
9. [Advanced Use Cases](#advanced-use-cases)

---

## 🚀 Getting Started

### Base URL
```
http://localhost:8000  # Development
https://your-domain.com  # Production
```

### Authentication
```python
# API Key Authentication
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
```

---

## 📄 Basic OCR Processing

### Example 1: Process Single Image

**Python:**
```python
import requests

url = "http://localhost:8000/process"

# Upload file
with open("document.jpg", "rb") as f:
    files = {"files": ("document.jpg", f, "image/jpeg")}
    data = {
        "strategy": "fallback",
        "providers": "mistral,google_vision"
    }

    response = requests.post(url, files=files, data=data)
    result = response.json()

print(f"Extracted Text: {result['results'][0]['text']}")
print(f"Provider Used: {result['results'][0]['provider_used']}")
print(f"Processing Time: {result['results'][0]['processing_time']}s")
```

**JavaScript:**
```javascript
const formData = new FormData();
formData.append('files', fileInput.files[0]);
formData.append('strategy', 'fallback');
formData.append('providers', 'mistral,google_vision');

fetch('http://localhost:8000/process', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Extracted Text:', data.results[0].text);
    console.log('Provider Used:', data.results[0].provider_used);
})
.catch(error => console.error('Error:', error));
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/process" \
  -F "files=@document.jpg" \
  -F "strategy=fallback" \
  -F "providers=mistral,google_vision"
```

### Example 2: Process PDF with Multiple Providers

**Python:**
```python
import requests

def process_pdf_with_ensemble(pdf_path):
    """Process PDF using ensemble strategy for best accuracy."""

    url = "http://localhost:8000/process"

    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path, f, "application/pdf")}
        data = {
            "strategy": "ensemble",  # Combine multiple providers
            "providers": "mistral,google_vision,aws_textract"
        }

        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            return {
                "text": result["results"][0]["text"],
                "confidence": result["results"][0].get("confidence", 0),
                "providers_used": result["results"][0]["provider_used"],
                "processing_time": result["results"][0]["processing_time"]
            }
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

# Usage
result = process_pdf_with_ensemble("invoice.pdf")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Text:\n{result['text']}")
```

### Example 3: Process Multiple Files

**Python:**
```python
import requests
from pathlib import Path

def process_multiple_files(file_paths, provider="mistral"):
    """Process multiple files in a single request."""

    url = "http://localhost:8000/process"

    files = []
    for file_path in file_paths:
        path = Path(file_path)
        files.append(
            ("files", (path.name, open(file_path, "rb"), "image/jpeg"))
        )

    data = {
        "strategy": "single",
        "providers": provider
    }

    response = requests.post(url, files=files, data=data)

    # Close file handles
    for _, (_, f, _) in files:
        f.close()

    return response.json()

# Usage
files = ["doc1.jpg", "doc2.jpg", "doc3.jpg"]
results = process_multiple_files(files)

for i, result in enumerate(results["results"]):
    print(f"\n--- File {i+1}: {files[i]} ---")
    print(f"Text: {result['text'][:100]}...")
    print(f"Time: {result['processing_time']}s")
```

---

## 📦 Batch Processing

### Example 4: Create Batch Job

**Python:**
```python
import requests

def create_batch_job(file_paths, job_name, priority="NORMAL"):
    """Create a batch OCR job."""

    url = "http://localhost:8000/api/batch/upload-batch"

    files = [("files", open(fp, "rb")) for fp in file_paths]

    data = {
        "job_name": job_name,
        "provider": "mistral",
        "strategy": "fallback",
        "priority": priority
    }

    response = requests.post(url, files=files, data=data)

    # Close file handles
    for _, f in files:
        f.close()

    return response.json()

# Usage
files = [f"invoice_{i}.pdf" for i in range(1, 101)]  # 100 invoices
job = create_batch_job(files, "Monthly Invoices - October", priority="HIGH")

print(f"Job ID: {job['job_id']}")
print(f"Total Files: {job['total_files']}")
print(f"Status: {job['message']}")
```

### Example 5: Monitor Batch Job Progress

**Python:**
```python
import requests
import time

def monitor_batch_job(job_id, check_interval=5):
    """Monitor batch job progress until completion."""

    url = f"http://localhost:8000/api/batch/jobs/{job_id}"

    while True:
        response = requests.get(url)
        job = response.json()

        status = job["status"]
        progress = job["progress_percent"]
        completed = job["completed_count"]
        total = job["total_files"]

        print(f"Status: {status} | Progress: {progress:.1f}% | Files: {completed}/{total}")

        if status in ["COMPLETED", "FAILED", "CANCELLED"]:
            break

        time.sleep(check_interval)

    return job

# Usage
job_id = "batch_abc123"
final_job = monitor_batch_job(job_id)

print(f"\n✅ Job Complete!")
print(f"Completed: {final_job['completed_count']}")
print(f"Failed: {final_job['failed_count']}")
```

### Example 6: Create Scheduled Job

**Python:**
```python
import requests

def create_daily_schedule(name, time, source_path, provider="mistral"):
    """Create a daily scheduled job."""

    url = "http://localhost:8000/api/batch/schedules"

    payload = {
        "schedule_name": name,
        "schedule_type": "DAILY",
        "schedule_config": {
            "time": time,
            "timezone": "UTC"
        },
        "source_path": source_path,
        "provider": provider,
        "strategy": "fallback",
        "enabled": True
    }

    response = requests.post(url, json=payload)
    return response.json()

# Usage - Process all PDFs in a folder daily at 2 AM
schedule = create_daily_schedule(
    name="Daily Document Processing",
    time="02:00",
    source_path="/data/incoming/*.pdf",
    provider="google_vision"
)

print(f"Schedule ID: {schedule['schedule_id']}")
print(f"Next Run: {schedule['next_run']}")
```

### Example 7: CRON Schedule

**Python:**
```python
def create_cron_schedule(name, cron_expression, source_path):
    """Create a CRON-based scheduled job."""

    url = "http://localhost:8000/api/batch/schedules"

    payload = {
        "schedule_name": name,
        "schedule_type": "CRON",
        "schedule_config": {
            "cron_expression": cron_expression
        },
        "source_path": source_path,
        "provider": "mistral",
        "strategy": "fallback",
        "enabled": True
    }

    response = requests.post(url, json=payload)
    return response.json()

# Usage - Every 6 hours
schedule = create_cron_schedule(
    name="Every 6 Hours Processing",
    cron_expression="0 */6 * * *",
    source_path="/data/batch/*.pdf"
)

print(f"Schedule created: {schedule['schedule_id']}")
```

---

## 🔍 Document Tracking

### Example 8: Check for Duplicates

**Python:**
```python
import requests

def check_duplicate(file_path):
    """Check if a document has been processed before."""

    url = "http://localhost:8000/api/document-tracking/check-duplicate"

    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    result = response.json()

    if result["is_duplicate"]:
        existing = result["existing_record"]
        return {
            "is_duplicate": True,
            "first_processed": existing["first_processed_at"],
            "processing_count": existing["processing_count"],
            "last_provider": existing["provider_used"],
            "extracted_text": existing["extracted_text_preview"]
        }
    else:
        return {"is_duplicate": False}

# Usage
check = check_duplicate("document.pdf")

if check["is_duplicate"]:
    print(f"⚠️ Duplicate detected!")
    print(f"First processed: {check['first_processed']}")
    print(f"Processed {check['processing_count']} times")
    print(f"Previous text preview: {check['extracted_text'][:100]}...")
else:
    print("✅ New document - proceed with processing")
```

### Example 9: Get Duplicate Prevention Stats

**Python:**
```python
def get_duplicate_stats(user_id=None):
    """Get duplicate prevention statistics."""

    url = "http://localhost:8000/api/document-tracking/stats"

    params = {}
    if user_id:
        params["user_id"] = user_id

    response = requests.get(url, params=params)
    stats = response.json()

    print("📊 Duplicate Prevention Statistics")
    print(f"  Total Unique Documents: {stats['total_unique_documents']}")
    print(f"  Total Processing Attempts: {stats['total_processing_count']}")
    print(f"  Duplicates Avoided: {stats['duplicates_avoided']}")
    print(f"  Duplicate Rate: {stats['duplicate_percentage']:.1f}%")
    print(f"  Total Cost: ${stats['total_cost_usd']:.2f}")
    print(f"  Total Tokens: {stats['total_tokens']:,}")
    print(f"  Avg Processing Time: {stats['avg_processing_time_seconds']:.2f}s")

    return stats

# Usage
stats = get_duplicate_stats()
```

---

## 🚀 Model Deployment

### Example 10: Deploy Model to RunPod

**Python:**
```python
import requests

def deploy_model_to_runpod(model_id, gpu_type="A40", gpu_count=1):
    """Deploy a model to RunPod."""

    url = "http://localhost:8000/api/deployments"

    payload = {
        "model_id": model_id,
        "provider_id": 1,  # RunPod provider ID
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "deployment_type": "vllm",
        "config": {
            "max_model_len": 4096,
            "tensor_parallel_size": gpu_count,
            "gpu_memory_utilization": 0.9
        }
    }

    response = requests.post(url, json=payload)
    return response.json()

# Usage
deployment = deploy_model_to_runpod(
    model_id=1,  # OLMocr model
    gpu_type="A40",
    gpu_count=1
)

print(f"Deployment ID: {deployment['id']}")
print(f"Status: {deployment['status']}")
print(f"Endpoint: {deployment['endpoint_url']}")
```

### Example 11: Create Model Version with A/B Testing

**Python:**
```python
def create_model_version_with_ab_test(model_id, version_info, traffic_split=20):
    """Create new model version and setup A/B test."""

    # Create new version
    version_url = "http://localhost:8000/api/versions"
    version_payload = {
        "model_id": model_id,
        "version_number": version_info["version"],
        "version_name": version_info["name"],
        "version_tag": "beta",
        "changelog": version_info["changelog"],
        "docker_image": version_info["docker_image"]
    }

    response = requests.post(version_url, json=version_payload)
    new_version = response.json()

    # Get current active version
    active_url = f"http://localhost:8000/api/models/{model_id}/versions/active"
    active_version = requests.get(active_url).json()

    # Setup A/B test
    ab_test_url = "http://localhost:8000/api/versions/ab-test/setup"
    ab_payload = {
        "version_id_a": active_version["id"],
        "version_id_b": new_version["id"],
        "traffic_split": traffic_split,  # % to new version
        "duration_days": 7
    }

    requests.post(ab_test_url, json=ab_payload)

    return new_version

# Usage
new_version = create_model_version_with_ab_test(
    model_id=1,
    version_info={
        "version": "v2.1.0",
        "name": "Performance Improvements",
        "changelog": "25% faster inference, improved accuracy",
        "docker_image": "registry/olmocr:v2.1.0"
    },
    traffic_split=20  # 20% of traffic to new version
)

print(f"New version created: {new_version['version_number']}")
print(f"A/B test active: 20% traffic to new version")
```

---

## 📊 Analytics & Monitoring

### Example 12: Get Deployment Analytics

**Python:**
```python
def get_deployment_analytics(days=7):
    """Get deployment analytics for the past N days."""

    # Get dashboard summary
    summary_url = "http://localhost:8000/api/analytics/dashboard"
    summary = requests.get(summary_url).json()

    # Get deployment timeline
    timeline_url = f"http://localhost:8000/api/analytics/deployment-timeline?days={days}"
    timeline = requests.get(timeline_url).json()

    # Get cost analytics
    cost_url = f"http://localhost:8000/api/analytics/cost-analytics?days={days}"
    costs = requests.get(cost_url).json()

    print(f"📊 Analytics Summary (Last {days} days)")
    print(f"\nDeployments:")
    print(f"  Active: {summary['active_deployments']}")
    print(f"  24h Change: {summary['active_deployments_24h_change']:+d}")

    print(f"\nGPU Utilization:")
    print(f"  {summary['gpu_utilization_percent']:.1f}%")
    print(f"  Active GPUs: {summary['active_gpus']}")

    print(f"\nCosts:")
    print(f"  Daily Estimate: ${summary['daily_cost_estimate']:.2f}")
    print(f"  Total ({days}d): ${costs['total_cost']:.2f}")

    return {
        "summary": summary,
        "timeline": timeline,
        "costs": costs
    }

# Usage
analytics = get_deployment_analytics(days=30)
```

### Example 13: Monitor Observability Logs

**Python:**
```python
def stream_logs(service=None, level="INFO", limit=100):
    """Stream recent logs from observability system."""

    url = "http://localhost:8000/api/observability/logs"

    params = {
        "limit": limit,
        "level": level
    }

    if service:
        params["service"] = service

    response = requests.get(url, params=params)
    logs = response.json()

    for log in logs["logs"]:
        timestamp = log["timestamp"]
        level = log["level"]
        service = log["service"]
        message = log["message"]

        print(f"[{timestamp}] [{level}] [{service}] {message}")

    return logs

# Usage
logs = stream_logs(service="ocr", level="ERROR", limit=50)
```

---

## 🔐 Authentication & API Keys

### Example 14: Create API Key

**Python:**
```python
def create_api_key(user_id, key_name, permissions=None):
    """Create a new API key."""

    url = "http://localhost:8000/api/auth/api-keys"

    payload = {
        "user_id": user_id,
        "key_name": key_name,
        "permissions": permissions or ["ocr:process", "batch:create"],
        "expires_at": None  # No expiration
    }

    response = requests.post(url, json=payload)
    key = response.json()

    print(f"✅ API Key Created")
    print(f"Key ID: {key['id']}")
    print(f"Key: {key['key']}")  # Save this - shown only once!
    print(f"Permissions: {', '.join(key['permissions'])}")

    return key

# Usage
api_key = create_api_key(
    user_id=1,
    key_name="Production API Key",
    permissions=["ocr:process", "batch:create", "analytics:read"]
)
```

### Example 15: Use API Key for Authentication

**Python:**
```python
def process_with_api_key(file_path, api_key):
    """Process document using API key authentication."""

    url = "http://localhost:8000/process"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    with open(file_path, "rb") as f:
        files = {"files": f}
        data = {"strategy": "fallback"}

        response = requests.post(
            url,
            files=files,
            data=data,
            headers=headers
        )

    return response.json()

# Usage
result = process_with_api_key(
    "document.pdf",
    api_key="sk_live_abc123..."
)
```

---

## 💰 Cost Optimization

### Example 16: Optimize Image Before Processing

**Python:**
```python
from PIL import Image
import io
import requests

def optimize_and_process(image_path):
    """Optimize image then process with OCR."""

    # Optimize image
    with Image.open(image_path) as img:
        # Resize if too large
        max_dimension = 2048
        if img.width > max_dimension or img.height > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        # Save optimized
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        optimized_bytes = buffer.getvalue()

    print(f"Original size: {len(open(image_path, 'rb').read())} bytes")
    print(f"Optimized size: {len(optimized_bytes)} bytes")
    print(f"Reduction: {(1 - len(optimized_bytes)/len(open(image_path, 'rb').read()))*100:.1f}%")

    # Process optimized image
    url = "http://localhost:8000/process"
    files = {"files": ("optimized.jpg", optimized_bytes, "image/jpeg")}
    data = {"strategy": "single", "providers": "mistral"}

    response = requests.post(url, files=files, data=data)
    return response.json()

# Usage
result = optimize_and_process("large_image.png")
```

### Example 17: Compare Provider Costs

**Python:**
```python
def compare_provider_costs(file_path):
    """Process with different providers and compare costs."""

    providers = ["mistral", "google_vision", "aws_textract", "tesseract"]
    results = {}

    for provider in providers:
        url = "http://localhost:8000/process"

        with open(file_path, "rb") as f:
            files = {"files": f}
            data = {"strategy": "single", "providers": provider}

            response = requests.post(url, files=files, data=data)
            result = response.json()["results"][0]

            results[provider] = {
                "time": result["processing_time"],
                "cost": result.get("cost_usd", 0),
                "tokens": result.get("token_count", 0)
            }

    print("💰 Provider Cost Comparison")
    for provider, data in results.items():
        print(f"\n{provider.upper()}:")
        print(f"  Time: {data['time']:.2f}s")
        print(f"  Cost: ${data['cost']:.4f}")
        print(f"  Tokens: {data['tokens']}")

    # Find cheapest
    cheapest = min(results.items(), key=lambda x: x[1]['cost'])
    print(f"\n✅ Cheapest: {cheapest[0].upper()} (${cheapest[1]['cost']:.4f})")

    return results

# Usage
comparison = compare_provider_costs("document.pdf")
```

---

## 🎯 Advanced Use Cases

### Example 18: Parallel Batch Processing with Progress

**Python:**
```python
import requests
import concurrent.futures
from pathlib import Path

def process_file_batch(file_paths, max_workers=5):
    """Process multiple files in parallel with progress tracking."""

    def process_single_file(file_path):
        url = "http://localhost:8000/process"

        with open(file_path, "rb") as f:
            files = {"files": f}
            data = {"strategy": "fallback"}

            response = requests.post(url, files=files, data=data)
            return {
                "file": file_path,
                "result": response.json()["results"][0] if response.ok else None,
                "error": None if response.ok else response.text
            }

    results = []
    total = len(file_paths)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, fp): fp for fp in file_paths}

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)

            status = "✅" if result["result"] else "❌"
            print(f"{status} [{i}/{total}] {Path(result['file']).name}")

    return results

# Usage
files = [f"document_{i}.pdf" for i in range(1, 21)]
results = process_file_batch(files, max_workers=10)

successful = sum(1 for r in results if r["result"])
print(f"\n📊 Summary: {successful}/{len(results)} successful")
```

### Example 19: Webhook Integration for Batch Jobs

**Python:**
```python
from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/webhook/batch-complete', methods=['POST'])
def batch_complete_webhook():
    """Webhook endpoint for batch job completion."""

    data = request.json
    job_id = data["job_id"]
    status = data["status"]
    completed = data["completed_count"]
    failed = data["failed_count"]

    print(f"📬 Batch job {job_id} completed!")
    print(f"Status: {status}")
    print(f"Completed: {completed}, Failed: {failed}")

    # Process results
    if status == "COMPLETED":
        # Download results, send notifications, etc.
        pass

    return {"received": True}

def create_batch_with_webhook(file_paths):
    """Create batch job with webhook callback."""

    url = "http://localhost:8000/api/batch/jobs"

    payload = {
        "job_name": "Webhook Test Job",
        "file_paths": file_paths,
        "provider": "mistral",
        "webhook_url": "https://your-server.com/webhook/batch-complete"
    }

    response = requests.post(url, json=payload)
    return response.json()

# Start Flask app to receive webhooks
if __name__ == '__main__':
    app.run(port=5000)
```

### Example 20: Complete OCR Pipeline with All Features

**Python:**
```python
import requests
from pathlib import Path

class OcularOCRPipeline:
    """Complete OCR pipeline with all platform features."""

    def __init__(self, api_key, base_url="http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def process_with_duplicate_check(self, file_path):
        """Process file with duplicate detection."""

        # Check for duplicate
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{self.base_url}/api/document-tracking/check-duplicate",
                files=files,
                headers=self.headers
            )

        check = response.json()

        if check["is_duplicate"]:
            print(f"⚠️ Duplicate detected - retrieving cached result")
            return {
                "text": check["existing_record"]["extracted_text_preview"],
                "from_cache": True,
                "processing_count": check["existing_record"]["processing_count"]
            }

        # Process new document
        print(f"✅ New document - processing with OCR")
        with open(file_path, "rb") as f:
            files = {"files": f}
            data = {"strategy": "fallback", "providers": "mistral,google_vision"}

            response = requests.post(
                f"{self.base_url}/process",
                files=files,
                data=data,
                headers=self.headers
            )

        result = response.json()["results"][0]

        return {
            "text": result["text"],
            "from_cache": False,
            "provider_used": result["provider_used"],
            "processing_time": result["processing_time"]
        }

    def create_scheduled_batch(self, name, cron, source_path):
        """Create scheduled batch job."""

        payload = {
            "schedule_name": name,
            "schedule_type": "CRON",
            "schedule_config": {"cron_expression": cron},
            "source_path": source_path,
            "provider": "mistral",
            "enabled": True
        }

        response = requests.post(
            f"{self.base_url}/api/batch/schedules",
            json=payload,
            headers=self.headers
        )

        return response.json()

    def get_analytics_summary(self):
        """Get complete analytics summary."""

        response = requests.get(
            f"{self.base_url}/api/analytics/dashboard",
            headers=self.headers
        )

        return response.json()

# Usage
pipeline = OcularOCRPipeline(api_key="sk_live_abc123...")

# Process with duplicate check
result = pipeline.process_with_duplicate_check("invoice.pdf")
print(f"Text: {result['text'][:100]}...")
print(f"From Cache: {result['from_cache']}")

# Create scheduled job
schedule = pipeline.create_scheduled_batch(
    name="Nightly Processing",
    cron="0 2 * * *",  # 2 AM daily
    source_path="/data/*.pdf"
)
print(f"Schedule created: {schedule['schedule_id']}")

# Get analytics
analytics = pipeline.get_analytics_summary()
print(f"Active Deployments: {analytics['active_deployments']}")
print(f"Daily Cost: ${analytics['daily_cost_estimate']:.2f}")
```

---

## 🔗 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Complete Features List**: See COMPLETE_FEATURES_LIST.md
- **Batch Processing Guide**: See BATCH_PROCESSING_GUIDE.md
- **Deployment Guide**: See DEPLOYMENT_GUIDE.md

---

**Last Updated**: 2025-10-06
**Version**: 3.0
**Support**: See GitHub Issues for questions and support
