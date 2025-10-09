# OLMocr RunPod Serverless Deployment Guide

Complete guide for deploying **OLMocr (allenai/olmOCR-7B-0825)** as a serverless endpoint on RunPod with S3 storage integration.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Cost Analysis](#cost-analysis)

## Overview

### What is OLMocr?

OLMocr (Open Language Model for OCR) is a 7B parameter vision-language model from AllenAI designed for optical character recognition tasks. It excels at:

- General purpose OCR with high accuracy
- Structured document extraction
- Table and form recognition
- Multi-page document processing
- Custom prompt-based extraction

### Why RunPod Serverless?

- **Cost Efficiency**: Pay only for actual compute time
- **Auto-scaling**: Automatically scales from 0 to max workers
- **GPU Access**: Access to high-end GPUs (RTX 4090, A100)
- **S3 Storage**: Built-in S3-compatible storage for results
- **Global CDN**: Low-latency access worldwide

## Prerequisites

### 1. Accounts Required

- [x] **RunPod Account** - [Sign up](https://runpod.io/)
- [x] **Docker Hub Account** - For hosting container images
- [x] **Hugging Face Account** - For model access (optional)

### 2. Local Requirements

```bash
# Check Docker installation
docker --version  # Should be 20.10+

# Check Python installation
python3 --version  # Should be 3.8+

# Install required Python packages
pip install runpod python-dotenv requests
```

### 3. API Keys

You'll need the following API keys:

1. **RunPod API Key**
   - Go to [RunPod Settings](https://console.runpod.io/settings)
   - Create new API key
   - Save securely

2. **RunPod S3 Credentials**
   - Go to [RunPod Settings > API Keys](https://console.runpod.io/settings)
   - Click "Create S3 API Key"
   - Save both Access Key ID and Secret Access Key

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Client    │────▶│  RunPod Endpoint │────▶│  OLMocr     │
│ Application │     │  (Your Handler)  │     │  7B Model   │
└─────────────┘     └──────────────────┘     └─────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  RunPod S3      │
                    │  Storage        │
                    │  (Results)      │
                    └─────────────────┘
```

### Components

1. **Docker Container**: Hosts OLMocr model and handler code
2. **RunPod Serverless**: Manages scaling, routing, and execution
3. **S3 Storage**: Stores OCR results and model cache
4. **Network Volume**: Optional persistent storage for model files

## Step-by-Step Deployment

### Step 1: Clone and Setup

```bash
# Navigate to deployment directory
cd /mnt/developer/git/aecs4u.it/ocular/runpod.io/olmocr

# Copy environment template
cp .env.example .env

# Edit .env with your actual values
nano .env
```

### Step 2: Configure Environment

Edit `.env` file with your credentials:

```bash
# Docker Configuration
DOCKER_USERNAME=your-dockerhub-username
IMAGE_TAG=latest

# RunPod API Key
RUNPOD_API_KEY=your_runpod_api_key

# RunPod S3 Storage
RUNPOD_ACCESS_KEY_ID=user_***...
RUNPOD_SECRET_ACCESS_KEY=rps_***...
RUNPOD_BUCKET_ENDPOINT_URL=https://s3api-eur-is-1.runpod.io/
RUNPOD_NETWORK_VOLUME_ID=your-volume-id
RUNPOD_REGION=eur-is-1

# Endpoint Configuration
RUNPOD_ENDPOINT_NAME=olmocr-endpoint
GPU_TYPE=NVIDIA RTX 4090
CONTAINER_DISK_GB=30
MIN_WORKERS=0
MAX_WORKERS=3
IDLE_TIMEOUT=30
EXECUTION_TIMEOUT=600
```

### Step 3: Create Network Volume (Optional but Recommended)

Network volumes cache the model between cold starts, significantly reducing initialization time.

1. Go to [RunPod Console > Storage > Network Volumes](https://console.runpod.io/storage)
2. Click "Create Network Volume"
3. Configure:
   - **Name**: `olmocr-model-cache`
   - **Size**: **50GB** (model is ~14GB)
   - **Datacenter**: Select closest to your users
4. Copy the **Network Volume ID** to `.env`

### Step 4: Build and Deploy

#### Option A: Automated Deployment (Recommended)

```bash
# Run deployment script
./deploy_olmocr.sh
```

This script will:
1. Build Docker image
2. Push to Docker Hub
3. Deploy to RunPod
4. Save endpoint ID for testing

#### Option B: Manual Steps

```bash
# 1. Build Docker image
docker build -t your-username/olmocr-runpod:latest .

# 2. Login to Docker Hub
docker login

# 3. Push image
docker push your-username/olmocr-runpod:latest

# 4. Deploy using Python API
python deploy_olmocr_api.py

# OR deploy manually via RunPod Console
```

#### Option C: Manual Deployment via Console

1. Go to [RunPod Console > Serverless](https://console.runpod.io/serverless)
2. Click "Deploy Endpoint"
3. Select "Custom" template
4. Configure:

   **Basic Settings:**
   - Name: `olmocr-endpoint`
   - Docker Image: `your-username/olmocr-runpod:latest`

   **Compute:**
   - GPU Type: RTX 4090 (recommended) or A100
   - Container Disk: 30GB
   - Network Volume: Select your volume (optional)

   **Scaling:**
   - Min Workers: 0 (for dev) or 1 (for prod)
   - Max Workers: 3
   - Idle Timeout: 30 seconds
   - Execution Timeout: 600 seconds (10 minutes)

   **Environment Variables:**
   ```
   RUNPOD_ACCESS_KEY_ID=user_***...
   RUNPOD_SECRET_ACCESS_KEY=rps_***...
   RUNPOD_BUCKET_ENDPOINT_URL=https://s3api-eur-is-1.runpod.io/
   RUNPOD_NETWORK_VOLUME_ID=your-volume-id
   RUNPOD_REGION=eur-is-1
   MODEL_NAME=allenai/olmOCR-7B-0825
   ```

5. Click "Deploy"

### Step 5: Wait for Deployment

- Initial deployment: **2-5 minutes**
- Check status: [RunPod Console > Serverless](https://console.runpod.io/serverless)
- Status should change: `Deploying` → `Active`

### Step 6: Test Endpoint

```bash
# Test with saved endpoint ID
python test_endpoint.py

# Or specify endpoint ID manually
python test_endpoint.py your-endpoint-id

# Test with custom image
python test_endpoint.py your-endpoint-id path/to/image.png
```

## Configuration

### GPU Selection Guide

| GPU | VRAM | Speed | Cost/hr | Best For |
|-----|------|-------|---------|----------|
| **RTX 4090** | 24GB | Fast | ~$0.30 | ⭐ Best value for most use cases |
| **A100 40GB** | 40GB | Very Fast | ~$1.50 | High-throughput production |
| **A6000** | 48GB | Fast | ~$0.80 | Sustained workloads |
| **RTX 3090** | 24GB | Medium | ~$0.20 | Budget development |

**Recommendation**: Start with RTX 4090 for optimal price/performance.

### Scaling Configuration

#### Development/Testing
```env
MIN_WORKERS=0        # No idle cost
MAX_WORKERS=1        # Single worker
IDLE_TIMEOUT=30      # Fast shutdown
EXECUTION_TIMEOUT=300  # 5 minutes
```

#### Production
```env
MIN_WORKERS=1        # Always ready (no cold starts)
MAX_WORKERS=5        # Auto-scale up to 5
IDLE_TIMEOUT=60      # Keep warm longer
EXECUTION_TIMEOUT=600  # 10 minutes for large docs
```

#### High-Volume Production
```env
MIN_WORKERS=3        # Multiple warm instances
MAX_WORKERS=10       # Scale to 10 workers
IDLE_TIMEOUT=120     # Keep warm for 2 minutes
EXECUTION_TIMEOUT=900  # 15 minutes max
```

### S3 Endpoint URLs

| Datacenter | Location | Endpoint URL |
|------------|----------|--------------|
| **EUR-IS-1** | Iceland | `https://s3api-eur-is-1.runpod.io/` |
| **US-CA-2** | California | `https://s3api-us-ca-2.runpod.io/` |
| **US-OR-1** | Oregon | `https://s3api-us-or-1.runpod.io/` |

Choose datacenter closest to your primary users for lowest latency.

## Testing

### Basic Test

```python
import runpod
import base64

runpod.api_key = "your-runpod-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

# Load image
with open("test.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Run OCR
result = endpoint.run_sync({
    "input": {
        "image": image_b64,
        "prompt": "Extract all text from this image."
    }
})

print(result["text"])
```

### Advanced Test with S3 Storage

```python
result = endpoint.run_sync({
    "input": {
        "image": image_b64,
        "prompt": "Extract invoice details including number, date, and total.",
        "upload_result": True  # Save to S3
    }
})

print(f"Text: {result['text']}")
print(f"Saved to: {result['s3_url']}")
```

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_image(image_path):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    return endpoint.run_sync({
        "input": {"image": image_b64}
    })

# Process multiple images in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(process_image, image_paths))
```

## Performance Optimization

### 1. Reduce Cold Start Time

**Option A: Pre-download Model in Docker Image**

Uncomment in `Dockerfile`:
```dockerfile
RUN python -c "from transformers import AutoModel, AutoProcessor; \
    model = AutoModel.from_pretrained('allenai/olmOCR-7B-0825', trust_remote_code=True); \
    processor = AutoProcessor.from_pretrained('allenai/olmOCR-7B-0825', trust_remote_code=True)"
```

**Trade-off**:
- ✅ Faster cold starts (30s → 10s)
- ❌ Larger image size (~14GB more)
- ❌ Longer build/push time

**Option B: Use Network Volume for Model Cache**

```env
RUNPOD_NETWORK_VOLUME_ID=your-volume-id
```

Set in `Dockerfile`:
```dockerfile
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache
```

**Trade-off**:
- ✅ Persistent cache across cold starts
- ✅ Smaller Docker image
- ❌ Network volume costs ($0.10/GB/month)

### 2. Optimize for Throughput

```env
# Keep multiple workers warm
MIN_WORKERS=3
MAX_WORKERS=10

# Longer idle timeout
IDLE_TIMEOUT=120

# Larger execution timeout for big docs
EXECUTION_TIMEOUT=900
```

### 3. Optimize for Cost

```env
# Scale to zero when idle
MIN_WORKERS=0

# Aggressive shutdown
IDLE_TIMEOUT=30

# Smaller container disk
CONTAINER_DISK_GB=20
```

## Troubleshooting

### Issue: Cold Start Too Slow (>60 seconds)

**Solutions**:
1. Pre-download model in Docker image
2. Use network volume for model cache
3. Set `MIN_WORKERS=1` to keep instance warm
4. Choose faster datacenter closer to model source

### Issue: Out of Memory Error

**Solutions**:
1. Use GPU with more VRAM (minimum 24GB)
2. Reduce `max_new_tokens` in request
3. Process smaller images or regions
4. Use FP16 precision (already enabled)

### Issue: S3 Upload Fails

**Check**:
```python
import boto3

s3 = boto3.client(
    's3',
    endpoint_url='https://s3api-eur-is-1.runpod.io/',
    aws_access_key_id='user_***',
    aws_secret_access_key='rps_***'
)

# Test connection
print(s3.list_buckets())
```

**Common Causes**:
- Incorrect S3 credentials
- Wrong endpoint URL for datacenter
- Network volume ID mismatch

### Issue: Model Download Fails

**Solutions**:
1. Check Hugging Face is accessible
2. Set `HF_TOKEN` if model requires authentication
3. Increase execution timeout for first download
4. Pre-download model in Docker image

### Issue: Endpoint Timeout

**Solutions**:
1. Increase `EXECUTION_TIMEOUT` (max 900s)
2. Process smaller documents
3. Use faster GPU
4. Optimize prompt to be more specific

## Cost Analysis

### Cost Components

1. **Compute**: GPU runtime cost
2. **Storage**: Container disk + Network volume
3. **Network**: Usually included (check RunPod pricing)

### Example Calculations

#### Low Volume (100 requests/day, avg 2s each)

```
Daily compute: 100 requests × 2s = 200s = 0.056 hours
Monthly compute: 0.056h × 30 days = 1.68 hours

Cost (RTX 4090 @ $0.30/hr):
- Compute: 1.68h × $0.30 = $0.50/month
- Storage (20GB): $0.20/month
- Network Volume (50GB): $5.00/month
Total: ~$5.70/month
```

#### Medium Volume (1,000 requests/day, avg 3s each)

```
Monthly compute: (1000 × 3s × 30) / 3600 = 25 hours

Cost (RTX 4090 @ $0.30/hr):
- Compute: 25h × $0.30 = $7.50/month
- Storage: $0.20/month
- Network Volume: $5.00/month
Total: ~$12.70/month
```

#### High Volume (10,000 requests/day, avg 3s each)

```
Monthly compute: (10000 × 3s × 30) / 3600 = 250 hours

Cost (RTX 4090 @ $0.30/hr):
- Compute: 250h × $0.30 = $75/month
- Storage: $0.30/month
- Network Volume: $5.00/month
Total: ~$80/month

With MIN_WORKERS=1 (always on):
- Compute: 720h × $0.30 = $216/month
- Other: $5.30/month
Total: ~$221/month
```

### Cost Optimization Tips

1. **Use MIN_WORKERS=0** for development and low-volume production
2. **Set aggressive IDLE_TIMEOUT** to scale down quickly
3. **Use RTX 4090** instead of A100 when possible (60% cheaper)
4. **Batch requests** to maximize GPU utilization
5. **Monitor and optimize** prompt complexity and max_new_tokens

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile
import runpod
import base64

app = FastAPI()
runpod.api_key = "your-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

@app.post("/ocr")
async def ocr_document(file: UploadFile):
    # Read and encode image
    content = await file.read()
    image_b64 = base64.b64encode(content).decode()

    # Process with OLMocr
    result = endpoint.run_sync({
        "input": {
            "image": image_b64,
            "prompt": "Extract all text preserving layout."
        }
    })

    return {"text": result["text"]}
```

### Async Processing with Queue

```python
import asyncio
from runpod import Endpoint

async def process_queue(image_queue):
    endpoint = Endpoint("your-endpoint-id")

    async def process_one(image_path):
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        return await endpoint.run({
            "input": {"image": image_b64}
        })

    # Process in parallel
    tasks = [process_one(img) for img in image_queue]
    results = await asyncio.gather(*tasks)

    return results
```

## Additional Resources

- [OLMocr Model Card](https://huggingface.co/allenai/olmOCR-7B-0825)
- [OLMocr GitHub Repository](https://github.com/allenai/olmocr)
- [RunPod Documentation](https://docs.runpod.io/)
- [RunPod Serverless Guide](https://docs.runpod.io/serverless/overview)
- [RunPod S3 API Documentation](https://docs.runpod.io/serverless/storage/s3-api)

## Support

- **OLMocr Issues**: [GitHub Issues](https://github.com/allenai/olmocr/issues)
- **RunPod Support**: support@runpod.io
- **RunPod Discord**: [Join Community](https://discord.gg/runpod)

## Next Steps

1. ✅ Deploy OLMocr endpoint
2. ✅ Test with sample documents
3. ⬜ Integrate into your application
4. ⬜ Monitor performance and costs
5. ⬜ Optimize based on usage patterns
6. ⬜ Set up monitoring and alerts
7. ⬜ Scale for production workloads
