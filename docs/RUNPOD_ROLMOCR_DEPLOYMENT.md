# RunPod Serverless Deployment Guide for RolmOCR/OlmOCR

This guide provides step-by-step instructions for deploying the RolmOCR model as a serverless endpoint on RunPod with personal S3 bucket storage.

## Overview

**RolmOCR** is Reducto's implementation based on olmOCR (allenai/olmOCR-7B), using Qwen2.5-VL-7B as the base model. It's optimized for OCR tasks including handwritten and slanted character recognition.

## Prerequisites

- RunPod account with API access
- Docker installed locally (for custom handler development)
- AWS CLI or boto3 library for S3 operations
- Network volume created in RunPod datacenter

## Step 1: Create RunPod S3 Bucket (Network Volume)

### 1.1 Create Network Volume
1. Log into [RunPod Console](https://console.runpod.io/)
2. Navigate to **Storage** > **Network Volumes**
3. Click **Create Network Volume**
4. Select datacenter (e.g., `EUR-IS-1`, `US-CA-2`)
5. Configure size and name
6. Note the **Network Volume ID** (acts as bucket name)

### 1.2 Generate S3 API Keys
1. Go to **Settings** > **API Keys**
2. Click **Create S3 API Key**
3. Save the credentials:
   - **Access Key ID**: `user_***...`
   - **Secret Access Key**: `rps_***...`

## Step 2: Configure S3 Environment Variables

### Required Environment Variables

For RunPod serverless endpoint, configure these environment variables:

```bash
# RunPod S3 Configuration
RUNPOD_ACCESS_KEY_ID=user_***...          # Your RunPod S3 access key
RUNPOD_SECRET_ACCESS_KEY=rps_***...       # Your RunPod S3 secret key
RUNPOD_BUCKET_ENDPOINT_URL=https://s3api-DATACENTER.runpod.io/
RUNPOD_NETWORK_VOLUME_ID=your-volume-id   # Network Volume ID (bucket name)
RUNPOD_REGION=eur-is-1                    # Datacenter region (e.g., eur-is-1, us-ca-2)
```

### S3 Endpoint URLs by Datacenter

| Datacenter | Endpoint URL |
|------------|--------------|
| EUR-IS-1 | `https://s3api-eur-is-1.runpod.io/` |
| US-CA-2 | `https://s3api-us-ca-2.runpod.io/` |
| US-OR-1 | `https://s3api-us-or-1.runpod.io/` |

## Step 3: Create Custom Handler for RolmOCR

Create a `handler.py` file for your RunPod serverless endpoint:

```python
"""
RolmOCR Serverless Handler for RunPod
Handles OCR requests with S3 storage integration
"""

import runpod
import boto3
import base64
import os
import json
from io import BytesIO
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=os.environ.get('RUNPOD_BUCKET_ENDPOINT_URL'),
    aws_access_key_id=os.environ.get('RUNPOD_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('RUNPOD_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('RUNPOD_REGION', 'eur-is-1')
)

# Load RolmOCR model (or olmOCR)
MODEL_NAME = "reducto/RolmOCR"  # or "allenai/olmOCR-7B-0225-preview"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME}")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Model loaded successfully")


def upload_to_s3(file_path, s3_key):
    """Upload file to RunPod S3 bucket."""
    bucket_name = os.environ.get('RUNPOD_NETWORK_VOLUME_ID')
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        print(f"S3 upload error: {e}")
        return None


def download_from_s3(s3_key):
    """Download file from RunPod S3 bucket."""
    bucket_name = os.environ.get('RUNPOD_NETWORK_VOLUME_ID')
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        print(f"S3 download error: {e}")
        return None


def process_image(image_data, prompt="Extract all text from this image."):
    """Process image with RolmOCR model."""
    try:
        # Decode base64 image if needed
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data

        # Load image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Prepare inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate OCR output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024)

        # Decode result
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return {
            "success": True,
            "text": result,
            "model": MODEL_NAME
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def handler(job):
    """
    RunPod handler function.

    Expected input format:
    {
        "input": {
            "image": "base64_encoded_image_or_s3_key",
            "prompt": "Optional custom prompt",
            "upload_result": true,  # Upload result to S3
            "s3_input": false  # If true, treat 'image' as S3 key
        }
    }
    """
    try:
        job_input = job["input"]

        # Get image data
        if job_input.get("s3_input", False):
            # Download from S3
            s3_key = job_input["image"]
            image_data = download_from_s3(s3_key)
            if not image_data:
                return {"error": "Failed to download image from S3"}
        else:
            # Use provided base64 image
            image_data = job_input["image"]

        # Get custom prompt (optional)
        prompt = job_input.get("prompt", "Extract all text from this image.")

        # Process image
        result = process_image(image_data, prompt)

        # Upload result to S3 if requested
        if job_input.get("upload_result", False) and result["success"]:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(result, f)
                temp_path = f.name

            s3_key = f"ocr_results/{job['id']}_result.json"
            s3_url = upload_to_s3(temp_path, s3_key)
            os.unlink(temp_path)

            if s3_url:
                result["s3_url"] = s3_url

        return result

    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
```

## Step 4: Create Dockerfile

Create a `Dockerfile` for your serverless endpoint:

```dockerfile
# Use official PyTorch image with CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Set Python to unbuffered mode
ENV PYTHONUNBUFFERED=1

# Command to run handler
CMD ["python", "-u", "handler.py"]
```

## Step 5: Create requirements.txt

```txt
runpod>=1.7.0
boto3>=1.34.0
torch>=2.1.0
transformers>=4.36.0
accelerate>=0.25.0
Pillow>=10.0.0
```

## Step 6: Build and Push Docker Image

```bash
# Build Docker image
docker build -t your-dockerhub-username/rolmocr-runpod:latest .

# Login to Docker Hub
docker login

# Push image
docker push your-dockerhub-username/rolmocr-runpod:latest
```

## Step 7: Deploy to RunPod Serverless

### Option 1: Using RunPod Console

1. Go to [RunPod Console](https://console.runpod.io/) > **Serverless**
2. Click **Deploy Endpoint**
3. Select **Custom** template
4. Configure:
   - **Name**: `rolmocr-endpoint`
   - **Docker Image**: `your-dockerhub-username/rolmocr-runpod:latest`
   - **GPU Type**: Select GPU (e.g., RTX 4090, A100)
   - **Container Disk**: 20GB minimum
   - **Environment Variables**: Add all S3 configuration variables
5. Click **Deploy**

### Option 2: Using RunPod API

```python
import requests

RUNPOD_API_KEY = "your-runpod-api-key"

endpoint_config = {
    "name": "rolmocr-endpoint",
    "docker_image": "your-dockerhub-username/rolmocr-runpod:latest",
    "gpu_type": "NVIDIA RTX 4090",
    "container_disk_in_gb": 20,
    "env": {
        "RUNPOD_ACCESS_KEY_ID": "user_***...",
        "RUNPOD_SECRET_ACCESS_KEY": "rps_***...",
        "RUNPOD_BUCKET_ENDPOINT_URL": "https://s3api-eur-is-1.runpod.io/",
        "RUNPOD_NETWORK_VOLUME_ID": "your-volume-id",
        "RUNPOD_REGION": "eur-is-1"
    }
}

response = requests.post(
    "https://api.runpod.io/v2/endpoints",
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    json=endpoint_config
)

print(response.json())
```

## Step 8: Test Your Endpoint

### Python Example

```python
import runpod
import base64
from pathlib import Path

# Configure RunPod
runpod.api_key = "your-runpod-api-key"
endpoint_id = "your-endpoint-id"

# Load and encode image
image_path = Path("test_image.png")
with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Run inference
endpoint = runpod.Endpoint(endpoint_id)
result = endpoint.run_sync({
    "input": {
        "image": image_base64,
        "prompt": "Extract all text from this document.",
        "upload_result": True
    }
})

print(result)
```

### cURL Example

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "base64_encoded_image_data",
      "prompt": "Extract all text from this image.",
      "upload_result": true
    }
  }'
```

## Step 9: Configure S3 Access in Code

### Using boto3 for S3 Operations

```python
import boto3
import os

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=os.environ['RUNPOD_BUCKET_ENDPOINT_URL'],
    aws_access_key_id=os.environ['RUNPOD_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['RUNPOD_SECRET_ACCESS_KEY'],
    region_name=os.environ.get('RUNPOD_REGION', 'eur-is-1')
)

# Upload file
bucket_name = os.environ['RUNPOD_NETWORK_VOLUME_ID']
s3_client.upload_file('local_file.json', bucket_name, 'remote/path/file.json')

# Download file
s3_client.download_file(bucket_name, 'remote/path/file.json', 'local_file.json')

# List files
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='ocr_results/')
for obj in response.get('Contents', []):
    print(obj['Key'])
```

## Performance Optimization Tips

### 1. Model Caching
- Cache the model in Docker image to avoid download on cold starts
- Use RunPod's model caching feature

### 2. Batch Processing
- Process multiple images in a single request when possible
- Implement queue-based processing for high volumes

### 3. GPU Selection
- **RTX 4090**: Good balance of cost and performance
- **A100 40GB**: Best for large batches and high throughput
- **A6000**: Good for production workloads

### 4. Scaling Configuration
```json
{
  "min_workers": 0,
  "max_workers": 5,
  "idle_timeout": 30,
  "execution_timeout": 600
}
```

## Cost Optimization

### Storage Costs
- RunPod S3 storage: ~$0.10/GB/month
- Network transfer: Free within same datacenter

### Compute Costs (Serverless)
- Pay only for active inference time
- Cold start: ~30-60 seconds (cached model reduces this)
- Idle workers shut down automatically

### Estimated Costs
- **Light usage** (100 requests/day): ~$5-10/month
- **Medium usage** (1000 requests/day): ~$50-100/month
- **Heavy usage** (10,000 requests/day): ~$500-1000/month

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify model download
huggingface-cli download reducto/RolmOCR
```

**2. S3 Connection Errors**
```python
# Test S3 connectivity
import boto3
s3 = boto3.client('s3', endpoint_url='https://s3api-eur-is-1.runpod.io/')
s3.list_buckets()  # Should not error
```

**3. Out of Memory (OOM)**
- Use smaller batch sizes
- Enable mixed precision (fp16)
- Upgrade to GPU with more VRAM

**4. Cold Start Timeout**
- Increase execution timeout
- Keep min_workers > 0 for critical applications
- Optimize Docker image size

## Security Best Practices

1. **API Keys**: Never commit API keys to version control
2. **Environment Variables**: Use RunPod's secrets management
3. **S3 Permissions**: Use least-privilege access policies
4. **Network**: Use RunPod's private networking for S3
5. **Logging**: Enable CloudWatch or custom logging for audit trails

## Additional Resources

- [RunPod Documentation](https://docs.runpod.io/)
- [RunPod S3 API Guide](https://docs.runpod.io/serverless/storage/s3-api)
- [RolmOCR GitHub](https://github.com/reducto/RolmOCR)
- [RunPod Community Discord](https://discord.gg/runpod)

## Support

For issues specific to:
- **RunPod**: [support@runpod.io](mailto:support@runpod.io)
- **RolmOCR**: Open issue on GitHub
- **This integration**: Contact your development team
