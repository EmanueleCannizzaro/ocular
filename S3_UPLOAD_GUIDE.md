# S3 Model Upload Guide

## Overview

The Ocular platform now supports uploading AI models and related files to S3-compatible storage (RunPod, AWS S3, or custom S3-compatible services). This allows you to:

- Store model weights in cloud storage
- Share models across deployments
- Build Docker images from S3-hosted models
- Version control model artifacts

## Supported Storage Providers

### 1. RunPod S3-Compatible Storage
- **Provider ID**: `runpod`
- **Endpoint**: `https://storage.runpod.io`
- **Region**: `us-east-1`
- **Best for**: RunPod deployments, fast access within RunPod infrastructure

### 2. AWS S3
- **Provider ID**: `aws`
- **Endpoint**: Standard AWS S3 endpoints
- **Region**: Configurable (e.g., `us-east-1`, `eu-west-1`)
- **Best for**: Enterprise deployments, AWS infrastructure

### 3. Custom S3-Compatible Storage
- **Provider ID**: `custom`
- **Endpoint**: Your custom S3-compatible endpoint
- **Region**: Configurable
- **Best for**: On-premise storage, MinIO, Wasabi, etc.

## Environment Configuration

### RunPod Storage

Add to `.env`:

```bash
# RunPod S3-Compatible Storage
RUNPOD_BUCKET_ACCESS_KEY_ID=user_xxxxxxxxxxxxxxxxxxxxxxxx
RUNPOD_BUCKET_SECRET_ACCESS_KEY=rps_xxxxxxxxxxxxxxxxxxxxxxxx
RUNPOD_BUCKET_ENDPOINT_URL=https://storage.runpod.io
```

### AWS S3

Add to `.env`:

```bash
# AWS S3
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_REGION=us-east-1
```

## API Endpoint

### Upload Model File

**Endpoint**: `POST /api/deployments/models/{model_id}/upload`

**Content-Type**: `multipart/form-data`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Model file to upload (weights, Docker image, etc.) |
| `bucket` | String | Yes | S3 bucket name |
| `provider` | String | No | Storage provider: `runpod`, `aws`, `custom` (default: `runpod`) |
| `make_public` | Boolean | No | Make file publicly accessible (default: `false`) |
| `access_key_id` | String | No | S3 access key (uses env var if not provided) |
| `secret_access_key` | String | No | S3 secret key (uses env var if not provided) |

**Response**:

```json
{
  "success": true,
  "message": "Model uploaded successfully",
  "model_id": 1,
  "model_name": "olmocr",
  "upload": {
    "filename": "model-weights.safetensors",
    "url": "https://storage.runpod.io/my-bucket/models/olmocr/v1.0/model-weights.safetensors",
    "bucket": "my-bucket",
    "key": "models/olmocr/v1.0/model-weights.safetensors",
    "size_bytes": 14536789,
    "provider": "runpod",
    "public": false
  },
  "model": {
    "id": 1,
    "name": "olmocr",
    "version": "v1.0",
    "metadata": {
      "s3_uploads": [
        {
          "filename": "model-weights.safetensors",
          "bucket": "my-bucket",
          "key": "models/olmocr/v1.0/model-weights.safetensors",
          "url": "https://storage.runpod.io/my-bucket/models/olmocr/v1.0/model-weights.safetensors",
          "size_bytes": 14536789,
          "provider": "runpod",
          "uploaded_at": "2025-10-05T12:34:56.789Z",
          "public": false
        }
      ]
    }
  }
}
```

## Usage Examples

### Example 1: Upload Model Weights to RunPod

Using `curl`:

```bash
curl -X POST "http://localhost:8000/api/deployments/models/1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/model-weights.safetensors" \
  -F "bucket=my-runpod-bucket" \
  -F "provider=runpod" \
  -F "make_public=false"
```

Using Python `requests`:

```python
import requests

url = "http://localhost:8000/api/deployments/models/1/upload"

with open("/path/to/model-weights.safetensors", "rb") as f:
    files = {"file": f}
    data = {
        "bucket": "my-runpod-bucket",
        "provider": "runpod",
        "make_public": "false"
    }

    response = requests.post(url, files=files, data=data)
    print(response.json())
```

### Example 2: Upload Docker Image Archive to AWS S3

```bash
curl -X POST "http://localhost:8000/api/deployments/models/1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@docker-olmocr.tar.gz" \
  -F "bucket=my-aws-bucket" \
  -F "provider=aws" \
  -F "make_public=true"
```

### Example 3: Upload with Custom Credentials

If you don't want to use environment variables:

```bash
curl -X POST "http://localhost:8000/api/deployments/models/1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@model-config.json" \
  -F "bucket=my-bucket" \
  -F "provider=runpod" \
  -F "access_key_id=user_xxxxxxxx" \
  -F "secret_access_key=rps_xxxxxxxx"
```

### Example 4: Frontend JavaScript

```javascript
async function uploadModelFile(modelId, file, bucket, provider = 'runpod') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('bucket', bucket);
    formData.append('provider', provider);
    formData.append('make_public', 'false');

    const response = await fetch(`/api/deployments/models/${modelId}/upload`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
    }

    return await response.json();
}

// Usage
const fileInput = document.getElementById('model-file');
const file = fileInput.files[0];

uploadModelFile(1, file, 'my-runpod-bucket', 'runpod')
    .then(result => {
        console.log('Upload successful:', result.upload.url);
        alert(`Model uploaded to: ${result.upload.url}`);
    })
    .catch(error => {
        console.error('Upload failed:', error);
        alert(`Upload failed: ${error.message}`);
    });
```

## S3 Object Key Structure

Files are organized in S3 using the following structure:

```
models/
  {model_name}/
    {version}/
      {filename}
```

**Example**:
- Model: `olmocr`
- Version: `allenai/olmOCR-7B-0825`
- File: `model-weights.safetensors`
- **S3 Key**: `models/olmocr/allenai/olmOCR-7B-0825/model-weights.safetensors`

This structure:
- Groups all files for a model together
- Supports multiple versions of the same model
- Preserves original filename
- Enables easy browsing in S3 console

## Model Metadata Tracking

Each upload is tracked in the model's `extra_metadata` field:

```json
{
  "s3_uploads": [
    {
      "filename": "model-weights.safetensors",
      "bucket": "my-bucket",
      "key": "models/olmocr/v1.0/model-weights.safetensors",
      "url": "https://storage.runpod.io/my-bucket/models/olmocr/v1.0/model-weights.safetensors",
      "size_bytes": 14536789,
      "provider": "runpod",
      "uploaded_at": "2025-10-05T12:34:56.789Z",
      "public": false
    }
  ]
}
```

This allows you to:
- View upload history
- Track multiple versions
- Find S3 URLs for deployments
- Monitor storage usage

## Docker Image Uploads

When uploading Docker images (`.tar`, `.tar.gz`, `.tgz` files with "docker" in the name), the model's `docker_image` field is automatically updated with the S3 URL.

**Example**:

```bash
# Upload Docker image
curl -X POST "http://localhost:8000/api/deployments/models/1/upload" \
  -F "file=@olmocr-docker-image.tar.gz" \
  -F "bucket=my-bucket" \
  -F "provider=runpod"
```

After upload, the model's `docker_image` field will be:
```
https://storage.runpod.io/my-bucket/models/olmocr/v1.0/olmocr-docker-image.tar.gz
```

This URL can then be used in deployment configurations.

## Access Control

### Public Files

Set `make_public=true` to make files publicly accessible:

```bash
curl -X POST "http://localhost:8000/api/deployments/models/1/upload" \
  -F "file=@model-config.json" \
  -F "bucket=my-bucket" \
  -F "make_public=true"
```

Public files can be accessed without authentication:
```
https://storage.runpod.io/my-bucket/models/olmocr/v1.0/model-config.json
```

### Private Files

By default, files are private and require authenticated access.

For deployments, ensure your deployment environment has access to the S3 credentials:

```bash
# In deployment environment
export RUNPOD_BUCKET_ACCESS_KEY_ID="user_xxx"
export RUNPOD_BUCKET_SECRET_ACCESS_KEY="rps_xxx"
```

## Presigned URLs (Future Enhancement)

For temporary access to private files, the S3 service supports presigned URLs:

```python
from ocular.services.s3_service import get_s3_service

s3_service = get_s3_service("runpod")

# Generate URL valid for 1 hour
url = s3_service.generate_presigned_url(
    bucket="my-bucket",
    object_key="models/olmocr/v1.0/weights.safetensors",
    expiration=3600  # seconds
)

print(f"Temporary URL: {url}")
```

## Best Practices

### 1. Use Versioned Paths

Always include version information in your uploads:

```
✅ Good: models/olmocr/v1.0/weights.safetensors
✅ Good: models/olmocr/allenai-olmOCR-7B-0825/weights.safetensors
❌ Bad:  models/olmocr/weights.safetensors
```

### 2. Organize by Model and Version

Keep all files for a model version together:

```
models/olmocr/v1.0/
  ├── weights.safetensors
  ├── config.json
  ├── tokenizer.json
  └── docker-image.tar.gz
```

### 3. Use Descriptive Filenames

```
✅ Good: olmocr-v1.0-fp16-weights.safetensors
✅ Good: olmocr-dockerfile-cuda11.8.tar.gz
❌ Bad:  model.bin
❌ Bad:  docker.tar
```

### 4. Track Upload Metadata

The model's `extra_metadata.s3_uploads` array tracks all uploads. Use this to:
- Verify uploads completed
- Find S3 URLs for deployments
- Monitor storage costs

### 5. Security

- **Never commit credentials** to version control
- Use **environment variables** for credentials
- Set `make_public=false` for sensitive models
- Rotate credentials regularly

## Integration with Deployments

Once models are uploaded to S3, you can reference them in deployments:

### 1. Update Model Configuration

```python
# Get model from database
model = db.query(Model).filter(Model.id == 1).first()

# Get S3 URL from uploads
s3_uploads = model.extra_metadata.get("s3_uploads", [])
if s3_uploads:
    latest_upload = s3_uploads[-1]
    model_url = latest_upload["url"]

    # Use in deployment
    deployment_config = {
        "model_url": model_url,
        "model_weights_path": "/models/weights.safetensors"
    }
```

### 2. Reference in Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Download model from S3 during build
ARG MODEL_URL
RUN apt-get update && apt-get install -y wget
RUN wget ${MODEL_URL} -O /models/weights.safetensors

# ... rest of Dockerfile
```

Build with:

```bash
docker build \
  --build-arg MODEL_URL="https://storage.runpod.io/my-bucket/models/olmocr/v1.0/weights.safetensors" \
  -t olmocr:latest .
```

### 3. Download at Runtime

```python
# In deployment handler
import boto3
import os

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('RUNPOD_BUCKET_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('RUNPOD_BUCKET_SECRET_ACCESS_KEY'),
    endpoint_url='https://storage.runpod.io'
)

# Download model weights
s3_client.download_file(
    'my-bucket',
    'models/olmocr/v1.0/weights.safetensors',
    '/tmp/weights.safetensors'
)

# Load model
model = load_model('/tmp/weights.safetensors')
```

## Troubleshooting

### Error: "S3 credentials not provided"

**Solution**: Add credentials to `.env`:

```bash
# For RunPod
RUNPOD_BUCKET_ACCESS_KEY_ID=user_xxx
RUNPOD_BUCKET_SECRET_ACCESS_KEY=rps_xxx

# For AWS
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

Or pass credentials in the request:

```bash
curl -X POST "..." \
  -F "access_key_id=user_xxx" \
  -F "secret_access_key=rps_xxx"
```

### Error: "Upload failed: Access Denied"

**Causes**:
1. Invalid credentials
2. Bucket doesn't exist
3. Insufficient permissions

**Solution**:
1. Verify credentials are correct
2. Check bucket exists: `aws s3 ls s3://my-bucket --endpoint-url https://storage.runpod.io`
3. Ensure credentials have `s3:PutObject` permission

### Error: "Upload failed: No such bucket"

**Solution**: Create the bucket first:

```bash
# RunPod
aws s3 mb s3://my-bucket --endpoint-url https://storage.runpod.io

# AWS S3
aws s3 mb s3://my-bucket --region us-east-1
```

### Error: "Upload failed: File too large"

**Causes**:
1. Request timeout (default 2 minutes)
2. Memory limits

**Solution**:
- Use multipart upload for files > 5GB (future enhancement)
- Increase timeout in deployment configuration
- Upload from server with better bandwidth

## Future Enhancements

### 1. Multipart Upload
For large files (>5GB), implement multipart uploads:

```python
# Future API
POST /api/deployments/models/{id}/upload/initiate
POST /api/deployments/models/{id}/upload/part
POST /api/deployments/models/{id}/upload/complete
```

### 2. Direct Browser Upload
Generate presigned POST URLs for direct browser-to-S3 uploads:

```javascript
// Future API
const presignedPost = await fetch('/api/s3/presigned-post');
// Upload directly to S3 from browser
```

### 3. Model Registry
Browse uploaded models via UI:

```
/models → View all models
/models/{id} → View model details
/models/{id}/files → List S3 uploads
```

### 4. Automatic Cleanup
Delete old model versions from S3:

```
POST /api/deployments/models/{id}/cleanup
  - Keep only last N versions
  - Delete files older than X days
```

## References

- **RunPod Storage Docs**: https://docs.runpod.io/pods/storage/network-volumes
- **AWS S3 API**: https://docs.aws.amazon.com/s3/
- **Boto3 Documentation**: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
- **S3 Service Implementation**: [ocular/services/s3_service.py](ocular/services/s3_service.py)
- **API Endpoint**: [ocular/routers/api.py](ocular/routers/api.py) (line 274)

## Summary

The S3 upload functionality provides:

✅ Multi-provider support (RunPod, AWS, custom)
✅ Automatic metadata tracking
✅ Docker image detection and linking
✅ Flexible credential management
✅ Public/private access control
✅ Version-aware file organization
✅ Integration with deployment system

This enables seamless model management and deployment workflows for the Ocular platform.
