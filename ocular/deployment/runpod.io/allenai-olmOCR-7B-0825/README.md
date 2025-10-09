# OLMocr RunPod Serverless Deployment

Complete guide for deploying **OLMocr (allenai/olmOCR-7B-0825)** on RunPod serverless platform.

## 📦 Deployment Options

This directory supports **two deployment approaches**:

### Option 1: Custom Docker + S3 (Recommended for Production)

**Best for**: Full control, custom handlers, S3 integration

**Features**:
- ✅ Custom Docker container with transformers
- ✅ S3 storage integration for results
- ✅ Complete control over inference pipeline
- ✅ Network volume support for model caching
- ✅ Automated deployment scripts

**Requirements**:
- Docker installed locally
- Docker Hub account
- RunPod S3 credentials

**Quick Start**:
```bash
# Use the unified deployment script
./deploy_olmocr.sh --type docker --docker-user your-username
```

---

### Option 2: vLLM + RunPod CLI (Recommended for Quick Testing)

**Best for**: Quick deployment, vLLM optimization, maximum throughput

**Features**:
- ✅ vLLM optimization for faster inference
- ✅ Uses RunPod CLI (`runpodctl`)
- ✅ Simplified deployment process
- ✅ Optimized for throughput
- ✅ Three deployment modes: quick, custom, template

**Requirements**:
- RunPod CLI (`runpodctl`)
- RunPod API key
- Linux/WSL environment

**Quick Start**:
```bash
# Install RunPod CLI first
wget -qO- cli.runpod.net | sudo bash

# Use the unified deployment script
./deploy_olmocr.sh --type vllm --api-key $RUNPOD_API_KEY
```

---

## 🤔 Which Option to Choose?

| Factor | Custom Docker | vLLM + CLI |
|--------|---------------|------------|
| **Setup Complexity** | Medium | Low |
| **Customization** | High | Medium |
| **S3 Integration** | ✅ Built-in | Manual |
| **Performance** | Good | Better (vLLM) |
| **Local Docker Required** | ✅ Yes | ❌ No |
| **Best For** | Production apps | Quick testing |

### Recommendations:

- **Choose Custom Docker** if you:
  - Need S3 result storage
  - Want full control over the handler
  - Are building a production application
  - Need custom preprocessing/postprocessing

- **Choose vLLM** if you:
  - Want fastest deployment
  - Need maximum throughput
  - Prefer command-line tools
  - Don't need S3 integration

---

## 🚀 Quick Start Guide

### Prerequisites

1. **RunPod Account**: Sign up at [console.runpod.io](https://console.runpod.io)
2. **RunPod API Key**: Get from [console.runpod.io/user/settings](https://console.runpod.io/user/settings)
3. **For Docker**: Docker + Docker Hub account
4. **For vLLM**: RunPod CLI (`runpodctl`)
5. **Optional**: Hugging Face token for gated models

### Unified Deployment Script

The `deploy_olmocr.sh` script supports both deployment types:

```bash
# Show all options
./deploy_olmocr.sh --help

# Docker deployment
./deploy_olmocr.sh --type docker --docker-user myusername

# vLLM quick deployment
./deploy_olmocr.sh --type vllm --mode quick --api-key $RUNPOD_API_KEY

# vLLM custom deployment with specific GPU
./deploy_olmocr.sh --type vllm --mode custom --gpu "NVIDIA A100" --region "EU-RO-1"

# Dry run to preview deployment
./deploy_olmocr.sh --dry-run --verbose
```

---

## 📋 Custom Docker Deployment Details

### 1. Setup RunPod S3 Storage

1. **Create Network Volume**:
   - Go to [RunPod Console](https://console.runpod.io/) > Storage > Network Volumes
   - Click "Create Network Volume"
   - Select datacenter (e.g., EUR-IS-1, US-CA-2)
   - Configure size: **50GB minimum** (for model cache)
   - Note the **Network Volume ID**

2. **Generate S3 API Keys**:
   - Go to Settings > API Keys
   - Click "Create S3 API Key"
   - Save both keys securely:
     - Access Key ID (starts with `user_`)
     - Secret Access Key (starts with `rps_`)

3. **Configure Environment**:
   ```bash
   # Copy environment template
   cp ../.env.example .env

   # Edit .env with your values:
   RUNPOD_ACCESS_KEY_ID=user_***...
   RUNPOD_SECRET_ACCESS_KEY=rps_***...
   RUNPOD_BUCKET_ENDPOINT_URL=https://s3api-eur-is-1.runpod.io/
   RUNPOD_NETWORK_VOLUME_ID=your-volume-id
   RUNPOD_REGION=eur-is-1
   DOCKER_USERNAME=your-dockerhub-username
   ```

### 2. Docker Deployment Options

```bash
# Automated deployment
./deploy_olmocr.sh --type docker

# Manual deployment
docker build -t $DOCKER_USERNAME/olmocr-runpod:latest olmocr/
docker push $DOCKER_USERNAME/olmocr-runpod:latest
python olmocr/deploy_olmocr_api.py
```

### 3. Docker API Usage

**Input Format**:
```json
{
  "input": {
    "image": "base64_encoded_image_or_s3_key",
    "prompt": "Extract all text from this document.",
    "upload_result": true,
    "s3_input": false,
    "max_new_tokens": 2048
  }
}
```

**Output Format**:
```json
{
  "success": true,
  "text": "Extracted text content...",
  "model": "allenai/olmOCR-7B-0825",
  "device": "cuda",
  "s3_url": "s3://bucket-id/ocr_results/job-id_result.json"
}
```

---

## 🔧 vLLM Deployment Details

### 1. Install RunPod CLI

```bash
# Install runpodctl
wget -qO- cli.runpod.net | sudo bash

# Configure CLI
runpodctl config --apiKey $RUNPOD_API_KEY
```

### 2. vLLM Deployment Modes

**Quick Deploy** (Recommended):
```bash
./deploy_olmocr.sh --type vllm --mode quick --api-key $RUNPOD_API_KEY
```

**Custom Deploy** (Uses runpod.toml):
```bash
./deploy_olmocr.sh --type vllm --mode custom --api-key $RUNPOD_API_KEY
```

**Template Deploy**:
```bash
./deploy_olmocr.sh --type vllm --mode template --api-key $RUNPOD_API_KEY
```

### 3. vLLM Configuration

Key environment variables in `runpod-config/runpod.toml`:

```toml
[env]
MODEL_NAME = "allenai/olmOCR-7B-0825"
MODEL_REVISION = "main"
BASE_PATH = "/runpod-volume"
VLLM_ENGINE_ARGS = "--model=allenai/olmOCR-7B-0825 --max-model-len=4096 --gpu-memory-utilization=0.9"
WORKER_CUDA_VERSION = "12.1.0"
HF_TOKEN = "${HF_TOKEN}"
```

### 4. vLLM API Usage

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "model": "allenai/olmOCR-7B-0825",
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Convert this document to markdown."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
          ]
        }
      ],
      "max_tokens": 4096,
      "temperature": 0.1
    }
  }'
```

---

## 💻 GPU Requirements

### For OLMocr-7B Model:

| GPU | VRAM | Performance | Cost/hr | Recommendation |
|-----|------|-------------|---------|----------------|
| **RTX 4090** | 24GB | Excellent | ~$0.30 | ⭐ Best value |
| **A100 40GB** | 40GB | Excellent | ~$1.89 | Production |
| **L40S** | 48GB | Excellent | ~$0.79 | vLLM recommended |
| **H100 80GB** | 80GB | Excellent | ~$2.69 | High throughput |
| **A6000** | 48GB | Good | ~$0.80 | Sustained workloads |

**💡 Minimum Requirements**:
- Docker: 24GB VRAM for FP16 inference
- vLLM: 20GB VRAM (more efficient)

---

## ⚡ Performance Optimization

### Reduce Cold Start Time

**For Docker**:
```dockerfile
# Option A: Pre-download model in Dockerfile (adds ~14GB to image)
RUN python -c "from transformers import AutoModel, AutoProcessor; \
    model = AutoModel.from_pretrained('allenai/olmOCR-7B-0825', trust_remote_code=True); \
    processor = AutoProcessor.from_pretrained('allenai/olmOCR-7B-0825', trust_remote_code=True)"
```

**For vLLM**:
- Use network storage for model caching
- Increase `container_disk_size_gb` in `runpod.toml`
- Set `BASE_PATH=/runpod-volume`

### Set Minimum Workers

Keep instances warm for production:
```bash
# Docker
./deploy_olmocr.sh --min-workers 1 --max-workers 5

# vLLM (edit runpod.toml)
[serverless]
min_workers = 1
max_workers = 5
```

### vLLM Optimization

```toml
[env]
# Use 90% of GPU memory
VLLM_ENGINE_ARGS = "--gpu-memory-utilization=0.9 --max-model-len=4096 --dtype=bfloat16"
```

---

## 💰 Cost Estimates

### Docker Deployment (RTX 4090 @ $0.30/hr):

| Usage | Requests/Day | Monthly Cost |
|-------|--------------|--------------|
| Light | 100 | ~$10-20 |
| Medium | 1,000 | ~$80-150 |
| Heavy | 10,000 | ~$800-1,500 |

### vLLM Deployment (L40S @ $0.79/hr):

| Usage | Requests/Day | Monthly Cost |
|-------|--------------|--------------|
| Light | 100 | ~$25-50 |
| Medium | 1,000 | ~$200-400 |
| Heavy | 10,000 | ~$2,000-4,000 |

**Note**: Serverless pricing means you only pay when actively processing requests.

---

## 🔍 Testing Your Endpoint

### Docker Endpoint Test

```python
import runpod
import base64
from pathlib import Path

runpod.api_key = "your-runpod-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

# Load and encode image
with open("test.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Run inference
result = endpoint.run_sync({
    "input": {
        "image": image_base64,
        "prompt": "Extract all text preserving layout.",
        "upload_result": True
    }
})

print(result["text"])
```

### vLLM Endpoint Test

```python
import runpod

runpod.api_key = "your-runpod-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

result = endpoint.run_sync({
    "input": {
        "model": "allenai/olmOCR-7B-0825",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract text from this document."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }],
        "max_tokens": 4096
    }
})
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. Deployment Fails with Memory Error**
- Use a GPU with more VRAM
- Docker: Reduce `max_new_tokens`
- vLLM: Reduce `max_model_len` in configuration

**2. Model Download Timeout**
- Docker: Increase execution timeout
- vLLM: Increase `container_disk_size_gb`
- Use network storage for model caching

**3. Authentication Errors**
- Verify `RUNPOD_API_KEY` is correct
- Check API key permissions in RunPod console
- For vLLM: Ensure `HF_TOKEN` is set for gated models

**4. Cold Start Too Slow**
- Pre-download model in Docker image
- Use network volume for model caching
- Set `min_workers > 0` to keep instances warm
- vLLM is generally faster for cold starts

### Debug Mode

```bash
# Docker deployment debug
./deploy_olmocr.sh --type docker --verbose --dry-run

# vLLM deployment debug
./deploy_olmocr.sh --type vllm --verbose --dry-run
```

### Manual CLI Commands (vLLM)

```bash
# List endpoints
runpodctl list endpoints

# Check specific endpoint
runpodctl describe endpoint YOUR_ENDPOINT_ID

# View logs
runpodctl logs YOUR_ENDPOINT_ID

# Update endpoint
runpodctl update endpoint YOUR_ENDPOINT_ID --env NEW_VAR=value

# Delete endpoint
runpodctl delete endpoint YOUR_ENDPOINT_ID
```

---

## 📊 Monitoring and Management

### View Deployment Status

**Docker**: Check [RunPod Console > Serverless](https://console.runpod.io/serverless)

**vLLM**:
```bash
runpodctl list endpoints | grep olmocr
```

### Performance Metrics

Monitor in RunPod dashboard:
- Request latency
- GPU utilization
- Memory usage
- Error rates
- Cost tracking

---

## 🌍 S3 Endpoint URLs by Datacenter

| Datacenter | Location | Endpoint URL |
|------------|----------|--------------|
| EUR-IS-1 | Iceland | `https://s3api-eur-is-1.runpod.io/` |
| US-CA-2 | California | `https://s3api-us-ca-2.runpod.io/` |
| US-OR-1 | Oregon | `https://s3api-us-or-1.runpod.io/` |

---

## 📁 File Structure

```
allenai-olmOCR-7B-0825/
├── README.md                   # This file - unified documentation
├── deploy_olmocr.sh           # Unified deployment script (Docker + vLLM)
│
├── olmocr/                    # Docker deployment files
│   ├── handler.py            # RunPod serverless handler
│   ├── Dockerfile            # Container configuration
│   ├── requirements.txt      # Python dependencies
│   ├── deploy_olmocr_api.py  # API deployment script
│   ├── test_endpoint.py      # Testing tool
│   └── .env.example          # Configuration template
│
└── runpod-config/            # vLLM deployment files
    ├── runpod.toml           # RunPod configuration
    ├── requirements.txt      # vLLM dependencies
    ├── .runpodignore         # Deployment exclusions
    ├── runpod_api_example.py # API usage examples
    └── test_deployment.sh    # Testing script
```

---

## 💡 Best Practices

### For Development
- Use vLLM quick deploy for rapid iteration
- Set `min_workers=0` to minimize costs
- Use dry-run mode to preview changes

### For Production
- Use Docker for full control and S3 integration
- Set `min_workers >= 1` to avoid cold starts
- Enable network volumes for model caching
- Monitor costs and performance metrics
- Set up appropriate timeout values
- Use specific GPU types for consistency

### Security
1. Store API keys securely
2. Rotate keys regularly
3. Use environment variables for sensitive data
4. Configure endpoint access permissions
5. Enable network storage encryption

---

## 🔗 Additional Resources

- [OLMocr GitHub Repository](https://github.com/allenai/olmocr)
- [OLMocr Model Card](https://huggingface.co/allenai/olmOCR-7B-0825)
- [RunPod Documentation](https://docs.runpod.io/)
- [RunPod CLI Guide](https://docs.runpod.io/cli/install-runpodctl)
- [vLLM Documentation](https://docs.vllm.ai/)
- [RunPod S3 API Guide](https://docs.runpod.io/serverless/storage/s3-api)
- [Complete Deployment Guide](../../docs/RUNPOD_OLMOCR_DEPLOYMENT.md)
- [RunPod Community Discord](https://discord.gg/runpod)

---

## 📞 Support

For issues and questions:
- **OLMocr Issues**: [GitHub Issues](https://github.com/allenai/olmocr/issues)
- **RunPod Support**: support@runpod.io
- **RunPod Community**: [Discord](https://discord.gg/runpod)
- **This Integration**: Contact development team

---

## 📝 License

This configuration is provided under the same license as the main Ocular project.

---

## 🆚 Model Comparison

| Model | Size | VRAM | Speed | Accuracy | Best For |
|-------|------|------|-------|----------|----------|
| **OLMocr-7B** | 7B | 24GB | Medium | High | General OCR, structured documents |
| **RolmOCR** | ~7B | 24GB | Fast | Very High | High-accuracy OCR, complex layouts |

**Choose OLMocr for**:
- General purpose OCR
- Research and experimentation
- Open-source deployments
- Custom fine-tuning

**Choose RolmOCR for**:
- Production deployments requiring highest accuracy
- Complex document layouts
- Forms and tables
