# Multi-Platform AI Model Deployment Templates

Deploy OLMocr (allenai/olmOCR-7B-0225-preview) with vLLM across multiple cloud GPU platforms using a unified configuration system.

## 🚀 Supported Platforms

| Platform | GPU Types | Scaling | Deployment Method | Cost |
|----------|-----------|---------|-------------------|------|
| **RunPod** | L40S, A100, H100 | Serverless | CLI/API | ⭐⭐⭐ |
| **Modal** | T4, L4, A100, H100 | Serverless | Python SDK | ⭐⭐ |
| **Replicate** | T4, A40, A100 | Serverless | Cog/Docker | ⭐⭐ |
| **Google Cloud Run** | L4, T4, V100, A100 | Managed | Docker/gcloud | ⭐ |
| **DigitalOcean** | RTX 6000, H100 | App Platform | CLI/Git | ⭐⭐ |
| **Lambda Labs** | A6000, A100, H100 | Instances | API/SSH | ⭐⭐⭐ |

## 📁 Directory Structure

```
gpu-deployment-templates/
├── multi-platform-config.yaml    # Unified configuration
├── deploy.py                     # Python deployment orchestrator
├── README.md                     # This file
├── runpod/
│   ├── runpod.toml
│   ├── deploy_runpod.sh
│   └── test_deployment.sh
├── modal/
│   ├── modal_app.py
│   └── requirements.txt
├── replicate/
│   ├── cog.yaml
│   ├── predict.py
│   └── requirements.txt
├── google-cloud-run/
│   ├── Dockerfile
│   ├── cloud_run_app.py
│   └── deploy_cloudrun.sh
├── digitalocean/
│   ├── .do/app.yaml
│   └── deploy_digitalocean.sh
└── lambda-labs/
    └── deploy_lambda.sh
```

## 🛠 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd gpu-deployment-templates

# Install dependencies for the orchestrator
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your API keys:

```bash
# Required for most platforms
HF_TOKEN=your_huggingface_token

# Platform-specific API keys
RUNPOD_API_KEY=your_runpod_key
MODAL_TOKEN=your_modal_token
REPLICATE_API_TOKEN=your_replicate_token
GOOGLE_CLOUD_PROJECT=your_gcp_project
DIGITALOCEAN_ACCESS_TOKEN=your_do_token
LAMBDA_API_KEY=your_lambda_key

# Optional: Model configuration
MODEL_NAME=allenai/olmOCR-7B-0225-preview
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.9
```

### 3. Deploy to Single Platform

Choose your platform and deploy:

#### RunPod (Recommended for cost-effectiveness)
```bash
cd runpod
./deploy_runpod.sh --api-key $RUNPOD_API_KEY --hf-token $HF_TOKEN
```

#### Modal (Recommended for ease of use)
```bash
cd modal
modal deploy modal_app.py
```

#### Google Cloud Run (Recommended for enterprise)
```bash
cd google-cloud-run
./deploy_cloudrun.sh --project-id $GOOGLE_CLOUD_PROJECT --hf-token $HF_TOKEN
```

#### Replicate (Recommended for public APIs)
```bash
cd replicate
cog deploy
```

#### DigitalOcean (Recommended for simplicity)
```bash
cd digitalocean
./deploy_digitalocean.sh --repo-url https://github.com/user/repo --hf-token $HF_TOKEN
```

#### Lambda Labs (Recommended for dedicated instances)
```bash
cd lambda-labs
./deploy_lambda.sh --lambda-key $LAMBDA_API_KEY --hf-token $HF_TOKEN
```

### 4. Multi-Platform Deployment

Use the unified orchestrator to deploy to multiple platforms:

```bash
# Deploy to all enabled platforms
python deploy.py --config multi-platform-config.yaml --strategy multi_platform

# Deploy to specific platforms
python deploy.py --platforms runpod,modal --hf-token $HF_TOKEN

# Cost-optimized deployment
python deploy.py --strategy cost_optimized

# Performance-optimized deployment
python deploy.py --strategy performance_optimized
```

## 📋 Platform-Specific Guides

### RunPod Serverless

**Best for**: Cost-effective serverless inference with high-end GPUs

**Setup**:
1. Install RunPod CLI: `pip install runpod`
2. Configure: `runpod config`
3. Deploy: `./runpod/deploy_runpod.sh`

**Features**:
- Auto-scaling from 0 to N instances
- Pay-per-request pricing
- Global edge locations
- Custom container support

**GPU Options**: NVIDIA L40S (48GB), A100 (80GB), H100 (80GB)

### Modal

**Best for**: Python developers wanting seamless serverless deployment

**Setup**:
1. Install Modal: `pip install modal`
2. Set token: `modal token set`
3. Deploy: `modal deploy modal/modal_app.py`

**Features**:
- Python-native deployment
- Automatic scaling and GPU allocation
- Built-in volume management
- Cold start optimization

**GPU Options**: T4, L4, L40S, A10, A100, H100, H200, B200

### Replicate

**Best for**: Public API deployment with versioning

**Setup**:
1. Install Cog: `curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m) && chmod +x /usr/local/bin/cog`
2. Login: `cog login`
3. Deploy: `cog deploy`

**Features**:
- Public API with automatic documentation
- Model versioning and rollbacks
- Integrated monitoring and analytics
- OpenAI-compatible endpoints

**GPU Options**: T4, A40 (48GB), A100 (40GB/80GB)

### Google Cloud Run

**Best for**: Enterprise deployments with Google Cloud integration

**Setup**:
1. Install gcloud: [Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Authenticate: `gcloud auth login`
3. Deploy: `./google-cloud-run/deploy_cloudrun.sh`

**Features**:
- Fully managed serverless platform
- Integrated with Google Cloud services
- Advanced monitoring and logging
- VPC and IAM integration

**GPU Options**: L4, T4, V100, A100

### DigitalOcean App Platform

**Best for**: Simple Git-based deployments

**Setup**:
1. Install doctl: [CLI Guide](https://docs.digitalocean.com/reference/doctl/how-to/install/)
2. Authenticate: `doctl auth init`
3. Deploy: `./digitalocean/deploy_digitalocean.sh`

**Features**:
- Git-based continuous deployment
- Automatic builds from GitHub/GitLab
- Integrated databases and caching
- Simple pricing and scaling

**GPU Options**: RTX 6000 (24GB), H100 (80GB)

### Lambda Labs

**Best for**: Dedicated GPU instances with full control

**Setup**:
1. Get API key: [Lambda Labs](https://cloud.lambdalabs.com/api-keys)
2. Deploy: `./lambda-labs/deploy_lambda.sh`

**Features**:
- Bare-metal GPU instances
- Full root access and customization
- High-performance networking
- Cost-effective for long-running workloads

**GPU Options**: A6000 (48GB), A100 SXM4 (80GB), H100 PCIe (80GB)

## 🔧 Configuration

### Global Configuration

Edit `multi-platform-config.yaml` to customize:

```yaml
global:
  model_name: "allenai/olmOCR-7B-0225-preview"
  resources:
    min_gpu_memory_gb: 16
    recommended_gpu_memory_gb: 20
  scaling:
    min_instances: 0
    max_instances: 5
```

### Platform-Specific Settings

Each platform has dedicated configuration sections:

```yaml
platforms:
  runpod:
    enabled: true
    config:
      gpu_types: ["NVIDIA L40S", "NVIDIA A100"]
      regions: ["US-OR-1", "US-TX-1"]

  modal:
    enabled: true
    config:
      gpu_config:
        type: "A100"
        count: 1
```

### Environment Variables

Common environment variables across platforms:

```bash
# Model Configuration
MODEL_NAME=allenai/olmOCR-7B-0225-preview
MODEL_REVISION=main
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.9
DTYPE=bfloat16
TRUST_REMOTE_CODE=true

# Performance Tuning
VLLM_ENGINE_ARGS="--max-model-len=4096 --gpu-memory-utilization=0.9"
TENSOR_PARALLEL_SIZE=1

# Security
HF_TOKEN=your_token_here
API_SECRET_KEY=your_secret_key
```

## 🧪 Testing

### Health Checks

All platforms expose standard health check endpoints:

```bash
# Health check
curl https://your-endpoint/health

# Model information
curl https://your-endpoint/v1/models

# Test completion
curl -X POST https://your-endpoint/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "allenai/olmOCR-7B-0225-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract text from this image"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ],
    "max_tokens": 512
  }'
```

### Platform-Specific Testing

Each platform includes testing scripts:

```bash
# RunPod
./runpod/test_deployment.sh --endpoint-id abc123 --api-key your_key

# Modal
modal run modal/modal_app.py

# Google Cloud Run
curl https://your-service-url/health
```

## 📊 Monitoring

### Cost Monitoring

Track costs across platforms:

```yaml
cost_optimization:
  cost_alerts:
    daily_limit_usd: 100
    weekly_limit_usd: 500
    monthly_limit_usd: 2000
```

### Performance Monitoring

Monitor key metrics:

- Response time (target: <5s)
- Error rate (target: <5%)
- GPU utilization (target: >80%)
- Throughput (requests/minute)

### Platform-Specific Monitoring

- **RunPod**: Built-in dashboard + custom webhooks
- **Modal**: Integrated logging and metrics
- **Replicate**: Analytics dashboard
- **Google Cloud Run**: Cloud Monitoring integration
- **DigitalOcean**: App Platform monitoring
- **Lambda Labs**: Custom monitoring via SSH

## 🛡 Security

### API Keys Management

Store sensitive data securely:

```bash
# Use environment variables
export HF_TOKEN="your_token"

# Use platform secret managers
# Google Cloud: Secret Manager
# AWS: Systems Manager Parameter Store
# Azure: Key Vault
```

### Network Security

- Enable HTTPS on all endpoints
- Use API keys for authentication
- Implement rate limiting
- Configure CORS appropriately

### Input Validation

All platforms include:

- File type validation (PDF, PNG, JPEG)
- File size limits (50MB default)
- Image dimension limits (1024px max)
- Content security scanning

## 💰 Cost Comparison

### Estimated Costs (per 1000 requests)

| Platform | GPU Type | Cold Start | Warm Request | Monthly (1K req/day) |
|----------|----------|------------|--------------|----------------------|
| RunPod | L40S | $0.05 | $0.10 | ~$30 |
| Modal | A100 | $0.08 | $0.15 | ~$50 |
| Replicate | A100 | $0.10 | $0.20 | ~$65 |
| Google Cloud Run | L4 | $0.12 | $0.18 | ~$58 |
| DigitalOcean | H100 | $0.15 | $0.25 | ~$80 |
| Lambda Labs | A100 | $0.20/hour | $0.20/hour | ~$150 (24/7) |

*Costs are estimates and vary based on usage patterns, regions, and promotional pricing.*

### Cost Optimization Tips

1. **Use serverless platforms** (RunPod, Modal, Replicate) for variable workloads
2. **Choose appropriate GPU types** - L40S for cost, H100 for performance
3. **Implement auto-scaling** to minimize idle time
4. **Use model quantization** (bfloat16) to reduce memory usage
5. **Enable caching** for repeated requests
6. **Monitor usage** and set cost alerts

## 🚨 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check GPU memory
nvidia-smi

# Verify model access
huggingface-cli login

# Check model compatibility
python -c "from transformers import AutoModel; AutoModel.from_pretrained('allenai/olmOCR-7B-0225-preview')"
```

#### API Connectivity Issues
```bash
# Test network connectivity
curl -I https://api.platform.com

# Verify API keys
curl -H "Authorization: Bearer $API_KEY" https://api.platform.com/v1/user

# Check rate limits
curl -I https://your-endpoint/health
```

#### Performance Issues
```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Check memory usage
free -h

# Profile request handling
time curl -X POST ... your-endpoint
```

### Platform-Specific Issues

#### RunPod
- **Cold starts**: Enable keep-warm instances
- **Timeouts**: Increase worker timeout in configuration
- **GPU allocation**: Check regional availability

#### Modal
- **Import errors**: Ensure all dependencies in requirements.txt
- **Volume mounting**: Verify volume configuration and permissions
- **Secret access**: Check secret names and scopes

#### Google Cloud Run
- **Memory limits**: Increase memory allocation for large models
- **Cold starts**: Use min-instances > 0
- **Permissions**: Verify service account roles

### Getting Help

1. **Platform Documentation**:
   - [RunPod Docs](https://docs.runpod.io/)
   - [Modal Docs](https://modal.com/docs)
   - [Replicate Docs](https://replicate.com/docs)
   - [Google Cloud Run Docs](https://cloud.google.com/run/docs)
   - [DigitalOcean Docs](https://docs.digitalocean.com/products/app-platform/)
   - [Lambda Labs Docs](https://docs.lambdalabs.com/)

2. **Community Support**:
   - Platform-specific Discord/Slack channels
   - GitHub issues and discussions
   - Stack Overflow with platform tags

3. **Professional Support**:
   - Enterprise support plans available for most platforms
   - Dedicated account managers for high-volume usage

## 🎯 Best Practices

### Development Workflow

1. **Local Testing**: Test with small models locally first
2. **Staging Deployment**: Deploy to low-cost platform for testing
3. **Performance Testing**: Load test with realistic traffic
4. **Production Deployment**: Deploy to multiple platforms for redundancy
5. **Monitoring**: Set up comprehensive monitoring and alerting

### Model Optimization

1. **Quantization**: Use bfloat16 or int8 quantization
2. **Batching**: Implement dynamic batching for better throughput
3. **Caching**: Cache model outputs for repeated requests
4. **Preprocessing**: Optimize image preprocessing pipelines

### Scaling Strategy

1. **Start Small**: Begin with single platform deployment
2. **Monitor Usage**: Track request patterns and costs
3. **Scale Gradually**: Add platforms based on demand
4. **Optimize Continuously**: Regular performance and cost reviews

## 📝 License

This deployment template is provided under the MIT License. Individual platforms and services have their own terms of service.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-platform`
3. Add your platform-specific templates
4. Update documentation
5. Submit pull request

## 📞 Support

For issues with this deployment template:
- Open GitHub issue with platform-specific labels
- Include logs and configuration details
- Specify which platform(s) are affected

For platform-specific issues:
- Contact platform support directly
- Check platform status pages
- Review platform documentation

---

**Happy Deploying! 🚀**