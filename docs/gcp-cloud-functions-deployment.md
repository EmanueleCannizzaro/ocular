# Google Cloud Functions Deployment Guide

This guide explains how to deploy the Ocular OCR Service to Google Cloud Functions for serverless operation.

## Overview

Google Cloud Functions provides a serverless environment that automatically scales based on demand, making it ideal for OCR services that may have variable usage patterns. This deployment approach offers:

- **Zero server management**: No need to manage infrastructure
- **Automatic scaling**: Scales from 0 to thousands of instances automatically
- **Pay-per-use**: Only pay when your function is executing
- **High availability**: Built-in reliability and fault tolerance

## Prerequisites

Before deploying, ensure you have:

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** (gcloud) installed and configured
3. **Project with required APIs enabled**:
   - Cloud Functions API
   - Cloud Build API
   - Cloud Run API (used by Cloud Functions Gen2)

### Install Google Cloud SDK

```bash
# On macOS
brew install --cask google-cloud-sdk

# On Ubuntu/Debian
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize and authenticate
gcloud init
gcloud auth login
```

## Quick Deployment

The fastest way to deploy is using the provided deployment script:

```bash
# Make the script executable (if not already done)
chmod +x deploy.sh

# Deploy with your project ID
./deploy.sh -p your-project-id

# Deploy with custom settings
./deploy.sh -p your-project-id -r europe-west1 -m 4GB -t 600s
```

## Manual Deployment Steps

If you prefer to deploy manually or need more control:

### 1. Set Up Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

**Important**: For Cloud Functions, you'll need to set environment variables through the deployment command or Cloud Console, not through .env files.

### 2. Configure Your Project

```bash
# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

### 3. Deploy the Function

```bash
gcloud functions deploy ocular-ocr-service \
  --source . \
  --entry-point ocular_ocr \
  --runtime python311 \
  --trigger http \
  --allow-unauthenticated \
  --region us-central1 \
  --memory 2GB \
  --timeout 540s \
  --max-instances 10 \
  --set-env-vars ENVIRONMENT=production,MISTRAL_API_KEY=your_api_key_here
```

### 4. Verify Deployment

Test the deployed function:

```bash
# Get the function URL
FUNCTION_URL=$(gcloud functions describe ocular-ocr-service --region=us-central1 --format="value(httpsTrigger.url)")

# Test health endpoint
curl "$FUNCTION_URL/health"

# Test the web interface
open "$FUNCTION_URL"
```

## Configuration Options

### Function Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--memory` | 2GB | Memory allocation (128MB to 8GB) |
| `--timeout` | 540s | Function timeout (1s to 540s) |
| `--max-instances` | 10 | Maximum concurrent instances |
| `--region` | us-central1 | Deployment region |

### Environment Variables

Set these through the deployment command or Cloud Console:

#### Required Variables

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
```

#### Optional Variables

```bash
# Additional OCR providers
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_PROJECT_ID=your-project-id
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AZURE_DOC_INTEL_ENDPOINT=your-azure-endpoint
AZURE_DOC_INTEL_API_KEY=your-azure-key

# Application settings
MAX_FILE_SIZE_MB=10
TIMEOUT_SECONDS=300
MAX_RETRIES=3
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Setting Environment Variables

#### During Deployment

```bash
gcloud functions deploy ocular-ocr-service \
  --set-env-vars MISTRAL_API_KEY=your_key,MAX_FILE_SIZE_MB=20,LOG_LEVEL=DEBUG \
  # ... other parameters
```

#### After Deployment

```bash
gcloud functions deploy ocular-ocr-service \
  --update-env-vars NEW_VAR=new_value \
  --source .
```

## Advanced Deployment Options

### Using Cloud Build for CI/CD

The included `cloudbuild.yaml` enables automated deployment:

1. **Connect your repository** to Cloud Build
2. **Create a build trigger** pointing to your repository
3. **Configure trigger settings**:
   - Trigger type: Push to branch
   - Branch: `^main$` or your preferred branch
   - Configuration: Cloud Build configuration file
   - Location: `cloudbuild.yaml`

### Custom Deployment Script Options

The `deploy.sh` script supports various options:

```bash
# Full deployment with authentication required
./deploy.sh -p my-project -r europe-west1 -m 4GB -t 600s -i 20 --auth

# Development deployment
./deploy.sh -p my-dev-project -e development

# High-performance deployment
./deploy.sh -p my-project -m 8GB -t 540s -i 50
```

### Multi-region Deployment

Deploy to multiple regions for better global performance:

```bash
# Deploy to US
./deploy.sh -p my-project -r us-central1 -n ocular-ocr-us

# Deploy to Europe
./deploy.sh -p my-project -r europe-west1 -n ocular-ocr-eu

# Deploy to Asia
./deploy.sh -p my-project -r asia-northeast1 -n ocular-ocr-asia
```

## Monitoring and Troubleshooting

### View Logs

```bash
# View recent logs
gcloud functions logs read ocular-ocr-service --region=us-central1

# Follow logs in real-time
gcloud functions logs tail ocular-ocr-service --region=us-central1
```

### Monitor Performance

Use Cloud Monitoring to track:
- **Invocation count**: Number of function calls
- **Duration**: Execution time per request
- **Memory usage**: Peak memory consumption
- **Error rate**: Failed requests percentage

Access metrics at: https://console.cloud.google.com/monitoring

### Common Issues and Solutions

#### 1. Function Timeout

**Problem**: Requests timeout after 540s (maximum)

**Solutions**:
- Optimize OCR processing for large files
- Implement file chunking for very large documents
- Use asynchronous processing with Cloud Tasks

#### 2. Memory Errors

**Problem**: Function runs out of memory

**Solutions**:
```bash
# Increase memory allocation
gcloud functions deploy ocular-ocr-service \
  --update-config \
  --memory 4GB
```

#### 3. Cold Start Latency

**Problem**: First request after idle period is slow

**Solutions**:
- Use Cloud Scheduler for keep-alive requests
- Consider Cloud Run for always-warm instances
- Optimize import statements and startup code

#### 4. Environment Variable Issues

**Problem**: API keys not loading properly

**Check**:
```bash
# List current environment variables
gcloud functions describe ocular-ocr-service \
  --region=us-central1 \
  --format="value(environmentVariables)"
```

## Cost Optimization

### Pricing Factors

Cloud Functions pricing is based on:
- **Invocations**: Number of function calls
- **Compute time**: GB-seconds of execution
- **Networking**: Egress data transfer

### Optimization Tips

1. **Right-size memory allocation**:
   - Start with 2GB and monitor usage
   - Adjust based on actual memory consumption

2. **Optimize timeout settings**:
   - Set timeout as low as safely possible
   - Implement early termination for failed requests

3. **Use appropriate max instances**:
   - Prevent cost spikes from traffic bursts
   - Balance between performance and cost

4. **Monitor usage patterns**:
   - Use Cloud Monitoring to identify optimization opportunities
   - Consider switching to Cloud Run for consistent high traffic

### Example Cost Analysis

For a function with:
- 2GB memory allocation
- 300s average execution time
- 1,000 requests per month

Estimated cost: ~$8-12 USD per month

## Security Best Practices

### 1. API Authentication

For production deployments, remove `--allow-unauthenticated`:

```bash
gcloud functions deploy ocular-ocr-service \
  --remove-flags allow-unauthenticated
  # ... other parameters
```

Then configure authentication:
```bash
# Allow specific users
gcloud functions add-iam-policy-binding ocular-ocr-service \
  --member="user:user@example.com" \
  --role="roles/cloudfunctions.invoker" \
  --region=us-central1

# Allow service accounts
gcloud functions add-iam-policy-binding ocular-ocr-service \
  --member="serviceAccount:service@project.iam.gserviceaccount.com" \
  --role="roles/cloudfunctions.invoker" \
  --region=us-central1
```

### 2. Environment Variable Security

- Store sensitive values in Secret Manager
- Use IAM roles for service-to-service authentication
- Regularly rotate API keys

### 3. Network Security

```bash
# Deploy with VPC connector for private networks
gcloud functions deploy ocular-ocr-service \
  --vpc-connector=projects/PROJECT_ID/locations/REGION/connectors/CONNECTOR_NAME \
  # ... other parameters
```

## Updating and Maintenance

### Update Function Code

```bash
# Deploy updates
./deploy.sh -p your-project-id

# Or manually
gcloud functions deploy ocular-ocr-service \
  --source . \
  # ... other parameters (same as original deployment)
```

### Update Dependencies

1. Update `requirements.txt`
2. Redeploy the function
3. Test thoroughly

### Rollback if Needed

```bash
# List recent deployments
gcloud functions describe ocular-ocr-service --region=us-central1

# Rollback is not directly supported, but you can:
# 1. Redeploy from a previous Git commit
# 2. Keep backup versions with different names
```

## Alternative Deployment Options

### Cloud Run

For more control or higher resource requirements:

```bash
# Build container image
gcloud builds submit --tag gcr.io/PROJECT_ID/ocular-ocr

# Deploy to Cloud Run
gcloud run deploy ocular-ocr-service \
  --image gcr.io/PROJECT_ID/ocular-ocr \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600
```

### App Engine

For applications requiring longer execution times:

Create `app.yaml`:
```yaml
runtime: python311
service: ocular-ocr

automatic_scaling:
  min_instances: 0
  max_instances: 10

resources:
  cpu: 2
  memory_gb: 4
  disk_size_gb: 10

env_variables:
  MISTRAL_API_KEY: "your_api_key_here"
  ENVIRONMENT: "production"
```

Deploy:
```bash
gcloud app deploy app.yaml
```

## Support and Troubleshooting

### Getting Help

1. **Google Cloud Support**: Available with paid support plans
2. **Stack Overflow**: Use tags `google-cloud-functions`, `python`, `ocr`
3. **Google Cloud Community**: https://cloud.google.com/community
4. **Project Issues**: Submit issues to the project repository

### Useful Commands

```bash
# Function status
gcloud functions describe ocular-ocr-service --region=us-central1

# Delete function
gcloud functions delete ocular-ocr-service --region=us-central1

# List all functions
gcloud functions list

# Get function URL
gcloud functions describe FUNCTION_NAME --region=REGION --format="value(httpsTrigger.url)"
```

## Conclusion

Google Cloud Functions provides an excellent serverless platform for deploying the Ocular OCR Service. With proper configuration and monitoring, you can achieve high availability, automatic scaling, and cost-effective operation.

For questions or issues specific to this deployment, please refer to the project documentation or submit an issue to the repository.