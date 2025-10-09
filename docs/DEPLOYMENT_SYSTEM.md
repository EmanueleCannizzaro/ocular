# AI Model Deployment System

## Overview

The Ocular deployment system enables automated deployment of AI OCR models to cloud GPU providers. When a deployment is created via the API, the system executes actual deployment scripts to provision cloud infrastructure.

## Architecture

### Components

1. **Database Layer** (`ocular/database/`)
   - Normalized schema with 3 tables: `providers`, `models`, `deployments`
   - Tracks deployment state and configuration
   - Stores deployment metadata and costs

2. **Service Layer** (`ocular/services/deployment_service.py`)
   - `DeploymentService` - Executes deployment scripts
   - Provider-specific deployment methods:
     - `deploy_to_runpod()` - RunPod serverless deployment via GraphQL API
     - `deploy_to_vastai()` - Vast.ai deployment (coming soon)
     - `deploy_to_lambda()` - Lambda Labs deployment (coming soon)

3. **API Layer** (`ocular/routers/api.py`)
   - REST endpoints for deployment management
   - Integrates database and deployment service
   - Handles deployment lifecycle

4. **Frontend** (`ocular/templates/deployments.html`)
   - Web interface for managing deployments
   - Real-time status updates
   - Cost tracking and metrics

## Deployment Flow

### Creating a Deployment

```
User Request → API Endpoint → Database Record → Deployment Service → Cloud Provider
```

1. **User submits deployment request** via `/api/deployments` endpoint
2. **API validates** provider and model exist
3. **Database record created** with status `PENDING`
4. **Deployment service executes**:
   - Builds configuration from deployment + model + provider data
   - Calls cloud provider API (e.g., RunPod GraphQL)
   - Updates deployment with endpoint URL and instance ID
   - Sets status to `DEPLOYING`
5. **Provider provisions** GPU instance and deploys Docker container
6. **Endpoint becomes active** (2-5 minutes typically)

### Example: OLMocr Deployment to RunPod

```python
# POST /api/deployments
{
    "deployment_name": "olmocr-production",
    "provider_id": 1,  # RunPod
    "model_id": 1,     # OLMocr 7B
    "gpu_type": "RTX 4090",
    "region": "eur-is-1",
    "min_workers": 0,
    "max_workers": 3,
    "cost_per_hour": 0.39,
    "api_key": "RUNPOD_API_KEY"
}
```

**Result:**
- Deployment record created in database
- RunPod GraphQL API called to create serverless endpoint
- Docker image `ocular/olmocr-runpod:latest` deployed
- Endpoint URL returned: `https://api.runpod.ai/v2/{endpoint_id}/run`
- Status updated to `DEPLOYING`

## Database Schema

### Providers Table
```sql
CREATE TABLE providers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) UNIQUE,
    display_name VARCHAR(255),
    status ENUM('ACTIVE', 'INACTIVE', 'MAINTENANCE'),
    available_gpus JSON,
    min_cost_per_hour FLOAT,
    max_cost_per_hour FLOAT,
    extra_metadata JSON
);
```

### Models Table
```sql
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) UNIQUE,
    version VARCHAR(100),
    min_vram_gb INTEGER,
    docker_image VARCHAR(512),
    supports_vllm BOOLEAN,
    extra_metadata JSON
);
```

### Deployments Table
```sql
CREATE TABLE deployments (
    id INTEGER PRIMARY KEY,
    deployment_id VARCHAR(36) UNIQUE,
    deployment_name VARCHAR(255),
    provider_id INTEGER REFERENCES providers(id),
    model_id INTEGER REFERENCES models(id),
    status ENUM('PENDING', 'DEPLOYING', 'RUNNING', 'STOPPED', 'FAILED', 'DELETED'),
    gpu_type VARCHAR(100),
    region VARCHAR(100),
    min_workers INTEGER,
    max_workers INTEGER,
    cost_per_hour FLOAT,
    endpoint_url VARCHAR(512),
    instance_id VARCHAR(255),
    started_at TIMESTAMP,
    stopped_at TIMESTAMP,
    config JSON,
    extra_metadata JSON
);
```

## API Endpoints

### Deployment Management

- **GET** `/api/deployments/providers` - List cloud providers
- **GET** `/api/deployments/models` - List available models
- **POST** `/api/deployments` - Create new deployment (executes deployment script)
- **GET** `/api/deployments` - List all deployments
- **GET** `/api/deployments/{id}` - Get deployment details
- **PATCH** `/api/deployments/{id}` - Update deployment configuration
- **DELETE** `/api/deployments/{id}` - Delete deployment
- **POST** `/api/deployments/{id}/start` - Start stopped deployment
- **POST** `/api/deployments/{id}/stop` - Stop running deployment
- **GET** `/api/deployments/metrics` - Get deployment metrics

### Page Endpoints

- **GET** `/deployments` - Deployment management UI

## Configuration

### Environment Variables

Required for RunPod deployments:

```bash
# RunPod API
RUNPOD_API_KEY=your_runpod_api_key

# RunPod S3 Storage
RUNPOD_ACCESS_KEY_ID=user_***
RUNPOD_SECRET_ACCESS_KEY=rps_***
RUNPOD_BUCKET_ENDPOINT_URL=https://s3api-eur-is-1.runpod.io/
RUNPOD_NETWORK_VOLUME_ID=your-network-volume-id
RUNPOD_REGION=eur-is-1

# Docker Registry
DOCKER_USERNAME=ocular

# Database
DATABASE_URL=sqlite:///./ocular_deployments.db
```

## Testing

### Unit Tests

Run OLMocr deployment tests:

```bash
# All deployment tests
pytest tests/test_olmocr_deployment.py -v

# Specific test
pytest tests/test_olmocr_deployment.py::TestOLMocrDeployment::test_deploy_olmocr_to_runpod_success -v
```

### Integration Tests

Test real deployment (requires RUNPOD_API_KEY):

```bash
# WARNING: Creates real deployment and may incur costs
RUNPOD_API_KEY=your_key pytest tests/test_olmocr_deployment.py::TestOLMocrDeploymentIntegration::test_real_olmocr_deployment -v
```

## Provider-Specific Details

### RunPod Serverless

**Deployment Method:** GraphQL API
**Configuration:**
- Endpoint name, GPU type, worker scaling
- Docker image with model
- Environment variables (S3 credentials, model name)
- Timeout and resource limits

**API Endpoint:** `https://api.runpod.io/graphql`
**Console:** `https://console.runpod.io/serverless`

**Example Configuration:**
```python
{
    "name": "olmocr-production",
    "gpu_ids": "RTX 4090",
    "docker_image": "ocular/olmocr-runpod:latest",
    "min_workers": 0,
    "max_workers": 3,
    "execution_timeout": 600,
    "container_disk_in_gb": 30,
    "env": {
        "MODEL_NAME": "allenai/olmOCR-7B-0825",
        "RUNPOD_REGION": "eur-is-1"
    }
}
```

### Vast.ai (Coming Soon)

**Status:** Not yet implemented
**Planned Features:**
- Instance rental via API
- SSH deployment
- Cost optimization

### Lambda Labs (Coming Soon)

**Status:** Not yet implemented
**Planned Features:**
- On-demand GPU instances
- Reserved capacity
- API-based deployment

## Cost Management

### Cost Tracking

The system tracks deployment costs:

```python
# Cost per hour from deployment configuration
cost_per_hour = 0.39  # USD

# Estimated monthly cost (730 hours)
monthly_cost = cost_per_hour * 730  # $284.70

# Cost classes
if cost_per_hour < 0.5:
    cost_class = "low"
elif cost_per_hour < 2.0:
    cost_class = "medium"
else:
    cost_class = "high"
```

### Metrics Endpoint

```bash
GET /api/deployments/metrics
```

Returns:
```json
{
    "active_deployments": 3,
    "total_gpu_count": 5,
    "total_monthly_cost": 854.10,
    "deployments_by_provider": {
        "runpod": 2,
        "vastai": 1
    },
    "deployments_by_status": {
        "running": 3,
        "stopped": 1
    }
}
```

## Future Enhancements

- [ ] Implement Vast.ai deployment
- [ ] Implement Lambda Labs deployment
- [ ] Add deployment health checks
- [ ] Add cost alerts and budgets
- [ ] Add deployment logs viewer
- [ ] Add GPU metrics monitoring
- [ ] Add auto-stop for idle deployments
- [ ] Add deployment templates/presets
- [ ] Add multi-region deployment
- [ ] Add A/B testing for models

## Troubleshooting

### Common Issues

**1. Deployment fails with "Invalid API key"**
- Verify `RUNPOD_API_KEY` is set correctly
- Check API key has not expired
- Verify API key has permissions to create endpoints

**2. Deployment status stuck in DEPLOYING**
- Check RunPod console for deployment status
- Verify Docker image exists and is accessible
- Check RunPod account has sufficient credits

**3. Endpoint URL not set**
- Check deployment service logs for errors
- Verify RunPod API returned endpoint ID
- Check database record was updated

**4. Tests failing with database errors**
- Tests use in-memory SQLite database
- Check fixtures are creating test data properly
- Ensure database schema is up to date

## References

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless)
- [RunPod GraphQL API](https://graphql-spec.runpod.io/)
- [OLMocr Model](https://huggingface.co/allenai/olmOCR-7B-0825)
- [Deployment Templates](../deployment/runpod.io/)
