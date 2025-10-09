# Ocular OCR Platform

**Enterprise-Grade OCR Model Deployment & Management Platform**

[![Version](https://img.shields.io/badge/version-3.0-blue.svg)](https://github.com/your-repo/ocular)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)

Ocular is a comprehensive platform for deploying, managing, and monitoring OCR models across multiple cloud GPU providers. Built with FastAPI, it provides enterprise-grade features including authentication, billing, cost tracking, and advanced deployment configuration.

---

## 🌟 Key Features

### Core Platform
- 🚀 **Multi-Provider Support**: Deploy across RunPod, Vast.ai, Lambda Labs, and Replicate
- 📊 **Advanced Analytics**: Real-time metrics, performance tracking, and cost analysis
- 🔄 **Version Control**: A/B testing, rollback, and version comparison
- ⚙️ **Advanced Configuration**: Autoscaling, health checks, resource limits
- 🧪 **Model Testing**: Automated testing, benchmarking, and validation
- 📈 **Provider Comparison**: Cost optimization and smart recommendations

### Enterprise Features
- 🔐 **Authentication**: Clerk.com enterprise authentication with OAuth
- 👥 **Authorization**: Role-based access control (Admin, User, Viewer)
- 🔑 **API Keys**: Secure key management with SHA-256 hashing
- 📝 **Audit Logging**: Complete compliance-ready audit trail
- 💳 **Billing System**: Stripe integration with automated invoicing
- 💰 **Cost Tracking**: Real-time cost monitoring across all resources
- 🔔 **Budget Alerts**: Threshold-based notifications (Email/Slack)
- 📊 **Usage Quotas**: API calls, deployments, and storage limits

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Deployment](#deployment)
- [Development](#development)
- [Contributing](#contributing)

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- PostgreSQL (or SQLite for development)
- Node.js 16+ (for frontend development)
- Docker (optional, for containerized deployment)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-repo/ocular.git
cd ocular

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Or using uv (recommended)
uv add -e .
```

### Install Additional Providers

```bash
# For specific OCR providers
uv add pdf2image          # Mistral AI PDF processing
uv add google-cloud-vision  # Google Vision API
uv add boto3             # AWS Textract
uv add azure-ai-documentintelligence  # Azure Document Intelligence
uv add pytesseract pillow # Tesseract OCR
```

---

## ⚡ Quick Start

### 1. Environment Setup

Create a `.env` file in the project root:

```bash
# Database
DATABASE_URL=sqlite:///./ocular.db  # or PostgreSQL URL

# Authentication (Clerk.com)
CLERK_SECRET_KEY=sk_test_your_key
CLERK_PUBLISHABLE_KEY=pk_test_your_key

# Payments (Stripe)
STRIPE_SECRET_KEY=sk_test_your_key
STRIPE_PUBLISHABLE_KEY=pk_test_your_key

# Cloud Providers
RUNPOD_API_KEY=your_runpod_key
VAST_AI_API_KEY=your_vastai_key
LAMBDA_LABS_API_KEY=your_lambda_key
REPLICATE_API_TOKEN=your_replicate_key

# Optional
LOGFIRE_TOKEN=your_logfire_token
```

### 2. Initialize Database

```bash
# Run database migrations
python ocular/database/migrate_add_auth.py
python ocular/database/migrate_add_billing.py

# Seed initial data (optional)
python -c "from ocular.database.seed import seed_database; seed_database()"
```

### 3. Start the Application

```bash
# Development server
cd ocular
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# The application will be available at:
# http://localhost:8000
```

### 4. Access the Platform

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🔧 Configuration

### Provider Configuration

Configure cloud GPU providers in the web UI under **Providers** or via API:

```python
import httpx

# Create provider
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/providers",
        json={
            "name": "runpod",
            "display_name": "RunPod Serverless",
            "api_endpoint": "https://api.runpod.io/v2",
            "api_key": "your_api_key",
            "available_gpus": ["RTX A4000", "RTX 4090"],
            "min_cost_per_hour": 0.30,
            "max_cost_per_hour": 2.00
        },
        headers={"Authorization": f"Bearer {api_key}"}
    )
```

### Model Configuration

Add OCR models via the web UI or API:

```python
# Create model
response = await client.post(
    "http://localhost:8000/api/models",
    json={
        "name": "allenai/OLMoE-1B-7B-0924-Instruct",
        "display_name": "OLMoE OCR",
        "version": "1.0.0",
        "description": "Advanced OCR model",
        "model_type": "ocr",
        "framework": "vllm",
        "min_gpu_memory_gb": 8,
        "recommended_gpu": "RTX A4000"
    },
    headers={"Authorization": f"Bearer {api_key}"}
)
```

---

## 📖 Usage Examples

### Deploy a Model

```python
# Create deployment
deployment = await client.post(
    "http://localhost:8000/api/deployments",
    json={
        "deployment_name": "production-ocr",
        "model_id": 1,
        "provider_id": 1,
        "gpu_type": "RTX A4000",
        "gpu_count": 1,
        "deployment_type": "vllm",
        "max_workers": 3,
        "min_workers": 0,
        "autoscaling_enabled": True
    },
    headers={"Authorization": f"Bearer {api_key}"}
)
```

### Create API Key

```python
# Generate API key
api_key_response = await client.post(
    "http://localhost:8000/api/auth/api-keys",
    json={
        "key_name": "Production API Key",
        "scopes": ["deployment:read", "deployment:create"],
        "expires_in_days": 90
    },
    headers={"Authorization": f"Bearer {session_token}"}
)

# Save the returned API key securely!
full_key = api_key_response.json()["full_key"]
```

### Track Costs

```python
# Get current month costs
costs = await client.get(
    "http://localhost:8000/api/billing/costs/current-month",
    headers={"Authorization": f"Bearer {api_key}"}
)

print(f"Total: ${costs['total_cost']}")
print(f"Breakdown: {costs['breakdown']}")
```

### Set Budget Alert

```python
# Create budget alert
alert = await client.post(
    "http://localhost:8000/api/billing/budget-alerts",
    json={
        "alert_name": "Monthly Budget",
        "threshold_amount": 100.00,
        "period_type": "monthly",
        "notify_email": True
    },
    headers={"Authorization": f"Bearer {api_key}"}
)
```

---

## 📚 API Documentation

### Available Endpoints

The platform provides 105+ REST API endpoints across multiple categories:

#### Core API (48 endpoints)
- `GET /health` - Health check
- `GET /api/models` - List models
- `POST /api/models` - Create model
- `GET /api/deployments` - List deployments
- `POST /api/deployments` - Create deployment
- `POST /api/deployments/{id}/start` - Start deployment
- `POST /api/deployments/{id}/stop` - Stop deployment

#### Analytics (8 endpoints)
- `GET /api/analytics/overview` - System overview
- `GET /api/analytics/deployment-metrics` - Deployment metrics
- `GET /api/analytics/cost-analysis` - Cost analysis

#### Version Control (13 endpoints)
- `POST /api/versions` - Create version
- `POST /api/versions/compare` - Compare versions
- `POST /api/versions/ab-test/setup` - Setup A/B test
- `POST /api/models/{id}/rollback/{version_id}` - Rollback

#### Authentication (18 endpoints)
- `GET /api/auth/me` - Get current user
- `POST /api/auth/api-keys` - Create API key
- `GET /api/auth/audit-logs/me` - Get audit logs

#### Billing (15 endpoints)
- `GET /api/billing/invoices` - List invoices
- `POST /api/billing/invoices/generate-monthly` - Generate invoice
- `GET /api/billing/costs/summary` - Cost summary
- `POST /api/billing/budget-alerts` - Create budget alert

### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Frontend Layer                     │
│  (Bootstrap 5 + Vanilla JS + Chart.js)            │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│                  API Layer (FastAPI)                │
│  • REST Endpoints (105+)                           │
│  • Authentication Middleware (Clerk + API Keys)    │
│  • Rate Limiting & Validation                      │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│               Service Layer (18 Services)           │
│  • Auth Service        • Billing Service           │
│  • Deployment Service  • Cost Tracking Service     │
│  • Version Service     • Budget Alert Service      │
│  • Analytics Service   • Provider Service          │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│            Database Layer (SQLAlchemy ORM)          │
│  • 14 Tables, 320+ Columns                         │
│  • PostgreSQL / SQLite                             │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              External Services                      │
│  • Clerk.com (Auth)    • RunPod (GPU)             │
│  • Stripe (Billing)    • Vast.ai (GPU)            │
│  • Logfire (Logs)      • Lambda Labs (GPU)        │
│  • Sentry (Errors)     • Replicate (GPU)          │
└─────────────────────────────────────────────────────┘
```

### Database Schema

**14 Tables**:
- `providers` - Cloud GPU providers
- `models` - OCR models
- `deployments` - Active deployments
- `model_versions` - Version control
- `version_deployments` - A/B testing
- `test_results` - Test execution
- `benchmark_results` - Performance benchmarks
- `users` - User accounts
- `api_keys` - API key management
- `audit_logs` - Audit trail
- `invoices` - Billing
- `cost_tracking` - Cost monitoring
- `budget_alerts` - Budget management

---

## 🚢 Deployment

### Docker Deployment

```bash
# Build image
docker build -t ocular-platform .

# Run container
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e CLERK_SECRET_KEY=... \
  -e STRIPE_SECRET_KEY=... \
  --name ocular \
  ocular-platform
```

### Docker Compose

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/ocular
      - CLERK_SECRET_KEY=${CLERK_SECRET_KEY}
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=ocular
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Google Cloud Functions

```bash
# Deploy to Cloud Functions
gcloud functions deploy ocular-platform \
  --source . \
  --entry-point ocular_ocr \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --region us-central1 \
  --memory 2GB \
  --timeout 540s \
  --set-env-vars CLERK_SECRET_KEY=$CLERK_SECRET_KEY
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocular-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ocular
  template:
    metadata:
      labels:
        app: ocular
    spec:
      containers:
      - name: ocular
        image: ocular-platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ocular-secrets
              key: database-url
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
uv add --dev pytest pytest-cov black isort mypy

# Run tests
pytest

# Run with coverage
pytest --cov=ocular --cov-report=html

# Format code
black ocular/
isort ocular/

# Type checking
mypy ocular/
```

### Project Structure

```
ocular/
├── core/               # Core functionality (enums, interfaces, models)
├── providers/          # OCR provider implementations
├── services/           # Business logic services (18 services)
├── routers/            # API route handlers (12 routers)
├── database/           # Database models and migrations
├── static/             # Static assets (JS, CSS)
├── templates/          # HTML templates
└── tests/              # Test suite
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_billing_service.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=ocular --cov-report=term-missing
```

---

## 📊 Performance & Scalability

### Performance Metrics
- API Latency: <200ms (p95)
- Database Queries: Optimized with indexes
- Frontend Load: <2s initial load
- Deployment Time: <60 seconds

### Scalability
- Horizontal scaling ready
- Database connection pooling
- Async/await throughout
- Stateless architecture

---

## 🔒 Security

### Authentication
- Enterprise-grade via Clerk.com
- OAuth provider support
- Multi-factor authentication
- Session management

### Authorization
- Role-based access control (RBAC)
- 26 fine-grained permissions
- Resource ownership validation
- API key scopes

### Data Security
- SHA-256 API key hashing
- Secure secret storage
- Input validation (Pydantic)
- SQL injection prevention (ORM)
- XSS protection

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

- **Documentation**: See `/docs` directory
- **Issues**: [GitHub Issues](https://github.com/your-repo/ocular/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/ocular/discussions)

---

## 🎯 Roadmap

### Completed ✅
- Multi-provider support (RunPod, Vast.ai, Lambda Labs, Replicate)
- Enterprise authentication & authorization
- Complete billing system
- Cost tracking & budget alerts
- Version control & A/B testing
- Advanced deployment configuration

### Planned 🚀
- Mobile applications (iOS/Android)
- Model marketplace
- Kubernetes auto-scaling
- Advanced analytics & ML insights
- Team collaboration features
- Additional provider integrations

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - ORM
- [Clerk.com](https://clerk.com/) - Authentication
- [Stripe](https://stripe.com/) - Payment processing
- [Bootstrap](https://getbootstrap.com/) - UI framework
- [Chart.js](https://www.chartjs.org/) - Data visualization

---

**Built with ❤️ by the Ocular Team**

**Version**: 3.0 - Enterprise Production Ready
**Last Updated**: 2025-10-05
