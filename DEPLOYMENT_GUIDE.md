# Ocular OCR Platform - Deployment Guide

**Version**: 3.0.0
**Last Updated**: 2025-10-05
**Status**: Production Ready ✅

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Database Configuration](#database-configuration)
4. [Service Configuration](#service-configuration)
5. [Deployment Options](#deployment-options)
6. [Post-Deployment](#post-deployment)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum:**
- Python 3.11+
- 2GB RAM
- 10GB disk space
- PostgreSQL 14+ (production) or SQLite (development)

**Recommended:**
- Python 3.11+
- 4GB+ RAM
- 20GB+ disk space
- PostgreSQL 15+
- Redis (optional, for caching)

### Required Accounts

1. **Clerk.com** - Authentication (https://clerk.com)
   - Free tier: Up to 10,000 MAU
   - Sign up and create application
   - Get Secret Key and Publishable Key

2. **Stripe** - Payment Processing (https://stripe.com)
   - Create account
   - Get API keys (test and live)
   - Setup webhooks

3. **Cloud Provider** (Choose one or more)
   - **Google Cloud Platform** - For Cloud Functions/Run
   - **AWS** - For EC2/ECS deployment
   - **DigitalOcean** - For Droplet deployment

4. **OCR Provider Keys** (At least one required)
   - Mistral AI - https://console.mistral.ai/
   - Google Cloud Vision - https://console.cloud.google.com/
   - AWS Textract - https://aws.amazon.com/textract/
   - Azure Document Intelligence - https://portal.azure.com/

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/ocular.git
cd ocular
```

### 2. Install Dependencies

**Using uv (recommended):**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install in editable mode
uv add -e .
```

**Using pip:**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or your preferred editor
```

**Required Environment Variables:**

```bash
# Application Settings
APP_TITLE=Ocular OCR Platform
APP_DESCRIPTION=Enterprise OCR and AI Model Deployment
APP_VERSION=3.0.0
ENVIRONMENT=production  # or development/staging
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ocular
# For SQLite (development only):
# DATABASE_URL=sqlite:///./ocular.db

# Authentication (Clerk.com)
CLERK_SECRET_KEY=sk_live_...
CLERK_PUBLISHABLE_KEY=pk_live_...
JWT_SECRET_KEY=<generate_random_256bit_key>
AUTH_SESSION_TIMEOUT_MINUTES=60

# Payments (Stripe)
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# OCR Providers (configure at least one)
MISTRAL_API_KEY=your_mistral_key
MISTRAL_MODEL=pixtral-12b-2409

# Optional: Google Cloud Vision
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_PROJECT_ID=your_project_id

# Optional: AWS Textract
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Optional: Azure Document Intelligence
AZURE_DOC_INTEL_ENDPOINT=https://your_resource.cognitiveservices.azure.com/
AZURE_DOC_INTEL_API_KEY=your_api_key

# Optional: Tesseract (local processing)
TESSERACT_LANGUAGE=eng
TESSERACT_DPI=300

# Cloud GPU Providers (configure at least one)
RUNPOD_API_KEY=your_runpod_key
VASTAI_API_KEY=your_vastai_key
LAMBDA_LABS_API_KEY=your_lambda_key
REPLICATE_API_TOKEN=your_replicate_token

# Storage
RUNPOD_BUCKET_ENDPOINT_URL=https://storage.runpod.io
RUNPOD_BUCKET_ACCESS_KEY_ID=your_access_key
RUNPOD_BUCKET_SECRET_ACCESS_KEY=your_secret_key

# Logging (optional)
LOG_LEVEL=INFO
LOGFIRE_TOKEN=your_logfire_token  # Optional

# Security
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
MAX_FILE_SIZE_MB=10
RATE_LIMIT_RPM=100
API_SECRET_KEY=<generate_random_256bit_key>
ENCRYPTION_KEY=<generate_random_256bit_key>
ENABLE_HTTPS_ONLY=true

# Cache (optional)
REDIS_URL=redis://localhost:6379
ENABLE_CACHING=true
CACHE_TTL_SECONDS=300
```

### 4. Generate Secret Keys

```bash
# Generate secure random keys
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Use output for JWT_SECRET_KEY, API_SECRET_KEY, ENCRYPTION_KEY
```

---

## Database Configuration

### PostgreSQL (Production)

**1. Install PostgreSQL:**

```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start service
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS
```

**2. Create Database:**

```bash
# Login to PostgreSQL
sudo -u postgres psql

# Create database and user
CREATE DATABASE ocular;
CREATE USER ocular_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE ocular TO ocular_user;

# Exit
\q
```

**3. Update DATABASE_URL:**

```bash
DATABASE_URL=postgresql://ocular_user:your_secure_password@localhost:5432/ocular
```

### SQLite (Development Only)

```bash
# No setup required, just use:
DATABASE_URL=sqlite:///./ocular.db
```

### Run Migrations

```bash
# Initialize database
python -c "from ocular.database.session import init_db; init_db()"

# Run migrations
python ocular/database/migrate_add_versions.py
python ocular/database/migrate_add_auth.py
python ocular/database/migrate_add_billing.py

# Seed initial data
python ocular/database/seed.py
```

---

## Service Configuration

### Clerk.com Setup

1. **Create Application:**
   - Go to https://dashboard.clerk.com/
   - Create new application
   - Choose authentication methods (Email, Google, GitHub, etc.)

2. **Get API Keys:**
   - Navigate to API Keys section
   - Copy Secret Key → `CLERK_SECRET_KEY`
   - Copy Publishable Key → `CLERK_PUBLISHABLE_KEY`

3. **Configure Webhooks (Optional):**
   - Add webhook endpoint: `https://yourdomain.com/api/auth/webhook`
   - Select events: `user.created`, `user.updated`, `session.created`

### Stripe Setup

1. **Create Account:**
   - Go to https://dashboard.stripe.com/
   - Complete account setup

2. **Get API Keys:**
   - Navigate to Developers → API Keys
   - Copy Secret Key → `STRIPE_SECRET_KEY`
   - Copy Publishable Key → `STRIPE_PUBLISHABLE_KEY`

3. **Setup Webhooks:**
   - Navigate to Developers → Webhooks
   - Add endpoint: `https://yourdomain.com/api/billing/webhook`
   - Select events:
     - `payment_intent.succeeded`
     - `payment_intent.payment_failed`
     - `invoice.paid`
     - `invoice.payment_failed`
   - Copy Webhook Secret → `STRIPE_WEBHOOK_SECRET`

### OCR Provider Setup

**Mistral AI:**
```bash
# Get API key from https://console.mistral.ai/
MISTRAL_API_KEY=your_key
MISTRAL_MODEL=pixtral-12b-2409
```

**Google Cloud Vision:**
```bash
# Create service account and download JSON key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GOOGLE_PROJECT_ID=your-gcp-project-id
```

**AWS Textract:**
```bash
# Create IAM user with Textract permissions
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

---

## Deployment Options

### Option 1: Local Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Run application
uvicorn ocular.app:app --host 0.0.0.0 --port 8000 --reload

# Access at http://localhost:8000
```

### Option 2: Google Cloud Functions

**Prerequisites:**
- Google Cloud SDK installed
- GCP project created
- Billing enabled

**Deploy:**

```bash
# Login to Google Cloud
gcloud auth login
gcloud config set project your-project-id

# Deploy to staging
gcloud functions deploy ocular-ocr-staging \
  --source . \
  --entry-point ocular_ocr \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --region us-central1 \
  --memory 2GB \
  --timeout 540s \
  --set-env-vars-file .env.staging

# Deploy to production
gcloud functions deploy ocular-ocr-production \
  --source . \
  --entry-point ocular_ocr \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --region us-central1 \
  --memory 2GB \
  --timeout 540s \
  --set-env-vars-file .env.production
```

**Automated Deployment (GitHub Actions):**

The repository includes `.github/workflows/deploy-gcp.yml` for CI/CD:
- Automatically deploys to staging on push to `main`
- Manually deploy to production via GitHub Actions

### Option 3: Google Cloud Run

```bash
# Build container
docker build -t gcr.io/your-project/ocular:latest .

# Push to GCR
docker push gcr.io/your-project/ocular:latest

# Deploy to Cloud Run
gcloud run deploy ocular \
  --image gcr.io/your-project/ocular:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars-file .env.production
```

### Option 4: Docker (Self-Hosted)

**1. Create Dockerfile (if not exists):**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install application
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "ocular.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**2. Build and Run:**

```bash
# Build image
docker build -t ocular:3.0.0 .

# Run container
docker run -d \
  --name ocular \
  -p 8000:8000 \
  --env-file .env \
  -v /path/to/data:/app/data \
  ocular:3.0.0
```

**3. Docker Compose:**

```yaml
version: '3.8'

services:
  ocular:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ocular
      POSTGRES_USER: ocular_user
      POSTGRES_PASSWORD: your_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

```bash
# Start all services
docker-compose up -d
```

### Option 5: VPS/VM Deployment

**1. Setup Server (Ubuntu 22.04):**

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.11
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Install PostgreSQL
sudo apt-get install -y postgresql postgresql-contrib

# Install Nginx (reverse proxy)
sudo apt-get install -y nginx

# Install Tesseract (optional)
sudo apt-get install -y tesseract-ocr
```

**2. Deploy Application:**

```bash
# Clone repository
git clone https://github.com/your-org/ocular.git
cd ocular

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.example .env
nano .env  # Edit with production values

# Setup database
python ocular/database/migrate_add_versions.py
python ocular/database/migrate_add_auth.py
python ocular/database/migrate_add_billing.py
python ocular/database/seed.py
```

**3. Configure Systemd Service:**

Create `/etc/systemd/system/ocular.service`:

```ini
[Unit]
Description=Ocular OCR Service
After=network.target postgresql.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/var/www/ocular
Environment="PATH=/var/www/ocular/.venv/bin"
ExecStart=/var/www/ocular/.venv/bin/uvicorn ocular.app:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable ocular
sudo systemctl start ocular
sudo systemctl status ocular
```

**4. Configure Nginx:**

Create `/etc/nginx/sites-available/ocular`:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    client_max_body_size 10M;
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/ocular /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**5. Setup SSL with Let's Encrypt:**

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

---

## Post-Deployment

### 1. Health Check

```bash
# Check application health
curl https://yourdomain.com/health

# Expected response:
{
  "status": "healthy",
  "version": "3.0.0",
  "database": "connected"
}
```

### 2. Create Admin User

```bash
# Create first admin user via Clerk dashboard or API
curl -X POST https://yourdomain.com/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@yourdomain.com",
    "password": "secure_password",
    "role": "admin"
  }'
```

### 3. Configure Webhook Endpoints

**Clerk Webhooks:**
- Endpoint: `https://yourdomain.com/api/auth/webhook`
- Events: `user.created`, `user.updated`, `session.created`

**Stripe Webhooks:**
- Endpoint: `https://yourdomain.com/api/billing/webhook`
- Events: `payment_intent.succeeded`, `invoice.paid`, etc.

### 4. Test Core Functionality

```bash
# Test OCR processing
curl -X POST https://yourdomain.com/api/ocr/process \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@test.pdf"

# Test model deployment
curl -X POST https://yourdomain.com/api/deployments \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": 1,
    "provider": "runpod",
    "gpu_type": "A40"
  }'
```

---

## Monitoring & Maintenance

### Observability Dashboard

Access at: `https://yourdomain.com/observability`

**Key Metrics:**
- Real-time logs with filtering
- Request traces
- Performance profiles
- Error tracking
- Active alerts

### Database Backups

```bash
# PostgreSQL backup
pg_dump -U ocular_user ocular > backup_$(date +%Y%m%d).sql

# Restore
psql -U ocular_user ocular < backup_20251005.sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/var/backups/ocular"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -U ocular_user ocular | gzip > $BACKUP_DIR/ocular_$TIMESTAMP.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "ocular_*.sql.gz" -mtime +30 -delete
```

### Log Rotation

```bash
# Configure logrotate
sudo nano /etc/logrotate.d/ocular

# Add:
/var/log/ocular/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload ocular
    endscript
}
```

### Performance Monitoring

**Enable Logfire (Optional):**
```bash
# Add to .env
LOGFIRE_TOKEN=your_token

# Logfire automatically tracks:
# - API request/response times
# - Database query performance
# - Error rates and tracebacks
```

**Custom Metrics:**
```python
from ocular.services.observability_service import get_observability_service, MetricType

obs = get_observability_service()

# Record custom metrics
obs.record_metric(
    name="custom.metric.name",
    value=123.45,
    metric_type=MetricType.GAUGE,
    tags={"environment": "production"}
)
```

---

## Troubleshooting

### Common Issues

**1. Database Connection Errors**

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -U ocular_user -d ocular -h localhost

# Check DATABASE_URL format
# Correct: postgresql://user:pass@host:5432/dbname
```

**2. Clerk Authentication Failures**

```bash
# Verify keys are correct
echo $CLERK_SECRET_KEY  # Should start with sk_
echo $CLERK_PUBLISHABLE_KEY  # Should start with pk_

# Test Clerk API
curl https://api.clerk.com/v1/sessions/verify \
  -H "Authorization: Bearer $CLERK_SECRET_KEY"
```

**3. Stripe Webhook Failures**

```bash
# Test webhook locally with Stripe CLI
stripe listen --forward-to localhost:8000/api/billing/webhook

# Verify webhook secret
echo $STRIPE_WEBHOOK_SECRET  # Should start with whsec_
```

**4. OCR Provider Errors**

```bash
# Test Mistral API
curl https://api.mistral.ai/v1/models \
  -H "Authorization: Bearer $MISTRAL_API_KEY"

# Test Google Vision
gcloud auth application-default login
python -c "from google.cloud import vision; client = vision.ImageAnnotatorClient()"
```

**5. Deployment Failures**

```bash
# Check Cloud Functions logs
gcloud functions logs read ocular-ocr-production --limit 50

# Check local logs
tail -f /var/log/ocular/app.log

# Enable debug mode temporarily
export DEBUG=true
uvicorn ocular.app:app --reload
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run application
python -m ocular.app

# Or with uvicorn
uvicorn ocular.app:app --log-level debug
```

### Support Resources

- **Documentation**: `/docs` (Swagger UI)
- **Health Check**: `/health`
- **Debug Info**: `/debug` (development only)
- **Observability**: `/observability`

---

## Security Checklist

- [ ] All environment variables set correctly
- [ ] Database credentials secured
- [ ] API keys stored as secrets (not in code)
- [ ] HTTPS/TLS enabled
- [ ] CORS origins configured
- [ ] Rate limiting enabled
- [ ] Webhook signatures verified
- [ ] Debug mode disabled in production
- [ ] Database backups automated
- [ ] Log rotation configured
- [ ] Monitoring and alerting set up

---

## Rollback Procedure

**If deployment fails:**

```bash
# Rollback to previous version
gcloud functions deploy ocular-ocr-production \
  --source gs://your-bucket/previous-version.zip \
  --entry-point ocular_ocr \
  --runtime python311

# Or restore database backup
psql -U ocular_user ocular < backup_20251004.sql

# Or use Git
git checkout v2.0.0
# Re-deploy
```

---

## Production Checklist

Before going live:

- [ ] All environment variables configured
- [ ] Database migrations completed
- [ ] SSL certificates installed
- [ ] Webhooks configured and tested
- [ ] Backups automated
- [ ] Monitoring dashboards set up
- [ ] Error alerting configured
- [ ] Load testing completed
- [ ] Security audit performed
- [ ] Documentation updated
- [ ] Team trained on platform

---

**Deployment Guide v3.0.0**
**Last Updated**: 2025-10-05
**Status**: Complete ✅
