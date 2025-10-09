# Ocular OCR - Development Plan
## Improvement Opportunities Implementation Roadmap

**Version**: 1.0
**Date**: October 2025
**Status**: Planning Phase
**Estimated Duration**: 4-6 weeks

---

## Executive Summary

This development plan addresses 14 identified improvement opportunities across security, architecture, code quality, and optimization. The plan is organized into 4 phases (weeks), prioritized by impact and risk, with clear deliverables, acceptance criteria, and testing requirements.

**Key Metrics**:
- **Security Issues**: 3 critical items
- **Architecture Issues**: 3 high-priority items
- **Code Quality**: 3 medium-priority items
- **Optimization**: 5 low-priority items
- **Total Effort**: ~160-200 developer hours

---

## Phase 1: Security & Critical Fixes (Week 1)
**Focus**: Eliminate security vulnerabilities and prevent production incidents
**Priority**: 🔴 CRITICAL
**Effort**: 40 hours

### 1.1 Rotate Exposed API Keys
**Issue**: Real credentials exposed in .env file
**Risk**: HIGH - Active credentials compromised
**Effort**: 2 hours

**Tasks**:
- [ ] Rotate all exposed tokens immediately
  - [ ] Generate new RUNPOD_API_KEY
  - [ ] Generate new RUNPOD_AWS_ACCESS_KEY_ID
  - [ ] Generate new RUNPOD_AWS_SECRET_ACCESS_KEY
  - [ ] Generate new HF_TOKEN
- [ ] Update .env with new credentials
- [ ] Update all deployed instances
- [ ] Verify old credentials are revoked
- [ ] Update .env.example to use placeholder values only

**Acceptance Criteria**:
- ✅ All old credentials revoked at provider
- ✅ New credentials tested and working
- ✅ .env.example contains only placeholders
- ✅ Documentation updated with credential rotation process

**Testing**:
```bash
# Verify old credentials fail
curl -H "Authorization: Bearer OLD_TOKEN" https://api.runpod.ai/v2/status
# Expected: 401 Unauthorized

# Verify new credentials work
curl -H "Authorization: Bearer NEW_TOKEN" https://api.runpod.ai/v2/status
# Expected: 200 OK
```

---

### 1.2 Add API Authentication
**Issue**: Public endpoints allow unauthorized deployment creation
**Risk**: HIGH - Anyone can create costly deployments
**Effort**: 16 hours

**Tasks**:
- [ ] Implement API key authentication system
  - [ ] Create `api_keys` database table
  - [ ] Generate secure API keys with `secrets.token_urlsafe(32)`
  - [ ] Add `APIKeyAuth` dependency
- [ ] Protect deployment endpoints
  - [ ] Add auth to POST `/api/deployments`
  - [ ] Add auth to PUT `/api/deployments/{id}`
  - [ ] Add auth to DELETE `/api/deployments/{id}`
- [ ] Create API key management UI
  - [ ] Generate new API keys
  - [ ] Revoke API keys
  - [ ] List active keys (show last 4 chars only)
- [ ] Add rate limiting
  - [ ] Implement `slowapi` rate limiter
  - [ ] Set limits: 100 req/hour per API key

**Implementation**:
```python
# ocular/middleware/auth.py
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib

security = HTTPBearer()

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
):
    api_key_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()

    # Check database for valid key
    db_key = db.query(APIKey).filter(
        APIKey.key_hash == api_key_hash,
        APIKey.is_active == True,
        APIKey.expires_at > datetime.utcnow()
    ).first()

    if not db_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Update last used timestamp
    db_key.last_used_at = datetime.utcnow()
    db.commit()

    return db_key

# Usage in endpoints
@router.post("/deployments")
async def create_deployment(
    deployment_data: DeploymentCreate,
    api_key: APIKey = Depends(verify_api_key),  # ← Add this
    db: Session = Depends(get_db)
):
    # ... deployment logic
```

**Database Migration**:
```sql
CREATE TABLE api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
```

**Acceptance Criteria**:
- ✅ All protected endpoints require valid API key
- ✅ Invalid/missing keys return 401 Unauthorized
- ✅ Rate limiting prevents abuse (100 req/hour)
- ✅ API key management UI working
- ✅ Documentation includes API key usage guide

**Testing**:
```bash
# Test without API key
curl -X POST http://localhost:8000/api/deployments
# Expected: 401 Unauthorized

# Test with valid API key
curl -X POST http://localhost:8000/api/deployments \
  -H "Authorization: Bearer sk_test_abc123..."
# Expected: 200 OK (or validation error)

# Test rate limiting
for i in {1..105}; do curl -H "Authorization: Bearer sk_test_abc123..." ...; done
# Expected: First 100 succeed, then 429 Too Many Requests
```

---

### 1.3 Migrate to Production Database
**Issue**: SQLite not suitable for concurrent deployments
**Risk**: MEDIUM - Data loss, corruption, poor performance
**Effort**: 22 hours

**Tasks**:
- [ ] Set up PostgreSQL database
  - [ ] Create Google Cloud SQL instance (or AWS RDS)
  - [ ] Configure connection pooling
  - [ ] Enable SSL/TLS connections
- [ ] Update database configuration
  - [ ] Add asyncpg driver: `uv add asyncpg sqlalchemy[asyncio]`
  - [ ] Update DATABASE_URL format
  - [ ] Configure connection pool size
- [ ] Implement async database access
  - [ ] Replace sync `Session` with `AsyncSession`
  - [ ] Update all queries to use `await`
  - [ ] Update dependency injection
- [ ] Data migration
  - [ ] Export existing SQLite data
  - [ ] Import into PostgreSQL
  - [ ] Verify data integrity
- [ ] Update deployment scripts

**Implementation**:
```python
# ocular/database/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# PostgreSQL connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://user:pass@localhost/ocular_prod"
)

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# Update endpoints to async
@router.get("/models")
async def list_ai_models(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Model).order_by(Model.name))
    models = result.scalars().all()
    return {"models": [m.to_dict() for m in models]}
```

**Environment Variables**:
```bash
# Production
DATABASE_URL=postgresql+asyncpg://user:pass@35.1.2.3:5432/ocular_prod?sslmode=require

# Development
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ocular_dev
```

**Acceptance Criteria**:
- ✅ PostgreSQL instance created and accessible
- ✅ All queries migrated to async
- ✅ Connection pooling configured (20 connections)
- ✅ All existing data migrated successfully
- ✅ Performance improved (>50% faster queries)
- ✅ Concurrent deployments work without errors

**Testing**:
```python
# Test concurrent writes
import asyncio
import aiohttp

async def create_deployment(session, i):
    async with session.post(
        "http://localhost:8000/api/deployments",
        json={"provider_id": 1, "model_id": 1, "gpu_type": "A100"}
    ) as response:
        return await response.json()

async def test_concurrent():
    async with aiohttp.ClientSession() as session:
        tasks = [create_deployment(session, i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 50
        assert all(r["success"] for r in results)

# Expected: All 50 deployments created successfully
```

---

## Phase 2: Architecture & Core Features (Week 2)
**Focus**: Implement critical missing features and improve architecture
**Priority**: 🟠 HIGH
**Effort**: 50 hours

### 2.1 Implement Actual Deployment Logic
**Issue**: Deployments create records but don't actually deploy
**Risk**: HIGH - Production-ready feature marked as TODO
**Effort**: 24 hours

**Tasks**:
- [ ] Implement RunPod deployment
  - [ ] Create `RunPodDeploymentProvider` class
  - [ ] Implement `create_serverless_endpoint()`
  - [ ] Implement `stop_endpoint()`
  - [ ] Implement `delete_endpoint()`
  - [ ] Handle RunPod API responses
- [ ] Implement Vast.ai deployment
  - [ ] Create `VastAIDeploymentProvider` class
  - [ ] Implement instance rental via API
  - [ ] Implement instance termination
- [ ] Implement deployment state machine
  - [ ] PENDING → DEPLOYING (trigger deployment)
  - [ ] DEPLOYING → RUNNING (poll status)
  - [ ] RUNNING → STOPPED (graceful shutdown)
  - [ ] STOPPED → DELETED (cleanup resources)
- [ ] Add deployment monitoring
  - [ ] Poll provider APIs for status
  - [ ] Update database with current state
  - [ ] Handle deployment failures
- [ ] Implement webhook handlers
  - [ ] RunPod webhook for status updates
  - [ ] Vast.ai webhook for instance events

**Implementation**:
```python
# ocular/services/deployment_providers/runpod.py
from runpod import RunPodClient

class RunPodDeploymentProvider:
    def __init__(self, api_key: str):
        self.client = RunPodClient(api_key)

    async def create_deployment(
        self,
        model: Model,
        gpu_type: str,
        gpu_count: int,
        config: dict
    ) -> dict:
        """Create RunPod serverless endpoint"""
        endpoint = await self.client.create_endpoint(
            name=f"{model.name}-{uuid.uuid4().hex[:8]}",
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            docker_image=model.docker_image,
            environment_variables=config.get("env", {}),
            max_workers=config.get("max_workers", 3),
            idle_timeout=config.get("idle_timeout", 5)
        )

        return {
            "endpoint_id": endpoint.id,
            "endpoint_url": endpoint.url,
            "status": "DEPLOYING"
        }

    async def get_status(self, endpoint_id: str) -> str:
        """Poll endpoint status"""
        endpoint = await self.client.get_endpoint(endpoint_id)
        return self._map_status(endpoint.status)

    async def stop_deployment(self, endpoint_id: str):
        """Stop serverless endpoint"""
        await self.client.terminate_endpoint(endpoint_id)

    def _map_status(self, runpod_status: str) -> DeploymentStatus:
        mapping = {
            "INITIALIZING": DeploymentStatus.DEPLOYING,
            "RUNNING": DeploymentStatus.RUNNING,
            "THROTTLED": DeploymentStatus.RUNNING,
            "TERMINATED": DeploymentStatus.STOPPED
        }
        return mapping.get(runpod_status, DeploymentStatus.ERROR)

# Update API endpoint
@router.post("/deployments")
async def create_deployment(
    deployment_data: DeploymentCreate,
    api_key: APIKey = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    # Create database record
    deployment = Deployment(
        provider_id=deployment_data.provider_id,
        model_id=deployment_data.model_id,
        gpu_type=deployment_data.gpu_type,
        status=DeploymentStatus.PENDING
    )
    db.add(deployment)
    await db.commit()

    # Trigger actual deployment
    provider = get_deployment_provider(deployment.provider)
    result = await provider.create_deployment(
        model=deployment.model,
        gpu_type=deployment.gpu_type,
        config=deployment_data.config
    )

    # Update with provider details
    deployment.instance_id = result["endpoint_id"]
    deployment.endpoint_url = result["endpoint_url"]
    deployment.status = DeploymentStatus.DEPLOYING
    deployment.started_at = datetime.utcnow()
    await db.commit()

    # Start background status polling
    background_tasks.add_task(poll_deployment_status, deployment.id)

    return {"success": True, "deployment": deployment.to_dict()}
```

**Background Task**:
```python
async def poll_deployment_status(deployment_id: int):
    """Poll provider API until deployment is running or failed"""
    while True:
        async with get_db() as db:
            deployment = await db.get(Deployment, deployment_id)
            if deployment.status in [DeploymentStatus.RUNNING, DeploymentStatus.ERROR]:
                break

            provider = get_deployment_provider(deployment.provider)
            status = await provider.get_status(deployment.instance_id)

            deployment.status = status
            await db.commit()

        await asyncio.sleep(10)  # Poll every 10 seconds
```

**Acceptance Criteria**:
- ✅ POST `/api/deployments` actually creates RunPod endpoint
- ✅ Endpoint URL returned and accessible
- ✅ Status polling updates deployment state
- ✅ Stop/delete actually terminates provider resources
- ✅ Failed deployments properly handled
- ✅ Cost tracking accurate

**Testing**:
```python
# Integration test
async def test_full_deployment_lifecycle():
    # Create deployment
    response = await client.post("/api/deployments", json={
        "provider_id": 1,  # RunPod
        "model_id": 1,      # OLMocr
        "gpu_type": "RTX A4000",
        "config": {"max_workers": 2}
    })
    deployment_id = response.json()["deployment"]["id"]

    # Wait for RUNNING status (max 5 minutes)
    for _ in range(30):
        deployment = await client.get(f"/api/deployments/{deployment_id}")
        if deployment.json()["status"] == "running":
            break
        await asyncio.sleep(10)

    assert deployment.json()["status"] == "running"
    assert deployment.json()["endpoint_url"]

    # Test endpoint
    ocr_response = await client.post(
        deployment.json()["endpoint_url"],
        files={"file": open("test.pdf", "rb")}
    )
    assert ocr_response.status_code == 200

    # Stop deployment
    await client.delete(f"/api/deployments/{deployment_id}")

    # Verify stopped on provider
    time.sleep(30)
    deployment = await client.get(f"/api/deployments/{deployment_id}")
    assert deployment.json()["status"] == "stopped"
```

---

### 2.2 Add Database Migrations with Alembic
**Issue**: No version control for schema changes
**Risk**: MEDIUM - Production deployments break on updates
**Effort**: 12 hours

**Tasks**:
- [ ] Install and configure Alembic
  - [ ] `uv add alembic`
  - [ ] `alembic init alembic`
  - [ ] Configure `alembic.ini`
- [ ] Create initial migration
  - [ ] Generate from current models
  - [ ] Review and test migration
- [ ] Add migration scripts
  - [ ] Create `migrate.sh` script
  - [ ] Add to deployment pipeline
- [ ] Document migration process
  - [ ] Developer guide for creating migrations
  - [ ] Production migration checklist

**Implementation**:
```bash
# Initialize Alembic
alembic init alembic

# Edit alembic/env.py
from ocular.database.models import Base
target_metadata = Base.metadata

# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Review generated migration
cat alembic/versions/001_initial_schema.py

# Apply migration
alembic upgrade head

# Future schema changes
# 1. Modify models in ocular/database/models.py
# 2. Generate migration
alembic revision --autogenerate -m "Add deployment_logs table"

# 3. Review migration
cat alembic/versions/002_add_deployment_logs.py

# 4. Apply migration
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

**migrate.sh**:
```bash
#!/bin/bash
set -e

echo "🔄 Running database migrations..."

# Check current version
current_version=$(alembic current)
echo "Current version: $current_version"

# Show pending migrations
alembic show head

# Apply migrations
alembic upgrade head

# Verify success
new_version=$(alembic current)
echo "✅ Migrated to version: $new_version"

# Run sanity checks
python -c "from ocular.database.session import engine; from sqlalchemy import text; engine.connect().execute(text('SELECT 1'))"
echo "✅ Database connection verified"
```

**Acceptance Criteria**:
- ✅ Alembic configured and working
- ✅ Initial migration captures current schema
- ✅ New migrations can be auto-generated
- ✅ Migrations apply cleanly on dev/staging/prod
- ✅ Rollback process tested
- ✅ Documentation complete

**Testing**:
```bash
# Test migration workflow
# 1. Create fresh database
dropdb ocular_test && createdb ocular_test

# 2. Apply migrations from scratch
DATABASE_URL=postgresql://localhost/ocular_test alembic upgrade head

# 3. Verify schema matches models
python tests/test_schema_integrity.py
# Expected: All tables present, all columns match

# 4. Test downgrade
alembic downgrade base
alembic upgrade head
# Expected: No errors, data preserved (if safe migrations)
```

---

### 2.3 Consolidate Settings Systems
**Issue**: 3 different settings mechanisms causing confusion
**Risk**: MEDIUM - Maintenance burden, bugs from inconsistency
**Effort**: 14 hours

**Tasks**:
- [ ] Audit all settings usage
  - [ ] Find all `get_settings()` calls
  - [ ] Find all `OcularSettings` usage
  - [ ] Document current dependencies
- [ ] Create unified settings class
  - [ ] Merge functionality into single class
  - [ ] Support both env vars and ocular.json
  - [ ] Use Pydantic for validation
- [ ] Update all imports
  - [ ] Replace old settings imports
  - [ ] Update type hints
- [ ] Remove deprecated settings
  - [ ] Delete `ocular.providers.settings`
  - [ ] Clean up unused code
- [ ] Update documentation

**Implementation**:
```python
# ocular/config.py - Single source of truth
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import json
from pathlib import Path

class OcularConfig(BaseSettings):
    """Unified configuration for Ocular OCR application"""

    # Application settings (from env or ocular.json)
    app_title: str = Field(default="Ocular OCR Service", env="APP_TITLE")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # Database (from env only - secret)
    database_url: str = Field(..., env="DATABASE_URL")

    # API Keys (from env only - secrets)
    mistral_api_key: str = Field(default="", env="MISTRAL_API_KEY")
    runpod_api_key: str = Field(default="", env="RUNPOD_API_KEY")
    hf_token: str = Field(default="", env="HF_TOKEN")

    # Provider/Model catalogs (from ocular.json)
    _cloud_providers: list = []
    _ai_models: list = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_json_config()

    def _load_json_config(self):
        """Load non-secret config from ocular.json"""
        config_file = Path(__file__).parent.parent / "ocular.json"
        if config_file.exists():
            with open(config_file) as f:
                data = json.load(f)
                self._cloud_providers = data.get("cloud_providers", [])
                self._ai_models = data.get("ai_models", [])

    @property
    def cloud_providers(self) -> list:
        return self._cloud_providers

    @property
    def ai_models(self) -> list:
        return self._ai_models

    @validator("database_url")
    def validate_database_url(cls, v):
        if v.startswith("sqlite") and "production" in os.getenv("ENVIRONMENT", ""):
            raise ValueError("SQLite not allowed in production")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Singleton instance
_config_instance = None

def get_config() -> OcularConfig:
    """Get application configuration"""
    global _config_instance
    if _config_instance is None:
        _config_instance = OcularConfig()
    return _config_instance

# Usage everywhere
from ocular.config import get_config

config = get_config()
print(config.app_title)
print(config.mistral_api_key)
print(config.cloud_providers)
```

**Migration Guide**:
```python
# Old (multiple places)
from ocular.settings import get_settings
from ocular.providers.settings import OcularSettings

settings = get_settings()
provider_settings = OcularSettings()

# New (single import)
from ocular.config import get_config

config = get_config()
```

**Acceptance Criteria**:
- ✅ All code uses `get_config()`
- ✅ Old settings modules deleted
- ✅ All tests passing
- ✅ Documentation updated
- ✅ No duplicate configuration loading

---

## Phase 3: Code Quality & Maintainability (Week 3)
**Focus**: Clean up code, improve maintainability, reduce technical debt
**Priority**: 🟡 MEDIUM
**Effort**: 40 hours

### 3.1 Clean Up Console Logging
**Issue**: 68 console.log() statements in production
**Risk**: LOW - Performance impact, potential data leaks
**Effort**: 12 hours

**Tasks**:
- [ ] Create logger utility
  - [ ] Implement log levels (debug, info, warn, error)
  - [ ] Environment-aware (only debug in dev)
  - [ ] Add timestamps
- [ ] Replace all console.log()
  - [ ] deployments.js (37 instances)
  - [ ] providers.js (9 instances)
  - [ ] models.js (7 instances)
  - [ ] Other files (15 instances)
- [ ] Add structured logging
  - [ ] Log deployment events
  - [ ] Log API errors
  - [ ] Log user actions
- [ ] Configure log rotation

**Implementation**:
```javascript
// ocular/static/js/logger.js
class Logger {
    constructor() {
        this.isDevelopment = window.location.hostname === 'localhost' ||
                           window.ENV === 'development';
    }

    debug(...args) {
        if (this.isDevelopment) {
            console.log(`[${new Date().toISOString()}] [DEBUG]`, ...args);
        }
    }

    info(...args) {
        console.info(`[${new Date().toISOString()}] [INFO]`, ...args);
    }

    warn(...args) {
        console.warn(`[${new Date().toISOString()}] [WARN]`, ...args);
    }

    error(...args) {
        console.error(`[${new Date().toISOString()}] [ERROR]`, ...args);
        // Optionally send to error tracking service
        if (window.Sentry) {
            Sentry.captureException(new Error(args[0]));
        }
    }
}

const logger = new Logger();

// Usage
// Before
console.log('Loaded providers:', providers);

// After
logger.debug('Loaded providers:', providers);  // Only in dev
logger.error('Failed to load:', error);  // Always logged + sent to Sentry
```

**Acceptance Criteria**:
- ✅ All console.log() replaced with logger.debug()
- ✅ Debug logs only appear in development
- ✅ Error logs include timestamps
- ✅ Critical errors sent to error tracking
- ✅ No performance impact in production

---

### 3.2 Standardize Async/Sync Patterns
**Issue**: Mixed async/sync causing confusion and inefficiency
**Risk**: MEDIUM - Potential deadlocks, blocking I/O
**Effort**: 16 hours

**Tasks**:
- [ ] Audit all async usage
  - [ ] List all async functions
  - [ ] Identify blocking calls in async contexts
- [ ] Choose strategy: Full async or full sync
  - [ ] Decision: Go full async (FastAPI best practice)
- [ ] Update database access
  - [ ] Migrate to AsyncSession (done in Phase 1.3)
- [ ] Update all endpoints
  - [ ] Keep `async def` for all routes
  - [ ] Add `await` to all async calls
- [ ] Update tests
  - [ ] Use `pytest-asyncio`
  - [ ] Update test fixtures

**Implementation**:
```python
# Before (mixed)
@router.get("/models")
async def list_models(db: Session = Depends(get_db)):  # async
    models = db.query(Model).all()  # sync blocking call!
    return {"models": models}

# After (fully async)
@router.get("/models")
async def list_models(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Model).order_by(Model.name))
    models = result.scalars().all()
    return {"models": [m.to_dict() for m in models]}

# Test updates
# Before
def test_list_models():
    response = client.get("/api/models")
    assert response.status_code == 200

# After
@pytest.mark.asyncio
async def test_list_models():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/models")
        assert response.status_code == 200
```

**Acceptance Criteria**:
- ✅ All endpoints consistently async
- ✅ No blocking calls in async contexts
- ✅ All database queries use await
- ✅ Tests updated and passing
- ✅ Performance improved (measured with benchmarks)

---

### 3.3 Add Comprehensive Input Validation
**Issue**: Minimal validation on deployment creation
**Risk**: MEDIUM - Invalid deployments, cost overruns
**Effort**: 12 hours

**Tasks**:
- [ ] Create validation models
  - [ ] Extend Pydantic models with validators
  - [ ] Add custom validation rules
- [ ] Validate deployment inputs
  - [ ] GPU exists for provider
  - [ ] Model compatible with GPU
  - [ ] Budget limits enforced
  - [ ] Provider API key configured
- [ ] Add budget controls
  - [ ] Max cost per hour
  - [ ] Total budget limits
  - [ ] Alert thresholds
- [ ] Validate before API calls

**Implementation**:
```python
from pydantic import BaseModel, Field, validator, root_validator

class DeploymentCreate(BaseModel):
    provider_id: int = Field(gt=0)
    model_id: int = Field(gt=0)
    gpu_type: str = Field(min_length=1, max_length=100)
    gpu_count: int = Field(ge=1, le=8, description="Max 8 GPUs")
    max_cost_per_hour: float = Field(
        ge=0,
        le=100,
        description="Budget limit per hour"
    )
    max_total_cost: float = Field(
        default=1000,
        ge=0,
        le=10000,
        description="Total budget limit"
    )
    auto_stop_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Auto-stop after N hours"
    )

    @validator('gpu_type')
    def validate_gpu_exists(cls, v, values):
        """Check GPU type is available for selected provider"""
        if 'provider_id' not in values:
            return v

        provider = get_provider(values['provider_id'])
        if not provider:
            raise ValueError(f"Provider {values['provider_id']} not found")

        if v not in provider.available_gpus:
            raise ValueError(
                f"GPU '{v}' not available for {provider.display_name}. "
                f"Available: {', '.join(provider.available_gpus[:5])}"
            )
        return v

    @root_validator
    def validate_model_gpu_compatibility(cls, values):
        """Check model requirements match GPU"""
        model = get_model(values.get('model_id'))
        gpu_type = values.get('gpu_type')
        gpu_count = values.get('gpu_count')

        if not model:
            raise ValueError(f"Model {values.get('model_id')} not found")

        # Extract VRAM from GPU type (e.g., "A100 (80GB)" -> 80)
        vram_match = re.search(r'\((\d+)GB\)', gpu_type)
        if vram_match:
            gpu_vram = int(vram_match.group(1)) * gpu_count
            if gpu_vram < model.min_vram_gb:
                raise ValueError(
                    f"Model {model.display_name} requires {model.min_vram_gb}GB VRAM, "
                    f"but {gpu_type} x{gpu_count} only provides {gpu_vram}GB"
                )

        return values

    @validator('provider_id')
    def validate_provider_configured(cls, v):
        """Check provider API key is configured"""
        provider = get_provider(v)
        if not provider:
            raise ValueError(f"Provider {v} not found")

        if provider.name == "runpod" and not get_config().runpod_api_key:
            raise ValueError("RunPod API key not configured")

        if provider.status != ProviderStatus.ACTIVE:
            raise ValueError(f"Provider {provider.display_name} is not active")

        return v
```

**Acceptance Criteria**:
- ✅ All validation rules enforced
- ✅ Invalid requests rejected with clear errors
- ✅ Budget limits prevent cost overruns
- ✅ GPU/model compatibility checked
- ✅ Helpful error messages for users

**Testing**:
```python
# Test GPU validation
def test_invalid_gpu_rejected():
    response = client.post("/api/deployments", json={
        "provider_id": 1,  # RunPod
        "model_id": 1,
        "gpu_type": "NonExistent GPU",  # Invalid
        "gpu_count": 1
    })
    assert response.status_code == 422
    assert "not available" in response.json()["detail"]

# Test VRAM validation
def test_insufficient_vram_rejected():
    response = client.post("/api/deployments", json={
        "provider_id": 1,
        "model_id": 1,  # Requires 16GB
        "gpu_type": "T4 (8GB)",  # Insufficient
        "gpu_count": 1
    })
    assert response.status_code == 422
    assert "requires 16GB VRAM" in response.json()["detail"]

# Test budget enforcement
def test_budget_limit_enforced():
    response = client.post("/api/deployments", json={
        "provider_id": 1,
        "model_id": 1,
        "gpu_type": "H100 (80GB)",  # $4/hour
        "gpu_count": 2,
        "max_cost_per_hour": 5.0  # Budget: $5/hour, actual: $8/hour
    })
    assert response.status_code == 422
    assert "exceeds budget" in response.json()["detail"]
```

---

## Phase 4: Optimization & Polish (Week 4)
**Focus**: Performance improvements, monitoring, user experience
**Priority**: 🟢 LOW
**Effort**: 30 hours

### 4.1 Add Redis Caching
**Issue**: Redundant database queries on every page load
**Risk**: LOW - Slow page loads, unnecessary DB load
**Effort**: 10 hours

**Tasks**:
- [ ] Set up Redis
  - [ ] Install Redis: `docker run -d -p 6379:6379 redis`
  - [ ] Add client: `uv add redis aioredis`
- [ ] Implement caching layer
  - [ ] Create cache decorator
  - [ ] Cache provider list (5 min TTL)
  - [ ] Cache model list (5 min TTL)
- [ ] Add cache invalidation
  - [ ] Clear on provider update
  - [ ] Clear on model update
- [ ] Monitor cache hit rate

**Implementation**:
```python
# ocular/cache.py
from redis.asyncio import Redis
import json
from functools import wraps

redis_client = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

def cache(ttl: int = 300, key_prefix: str = ""):
    """Cache decorator with TTL in seconds"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Check cache
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            await redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )

            return result
        return wrapper
    return decorator

# Usage
@router.get("/providers")
@cache(ttl=300, key_prefix="api")  # 5 minute cache
async def list_providers(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Provider).order_by(Provider.name))
    providers = result.scalars().all()
    return {"providers": [p.to_dict() for p in providers]}

# Cache invalidation
@router.put("/providers/{provider_id}")
async def update_provider(provider_id: int, ...):
    # Update provider
    ...

    # Clear cache
    await redis_client.delete("api:list_providers:*")

    return {"success": True}
```

**Acceptance Criteria**:
- ✅ Redis running and connected
- ✅ Provider/model lists cached
- ✅ Cache cleared on updates
- ✅ Page load time improved >50%
- ✅ Cache hit rate >80%

**Testing**:
```python
# Test caching
async def test_caching():
    # First request (cache miss)
    start = time.time()
    response1 = await client.get("/api/providers")
    time1 = time.time() - start

    # Second request (cache hit)
    start = time.time()
    response2 = await client.get("/api/providers")
    time2 = time.time() - start

    assert response1.json() == response2.json()
    assert time2 < time1 / 2  # Second request >2x faster

    # Verify cache hit
    stats = await redis_client.info("stats")
    assert stats["keyspace_hits"] > 0
```

---

### 4.2 Implement Database Backups
**Issue**: No backup strategy for SQLite database
**Risk**: MEDIUM - Data loss risk
**Effort**: 8 hours

**Tasks**:
- [ ] Automated backups (PostgreSQL)
  - [ ] Enable point-in-time recovery on Cloud SQL
  - [ ] Configure backup schedule (daily)
  - [ ] Set retention period (30 days)
- [ ] Backup monitoring
  - [ ] Alert on backup failures
  - [ ] Monitor backup size
- [ ] Test restore process
  - [ ] Document restore procedure
  - [ ] Practice restore (quarterly)
- [ ] Backup encryption

**Implementation**:
```bash
# Google Cloud SQL automated backups
gcloud sql instances patch ocular-prod \
  --backup-start-time=02:00 \
  --enable-bin-log \
  --retained-backups-count=30 \
  --transaction-log-retention-days=7

# AWS RDS automated backups
aws rds modify-db-instance \
  --db-instance-identifier ocular-prod \
  --backup-retention-period 30 \
  --preferred-backup-window "02:00-03:00"

# Manual backup script (for testing)
#!/bin/bash
# backup.sh
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/ocular_backup_$TIMESTAMP.sql"

# PostgreSQL backup
pg_dump $DATABASE_URL > $BACKUP_FILE

# Compress
gzip $BACKUP_FILE

# Upload to cloud storage
gsutil cp $BACKUP_FILE.gz gs://ocular-backups/

# Clean up old backups (keep 30 days)
find $BACKUP_DIR -name "ocular_backup_*.sql.gz" -mtime +30 -delete

echo "✅ Backup completed: $BACKUP_FILE.gz"
```

**Restore Procedure**:
```bash
# Restore from backup
# 1. Download backup
gsutil cp gs://ocular-backups/ocular_backup_20251005_020000.sql.gz .

# 2. Decompress
gunzip ocular_backup_20251005_020000.sql.gz

# 3. Restore to database
psql $DATABASE_URL < ocular_backup_20251005_020000.sql

# 4. Verify
psql $DATABASE_URL -c "SELECT COUNT(*) FROM deployments;"
```

**Acceptance Criteria**:
- ✅ Daily automated backups configured
- ✅ Backups encrypted at rest
- ✅ 30-day retention policy
- ✅ Restore tested successfully
- ✅ Monitoring alerts configured

---

### 4.3 Add Monitoring & Observability
**Issue**: No application monitoring, hard to diagnose issues
**Risk**: LOW - Slow incident response
**Effort**: 12 hours

**Tasks**:
- [ ] Add health check endpoint
  - [ ] Database connectivity
  - [ ] Redis connectivity
  - [ ] Provider API reachability
- [ ] Integrate error tracking
  - [ ] Set up Sentry
  - [ ] Configure error alerts
- [ ] Add performance monitoring
  - [ ] Track endpoint response times
  - [ ] Monitor database query performance
- [ ] Create metrics dashboard
  - [ ] Deployment success/failure rate
  - [ ] API request count
  - [ ] Active deployments
  - [ ] Total cost spent

**Implementation**:
```python
# Health check endpoint
@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        health["checks"]["database"] = "ok"
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["database"] = f"error: {str(e)}"

    # Check Redis
    try:
        await redis_client.ping()
        health["checks"]["redis"] = "ok"
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["redis"] = f"error: {str(e)}"

    # Check provider APIs
    try:
        # Quick API call to verify connectivity
        runpod_client = RunPodClient(get_config().runpod_api_key)
        await runpod_client.get_endpoints(limit=1)
        health["checks"]["runpod_api"] = "ok"
    except Exception as e:
        health["checks"]["runpod_api"] = f"warning: {str(e)}"

    # Get metrics
    result = await db.execute(
        select(func.count(Deployment.id))
        .filter(Deployment.status == DeploymentStatus.RUNNING)
    )
    health["metrics"] = {
        "active_deployments": result.scalar(),
        "uptime_seconds": (datetime.utcnow() - app_start_time).total_seconds()
    }

    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)

# Sentry integration
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastAPIIntegration

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "development"),
    traces_sample_rate=0.1,
    integrations=[FastAPIIntegration()]
)

# Metrics middleware
from prometheus_client import Counter, Histogram

request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    endpoint = request.url.path

    request_count.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()

    request_duration.labels(
        method=request.method,
        endpoint=endpoint
    ).observe(duration)

    return response

# Metrics endpoint
from prometheus_client import generate_latest

@router.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

**Grafana Dashboard**:
```yaml
# dashboard.json (import to Grafana)
{
  "title": "Ocular OCR Metrics",
  "panels": [
    {
      "title": "Request Rate",
      "targets": [
        {"expr": "rate(http_requests_total[5m])"}
      ]
    },
    {
      "title": "Active Deployments",
      "targets": [
        {"expr": "deployments_active_total"}
      ]
    },
    {
      "title": "Error Rate",
      "targets": [
        {"expr": "rate(http_requests_total{status=~\"5..\"}[5m])"}
      ]
    }
  ]
}
```

**Acceptance Criteria**:
- ✅ /health endpoint returns detailed status
- ✅ Sentry captures all errors
- ✅ Prometheus metrics exported
- ✅ Grafana dashboard created
- ✅ Alerts configured for critical issues

---

## Testing Strategy

### Unit Tests
```python
# tests/test_validation.py
def test_gpu_validation():
    with pytest.raises(ValueError, match="not available"):
        DeploymentCreate(
            provider_id=1,
            model_id=1,
            gpu_type="InvalidGPU",
            gpu_count=1
        )

def test_vram_validation():
    with pytest.raises(ValueError, match="requires.*GB VRAM"):
        DeploymentCreate(
            provider_id=1,
            model_id=1,  # Requires 16GB
            gpu_type="T4 (8GB)",
            gpu_count=1
        )
```

### Integration Tests
```python
# tests/test_deployment_flow.py
@pytest.mark.asyncio
async def test_full_deployment_lifecycle():
    # Create
    response = await client.post("/api/deployments", json={...})
    deployment_id = response.json()["deployment"]["id"]

    # Wait for RUNNING
    await wait_for_status(deployment_id, "running", timeout=300)

    # Test endpoint
    deployment = await client.get(f"/api/deployments/{deployment_id}")
    ocr_response = await client.post(
        deployment.json()["endpoint_url"],
        files={"file": test_pdf}
    )
    assert "text" in ocr_response.json()

    # Stop
    await client.delete(f"/api/deployments/{deployment_id}")
    await wait_for_status(deployment_id, "stopped", timeout=60)
```

### Load Tests
```python
# tests/load_test.py
from locust import HttpUser, task, between

class OcularUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def list_providers(self):
        self.client.get("/api/providers")

    @task(2)
    def list_models(self):
        self.client.get("/api/models")

    @task(1)
    def create_deployment(self):
        self.client.post("/api/deployments", json={...})

# Run: locust -f tests/load_test.py --host=http://localhost:8000
# Target: 100 concurrent users, <200ms p95 latency
```

---

## Deployment Plan

### Phase Rollout
1. **Phase 1** → Deploy to staging, test thoroughly
2. **Phase 2** → Deploy to staging, run integration tests
3. **Phase 3** → Deploy incrementally, monitor errors
4. **Phase 4** → Deploy during low-traffic window

### Rollback Plan
Each phase has a rollback procedure:
```bash
# Rollback database migration
alembic downgrade -1

# Rollback code deployment
git revert <commit-hash>
git push origin main

# Restore from backup (if needed)
gsutil cp gs://ocular-backups/latest.sql.gz .
gunzip latest.sql.gz
psql $DATABASE_URL < latest.sql
```

---

## Success Metrics

### Security (Phase 1)
- ✅ 0 exposed credentials
- ✅ 100% protected endpoints require auth
- ✅ 0 SQL injection vulnerabilities
- ✅ Database backups: 100% success rate

### Performance (Phase 2-4)
- ✅ Page load time: <1s (down from ~3s)
- ✅ API response time: p95 <200ms
- ✅ Cache hit rate: >80%
- ✅ Database query time: <50ms average

### Reliability (Phase 2-4)
- ✅ Uptime: 99.9%
- ✅ Deployment success rate: >95%
- ✅ Error rate: <0.1%
- ✅ Mean time to recovery: <15 minutes

### Code Quality (Phase 3)
- ✅ Test coverage: >80%
- ✅ No console.log in production
- ✅ All async patterns consistent
- ✅ 0 critical code smells (SonarQube)

---

## Risk Management

### High Risk Items
1. **Database Migration** (Phase 1.3)
   - **Risk**: Data loss during migration
   - **Mitigation**: Full backup before migration, test on staging first
   - **Rollback**: Restore from backup, revert to SQLite temporarily

2. **Deployment Logic** (Phase 2.1)
   - **Risk**: Failed deployments cost money
   - **Mitigation**: Extensive testing, budget limits, auto-stop timers
   - **Rollback**: Feature flag to disable actual deployments

3. **Settings Consolidation** (Phase 2.3)
   - **Risk**: Breaking all imports across codebase
   - **Mitigation**: Update all imports atomically, comprehensive testing
   - **Rollback**: Revert commit, deploy previous version

### Medium Risk Items
- API Authentication: Test thoroughly, provide migration guide for API users
- Async Migration: Run performance benchmarks before/after
- Input Validation: Ensure existing deployments still work

---

## Resource Requirements

### Infrastructure
- **PostgreSQL**: Cloud SQL instance (db-n1-standard-2, ~$100/month)
- **Redis**: Cloud Memorystore (1GB, ~$30/month)
- **Monitoring**: Sentry (Team plan, ~$26/month), Grafana Cloud (free tier)
- **Total**: ~$156/month additional infrastructure cost

### Team
- **Backend Developer**: Full-time (Weeks 1-4)
- **Frontend Developer**: Part-time (Week 3)
- **DevOps Engineer**: Part-time (Weeks 1, 2, 4)
- **QA Engineer**: Part-time (Weeks 2-4)

---

## Timeline Summary

| Phase | Focus | Duration | Effort | Risk |
|-------|-------|----------|--------|------|
| Phase 1 | Security & Critical Fixes | Week 1 | 40h | 🔴 High |
| Phase 2 | Architecture & Core Features | Week 2 | 50h | 🟠 High |
| Phase 3 | Code Quality | Week 3 | 40h | 🟡 Medium |
| Phase 4 | Optimization & Polish | Week 4 | 30h | 🟢 Low |
| **Total** | | **4 weeks** | **160h** | |

---

## Appendix

### A. Dependencies to Add
```bash
# Phase 1
uv add asyncpg sqlalchemy[asyncio] alembic slowapi

# Phase 4
uv add redis aioredis prometheus-client sentry-sdk
```

### B. Environment Variables
```bash
# New variables needed
DATABASE_URL=postgresql+asyncpg://...
REDIS_HOST=localhost
REDIS_PORT=6379
SENTRY_DSN=https://...@sentry.io/...
ENVIRONMENT=production
```

### C. Documentation Updates
- [ ] API authentication guide
- [ ] Database migration procedures
- [ ] Deployment guide
- [ ] Monitoring & alerts setup
- [ ] Incident response playbook

---

**Next Steps**:
1. Review and approve this plan
2. Set up development/staging environments
3. Begin Phase 1 implementation
4. Schedule weekly progress reviews

**Questions/Feedback**: [Contact: dev-team@ocular.ai]
