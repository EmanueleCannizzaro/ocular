"""
SQLAlchemy models for deployment tracking with normalized schema.

Database Structure:
- providers: Cloud GPU providers (RunPod, Vast.ai, Lambda, etc.)
- models: AI models available for deployment (OLMocr, RoLMocr, etc.)
- deployments: Active deployment instances linking providers and models
"""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Enum, Boolean, Text, JSON, ForeignKey
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ProviderStatus(str, PyEnum):
    """Provider status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"


class ModelStatus(str, PyEnum):
    """Model status enumeration."""
    AVAILABLE = "available"
    DEPRECATED = "deprecated"
    BETA = "beta"


class DeploymentStatus(str, PyEnum):
    """Deployment instance status enumeration."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DELETED = "deleted"


class DeploymentType(str, PyEnum):
    """Deployment type enumeration."""
    DOCKER = "docker"
    VLLM = "vllm"
    CUSTOM = "custom"


class VersionStatus(str, PyEnum):
    """Model version status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class UserRole(str, PyEnum):
    """User role enumeration for RBAC."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class APIKeyStatus(str, PyEnum):
    """API key status enumeration."""
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class AuditAction(str, PyEnum):
    """Audit log action types."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    DEPLOY = "deploy"
    STOP = "stop"
    LOGIN = "login"
    LOGOUT = "logout"
    API_CALL = "api_call"


class InvoiceStatus(str, PyEnum):
    """Invoice status enumeration."""
    DRAFT = "draft"
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentMethod(str, PyEnum):
    """Payment method enumeration."""
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    STRIPE = "stripe"


class BudgetAlertType(str, PyEnum):
    """Budget alert type enumeration."""
    THRESHOLD = "threshold"
    LIMIT = "limit"
    FORECAST = "forecast"


class Provider(Base):
    """
    Cloud GPU provider table.

    Stores information about cloud providers like RunPod, Vast.ai, Lambda Labs.
    """
    __tablename__ = "providers"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Provider Information
    name = Column(String(100), unique=True, nullable=False, index=True)  # e.g., 'runpod', 'vastai'
    display_name = Column(String(255), nullable=False)  # e.g., 'RunPod Serverless'
    description = Column(Text, nullable=True)
    website_url = Column(String(512), nullable=True)
    documentation_url = Column(String(512), nullable=True)

    # Status
    status = Column(Enum(ProviderStatus), nullable=False, default=ProviderStatus.ACTIVE)

    # Configuration
    api_endpoint = Column(String(512), nullable=True)  # Base API endpoint
    supports_autoscaling = Column(Boolean, nullable=False, default=False)
    supports_spot_instances = Column(Boolean, nullable=False, default=False)

    # GPU Types Available (JSON array)
    available_gpus = Column(JSON, nullable=True)  # ['RTX A4000', 'RTX 4000 Ada', etc.]

    # Pricing Information
    min_cost_per_hour = Column(Float, nullable=True)
    max_cost_per_hour = Column(Float, nullable=True)

    # Metadata
    extra_metadata = Column(JSON, nullable=True)  # Additional provider-specific config

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    deployments = relationship("Deployment", back_populates="provider")

    def __repr__(self):
        return f"<Provider(id={self.id}, name='{self.name}', status={self.status.value})>"

    def to_dict(self) -> dict:
        """Convert provider to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "website_url": self.website_url,
            "documentation_url": self.documentation_url,
            "status": self.status.value,
            "api_endpoint": self.api_endpoint,
            "supports_autoscaling": self.supports_autoscaling,
            "supports_spot_instances": self.supports_spot_instances,
            "available_gpus": self.available_gpus or [],
            "min_cost_per_hour": self.min_cost_per_hour,
            "max_cost_per_hour": self.max_cost_per_hour,
            "metadata": self.extra_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Model(Base):
    """
    AI model table.

    Stores information about OCR models available for deployment.
    """
    __tablename__ = "models"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Model Information
    name = Column(String(100), unique=True, nullable=False, index=True)  # e.g., 'olmocr', 'rolmocr'
    display_name = Column(String(255), nullable=False)  # e.g., 'OLMocr 7B'
    version = Column(String(100), nullable=False)  # e.g., 'allenai/olmOCR-7B-0825'
    description = Column(Text, nullable=True)

    # Status
    status = Column(Enum(ModelStatus), nullable=False, default=ModelStatus.AVAILABLE)

    # Model Specifications
    parameter_count = Column(String(50), nullable=True)  # e.g., '7B'
    model_type = Column(String(100), nullable=True)  # e.g., 'OCR', 'Vision-Language'
    base_model = Column(String(255), nullable=True)  # e.g., 'allenai/olmOCR-7B-0825'
    license = Column(String(100), nullable=True)
    pipeline_tag = Column(String(100), nullable=True)

    # HuggingFace metrics
    downloads = Column(Integer, nullable=True)
    likes = Column(Integer, nullable=True)
    last_modified = Column(DateTime, nullable=True)

    # Resource Requirements
    min_vram_gb = Column(Integer, nullable=False, default=16)  # Minimum VRAM required
    recommended_gpu = Column(String(100), nullable=True)  # e.g., 'RTX A4000+'

    # Docker Configuration
    docker_image = Column(String(512), nullable=True)  # Docker image for deployment
    dockerfile_path = Column(String(512), nullable=True)  # Path to Dockerfile in repo

    # Deployment Configuration
    default_deployment_type = Column(Enum(DeploymentType), nullable=False, default=DeploymentType.DOCKER)
    supports_vllm = Column(Boolean, nullable=False, default=False)

    # Metadata
    config_template = Column(JSON, nullable=True)  # Default configuration template
    extra_metadata = Column(JSON, nullable=True)  # Additional model-specific info

    # Repository Information
    repo_url = Column(String(512), nullable=True)
    documentation_url = Column(String(512), nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    deployments = relationship("Deployment", back_populates="model")
    versions = relationship("ModelVersion", back_populates="model", order_by="desc(ModelVersion.created_at)")

    def __repr__(self):
        return f"<Model(id={self.id}, name='{self.name}', version='{self.version}')>"

    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        metadata = dict(self.extra_metadata or {})

        closed_licenses = {
            "unknown",
            "proprietary",
            "llama2",
            "facebooklicense",
            "bigscience-bloom-rail-1.0",
            "cc-by-nc-4.0",
        }

        license_value = (self.license or metadata.get('huggingface_stats', {}).get('license') or '').lower()
        if 'is_opensource' not in metadata:
            metadata['is_opensource'] = bool(license_value) and license_value not in closed_licenses

        hf_stats = dict(metadata.get('huggingface_stats') or {})
        hf_stats.setdefault('downloads', self.downloads)
        hf_stats.setdefault('likes', self.likes)
        hf_stats.setdefault('last_modified', self.last_modified.isoformat() if self.last_modified else None)
        hf_stats.setdefault('license', self.license)
        metadata['huggingface_stats'] = hf_stats

        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "status": self.status.value,
            "parameter_count": self.parameter_count,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "license": self.license,
            "pipeline_tag": self.pipeline_tag,
            "min_vram_gb": self.min_vram_gb,
            "recommended_gpu": self.recommended_gpu,
            "docker_image": self.docker_image,
            "dockerfile_path": self.dockerfile_path,
            "default_deployment_type": self.default_deployment_type.value,
            "supports_vllm": self.supports_vllm,
            "config_template": self.config_template or {},
            "metadata": metadata,
            "downloads": self.downloads,
            "likes": self.likes,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "repo_url": self.repo_url,
            "documentation_url": self.documentation_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ModelVersion(Base):
    """
    Model version table.

    Tracks different versions of the same model for version management,
    rollback capabilities, and A/B testing.
    """
    __tablename__ = "model_versions"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Key
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)

    # Version Information
    version_number = Column(String(100), nullable=False)  # e.g., 'v1.0.0', '2024-01-05'
    version_tag = Column(String(100), nullable=True, index=True)  # e.g., 'latest', 'stable', 'beta'
    version_name = Column(String(255), nullable=True)  # Human-readable name
    changelog = Column(Text, nullable=True)  # Description of changes

    # Status
    status = Column(Enum(VersionStatus), nullable=False, default=VersionStatus.DRAFT)
    is_active = Column(Boolean, nullable=False, default=False)  # Current production version

    # Model Metadata (can differ per version)
    docker_image = Column(String(512), nullable=True)
    huggingface_repo = Column(String(512), nullable=True)  # HF model repo for this version
    model_hash = Column(String(64), nullable=True)  # SHA256 hash for verification
    model_size_gb = Column(Float, nullable=True)

    # Configuration
    config = Column(JSON, nullable=True)  # Version-specific configuration
    deployment_config = Column(JSON, nullable=True)  # Deployment settings for this version

    # Performance Metrics (from testing)
    benchmark_scores = Column(JSON, nullable=True)  # Performance benchmarks
    avg_inference_time_ms = Column(Float, nullable=True)
    avg_tokens_per_second = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)  # Overall quality metric

    # A/B Testing
    ab_test_enabled = Column(Boolean, nullable=False, default=False)
    ab_test_traffic_percentage = Column(Float, nullable=True)  # % of traffic for A/B test
    ab_test_start_date = Column(DateTime, nullable=True)
    ab_test_end_date = Column(DateTime, nullable=True)

    # Storage Information
    cloud_storage_path = Column(String(1024), nullable=True)  # S3/cloud storage path
    local_cache_path = Column(String(1024), nullable=True)  # Local cache path

    # Metadata
    created_by = Column(String(255), nullable=True)  # User who created this version
    extra_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deployed_at = Column(DateTime, nullable=True)  # When this version was first deployed
    deprecated_at = Column(DateTime, nullable=True)

    # Relationships
    model = relationship("Model", back_populates="versions")
    version_deployments = relationship("VersionDeployment", back_populates="version")

    def __repr__(self):
        return f"<ModelVersion(id={self.id}, model_id={self.model_id}, version='{self.version_number}', status={self.status.value})>"

    def to_dict(self) -> dict:
        """Convert model version to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_name": self.model.name if self.model else None,
            "version_number": self.version_number,
            "version_tag": self.version_tag,
            "version_name": self.version_name,
            "changelog": self.changelog,
            "status": self.status.value,
            "is_active": self.is_active,
            "docker_image": self.docker_image,
            "huggingface_repo": self.huggingface_repo,
            "model_hash": self.model_hash,
            "model_size_gb": self.model_size_gb,
            "config": self.config or {},
            "deployment_config": self.deployment_config or {},
            "benchmark_scores": self.benchmark_scores or {},
            "avg_inference_time_ms": self.avg_inference_time_ms,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "quality_score": self.quality_score,
            "ab_test_enabled": self.ab_test_enabled,
            "ab_test_traffic_percentage": self.ab_test_traffic_percentage,
            "ab_test_start_date": self.ab_test_start_date.isoformat() if self.ab_test_start_date else None,
            "ab_test_end_date": self.ab_test_end_date.isoformat() if self.ab_test_end_date else None,
            "cloud_storage_path": self.cloud_storage_path,
            "created_by": self.created_by,
            "metadata": self.extra_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
        }


class VersionDeployment(Base):
    """
    Version-specific deployment tracking table.

    Links deployments to specific model versions for rollback and A/B testing.
    """
    __tablename__ = "version_deployments"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Keys
    deployment_id = Column(Integer, ForeignKey("deployments.id"), nullable=False, index=True)
    version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False, index=True)

    # Deployment Info
    is_primary = Column(Boolean, nullable=False, default=True)  # Primary or A/B test deployment
    traffic_weight = Column(Float, nullable=False, default=100.0)  # Traffic percentage

    # Performance Tracking
    total_requests = Column(Integer, nullable=False, default=0)
    successful_requests = Column(Integer, nullable=False, default=0)
    failed_requests = Column(Integer, nullable=False, default=0)
    avg_latency_ms = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    # Relationships
    deployment = relationship("Deployment")
    version = relationship("ModelVersion", back_populates="version_deployments")

    def __repr__(self):
        return f"<VersionDeployment(id={self.id}, deployment_id={self.deployment_id}, version_id={self.version_id})>"

    def to_dict(self) -> dict:
        """Convert version deployment to dictionary."""
        return {
            "id": self.id,
            "deployment_id": self.deployment_id,
            "version_id": self.version_id,
            "is_primary": self.is_primary,
            "traffic_weight": self.traffic_weight,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round((self.successful_requests / self.total_requests * 100), 2) if self.total_requests > 0 else 0,
            "avg_latency_ms": self.avg_latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


class Deployment(Base):
    """
    Deployment instance table.

    Represents an active deployment of a model on a provider.
    Links Provider and Model tables.
    """
    __tablename__ = "deployments"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Keys
    provider_id = Column(Integer, ForeignKey("providers.id"), nullable=False, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)

    # Deployment Identification
    deployment_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    deployment_name = Column(String(255), nullable=False, index=True)

    # Deployment Configuration
    deployment_type = Column(Enum(DeploymentType), nullable=False, default=DeploymentType.DOCKER)

    # GPU Configuration
    gpu_type = Column(String(100), nullable=False)
    gpu_count = Column(Integer, nullable=False, default=1)

    # Status
    status = Column(Enum(DeploymentStatus), nullable=False, default=DeploymentStatus.PENDING, index=True)

    # Endpoint Information
    endpoint_url = Column(String(1024), nullable=True)
    api_key_hash = Column(String(255), nullable=True)  # Hashed for security

    # Cost Information
    cost_per_hour = Column(Float, nullable=False, default=0.0)
    total_cost = Column(Float, nullable=False, default=0.0)

    # Scaling Configuration
    max_workers = Column(Integer, nullable=False, default=1)
    min_workers = Column(Integer, nullable=False, default=0)
    timeout_seconds = Column(Integer, nullable=False, default=300)
    autoscaling_enabled = Column(Boolean, nullable=False, default=False)

    # Instance Configuration
    instance_id = Column(String(255), nullable=True)  # Provider's instance/pod ID
    region = Column(String(100), nullable=True)  # Deployment region

    # Configuration
    config = Column(JSON, nullable=True)  # Deployment-specific configuration
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)

    # Error Tracking
    error_message = Column(Text, nullable=True)
    last_error_at = Column(DateTime, nullable=True)

    # Usage Statistics
    total_requests = Column(Integer, nullable=False, default=0)
    last_request_at = Column(DateTime, nullable=True)
    avg_response_time_ms = Column(Float, nullable=True)

    # Relationships
    provider = relationship("Provider", back_populates="deployments")
    model = relationship("Model", back_populates="deployments")

    def __repr__(self):
        return f"<Deployment(id={self.id}, name='{self.deployment_name}', status={self.status.value})>"

    def to_dict(self) -> dict:
        """Convert deployment to dictionary representation."""
        return {
            "id": self.deployment_id,
            "deployment_name": self.deployment_name,
            "provider_id": self.provider_id,
            "provider_name": self.provider.name if self.provider else None,
            "model_id": self.model_id,
            "model_name": self.model.name if self.model else None,
            "model_display_name": self.model.display_name if self.model else None,
            "model_version": self.model.version if self.model else None,
            "deployment_type": self.deployment_type.value,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "status": self.status.value,
            "endpoint": self.endpoint_url,
            "cost_per_hour": round(self.cost_per_hour, 4),
            "total_cost": round(self.total_cost, 2),
            "max_workers": self.max_workers,
            "min_workers": self.min_workers,
            "timeout": self.timeout_seconds,
            "autoscaling": self.autoscaling_enabled,
            "instance_id": self.instance_id,
            "region": self.region,
            "config": self.config or {},
            "metadata": self.extra_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "error_message": self.error_message,
            "total_requests": self.total_requests,
            "last_request_at": self.last_request_at.isoformat() if self.last_request_at else None,
            "avg_response_time_ms": self.avg_response_time_ms,
        }

    @property
    def cost_class(self) -> str:
        """Get cost classification for UI display."""
        if self.cost_per_hour < 0.5:
            return "low"
        elif self.cost_per_hour < 1.0:
            return "medium"
        else:
            return "high"

    @property
    def uptime_hours(self) -> float:
        """Calculate uptime in hours if deployment is running."""
        if self.started_at and self.status == DeploymentStatus.RUNNING:
            delta = datetime.utcnow() - self.started_at
            return delta.total_seconds() / 3600
        return 0.0


class User(Base):
    """
    User table for authentication and authorization.

    Integrates with Clerk.com for authentication while maintaining local user data.
    """
    __tablename__ = "users"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Clerk Integration
    clerk_user_id = Column(String(255), unique=True, nullable=False, index=True)  # Clerk's user ID

    # User Information
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=True, index=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    avatar_url = Column(String(512), nullable=True)

    # Role-Based Access Control
    role = Column(Enum(UserRole), nullable=False, default=UserRole.USER, index=True)

    # Organization/Team (for future multi-tenant support)
    organization_id = Column(String(255), nullable=True, index=True)
    organization_name = Column(String(255), nullable=True)

    # Status
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    is_verified = Column(Boolean, nullable=False, default=False)

    # Usage Quotas
    api_calls_quota = Column(Integer, nullable=False, default=1000)  # Monthly quota
    api_calls_used = Column(Integer, nullable=False, default=0)
    deployment_quota = Column(Integer, nullable=False, default=10)  # Max concurrent deployments
    storage_quota_gb = Column(Float, nullable=False, default=10.0)

    # Billing
    stripe_customer_id = Column(String(255), nullable=True, unique=True)
    subscription_tier = Column(String(50), nullable=False, default="free")  # free, pro, enterprise

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    # Metadata
    user_metadata = Column(JSON, nullable=True)

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role={self.role.value})>"

    def to_dict(self) -> dict:
        """Convert user to dictionary representation."""
        return {
            "id": self.id,
            "clerk_user_id": self.clerk_user_id,
            "email": self.email,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "avatar_url": self.avatar_url,
            "role": self.role.value,
            "organization_id": self.organization_id,
            "organization_name": self.organization_name,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "subscription_tier": self.subscription_tier,
            "quotas": {
                "api_calls": {
                    "limit": self.api_calls_quota,
                    "used": self.api_calls_used,
                    "remaining": self.api_calls_quota - self.api_calls_used
                },
                "deployments": self.deployment_quota,
                "storage_gb": self.storage_quota_gb
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }

    @property
    def full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username or self.email

    @property
    def api_quota_remaining(self) -> int:
        """Get remaining API quota."""
        return max(0, self.api_calls_quota - self.api_calls_used)

    @property
    def api_quota_percentage(self) -> float:
        """Get API quota usage percentage."""
        if self.api_calls_quota == 0:
            return 100.0
        return (self.api_calls_used / self.api_calls_quota) * 100


class APIKey(Base):
    """
    API key table for programmatic access.

    Allows users to create API keys for authenticating API requests.
    """
    __tablename__ = "api_keys"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Key Information
    key_name = Column(String(255), nullable=False)  # User-defined name
    key_prefix = Column(String(20), nullable=False, index=True)  # First 8 chars for identification
    key_hash = Column(String(255), nullable=False, unique=True)  # SHA-256 hash of the full key

    # Permissions
    scopes = Column(JSON, nullable=False, default=list)  # List of allowed operations

    # Status
    status = Column(Enum(APIKeyStatus), nullable=False, default=APIKeyStatus.ACTIVE, index=True)

    # Expiration
    expires_at = Column(DateTime, nullable=True)

    # Usage Tracking
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, nullable=False, default=0)

    # Rate Limiting (per key)
    rate_limit_per_minute = Column(Integer, nullable=True)
    rate_limit_per_hour = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    revoked_at = Column(DateTime, nullable=True)

    # Metadata
    key_metadata = Column(JSON, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.key_name}', prefix='{self.key_prefix}', status={self.status.value})>"

    def to_dict(self, include_hash: bool = False) -> dict:
        """Convert API key to dictionary representation."""
        result = {
            "id": self.id,
            "user_id": self.user_id,
            "key_name": self.key_name,
            "key_prefix": self.key_prefix,
            "scopes": self.scopes or [],
            "status": self.status.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "rate_limit_per_hour": self.rate_limit_per_hour,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
        }
        if include_hash:
            result["key_hash"] = self.key_hash
        return result

    @property
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if API key is valid (active and not expired)."""
        return self.status == APIKeyStatus.ACTIVE and not self.is_expired


class AuditLog(Base):
    """
    Audit log table for tracking all user actions.

    Provides complete audit trail for security and compliance.
    """
    __tablename__ = "audit_logs"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # Nullable for system actions

    # Action Information
    action = Column(Enum(AuditAction), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)  # e.g., 'deployment', 'model', 'api_key'
    resource_id = Column(String(255), nullable=True, index=True)  # ID of affected resource

    # Request Information
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(String(512), nullable=True)
    request_method = Column(String(10), nullable=True)  # GET, POST, PUT, DELETE
    request_path = Column(String(512), nullable=True)

    # Details
    description = Column(Text, nullable=True)
    changes = Column(JSON, nullable=True)  # Before/after changes
    log_metadata = Column(JSON, nullable=True)

    # Status
    status = Column(String(50), nullable=False, default="success")  # success, failure, error
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action.value}, resource_type='{self.resource_type}', user_id={self.user_id})>"

    def to_dict(self) -> dict:
        """Convert audit log to dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_method": self.request_method,
            "request_path": self.request_path,
            "description": self.description,
            "changes": self.changes,
            "metadata": self.log_metadata,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Invoice(Base):
    """
    Invoice table for billing and payment tracking.

    Stores invoices for user subscriptions and usage-based billing.
    """
    __tablename__ = "invoices"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Invoice Information
    invoice_number = Column(String(100), unique=True, nullable=False, index=True)
    invoice_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    due_date = Column(DateTime, nullable=False, index=True)

    # Status
    status = Column(Enum(InvoiceStatus), nullable=False, default=InvoiceStatus.DRAFT, index=True)

    # Amounts (in USD)
    subtotal = Column(Float, nullable=False, default=0.0)
    tax = Column(Float, nullable=False, default=0.0)
    total = Column(Float, nullable=False, default=0.0)
    amount_paid = Column(Float, nullable=False, default=0.0)

    # Billing Period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    # Line Items (JSON array)
    line_items = Column(JSON, nullable=False, default=list)

    # Payment Information
    payment_method = Column(Enum(PaymentMethod), nullable=True)
    stripe_invoice_id = Column(String(255), nullable=True, unique=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)

    # Payment Dates
    paid_at = Column(DateTime, nullable=True)

    # Notes
    notes = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="invoices")

    def __repr__(self):
        return f"<Invoice(id={self.id}, number='{self.invoice_number}', status={self.status.value}, total={self.total})>"

    def to_dict(self) -> dict:
        """Convert invoice to dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date.isoformat() if self.invoice_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "status": self.status.value,
            "subtotal": round(self.subtotal, 2),
            "tax": round(self.tax, 2),
            "total": round(self.total, 2),
            "amount_paid": round(self.amount_paid, 2),
            "amount_due": round(self.total - self.amount_paid, 2),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "line_items": self.line_items or [],
            "payment_method": self.payment_method.value if self.payment_method else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @property
    def is_paid(self) -> bool:
        """Check if invoice is paid."""
        return self.status == InvoiceStatus.PAID

    @property
    def is_overdue(self) -> bool:
        """Check if invoice is overdue."""
        if self.is_paid:
            return False
        return datetime.utcnow() > self.due_date


class CostTracking(Base):
    """
    Cost tracking table for real-time usage and cost monitoring.

    Records deployment costs and API usage costs.
    """
    __tablename__ = "cost_tracking"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"), nullable=True, index=True)

    # Cost Information
    resource_type = Column(String(50), nullable=False, index=True)  # deployment, api_call, storage
    cost_usd = Column(Float, nullable=False, default=0.0)

    # Usage Details
    usage_amount = Column(Float, nullable=True)  # hours, calls, GB
    usage_unit = Column(String(20), nullable=True)  # hours, calls, GB

    # Provider and Resource Info
    provider_name = Column(String(100), nullable=True)
    resource_name = Column(String(255), nullable=True)

    # Time Period
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)

    # Additional Details
    cost_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", backref="cost_records")
    deployment = relationship("Deployment", backref="cost_records")

    def __repr__(self):
        return f"<CostTracking(id={self.id}, type='{self.resource_type}', cost=${self.cost_usd})>"

    def to_dict(self) -> dict:
        """Convert cost record to dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "deployment_id": self.deployment_id,
            "resource_type": self.resource_type,
            "cost_usd": round(self.cost_usd, 4),
            "usage_amount": self.usage_amount,
            "usage_unit": self.usage_unit,
            "provider_name": self.provider_name,
            "resource_name": self.resource_name,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "metadata": self.cost_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BudgetAlert(Base):
    """
    Budget alert table for cost monitoring and notifications.

    Allows users to set budget thresholds and receive alerts.
    """
    __tablename__ = "budget_alerts"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Alert Configuration
    alert_name = Column(String(255), nullable=False)
    alert_type = Column(Enum(BudgetAlertType), nullable=False, default=BudgetAlertType.THRESHOLD)

    # Budget Thresholds (in USD)
    threshold_amount = Column(Float, nullable=False)
    current_amount = Column(Float, nullable=False, default=0.0)

    # Time Period
    period_type = Column(String(20), nullable=False, default="monthly")  # daily, weekly, monthly, yearly
    period_start = Column(DateTime, nullable=True)
    period_end = Column(DateTime, nullable=True)

    # Alert Status
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    is_triggered = Column(Boolean, nullable=False, default=False)
    last_triggered_at = Column(DateTime, nullable=True)

    # Notification Settings
    notify_email = Column(Boolean, nullable=False, default=True)
    notify_slack = Column(Boolean, nullable=False, default=False)
    slack_webhook_url = Column(String(512), nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="budget_alerts")

    def __repr__(self):
        return f"<BudgetAlert(id={self.id}, name='{self.alert_name}', threshold=${self.threshold_amount})>"

    def to_dict(self) -> dict:
        """Convert budget alert to dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "alert_name": self.alert_name,
            "alert_type": self.alert_type.value,
            "threshold_amount": round(self.threshold_amount, 2),
            "current_amount": round(self.current_amount, 2),
            "percentage": round((self.current_amount / self.threshold_amount * 100), 2) if self.threshold_amount > 0 else 0,
            "period_type": self.period_type,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "is_active": self.is_active,
            "is_triggered": self.is_triggered,
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "notify_email": self.notify_email,
            "notify_slack": self.notify_slack,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @property
    def percentage_used(self) -> float:
        """Get percentage of budget used."""
        if self.threshold_amount == 0:
            return 0.0
        return (self.current_amount / self.threshold_amount) * 100

    @property
    def should_trigger(self) -> bool:
        """Check if alert should be triggered."""
        return self.is_active and self.current_amount >= self.threshold_amount and not self.is_triggered


class ProcessedDocument(Base):
    """
    Tracks processed documents to prevent duplicate processing.
    Documents are identified by content hash (SHA-256) instead of filename.
    """
    __tablename__ = "processed_documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Document Identification
    file_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256 hash
    file_name = Column(String(512), nullable=True)  # Original filename (for reference)
    file_size_bytes = Column(Integer, nullable=True)  # File size in bytes
    mime_type = Column(String(100), nullable=True)  # e.g., 'application/pdf', 'image/jpeg'

    # Processing Information
    first_processed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    last_processed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    processing_count = Column(Integer, nullable=False, default=1)  # Number of times processed

    # OCR Results (latest)
    provider_used = Column(String(100), nullable=True)  # e.g., 'mistral', 'google_vision'
    strategy_used = Column(String(50), nullable=True)  # e.g., 'single', 'fallback'
    extracted_text_preview = Column(Text, nullable=True)  # First 500 chars of extracted text
    confidence_score = Column(Float, nullable=True)  # OCR confidence score (0.0-1.0)

    # Processing Results
    processing_time_seconds = Column(Float, nullable=True)  # Processing duration
    token_count = Column(Integer, nullable=True)  # Number of tokens used (if applicable)
    cost_usd = Column(Float, nullable=True)  # Processing cost in USD

    # Status
    processing_status = Column(String(50), nullable=False, default="completed")  # completed, failed, skipped
    error_message = Column(Text, nullable=True)  # Error message if processing failed

    # Metadata
    user_id = Column(Integer, nullable=True, index=True)  # User who processed the document
    job_id = Column(String(100), nullable=True, index=True)  # Batch job ID if part of batch
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ProcessedDocument(hash='{self.file_hash[:16]}...', name='{self.file_name}')>"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_hash": self.file_hash,
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "mime_type": self.mime_type,
            "first_processed_at": self.first_processed_at.isoformat() if self.first_processed_at else None,
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
            "processing_count": self.processing_count,
            "provider_used": self.provider_used,
            "strategy_used": self.strategy_used,
            "extracted_text_preview": self.extracted_text_preview,
            "confidence_score": self.confidence_score,
            "processing_time_seconds": self.processing_time_seconds,
            "token_count": self.token_count,
            "cost_usd": self.cost_usd,
            "processing_status": self.processing_status,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "job_id": self.job_id,
            "extra_metadata": self.extra_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Settings(Base):
    """
    Application settings storage.
    Stores configuration for all platform features.
    """
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Settings category (global, user-specific, deployment-specific)
    category = Column(String(100), nullable=False, default="global", index=True)
    user_id = Column(Integer, nullable=True, index=True)  # NULL for global settings

    # Settings data stored as JSON
    settings_json = Column(Text, nullable=False)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Settings(id={self.id}, category={self.category})>"


class Notification(Base):
    """
    User notifications for system events, deployments, batch jobs, etc.
    """
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Notification details
    type = Column(String(50), nullable=False, index=True)  # deployment, batch, alert, success, error, info
    message = Column(Text, nullable=False)
    link = Column(String(512), nullable=True)  # Optional link to related resource

    # User targeting
    user_id = Column(Integer, nullable=True, index=True)  # NULL for global notifications

    # Read status
    read = Column(Boolean, nullable=False, default=False, index=True)
    read_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<Notification(id={self.id}, type={self.type}, read={self.read})>"
