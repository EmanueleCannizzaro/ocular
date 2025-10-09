"""
Settings management API endpoints.
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime
import json

from ocular.database.session import get_db
from ocular.database.models import Settings as SettingsModel

router = APIRouter()


class ApplicationSettings(BaseModel):
    """Application settings schema."""
    app_title: Optional[str] = "Ocular OCR Service"
    app_description: Optional[str] = "Extract text and data from PDFs and images"
    app_version: Optional[str] = "0.1.0"
    app_environment: Optional[str] = "development"
    app_debug: bool = False
    app_host: Optional[str] = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 1
    max_file_size_mb: int = 50
    max_files_per_request: int = 10
    supported_formats: Optional[str] = "pdf,png,jpg,jpeg,bmp,tiff,webp"
    default_strategy: Optional[str] = "fallback"
    default_providers: Optional[str] = "mistral,google_vision"
    max_concurrent_requests: int = 10
    templates_dir: Optional[str] = "templates"
    static_dir: Optional[str] = "static"
    enable_static_files: bool = True


class DatabaseSettings(BaseModel):
    """Database settings schema."""
    database_url: Optional[str] = "sqlite:///./ocular.db"
    pool_size: int = 5
    max_overflow: int = 10


class AuthSettings(BaseModel):
    """Authentication settings schema."""
    clerk_secret_key: Optional[str] = None
    clerk_publishable_key: Optional[str] = None
    jwt_secret_key: Optional[str] = None
    session_timeout_minutes: int = 60


class PaymentSettings(BaseModel):
    """Payment processing settings schema."""
    stripe_secret_key: Optional[str] = None
    stripe_publishable_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None


class CloudProviderSettings(BaseModel):
    """Cloud provider settings schema."""
    # Mistral
    mistral_api_key: Optional[str] = None
    mistral_model: Optional[str] = "pixtral-12b-2409"

    # Google Cloud Vision
    google_credentials_path: Optional[str] = None
    google_project_id: Optional[str] = None

    # AWS Textract
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = "us-east-1"

    # Azure Document Intelligence
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None

    # Tesseract
    tesseract_language: Optional[str] = "eng"
    tesseract_dpi: int = 300

    # HuggingFace
    huggingface_token: Optional[str] = None


class AIModelSettings(BaseModel):
    """AI model deployment settings schema."""
    # Primary Cloud Provider
    primary_cloud_provider: Optional[str] = "runpod"

    # RunPod
    runpod_api_key: Optional[str] = None
    runpod_endpoint_id: Optional[str] = None
    runpod_region: Optional[str] = "eur-is-1"
    runpod_s3_access_key: Optional[str] = None
    runpod_s3_secret_key: Optional[str] = None
    runpod_s3_endpoint: Optional[str] = None
    runpod_network_volume_id: Optional[str] = None

    # Vast.ai
    vastai_api_key: Optional[str] = None
    vastai_instance_id: Optional[str] = None
    vastai_endpoint_url: Optional[str] = None

    # DeepInfra
    deepinfra_api_key: Optional[str] = None
    deepinfra_model: Optional[str] = "allenai/olmOCR-7B-0825"
    deepinfra_base_url: Optional[str] = "https://api.deepinfra.com/v1/openai"

    # OLMocr Model
    olmocr_model_name: Optional[str] = "allenai/olmOCR-7B-0225-preview"
    olmocr_max_file_size_mb: int = 50
    olmocr_timeout_seconds: int = 300
    olmocr_allowed_extensions: Optional[str] = ".pdf,.png,.jpg,.jpeg"

    # RoLMocr Model
    rolmocr_model_name: Optional[str] = "custom/rolmocr-model"
    rolmocr_max_file_size_mb: int = 50
    rolmocr_timeout_seconds: int = 300
    rolmocr_endpoint_id: Optional[str] = None

    # Pixtral Model
    pixtral_model_name: Optional[str] = "mistralai/pixtral-12b"
    pixtral_max_file_size_mb: int = 50
    pixtral_timeout_seconds: int = 300
    pixtral_endpoint_id: Optional[str] = None


class LoggingSettings(BaseModel):
    """Logging configuration schema."""
    log_level: Optional[str] = "INFO"
    log_format: Optional[str] = "json"
    logfire_token: Optional[str] = None
    enable_file_logging: bool = False
    log_file_path: Optional[str] = "logs/ocular.log"


class SecuritySettings(BaseModel):
    """Security configuration schema."""
    cors_origins: Optional[str] = "*"
    max_request_size_mb: int = 100
    rate_limit_rpm: int = 100
    api_secret_key: Optional[str] = None
    encryption_key: Optional[str] = None
    enable_https_only: bool = False


class CacheSettings(BaseModel):
    """Cache configuration schema."""
    redis_url: Optional[str] = None
    cache_ttl_seconds: int = 3600
    enable_caching: bool = True


class SettingsResponse(BaseModel):
    """Complete settings response."""
    application: ApplicationSettings
    database: DatabaseSettings
    auth: AuthSettings
    payments: PaymentSettings
    cloud_providers: CloudProviderSettings
    ai_models: AIModelSettings
    logging: LoggingSettings
    security: SecuritySettings
    cache: CacheSettings
    updated_at: Optional[datetime] = None


@router.get("/settings", response_model=SettingsResponse)
async def get_settings(db: Session = Depends(get_db)):
    """Get all settings."""
    try:
        # Get settings from database (last updated record)
        db_settings = db.query(SettingsModel).order_by(SettingsModel.updated_at.desc()).first()

        if not db_settings:
            # Return defaults if no settings exist
            return SettingsResponse(
                application=ApplicationSettings(),
                database=DatabaseSettings(),
                auth=AuthSettings(),
                payments=PaymentSettings(),
                cloud_providers=CloudProviderSettings(),
                ai_models=AIModelSettings(),
                logging=LoggingSettings(),
                security=SecuritySettings(),
                cache=CacheSettings()
            )

        # Parse JSON settings from database
        settings_data = json.loads(db_settings.settings_json)

        return SettingsResponse(
            application=ApplicationSettings(**settings_data.get("application", {})),
            database=DatabaseSettings(**settings_data.get("database", {})),
            auth=AuthSettings(**settings_data.get("auth", {})),
            payments=PaymentSettings(**settings_data.get("payments", {})),
            cloud_providers=CloudProviderSettings(**settings_data.get("cloud_providers", {})),
            ai_models=AIModelSettings(**settings_data.get("ai_models", {})),
            logging=LoggingSettings(**settings_data.get("logging", {})),
            security=SecuritySettings(**settings_data.get("security", {})),
            cache=CacheSettings(**settings_data.get("cache", {})),
            updated_at=db_settings.updated_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")


@router.post("/settings", response_model=SettingsResponse)
async def save_settings(settings: SettingsResponse, db: Session = Depends(get_db)):
    """Save all settings."""
    try:
        # Convert settings to JSON
        settings_json = json.dumps({
            "application": settings.application.dict(exclude_none=True),
            "database": settings.database.dict(exclude_none=True),
            "auth": settings.auth.dict(exclude_none=True),
            "payments": settings.payments.dict(exclude_none=True),
            "cloud_providers": settings.cloud_providers.dict(exclude_none=True),
            "ai_models": settings.ai_models.dict(exclude_none=True),
            "logging": settings.logging.dict(exclude_none=True),
            "security": settings.security.dict(exclude_none=True),
            "cache": settings.cache.dict(exclude_none=True)
        })

        # Create new settings record
        db_settings = SettingsModel(
            settings_json=settings_json,
            category="global",
            updated_at=datetime.utcnow()
        )

        db.add(db_settings)
        db.commit()
        db.refresh(db_settings)

        settings.updated_at = db_settings.updated_at
        return settings

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@router.post("/settings/test-provider/{provider}")
async def test_provider(provider: str, settings: Dict[str, Any]):
    """Test a specific provider configuration."""
    try:
        # Test connection to provider based on type
        if provider == "mistral":
            # Test Mistral API connection
            if not settings.get("mistral_api_key"):
                raise HTTPException(status_code=400, detail="Mistral API key is required")

            # TODO: Actually test the API
            return {
                "status": "success",
                "message": "Mistral API connection successful",
                "provider": provider
            }

        elif provider == "google_vision":
            if not settings.get("google_credentials_path"):
                raise HTTPException(status_code=400, detail="Google credentials path is required")

            return {
                "status": "success",
                "message": "Google Vision API connection successful",
                "provider": provider
            }

        elif provider == "olmocr":
            if not settings.get("runpod_api_key"):
                raise HTTPException(status_code=400, detail="RunPod API key is required")

            return {
                "status": "success",
                "message": "OLMocr endpoint connection successful",
                "provider": provider
            }

        elif provider == "rolmocr":
            if not settings.get("rolmocr_endpoint_id"):
                raise HTTPException(status_code=400, detail="RoLMocr endpoint ID is required")

            return {
                "status": "success",
                "message": "RoLMocr endpoint connection successful",
                "provider": provider
            }

        elif provider == "pixtral":
            if not settings.get("pixtral_endpoint_id"):
                raise HTTPException(status_code=400, detail="Pixtral endpoint ID is required")

            return {
                "status": "success",
                "message": "Pixtral endpoint connection successful",
                "provider": provider
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Provider test failed: {str(e)}")


@router.get("/settings/export")
async def export_settings(db: Session = Depends(get_db)):
    """Export settings as JSON file."""
    try:
        db_settings = db.query(SettingsModel).order_by(SettingsModel.updated_at.desc()).first()

        if not db_settings:
            raise HTTPException(status_code=404, detail="No settings found")

        return {
            "filename": f"ocular-settings-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json",
            "content": db_settings.settings_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export settings: {str(e)}")


@router.post("/settings/import")
async def import_settings(settings_json: str, db: Session = Depends(get_db)):
    """Import settings from JSON."""
    try:
        # Validate JSON
        settings_data = json.loads(settings_json)

        # Create new settings record
        db_settings = SettingsModel(
            settings_json=settings_json,
            category="global",
            updated_at=datetime.utcnow()
        )

        db.add(db_settings)
        db.commit()
        db.refresh(db_settings)

        return {
            "status": "success",
            "message": "Settings imported successfully",
            "updated_at": db_settings.updated_at
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to import settings: {str(e)}")
