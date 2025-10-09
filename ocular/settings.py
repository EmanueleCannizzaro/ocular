"""
Centralized settings and configuration for Ocular OCR application.
Uses Pydantic BaseSettings for environment variable management.
Loads configuration from ocular.json and .env files.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import json


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    pool_size: int = Field(default=5)
    max_overflow: int = Field(default=10)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DB_",
        extra="ignore",
    )


class AuthenticationSettings(BaseSettings):
    """Authentication and authorization settings"""
    clerk_secret_key: Optional[str] = Field(default=None, alias="CLERK_SECRET_KEY")
    clerk_publishable_key: Optional[str] = Field(default=None, alias="CLERK_PUBLISHABLE_KEY")
    jwt_secret_key: Optional[str] = Field(default=None, alias="JWT_SECRET_KEY")
    session_timeout_minutes: int = Field(default=60)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AUTH_",
        extra="ignore",
    )


class PaymentSettings(BaseSettings):
    """Payment processing settings"""
    stripe_secret_key: Optional[str] = Field(default=None, alias="STRIPE_SECRET_KEY")
    stripe_publishable_key: Optional[str] = Field(default=None, alias="STRIPE_PUBLISHABLE_KEY")
    stripe_webhook_secret: Optional[str] = Field(default=None, alias="STRIPE_WEBHOOK_SECRET")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="STRIPE_",
        extra="ignore"
    )


class OLMocrSettings(BaseSettings):
    """OLMocr Cloud GPU integration settings (RunPod/Vast.ai/DeepInfra)"""
    # Provider selection
    provider: str = Field(default="runpod", alias="OLMOCR_PROVIDER")  # runpod, vastai, or deepinfra

    # RunPod configuration
    runpod_endpoint_id: str = Field(default="", alias="RUNPOD_ENDPOINT_ID")
    runpod_api_key: str = Field(default="", alias="RUNPOD_API_KEY")

    # RunPod S3 Storage configuration
    runpod_access_key_id: str = Field(default="", alias="RUNPOD_ACCESS_KEY_ID")
    runpod_secret_access_key: str = Field(default="", alias="RUNPOD_SECRET_ACCESS_KEY")
    runpod_bucket_endpoint_url: str = Field(default="", alias="RUNPOD_BUCKET_ENDPOINT_URL")
    runpod_network_volume_id: str = Field(default="", alias="RUNPOD_NETWORK_VOLUME_ID")
    runpod_region: str = Field(default="eur-is-1", alias="RUNPOD_REGION")

    # Vast.ai configuration
    vastai_instance_id: str = Field(default="", alias="VASTAI_INSTANCE_ID")
    vastai_api_key: str = Field(default="", alias="VASTAI_API_KEY")
    vastai_endpoint_url: str = Field(default="", alias="VASTAI_ENDPOINT_URL")

    # DeepInfra configuration
    deepinfra_api_key: str = Field(default="", alias="DEEPINFRA_TOKEN")
    deepinfra_model: str = Field(default="allenai/olmOCR-7B-0825", alias="DEEPINFRA_MODEL")
    deepinfra_base_url: str = Field(default="https://api.deepinfra.com/v1/openai", alias="DEEPINFRA_BASE_URL")

    # Model configuration
    model_name: str = Field(default="allenai/olmOCR-7B-0225-preview", alias="OLMOCR_MODEL_NAME")
    max_file_size_mb: int = Field(default=50, alias="OLMOCR_MAX_FILE_SIZE_MB")
    timeout_seconds: int = Field(default=300, alias="OLMOCR_TIMEOUT_SECONDS")
    allowed_extensions: Union[str, List[str]] = Field(default=[".pdf", ".png", ".jpg", ".jpeg"], alias="OLMOCR_ALLOWED_EXTENSIONS")

    @property
    def runpod_url(self) -> str:
        """Generate RunPod API URL"""
        return f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}/run"

    @property
    def vastai_url(self) -> str:
        """Get Vast.ai endpoint URL"""
        return self.vastai_endpoint_url

    @property
    def active_endpoint_url(self) -> str:
        """Get the active endpoint URL based on provider"""
        if self.provider == "vastai":
            return self.vastai_url
        elif self.provider == "deepinfra":
            return self.deepinfra_base_url
        else:
            return self.runpod_url

    @property
    def active_api_key(self) -> str:
        """Get the active API key based on provider"""
        if self.provider == "vastai":
            return self.vastai_api_key
        elif self.provider == "deepinfra":
            return self.deepinfra_api_key
        else:
            return self.runpod_api_key

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def is_configured(self) -> bool:
        """Check if OLMocr is properly configured"""
        if self.provider == "vastai":
            return bool(self.vastai_instance_id and self.vastai_api_key and self.vastai_endpoint_url)
        elif self.provider == "deepinfra":
            return bool(self.deepinfra_api_key)
        else:
            return bool(self.runpod_endpoint_id and self.runpod_api_key)

    @field_validator('allowed_extensions', mode='before')
    @classmethod
    def validate_extensions(cls, v):
        """Ensure extensions start with dot"""
        if isinstance(v, str):
            v = v.split(',')
        return [ext.strip() if ext.startswith('.') else f'.{ext.strip()}' for ext in v]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OLMOCR_",
        extra="ignore"
    )


class OCRProviderSettings(BaseSettings):
    """OCR provider configuration settings"""
    # Mistral AI
    mistral_api_key: Optional[str] = Field(default=None, alias="MISTRAL_API_KEY")
    mistral_model: str = Field(default="pixtral-12b-2409", alias="MISTRAL_MODEL")

    # Google Cloud Vision
    google_application_credentials: Optional[str] = Field(default=None, alias="GOOGLE_APPLICATION_CREDENTIALS")
    google_project_id: Optional[str] = Field(default=None, alias="GOOGLE_PROJECT_ID")

    # AWS Textract
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # Azure Document Intelligence
    azure_doc_intel_endpoint: Optional[str] = Field(default=None, alias="AZURE_DOC_INTEL_ENDPOINT")
    azure_doc_intel_api_key: Optional[str] = Field(default=None, alias="AZURE_DOC_INTEL_API_KEY")

    # Tesseract
    tesseract_language: str = Field(default="eng", alias="TESSERACT_LANGUAGE")
    tesseract_dpi: int = Field(default=300, alias="TESSERACT_DPI")

    # Hugging Face Token (for various models)
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OCR_",
        extra="ignore"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings"""
    level: str = Field(default="INFO", alias="LOG_LEVEL")
    format: str = Field(default="json", alias="LOG_FORMAT")
    logfire_token: Optional[str] = Field(default=None, alias="LOGFIRE_TOKEN")
    enable_file_logging: bool = Field(default=False, alias="ENABLE_FILE_LOGGING")
    log_file_path: str = Field(default="logs/ocular.log", alias="LOG_FILE_PATH")

    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LOG_",
        extra="ignore"
    )


class SecuritySettings(BaseSettings):
    """Security configuration settings"""
    cors_origins: Union[str, List[str]] = Field(default=["*"], alias="CORS_ORIGINS")
    max_request_size_mb: int = Field(default=100, alias="MAX_REQUEST_SIZE_MB")
    rate_limit_requests_per_minute: int = Field(default=100, alias="RATE_LIMIT_RPM")
    api_secret_key: Optional[str] = Field(default=None, alias="API_SECRET_KEY")
    encryption_key: Optional[str] = Field(default=None, alias="ENCRYPTION_KEY")
    enable_https_only: bool = Field(default=True, alias="ENABLE_HTTPS_ONLY")

    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SECURITY_",
        extra="ignore"
    )


class CacheSettings(BaseSettings):
    """Caching configuration settings"""
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")  # 1 hour
    enable_caching: bool = Field(default=True, alias="ENABLE_CACHING")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CACHE_",
        extra="ignore"
    )


class CloudProviderModel(BaseModel):
    """Cloud GPU provider configuration model"""
    id: int
    name: str
    display_name: str
    description: str = ""
    website: str = ""
    status: str = "INACTIVE"  # ACTIVE, INACTIVE, TESTING
    available_gpus: List[str] = Field(default_factory=list)
    features: Dict[str, bool] = Field(default_factory=dict)
    configuration: Dict[str, Any] = Field(default_factory=dict)


class AIModelModel(BaseModel):
    """AI model configuration model"""
    id: int
    name: str
    display_name: str
    version: str
    description: str = ""
    docker_image: str = ""
    min_vram_gb: int = 0
    recommended_gpu: str = ""
    supports_vllm: bool = False
    status: str = "INACTIVE"  # ACTIVE, INACTIVE, TESTING, DEPRECATED
    configuration: Dict[str, Any] = Field(default_factory=dict)


class ApplicationSettings(BaseSettings):
    """Main application settings"""
    # Basic app configuration
    title: str = Field(default="Ocular OCR Service", alias="APP_TITLE")
    description: str = Field(default="Extract text and data from PDFs and images using multiple OCR providers", alias="APP_DESCRIPTION")
    version: str = Field(default="0.1.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Server configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")

    # Templates and static files
    templates_dir: str = Field(default="templates", alias="TEMPLATES_DIR")
    static_dir: str = Field(default="static", alias="STATIC_DIR")
    enable_static_files: bool = Field(default=True, alias="ENABLE_STATIC_FILES")

    # Default processing configuration
    default_strategy: str = Field(default="fallback", alias="DEFAULT_STRATEGY")
    default_providers: Union[str, List[str]] = Field(default=["mistral"], alias="DEFAULT_PROVIDERS")
    max_concurrent_requests: int = Field(default=10, alias="MAX_CONCURRENT_REQUESTS")

    # File processing limits
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    max_files_per_request: int = Field(default=10, alias="MAX_FILES_PER_REQUEST")
    supported_formats: Union[str, List[str]] = Field(
        default=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        alias="SUPPORTED_FORMATS"
    )

    @field_validator('default_providers', mode='before')
    @classmethod
    def parse_default_providers(cls, v):
        """Parse default providers from string or list"""
        if isinstance(v, str):
            return [provider.strip() for provider in v.split(',')]
        return v

    @field_validator('supported_formats', mode='before')
    @classmethod
    def parse_supported_formats(cls, v):
        """Parse supported formats from string or list"""
        if isinstance(v, str):
            return [fmt.strip().lower() for fmt in v.split(',')]
        return [fmt.lower() for fmt in v]

    @field_validator('environment')
    @classmethod
    def validate_env_value(cls, v):
        """Validate environment setting"""
        valid_environments = ['development', 'staging', 'production', 'testing']
        if v.lower() not in valid_environments:
            raise ValueError(f'Environment must be one of: {valid_environments}')
        return v.lower()

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def templates_path(self) -> Path:
        """Get absolute path to templates directory"""
        templates_path = Path(self.templates_dir)
        if not templates_path.is_absolute():
            # If relative, resolve from package root
            package_root = Path(__file__).parent
            templates_path = (package_root / self.templates_dir).resolve()
        return templates_path

    @property
    def static_path(self) -> Path:
        """Get absolute path to static files directory"""
        static_path = Path(self.static_dir)
        if not static_path.is_absolute():
            # If relative, resolve from package root
            package_root = Path(__file__).parent
            static_path = (package_root / self.static_dir).resolve()
        return static_path

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class Settings:
    """Unified settings container

    Loads configuration from two sources:
    1. ocular.json - Non-sensitive configuration (models, providers, app settings)
    2. .env - Secrets and credentials (API keys, passwords, tokens)
    """

    def __init__(self):
        # Load configuration from ocular.json
        self._config_data = self._load_ocular_config()

        # Initialize all setting groups
        self.app = ApplicationSettings()
        self.database = DatabaseSettings()
        self.auth = AuthenticationSettings()
        self.payments = PaymentSettings()
        self.olmocr = OLMocrSettings()
        self.ocr_providers = OCRProviderSettings()
        self.logging = LoggingSettings()
        self.security = SecuritySettings()
        self.cache = CacheSettings()

        # Load cloud providers and AI models from ocular.json
        self.cloud_providers: List[CloudProviderModel] = self._load_cloud_providers()
        self.ai_models: List[AIModelModel] = self._load_ai_models()

        # Validate configuration and show warnings
        self._validate_configuration()

    def _load_ocular_config(self) -> Dict[str, Any]:
        """Load configuration from ocular.json file"""
        # Try to find ocular.json in project root
        config_file = Path(__file__).parent.parent / "ocular.json"

        if not config_file.exists():
            # Fallback to current directory
            config_file = Path("ocular.json")

        if not config_file.exists():
            print(f"⚠️  Configuration file not found: {config_file}")
            print("   Using defaults and environment variables only")
            return {}

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if self.app.debug if hasattr(self, 'app') else os.getenv('DEBUG', 'false').lower() == 'true':
                    print(f"✅ Loaded configuration from: {config_file}")
                return data
        except Exception as e:
            print(f"⚠️  Error loading {config_file}: {e}")
            print("   Using defaults and environment variables only")
            return {}

    def _load_cloud_providers(self) -> List[CloudProviderModel]:
        """Load cloud providers from ocular.json"""
        providers_data = self._config_data.get('cloud_providers', [])
        providers = []

        for provider_data in providers_data:
            try:
                provider = CloudProviderModel(**provider_data)
                providers.append(provider)
            except Exception as e:
                print(f"⚠️  Error loading cloud provider {provider_data.get('name', 'unknown')}: {e}")

        return providers

    def _load_ai_models(self) -> List[AIModelModel]:
        """Load AI models from ocular.json"""
        models_data = self._config_data.get('ai_models', [])
        models = []

        for model_data in models_data:
            try:
                model = AIModelModel(**model_data)
                models.append(model)
            except Exception as e:
                print(f"⚠️  Error loading AI model {model_data.get('name', 'unknown')}: {e}")

        return models

    def to_dict(self) -> Dict[str, Any]:
        """Convert all settings to dictionary"""
        return {
            "app": self.app.model_dump(),
            "database": self.database.model_dump(),
            "auth": self.auth.model_dump(),
            "payments": self.payments.model_dump(),
            "olmocr": self.olmocr.model_dump(),
            "ocr_providers": self.ocr_providers.model_dump(),
            "logging": self.logging.model_dump(),
            "security": self.security.model_dump(),
            "cache": self.cache.model_dump(),
            "cloud_providers": [p.model_dump() for p in self.cloud_providers],
            "ai_models": [m.model_dump() for m in self.ai_models],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get settings summary (without sensitive values)"""
        sensitive_keys = {
            'api_key', 'secret_key', 'password', 'token', 'credentials',
            'webhook_secret', 'encryption_key'
        }

        summary = {}
        for group_name, group_settings in self.to_dict().items():
            # Handle lists (cloud_providers, ai_models)
            if isinstance(group_settings, list):
                summary[group_name] = group_settings
                continue

            # Handle dictionaries (normal settings groups)
            summary[group_name] = {}
            for key, value in group_settings.items():
                # Hide sensitive values
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    summary[group_name][key] = "***HIDDEN***" if value else None
                else:
                    summary[group_name][key] = value

        return summary

    def _validate_configuration(self):
        """Validate configuration and show warnings for missing critical parameters"""
        warnings_list = []

        # Check for OCR providers
        ocr_providers_configured = []
        ocr_warnings = []

        if self.ocr_providers.mistral_api_key:
            ocr_providers_configured.append("Mistral AI")
        else:
            ocr_warnings.append("MISTRAL_API_KEY not set - Mistral AI provider will be unavailable")

        if self.ocr_providers.google_application_credentials and self.ocr_providers.google_project_id:
            ocr_providers_configured.append("Google Vision")
        elif self.ocr_providers.google_application_credentials or self.ocr_providers.google_project_id:
            ocr_warnings.append("Google Vision partially configured - need both GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_PROJECT_ID")

        if self.ocr_providers.aws_access_key_id and self.ocr_providers.aws_secret_access_key:
            ocr_providers_configured.append("AWS Textract")
        elif self.ocr_providers.aws_access_key_id or self.ocr_providers.aws_secret_access_key:
            ocr_warnings.append("AWS Textract partially configured - need both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")

        if self.ocr_providers.azure_doc_intel_endpoint and self.ocr_providers.azure_doc_intel_api_key:
            ocr_providers_configured.append("Azure Document Intelligence")
        elif self.ocr_providers.azure_doc_intel_endpoint or self.ocr_providers.azure_doc_intel_api_key:
            ocr_warnings.append("Azure Document Intelligence partially configured - need both AZURE_DOC_INTEL_ENDPOINT and AZURE_DOC_INTEL_API_KEY")

        if self.olmocr.is_configured:
            provider_name = f"OLMocr ({self.olmocr.provider.title()})"
            ocr_providers_configured.append(provider_name)
        else:
            if self.olmocr.provider == "vastai":
                if any([self.olmocr.vastai_instance_id, self.olmocr.vastai_api_key, self.olmocr.vastai_endpoint_url]):
                    ocr_warnings.append("OLMocr (Vast.ai) partially configured - need VASTAI_INSTANCE_ID, VASTAI_API_KEY, and VASTAI_ENDPOINT_URL")
            else:
                if any([self.olmocr.runpod_endpoint_id, self.olmocr.runpod_api_key]):
                    ocr_warnings.append("OLMocr (RunPod) partially configured - need both RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY")

        # Always available if system dependencies are met
        ocr_providers_configured.append("Tesseract (local)")

        # Critical warnings
        if not ocr_providers_configured or len(ocr_providers_configured) == 1:  # Only Tesseract
            warnings_list.append("⚠️  CRITICAL: No cloud OCR providers configured! Consider setting up at least MISTRAL_API_KEY for basic functionality.")

        if not self.ocr_providers.mistral_api_key:
            warnings_list.append("⚠️  RECOMMENDED: MISTRAL_API_KEY not set - this is the recommended primary OCR provider")

        # Authentication warnings
        auth_configured = False
        if self.auth.clerk_secret_key and self.auth.clerk_publishable_key:
            auth_configured = True
        elif self.auth.clerk_secret_key or self.auth.clerk_publishable_key:
            warnings_list.append("⚠️  Clerk authentication partially configured - need both CLERK_SECRET_KEY and CLERK_PUBLISHABLE_KEY")

        if not auth_configured and self.app.environment == "production":
            warnings_list.append("⚠️  No authentication configured for production environment - consider setting up Clerk")

        # Payment processing warnings
        payments_configured = False
        if self.payments.stripe_secret_key and self.payments.stripe_publishable_key:
            payments_configured = True
        elif self.payments.stripe_secret_key or self.payments.stripe_publishable_key:
            warnings_list.append("⚠️  Stripe payment processing partially configured - need both STRIPE_SECRET_KEY and STRIPE_PUBLISHABLE_KEY")

        # Security warnings
        if self.app.environment == "production":
            if self.app.debug:
                warnings_list.append("⚠️  SECURITY: DEBUG=true in production environment - set DEBUG=false")

            if "*" in self.security.cors_origins:
                warnings_list.append("⚠️  SECURITY: CORS_ORIGINS set to '*' in production - specify exact origins")

            if not self.security.enable_https_only:
                warnings_list.append("⚠️  SECURITY: ENABLE_HTTPS_ONLY=false in production - enable HTTPS")

        # Performance warnings
        if not self.cache.redis_url and self.app.environment == "production":
            warnings_list.append("ℹ️  PERFORMANCE: No Redis caching configured - consider setting REDIS_URL for better performance")

        # Required environment variables check
        required_vars = {
            'APP_TITLE': self.app.title != "Ocular OCR Service",
            'ENVIRONMENT': True,  # Always has default
        }

        for var_name, is_set in required_vars.items():
            if not is_set:
                warnings_list.append(f"⚠️  REQUIRED: {var_name} environment variable not set")

        # Print configuration summary
        if self.app.debug or self.app.environment == "development":
            print("🔧 Ocular OCR Configuration Summary:")
            print(f"   📱 App: {self.app.title} v{self.app.version} ({self.app.environment})")
            print(f"   🤖 OCR Providers: {', '.join(ocr_providers_configured) if ocr_providers_configured else 'None configured'}")
            print(f"   🔐 Authentication: {'Configured' if auth_configured else 'Not configured'}")
            print(f"   💳 Payments: {'Configured' if payments_configured else 'Not configured'}")
            print(f"   📊 Caching: {'Redis' if self.cache.redis_url else 'In-memory'}")
            print(f"   🌐 Environment: {self.app.environment}")

        # Show warnings
        if warnings_list:
            print("\n" + "="*60)
            print("⚠️  CONFIGURATION WARNINGS:")
            print("="*60)
            for warning in warnings_list:
                print(f"   {warning}")
            print("\n💡 See .env.template for complete configuration options")
            print("="*60 + "\n")

        # Add OCR-specific warnings
        for warning in ocr_warnings:
            if self.app.debug:
                print(f"   ℹ️  {warning}")

    def get_configured_providers(self) -> List[str]:
        """Get list of properly configured OCR providers"""
        configured = []

        if self.ocr_providers.mistral_api_key:
            configured.append("mistral")

        if self.ocr_providers.google_application_credentials and self.ocr_providers.google_project_id:
            configured.append("google_vision")

        if self.ocr_providers.aws_access_key_id and self.ocr_providers.aws_secret_access_key:
            configured.append("aws_textract")

        if self.ocr_providers.azure_doc_intel_endpoint and self.ocr_providers.azure_doc_intel_api_key:
            configured.append("azure_document_intelligence")

        if self.olmocr.runpod_endpoint_id and self.olmocr.runpod_api_key:
            configured.append("olm_ocr")

        # Tesseract is always potentially available (depends on system installation)
        configured.append("tesseract")

        return configured

    def is_production_ready(self) -> bool:
        """Check if configuration is ready for production deployment"""
        issues = []

        # Must have at least one cloud OCR provider
        configured_providers = self.get_configured_providers()
        cloud_providers = [p for p in configured_providers if p != "tesseract"]

        if not cloud_providers:
            issues.append("No cloud OCR providers configured")

        if self.app.environment == "production":
            if self.app.debug:
                issues.append("DEBUG enabled in production")

            if "*" in self.security.cors_origins:
                issues.append("CORS origins too permissive")

            if not self.security.enable_https_only:
                issues.append("HTTPS not enforced")

        return len(issues) == 0


# Global settings instance
settings = Settings()


# Convenience functions for backward compatibility
def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


def get_app_settings() -> ApplicationSettings:
    """Get application settings"""
    return settings.app


def get_olmocr_settings() -> OLMocrSettings:
    """Get OLMocr settings"""
    return settings.olmocr


def get_ocr_provider_settings() -> OCRProviderSettings:
    """Get OCR provider settings"""
    return settings.ocr_providers
