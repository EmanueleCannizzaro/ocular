"""
Configuration settings for Ocular OCR system.
"""

import os
import re
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from ..core.enums import ProviderType
from ..core.exceptions import ValidationError, ConfigurationError
from ..core.interfaces import ConfigurationProvider


class ProviderSettings(BaseSettings):
    """Settings for OCR providers."""
    
    # Mistral AI settings
    mistral_api_key: Optional[str] = Field(None, env="MISTRAL_API_KEY")
    mistral_model: str = Field("pixtral-12b-2409", env="MISTRAL_MODEL")
    mistral_timeout: int = Field(30, env="MISTRAL_TIMEOUT")
    mistral_max_retries: int = Field(3, env="MISTRAL_MAX_RETRIES")
    
    # RunPod OLM OCR settings
    olm_api_endpoint: Optional[str] = Field(None, env="OLM_API_ENDPOINT")
    olm_api_key: Optional[str] = Field(None, env="OLM_API_KEY")
    olm_model_name: str = Field("allenai/OLMo-7B-0724-hf", env="OLM_MODEL_NAME")
    olm_timeout: int = Field(60, env="OLM_TIMEOUT")
    
    # RunPod RoLM OCR settings
    rolm_api_endpoint: Optional[str] = Field(None, env="ROLM_API_ENDPOINT")
    rolm_api_key: Optional[str] = Field(None, env="ROLM_API_KEY")
    rolm_model_name: str = Field("microsoft/trocr-large-printed", env="ROLM_MODEL_NAME")
    rolm_timeout: int = Field(60, env="ROLM_TIMEOUT")
    
    # Google Cloud Vision API settings
    google_credentials_path: Optional[str] = Field(None, env="GOOGLE_APPLICATION_CREDENTIALS")
    google_project_id: Optional[str] = Field(None, env="GOOGLE_PROJECT_ID")
    google_timeout: int = Field(30, env="GOOGLE_TIMEOUT")
    
    # AWS Textract settings
    aws_region: str = Field("us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = Field(None, env="AWS_SESSION_TOKEN")
    aws_timeout: int = Field(60, env="AWS_TIMEOUT")
    
    # Tesseract OCR settings
    tesseract_language: str = Field("eng", env="TESSERACT_LANGUAGE")
    tesseract_dpi: int = Field(300, env="TESSERACT_DPI")
    tesseract_timeout: int = Field(30, env="TESSERACT_TIMEOUT")
    tesseract_cmd: Optional[str] = Field(None, env="TESSERACT_CMD")
    
    # Azure Document Intelligence settings
    azure_doc_intel_endpoint: Optional[str] = Field(None, env="AZURE_DOC_INTEL_ENDPOINT")
    azure_doc_intel_api_key: Optional[str] = Field(None, env="AZURE_DOC_INTEL_API_KEY")
    azure_doc_intel_timeout: int = Field(120, env="AZURE_DOC_INTEL_TIMEOUT")
    
    # Local model settings (deprecated but kept for backward compatibility)
    device: str = Field("auto", env="DEVICE")  # auto, cpu, cuda
    
    # Provider priorities
    provider_priorities: Dict[str, int] = Field(
        default={
            "mistral": 1, 
            "google_vision": 2, 
            "aws_textract": 3, 
            "azure_document_intelligence": 4,
            "tesseract": 5,
            "olm_ocr": 6, 
            "rolm_ocr": 7
        },
        env="PROVIDER_PRIORITIES"
    )
    
    # Enabled providers
    enabled_providers: List[str] = Field(
        default=["mistral", "tesseract"],
        env="ENABLED_PROVIDERS"
    )
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        valid_devices = ['auto', 'cpu', 'cuda']
        if v not in valid_devices:
            raise ValueError(f'Device must be one of: {valid_devices}')
        return v
    
    @field_validator('enabled_providers', mode='before')
    @classmethod
    def parse_enabled_providers(cls, v):
        if isinstance(v, str):
            return [p.strip() for p in v.split(',') if p.strip()]
        return v
    
    class Config:
        env_prefix = ""  # No prefix for provider settings to use direct env vars
        case_sensitive = False
        extra = "ignore"


class FileSettings(BaseSettings):
    """Settings for file handling."""
    
    max_file_size_mb: int = Field(10, env="MAX_FILE_SIZE_MB")
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        env="ALLOWED_EXTENSIONS"
    )
    temp_dir: Optional[Path] = Field(None, env="TEMP_DIR")
    upload_dir: Optional[Path] = Field(None, env="UPLOAD_DIR")
    
    @field_validator('temp_dir', 'upload_dir', mode='before')
    @classmethod
    def parse_path(cls, v):
        return Path(v) if v else None
    
    @field_validator('allowed_extensions', mode='before')
    @classmethod
    def parse_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        return v
    
    class Config:
        env_prefix = "OCULAR_FILE_"
        extra = "ignore"


class ProcessingSettings(BaseSettings):
    """Settings for processing behavior."""
    
    default_strategy: str = Field("fallback", env="DEFAULT_STRATEGY")
    max_concurrent_requests: int = Field(5, env="MAX_CONCURRENT_REQUESTS")
    request_timeout_seconds: int = Field(300, env="REQUEST_TIMEOUT_SECONDS")
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    
    @field_validator('default_strategy')
    @classmethod
    def validate_strategy(cls, v):
        valid_strategies = ['single', 'fallback', 'ensemble', 'best']
        if v not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {valid_strategies}')
        return v
    
    class Config:
        env_prefix = "OCULAR_PROCESSING_"
        extra = "ignore"


class WebSettings(BaseSettings):
    """Settings for web application."""
    
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    reload: bool = Field(False, env="RELOAD")
    
    # Security settings
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    max_request_size_mb: int = Field(50, env="MAX_REQUEST_SIZE_MB")
    
    # Static files
    static_dir: Path = Field(Path("web/static"), env="STATIC_DIR")
    template_dir: Path = Field(Path("web/templates"), env="TEMPLATE_DIR")
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    @field_validator('static_dir', 'template_dir', mode='before')
    @classmethod
    def parse_path(cls, v):
        return Path(v) if isinstance(v, str) else v
    
    class Config:
        env_prefix = "OCULAR_WEB_"
        extra = "ignore"


class LoggingSettings(BaseSettings):
    """Settings for logging."""
    
    level: str = Field("INFO", env="LOG_LEVEL")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_path: Optional[Path] = Field(None, env="LOG_FILE")
    max_file_size_mb: int = Field(10, env="LOG_MAX_FILE_SIZE_MB")
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    @field_validator('file_path', mode='before')
    @classmethod
    def parse_path(cls, v):
        return Path(v) if v else None
    
    class Config:
        env_prefix = "OCULAR_LOG_"
        extra = "ignore"


class OcularSettings(BaseSettings, ConfigurationProvider):
    """Main settings class combining all configuration with validation and management."""
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    
    # Sub-configurations
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    files: FileSettings = Field(default_factory=FileSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Runtime configuration
    _runtime_config: Dict[str, Any] = {}
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        valid_envs = ['development', 'production', 'testing']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v
    
    @classmethod
    def from_env_file(cls, env_file: Optional[str] = None) -> "OcularSettings":
        """Load settings from environment file."""
        if env_file:
            return cls(_env_file=env_file)
        
        # Try to find .env file
        env_files = ['.env', '.env.local', '.env.development']
        for env_file in env_files:
            if os.path.exists(env_file):
                return cls(_env_file=env_file)
        
        # Return with environment variables only
        return cls()
    
    def get_provider_config(self, provider_type: ProviderType) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        if provider_type == ProviderType.MISTRAL:
            return {
                "api_key": self.providers.mistral_api_key,
                "model": self.providers.mistral_model,
                "timeout": self.providers.mistral_timeout,
                "max_retries": self.providers.mistral_max_retries,
            }
        elif provider_type == ProviderType.OLM_OCR:
            return {
                "api_endpoint": self.providers.olm_api_endpoint,
                "api_key": self.providers.olm_api_key,
                "model_name": self.providers.olm_model_name,
                "timeout": self.providers.olm_timeout,
            }
        elif provider_type == ProviderType.ROLM_OCR:
            return {
                "api_endpoint": self.providers.rolm_api_endpoint,
                "api_key": self.providers.rolm_api_key,
                "model_name": self.providers.rolm_model_name,
                "timeout": self.providers.rolm_timeout,
            }
        elif provider_type == ProviderType.GOOGLE_VISION:
            return {
                "credentials_path": self.providers.google_credentials_path,
                "project_id": self.providers.google_project_id,
                "timeout": self.providers.google_timeout,
                "max_retries": 3,
            }
        elif provider_type == ProviderType.AWS_TEXTRACT:
            return {
                "region": self.providers.aws_region,
                "access_key_id": self.providers.aws_access_key_id,
                "secret_access_key": self.providers.aws_secret_access_key,
                "session_token": self.providers.aws_session_token,
                "timeout": self.providers.aws_timeout,
                "max_retries": 3,
            }
        elif provider_type == ProviderType.TESSERACT:
            return {
                "language": self.providers.tesseract_language,
                "dpi": self.providers.tesseract_dpi,
                "timeout": self.providers.tesseract_timeout,
                "tesseract_cmd": self.providers.tesseract_cmd,
            }
        elif provider_type == ProviderType.AZURE_DOCUMENT_INTELLIGENCE:
            return {
                "endpoint": self.providers.azure_doc_intel_endpoint,
                "api_key": self.providers.azure_doc_intel_api_key,
                "timeout": self.providers.azure_doc_intel_timeout,
                "max_retries": 3,
            }
        else:
            return {}
    
    def is_provider_enabled(self, provider_type: ProviderType) -> bool:
        """Check if a provider is enabled."""
        return provider_type.value in self.providers.enabled_providers
    
    def get_provider_priority(self, provider_type: ProviderType) -> int:
        """Get priority for a provider (lower = higher priority)."""
        return self.providers.provider_priorities.get(provider_type.value, 999)
    
    # ========== Configuration Management Methods ==========
    
    async def get_config(self, key: str) -> Any:
        """Get configuration value."""
        # Check runtime config first
        if key in self._runtime_config:
            return self._runtime_config[key]
        
        # Navigate nested settings
        parts = key.split('.')
        value = self
        
        try:
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    raise AttributeError(f"No config key: {part}")
            return value
        except AttributeError:
            raise ConfigurationError(f"Configuration key not found: {key}")
    
    async def set_config(self, key: str, value: Any) -> None:
        """Set runtime configuration value."""
        self._runtime_config[key] = value
    
    async def validate_config(self) -> bool:
        """Validate all configuration."""
        try:
            # Validate API keys for enabled providers
            for provider in self.providers.enabled_providers:
                if provider == "mistral" and not self.providers.mistral_api_key:
                    raise ConfigurationError("Mistral API key is required when Mistral provider is enabled")
                elif provider == "olm_ocr" and not self.providers.olm_api_key:
                    raise ConfigurationError("OLM API key is required when OLM provider is enabled")
                elif provider == "rolm_ocr" and not self.providers.rolm_api_key:
                    raise ConfigurationError("RoLM API key is required when RoLM provider is enabled")
                elif provider == "google_vision" and not self.providers.google_credentials_path:
                    raise ConfigurationError("Google credentials path is required when Google Vision provider is enabled")
                elif provider == "aws_textract" and not self.providers.aws_access_key_id:
                    raise ConfigurationError("AWS credentials are required when AWS Textract provider is enabled")
                elif provider == "azure_document_intelligence" and not self.providers.azure_doc_intel_api_key:
                    raise ConfigurationError("Azure API key is required when Azure Document Intelligence provider is enabled")
            
            # Validate file paths
            if self.files.temp_dir and not self.files.temp_dir.exists():
                self.files.temp_dir.mkdir(parents=True, exist_ok=True)
            
            if self.files.upload_dir and not self.files.upload_dir.exists():
                self.files.upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate web directories
            if not self.web.static_dir.exists():
                raise ConfigurationError(f"Static directory not found: {self.web.static_dir}")
            
            if not self.web.template_dir.exists():
                raise ConfigurationError(f"Template directory not found: {self.web.template_dir}")
            
            # Validate settings using built-in validators
            self._validate_all_settings()
            
            return True
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    def load_from_file(self, config_file: Path) -> None:
        """Load configuration from JSON file."""
        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            self._runtime_config.update(config_data)
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file: {e}")
    
    def save_to_file(self, config_file: Path) -> None:
        """Save runtime configuration to JSON file."""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self._runtime_config, f, indent=2)
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration file: {e}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the runtime environment."""
        return {
            "environment": self.environment,
            "python_version": os.sys.version,
            "platform": os.name,
            "working_directory": str(Path.cwd()),
            "config_sources": self._get_config_sources(),
        }
    
    def _get_config_sources(self) -> Dict[str, str]:
        """Get information about configuration sources."""
        sources = {}
        
        # Check for .env files
        env_files = ['.env', '.env.local', '.env.development', '.env.production']
        for env_file in env_files:
            if Path(env_file).exists():
                sources[env_file] = "environment file"
        
        # Check for environment variables
        ocular_env_vars = [key for key in os.environ.keys() if key.startswith('OCULAR_')]
        if ocular_env_vars:
            sources["environment_variables"] = f"{len(ocular_env_vars)} variables"
        
        # Runtime config
        if self._runtime_config:
            sources["runtime_config"] = f"{len(self._runtime_config)} keys"
        
        return sources
    
    def reset_runtime_config(self) -> None:
        """Reset runtime configuration to defaults."""
        self._runtime_config.clear()
    
    def merge_config(self, config: Dict[str, Any]) -> None:
        """Merge additional configuration."""
        self._runtime_config.update(config)
    
    def export_config(self) -> Dict[str, Any]:
        """Export all configuration as dictionary."""
        return {
            "settings": self.dict(),
            "runtime_config": self._runtime_config,
            "environment_info": self.get_environment_info(),
        }
    
    # ========== Validation Methods ==========
    
    def _validate_all_settings(self) -> None:
        """Validate all settings using built-in validators."""
        # Validate API keys for enabled providers
        for provider in self.providers.enabled_providers:
            api_key = getattr(self.providers, f"{provider.replace('_', '_')}_api_key", None)
            if api_key:
                self.validate_api_key(api_key, provider)
        
        # Validate file extensions
        if self.files.allowed_extensions:
            self.validate_file_extensions(self.files.allowed_extensions)
        
        # Validate CORS origins
        if self.web.cors_origins:
            self.validate_cors_origins(self.web.cors_origins)
        
        # Validate providers
        self.validate_provider_list(self.providers.enabled_providers)
        
        # Validate processing strategy
        self.validate_processing_strategy(self.processing.default_strategy)
        
        # Validate log level
        self.validate_log_level(self.logging.level)
        
        # Validate environment
        self.validate_environment(self.environment)
    
    @staticmethod
    def validate_api_key(api_key: Optional[str], provider_name: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        
        # Basic validation - ensure it's not empty and has reasonable length
        if len(api_key.strip()) < 10:
            raise ValidationError(
                f"API key for {provider_name} appears to be too short",
                field="api_key",
                value=api_key[:10] + "..."
            )
        
        return True
    
    @staticmethod
    def validate_file_path(file_path: Any, field_name: str, must_exist: bool = True) -> Path:
        """Validate file path."""
        if not file_path:
            raise ValidationError(f"{field_name} is required", field=field_name)
        
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise ValidationError(
                f"{field_name} path does not exist: {path}",
                field=field_name,
                value=str(path)
            )
        
        return path
    
    @staticmethod
    def validate_directory_path(dir_path: Any, field_name: str, create_if_missing: bool = False) -> Path:
        """Validate directory path."""
        if not dir_path:
            raise ValidationError(f"{field_name} is required", field=field_name)
        
        path = Path(dir_path)
        
        if not path.exists():
            if create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValidationError(
                        f"Cannot create {field_name} directory: {e}",
                        field=field_name,
                        value=str(path)
                    )
            else:
                raise ValidationError(
                    f"{field_name} directory does not exist: {path}",
                    field=field_name,
                    value=str(path)
                )
        
        if not path.is_dir():
            raise ValidationError(
                f"{field_name} is not a directory: {path}",
                field=field_name,
                value=str(path)
            )
        
        return path
    
    @staticmethod
    def validate_file_extensions(extensions: List[str], field_name: str = "extensions") -> List[str]:
        """Validate file extensions."""
        if not extensions:
            raise ValidationError(f"{field_name} cannot be empty", field=field_name)
        
        valid_extensions = []
        extension_pattern = re.compile(r'^\.[a-z0-9]+$', re.IGNORECASE)
        
        for ext in extensions:
            ext = ext.strip().lower()
            if not ext.startswith('.'):
                ext = '.' + ext
            
            if not extension_pattern.match(ext):
                raise ValidationError(
                    f"Invalid file extension format: {ext}",
                    field=field_name,
                    value=ext
                )
            
            valid_extensions.append(ext)
        
        return valid_extensions
    
    @staticmethod
    def validate_provider_list(providers: List[str], field_name: str = "providers") -> List[str]:
        """Validate list of provider names."""
        if not providers:
            raise ValidationError(f"{field_name} cannot be empty", field=field_name)
        
        valid_providers = {'mistral', 'olm_ocr', 'rolm_ocr', 'google_vision', 'aws_textract', 'tesseract', 'azure_document_intelligence'}
        
        for provider in providers:
            if provider not in valid_providers:
                raise ValidationError(
                    f"Unknown provider: {provider}. Valid providers: {valid_providers}",
                    field=field_name,
                    value=provider
                )
        
        return providers
    
    @staticmethod
    def validate_processing_strategy(strategy: str, field_name: str = "strategy") -> str:
        """Validate processing strategy."""
        valid_strategies = {'single', 'fallback', 'ensemble', 'best'}
        
        if strategy not in valid_strategies:
            raise ValidationError(
                f"Invalid strategy: {strategy}. Valid strategies: {valid_strategies}",
                field=field_name,
                value=strategy
            )
        
        return strategy
    
    @staticmethod
    def validate_log_level(level: str, field_name: str = "log_level") -> str:
        """Validate log level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        level = level.upper()
        
        if level not in valid_levels:
            raise ValidationError(
                f"Invalid log level: {level}. Valid levels: {valid_levels}",
                field=field_name,
                value=level
            )
        
        return level
    
    @staticmethod
    def validate_environment(env: str, field_name: str = "environment") -> str:
        """Validate environment name."""
        valid_environments = {'development', 'production', 'testing'}
        
        if env not in valid_environments:
            raise ValidationError(
                f"Invalid environment: {env}. Valid environments: {valid_environments}",
                field=field_name,
                value=env
            )
        
        return env
    
    @staticmethod
    def validate_cors_origins(origins: List[str], field_name: str = "cors_origins") -> List[str]:
        """Validate CORS origins."""
        if not origins:
            return ["*"]  # Default to allow all
        
        url_pattern = re.compile(
            r'^(https?://)?'  # optional protocol
            r'(\*\.)?'        # optional wildcard subdomain
            r'([a-zA-Z0-9-]+\.)*'  # optional subdomains
            r'[a-zA-Z0-9-]+'  # domain
            r'(\.[a-zA-Z]{2,})?'   # optional TLD
            r'(:[0-9]{1,5})?$'     # optional port
        )
        
        valid_origins = []
        for origin in origins:
            origin = origin.strip()
            
            if origin == "*":
                valid_origins.append(origin)
                continue
            
            if not url_pattern.match(origin):
                raise ValidationError(
                    f"Invalid CORS origin format: {origin}",
                    field=field_name,
                    value=origin
                )
            
            valid_origins.append(origin)
        
        return valid_origins
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"