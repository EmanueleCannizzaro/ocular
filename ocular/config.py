"""
Configuration management for Ocular.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .exceptions import ConfigurationError
from .providers.settings import (
    OcularSettings,
    ProviderSettings,
    WebSettings,
    FileSettings,
    ProcessingSettings,
    LoggingSettings
)


class OcularConfig(BaseModel):
    """Configuration class for Ocular OCR client."""
    
    mistral_api_key: str = Field(..., description="Mistral AI API key")
    mistral_model: str = Field(default="pixtral-12b-2409", description="Mistral model to use for OCR")
    max_file_size_mb: int = Field(default=10, description="Maximum file size in MB")
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "OcularConfig":
        """Create configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
            
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "MISTRAL_API_KEY environment variable is required. "
                "Set it in your environment or .env file."
            )
            
        return cls(
            mistral_api_key=api_key,
            mistral_model=os.getenv("MISTRAL_MODEL", "pixtral-12b-2409"),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
        )