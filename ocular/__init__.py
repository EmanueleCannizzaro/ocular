"""
Ocular: A Python package for OCR and data extraction using Mistral AI.
"""

# Load environment variables when the package is imported
import os
from pathlib import Path

from ocular.config import OcularConfig
from ocular.core.enums import ProcessingStrategy, ProviderType
from ocular.deployment.mistralocr import MistralOCRClient
from ocular.exceptions import OcularError, OCRError, ConfigurationError
from ocular.processor import DocumentProcessor
from ocular.providers import ProviderFactory
from ocular.pydantic_client import PydanticOCRClient
from ocular.unified_processor import UnifiedDocumentProcessor


# Backward compatibility alias
OCRProvider = ProviderType


def _load_env_on_import():
    """Load environment variables on package import."""
    try:
        from dotenv import load_dotenv
        
        # Look for .env file in current directory and parent directories
        current_dir = Path.cwd()
        env_paths = [
            current_dir / '.env',
            current_dir.parent / '.env',
            Path(__file__).parent.parent / '.env'
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                break
        else:
            # Try loading from default locations
            load_dotenv()
            
    except ImportError:
        # dotenv not available, skip
        pass

# Load environment variables
_load_env_on_import()

__version__ = "0.1.0"
__all__ = [
    "MistralOCRClient",
    "DocumentProcessor",
    "PydanticOCRClient", 
    "UnifiedDocumentProcessor",
    "ProcessingStrategy",
    "ProviderType",
    "OCRProvider",  # Backward compatibility
    "ProviderFactory",
    "OcularConfig",
    "OcularError",
    "OCRError",
    "ConfigurationError",
]