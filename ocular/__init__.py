"""
Ocular: A Python package for OCR and data extraction using Mistral AI.
"""

# Load environment variables when the package is imported
import os
from pathlib import Path

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

from .client import MistralOCRClient
from .processor import DocumentProcessor
from .pydantic_client import PydanticOCRClient
from .unified_processor import UnifiedDocumentProcessor
from .core.enums import ProcessingStrategy, ProviderType
# Backward compatibility alias
OCRProvider = ProviderType
from .providers import ProviderFactory
from .config import OcularConfig
from .exceptions import OcularError, OCRError, ConfigurationError

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