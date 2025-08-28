"""
Core module containing base interfaces and common functionality.
"""

from .interfaces import OCRProvider, DocumentProcessor, ResultProcessor
from .models import OCRResult, ProcessingRequest, ProcessingResult
from .exceptions import OcularError, OCRError, ConfigurationError, ValidationError
from .enums import ProviderType, ProcessingStrategy, DocumentType

__all__ = [
    "OCRProvider",
    "DocumentProcessor", 
    "ResultProcessor",
    "OCRResult",
    "ProcessingRequest",
    "ProcessingResult",
    "OcularError",
    "OCRError",
    "ConfigurationError",
    "ValidationError",
    "ProviderType",
    "ProcessingStrategy",
    "DocumentType",
]