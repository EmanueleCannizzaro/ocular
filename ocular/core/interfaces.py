"""
Core interfaces for Ocular OCR system.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, AsyncGenerator
from pathlib import Path

from .models import OCRResult, ProcessingResult, ProcessingRequest
from .enums import ProviderType


class OCRProvider(ABC):
    """Abstract base class for OCR providers."""
    
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return human-readable provider name."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (lazy loading)."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available and properly configured."""
        pass
    
    @abstractmethod
    async def extract_text(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text from a document."""
        pass
    
    @abstractmethod
    async def extract_structured_data(
        self,
        file_path: Path,
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract structured data from a document."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate if file is supported by this provider."""
        return file_path.suffix.lower() in self.get_supported_formats()


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    @abstractmethod
    async def process_document(
        self, 
        request: ProcessingRequest
    ) -> ProcessingResult:
        """Process a single document."""
        pass
    
    @abstractmethod
    async def process_batch(
        self, 
        requests: List[ProcessingRequest]
    ) -> List[ProcessingResult]:
        """Process multiple documents."""
        pass
    
    @abstractmethod
    async def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        pass


class ResultProcessor(ABC):
    """Abstract base class for processing and analyzing OCR results."""
    
    @abstractmethod
    async def combine_results(
        self, 
        results: List[OCRResult]
    ) -> OCRResult:
        """Combine multiple OCR results into a single result."""
        pass
    
    @abstractmethod
    async def select_best_result(
        self, 
        results: List[OCRResult]
    ) -> OCRResult:
        """Select the best result from multiple options."""
        pass
    
    @abstractmethod
    async def validate_result(
        self, 
        result: OCRResult,
        validation_rules: Dict[str, Any]
    ) -> bool:
        """Validate an OCR result against rules."""
        pass


class FileHandler(ABC):
    """Abstract base class for file handling operations."""
    
    @abstractmethod
    async def read_file(self, file_path: Path) -> bytes:
        """Read file contents."""
        pass
    
    @abstractmethod
    async def validate_file(self, file_path: Path) -> bool:
        """Validate file format and size."""
        pass
    
    @abstractmethod
    async def preprocess_file(self, file_path: Path) -> Path:
        """Preprocess file if needed (e.g., image enhancement)."""
        pass


class ConfigurationProvider(ABC):
    """Abstract base class for configuration providers."""
    
    @abstractmethod
    async def get_config(self, key: str) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    async def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass
    
    @abstractmethod
    async def validate_config(self) -> bool:
        """Validate configuration."""
        pass


class EventHandler(ABC):
    """Abstract base class for handling processing events."""
    
    @abstractmethod
    async def on_processing_started(self, request: ProcessingRequest) -> None:
        """Handle processing started event."""
        pass
    
    @abstractmethod
    async def on_processing_completed(self, result: ProcessingResult) -> None:
        """Handle processing completed event."""
        pass
    
    @abstractmethod
    async def on_processing_failed(self, request: ProcessingRequest, error: Exception) -> None:
        """Handle processing failed event."""
        pass
    
    @abstractmethod
    async def on_provider_result(self, provider: str, result: OCRResult) -> None:
        """Handle individual provider result."""
        pass


class CacheProvider(ABC):
    """Abstract base class for caching OCR results."""
    
    @abstractmethod
    async def get_cached_result(self, key: str) -> Optional[OCRResult]:
        """Get cached OCR result."""
        pass
    
    @abstractmethod
    async def cache_result(self, key: str, result: OCRResult, ttl: Optional[int] = None) -> None:
        """Cache OCR result."""
        pass
    
    @abstractmethod
    async def invalidate_cache(self, key: str) -> None:
        """Invalidate cached result."""
        pass
    
    @abstractmethod
    async def clear_cache(self) -> None:
        """Clear all cached results."""
        pass