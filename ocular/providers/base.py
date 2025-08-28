"""
Base OCR provider implementation with robust error handling and logging.
"""

import time
import asyncio
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path
from abc import abstractmethod

from ..core.interfaces import OCRProvider
from ..core.models import OCRResult
from ..core.enums import ProviderType
from ..core.exceptions import (
    OCRError, ValidationError, TimeoutError, 
    RateLimitError, AuthenticationError, ResourceExhaustedError
)
from ..core.logging import get_logger, log_async_function_call
from ..core.validation import file_validator, config_validator


class BaseOCRProvider(OCRProvider):
    """Base implementation for OCR providers with robust error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        # Validate configuration
        self.config = config_validator.validate_config(config)
        
        self._initialized = False
        self._lock = asyncio.Lock()
        self.logger = get_logger(f"ocular.provider.{self.provider_type.value}")
        
        # Track provider state
        self._health_status = "unknown"
        self._last_health_check = None
        self._error_count = 0
        self._success_count = 0
        
        self.logger.info(
            f"Initialized {self.provider_name} provider",
            provider=self.provider_name,
            config_keys=list(config.keys())
        )
    
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
    
    @log_async_function_call()
    async def initialize(self) -> None:
        """Initialize the provider (lazy loading)."""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            self.logger.info(f"Initializing {self.provider_name} provider")
            
            try:
                await self._do_initialize()
                self._initialized = True
                self._health_status = "healthy"
                
                self.logger.info(
                    f"Successfully initialized {self.provider_name} provider",
                    provider=self.provider_name
                )
                
            except Exception as e:
                self._health_status = "failed"
                self.logger.error(
                    f"Failed to initialize {self.provider_name} provider: {e}",
                    provider=self.provider_name,
                    exc_info=True
                )
                raise
    
    @abstractmethod
    async def _do_initialize(self) -> None:
        """Perform actual initialization (implemented by subclasses)."""
        pass
    
    @log_async_function_call()
    async def extract_text(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text from a document with comprehensive error handling."""
        request_id = str(uuid.uuid4())
        
        # Validate inputs
        file_validation = file_validator.validate_file(file_path)
        
        self.logger.log_ocr_request(
            str(file_path), 
            self.provider_name,
            request_id=request_id
        )
        
        # Initialize provider if needed
        await self.initialize()
        
        if not self.validate_file(file_path):
            raise ValidationError(
                f"File format not supported by {self.provider_name}",
                field="file_path",
                value=str(file_path)
            )
        
        start_time = time.time()
        
        try:
            result = await self._extract_text_impl(file_path, prompt, **kwargs)
            processing_time = time.time() - start_time
            
            # Ensure result has required fields
            if not isinstance(result, OCRResult):
                result = OCRResult(
                    text=str(result),
                    provider=self.provider_name,
                    processing_time=processing_time
                )
            else:
                result.processing_time = processing_time
                result.provider = self.provider_name
            
            # Update success metrics
            self._success_count += 1
            
            # Log successful completion
            self.logger.log_ocr_result(
                str(file_path),
                self.provider_name,
                success=True,
                processing_time=processing_time,
                confidence=result.confidence,
                text_length=len(result.text) if result.text else 0,
                request_id=request_id
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._error_count += 1
            
            # Log error
            self.logger.log_ocr_result(
                str(file_path),
                self.provider_name,
                success=False,
                processing_time=processing_time,
                request_id=request_id
            )
            
            self.logger.log_error(
                e,
                context={
                    "file_path": str(file_path),
                    "provider": self.provider_name,
                    "processing_time": processing_time
                },
                request_id=request_id
            )
            
            # Convert to appropriate exception type
            if isinstance(e, (ValidationError, OCRError)):
                raise
            else:
                raise OCRError(
                    f"Text extraction failed: {str(e)}",
                    provider=self.provider_name,
                    file_path=str(file_path)
                ) from e
    
    @abstractmethod
    async def _extract_text_impl(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Implementation-specific text extraction."""
        pass
    
    @log_async_function_call()
    async def extract_structured_data(
        self,
        file_path: Path,
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract structured data from a document with robust error handling."""
        request_id = str(uuid.uuid4())
        
        # Validate inputs
        file_validation = file_validator.validate_file(file_path)
        
        self.logger.info(
            f"Starting structured data extraction",
            file_path=str(file_path),
            provider=self.provider_name,
            schema_keys=list(schema.keys()) if schema else [],
            request_id=request_id
        )
        
        await self.initialize()
        
        if not self.validate_file(file_path):
            raise ValidationError(
                f"File format not supported by {self.provider_name}",
                field="file_path", 
                value=str(file_path)
            )
        
        start_time = time.time()
        
        try:
            result = await self._extract_structured_data_impl(file_path, schema, prompt, **kwargs)
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Structured data extraction completed",
                file_path=str(file_path),
                provider=self.provider_name,
                processing_time=processing_time,
                extracted_fields=list(result.keys()) if isinstance(result, dict) else None,
                request_id=request_id
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._error_count += 1
            
            self.logger.log_error(
                e,
                context={
                    "file_path": str(file_path),
                    "provider": self.provider_name,
                    "operation": "structured_data_extraction",
                    "processing_time": processing_time
                },
                request_id=request_id
            )
            
            if isinstance(e, (ValidationError, OCRError)):
                raise
            else:
                raise OCRError(
                    f"Structured data extraction failed: {str(e)}",
                    provider=self.provider_name,
                    file_path=str(file_path)
                ) from e
    
    async def _extract_structured_data_impl(
        self,
        file_path: Path,
        schema: Dict[str, Any], 
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Default implementation using text extraction + parsing."""
        # Default behavior: extract text then try to parse structured data
        text_result = await self._extract_text_impl(file_path, prompt, **kwargs)
        
        # Simple structured data extraction (can be overridden by subclasses)
        return {
            "extracted_text": text_result.text,
            "confidence": text_result.confidence,
            "provider": self.provider_name,
            "schema": schema
        }
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate if file is supported by this provider."""
        if not file_path.exists():
            return False
        
        if not file_path.is_file():
            return False
        
        return file_path.suffix.lower() in self.get_supported_formats()
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    async def cleanup(self) -> None:
        """Clean up provider resources."""
        # Default implementation - override in subclasses if needed
        pass
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    @log_async_function_call()
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_info = {
            "provider": self.provider_name,
            "provider_type": self.provider_type.value,
            "status": self._health_status,
            "initialized": self._initialized,
            "error_count": self._error_count,
            "success_count": self._success_count,
            "last_health_check": None,
            "supported_formats": self.get_supported_formats(),
            "config_valid": False,
            "availability_test": False
        }
        
        try:
            # Validate configuration
            health_info["config_valid"] = self.validate_config()
            
            # Test availability
            health_info["availability_test"] = await self.is_available()
            
            # Update overall status
            if health_info["config_valid"] and health_info["availability_test"]:
                self._health_status = "healthy"
                health_info["status"] = "healthy"
            else:
                self._health_status = "unhealthy"
                health_info["status"] = "unhealthy"
                
            self._last_health_check = time.time()
            health_info["last_health_check"] = self._last_health_check
            
            self.logger.info(
                f"Health check completed for {self.provider_name}",
                status=health_info["status"],
                **{k: v for k, v in health_info.items() if k != 'provider'}
            )
            
        except Exception as e:
            self._health_status = "failed"
            health_info["status"] = "failed"
            health_info["error"] = str(e)
            
            self.logger.error(
                f"Health check failed for {self.provider_name}: {e}",
                provider=self.provider_name,
                exc_info=True
            )
        
        return health_info
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics."""
        total_requests = self._success_count + self._error_count
        success_rate = (self._success_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "provider": self.provider_name,
            "total_requests": total_requests,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate_percent": round(success_rate, 2),
            "health_status": self._health_status,
            "last_health_check": self._last_health_check,
            "initialized": self._initialized
        }
    
    def reset_metrics(self):
        """Reset provider metrics."""
        self._success_count = 0
        self._error_count = 0
        self.logger.info(f"Metrics reset for {self.provider_name}")
    
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        # Basic validation - override in subclasses
        return True
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available."""
        pass
    
    def __str__(self) -> str:
        return f"{self.provider_name} ({self.provider_type.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.provider_name}>"