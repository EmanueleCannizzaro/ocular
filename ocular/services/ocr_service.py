"""
Main OCR service orchestrating the processing pipeline.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..core.interfaces import DocumentProcessor
from ..core.models import ProcessingRequest, ProcessingResult, OCRResult
from ..core.enums import ProviderType, ProcessingStrategy, ProcessingStatus
from ..core.exceptions import OCRError, ValidationError
from ..providers.factory import ProviderManager
from ..providers.settings import OcularSettings
from .document_service import DocumentService
from .validation_service import ValidationService
from .cache_service import CacheService

logger = logging.getLogger(__name__)


class OCRService(DocumentProcessor):
    """Main OCR service coordinating all processing operations."""
    
    def __init__(self, settings: OcularSettings):
        self.settings = settings
        self.provider_manager = ProviderManager()
        self.document_service = DocumentService(settings)
        self.validation_service = ValidationService(settings)
        self.cache_service = CacheService(settings) if settings.processing.enable_caching else None
        
    async def process_document(self, request: ProcessingRequest) -> ProcessingResult:
        """Process a single document with the specified strategy."""
        
        # Create result object
        result = ProcessingResult(
            file_path=request.file_path,
            document_type=self.document_service.detect_document_type(request.file_path),
            strategy=request.strategy,
            status=ProcessingStatus.PENDING
        )
        
        try:
            # Validate request
            await self.validation_service.validate_processing_request(request)
            
            # Check cache if enabled
            if self.cache_service:
                cache_key = self._generate_cache_key(request)
                cached_result = await self.cache_service.get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Using cached result for {request.file_path}")
                    result.primary_result = cached_result
                    result.provider_results[cached_result.provider] = cached_result
                    result.mark_completed()
                    return result
            
            # Update status
            result.status = ProcessingStatus.IN_PROGRESS
            
            # Process based on strategy
            if request.strategy == ProcessingStrategy.SINGLE:
                await self._process_single(request, result)
            elif request.strategy == ProcessingStrategy.FALLBACK:
                await self._process_fallback(request, result)
            elif request.strategy == ProcessingStrategy.ENSEMBLE:
                await self._process_ensemble(request, result)
            elif request.strategy == ProcessingStrategy.BEST:
                await self._process_best(request, result)
            else:
                raise ValidationError(f"Unknown processing strategy: {request.strategy}")
            
            # Cache result if enabled
            if self.cache_service and result.primary_result:
                cache_key = self._generate_cache_key(request)
                await self.cache_service.cache_result(
                    cache_key, 
                    result.primary_result,
                    self.settings.processing.cache_ttl_seconds
                )
            
            result.mark_completed()
            logger.info(f"Successfully processed {request.file_path} with {request.strategy.value} strategy")
            
        except Exception as e:
            error_msg = f"Processing failed for {request.file_path}: {str(e)}"
            logger.error(error_msg)
            result.mark_failed(error_msg)
            raise OCRError(error_msg) from e
        
        return result
    
    async def process_batch(self, requests: List[ProcessingRequest]) -> List[ProcessingResult]:
        """Process multiple documents with concurrency control."""
        
        if not requests:
            return []
        
        # Limit concurrent processing
        semaphore = asyncio.Semaphore(self.settings.processing.max_concurrent_requests)
        
        async def process_with_semaphore(request: ProcessingRequest):
            async with semaphore:
                return await self.process_document(request)
        
        # Process all requests concurrently
        tasks = [process_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_result = ProcessingResult(
                    file_path=requests[i].file_path,
                    document_type=self.document_service.detect_document_type(requests[i].file_path),
                    strategy=requests[i].strategy,
                    status=ProcessingStatus.FAILED
                )
                failed_result.mark_failed(str(result))
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single(self, request: ProcessingRequest, result: ProcessingResult) -> None:
        """Process with a single provider."""
        
        if not request.providers:
            raise ValidationError("No providers specified for single processing")
        
        provider_type = request.providers[0]
        provider_config = self.settings.get_provider_config(provider_type)
        
        provider = await self.provider_manager.get_provider(provider_type, provider_config)
        
        ocr_result = await provider.extract_text(
            request.file_path,
            request.prompt,
            **request.options
        )
        
        result.add_provider_result(provider_type.value, ocr_result)
    
    async def _process_fallback(self, request: ProcessingRequest, result: ProcessingResult) -> None:
        """Process with fallback strategy - try providers until success."""
        
        if not request.providers:
            # Use all enabled providers in priority order
            providers = self._get_enabled_providers_by_priority()
        else:
            providers = request.providers
        
        last_error = None
        
        for provider_type in providers:
            try:
                provider_config = self.settings.get_provider_config(provider_type)
                provider = await self.provider_manager.get_provider(provider_type, provider_config)
                
                ocr_result = await provider.extract_text(
                    request.file_path,
                    request.prompt,
                    **request.options
                )
                
                result.add_provider_result(provider_type.value, ocr_result)
                return  # Success - exit early
                
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_type.value} failed: {e}")
                continue
        
        if not result.provider_results:
            raise OCRError(f"All providers failed. Last error: {last_error}")
    
    async def _process_ensemble(self, request: ProcessingRequest, result: ProcessingResult) -> None:
        """Process with ensemble strategy - use multiple providers concurrently."""
        
        if not request.providers:
            providers = self._get_enabled_providers_by_priority()
        else:
            providers = request.providers
        
        if len(providers) < 2:
            # Fall back to single processing
            await self._process_single(request, result)
            return
        
        # Process with all providers concurrently
        async def process_with_provider(provider_type: ProviderType):
            try:
                provider_config = self.settings.get_provider_config(provider_type)
                provider = await self.provider_manager.get_provider(provider_type, provider_config)
                
                return provider_type, await provider.extract_text(
                    request.file_path,
                    request.prompt,
                    **request.options
                )
            except Exception as e:
                logger.warning(f"Provider {provider_type.value} failed in ensemble: {e}")
                return provider_type, None
        
        # Run providers concurrently
        tasks = [process_with_provider(pt) for pt in providers]
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        for provider_type, ocr_result in provider_results:
            if not isinstance(ocr_result, Exception) and ocr_result is not None:
                result.add_provider_result(provider_type.value, ocr_result)
        
        if not result.provider_results:
            raise OCRError("All providers failed in ensemble processing")
    
    async def _process_best(self, request: ProcessingRequest, result: ProcessingResult) -> None:
        """Process with best strategy - use ensemble then select best result."""
        
        # Use ensemble processing first
        await self._process_ensemble(request, result)
        
        # The best result is automatically selected by ProcessingResult.add_provider_result
        # based on confidence scores, so no additional work needed here
    
    def _get_enabled_providers_by_priority(self) -> List[ProviderType]:
        """Get enabled providers sorted by priority."""
        
        enabled_providers = []
        
        for provider_type in ProviderType:
            if self.settings.is_provider_enabled(provider_type):
                enabled_providers.append((
                    provider_type,
                    self.settings.get_provider_priority(provider_type)
                ))
        
        # Sort by priority (lower number = higher priority)
        enabled_providers.sort(key=lambda x: x[1])
        
        return [pt for pt, _ in enabled_providers]
    
    def _generate_cache_key(self, request: ProcessingRequest) -> str:
        """Generate cache key for request."""
        import hashlib
        
        # Include file path, strategy, providers, and prompt in cache key
        key_data = f"{request.file_path}:{request.strategy.value}:{sorted([p.value for p in request.providers])}:{request.prompt or ''}"
        
        # Add file modification time to invalidate cache when file changes
        try:
            mtime = request.file_path.stat().st_mtime
            key_data += f":{mtime}"
        except:
            pass
        
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        
        available = []
        
        for provider_type in ProviderType:
            if self.settings.is_provider_enabled(provider_type):
                try:
                    provider_config = self.settings.get_provider_config(provider_type)
                    provider = await self.provider_manager.get_provider(provider_type, provider_config)
                    
                    if await provider.is_available():
                        available.append(provider_type.value)
                        
                except Exception as e:
                    logger.warning(f"Provider {provider_type.value} not available: {e}")
        
        return available
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        
        health_info = {
            "service": "OCRService",
            "status": "healthy",
            "providers": {},
            "settings": {
                "environment": self.settings.environment,
                "caching_enabled": self.settings.processing.enable_caching,
                "max_concurrent": self.settings.processing.max_concurrent_requests
            }
        }
        
        # Check each enabled provider
        for provider_type in ProviderType:
            if self.settings.is_provider_enabled(provider_type):
                try:
                    provider_config = self.settings.get_provider_config(provider_type)
                    provider = await self.provider_manager.get_provider(provider_type, provider_config)
                    
                    provider_health = await provider.health_check()
                    health_info["providers"][provider_type.value] = provider_health
                    
                except Exception as e:
                    health_info["providers"][provider_type.value] = {
                        "available": False,
                        "error": str(e)
                    }
        
        # Overall service health
        available_providers = [
            p for p in health_info["providers"].values() 
            if p.get("available", False)
        ]
        
        if not available_providers:
            health_info["status"] = "unhealthy"
            health_info["error"] = "No providers available"
        
        return health_info
    
    async def cleanup(self) -> None:
        """Clean up service resources."""
        
        try:
            await self.provider_manager.cleanup_all()
            
            if self.cache_service:
                await self.cache_service.clear_cache()
                
            logger.info("OCR service cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during OCR service cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()