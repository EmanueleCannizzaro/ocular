"""
Unified document processor supporting multiple OCR providers.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import time

from .providers import BaseOCRProvider, ProviderFactory
from .providers.settings import OcularSettings
from .core.models import OCRResult
from .core.enums import ProviderType, ProcessingStrategy
from .config import OcularConfig
from .exceptions import DocumentProcessingError, ConfigurationError


class UnifiedProcessingResult:
    """Result from unified processing with multiple providers."""
    
    def __init__(
        self,
        file_path: Path,
        results: Dict[str, OCRResult],
        strategy: ProcessingStrategy,
        primary_result: Optional[OCRResult] = None,
        total_processing_time: Optional[float] = None
    ):
        self.file_path = file_path
        self.results = results
        self.strategy = strategy
        self.primary_result = primary_result or self._select_best_result()
        self.total_processing_time = total_processing_time
    
    def _select_best_result(self) -> Optional[OCRResult]:
        """Select the best result based on confidence scores."""
        if not self.results:
            return None
        
        best_result = None
        best_confidence = 0.0
        
        for result in self.results.values():
            if result.confidence and result.confidence > best_confidence:
                best_confidence = result.confidence
                best_result = result
        
        # If no confidence scores, return first result
        if best_result is None:
            best_result = next(iter(self.results.values()))
        
        return best_result
    
    def get_text(self) -> str:
        """Get text from primary result."""
        return self.primary_result.text if self.primary_result else ""
    
    def get_all_results(self) -> Dict[str, str]:
        """Get text from all providers."""
        return {provider: result.text for provider, result in self.results.items()}
    
    def get_consensus_text(self, min_agreement: float = 0.7) -> str:
        """Get consensus text from multiple results."""
        if len(self.results) < 2:
            return self.get_text()
        
        # Simple consensus: return text that appears most frequently
        # In production, you might use more sophisticated text comparison
        texts = [result.text for result in self.results.values()]
        
        # For now, return the longest common substring or most confident result
        return self.primary_result.text if self.primary_result else ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": str(self.file_path),
            "strategy": self.strategy.value,
            "primary_text": self.get_text(),
            "primary_provider": self.primary_result.provider if self.primary_result else None,
            "primary_confidence": self.primary_result.confidence if self.primary_result else None,
            "all_results": {
                provider: {
                    "text": result.text,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time
                }
                for provider, result in self.results.items()
            },
            "total_processing_time": self.total_processing_time
        }


class UnifiedDocumentProcessor:
    """Unified document processor supporting multiple OCR providers."""
    
    def __init__(self, config: Optional[Union[OcularConfig, OcularSettings]] = None):
        """Initialize the unified processor."""
        # Use OcularSettings as default since it's more comprehensive
        if config is None:
            self.config = OcularSettings()
        else:
            self.config = config
        self.providers: Dict[ProviderType, BaseOCRProvider] = {}
        self._configs = self._prepare_provider_configs()
    
    def _prepare_provider_configs(self) -> Dict[ProviderType, Dict[str, Any]]:
        """Prepare configuration for each provider type."""
        configs = {}
        
        # Use get_provider_config method from OcularSettings
        for provider_type in ProviderType:
            try:
                provider_config = self.config.get_provider_config(provider_type)
                if provider_config and self._is_provider_config_valid(provider_type, provider_config):
                    configs[provider_type] = provider_config
            except Exception as e:
                print(f"Warning: Could not get config for {provider_type.value}: {e}")
        
        return configs
    
    def _is_provider_config_valid(self, provider_type: ProviderType, config: Dict[str, Any]) -> bool:
        """Check if provider configuration is valid."""
        if provider_type == ProviderType.MISTRAL:
            return bool(config.get("api_key"))
        elif provider_type == ProviderType.OLM_OCR:
            return bool(config.get("api_endpoint") and config.get("api_key"))
        elif provider_type == ProviderType.ROLM_OCR:
            return bool(config.get("api_endpoint") and config.get("api_key"))
        elif provider_type == ProviderType.GOOGLE_VISION:
            # Google Vision can work with default credentials or explicit credentials
            return True  # Validation happens in provider initialization
        elif provider_type == ProviderType.AWS_TEXTRACT:
            # AWS can work with default credentials or explicit credentials
            return True  # Validation happens in provider initialization
        elif provider_type == ProviderType.TESSERACT:
            # Tesseract just needs basic config
            return bool(config.get("language"))
        elif provider_type == ProviderType.AZURE_DOCUMENT_INTELLIGENCE:
            return bool(config.get("endpoint") and config.get("api_key"))
        return False
    
    async def _ensure_providers_initialized(self):
        """Ensure providers are initialized (lazy loading)."""
        if not self.providers:
            # Get available providers
            available_providers = await ProviderFactory.get_available_providers(self._configs)
            
            for provider_type in available_providers:
                if provider_type in self._configs:
                    try:
                        provider = ProviderFactory.create_provider(provider_type, self._configs[provider_type])
                        self.providers[provider_type] = provider
                    except Exception as e:
                        print(f"Warning: Could not initialize {provider_type.value}: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [provider.value for provider in self.providers.keys()]
    
    async def process_single_provider(
        self,
        file_path: Union[str, Path],
        provider: ProviderType,
        prompt: Optional[str] = None
    ) -> UnifiedProcessingResult:
        """Process document with a single provider."""
        file_path = Path(file_path)
        start_time = time.time()
        
        await self._ensure_providers_initialized()
        
        if provider not in self.providers:
            raise ConfigurationError(f"Provider {provider.value} not available")
        
        try:
            result = await self.providers[provider].extract_text(file_path, prompt)
            total_time = time.time() - start_time
            
            return UnifiedProcessingResult(
                file_path=file_path,
                results={provider.value: result},
                strategy=ProcessingStrategy.SINGLE,
                primary_result=result,
                total_processing_time=total_time
            )
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to process with {provider.value}: {str(e)}")
    
    async def process_with_fallback(
        self,
        file_path: Union[str, Path],
        providers: Optional[List[ProviderType]] = None,
        prompt: Optional[str] = None
    ) -> UnifiedProcessingResult:
        """Process document with fallback strategy."""
        file_path = Path(file_path)
        start_time = time.time()
        
        await self._ensure_providers_initialized()
        
        if providers is None:
            providers = list(self.providers.keys())
        
        results = {}
        last_error = None
        
        for provider in providers:
            if provider not in self.providers:
                continue
                
            try:
                result = await self.providers[provider].extract_text(file_path, prompt)
                results[provider.value] = result
                
                total_time = time.time() - start_time
                return UnifiedProcessingResult(
                    file_path=file_path,
                    results=results,
                    strategy=ProcessingStrategy.FALLBACK,
                    primary_result=result,
                    total_processing_time=total_time
                )
                
            except Exception as e:
                last_error = e
                continue
        
        if not results:
            raise DocumentProcessingError(f"All providers failed. Last error: {last_error}")
        
        # This shouldn't happen, but just in case
        total_time = time.time() - start_time
        return UnifiedProcessingResult(
            file_path=file_path,
            results=results,
            strategy=ProcessingStrategy.FALLBACK,
            total_processing_time=total_time
        )
    
    async def process_with_ensemble(
        self,
        file_path: Union[str, Path],
        providers: Optional[List[ProviderType]] = None,
        prompt: Optional[str] = None,
        max_concurrent: int = 3
    ) -> UnifiedProcessingResult:
        """Process document with multiple providers concurrently."""
        file_path = Path(file_path)
        start_time = time.time()
        
        await self._ensure_providers_initialized()
        
        if providers is None:
            providers = list(self.providers.keys())
        
        # Limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_provider(provider: ProviderType):
            if provider not in self.providers:
                return None
            
            async with semaphore:
                try:
                    return await self.providers[provider].extract_text(file_path, prompt)
                except Exception as e:
                    print(f"Warning: {provider.value} failed: {e}")
                    return None
        
        # Run all providers concurrently
        tasks = [process_with_provider(provider) for provider in providers]
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        results = {}
        for provider, result in zip(providers, provider_results):
            if result and not isinstance(result, Exception):
                results[provider.value] = result
        
        if not results:
            raise DocumentProcessingError("All providers failed in ensemble processing")
        
        total_time = time.time() - start_time
        return UnifiedProcessingResult(
            file_path=file_path,
            results=results,
            strategy=ProcessingStrategy.ENSEMBLE,
            total_processing_time=total_time
        )
    
    async def process_document(
        self,
        file_path: Union[str, Path],
        strategy: ProcessingStrategy = ProcessingStrategy.FALLBACK,
        providers: Optional[List[ProviderType]] = None,
        prompt: Optional[str] = None,
        **kwargs
    ) -> UnifiedProcessingResult:
        """Process document using specified strategy."""
        
        if strategy == ProcessingStrategy.SINGLE:
            provider = providers[0] if providers else list(self.providers.keys())[0]
            return await self.process_single_provider(file_path, provider, prompt)
        
        elif strategy == ProcessingStrategy.FALLBACK:
            return await self.process_with_fallback(file_path, providers, prompt)
        
        elif strategy == ProcessingStrategy.ENSEMBLE:
            max_concurrent = kwargs.get('max_concurrent', 3)
            return await self.process_with_ensemble(file_path, providers, prompt, max_concurrent)
        
        elif strategy == ProcessingStrategy.BEST:
            # Use ensemble then select best result
            ensemble_result = await self.process_with_ensemble(file_path, providers, prompt)
            ensemble_result.strategy = ProcessingStrategy.BEST
            return ensemble_result
        
        else:
            raise ValueError(f"Unknown processing strategy: {strategy}")
    
    async def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        strategy: ProcessingStrategy = ProcessingStrategy.FALLBACK,
        providers: Optional[List[ProviderType]] = None,
        prompt: Optional[str] = None,
        max_concurrent: int = 3,
        **kwargs
    ) -> List[UnifiedProcessingResult]:
        """Process multiple documents."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_document(
                    file_path, strategy, providers, prompt, **kwargs
                )
        
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about available providers."""
        stats = {}
        
        for provider_type, provider in self.providers.items():
            stats[provider_type.value] = {
                "name": provider.provider_name,
                "available": provider.is_available(),
                "type": "API" if provider_type == ProviderType.MISTRAL else "Local"
            }
        
        return stats