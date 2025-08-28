"""
Factory for creating OCR providers.
"""

from typing import Dict, Any, List, Optional, Type
import logging

from ..core.interfaces import OCRProvider as IOCRProvider
from ..core.enums import ProviderType
from ..core.exceptions import ConfigurationError, ProviderError
from .base import BaseOCRProvider
from .mistral import MistralProvider
from .olm import OLMProvider
from .rolm import RoLMProvider
from .google_vision import GoogleVisionProvider
from .aws_textract import AWSTextractProvider
from .tesseract import TesseractProvider
from .azure_document_intelligence import AzureDocumentIntelligenceProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating and managing OCR providers."""
    
    _provider_registry: Dict[ProviderType, Type[BaseOCRProvider]] = {
        ProviderType.MISTRAL: MistralProvider,
        ProviderType.OLM_OCR: OLMProvider,
        ProviderType.ROLM_OCR: RoLMProvider,
        ProviderType.GOOGLE_VISION: GoogleVisionProvider,
        ProviderType.AWS_TEXTRACT: AWSTextractProvider,
        ProviderType.TESSERACT: TesseractProvider,
        ProviderType.AZURE_DOCUMENT_INTELLIGENCE: AzureDocumentIntelligenceProvider,
    }
    
    @classmethod
    def create_provider(
        cls, 
        provider_type: ProviderType, 
        config: Dict[str, Any]
    ) -> BaseOCRProvider:
        """Create an OCR provider instance."""
        
        if provider_type not in cls._provider_registry:
            raise ProviderError(
                f"Unknown provider type: {provider_type}",
                provider_type=provider_type.value
            )
        
        provider_class = cls._provider_registry[provider_type]
        
        try:
            return provider_class(config)
        except Exception as e:
            raise ProviderError(
                f"Failed to create provider {provider_type.value}: {str(e)}",
                provider_type=provider_type.value
            ) from e
    
    @classmethod
    async def create_and_initialize_provider(
        cls,
        provider_type: ProviderType,
        config: Dict[str, Any]
    ) -> BaseOCRProvider:
        """Create and initialize an OCR provider."""
        
        provider = cls.create_provider(provider_type, config)
        
        try:
            await provider.initialize()
            return provider
        except Exception as e:
            await provider.cleanup()
            raise ProviderError(
                f"Failed to initialize provider {provider_type.value}: {str(e)}",
                provider_type=provider_type.value
            ) from e
    
    @classmethod
    async def get_available_providers(
        cls, 
        configs: Dict[ProviderType, Dict[str, Any]]
    ) -> List[ProviderType]:
        """Get list of available providers based on configuration."""
        available = []
        
        for provider_type in ProviderType:
            if provider_type not in configs:
                continue
            
            try:
                provider = cls.create_provider(provider_type, configs[provider_type])
                if await provider.is_available():
                    available.append(provider_type)
                    logger.info(f"Provider {provider_type.value} is available")
                else:
                    logger.warning(f"Provider {provider_type.value} is not available")
                
                await provider.cleanup()
                
            except Exception as e:
                logger.error(f"Error checking provider {provider_type.value}: {e}")
                continue
        
        return available
    
    @classmethod
    async def validate_provider_configs(
        cls,
        configs: Dict[ProviderType, Dict[str, Any]]
    ) -> Dict[ProviderType, bool]:
        """Validate configuration for all providers."""
        validation_results = {}
        
        for provider_type, config in configs.items():
            try:
                provider = cls.create_provider(provider_type, config)
                validation_results[provider_type] = provider.validate_config()
                await provider.cleanup()
                
            except Exception as e:
                logger.error(f"Config validation failed for {provider_type.value}: {e}")
                validation_results[provider_type] = False
        
        return validation_results
    
    @classmethod
    async def health_check_all(
        cls,
        configs: Dict[ProviderType, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all configured providers."""
        health_results = {}
        
        for provider_type, config in configs.items():
            try:
                provider = cls.create_provider(provider_type, config)
                health_results[provider_type.value] = await provider.health_check()
                await provider.cleanup()
                
            except Exception as e:
                health_results[provider_type.value] = {
                    "provider": provider_type.value,
                    "available": False,
                    "error": str(e)
                }
        
        return health_results
    
    @classmethod
    def register_provider(
        cls,
        provider_type: ProviderType,
        provider_class: Type[BaseOCRProvider]
    ) -> None:
        """Register a custom provider."""
        
        if not issubclass(provider_class, BaseOCRProvider):
            raise ProviderError(
                f"Provider class must inherit from BaseOCRProvider",
                provider_type=provider_type.value
            )
        
        cls._provider_registry[provider_type] = provider_class
        logger.info(f"Registered custom provider: {provider_type.value}")
    
    @classmethod
    def unregister_provider(cls, provider_type: ProviderType) -> None:
        """Unregister a provider."""
        
        if provider_type in cls._provider_registry:
            del cls._provider_registry[provider_type]
            logger.info(f"Unregistered provider: {provider_type.value}")
    
    @classmethod
    def get_registered_providers(cls) -> List[ProviderType]:
        """Get list of registered provider types."""
        return list(cls._provider_registry.keys())
    
    @classmethod
    def get_provider_info(cls, provider_type: ProviderType) -> Dict[str, Any]:
        """Get information about a provider."""
        
        if provider_type not in cls._provider_registry:
            raise ProviderError(
                f"Unknown provider type: {provider_type}",
                provider_type=provider_type.value
            )
        
        provider_class = cls._provider_registry[provider_type]
        
        # Create temporary instance to get info (without initializing)
        temp_config = {}
        if provider_type == ProviderType.MISTRAL:
            temp_config = {"api_key": "temp"}
        elif provider_type in [ProviderType.GOOGLE_VISION, ProviderType.AWS_TEXTRACT, ProviderType.AZURE_DOCUMENT_INTELLIGENCE]:
            temp_config = {"api_key": "temp", "endpoint": "temp"}
        elif provider_type == ProviderType.TESSERACT:
            temp_config = {"language": "eng"}
        else:
            temp_config = {"api_key": "temp", "api_endpoint": "temp"}
            
        temp_provider = provider_class(temp_config)
        
        return {
            "type": provider_type.value,
            "name": temp_provider.provider_name,
            "class": provider_class.__name__,
            "module": provider_class.__module__,
            "supported_formats": temp_provider.get_supported_formats(),
        }
    
    @classmethod
    def create_provider_from_name(
        cls,
        provider_name: str,
        config: Dict[str, Any]
    ) -> BaseOCRProvider:
        """Create provider from string name."""
        
        # Map string names to enum values
        name_to_type = {
            "mistral": ProviderType.MISTRAL,
            "mistral_ai": ProviderType.MISTRAL,
            "olm": ProviderType.OLM_OCR,
            "olm_ocr": ProviderType.OLM_OCR,
            "rolm": ProviderType.ROLM_OCR,
            "rolm_ocr": ProviderType.ROLM_OCR,
            "google_vision": ProviderType.GOOGLE_VISION,
            "google": ProviderType.GOOGLE_VISION,
            "aws_textract": ProviderType.AWS_TEXTRACT,
            "textract": ProviderType.AWS_TEXTRACT,
            "tesseract": ProviderType.TESSERACT,
            "azure_document_intelligence": ProviderType.AZURE_DOCUMENT_INTELLIGENCE,
            "azure_doc_intel": ProviderType.AZURE_DOCUMENT_INTELLIGENCE,
            "azure": ProviderType.AZURE_DOCUMENT_INTELLIGENCE,
        }
        
        provider_name_lower = provider_name.lower()
        
        if provider_name_lower not in name_to_type:
            raise ProviderError(
                f"Unknown provider name: {provider_name}. "
                f"Available: {list(name_to_type.keys())}",
                provider_type=provider_name
            )
        
        provider_type = name_to_type[provider_name_lower]
        return cls.create_provider(provider_type, config)


class ProviderManager:
    """Manages lifecycle of OCR providers."""
    
    def __init__(self):
        self._active_providers: Dict[ProviderType, BaseOCRProvider] = {}
        self._factory = ProviderFactory()
    
    async def get_provider(
        self,
        provider_type: ProviderType,
        config: Dict[str, Any]
    ) -> BaseOCRProvider:
        """Get or create a provider instance."""
        
        if provider_type not in self._active_providers:
            provider = await self._factory.create_and_initialize_provider(
                provider_type, config
            )
            self._active_providers[provider_type] = provider
        
        return self._active_providers[provider_type]
    
    async def cleanup_provider(self, provider_type: ProviderType) -> None:
        """Clean up a specific provider."""
        
        if provider_type in self._active_providers:
            await self._active_providers[provider_type].cleanup()
            del self._active_providers[provider_type]
    
    async def cleanup_all(self) -> None:
        """Clean up all active providers."""
        
        for provider in self._active_providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up provider: {e}")
        
        self._active_providers.clear()
    
    def get_active_providers(self) -> List[ProviderType]:
        """Get list of currently active providers."""
        return list(self._active_providers.keys())
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup_all()