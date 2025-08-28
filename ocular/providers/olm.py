"""
OLM (Optical Language Model) OCR provider implementation.
API-based provider for remote RunPod deployment.
"""

import base64
import aiohttp
import json
from typing import Optional, Dict, Any
from pathlib import Path

from .base import BaseOCRProvider
from ..core.models import OCRResult
from ..core.enums import ProviderType
from ..core.exceptions import OCRError, APIError, ConfigurationError


class OLMProvider(BaseOCRProvider):
    """OLM OCR provider using remote RunPod API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_endpoint = config.get("api_endpoint", "")
        self.api_key = config.get("api_key", "")
        self.model_name = config.get("model_name", "microsoft/trocr-base-printed")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLM_OCR
    
    @property
    def provider_name(self) -> str:
        return "OLM OCR"
    
    async def _do_initialize(self) -> None:
        """Initialize OLM API connection."""
        if not self.api_endpoint:
            raise ConfigurationError(
                "OLM API endpoint is required",
                config_key="api_endpoint"
            )
        
        if not self.api_key:
            raise ConfigurationError(
                "OLM API key is required", 
                config_key="api_key"
            )
        
        # Test API connectivity
        try:
            await self._test_api_connection()
        except Exception as e:
            raise ConfigurationError(f"Failed to connect to OLM API: {e}")
    
    async def is_available(self) -> bool:
        """Check if OLM API is available."""
        try:
            await self._test_api_connection()
            return True
        except Exception:
            return False
    
    async def _extract_text_impl(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text using OLM API."""
        
        # Encode image
        base64_image = await self._encode_image(file_path)
        
        # Prepare request payload
        payload = {
            "image": base64_image,
            "model_name": self.model_name,
            "prompt": prompt or "Extract all text from this image accurately",
            "options": kwargs
        }
        
        # Make API call with retries
        response_data = await self._make_api_call("/extract_text", payload)
        
        return OCRResult(
            text=response_data.get("text", ""),
            confidence=response_data.get("confidence", 0.85),
            language=response_data.get("language", "en"),
            provider=self.provider_name,
            metadata=response_data.get("metadata", {})
        )
    
    async def _extract_structured_data_impl(
        self,
        file_path: Path,
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract structured data using OLM API."""
        
        # Encode image
        base64_image = await self._encode_image(file_path)
        
        # Prepare structured extraction payload
        structured_prompt = (
            f"Extract structured data from this document according to this schema: {json.dumps(schema)}. "
            f"{prompt or ''} Return the data as valid JSON."
        )
        
        payload = {
            "image": base64_image,
            "model_name": self.model_name,
            "prompt": structured_prompt,
            "schema": schema,
            "extraction_type": "structured",
            "options": kwargs
        }
        
        # Make API call
        response_data = await self._make_api_call("/extract_structured", payload)
        
        return {
            **response_data.get("structured_data", {}),
            "extraction_confidence": response_data.get("confidence", 0.8),
            "provider": self.provider_name
        }
    
    async def _test_api_connection(self) -> None:
        """Test connection to OLM API."""
        headers = self._get_headers()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_endpoint}/health",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    raise APIError(
                        f"API health check failed with status {response.status}",
                        status_code=response.status
                    )
    
    async def _make_api_call(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call to OLM service with retries."""
        headers = self._get_headers()
        url = f"{self.api_endpoint.rstrip('/')}{endpoint}"
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        if response.status == 200:
                            return await response.json()
                        
                        elif response.status == 429:  # Rate limit
                            if attempt < self.max_retries - 1:
                                wait_time = 2 ** attempt  # Exponential backoff
                                await asyncio.sleep(wait_time)
                                continue
                        
                        error_body = await response.text()
                        raise APIError(
                            f"OLM API request failed with status {response.status}",
                            status_code=response.status,
                            response_body=error_body
                        )
                        
            except aiohttp.ClientError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                break
        
        raise APIError(f"OLM API request failed after {self.max_retries} attempts: {last_exception}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Ocular-OCR/0.1.0"
        }
    
    async def _encode_image(self, file_path: Path) -> str:
        """Encode image file as base64."""
        try:
            with open(file_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise OCRError(f"Failed to encode image: {e}", provider=self.provider_name)
    
    def validate_config(self) -> bool:
        """Validate OLM provider configuration."""
        if not self.api_endpoint:
            return False
        
        if not self.api_key:
            return False
        
        if self.timeout <= 0 or self.timeout > 300:
            return False
        
        if self.max_retries < 1 or self.max_retries > 10:
            return False
        
        return True
    
    def get_supported_formats(self) -> list[str]:
        """Get supported formats for OLM OCR API."""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    async def cleanup(self) -> None:
        """Clean up API resources."""
        # No persistent resources to clean up for API-based provider
        pass