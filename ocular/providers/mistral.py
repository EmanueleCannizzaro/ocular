"""
Mistral AI OCR provider implementation with robust error handling and logging.
"""

import base64
import aiohttp
import asyncio
import json
from typing import Optional, Dict, Any
from pathlib import Path

from .base import BaseOCRProvider
from ..core.models import OCRResult
from ..core.enums import ProviderType
from ..core.exceptions import (
    OCRError, APIError, ConfigurationError, TimeoutError, 
    RateLimitError, AuthenticationError
)
from ..core.logging import log_async_function_call
from ..core.validation import text_validator


class MistralProvider(BaseOCRProvider):
    """Mistral AI OCR provider using vision models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate and set API key
        self.api_key = text_validator.validate_api_key(
            config.get("api_key", ""), "mistral"
        )
        
        self.model = config.get("model", "pixtral-12b-2409")
        self.timeout = max(10, min(300, config.get("timeout", 30)))  # 10-300 seconds
        self.max_retries = max(1, min(10, config.get("max_retries", 3)))  # 1-10 retries
        self.base_url = text_validator.validate_url(
            config.get("base_url", "https://api.mistral.ai/v1")
        )
        
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MISTRAL
    
    @property
    def provider_name(self) -> str:
        return "Mistral AI"
    
    @log_async_function_call()
    async def _do_initialize(self) -> None:
        """Initialize Mistral provider with comprehensive validation."""
        if not self.api_key:
            raise ConfigurationError(
                "Mistral API key is required",
                config_key="api_key"
            )
        
        self.logger.info(
            f"Initializing Mistral provider",
            model=self.model,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        
        # Test API connectivity with timeout
        try:
            await asyncio.wait_for(
                self._test_api_connection(),
                timeout=30  # 30 second timeout for initialization
            )
            
            self.logger.info("Mistral API connection test successful")
            
        except asyncio.TimeoutError:
            raise TimeoutError(
                "Mistral API connection test timed out",
                timeout_seconds=30,
                operation="initialization"
            )
        except Exception as e:
            self.logger.error(f"Mistral API connection test failed: {e}")
            raise ConfigurationError(f"Failed to connect to Mistral API: {e}")
    
    @log_async_function_call()
    async def is_available(self) -> bool:
        """Check if Mistral API is available."""
        try:
            await asyncio.wait_for(self._test_api_connection(), timeout=10)
            return True
        except Exception as e:
            self.logger.warning(f"Mistral API availability check failed: {e}")
            return False
    
    async def _test_api_connection(self) -> None:
        """Test API connection with robust error handling."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Ocular-OCR/0.1.0"
        }
        
        try:
            # Simple API test - list models
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError(
                            "Invalid Mistral API key",
                            provider="mistral"
                        )
                    elif response.status == 429:
                        raise RateLimitError(
                            "Mistral API rate limit exceeded",
                            provider="mistral"
                        )
                    elif response.status != 200:
                        response_text = await response.text()
                        raise APIError(
                            f"API test failed with status {response.status}",
                            status_code=response.status,
                            response_body=response_text,
                            endpoint=f"{self.base_url}/models"
                        )
                        
        except aiohttp.ClientError as e:
            raise APIError(
                f"Network error during API test: {e}",
                endpoint=f"{self.base_url}/models"
            )
    
    @log_async_function_call()
    async def _extract_text_impl(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text using Mistral vision API with comprehensive error handling."""
        
        self.logger.info(
            f"Starting Mistral OCR extraction for {file_path.name}",
            model=self.model,
            file_size=file_path.stat().st_size
        )
        
        # Validate and sanitize prompt
        if prompt:
            prompt = text_validator.validate_text(prompt, "prompt", allow_empty=True)
        
        try:
            # Encode image
            base64_image = await self._encode_image(file_path)
            
            # Prepare prompt
            default_prompt = (
                "Extract all text from this image accurately. "
                "Maintain original formatting and structure. "
                "Return only the extracted text without additional commentary."
            )
            user_prompt = prompt or default_prompt
            
            # Prepare request
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.1,  # Low temperature for consistent extraction
                **kwargs  # Allow additional parameters
            }
            
            # Make API call with retries
            response_text = await self._make_api_call(payload)
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(response_text)
            
            return OCRResult(
                text=response_text,
                confidence=confidence,
                language="auto-detected",
                provider=self.provider_name,
                metadata={
                    "model": self.model,
                    "prompt_length": len(user_prompt),
                    "response_length": len(response_text)
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Mistral OCR extraction failed for {file_path.name}: {e}",
                exc_info=True
            )
            raise
    
    async def _extract_structured_data_impl(
        self,
        file_path: Path,
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract structured data using Mistral vision API."""
        
        # Encode image
        base64_image = await self._encode_image(file_path)
        
        # Prepare structured extraction prompt
        schema_str = json.dumps(schema, indent=2)
        structured_prompt = (
            f"Extract structured data from this document according to this schema:\n\n{schema_str}\n\n"
            f"{prompt or ''}\n\n"
            "Return the data as valid JSON matching the schema exactly. "
            "If a field cannot be found, use null."
        )
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": structured_prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.0  # Very low temperature for structured data
        }
        
        response_text = await self._make_api_call(payload)
        
        # Try to parse as JSON
        try:
            # Clean up response text (remove markdown code blocks if present)
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            structured_data = json.loads(clean_text.strip())
            return structured_data
            
        except json.JSONDecodeError:
            # Fallback: return as text with metadata
            return {
                "raw_response": response_text,
                "extraction_successful": False,
                "schema": schema
            }
    
    @log_async_function_call()
    async def _make_api_call(self, payload: Dict[str, Any]) -> str:
        """Make API call with comprehensive retry logic and error handling."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Ocular-OCR/0.1.0"
        }
        
        endpoint = f"{self.base_url}/chat/completions"
        last_exception = None
        
        self.logger.info(
            f"Making Mistral API call",
            endpoint=endpoint,
            model=payload.get("model"),
            max_retries=self.max_retries,
            timeout=self.timeout
        )
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        # Log API call
                        self.logger.log_api_call(
                            endpoint=endpoint,
                            method="POST",
                            status_code=response.status
                        )
                        
                        if response.status == 200:
                            result = await response.json()
                            response_text = result["choices"][0]["message"]["content"].strip()
                            
                            self.logger.info(
                                f"Mistral API call successful",
                                response_length=len(response_text),
                                attempt=attempt + 1
                            )
                            
                            return response_text
                        
                        elif response.status == 401:
                            raise AuthenticationError(
                                "Invalid Mistral API key",
                                provider="mistral"
                            )
                        
                        elif response.status == 429:  # Rate limit
                            retry_after = response.headers.get('Retry-After', '60')
                            wait_time = min(int(retry_after), 2 ** attempt)
                            
                            if attempt < self.max_retries - 1:
                                self.logger.warning(
                                    f"Rate limited, waiting {wait_time}s before retry",
                                    attempt=attempt + 1,
                                    retry_after=retry_after
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                raise RateLimitError(
                                    "Mistral API rate limit exceeded",
                                    retry_after=int(retry_after),
                                    provider="mistral"
                                )
                        
                        elif response.status >= 500:  # Server errors - retryable
                            if attempt < self.max_retries - 1:
                                wait_time = 2 ** attempt
                                self.logger.warning(
                                    f"Server error {response.status}, retrying in {wait_time}s",
                                    attempt=attempt + 1
                                )
                                await asyncio.sleep(wait_time)
                                continue
                        
                        error_body = await response.text()
                        raise APIError(
                            f"API request failed with status {response.status}",
                            status_code=response.status,
                            response_body=error_body,
                            endpoint=endpoint
                        )
                        
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"Request timeout, retrying attempt {attempt + 2}/{self.max_retries}"
                    )
                    continue
                else:
                    raise TimeoutError(
                        f"Mistral API request timed out after {self.timeout} seconds",
                        timeout_seconds=self.timeout,
                        operation="api_call"
                    )
                    
            except aiohttp.ClientError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 10)  # Max 10 second wait
                    self.logger.warning(
                        f"Network error, retrying in {wait_time}s: {e}",
                        attempt=attempt + 1
                    )
                    await asyncio.sleep(wait_time)
                    continue
                break
        
        # If we get here, all retries failed
        raise APIError(
            f"Mistral API request failed after {self.max_retries} attempts: {last_exception}",
            endpoint=endpoint
        )
    
    async def _encode_image(self, file_path: Path) -> str:
        """Encode image file as base64, with PDF-to-image conversion support."""
        try:
            # Check if it's a PDF file
            if file_path.suffix.lower() == '.pdf':
                return await self._convert_pdf_to_base64(file_path)
            else:
                # Handle regular image files
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise OCRError(f"Failed to encode image: {e}", provider=self.provider_name)
    
    async def _convert_pdf_to_base64(self, pdf_path: Path) -> str:
        """Convert PDF to image and encode as base64."""
        try:
            # Try to import pdf2image
            try:
                from pdf2image import convert_from_path
            except ImportError:
                raise OCRError(
                    "PDF processing requires pdf2image library. "
                    "Install with: pip install pdf2image",
                    provider=self.provider_name
                )
            
            def _convert_pdf():
                # Convert first page of PDF to image
                images = convert_from_path(
                    pdf_path, 
                    dpi=200,  # Good quality for OCR
                    first_page=1,
                    last_page=1,  # Only process first page
                    fmt='JPEG'
                )
                
                if not images:
                    raise OCRError(
                        "Failed to convert PDF: no pages found",
                        provider=self.provider_name
                    )
                
                # Convert PIL Image to bytes
                import io
                img_buffer = io.BytesIO()
                images[0].save(img_buffer, format='JPEG', quality=95)
                return img_buffer.getvalue()
            
            # Run PDF conversion in thread to avoid blocking
            image_data = await asyncio.to_thread(_convert_pdf)
            
            # Encode as base64
            return base64.b64encode(image_data).decode('utf-8')
            
        except Exception as e:
            if "pdf2image" in str(e):
                raise  # Re-raise pdf2image import error
            raise OCRError(f"Failed to convert PDF to image: {e}", provider=self.provider_name)
    
    def validate_config(self) -> bool:
        """Validate Mistral provider configuration."""
        if not self.api_key:
            return False
        
        if len(self.api_key.strip()) < 10:
            return False
        
        if self.timeout <= 0 or self.timeout > 300:
            return False
        
        if self.max_retries < 1 or self.max_retries > 10:
            return False
        
        return True
    
    def get_supported_formats(self) -> list[str]:
        """Get supported formats for Mistral vision API."""
        # Mistral vision API supports common image formats, plus PDF conversion
        return ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.pdf']
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence based on text quality metrics."""
        if not text:
            return 0.0
        
        # Base confidence for Mistral API
        confidence = 0.85
        
        # Adjust based on text characteristics
        if len(text.strip()) < 5:
            confidence -= 0.3  # Very short text might be less reliable
        elif len(text.strip()) > 1000:
            confidence += 0.1  # Longer text suggests good extraction
        
        # Check for common OCR issues
        suspicious_patterns = ['�', '□', '◦', '●']
        for pattern in suspicious_patterns:
            if pattern in text:
                confidence -= 0.1
                break
        
        # Check for reasonable character distribution
        if text and text.isascii() and any(c.isalpha() for c in text):
            confidence += 0.05
        
        return max(0.0, min(1.0, confidence))