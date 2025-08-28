"""
Google Cloud Vision API OCR provider implementation.
"""

import base64
import asyncio
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

from .base import BaseOCRProvider
from ..core.models import OCRResult
from ..core.enums import ProviderType
from ..core.exceptions import (
    OCRError, APIError, ConfigurationError, AuthenticationError, 
    RateLimitError, TimeoutError
)
from ..core.logging import log_async_function_call
from ..core.validation import text_validator, file_validator

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    from google.api_core import exceptions as google_exceptions
    HAS_GOOGLE_VISION = True
except ImportError:
    HAS_GOOGLE_VISION = False


class GoogleVisionProvider(BaseOCRProvider):
    """Google Cloud Vision API OCR provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_GOOGLE_VISION:
            raise ConfigurationError(
                "Google Cloud Vision API libraries not installed. "
                "Install with: pip install google-cloud-vision",
                config_key="google_vision"
            )
        
        # Configuration
        self.credentials_path = config.get("credentials_path")
        self.project_id = config.get("project_id")
        self.location = config.get("location", "us")
        self.language_hints = config.get("language_hints", ["en"])
        self.features = config.get("features", ["TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION"])
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
        # Client will be initialized lazily
        self._client = None
        
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE_VISION
    
    @property
    def provider_name(self) -> str:
        return "Google Cloud Vision"
    
    @log_async_function_call()
    async def _do_initialize(self) -> None:
        """Initialize Google Cloud Vision client."""
        try:
            # Initialize credentials
            if self.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self._client = vision.ImageAnnotatorClient(credentials=credentials)
            else:
                # Use default credentials (environment variable)
                self._client = vision.ImageAnnotatorClient()
            
            self.logger.info(
                f"Initialized Google Vision client",
                project_id=self.project_id,
                location=self.location,
                language_hints=self.language_hints
            )
            
            # Test the client
            await self._test_client()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Vision client: {e}")
            raise ConfigurationError(
                f"Google Vision initialization failed: {e}",
                config_key="google_vision"
            )
    
    async def _test_client(self) -> None:
        """Test Google Vision client connectivity."""
        try:
            # Create a simple test image (1x1 pixel white image)
            test_image_data = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            )
            
            def _test_api():
                image = vision.Image(content=test_image_data)
                response = self._client.text_detection(image=image)
                return response
            
            # Run in thread to avoid blocking
            await asyncio.to_thread(_test_api)
            
            self.logger.info("Google Vision API connectivity test successful")
            
        except google_exceptions.Unauthenticated:
            raise AuthenticationError(
                "Google Vision API authentication failed. Check credentials.",
                provider="google_vision"
            )
        except google_exceptions.PermissionDenied:
            raise AuthenticationError(
                "Google Vision API permission denied. Check API permissions.",
                provider="google_vision"
            )
        except Exception as e:
            raise ConfigurationError(f"Google Vision API test failed: {e}")
    
    @log_async_function_call()
    async def is_available(self) -> bool:
        """Check if Google Vision API is available."""
        try:
            if not self._client:
                await self.initialize()
            await self._test_client()
            return True
        except Exception as e:
            self.logger.warning(f"Google Vision availability check failed: {e}")
            return False
    
    @log_async_function_call()
    async def _extract_text_impl(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text using Google Vision API."""
        
        self.logger.info(
            f"Starting Google Vision OCR for {file_path.name}",
            file_size=file_path.stat().st_size
        )
        
        try:
            # Read and prepare image
            image_content = await self._read_image_file(file_path)
            image = vision.Image(content=image_content)
            
            # Configure image context
            image_context = vision.ImageContext(language_hints=self.language_hints)
            
            # Choose detection type based on use case
            detection_type = kwargs.get("detection_type", "document")
            
            def _perform_ocr():
                if detection_type == "document":
                    response = self._client.document_text_detection(
                        image=image, 
                        image_context=image_context
                    )
                else:
                    response = self._client.text_detection(
                        image=image, 
                        image_context=image_context
                    )
                
                if response.error.message:
                    raise OCRError(
                        f"Google Vision API error: {response.error.message}",
                        provider=self.provider_name
                    )
                
                return response
            
            # Run OCR in thread to avoid blocking
            response = await asyncio.to_thread(_perform_ocr)
            
            # Extract text
            if detection_type == "document" and response.full_text_annotation:
                extracted_text = response.full_text_annotation.text
                confidence = self._calculate_document_confidence(response.full_text_annotation)
            else:
                text_annotations = response.text_annotations
                if text_annotations:
                    extracted_text = text_annotations[0].description
                    confidence = self._calculate_text_confidence(text_annotations)
                else:
                    extracted_text = ""
                    confidence = 0.0
            
            # Detect language
            detected_language = self._detect_language(response)
            
            return OCRResult(
                text=extracted_text,
                confidence=confidence,
                language=detected_language,
                provider=self.provider_name,
                metadata={
                    "detection_type": detection_type,
                    "language_hints": self.language_hints,
                    "text_annotations_count": len(response.text_annotations) if response.text_annotations else 0
                }
            )
            
        except google_exceptions.Unauthenticated:
            raise AuthenticationError(
                "Google Vision API authentication failed",
                provider="google_vision"
            )
        except google_exceptions.ResourceExhausted:
            raise RateLimitError(
                "Google Vision API quota exceeded",
                provider="google_vision"
            )
        except google_exceptions.DeadlineExceeded:
            raise TimeoutError(
                "Google Vision API request timed out",
                timeout_seconds=self.timeout,
                operation="text_extraction"
            )
        except Exception as e:
            self.logger.error(
                f"Google Vision OCR failed for {file_path.name}: {e}",
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
        """Extract structured data using Google Vision API."""
        
        # First extract text
        ocr_result = await self._extract_text_impl(file_path, prompt, **kwargs)
        
        # Use document layout analysis if available
        image_content = await self._read_image_file(file_path)
        image = vision.Image(content=image_content)
        
        def _analyze_document():
            response = self._client.document_text_detection(image=image)
            return response
        
        response = await asyncio.to_thread(_analyze_document)
        
        # Extract structured information from Google's response
        structured_data = {
            "extracted_text": ocr_result.text,
            "confidence": ocr_result.confidence,
            "pages": [],
            "blocks": [],
            "paragraphs": [],
            "words": []
        }
        
        if response.full_text_annotation:
            # Extract page-level information
            for page in response.full_text_annotation.pages:
                page_info = {
                    "width": page.width if hasattr(page, 'width') else None,
                    "height": page.height if hasattr(page, 'height') else None,
                    "blocks": len(page.blocks)
                }
                structured_data["pages"].append(page_info)
                
                # Extract block-level information
                for block in page.blocks:
                    block_text = "".join([
                        "".join([
                            "".join([symbol.text for symbol in word.symbols])
                            for word in paragraph.words
                        ])
                        for paragraph in block.paragraphs
                    ])
                    
                    block_info = {
                        "text": block_text,
                        "confidence": block.confidence if hasattr(block, 'confidence') else None,
                        "bounding_box": self._extract_bounding_box(block.bounding_box) if hasattr(block, 'bounding_box') else None
                    }
                    structured_data["blocks"].append(block_info)
        
        return structured_data
    
    def _calculate_document_confidence(self, full_text_annotation) -> float:
        """Calculate confidence from document text detection."""
        if not full_text_annotation.pages:
            return 0.0
        
        total_confidence = 0.0
        total_words = 0
        
        for page in full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        if hasattr(word, 'confidence'):
                            total_confidence += word.confidence
                            total_words += 1
        
        return total_confidence / total_words if total_words > 0 else 0.5
    
    def _calculate_text_confidence(self, text_annotations) -> float:
        """Calculate confidence from text annotations."""
        if len(text_annotations) <= 1:
            return 0.5
        
        # Skip the first annotation (full text) and average the rest
        confidences = [
            ann.confidence for ann in text_annotations[1:] 
            if hasattr(ann, 'confidence')
        ]
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _detect_language(self, response) -> str:
        """Detect language from response."""
        if (response.full_text_annotation and 
            response.full_text_annotation.pages and
            response.full_text_annotation.pages[0].blocks):
            
            # Try to get language from the first block
            first_block = response.full_text_annotation.pages[0].blocks[0]
            if (hasattr(first_block, 'property') and 
                hasattr(first_block.property, 'detected_languages')):
                languages = first_block.property.detected_languages
                if languages:
                    return languages[0].language_code
        
        return "en"  # Default to English
    
    def _extract_bounding_box(self, bounding_box) -> Dict[str, int]:
        """Extract bounding box coordinates."""
        return {
            "x1": bounding_box.vertices[0].x,
            "y1": bounding_box.vertices[0].y,
            "x2": bounding_box.vertices[2].x,
            "y2": bounding_box.vertices[2].y
        }
    
    async def _read_image_file(self, file_path: Path) -> bytes:
        """Read image file as bytes."""
        try:
            def _read_file():
                with open(file_path, 'rb') as f:
                    return f.read()
            
            return await asyncio.to_thread(_read_file)
        except Exception as e:
            raise OCRError(f"Failed to read image file: {e}", provider=self.provider_name)
    
    def validate_config(self) -> bool:
        """Validate Google Vision provider configuration."""
        # Check if credentials are available
        if self.credentials_path and not Path(self.credentials_path).exists():
            return False
        
        # Check timeout
        if self.timeout <= 0 or self.timeout > 300:
            return False
        
        # Check language hints
        if not isinstance(self.language_hints, list):
            return False
        
        return True
    
    def get_supported_formats(self) -> List[str]:
        """Get supported formats for Google Vision API."""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.pdf', '.tiff']
    
    async def cleanup(self) -> None:
        """Clean up Google Vision resources."""
        if self._client:
            # Google Vision client doesn't require explicit cleanup
            self._client = None

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence based on text quality metrics."""
        if not text:
            return 0.0
        
        # Simple heuristic based on text characteristics
        confidence = 0.8
        
        # Boost confidence for longer text
        if len(text) > 100:
            confidence += 0.1
        
        # Reduce confidence for very short text
        if len(text) < 10:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))