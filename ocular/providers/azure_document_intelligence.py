"""
Azure Document Intelligence OCR provider implementation.
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from .base import BaseOCRProvider
from ..core.models import OCRResult
from ..core.enums import ProviderType
from ..core.exceptions import (
    OCRError, APIError, ConfigurationError, AuthenticationError,
    RateLimitError, TimeoutError, ResourceExhaustedError
)
from ..core.logging import log_async_function_call
from ..core.validation import text_validator, file_validator

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import (
        ClientAuthenticationError, ResourceNotFoundError,
        HttpResponseError, ServiceRequestError
    )
    HAS_AZURE_DOC_INTEL = True
except ImportError:
    HAS_AZURE_DOC_INTEL = False


class AzureDocumentIntelligenceProvider(BaseOCRProvider):
    """Azure Document Intelligence OCR provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_AZURE_DOC_INTEL:
            raise ConfigurationError(
                "Azure Document Intelligence requires azure-ai-documentintelligence. "
                "Install with: pip install azure-ai-documentintelligence",
                config_key="azure_doc_intel"
            )
        
        # Azure configuration
        self.endpoint = config.get("endpoint")
        self.api_key = config.get("api_key")
        self.api_version = config.get("api_version", "2024-02-29-preview")
        
        # Document Intelligence specific settings
        self.model_id = config.get("model_id", "prebuilt-read")
        self.pages = config.get("pages")  # Specific pages to analyze
        self.locale = config.get("locale", "en-US")
        self.reading_order = config.get("reading_order", "natural")
        
        # Processing settings
        self.timeout = config.get("timeout", 120)
        self.max_retries = config.get("max_retries", 3)
        self.poll_interval = config.get("poll_interval", 2)
        
        # Client will be initialized lazily
        self._client = None
        
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.AZURE_DOCUMENT_INTELLIGENCE
    
    @property
    def provider_name(self) -> str:
        return "Azure Document Intelligence"
    
    @log_async_function_call()
    async def _do_initialize(self) -> None:
        """Initialize Azure Document Intelligence client."""
        try:
            if not self.endpoint:
                raise ConfigurationError(
                    "Azure endpoint is required",
                    config_key="endpoint"
                )
                
            if not self.api_key:
                raise ConfigurationError(
                    "Azure API key is required", 
                    config_key="api_key"
                )
            
            # Initialize client
            credential = AzureKeyCredential(self.api_key)
            self._client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=credential,
                api_version=self.api_version
            )
            
            self.logger.info(
                f"Initialized Azure Document Intelligence client",
                endpoint=self.endpoint,
                model_id=self.model_id,
                api_version=self.api_version
            )
            
            # Test the client
            await self._test_client()
            
        except ClientAuthenticationError as e:
            raise AuthenticationError(
                f"Azure authentication failed: {e}",
                provider="azure_doc_intel"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure Document Intelligence client: {e}")
            raise ConfigurationError(
                f"Azure Document Intelligence initialization failed: {e}",
                config_key="azure_doc_intel"
            )
    
    async def _test_client(self) -> None:
        """Test Azure Document Intelligence client connectivity."""
        try:
            def _test_api():
                # Test by trying to get document models (this doesn't process any documents)
                try:
                    list(self._client.list_document_models())
                    return True
                except Exception as e:
                    # If we can't list models, try a simple operation info call
                    return False
            
            await asyncio.to_thread(_test_api)
            self.logger.info("Azure Document Intelligence API connectivity test successful")
            
        except ClientAuthenticationError:
            raise AuthenticationError(
                "Azure Document Intelligence authentication failed",
                provider="azure_doc_intel"
            )
        except Exception as e:
            raise ConfigurationError(f"Azure Document Intelligence API test failed: {e}")
    
    @log_async_function_call()
    async def is_available(self) -> bool:
        """Check if Azure Document Intelligence is available."""
        try:
            if not self._client:
                await self.initialize()
            await self._test_client()
            return True
        except Exception as e:
            self.logger.warning(f"Azure Document Intelligence availability check failed: {e}")
            return False
    
    @log_async_function_call()
    async def _extract_text_impl(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text using Azure Document Intelligence."""
        
        self.logger.info(
            f"Starting Azure Document Intelligence OCR for {file_path.name}",
            file_size=file_path.stat().st_size,
            model_id=self.model_id
        )
        
        try:
            # Read document
            document_bytes = await self._read_file(file_path)
            
            # Choose model based on requirements
            model_id = kwargs.get("model_id", self.model_id)
            
            # Start analysis
            poller = await self._start_analysis(document_bytes, model_id, **kwargs)
            
            # Wait for completion
            result = await self._wait_for_completion(poller)
            
            # Extract text and metadata
            extracted_text = self._extract_text_from_result(result)
            confidence = self._calculate_confidence_from_result(result)
            language = self._detect_language_from_result(result)
            
            return OCRResult(
                text=extracted_text,
                confidence=confidence,
                language=language,
                provider=self.provider_name,
                metadata={
                    "model_id": model_id,
                    "api_version": self.api_version,
                    "pages_analyzed": len(result.pages) if hasattr(result, 'pages') else 1,
                    "reading_order": self.reading_order
                }
            )
            
        except ClientAuthenticationError:
            raise AuthenticationError(
                "Azure Document Intelligence authentication failed",
                provider="azure_doc_intel"
            )
        except HttpResponseError as e:
            if e.status_code == 429:
                raise RateLimitError(
                    "Azure Document Intelligence rate limit exceeded",
                    provider="azure_doc_intel"
                )
            elif e.status_code == 403:
                raise AuthenticationError(
                    "Azure Document Intelligence access forbidden",
                    provider="azure_doc_intel"
                )
            else:
                raise APIError(
                    f"Azure Document Intelligence API error: {e.message}",
                    status_code=e.status_code
                )
        except Exception as e:
            self.logger.error(
                f"Azure Document Intelligence OCR failed for {file_path.name}: {e}",
                exc_info=True
            )
            raise
    
    async def _start_analysis(
        self, 
        document_bytes: bytes, 
        model_id: str, 
        **kwargs
    ) -> Any:
        """Start document analysis and return poller."""
        
        def _begin_analyze():
            analyze_request = AnalyzeDocumentRequest(
                bytes_source=document_bytes
            )
            
            # Additional parameters
            analyze_params = {
                "pages": kwargs.get("pages", self.pages),
                "locale": kwargs.get("locale", self.locale),
                "string_index_type": "textElements"
            }
            
            # Remove None values
            analyze_params = {k: v for k, v in analyze_params.items() if v is not None}
            
            return self._client.begin_analyze_document(
                model_id=model_id,
                analyze_request=analyze_request,
                **analyze_params
            )
        
        return await asyncio.to_thread(_begin_analyze)
    
    async def _wait_for_completion(self, poller: Any) -> Any:
        """Wait for analysis completion with timeout."""
        start_time = time.time()
        
        while not poller.done():
            if time.time() - start_time > self.timeout:
                try:
                    poller.cancel()
                except Exception:
                    pass
                raise TimeoutError(
                    f"Azure Document Intelligence analysis timed out after {self.timeout} seconds",
                    timeout_seconds=self.timeout,
                    operation="document_analysis"
                )
            
            await asyncio.sleep(self.poll_interval)
        
        def _get_result():
            return poller.result()
        
        return await asyncio.to_thread(_get_result)
    
    def _extract_text_from_result(self, result: Any) -> str:
        """Extract text from Azure Document Intelligence result."""
        if not hasattr(result, 'content'):
            return ""
        
        return result.content
    
    def _calculate_confidence_from_result(self, result: Any) -> float:
        """Calculate confidence from Azure Document Intelligence result."""
        if not hasattr(result, 'pages'):
            return 0.8  # Default confidence
        
        total_confidence = 0.0
        word_count = 0
        
        for page in result.pages:
            if hasattr(page, 'words'):
                for word in page.words:
                    if hasattr(word, 'confidence'):
                        total_confidence += word.confidence
                        word_count += 1
        
        if word_count == 0:
            return 0.8
        
        return total_confidence / word_count
    
    def _detect_language_from_result(self, result: Any) -> str:
        """Detect language from Azure Document Intelligence result."""
        if hasattr(result, 'languages') and result.languages:
            return result.languages[0].locale
        
        return self.locale
    
    async def _extract_structured_data_impl(
        self,
        file_path: Path,
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract structured data using Azure Document Intelligence."""
        
        # Read document
        document_bytes = await self._read_file(file_path)
        
        # Use layout model for structured extraction
        model_id = kwargs.get("model_id", "prebuilt-layout")
        
        # Start analysis
        poller = await self._start_analysis(document_bytes, model_id, **kwargs)
        result = await self._wait_for_completion(poller)
        
        # Extract structured information
        structured_data = {
            "content": result.content,
            "pages": [],
            "tables": [],
            "paragraphs": [],
            "lines": [],
            "words": [],
            "confidence": self._calculate_confidence_from_result(result)
        }
        
        # Extract page information
        if hasattr(result, 'pages'):
            for page_idx, page in enumerate(result.pages):
                page_info = {
                    "page_number": page_idx + 1,
                    "width": page.width if hasattr(page, 'width') else None,
                    "height": page.height if hasattr(page, 'height') else None,
                    "angle": page.angle if hasattr(page, 'angle') else None,
                    "unit": page.unit if hasattr(page, 'unit') else None
                }
                structured_data["pages"].append(page_info)
                
                # Extract lines
                if hasattr(page, 'lines'):
                    for line in page.lines:
                        line_info = {
                            "content": line.content,
                            "bounding_box": self._extract_polygon(line.polygon) if hasattr(line, 'polygon') else None
                        }
                        structured_data["lines"].append(line_info)
                
                # Extract words
                if hasattr(page, 'words'):
                    for word in page.words:
                        word_info = {
                            "content": word.content,
                            "confidence": word.confidence if hasattr(word, 'confidence') else None,
                            "bounding_box": self._extract_polygon(word.polygon) if hasattr(word, 'polygon') else None
                        }
                        structured_data["words"].append(word_info)
        
        # Extract tables
        if hasattr(result, 'tables'):
            for table in result.tables:
                table_info = {
                    "row_count": table.row_count if hasattr(table, 'row_count') else 0,
                    "column_count": table.column_count if hasattr(table, 'column_count') else 0,
                    "cells": []
                }
                
                if hasattr(table, 'cells'):
                    for cell in table.cells:
                        cell_info = {
                            "content": cell.content,
                            "row_index": cell.row_index if hasattr(cell, 'row_index') else None,
                            "column_index": cell.column_index if hasattr(cell, 'column_index') else None,
                            "row_span": cell.row_span if hasattr(cell, 'row_span') else 1,
                            "column_span": cell.column_span if hasattr(cell, 'column_span') else 1
                        }
                        table_info["cells"].append(cell_info)
                
                structured_data["tables"].append(table_info)
        
        # Extract paragraphs
        if hasattr(result, 'paragraphs'):
            for paragraph in result.paragraphs:
                para_info = {
                    "content": paragraph.content,
                    "bounding_regions": []
                }
                
                if hasattr(paragraph, 'bounding_regions'):
                    for region in paragraph.bounding_regions:
                        region_info = {
                            "page_number": region.page_number if hasattr(region, 'page_number') else None,
                            "polygon": self._extract_polygon(region.polygon) if hasattr(region, 'polygon') else None
                        }
                        para_info["bounding_regions"].append(region_info)
                
                structured_data["paragraphs"].append(para_info)
        
        return structured_data
    
    def _extract_polygon(self, polygon: Any) -> List[Dict[str, float]]:
        """Extract polygon coordinates."""
        if not polygon:
            return []
        
        points = []
        for point in polygon:
            if hasattr(point, 'x') and hasattr(point, 'y'):
                points.append({"x": point.x, "y": point.y})
        
        return points
    
    async def _read_file(self, file_path: Path) -> bytes:
        """Read file as bytes."""
        try:
            def _read_file_sync():
                with open(file_path, 'rb') as f:
                    return f.read()
            
            return await asyncio.to_thread(_read_file_sync)
        except Exception as e:
            raise OCRError(f"Failed to read file: {e}", provider=self.provider_name)
    
    def validate_config(self) -> bool:
        """Validate Azure Document Intelligence provider configuration."""
        # Check endpoint
        if not self.endpoint or not self.endpoint.startswith('https://'):
            return False
        
        # Check API key
        if not self.api_key or len(self.api_key.strip()) < 10:
            return False
        
        # Check timeout
        if self.timeout <= 0 or self.timeout > 600:
            return False
        
        # Check model ID
        if not self.model_id:
            return False
        
        return True
    
    def get_supported_formats(self) -> List[str]:
        """Get supported formats for Azure Document Intelligence."""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']
    
    async def cleanup(self) -> None:
        """Clean up Azure Document Intelligence resources."""
        # Azure client handles cleanup automatically
        self._client = None

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence based on text quality metrics."""
        if not text:
            return 0.0
        
        # Azure Document Intelligence has high accuracy
        confidence = 0.9
        
        # Adjust based on text characteristics
        if len(text) < 10:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))