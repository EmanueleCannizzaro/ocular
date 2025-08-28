"""
AWS Textract OCR provider implementation.
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
    RateLimitError, TimeoutError, ResourceExhaustedError
)
from ..core.logging import log_async_function_call
from ..core.validation import text_validator, file_validator

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


class AWSTextractProvider(BaseOCRProvider):
    """AWS Textract OCR provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_BOTO3:
            raise ConfigurationError(
                "AWS Textract requires boto3. Install with: pip install boto3",
                config_key="aws_textract"
            )
        
        # AWS Configuration
        self.region = config.get("region", "us-east-1")
        self.access_key_id = config.get("access_key_id")
        self.secret_access_key = config.get("secret_access_key")
        self.session_token = config.get("session_token")
        
        # Textract-specific configuration
        self.feature_types = config.get("feature_types", ["TABLES", "FORMS"])
        self.document_pages = config.get("document_pages")  # For multi-page documents
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        
        # Client will be initialized lazily
        self._client = None
        
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.AWS_TEXTRACT
    
    @property
    def provider_name(self) -> str:
        return "AWS Textract"
    
    @log_async_function_call()
    async def _do_initialize(self) -> None:
        """Initialize AWS Textract client."""
        try:
            # Configure session
            session_kwargs = {"region_name": self.region}
            
            if self.access_key_id and self.secret_access_key:
                session_kwargs.update({
                    "aws_access_key_id": self.access_key_id,
                    "aws_secret_access_key": self.secret_access_key
                })
                
                if self.session_token:
                    session_kwargs["aws_session_token"] = self.session_token
            
            # Create Textract client
            session = boto3.Session(**session_kwargs)
            self._client = session.client('textract')
            
            self.logger.info(
                f"Initialized AWS Textract client",
                region=self.region,
                feature_types=self.feature_types
            )
            
            # Test the client
            await self._test_client()
            
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise AuthenticationError(
                f"AWS credentials not found or incomplete: {e}",
                provider="aws_textract"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS Textract client: {e}")
            raise ConfigurationError(
                f"AWS Textract initialization failed: {e}",
                config_key="aws_textract"
            )
    
    async def _test_client(self) -> None:
        """Test AWS Textract client connectivity."""
        try:
            def _test_api():
                # Test with a simple API call - describe the service limits
                return self._client.get_document_text_detection(JobId="test-job-id-that-does-not-exist")
            
            try:
                await asyncio.to_thread(_test_api)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'InvalidJobIdException':
                    # This is expected - we're just testing connectivity
                    self.logger.info("AWS Textract API connectivity test successful")
                elif error_code == 'AccessDeniedException':
                    raise AuthenticationError(
                        "AWS Textract access denied. Check permissions.",
                        provider="aws_textract"
                    )
                else:
                    raise APIError(f"AWS Textract API error: {e}")
                    
        except Exception as e:
            if "access denied" in str(e).lower():
                raise AuthenticationError(
                    f"AWS Textract authentication failed: {e}",
                    provider="aws_textract"
                )
            else:
                raise ConfigurationError(f"AWS Textract API test failed: {e}")
    
    @log_async_function_call()
    async def is_available(self) -> bool:
        """Check if AWS Textract is available."""
        try:
            if not self._client:
                await self.initialize()
            await self._test_client()
            return True
        except Exception as e:
            self.logger.warning(f"AWS Textract availability check failed: {e}")
            return False
    
    @log_async_function_call()
    async def _extract_text_impl(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Extract text using AWS Textract."""
        
        self.logger.info(
            f"Starting AWS Textract OCR for {file_path.name}",
            file_size=file_path.stat().st_size
        )
        
        try:
            # Read and prepare document
            document_bytes = await self._read_file(file_path)
            
            # Choose between synchronous and asynchronous processing based on file size
            max_sync_size = 5 * 1024 * 1024  # 5MB limit for sync processing
            
            if len(document_bytes) <= max_sync_size and file_path.suffix.lower() != '.pdf':
                # Synchronous processing for small images
                response = await self._process_document_sync(document_bytes)
            else:
                # Asynchronous processing for large documents or PDFs
                response = await self._process_document_async(document_bytes, file_path)
            
            # Extract text from response
            extracted_text = self._extract_text_from_response(response)
            confidence = self._calculate_confidence_from_response(response)
            
            return OCRResult(
                text=extracted_text,
                confidence=confidence,
                language="auto-detected",
                provider=self.provider_name,
                metadata={
                    "feature_types": self.feature_types,
                    "processing_mode": "sync" if len(document_bytes) <= max_sync_size else "async",
                    "blocks_count": len(response.get("Blocks", []))
                }
            )
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'ThrottlingException':
                raise RateLimitError(
                    "AWS Textract rate limit exceeded",
                    provider="aws_textract"
                )
            elif error_code == 'AccessDeniedException':
                raise AuthenticationError(
                    "AWS Textract access denied",
                    provider="aws_textract"
                )
            elif error_code == 'LimitExceededException':
                raise ResourceExhaustedError(
                    "AWS Textract service limit exceeded",
                    provider="aws_textract"
                )
            else:
                raise APIError(
                    f"AWS Textract API error: {e.response['Error']['Message']}",
                    status_code=e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
                )
                
        except Exception as e:
            self.logger.error(
                f"AWS Textract OCR failed for {file_path.name}: {e}",
                exc_info=True
            )
            raise
    
    async def _process_document_sync(self, document_bytes: bytes) -> Dict[str, Any]:
        """Process document synchronously."""
        def _sync_process():
            if self.feature_types:
                return self._client.analyze_document(
                    Document={'Bytes': document_bytes},
                    FeatureTypes=self.feature_types
                )
            else:
                return self._client.detect_document_text(
                    Document={'Bytes': document_bytes}
                )
        
        return await asyncio.to_thread(_sync_process)
    
    async def _process_document_async(self, document_bytes: bytes, file_path: Path) -> Dict[str, Any]:
        """Process document asynchronously (for large files)."""
        
        def _start_job():
            if self.feature_types:
                return self._client.start_document_analysis(
                    Document={'Bytes': document_bytes},
                    FeatureTypes=self.feature_types
                )
            else:
                return self._client.start_document_text_detection(
                    Document={'Bytes': document_bytes}
                )
        
        # Start the job
        start_response = await asyncio.to_thread(_start_job)
        job_id = start_response['JobId']
        
        self.logger.info(f"Started async Textract job: {job_id}")
        
        # Poll for completion
        max_wait_time = 300  # 5 minutes
        poll_interval = 2  # seconds
        waited_time = 0
        
        while waited_time < max_wait_time:
            def _get_job_status():
                if self.feature_types:
                    return self._client.get_document_analysis(JobId=job_id)
                else:
                    return self._client.get_document_text_detection(JobId=job_id)
            
            response = await asyncio.to_thread(_get_job_status)
            
            status = response['JobStatus']
            
            if status == 'SUCCEEDED':
                self.logger.info(f"Textract job {job_id} completed successfully")
                return response
            elif status == 'FAILED':
                raise OCRError(
                    f"AWS Textract job failed: {response.get('StatusMessage', 'Unknown error')}",
                    provider=self.provider_name
                )
            elif status in ['IN_PROGRESS']:
                # Continue waiting
                await asyncio.sleep(poll_interval)
                waited_time += poll_interval
            else:
                raise OCRError(
                    f"Unknown AWS Textract job status: {status}",
                    provider=self.provider_name
                )
        
        raise TimeoutError(
            f"AWS Textract job timed out after {max_wait_time} seconds",
            timeout_seconds=max_wait_time,
            operation="async_processing"
        )
    
    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text from Textract response."""
        text_lines = []
        
        blocks = response.get("Blocks", [])
        
        # Group blocks by type
        line_blocks = [block for block in blocks if block.get("BlockType") == "LINE"]
        
        # Sort lines by geometry (top to bottom, left to right)
        line_blocks.sort(key=lambda b: (
            b.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0),
            b.get("Geometry", {}).get("BoundingBox", {}).get("Left", 0)
        ))
        
        # Extract text from sorted lines
        for block in line_blocks:
            text = block.get("Text", "").strip()
            if text:
                text_lines.append(text)
        
        return "\n".join(text_lines)
    
    def _calculate_confidence_from_response(self, response: Dict[str, Any]) -> float:
        """Calculate average confidence from Textract response."""
        blocks = response.get("Blocks", [])
        word_blocks = [block for block in blocks if block.get("BlockType") == "WORD"]
        
        if not word_blocks:
            return 0.5
        
        confidences = [block.get("Confidence", 0) for block in word_blocks]
        return sum(confidences) / len(confidences) / 100.0  # Convert percentage to decimal
    
    async def _extract_structured_data_impl(
        self,
        file_path: Path,
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract structured data using AWS Textract."""
        
        # Process document with form and table detection
        document_bytes = await self._read_file(file_path)
        
        # Enable all feature types for structured extraction
        original_features = self.feature_types
        self.feature_types = ["TABLES", "FORMS", "SIGNATURES", "QUERIES"]
        
        try:
            if len(document_bytes) <= 5 * 1024 * 1024:
                response = await self._process_document_sync(document_bytes)
            else:
                response = await self._process_document_async(document_bytes, file_path)
        finally:
            # Restore original features
            self.feature_types = original_features
        
        # Extract structured information
        structured_data = {
            "text": self._extract_text_from_response(response),
            "tables": self._extract_tables_from_response(response),
            "forms": self._extract_forms_from_response(response),
            "signatures": self._extract_signatures_from_response(response),
            "confidence": self._calculate_confidence_from_response(response)
        }
        
        return structured_data
    
    def _extract_tables_from_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract table data from Textract response."""
        tables = []
        blocks = response.get("Blocks", [])
        
        # Find table blocks
        table_blocks = [block for block in blocks if block.get("BlockType") == "TABLE"]
        
        for table_block in table_blocks:
            table_data = {
                "confidence": table_block.get("Confidence", 0) / 100.0,
                "rows": [],
                "row_count": table_block.get("RowCount", 0),
                "column_count": table_block.get("ColumnCount", 0)
            }
            
            # Extract cell relationships
            relationships = table_block.get("Relationships", [])
            for relationship in relationships:
                if relationship.get("Type") == "CHILD":
                    cell_ids = relationship.get("Ids", [])
                    # Process cells (simplified - full implementation would build table structure)
                    for cell_id in cell_ids:
                        cell_block = next((b for b in blocks if b.get("Id") == cell_id), None)
                        if cell_block and cell_block.get("BlockType") == "CELL":
                            # Extract cell text and position
                            pass
            
            tables.append(table_data)
        
        return tables
    
    def _extract_forms_from_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract form data from Textract response."""
        forms = []
        blocks = response.get("Blocks", [])
        
        # Find key-value sets
        key_blocks = [block for block in blocks if block.get("BlockType") == "KEY_VALUE_SET" and block.get("EntityTypes") == ["KEY"]]
        
        for key_block in key_blocks:
            form_field = {
                "key": "",
                "value": "",
                "confidence": key_block.get("Confidence", 0) / 100.0
            }
            
            # Extract key and value text (simplified implementation)
            relationships = key_block.get("Relationships", [])
            for relationship in relationships:
                if relationship.get("Type") == "VALUE":
                    # Find corresponding value block
                    pass
            
            forms.append(form_field)
        
        return forms
    
    def _extract_signatures_from_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract signature data from Textract response."""
        signatures = []
        blocks = response.get("Blocks", [])
        
        # Find signature blocks
        signature_blocks = [block for block in blocks if block.get("BlockType") == "SIGNATURE"]
        
        for sig_block in signature_blocks:
            signature_data = {
                "confidence": sig_block.get("Confidence", 0) / 100.0,
                "geometry": sig_block.get("Geometry", {})
            }
            signatures.append(signature_data)
        
        return signatures
    
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
        """Validate AWS Textract provider configuration."""
        # Check region
        if not self.region:
            return False
        
        # Check timeout
        if self.timeout <= 0 or self.timeout > 600:
            return False
        
        # Check feature types
        valid_features = ["TABLES", "FORMS", "SIGNATURES", "QUERIES"]
        if self.feature_types:
            if not all(feature in valid_features for feature in self.feature_types):
                return False
        
        return True
    
    def get_supported_formats(self) -> List[str]:
        """Get supported formats for AWS Textract."""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']
    
    async def cleanup(self) -> None:
        """Clean up AWS Textract resources."""
        # AWS clients handle cleanup automatically
        self._client = None

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence based on text quality metrics."""
        if not text:
            return 0.0
        
        # AWS Textract has high accuracy, so base confidence is high
        confidence = 0.9
        
        # Adjust based on text characteristics
        if len(text) < 10:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))