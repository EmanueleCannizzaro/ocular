"""
Validation service for Ocular OCR system.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..core.models import ProcessingRequest, BatchProcessingRequest, OCRResult
from ..core.enums import ProviderType, ProcessingStrategy
from ..core.exceptions import ValidationError
from ..providers.settings import OcularSettings

logger = logging.getLogger(__name__)


class ValidationService:
    """Service for validating requests and data."""
    
    def __init__(self, settings: OcularSettings):
        self.settings = settings
    
    async def validate_processing_request(self, request: ProcessingRequest) -> bool:
        """Validate a processing request."""
        
        # Validate file path
        if not request.file_path or not isinstance(request.file_path, Path):
            raise ValidationError("file_path must be a valid Path object", field="file_path")
        
        if not request.file_path.exists():
            raise ValidationError(
                f"File does not exist: {request.file_path}",
                field="file_path",
                value=str(request.file_path)
            )
        
        if not request.file_path.is_file():
            raise ValidationError(
                f"Path is not a file: {request.file_path}",
                field="file_path", 
                value=str(request.file_path)
            )
        
        # Validate file size
        file_size_mb = request.file_path.stat().st_size / (1024 * 1024)
        max_size = self.settings.files.max_file_size_mb
        
        if file_size_mb > max_size:
            raise ValidationError(
                f"File size ({file_size_mb:.2f}MB) exceeds limit of {max_size}MB",
                field="file_size",
                value=file_size_mb
            )
        
        # Validate file extension
        suffix = request.file_path.suffix.lower()
        allowed_extensions = [ext.lower() for ext in self.settings.files.allowed_extensions]
        
        if suffix not in allowed_extensions:
            raise ValidationError(
                f"File extension {suffix} not allowed. Allowed: {allowed_extensions}",
                field="file_extension",
                value=suffix
            )
        
        # Validate providers
        if not request.providers:
            raise ValidationError("At least one provider must be specified", field="providers")
        
        for provider in request.providers:
            if not isinstance(provider, ProviderType):
                raise ValidationError(
                    f"Invalid provider type: {provider}",
                    field="providers",
                    value=str(provider)
                )
            
            if not self.settings.is_provider_enabled(provider):
                raise ValidationError(
                    f"Provider {provider.value} is not enabled",
                    field="providers",
                    value=provider.value
                )
        
        # Validate strategy
        if not isinstance(request.strategy, ProcessingStrategy):
            raise ValidationError(
                f"Invalid processing strategy: {request.strategy}",
                field="strategy",
                value=str(request.strategy)
            )
        
        # Validate prompt length
        if request.prompt and len(request.prompt) > 2000:
            raise ValidationError(
                "Prompt too long (max 2000 characters)",
                field="prompt",
                value=len(request.prompt)
            )
        
        return True
    
    async def validate_batch_request(self, request: BatchProcessingRequest) -> bool:
        """Validate a batch processing request."""
        
        # Validate file paths
        if not request.file_paths:
            raise ValidationError("At least one file path must be specified", field="file_paths")
        
        if len(request.file_paths) > 100:  # Reasonable batch size limit
            raise ValidationError(
                f"Too many files in batch ({len(request.file_paths)}). Maximum: 100",
                field="file_paths",
                value=len(request.file_paths)
            )
        
        # Validate each file path
        for i, file_path in enumerate(request.file_paths):
            try:
                individual_request = ProcessingRequest(
                    file_path=file_path,
                    strategy=request.strategy,
                    providers=request.providers,
                    prompt=request.prompt,
                    options=request.options
                )
                await self.validate_processing_request(individual_request)
            except ValidationError as e:
                raise ValidationError(
                    f"File {i+1} validation failed: {e.message}",
                    field="file_paths",
                    value=str(file_path)
                ) from e
        
        # Validate max_concurrent
        if request.max_concurrent < 1 or request.max_concurrent > 10:
            raise ValidationError(
                "max_concurrent must be between 1 and 10",
                field="max_concurrent",
                value=request.max_concurrent
            )
        
        return True
    
    def validate_ocr_result(self, result: OCRResult) -> bool:
        """Validate an OCR result."""
        
        if not result.text and result.text != "":  # Allow empty string but not None
            raise ValidationError("OCR result must have text field", field="text")
        
        if result.confidence is not None:
            if not (0.0 <= result.confidence <= 1.0):
                raise ValidationError(
                    "Confidence must be between 0.0 and 1.0",
                    field="confidence",
                    value=result.confidence
                )
        
        if not result.provider:
            raise ValidationError("OCR result must specify provider", field="provider")
        
        if result.processing_time is not None:
            if result.processing_time < 0:
                raise ValidationError(
                    "Processing time cannot be negative",
                    field="processing_time",
                    value=result.processing_time
                )
        
        return True
    
    def validate_structured_data_schema(self, schema: Dict[str, Any]) -> bool:
        """Validate a structured data extraction schema."""
        
        if not schema:
            raise ValidationError("Schema cannot be empty", field="schema")
        
        if not isinstance(schema, dict):
            raise ValidationError("Schema must be a dictionary", field="schema")
        
        # Check for reasonable complexity
        if len(schema) > 50:
            raise ValidationError(
                "Schema too complex (max 50 fields)",
                field="schema",
                value=len(schema)
            )
        
        # Validate field names
        for field_name in schema.keys():
            if not self._is_valid_field_name(field_name):
                raise ValidationError(
                    f"Invalid field name: {field_name}",
                    field="schema",
                    value=field_name
                )
        
        return True
    
    def _is_valid_field_name(self, name: str) -> bool:
        """Check if field name is valid."""
        
        if not name or len(name) > 100:
            return False
        
        # Must start with letter or underscore, contain only alphanumeric and underscore
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, name))
    
    def validate_upload_file(self, content: bytes, filename: str) -> bool:
        """Validate uploaded file content and filename."""
        
        if not content:
            raise ValidationError("File content cannot be empty", field="content")
        
        if not filename:
            raise ValidationError("Filename is required", field="filename")
        
        # Check file size
        size_mb = len(content) / (1024 * 1024)
        max_size = self.settings.files.max_file_size_mb
        
        if size_mb > max_size:
            raise ValidationError(
                f"File size ({size_mb:.2f}MB) exceeds limit of {max_size}MB",
                field="file_size",
                value=size_mb
            )
        
        # Check filename
        if not self._is_safe_filename(filename):
            raise ValidationError(
                f"Unsafe filename: {filename}",
                field="filename",
                value=filename
            )
        
        # Check file extension
        path = Path(filename)
        suffix = path.suffix.lower()
        allowed_extensions = [ext.lower() for ext in self.settings.files.allowed_extensions]
        
        if suffix not in allowed_extensions:
            raise ValidationError(
                f"File extension {suffix} not allowed. Allowed: {allowed_extensions}",
                field="file_extension",
                value=suffix
            )
        
        # Basic content validation
        if not self._is_valid_file_content(content, suffix):
            raise ValidationError(
                "File content appears to be corrupted or invalid",
                field="content"
            )
        
        return True
    
    def _is_safe_filename(self, filename: str) -> bool:
        """Check if filename is safe."""
        
        if not filename or len(filename) > 255:
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'\.\./',  # Path traversal
            r'^\.',    # Hidden files
            r'[<>:"|?*]',  # Invalid chars
            r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$'  # Windows reserved names
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return False
        
        return True
    
    def _is_valid_file_content(self, content: bytes, extension: str) -> bool:
        """Basic validation of file content."""
        
        if not content:
            return False
        
        # Check for common file signatures
        if extension == '.pdf':
            return content.startswith(b'%PDF-')
        elif extension in ['.jpg', '.jpeg']:
            return content.startswith(b'\xff\xd8\xff')
        elif extension == '.png':
            return content.startswith(b'\x89PNG\r\n\x1a\n')
        elif extension == '.bmp':
            return content.startswith(b'BM')
        elif extension == '.webp':
            return b'WEBP' in content[:12]
        
        # For other formats, just check it's not empty
        return len(content) > 0
    
    def validate_api_request_size(self, content_length: Optional[int]) -> bool:
        """Validate API request size."""
        
        if content_length is None:
            return True  # Cannot validate unknown size
        
        max_size_bytes = self.settings.web.max_request_size_mb * 1024 * 1024
        
        if content_length > max_size_bytes:
            max_size_mb = self.settings.web.max_request_size_mb
            actual_size_mb = content_length / (1024 * 1024)
            
            raise ValidationError(
                f"Request size ({actual_size_mb:.2f}MB) exceeds limit of {max_size_mb}MB",
                field="request_size",
                value=actual_size_mb
            )
        
        return True
    
    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitize user prompt input."""
        
        if not prompt:
            return ""
        
        # Remove potentially dangerous content
        # This is a basic implementation - enhance based on security requirements
        sanitized = prompt.strip()
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Limit length
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000]
            logger.warning("Prompt truncated to 2000 characters")
        
        return sanitized
    
    def validate_processing_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize processing options."""
        
        if not options:
            return {}
        
        validated_options = {}
        
        # Define allowed options and their types
        allowed_options = {
            "enhance_image": bool,
            "use_preprocessing": bool,
            "ensemble_size": int,
            "beam_size": int,
            "max_tokens": int,
            "temperature": float,
        }
        
        for key, value in options.items():
            if key not in allowed_options:
                logger.warning(f"Unknown processing option ignored: {key}")
                continue
            
            expected_type = allowed_options[key]
            
            try:
                # Type coercion
                if expected_type == bool:
                    validated_options[key] = bool(value)
                elif expected_type == int:
                    validated_options[key] = int(value)
                elif expected_type == float:
                    validated_options[key] = float(value)
                else:
                    validated_options[key] = value
                    
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for option {key}: {value}")
                continue
        
        return validated_options