"""
Tests for validation service.
"""

import pytest
from pathlib import Path

from ocular.services.validation_service import ValidationService
from ocular.core.models import ProcessingRequest, BatchProcessingRequest, OCRResult
from ocular.core.enums import ProviderType, ProcessingStrategy
from ocular.core.exceptions import ValidationError


class TestValidationService:
    """Test ValidationService class."""
    
    @pytest.fixture
    def validation_service(self, test_settings):
        """Create validation service with test settings."""
        return ValidationService(test_settings)
    
    @pytest.mark.asyncio
    async def test_validate_valid_processing_request(self, validation_service, sample_image_path):
        """Test validating valid processing request."""
        request = ProcessingRequest(
            file_path=sample_image_path,
            strategy=ProcessingStrategy.SINGLE,
            providers=[ProviderType.MISTRAL],
            prompt="Extract text"
        )
        
        result = await validation_service.validate_processing_request(request)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, validation_service, invalid_file_path):
        """Test validation fails for non-existent file."""
        request = ProcessingRequest(
            file_path=invalid_file_path,
            providers=[ProviderType.MISTRAL]
        )
        
        with pytest.raises(ValidationError, match="File does not exist"):
            await validation_service.validate_processing_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_large_file(self, validation_service, large_file_path):
        """Test validation fails for oversized file."""
        request = ProcessingRequest(
            file_path=large_file_path,
            providers=[ProviderType.MISTRAL]
        )
        
        with pytest.raises(ValidationError, match="exceeds limit"):
            await validation_service.validate_processing_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_unsupported_extension(self, validation_service, unsupported_file_path):
        """Test validation fails for unsupported file extension."""
        request = ProcessingRequest(
            file_path=unsupported_file_path,
            providers=[ProviderType.MISTRAL]
        )
        
        with pytest.raises(ValidationError, match="not allowed"):
            await validation_service.validate_processing_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_empty_providers(self, validation_service, sample_image_path):
        """Test validation fails for empty providers list."""
        request = ProcessingRequest(
            file_path=sample_image_path,
            providers=[]
        )
        
        with pytest.raises(ValidationError, match="At least one provider"):
            await validation_service.validate_processing_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_long_prompt(self, validation_service, sample_image_path):
        """Test validation fails for overly long prompt."""
        long_prompt = "x" * 2001  # Exceeds 2000 character limit
        
        request = ProcessingRequest(
            file_path=sample_image_path,
            providers=[ProviderType.MISTRAL],
            prompt=long_prompt
        )
        
        with pytest.raises(ValidationError, match="Prompt too long"):
            await validation_service.validate_processing_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_batch_request(self, validation_service, sample_image_path, sample_pdf_path):
        """Test validating valid batch request."""
        request = BatchProcessingRequest(
            file_paths=[sample_image_path, sample_pdf_path],
            strategy=ProcessingStrategy.FALLBACK,
            providers=[ProviderType.MISTRAL],
            max_concurrent=2
        )
        
        result = await validation_service.validate_batch_request(request)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_empty_batch(self, validation_service):
        """Test validation fails for empty batch."""
        request = BatchProcessingRequest(
            file_paths=[],
            providers=[ProviderType.MISTRAL]
        )
        
        with pytest.raises(ValidationError, match="At least one file path"):
            await validation_service.validate_batch_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_large_batch(self, validation_service, sample_image_path):
        """Test validation fails for oversized batch."""
        # Create batch with too many files
        file_paths = [sample_image_path] * 101  # Exceeds limit of 100
        
        request = BatchProcessingRequest(
            file_paths=file_paths,
            providers=[ProviderType.MISTRAL]
        )
        
        with pytest.raises(ValidationError, match="Too many files in batch"):
            await validation_service.validate_batch_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_invalid_concurrent_limit(self, validation_service, sample_image_path):
        """Test validation fails for invalid concurrency limit."""
        request = BatchProcessingRequest(
            file_paths=[sample_image_path],
            providers=[ProviderType.MISTRAL],
            max_concurrent=15  # Exceeds limit of 10
        )
        
        with pytest.raises(ValidationError, match="max_concurrent must be between"):
            await validation_service.validate_batch_request(request)
    
    def test_validate_ocr_result(self, validation_service):
        """Test validating OCR result."""
        result = OCRResult(
            text="Sample text",
            confidence=0.95,
            provider="test_provider",
            processing_time=1.5
        )
        
        is_valid = validation_service.validate_ocr_result(result)
        assert is_valid is True
    
    def test_validate_invalid_confidence(self, validation_service):
        """Test validation fails for invalid confidence."""
        result = OCRResult(
            text="Sample text",
            confidence=1.5,  # Invalid - over 1.0
            provider="test_provider"
        )
        
        with pytest.raises(ValidationError, match="Confidence must be between"):
            validation_service.validate_ocr_result(result)
    
    def test_validate_negative_processing_time(self, validation_service):
        """Test validation fails for negative processing time."""
        result = OCRResult(
            text="Sample text",
            provider="test_provider",
            processing_time=-1.0  # Invalid - negative
        )
        
        with pytest.raises(ValidationError, match="cannot be negative"):
            validation_service.validate_ocr_result(result)
    
    def test_validate_structured_data_schema(self, validation_service):
        """Test validating structured data schema."""
        valid_schema = {
            "invoice_number": "string",
            "date": "string",
            "total_amount": "number"
        }
        
        result = validation_service.validate_structured_data_schema(valid_schema)
        assert result is True
    
    def test_validate_empty_schema(self, validation_service):
        """Test validation fails for empty schema."""
        with pytest.raises(ValidationError, match="Schema cannot be empty"):
            validation_service.validate_structured_data_schema({})
    
    def test_validate_complex_schema(self, validation_service):
        """Test validation fails for overly complex schema."""
        complex_schema = {f"field_{i}": "string" for i in range(51)}  # 51 fields
        
        with pytest.raises(ValidationError, match="Schema too complex"):
            validation_service.validate_structured_data_schema(complex_schema)
    
    def test_validate_invalid_field_name(self, validation_service):
        """Test validation fails for invalid field names."""
        invalid_schema = {
            "123_invalid": "string",  # Starts with number
            "valid-field": "string"   # Contains dash
        }
        
        with pytest.raises(ValidationError, match="Invalid field name"):
            validation_service.validate_structured_data_schema(invalid_schema)
    
    def test_validate_upload_file(self, validation_service):
        """Test validating uploaded file."""
        content = b"fake image content"
        filename = "test_image.jpg"
        
        result = validation_service.validate_upload_file(content, filename)
        assert result is True
    
    def test_validate_empty_upload(self, validation_service):
        """Test validation fails for empty upload."""
        with pytest.raises(ValidationError, match="File content cannot be empty"):
            validation_service.validate_upload_file(b"", "test.jpg")
    
    def test_validate_unsafe_filename(self, validation_service):
        """Test validation fails for unsafe filename."""
        content = b"test content"
        unsafe_filename = "../../../etc/passwd"
        
        with pytest.raises(ValidationError, match="Unsafe filename"):
            validation_service.validate_upload_file(content, unsafe_filename)
    
    def test_sanitize_prompt(self, validation_service):
        """Test prompt sanitization."""
        # Normal prompt
        normal = "Extract text from this document"
        sanitized = validation_service.sanitize_prompt(normal)
        assert sanitized == normal
        
        # Prompt with extra whitespace
        messy = "  Extract   text  from    document  "
        sanitized = validation_service.sanitize_prompt(messy)
        assert sanitized == "Extract text from document"
        
        # Overly long prompt gets truncated
        long_prompt = "x" * 2500
        sanitized = validation_service.sanitize_prompt(long_prompt)
        assert len(sanitized) == 2000
    
    def test_validate_processing_options(self, validation_service):
        """Test validating processing options."""
        options = {
            "enhance_image": True,
            "beam_size": 5,
            "temperature": 0.7,
            "unknown_option": "ignored"  # Should be filtered out
        }
        
        validated = validation_service.validate_processing_options(options)
        
        assert validated["enhance_image"] is True
        assert validated["beam_size"] == 5
        assert validated["temperature"] == 0.7
        assert "unknown_option" not in validated
    
    def test_validate_api_request_size(self, validation_service):
        """Test API request size validation."""
        # Valid size
        result = validation_service.validate_api_request_size(1024 * 1024)  # 1MB
        assert result is True
        
        # Oversized request
        large_size = 100 * 1024 * 1024  # 100MB (exceeds default limit)
        
        with pytest.raises(ValidationError, match="Request size.*exceeds limit"):
            validation_service.validate_api_request_size(large_size)
