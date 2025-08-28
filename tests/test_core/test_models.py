"""
Tests for core models.
"""

import pytest
from datetime import datetime
from pathlib import Path

from ocular.core.models import (
    OCRResult, ProcessingRequest, ProcessingResult, 
    InvoiceData, BatchProcessingRequest
)
from ocular.core.enums import ProviderType, ProcessingStrategy, DocumentType, ProcessingStatus


class TestOCRResult:
    """Test OCRResult model."""
    
    def test_create_basic_result(self):
        """Test creating basic OCR result."""
        result = OCRResult(
            text="Sample text",
            provider="test_provider"
        )
        
        assert result.text == "Sample text"
        assert result.provider == "test_provider"
        assert result.confidence is None
        assert result.language is None
        assert result.processing_time is None
        assert result.metadata == {}
    
    def test_create_full_result(self):
        """Test creating OCR result with all fields."""
        metadata = {"key": "value"}
        
        result = OCRResult(
            text="Full text result",
            confidence=0.95,
            language="en",
            provider="full_provider",
            processing_time=1.5,
            metadata=metadata
        )
        
        assert result.text == "Full text result"
        assert result.confidence == 0.95
        assert result.language == "en"
        assert result.provider == "full_provider"
        assert result.processing_time == 1.5
        assert result.metadata == metadata
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        result = OCRResult(text="test", confidence=0.5, provider="test")
        assert result.confidence == 0.5
        
        # Invalid confidence - should raise validation error
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            OCRResult(text="test", confidence=1.5, provider="test")
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            OCRResult(text="test", confidence=-0.1, provider="test")


class TestProcessingRequest:
    """Test ProcessingRequest model."""
    
    def test_create_basic_request(self, sample_image_path):
        """Test creating basic processing request."""
        request = ProcessingRequest(
            file_path=sample_image_path
        )
        
        assert request.file_path == sample_image_path
        assert request.strategy == ProcessingStrategy.FALLBACK
        assert request.providers == [ProviderType.MISTRAL]
        assert request.prompt is None
        assert request.options == {}
    
    def test_create_full_request(self, sample_image_path):
        """Test creating processing request with all fields."""
        providers = [ProviderType.MISTRAL, ProviderType.OLM_OCR]
        options = {"enhance": True}
        
        request = ProcessingRequest(
            file_path=sample_image_path,
            strategy=ProcessingStrategy.ENSEMBLE,
            providers=providers,
            prompt="Extract all text",
            options=options
        )
        
        assert request.file_path == sample_image_path
        assert request.strategy == ProcessingStrategy.ENSEMBLE
        assert request.providers == providers
        assert request.prompt == "Extract all text"
        assert request.options == options


class TestProcessingResult:
    """Test ProcessingResult model."""
    
    def test_create_basic_result(self, sample_image_path):
        """Test creating basic processing result."""
        result = ProcessingResult(
            file_path=sample_image_path,
            document_type=DocumentType.IMAGE,
            strategy=ProcessingStrategy.SINGLE,
            status=ProcessingStatus.PENDING
        )
        
        assert result.file_path == sample_image_path
        assert result.document_type == DocumentType.IMAGE
        assert result.strategy == ProcessingStrategy.SINGLE
        assert result.status == ProcessingStatus.PENDING
        assert result.provider_results == {}
        assert result.primary_result is None
        assert result.errors == []
    
    def test_add_provider_result(self, sample_image_path):
        """Test adding provider results."""
        result = ProcessingResult(
            file_path=sample_image_path,
            document_type=DocumentType.IMAGE,
            strategy=ProcessingStrategy.ENSEMBLE,
            status=ProcessingStatus.IN_PROGRESS
        )
        
        ocr_result1 = OCRResult(
            text="First result",
            confidence=0.8,
            provider="provider1"
        )
        
        ocr_result2 = OCRResult(
            text="Second result", 
            confidence=0.9,
            provider="provider2"
        )
        
        result.add_provider_result("provider1", ocr_result1)
        result.add_provider_result("provider2", ocr_result2)
        
        assert len(result.provider_results) == 2
        assert result.provider_results["provider1"] == ocr_result1
        assert result.provider_results["provider2"] == ocr_result2
        
        # Primary result should be the one with higher confidence
        assert result.primary_result == ocr_result2
    
    def test_get_text(self, sample_image_path):
        """Test getting primary text."""
        result = ProcessingResult(
            file_path=sample_image_path,
            document_type=DocumentType.IMAGE,
            strategy=ProcessingStrategy.SINGLE,
            status=ProcessingStatus.COMPLETED
        )
        
        # No results yet
        assert result.get_text() == ""
        
        # Add result
        ocr_result = OCRResult(
            text="Extracted text",
            provider="test_provider"
        )
        result.add_provider_result("test_provider", ocr_result)
        
        assert result.get_text() == "Extracted text"
    
    def test_get_best_result(self, sample_image_path):
        """Test getting best result by confidence."""
        result = ProcessingResult(
            file_path=sample_image_path,
            document_type=DocumentType.IMAGE,
            strategy=ProcessingStrategy.ENSEMBLE,
            status=ProcessingStatus.IN_PROGRESS
        )
        
        # No results
        assert result.get_best_result() is None
        
        # Add results with different confidence
        low_conf = OCRResult(text="Low", confidence=0.6, provider="low")
        high_conf = OCRResult(text="High", confidence=0.9, provider="high")
        no_conf = OCRResult(text="None", provider="none")
        
        result.add_provider_result("low", low_conf)
        result.add_provider_result("high", high_conf)
        result.add_provider_result("none", no_conf)
        
        best = result.get_best_result()
        assert best == high_conf
    
    def test_mark_completed(self, sample_image_path):
        """Test marking result as completed."""
        result = ProcessingResult(
            file_path=sample_image_path,
            document_type=DocumentType.IMAGE,
            strategy=ProcessingStrategy.SINGLE,
            status=ProcessingStatus.IN_PROGRESS
        )
        
        # Set started_at for timing calculation
        result.started_at = datetime.now()
        
        result.mark_completed()
        
        assert result.status == ProcessingStatus.COMPLETED
        assert result.completed_at is not None
        assert result.total_processing_time is not None
        assert result.total_processing_time > 0
    
    def test_mark_failed(self, sample_image_path):
        """Test marking result as failed."""
        result = ProcessingResult(
            file_path=sample_image_path,
            document_type=DocumentType.IMAGE,
            strategy=ProcessingStrategy.SINGLE,
            status=ProcessingStatus.IN_PROGRESS
        )
        
        error_msg = "Processing failed"
        result.mark_failed(error_msg)
        
        assert result.status == ProcessingStatus.FAILED
        assert error_msg in result.errors
        assert result.completed_at is not None


class TestInvoiceData:
    """Test InvoiceData structured model."""
    
    def test_create_empty_invoice(self):
        """Test creating empty invoice data."""
        invoice = InvoiceData()
        
        assert invoice.invoice_number is None
        assert invoice.date is None
        assert invoice.total_amount is None
        assert invoice.vendor_name is None
        assert invoice.line_items == []
        assert invoice.extraction_confidence is None
        assert invoice.source_provider is None
    
    def test_create_full_invoice(self):
        """Test creating full invoice data."""
        line_items = [
            {"description": "Item 1", "quantity": 2, "price": 10.00},
            {"description": "Item 2", "quantity": 1, "price": 20.00}
        ]
        
        invoice = InvoiceData(
            invoice_number="INV-001",
            date="2024-01-15",
            total_amount=40.00,
            vendor_name="Test Vendor",
            line_items=line_items,
            extraction_confidence=0.95,
            source_provider="mistral"
        )
        
        assert invoice.invoice_number == "INV-001"
        assert invoice.date == "2024-01-15"
        assert invoice.total_amount == 40.00
        assert invoice.vendor_name == "Test Vendor"
        assert invoice.line_items == line_items
        assert invoice.extraction_confidence == 0.95
        assert invoice.source_provider == "mistral"


class TestBatchProcessingRequest:
    """Test BatchProcessingRequest model."""
    
    def test_create_batch_request(self, sample_image_path, sample_pdf_path):
        """Test creating batch processing request."""
        file_paths = [sample_image_path, sample_pdf_path]
        
        request = BatchProcessingRequest(
            file_paths=file_paths
        )
        
        assert request.file_paths == file_paths
        assert request.strategy == ProcessingStrategy.FALLBACK
        assert request.providers == [ProviderType.MISTRAL]
        assert request.max_concurrent == 3
        assert request.prompt is None
        assert request.options == {}
    
    def test_batch_request_validation(self):
        """Test batch request validation."""
        # Empty file paths should fail
        with pytest.raises(ValueError):
            BatchProcessingRequest(file_paths=[])
        
        # max_concurrent validation
        with pytest.raises(ValueError):
            BatchProcessingRequest(
                file_paths=[Path("test.jpg")],
                max_concurrent=0
            )
        
        with pytest.raises(ValueError):
            BatchProcessingRequest(
                file_paths=[Path("test.jpg")],
                max_concurrent=15
            )