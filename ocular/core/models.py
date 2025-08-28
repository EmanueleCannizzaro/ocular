"""
Core data models for Ocular OCR system.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, validator

from .enums import ProviderType, ProcessingStrategy, DocumentType, ProcessingStatus


class OCRResult(BaseModel):
    """Standardized OCR result format."""
    
    text: str = Field(..., description="Extracted text content")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    language: Optional[str] = Field(None, description="Detected language")
    provider: str = Field(..., description="Provider that generated this result")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class ProcessingRequest(BaseModel):
    """Request for OCR processing."""
    
    file_path: Path = Field(..., description="Path to the document to process")
    strategy: ProcessingStrategy = Field(ProcessingStrategy.FALLBACK, description="Processing strategy")
    providers: List[ProviderType] = Field(default=[ProviderType.MISTRAL], description="OCR providers to use")
    prompt: Optional[str] = Field(None, description="Custom processing prompt")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional processing options")
    
    class Config:
        use_enum_values = True


class ProcessingResult(BaseModel):
    """Result from document processing."""
    
    file_path: Path = Field(..., description="Original file path")
    document_type: DocumentType = Field(..., description="Type of document processed")
    strategy: ProcessingStrategy = Field(..., description="Strategy used for processing")
    status: ProcessingStatus = Field(..., description="Processing status")
    
    # Results from different providers
    provider_results: Dict[str, OCRResult] = Field(default_factory=dict, description="Results from each provider")
    primary_result: Optional[OCRResult] = Field(None, description="Primary/best result")
    
    # Processing metadata
    started_at: datetime = Field(default_factory=datetime.now, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    total_processing_time: Optional[float] = Field(None, ge=0.0, description="Total processing time")
    
    # Error information
    errors: List[str] = Field(default_factory=list, description="Any errors that occurred")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat()
        }
    
    def get_text(self) -> str:
        """Get the primary extracted text."""
        if self.primary_result:
            return self.primary_result.text
        if self.provider_results:
            # Return text from first available result
            return next(iter(self.provider_results.values())).text
        return ""
    
    def get_best_result(self) -> Optional[OCRResult]:
        """Get the result with highest confidence."""
        if not self.provider_results:
            return None
        
        best_result = None
        best_confidence = 0.0
        
        for result in self.provider_results.values():
            if result.confidence and result.confidence > best_confidence:
                best_confidence = result.confidence
                best_result = result
        
        return best_result or next(iter(self.provider_results.values()))
    
    def add_provider_result(self, provider: str, result: OCRResult) -> None:
        """Add a result from a provider."""
        self.provider_results[provider] = result
        
        # Update primary result if this one is better
        if not self.primary_result or (
            result.confidence and 
            (not self.primary_result.confidence or result.confidence > self.primary_result.confidence)
        ):
            self.primary_result = result
    
    def mark_completed(self) -> None:
        """Mark processing as completed."""
        self.completed_at = datetime.now()
        self.status = ProcessingStatus.COMPLETED
        if self.started_at:
            self.total_processing_time = (self.completed_at - self.started_at).total_seconds()
    
    def mark_failed(self, error: str) -> None:
        """Mark processing as failed."""
        self.status = ProcessingStatus.FAILED
        self.errors.append(error)
        self.completed_at = datetime.now()


class StructuredData(BaseModel):
    """Base class for structured data extraction."""
    
    extraction_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_provider: Optional[str] = Field(None)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)


class InvoiceData(StructuredData):
    """Structured data for invoice extraction."""
    
    invoice_number: Optional[str] = Field(None)
    date: Optional[str] = Field(None)
    due_date: Optional[str] = Field(None)
    total_amount: Optional[float] = Field(None)
    tax_amount: Optional[float] = Field(None)
    vendor_name: Optional[str] = Field(None)
    vendor_address: Optional[str] = Field(None)
    customer_name: Optional[str] = Field(None)
    customer_address: Optional[str] = Field(None)
    line_items: List[Dict[str, Any]] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    """Configuration for an OCR provider."""
    
    provider_type: ProviderType = Field(..., description="Type of provider")
    enabled: bool = Field(True, description="Whether provider is enabled")
    priority: int = Field(1, description="Provider priority (lower = higher priority)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific configuration")
    
    class Config:
        use_enum_values = True


class BatchProcessingRequest(BaseModel):
    """Request for batch processing of multiple documents."""
    
    file_paths: List[Path] = Field(..., min_items=1, description="Paths to documents to process")
    strategy: ProcessingStrategy = Field(ProcessingStrategy.FALLBACK, description="Processing strategy")
    providers: List[ProviderType] = Field(default=[ProviderType.MISTRAL], description="OCR providers to use")
    prompt: Optional[str] = Field(None, description="Custom processing prompt")
    max_concurrent: int = Field(3, ge=1, le=10, description="Maximum concurrent processing")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional processing options")
    
    class Config:
        use_enum_values = True