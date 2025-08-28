"""
Enumerations for Ocular OCR system.
"""

from enum import Enum


class ProviderType(Enum):
    """Available OCR providers."""
    MISTRAL = "mistral"
    OLM_OCR = "olm_ocr" 
    ROLM_OCR = "rolm_ocr"
    GOOGLE_VISION = "google_vision"
    AWS_TEXTRACT = "aws_textract"
    TESSERACT = "tesseract"
    AZURE_DOCUMENT_INTELLIGENCE = "azure_document_intelligence"


class ProcessingStrategy(Enum):
    """Strategy for processing with multiple providers."""
    SINGLE = "single"
    FALLBACK = "fallback"
    ENSEMBLE = "ensemble"
    BEST = "best"


class DocumentType(Enum):
    """Supported document types."""
    IMAGE = "image"
    PDF = "pdf"


class ConfidenceLevel(Enum):
    """Confidence levels for OCR results."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ProcessingStatus(Enum):
    """Processing status for documents."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"