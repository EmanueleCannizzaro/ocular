"""
Custom exceptions for Ocular package.
"""


class OcularError(Exception):
    """Base exception for Ocular package."""
    pass


class OCRError(OcularError):
    """Raised when OCR processing fails."""
    pass


class ConfigurationError(OcularError):
    """Raised when configuration is invalid."""
    pass


class APIError(OcularError):
    """Raised when API calls fail."""
    pass


class DocumentProcessingError(OcularError):
    """Raised when document processing fails."""
    pass