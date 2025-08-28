"""
Custom exceptions for Ocular package with enhanced error tracking.
"""

import uuid
from typing import Optional, Dict, Any
from datetime import datetime


class OcularError(Exception):
    """Base exception for Ocular package with enhanced error tracking."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None,
        retryable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "OCULAR_ERROR"
        self.details = details or {}
        self.error_id = error_id or str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.retryable = retryable
        
        # Add standard error details
        self.details.update({
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "retryable": self.retryable
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "retryable": self.retryable,
            "details": self.details
        }


class ConfigurationError(OcularError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(
            message, 
            error_code="CONFIG_ERROR",
            details=details,
            retryable=False,  # Config errors typically not retryable
            **kwargs
        )


class ValidationError(OcularError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, **kwargs):
        # Sanitize value to avoid logging sensitive data
        safe_value = str(value)[:100] + "..." if value and len(str(value)) > 100 else value
        details = {"field": field, "value": safe_value}
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            details=details,
            retryable=False,  # Validation errors not retryable without input changes
            **kwargs
        )


class OCRError(OcularError):
    """Raised when OCR processing fails."""
    
    def __init__(self, message: str, provider: Optional[str] = None, file_path: Optional[str] = None, **kwargs):
        details = {}
        if provider:
            details["provider"] = provider
        if file_path:
            details["file_path"] = file_path
        super().__init__(
            message,
            error_code="OCR_ERROR", 
            details=details,
            retryable=True,  # OCR errors might be retryable
            **kwargs
        )


class APIError(OcularError):
    """Raised when API calls fail."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        endpoint: Optional[str] = None,
        **kwargs
    ):
        # Determine if error is retryable based on status code
        retryable = status_code in [429, 500, 502, 503, 504] if status_code else True
        
        # Truncate response body to avoid logging huge responses
        safe_response = response_body[:500] + "..." if response_body and len(response_body) > 500 else response_body
        
        details = {
            "status_code": status_code,
            "response_body": safe_response,
            "endpoint": endpoint
        }
        super().__init__(
            message,
            error_code="API_ERROR",
            details=details,
            retryable=retryable,
            **kwargs
        )


class DocumentProcessingError(OcularError):
    """Raised when document processing fails."""
    
    def __init__(self, message: str, document_path: Optional[str] = None, file_size: Optional[int] = None, **kwargs):
        details = {}
        if document_path:
            details["document_path"] = document_path
        if file_size:
            details["file_size"] = file_size
        super().__init__(
            message,
            error_code="DOCUMENT_PROCESSING_ERROR",
            details=details,
            retryable=True,  # Document processing might be retryable
            **kwargs
        )


class ProviderError(OcularError):
    """Raised when provider operations fail."""
    
    def __init__(self, message: str, provider_type: Optional[str] = None, operation: Optional[str] = None, **kwargs):
        details = {}
        if provider_type:
            details["provider_type"] = provider_type
        if operation:
            details["operation"] = operation
        super().__init__(
            message,
            error_code="PROVIDER_ERROR",
            details=details,
            retryable=True,  # Provider errors often retryable
            **kwargs
        )


class TimeoutError(OcularError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, operation: Optional[str] = None, **kwargs):
        details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            details=details,
            retryable=True,  # Timeouts are often retryable
            **kwargs
        )


class RateLimitError(OcularError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, provider: Optional[str] = None, **kwargs):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        if provider:
            details["provider"] = provider
        super().__init__(
            message,
            error_code="RATE_LIMIT_ERROR",
            details=details,
            retryable=True,  # Rate limit errors are retryable after delay
            **kwargs
        )


class AuthenticationError(OcularError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        details = {"provider": provider} if provider else {}
        super().__init__(
            message,
            error_code="AUTHENTICATION_ERROR",
            details=details,
            retryable=False,  # Auth errors not retryable without fixing credentials
            **kwargs
        )


class ResourceExhaustedError(OcularError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, current_usage: Optional[str] = None, **kwargs):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if current_usage:
            details["current_usage"] = current_usage
        super().__init__(
            message,
            error_code="RESOURCE_EXHAUSTED_ERROR",
            details=details,
            retryable=True,  # Resource exhaustion might be temporary
            **kwargs
        )


class SecurityError(OcularError):
    """Raised when security violations are detected."""
    
    def __init__(self, message: str, violation_type: Optional[str] = None, **kwargs):
        details = {"violation_type": violation_type} if violation_type else {}
        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            details=details,
            retryable=False,  # Security errors should not be automatically retried
            **kwargs
        )