"""
Robust input validation and sanitization for Ocular OCR system.
"""

import re
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from urllib.parse import urlparse

from .exceptions import ValidationError, SecurityError
from .logging import get_logger

logger = get_logger(__name__)


class FileValidator:
    """Comprehensive file validation and security checks."""
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        'image': 50 * 1024 * 1024,  # 50MB for images
        'pdf': 100 * 1024 * 1024,   # 100MB for PDFs
    }
    
    # Allowed file types
    ALLOWED_MIME_TYPES = {
        'image/jpeg',
        'image/jpg', 
        'image/png',
        'image/bmp',
        'image/tiff',
        'image/webp',
        'application/pdf'
    }
    
    ALLOWED_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.pdf'
    }
    
    # Dangerous file patterns
    DANGEROUS_PATTERNS = [
        r'\.exe$', r'\.bat$', r'\.cmd$', r'\.scr$', r'\.pif$',
        r'\.com$', r'\.jar$', r'\.vbs$', r'\.js$', r'\.ps1$',
        r'\.sh$', r'\.py$', r'\.php$', r'\.asp$', r'\.jsp$'
    ]
    
    def __init__(self):
        self.dangerous_regex = re.compile('|'.join(self.DANGEROUS_PATTERNS), re.IGNORECASE)
    
    def validate_file(self, file_path: Union[str, Path], check_content: bool = True) -> Dict[str, Any]:
        """
        Comprehensive file validation.
        
        Args:
            file_path: Path to the file to validate
            check_content: Whether to perform content-based validation
            
        Returns:
            Validation result dictionary
            
        Raises:
            ValidationError: If file validation fails
            SecurityError: If security violations are detected
        """
        file_path = Path(file_path)
        
        logger.debug(f"Validating file: {file_path}")
        
        # Check if file exists
        if not file_path.exists():
            raise ValidationError(
                f"File does not exist: {file_path}",
                field="file_path",
                value=str(file_path)
            )
        
        # Check if it's actually a file
        if not file_path.is_file():
            raise ValidationError(
                f"Path is not a file: {file_path}",
                field="file_path",
                value=str(file_path)
            )
        
        # Get file stats
        stat = file_path.stat()
        file_size = stat.st_size
        
        # Validate file name
        self._validate_filename(file_path.name)
        
        # Validate file extension
        file_extension = file_path.suffix.lower()
        if file_extension not in self.ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"File extension not allowed: {file_extension}. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}",
                field="file_extension",
                value=file_extension
            )
        
        # Determine file type category
        file_type = 'pdf' if file_extension == '.pdf' else 'image'
        
        # Validate file size
        max_size = self.MAX_FILE_SIZES[file_type]
        if file_size > max_size:
            raise ValidationError(
                f"File size {file_size} exceeds maximum allowed size {max_size} for {file_type} files",
                field="file_size",
                value=file_size
            )
        
        # Validate MIME type if possible
        if check_content:
            self._validate_file_content(file_path, file_extension)
        
        logger.info(
            f"File validation passed: {file_path.name}",
            file_path=str(file_path),
            file_size=file_size,
            file_type=file_type
        )
        
        return {
            'valid': True,
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': file_size,
            'file_type': file_type,
            'file_extension': file_extension
        }
    
    def _validate_filename(self, filename: str):
        """Validate filename for security issues."""
        
        # Check for dangerous extensions
        if self.dangerous_regex.search(filename):
            raise SecurityError(
                f"Potentially dangerous file extension detected: {filename}",
                violation_type="dangerous_extension"
            )
        
        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            raise SecurityError(
                f"Path traversal attempt detected in filename: {filename}",
                violation_type="path_traversal"
            )
        
        # Check for null bytes
        if '\x00' in filename:
            raise SecurityError(
                f"Null byte detected in filename: {filename}",
                violation_type="null_byte_injection"
            )
        
        # Check filename length
        if len(filename) > 255:
            raise ValidationError(
                f"Filename too long: {len(filename)} characters (max 255)",
                field="filename",
                value=filename[:50] + "..."
            )
        
        # Check for empty filename
        if not filename or filename.isspace():
            raise ValidationError(
                "Filename cannot be empty",
                field="filename",
                value=filename
            )
    
    def _validate_file_content(self, file_path: Path, expected_extension: str):
        """Validate file content matches expected type."""
        try:
            # Get MIME type from file content
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            if mime_type and mime_type not in self.ALLOWED_MIME_TYPES:
                raise ValidationError(
                    f"MIME type not allowed: {mime_type}. Allowed: {', '.join(self.ALLOWED_MIME_TYPES)}",
                    field="mime_type",
                    value=mime_type
                )
            
            # Additional content validation for specific file types
            if expected_extension == '.pdf':
                self._validate_pdf_content(file_path)
            else:
                self._validate_image_content(file_path)
                
        except Exception as e:
            if isinstance(e, (ValidationError, SecurityError)):
                raise
            logger.error(f"Content validation error for {file_path}: {e}")
            raise ValidationError(
                f"Unable to validate file content: {e}",
                field="file_content"
            )
    
    def _validate_pdf_content(self, file_path: Path):
        """Validate PDF file content."""
        try:
            # Read first few bytes to check PDF magic number
            with open(file_path, 'rb') as f:
                header = f.read(8)
                
            if not header.startswith(b'%PDF-'):
                raise ValidationError(
                    "File does not appear to be a valid PDF (missing PDF header)",
                    field="pdf_header"
                )
                
        except IOError as e:
            raise ValidationError(
                f"Cannot read PDF file: {e}",
                field="file_access"
            )
    
    def _validate_image_content(self, file_path: Path):
        """Validate image file content."""
        try:
            # Read first few bytes to check image magic numbers
            with open(file_path, 'rb') as f:
                header = f.read(12)
            
            # Check common image format magic numbers
            valid_image = False
            
            if header.startswith(b'\xff\xd8\xff'):  # JPEG
                valid_image = True
            elif header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                valid_image = True
            elif header.startswith((b'BM', b'BA', b'CI', b'CP', b'IC', b'PT')):  # BMP
                valid_image = True
            elif header.startswith((b'II*\x00', b'MM\x00*')):  # TIFF
                valid_image = True
            elif header.startswith(b'RIFF') and b'WEBP' in header:  # WebP
                valid_image = True
            
            if not valid_image:
                raise ValidationError(
                    "File does not appear to be a valid image format",
                    field="image_header"
                )
                
        except IOError as e:
            raise ValidationError(
                f"Cannot read image file: {e}",
                field="file_access"
            )


class TextValidator:
    """Validate and sanitize text inputs."""
    
    # Maximum text lengths
    MAX_LENGTHS = {
        'prompt': 5000,
        'filename': 255,
        'provider_name': 100,
        'strategy': 50,
        'api_key': 500
    }
    
    # Dangerous patterns in text
    DANGEROUS_TEXT_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'vbscript:',                 # VBScript URLs
        r'data:.*base64',             # Base64 data URLs
        r'<?php.*?\?>',               # PHP code
        r'<%.*?%>',                   # ASP code
    ]
    
    def __init__(self):
        self.dangerous_text_regex = re.compile('|'.join(self.DANGEROUS_TEXT_PATTERNS), re.IGNORECASE | re.DOTALL)
    
    def validate_text(self, text: str, field_type: str, allow_empty: bool = False) -> str:
        """
        Validate and sanitize text input.
        
        Args:
            text: Text to validate
            field_type: Type of field (for length limits)
            allow_empty: Whether empty text is allowed
            
        Returns:
            Sanitized text
            
        Raises:
            ValidationError: If validation fails
            SecurityError: If security violations detected
        """
        if text is None:
            text = ""
        
        # Check if empty text is allowed
        if not allow_empty and not text.strip():
            raise ValidationError(
                f"Empty {field_type} not allowed",
                field=field_type,
                value=text
            )
        
        # Check length limits
        max_length = self.MAX_LENGTHS.get(field_type, 1000)
        if len(text) > max_length:
            raise ValidationError(
                f"{field_type} too long: {len(text)} characters (max {max_length})",
                field=field_type,
                value=text[:100] + "..."
            )
        
        # Check for dangerous patterns
        if self.dangerous_text_regex.search(text):
            raise SecurityError(
                f"Potentially dangerous content detected in {field_type}",
                violation_type="dangerous_text_pattern"
            )
        
        # Check for control characters (except common whitespace)
        if any(ord(c) < 32 and c not in '\t\n\r' for c in text):
            raise SecurityError(
                f"Control characters detected in {field_type}",
                violation_type="control_characters"
            )
        
        # Sanitize text
        sanitized = self._sanitize_text(text)
        
        logger.debug(
            f"Text validation passed for {field_type}",
            field_type=field_type,
            original_length=len(text),
            sanitized_length=len(sanitized)
        )
        
        return sanitized
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing/replacing dangerous content."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def validate_api_key(self, api_key: str, provider: str) -> str:
        """Validate API key format."""
        if not api_key:
            raise ValidationError(
                f"API key required for {provider}",
                field="api_key"
            )
        
        # Basic format validation for different providers
        if provider == "mistral":
            if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
                raise ValidationError(
                    "Invalid Mistral API key format",
                    field="api_key"
                )
        
        return api_key.strip()
    
    def validate_url(self, url: str, field_name: str = "url") -> str:
        """Validate URL format and security."""
        if not url:
            raise ValidationError(
                f"URL cannot be empty",
                field=field_name
            )
        
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(
                f"Invalid URL format: {e}",
                field=field_name,
                value=url
            )
        
        # Check scheme
        if parsed.scheme not in ('http', 'https'):
            raise ValidationError(
                f"Only HTTP(S) URLs allowed, got: {parsed.scheme}",
                field=field_name,
                value=url
            )
        
        # Check for localhost/private networks in production
        if parsed.hostname:
            if parsed.hostname.lower() in ('localhost', '127.0.0.1', '0.0.0.0'):
                logger.warning(f"Localhost URL detected: {url}")
            
            # Check for private IP ranges (basic check)
            if parsed.hostname.startswith(('10.', '192.168.', '172.')):
                logger.warning(f"Private network URL detected: {url}")
        
        return url


class ConfigValidator:
    """Validate configuration values."""
    
    def __init__(self):
        self.text_validator = TextValidator()
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entire configuration dictionary."""
        validated = {}
        
        for key, value in config.items():
            try:
                validated[key] = self.validate_config_value(key, value)
            except (ValidationError, SecurityError) as e:
                logger.error(f"Config validation failed for {key}: {e}")
                raise ValidationError(
                    f"Invalid configuration value for '{key}': {e.message}",
                    field=key,
                    value=value
                )
        
        return validated
    
    def validate_config_value(self, key: str, value: Any) -> Any:
        """Validate individual configuration value."""
        if value is None:
            return value
        
        # String validation
        if isinstance(value, str):
            # Determine field type from key name
            field_type = self._get_field_type(key)
            return self.text_validator.validate_text(value, field_type, allow_empty=True)
        
        # Numeric validation
        elif isinstance(value, (int, float)):
            return self._validate_numeric(key, value)
        
        # Boolean validation
        elif isinstance(value, bool):
            return value
        
        # List validation
        elif isinstance(value, list):
            return [self.validate_config_value(f"{key}_item", item) for item in value]
        
        # Dict validation
        elif isinstance(value, dict):
            return {k: self.validate_config_value(f"{key}_{k}", v) for k, v in value.items()}
        
        else:
            logger.warning(f"Unknown config value type for {key}: {type(value)}")
            return value
    
    def _get_field_type(self, key: str) -> str:
        """Determine field type from configuration key."""
        key_lower = key.lower()
        
        if 'api_key' in key_lower:
            return 'api_key'
        elif 'prompt' in key_lower:
            return 'prompt'
        elif 'url' in key_lower or 'endpoint' in key_lower:
            return 'filename'  # Use filename limits for URLs
        elif 'provider' in key_lower:
            return 'provider_name'
        elif 'strategy' in key_lower:
            return 'strategy'
        else:
            return 'filename'  # Default
    
    def _validate_numeric(self, key: str, value: Union[int, float]) -> Union[int, float]:
        """Validate numeric configuration values."""
        key_lower = key.lower()
        
        # Timeout validation
        if 'timeout' in key_lower:
            if value <= 0 or value > 600:  # 10 minutes max
                raise ValidationError(
                    f"Timeout must be between 0 and 600 seconds, got: {value}",
                    field=key,
                    value=value
                )
        
        # Retry validation
        elif 'retry' in key_lower or 'retries' in key_lower:
            if value < 0 or value > 10:
                raise ValidationError(
                    f"Retry count must be between 0 and 10, got: {value}",
                    field=key,
                    value=value
                )
        
        # Ensemble size validation
        elif 'ensemble' in key_lower:
            if value < 1 or value > 10:
                raise ValidationError(
                    f"Ensemble size must be between 1 and 10, got: {value}",
                    field=key,
                    value=value
                )
        
        return value


# Global validator instances
file_validator = FileValidator()
text_validator = TextValidator()
config_validator = ConfigValidator()