"""
Robust logging configuration with Logfire integration for Ocular OCR system.
"""

import os
import sys
import logging
import functools
import asyncio
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from datetime import datetime

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

from .exceptions import OcularError


class LogfireHandler(logging.Handler):
    """Custom logging handler that sends logs to Logfire."""
    
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.logfire_available = LOGFIRE_AVAILABLE
        
    def emit(self, record: logging.LogRecord):
        """Emit a log record to Logfire."""
        if not self.logfire_available:
            return
            
        try:
            # Format the record
            message = self.format(record)
            
            # Convert log level to Logfire level
            level_mapping = {
                logging.DEBUG: 'debug',
                logging.INFO: 'info', 
                logging.WARNING: 'warn',
                logging.ERROR: 'error',
                logging.CRITICAL: 'error'
            }
            
            level = level_mapping.get(record.levelno, 'info')
            
            # Extract additional context
            extra = {}
            if hasattr(record, 'provider'):
                extra['provider'] = record.provider
            if hasattr(record, 'file_path'):
                extra['file_path'] = record.file_path
            if hasattr(record, 'processing_time'):
                extra['processing_time'] = record.processing_time
            if hasattr(record, 'user_id'):
                extra['user_id'] = record.user_id
            if hasattr(record, 'request_id'):
                extra['request_id'] = record.request_id
            
            # Add standard fields
            extra.update({
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            })
            
            # Send to Logfire
            if level == 'debug':
                logfire.debug(message, **extra)
            elif level == 'info':
                logfire.info(message, **extra)
            elif level == 'warn':
                logfire.warn(message, **extra)
            elif level == 'error':
                logfire.error(message, **extra)
                
        except Exception as e:
            # Fallback to standard error output if Logfire fails
            print(f"Logfire logging error: {e}", file=sys.stderr)


class OcularLogger:
    """Enhanced logger for Ocular OCR system with Logfire integration."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(level)
    
    def _setup_handlers(self, level: str):
        """Set up logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Enhanced formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Logfire handler
        if LOGFIRE_AVAILABLE:
            logfire_handler = LogfireHandler(getattr(logging, level.upper()))
            logfire_handler.setFormatter(formatter)
            self.logger.addHandler(logfire_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exc_info=None, **kwargs):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def critical(self, message: str, exc_info=None, **kwargs):
        """Log critical message."""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)
    
    def log_ocr_request(self, file_path: str, provider: str, request_id: Optional[str] = None):
        """Log OCR processing request."""
        self.info(
            f"Starting OCR processing",
            file_path=file_path,
            provider=provider,
            request_id=request_id
        )
    
    def log_ocr_result(self, file_path: str, provider: str, success: bool, 
                       processing_time: Optional[float] = None, confidence: Optional[float] = None,
                       text_length: Optional[int] = None, request_id: Optional[str] = None):
        """Log OCR processing result."""
        status = "completed" if success else "failed"
        self.info(
            f"OCR processing {status}",
            file_path=file_path,
            provider=provider,
            processing_time=processing_time,
            confidence=confidence,
            text_length=text_length,
            request_id=request_id
        )
    
    def log_api_call(self, endpoint: str, method: str, status_code: Optional[int] = None,
                     response_time: Optional[float] = None, request_id: Optional[str] = None):
        """Log API call."""
        self.info(
            f"API call: {method} {endpoint}",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            request_id=request_id
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                  request_id: Optional[str] = None):
        """Log error with full context."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "request_id": request_id
        }
        
        if isinstance(error, OcularError):
            error_info.update({
                "error_code": error.error_code,
                "error_details": error.details
            })
        
        if context:
            error_info.update(context)
        
        self.error(
            f"Error occurred: {error}",
            exc_info=True,
            **error_info
        )


# Global logger instances
_loggers: Dict[str, OcularLogger] = {}


def get_logger(name: str, level: str = "INFO") -> OcularLogger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = OcularLogger(name, level)
    return _loggers[name]


def setup_logfire(
    token: Optional[str] = None,
    service_name: str = "ocular-ocr",
    service_version: str = "0.1.0",
    environment: str = "production"
):
    """Set up Logfire configuration."""
    if not LOGFIRE_AVAILABLE:
        print("Warning: Logfire not available. Install with: pip install logfire")
        return
    
    try:
        # Get token from environment if not provided
        if not token:
            token = os.getenv("LOGFIRE_TOKEN")
        
        if not token:
            print("Warning: No Logfire token provided. Set LOGFIRE_TOKEN environment variable.")
            return
        
        # Configure Logfire
        logfire.configure(
            token=token,
            service_name=service_name,
            service_version=service_version,
            environment=environment,
            # Additional configuration
            send_to_logfire=True,
            console=False,  # We handle console logging separately
        )
        
        print(f"Logfire configured for {service_name} v{service_version} in {environment}")
        
    except Exception as e:
        print(f"Failed to configure Logfire: {e}")


def log_function_call(logger_name: Optional[str] = None):
    """Decorator to log function calls with Logfire."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            # Log function entry
            logger.debug(
                f"Entering {func.__name__}",
                function=func.__name__,
                module=func.__module__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(
                    f"Completed {func.__name__}",
                    function=func.__name__,
                    execution_time=execution_time,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # Log exception
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Exception in {func.__name__}: {e}",
                    function=func.__name__,
                    execution_time=execution_time,
                    success=False,
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_async_function_call(logger_name: Optional[str] = None):
    """Decorator to log async function calls with Logfire."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            # Log function entry
            logger.debug(
                f"Entering async {func.__name__}",
                function=func.__name__,
                module=func.__module__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            start_time = datetime.now()
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful completion
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(
                    f"Completed async {func.__name__}",
                    function=func.__name__,
                    execution_time=execution_time,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # Log exception
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Exception in async {func.__name__}: {e}",
                    function=func.__name__,
                    execution_time=execution_time,
                    success=False,
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Initialize default Logfire configuration
if LOGFIRE_AVAILABLE:
    setup_logfire(
        environment=os.getenv("ENVIRONMENT", "development")
    )