"""
Health and debug endpoints router for Ocular OCR application.
"""

import os
import platform
from typing import Dict, Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ocular.routers import templates
from ocular.settings import get_settings
from ocular.unified_processor import UnifiedDocumentProcessor


router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ocular-ocr"}


@router.get("/debug")
async def debug_info(request: Request):
    """Debug information endpoint"""
    try:
        # Initialize processor to check available providers
        settings = get_settings()
        processor = UnifiedDocumentProcessor(settings)

        available_providers = []
        provider_statuses = {}

        # Check each provider's availability
        from ocular.providers.factory import ProviderFactory
        factory = ProviderFactory(settings)

        for provider_type in factory.get_available_providers():
            try:
                provider = factory.create_provider(provider_type)
                available_providers.append(provider_type.value)
                provider_statuses[provider_type.value] = "available"
            except Exception as e:
                provider_statuses[provider_type.value] = f"error: {str(e)}"

        debug_data = {
            "status": "debug_info",
            "timestamp": "2024-01-01T00:00:00Z",  # You might want to use datetime.utcnow()
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "architecture": platform.architecture()[0]
            },
            "ocr_providers": {
                "available": available_providers,
                "count": len(available_providers),
                "status_details": provider_statuses
            },
            "configuration": {
                "max_file_size": "10MB",
                "supported_formats": ["PDF", "PNG", "JPEG", "JPG"],
                "timeout": "30 seconds"
            },
            "request_info": {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client": request.client.host if request.client else "unknown"
            }
        }

        return JSONResponse(content=debug_data)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Debug info failed: {str(e)}"
            }
        )