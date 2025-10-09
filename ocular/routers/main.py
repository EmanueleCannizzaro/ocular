"""Main router for Ocular OCR web application."""

from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
import httpx
import io
import os
from PIL import Image
from pathlib import Path
from pydantic import BaseModel
import tempfile
from typing import Any, Dict, List, Optional
import uuid

from ocular.core.enums import ProcessingStrategy, ProviderType
from ocular.core.exceptions import ConfigurationError, DocumentProcessingError
from ocular.routers import templates
from ocular.settings import get_olmocr_settings, get_settings
from ocular.unified_processor import UnifiedDocumentProcessor


# Router instance
router = APIRouter()

# Global processor / clients (lazy initialisation where appropriate)
processor: Optional[UnifiedDocumentProcessor] = None


def get_processor() -> UnifiedDocumentProcessor:
    """Return a shared instance of the unified processor."""
    global processor
    if processor is None:
        try:
            processor = UnifiedDocumentProcessor(get_settings())
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Failed to initialise OCR processor: {exc}")
    return processor


class ProcessingResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total_files: int
    processing_time: float
    result_id: str


class BatchProcessRequest(BaseModel):
    urls: List[str]
    strategy: str = "fallback"
    providers: List[str] = ["mistral"]
    prompt: Optional[str] = None


class AuthStatus(BaseModel):
    authenticated: bool
    user_id: Optional[str] = None
    credits_remaining: Optional[int] = None


class OLMocrResponse(BaseModel):
    success: bool
    markdown_content: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OLMocrClient:
    """Client wrapper for interacting with serverless OLMocr deployments."""

    def __init__(self) -> None:
        self.settings = get_olmocr_settings()
        self.timeout = httpx.Timeout(self.settings.timeout_seconds)

    async def process_document(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        payload = {
            "input": {
                "model": self.settings.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Convert this document to markdown format. Preserve the structure, tables, "
                                    "equations, and reading order. Output clean markdown without any additional "
                                    "commentary."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 8192,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False,
            }
        }

        headers = {
            "Authorization": f"Bearer {self.settings.active_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.settings.active_endpoint_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()

                result = response.json()

                if "output" in result and "choices" in result["output"]:
                    extracted_text = result["output"]["choices"][0]["message"]["content"]
                elif "choices" in result:
                    extracted_text = result["choices"][0]["message"]["content"]
                else:
                    extracted_text = str(result)

                return {
                    "success": True,
                    "markdown_content": extracted_text,
                    "metadata": {
                        "filename": filename,
                        "model": self.settings.model_name,
                        "provider": f"{self.settings.provider.title()} vLLM",
                    },
                }

            except httpx.TimeoutException as exc:
                raise HTTPException(status_code=408, detail="Request timeout - document processing took too long") from exc
            except httpx.HTTPStatusError as exc:
                raise HTTPException(
                    status_code=exc.response.status_code,
                    detail=f"{self.settings.provider.title()} API error: {exc.response.text}",
                ) from exc
            except Exception as exc:  # pragma: no cover - defensive
                raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc


olmocr_client = OLMocrClient()


def validate_file(file: UploadFile) -> None:
    settings = get_olmocr_settings()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}",
        )


def prepare_image_for_ocr(file_content: bytes, filename: str) -> bytes:
    try:
        if filename.lower().endswith(".pdf"):
            return file_content

        image = Image.open(io.BytesIO(file_content))

        max_dimension = max(image.size)
        if max_dimension > 1024:
            ratio = 1024 / max_dimension
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            output = io.BytesIO()
            format_map = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
            file_ext = Path(filename).suffix.lower()
            image_format = format_map.get(file_ext, "PNG")
            image.save(output, format=image_format)
            return output.getvalue()

        return file_content

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error processing image: {exc}") from exc


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def home(request: Request, tab: Optional[str] = None) -> HTMLResponse:
    """Render the unified home page containing both upload and OLMocr interfaces."""
    context: Dict[str, Any] = {"request": request}

    try:
        ocr_processor = get_processor()
        await ocr_processor._ensure_providers_initialized()
        context.update(
            {
                "available_providers": ocr_processor.get_available_providers(),
                "provider_stats": ocr_processor.get_provider_stats(),
                "strategies": [strategy.value for strategy in ProcessingStrategy],
            }
        )
    except Exception as exc:
        # Log but do not fail rendering
        print(f"⚠️ Unable to load provider metadata: {exc}")
        context.update(
            {
                "available_providers": [],
                "provider_stats": {},
                "strategies": [strategy.value for strategy in ProcessingStrategy],
            }
        )

    context["active_tab"] = tab or request.query_params.get("tab", "unified")
    return templates.TemplateResponse("index.html", context)


@router.get("/deployments", response_class=HTMLResponse)
async def deployments_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("deployments.html", {"request": request})


@router.get("/observability", response_class=HTMLResponse)
async def observability_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("observability.html", {"request": request})


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("settings.html", {"request": request})


@router.get("/models", response_class=HTMLResponse)
async def models_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("models.html", {"request": request})


@router.get("/providers", response_class=HTMLResponse)
async def providers_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("providers.html", {"request": request})


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("analytics.html", {"request": request})


@router.get("/batch", response_class=HTMLResponse)
async def batch_processing_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("batch.html", {"request": request})


@router.get("/document-history", response_class=HTMLResponse)
async def document_history_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("document_tracking.html", {"request": request})


@router.get("/olmocr", response_class=HTMLResponse)
async def olmocr_landing(request: Request) -> HTMLResponse:
    """Legacy route that opens the OLMocr tab on the unified homepage."""
    return await home(request, tab="olmocr")


# ---------------------------------------------------------------------------
# Unified OCR processing endpoints
# ---------------------------------------------------------------------------

@router.post("/process", response_model=ProcessingResponse)
async def process_files(
    files: List[UploadFile] = File(...),
    strategy: str = Form("fallback"),
    providers: str = Form("mistral"),
    prompt: Optional[str] = Form(None),
    session_info: Optional[Dict[str, Any]] = Depends(lambda: None),
) -> ProcessingResponse:
    ocr_processor = get_processor()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        processing_strategy = ProcessingStrategy(strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}") from exc

    provider_list = [provider.strip() for provider in providers.split(",") if provider.strip()]
    parsed_providers: List[ProviderType] = []
    for provider in provider_list or ["mistral"]:
        try:
            parsed_providers.append(ProviderType(provider))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}") from exc

    results: List[Dict[str, Any]] = []
    total_processing_time = 0.0
    result_id = str(uuid.uuid4())

    for file in files:
        try:
            if not file.filename:
                results.append({
                    "filename": "unknown",
                    "success": False,
                    "error": "No filename provided",
                })
                continue

            content = await file.read()
            if not content:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Empty file",
                })
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                result = await ocr_processor.process_document(
                    tmp_file_path,
                    strategy=processing_strategy,
                    providers=parsed_providers,
                    prompt=prompt,
                )

                results.append(
                    {
                        "filename": file.filename,
                        "success": True,
                        "extracted_text": result.extracted_text,
                        "processing_time": result.processing_time,
                        "provider_used": result.provider_used.value if result.provider_used else None,
                        "metadata": result.metadata,
                    }
                )
                total_processing_time += result.processing_time

            finally:
                try:
                    os.unlink(tmp_file_path)
                except OSError:
                    pass

        except DocumentProcessingError as exc:
            results.append(
                {
                    "filename": file.filename,
                    "success": False,
                    "error": str(exc),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            results.append(
                {
                    "filename": file.filename,
                    "success": False,
                    "error": f"Unexpected error: {exc}",
                }
            )

    return ProcessingResponse(
        success=any(result.get("success", False) for result in results),
        results=results,
        total_files=len(files),
        processing_time=total_processing_time,
        result_id=result_id,
    )


@router.get("/results/{result_id}")
async def get_results(result_id: str) -> Dict[str, Any]:
    return {
        "result_id": result_id,
        "status": "completed",
        "message": "Results would be stored and retrieved from database",
    }


@router.post("/batch-process")
async def batch_process_urls(
    request: BatchProcessRequest,
    session_info: Optional[Dict[str, Any]] = Depends(lambda: None),
) -> Dict[str, Any]:
    ocr_processor = get_processor()

    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    if len(request.urls) > 10:
        raise HTTPException(status_code=400, detail="Too many URLs. Maximum 10 allowed.")

    try:
        processing_strategy = ProcessingStrategy(request.strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}") from exc

    parsed_providers: List[ProviderType] = []
    for provider in request.providers:
        try:
            parsed_providers.append(ProviderType(provider))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}") from exc

    results = []
    for url in request.urls:
        results.append(
            {
                "url": url,
                "success": False,
                "error": "URL processing not implemented in this simplified version",
            }
        )

    return {
        "success": False,
        "results": results,
        "total_urls": len(request.urls),
        "message": "Batch processing is a placeholder implementation",
    }


@router.get("/api/ocr-providers/status")
async def get_available_providers() -> Dict[str, Any]:
    try:
        ocr_processor = get_processor()
        await ocr_processor._ensure_providers_initialized()

        provider_info: List[Dict[str, Any]] = []
        for provider_type, provider in ocr_processor.providers.items():
            try:
                available = await provider.is_available()
                provider_info.append(
                    {
                        "name": provider_type.value,
                        "status": "available" if available else "unavailable",
                        "description": f"{provider.provider_name} OCR provider",
                    }
                )
            except Exception as exc:
                provider_info.append(
                    {
                        "name": provider_type.value,
                        "status": "error",
                        "description": f"Error: {exc}",
                    }
                )

        return {
            "providers": provider_info,
            "total": len(provider_info),
            "available": len([provider for provider in provider_info if provider["status"] == "available"]),
        }

    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {exc}") from exc


@router.get("/download/{result_id}")
async def download_results(result_id: str) -> Dict[str, Any]:
    raise HTTPException(status_code=501, detail="Download functionality not yet implemented")


# ---------------------------------------------------------------------------
# OLMocr API endpoints
# ---------------------------------------------------------------------------

@router.post("/olmocr/file", response_model=OLMocrResponse)
async def olmocr_process_file(
    file: UploadFile = File(...),
    include_metadata: bool = Form(False),
) -> OLMocrResponse:
    try:
        validate_file(file)

        file_content = await file.read()
        settings = get_olmocr_settings()
        if len(file_content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
            )

        processed_content = prepare_image_for_ocr(file_content, file.filename)
        result = await olmocr_client.process_document(processed_content, file.filename)
        return OLMocrResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        return OLMocrResponse(success=False, markdown_content="", error=f"Processing failed: {exc}")


@router.post("/olmocr/url", response_model=OLMocrResponse)
async def olmocr_process_url(
    url: str = Form(...),
    include_metadata: bool = Form(False),
) -> OLMocrResponse:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            file_content = response.content

        filename = Path(url).name or "document.jpg"
        settings = get_olmocr_settings()

        if len(file_content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
            )

        file_ext = Path(filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}",
            )

        processed_content = prepare_image_for_ocr(file_content, filename)
        result = await olmocr_client.process_document(processed_content, filename)
        return OLMocrResponse(**result)

    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download from URL: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        return OLMocrResponse(success=False, markdown_content="", error=f"URL processing failed: {exc}")


@router.get("/olmocr/config")
async def olmocr_get_config() -> Dict[str, Any]:
    settings = get_olmocr_settings()
    return {
        "provider_configured": settings.is_configured,
        "provider_type": settings.provider,
        "max_file_size_mb": settings.max_file_size_mb,
        "supported_formats": settings.allowed_extensions,
        "timeout_seconds": settings.timeout_seconds,
        "model": settings.model_name,
        "provider": f"{settings.provider.title()} Serverless",
        "endpoint_url": settings.active_endpoint_url if settings.is_configured else None,
    }
