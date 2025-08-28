"""
FastAPI web application for Ocular OCR service.
"""

import asyncio
import json
import tempfile
import uuid
import os
from pathlib import Path
from typing import List, Optional

# Load environment variables first
try:
    from dotenv import load_dotenv
    # Look for .env file in project root (parent directory of web/)
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from {env_path}")
    else:
        load_dotenv()  # Try default locations
        print("Loaded .env from default location")
except ImportError:
    print("Warning: python-dotenv not available, environment variables may not load")
except Exception as e:
    print(f"Warning: Error loading .env file: {e}")

# Debug: Check if MISTRAL_API_KEY is loaded
mistral_key = os.getenv("MISTRAL_API_KEY")
print(f"MISTRAL_API_KEY loaded: {'Yes' if mistral_key else 'No'}")

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ocular import UnifiedDocumentProcessor, ProcessingStrategy, ProviderType
from ocular.providers.settings import OcularSettings
from ocular.exceptions import DocumentProcessingError, ConfigurationError


class ProcessingRequest(BaseModel):
    """Request model for OCR processing."""
    strategy: str = "fallback"
    providers: List[str] = ["mistral"]
    prompt: Optional[str] = None


class ProcessingResponse(BaseModel):
    """Response model for OCR processing."""
    success: bool
    results: List[dict]
    errors: List[str] = []
    processing_time: Optional[float] = None


app = FastAPI(
    title="Ocular OCR Service",
    description="Extract text and data from PDFs and images using multiple OCR providers",
    version="0.1.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Global processor instance
processor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the OCR processor on startup."""
    global processor
    try:
        print("Initializing OCR processor...")
        
        # Create config
        config = OcularSettings()
        print(f"Config created with Mistral API key: {'Set' if config.providers.mistral_api_key else 'Not set'}")
        
        # Create processor
        processor = UnifiedDocumentProcessor(config)
        print("Processor created successfully")
        
        # Get available providers (this will trigger lazy initialization)
        await processor._ensure_providers_initialized()
        available_providers = processor.get_available_providers()
        print(f"Available OCR providers: {available_providers}")
        
        if not available_providers:
            print("⚠️  WARNING: No OCR providers are available! Check your configuration.")
        else:
            print(f"✅ {len(available_providers)} OCR provider(s) ready")
            
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
        print("Check your .env file and ensure all required environment variables are set.")
        processor = None
    except Exception as e:
        print(f"❌ Failed to initialize OCR processor: {e}")
        import traceback
        traceback.print_exc()
        processor = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main upload form."""
    available_providers = []
    provider_stats = {}
    
    if processor:
        try:
            await processor._ensure_providers_initialized()
            available_providers = processor.get_available_providers()
            provider_stats = processor.get_provider_stats()
        except Exception as e:
            print(f"Error getting provider info: {e}")
    
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "available_providers": available_providers,
        "provider_stats": provider_stats,
        "strategies": [s.value for s in ProcessingStrategy]
    })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    available_providers = []
    if processor:
        try:
            await processor._ensure_providers_initialized()
            available_providers = processor.get_available_providers()
        except Exception as e:
            print(f"Error in health check: {e}")
    
    return {
        "status": "healthy",
        "processor_available": processor is not None,
        "available_providers": available_providers
    }


@app.post("/process", response_model=ProcessingResponse)
async def process_files(
    files: List[UploadFile] = File(...),
    strategy: str = Form("fallback"),
    providers: str = Form("mistral"),
    prompt: Optional[str] = Form(None)
):
    """Process uploaded files with OCR."""
    if not processor:
        raise HTTPException(
            status_code=500, 
            detail="OCR processor not available. Check configuration."
        )
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Parse providers
    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    if not provider_list:
        provider_list = ["mistral"]
    
    # Convert to ProviderType enums
    try:
        ocr_providers = []
        # Create a mapping from provider names to enums
        provider_mapping = {
            "mistral": ProviderType.MISTRAL,
            "olm_ocr": ProviderType.OLM_OCR,
            "rolm_ocr": ProviderType.ROLM_OCR,
            "google_vision": ProviderType.GOOGLE_VISION,
            "aws_textract": ProviderType.AWS_TEXTRACT,
            "tesseract": ProviderType.TESSERACT,
            "azure_document_intelligence": ProviderType.AZURE_DOCUMENT_INTELLIGENCE,
        }
        
        for p in provider_list:
            if p in provider_mapping:
                ocr_providers.append(provider_mapping[p])
            else:
                print(f"Warning: Unknown provider '{p}', skipping")
                
        # Default to Mistral if no valid providers found
        if not ocr_providers:
            ocr_providers = [ProviderType.MISTRAL]
            
    except Exception as e:
        print(f"Error converting providers: {e}")
        ocr_providers = [ProviderType.MISTRAL]
    
    # Parse strategy
    try:
        processing_strategy = ProcessingStrategy(strategy)
    except ValueError:
        processing_strategy = ProcessingStrategy.FALLBACK
    
    results = []
    errors = []
    total_start_time = asyncio.get_event_loop().time()
    
    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        file_paths = []
        
        # Save uploaded files
        for file in files:
            if not file.filename:
                continue
                
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix
            temp_file_path = temp_path / f"{file_id}{file_extension}"
            
            try:
                # Save file
                content = await file.read()
                with open(temp_file_path, "wb") as f:
                    f.write(content)
                
                file_paths.append({
                    "path": temp_file_path,
                    "original_name": file.filename,
                    "size": len(content)
                })
                
            except Exception as e:
                errors.append(f"Failed to save {file.filename}: {str(e)}")
        
        # Process files
        for file_info in file_paths:
            try:
                result = await processor.process_document(
                    file_info["path"],
                    strategy=processing_strategy,
                    providers=ocr_providers,
                    prompt=prompt
                )
                
                # Convert to serializable format
                result_dict = result.to_dict()
                result_dict["original_filename"] = file_info["original_name"]
                result_dict["file_size"] = file_info["size"]
                
                results.append(result_dict)
                
            except Exception as e:
                errors.append(f"Failed to process {file_info['original_name']}: {str(e)}")
    
    total_processing_time = asyncio.get_event_loop().time() - total_start_time
    
    return ProcessingResponse(
        success=len(results) > 0,
        results=results,
        errors=errors,
        processing_time=total_processing_time
    )


@app.get("/results/{result_id}")
async def get_result(result_id: str):
    """Get processing result by ID (for future implementation)."""
    # This would be implemented with a database or cache
    # For now, return a placeholder
    return {"message": "Result storage not implemented yet"}


@app.post("/batch-process")
async def batch_process_files(
    files: List[UploadFile] = File(...),
    request_data: str = Form(...)
):
    """Batch process multiple files with individual settings."""
    if not processor:
        raise HTTPException(status_code=500, detail="OCR processor not available")
    
    try:
        batch_request = json.loads(request_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request data")
    
    # Implementation for batch processing with individual settings per file
    # This would allow different strategies/providers for different files
    return {"message": "Batch processing endpoint - implementation pending"}


@app.get("/providers")
async def get_providers():
    """Get information about available OCR providers."""
    if not processor:
        return {"providers": [], "error": "Processor not available"}
    
    try:
        await processor._ensure_providers_initialized()
        available_providers = processor.get_available_providers()
        provider_stats = processor.get_provider_stats()
    except Exception as e:
        return {"providers": [], "error": f"Provider initialization failed: {e}"}
    
    return {
        "available_providers": available_providers,
        "provider_stats": provider_stats
    }


@app.get("/download/{result_id}")
async def download_result(result_id: str, format: str = "json"):
    """Download processing results in various formats."""
    # Implementation for downloading results as JSON, CSV, or text
    return {"message": "Download endpoint - implementation pending"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )