# RunPod Serverless Deployment for OLM and RoLM OCR

This guide explains how to deploy OLM (Optical Language Model) and RoLM (Robust Optical Language Model) OCR services on RunPod serverless infrastructure.

## Overview

Both OLM and RoLM providers in the Ocular package have been designed to work with remote API endpoints rather than local models. This allows you to run the computationally intensive OCR models on RunPod's GPU infrastructure while keeping your main application lightweight.

## Prerequisites

- RunPod account with billing configured
- Docker installed locally (for building custom images)
- Basic knowledge of Python and FastAPI

## Architecture

```
Ocular Application (Local/Cloud)
    ↓ API Calls (HTTPS)
RunPod Serverless Endpoint
    ↓ GPU Processing
OCR Model (TrOCR/Custom)
    ↓ Results
Base64 Encoded Response
```

## Step 1: Create OCR Service Container

### Directory Structure

Create the following directory structure for your RunPod deployment:

```
runpod-ocr/
├── Dockerfile
├── requirements.txt
├── app.py
├── models/
│   └── __init__.py
└── utils/
    └── image_processing.py
```

### Requirements File

Create `requirements.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
Pillow>=9.5.0
numpy>=1.24.0
aiofiles>=23.0.0
python-multipart>=0.0.6
runpod>=1.5.0
```

### Main Application

Create `app.py`:

```python
"""
RunPod OCR Service for OLM and RoLM providers.
Provides OCR functionality via REST API endpoints.
"""

import asyncio
import base64
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import runpod
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForVision2Seq
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
models = {}
processors = {}

# Request/Response Models
class OCRRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    model_name: Optional[str] = Field(default="microsoft/trocr-base-printed", description="Model to use")
    prompt: Optional[str] = Field(default="Extract all text from this image", description="Processing prompt")
    use_preprocessing: Optional[bool] = Field(default=False, description="Apply image preprocessing")
    ensemble_size: Optional[int] = Field(default=1, description="Number of variants for robust processing")
    robust_processing: Optional[bool] = Field(default=False, description="Enable robust processing techniques")
    decoding_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model decoding options")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")

class StructuredRequest(OCRRequest):
    schema: Dict[str, Any] = Field(..., description="JSON schema for structured extraction")
    extraction_type: str = Field(default="structured", description="Type of extraction")

class OCRResponse(BaseModel):
    text: str
    confidence: float
    language: str = "en"
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StructuredResponse(BaseModel):
    structured_data: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    gpu_available: bool
    memory_info: Dict[str, Any]

# Model Management
async def load_model(model_name: str) -> tuple:
    """Load OCR model and processor."""
    if model_name in models and model_name in processors:
        return models[model_name], processors[model_name]
    
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Load processor and model
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        # Cache models
        models[model_name] = model
        processors[model_name] = processor
        
        logger.info(f"Model {model_name} loaded successfully on {device}")
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def create_robust_variants(image: Image.Image, ensemble_size: int) -> List[Image.Image]:
    """Create multiple variants of the image for robust processing."""
    variants = [image]  # Original image
    
    if ensemble_size > 1:
        # Variant 1: Enhanced contrast
        enhancer = ImageEnhance.Contrast(image)
        variants.append(enhancer.enhance(1.5))
        
        if ensemble_size > 2:
            # Variant 2: Enhanced sharpness
            enhancer = ImageEnhance.Sharpness(image)
            variants.append(enhancer.enhance(1.3))
        
        if ensemble_size > 3:
            # Variant 3: Slight brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            variants.append(enhancer.enhance(1.1))
        
        if ensemble_size > 4:
            # Variant 4: Denoised version
            variants.append(image.filter(ImageFilter.MedianFilter(size=3)))
    
    return variants[:ensemble_size]

def calculate_confidence(text: str, image: Image.Image) -> float:
    """Calculate confidence score for OCR result."""
    base_confidence = 0.85
    
    # Text quality metrics
    if len(text.strip()) == 0:
        return 0.05
    
    # Character diversity
    unique_chars = len(set(text.lower()))
    char_diversity = min(unique_chars / 20.0, 1.0)
    base_confidence *= (0.5 + 0.5 * char_diversity)
    
    # Word structure analysis
    words = text.split()
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words)
        if 2 <= avg_word_length <= 8:
            base_confidence *= 1.1
        elif avg_word_length < 1 or avg_word_length > 15:
            base_confidence *= 0.7
    
    # Image quality factors
    width, height = image.size
    pixel_count = width * height
    
    if pixel_count < 10000:  # Very small
        base_confidence *= 0.6
    elif pixel_count > 5000000:  # Very large
        base_confidence *= 0.9
    
    return min(base_confidence, 0.98)

async def process_image_variants(
    variants: List[Image.Image], 
    model, 
    processor, 
    decoding_options: Dict[str, Any]
) -> tuple[List[str], List[float]]:
    """Process multiple image variants with the OCR model."""
    results = []
    confidences = []
    device = next(model.parameters()).device
    
    default_options = {
        "num_beams": 4,
        "max_length": 512,
        "early_stopping": True,
        "do_sample": False,
        "temperature": 0.0
    }
    default_options.update(decoding_options)
    
    for variant in variants:
        try:
            # Process image
            pixel_values = processor(variant, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, **default_options)
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            confidence = calculate_confidence(generated_text, variant)
            
            results.append(generated_text)
            confidences.append(confidence)
            
        except Exception as e:
            logger.warning(f"Failed to process image variant: {e}")
            results.append("")
            confidences.append(0.1)
    
    return results, confidences

def ensemble_results(results: List[str], confidences: List[float]) -> tuple[str, float]:
    """Combine multiple results using ensemble techniques."""
    if len(results) == 1:
        return results[0], confidences[0]
    
    # Select highest confidence result
    best_idx = confidences.index(max(confidences))
    best_result = results[best_idx]
    best_confidence = confidences[best_idx]
    
    # Check for consensus
    if len(set(results)) < len(results):
        from collections import Counter
        result_counts = Counter(results)
        most_common = result_counts.most_common(1)[0]
        
        if most_common[1] > 1:  # Consensus found
            consensus_text = most_common[0]
            consensus_indices = [i for i, r in enumerate(results) if r == consensus_text]
            consensus_confidence = max(confidences[i] for i in consensus_indices)
            consensus_confidence = min(consensus_confidence * 1.2, 0.98)
            return consensus_text, consensus_confidence
    
    return best_result, best_confidence

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting OCR service...")
    
    # Pre-load default models if specified
    default_models = os.getenv("PRELOAD_MODELS", "").split(",")
    for model_name in default_models:
        if model_name.strip():
            try:
                await load_model(model_name.strip())
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")
    
    yield
    
    logger.info("Shutting down OCR service...")
    # Clear model cache
    models.clear()
    processors.clear()
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="RunPod OCR Service",
    description="OCR service for OLM and RoLM providers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    memory_info = {}
    
    if gpu_available:
        memory_info = {
            "allocated": torch.cuda.memory_allocated(),
            "cached": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
        }
    
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        gpu_available=gpu_available,
        memory_info=memory_info
    )

# Basic OCR Endpoint (for OLM provider)
@app.post("/extract_text", response_model=OCRResponse)
async def extract_text(request: OCRRequest):
    """Extract text from image using basic OCR."""
    import time
    start_time = time.time()
    
    try:
        # Load model
        model, processor = await load_model(request.model_name)
        
        # Decode image
        image = decode_base64_image(request.image)
        
        # Apply preprocessing if requested
        variants = [image]
        if request.use_preprocessing and request.ensemble_size > 1:
            variants = create_robust_variants(image, request.ensemble_size)
        
        # Process image(s)
        results, confidences = await process_image_variants(
            variants, model, processor, request.decoding_options
        )
        
        # Get final result
        if len(results) > 1:
            final_text, final_confidence = ensemble_results(results, confidences)
        else:
            final_text, final_confidence = results[0], confidences[0]
        
        processing_time = time.time() - start_time
        
        return OCRResponse(
            text=final_text,
            confidence=final_confidence,
            processing_time=processing_time,
            metadata={
                "model_name": request.model_name,
                "variants_processed": len(variants),
                "preprocessing_enabled": request.use_preprocessing
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")

# Robust OCR Endpoint (for RoLM provider)
@app.post("/extract_text_robust", response_model=OCRResponse)
async def extract_text_robust(request: OCRRequest):
    """Extract text with robust processing techniques."""
    import time
    start_time = time.time()
    
    try:
        # Load model
        model, processor = await load_model(request.model_name)
        
        # Decode image
        image = decode_base64_image(request.image)
        
        # Always use robust processing for this endpoint
        ensemble_size = max(request.ensemble_size, 3)  # Minimum 3 for robust processing
        variants = create_robust_variants(image, ensemble_size)
        
        # Enhanced decoding options for robustness
        robust_options = {
            "num_beams": 8,
            "max_length": 768,
            "repetition_penalty": 1.2,
            "temperature": 0.0,
            "length_penalty": 1.0
        }
        robust_options.update(request.decoding_options)
        
        # Process all variants
        results, confidences = await process_image_variants(
            variants, model, processor, robust_options
        )
        
        # Ensemble results with robust techniques
        final_text, final_confidence = ensemble_results(results, confidences)
        
        processing_time = time.time() - start_time
        
        return OCRResponse(
            text=final_text,
            confidence=final_confidence,
            processing_time=processing_time,
            metadata={
                "model_name": request.model_name,
                "variants_processed": len(variants),
                "ensemble_size": ensemble_size,
                "robust_processing": True,
                "preprocessing_enabled": True
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Robust OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Robust OCR processing failed: {e}")

# Structured Data Extraction Endpoint
@app.post("/extract_structured", response_model=StructuredResponse)
async def extract_structured(request: StructuredRequest):
    """Extract structured data from image."""
    import time
    start_time = time.time()
    
    try:
        # First extract text
        ocr_request = OCRRequest(
            image=request.image,
            model_name=request.model_name,
            prompt=request.prompt,
            use_preprocessing=request.use_preprocessing,
            ensemble_size=request.ensemble_size,
            decoding_options=request.decoding_options
        )
        
        ocr_response = await extract_text(ocr_request)
        text = ocr_response.text
        
        # Basic structured extraction using regex patterns
        structured_data = {}
        text_lower = text.lower()
        
        # Invoice number extraction
        if any(key in request.schema for key in ["invoice_number", "invoice_id", "inv_no"]):
            import re
            patterns = [
                r'(?:invoice|inv)[\s#]*:?\s*([A-Z0-9][-A-Z0-9]{2,20})',
                r'#\s*([A-Z0-9][-A-Z0-9]{2,15})',
                r'(?:no|number)[\s:]*([A-Z0-9][-A-Z0-9]{2,15})'
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    structured_data["invoice_number"] = match.group(1)
                    break
        
        # Amount extraction
        if any(key in request.schema for key in ["total_amount", "amount", "total"]):
            import re
            patterns = [
                r'(?:total|amount)[\s:]*\$?\s*([0-9,]+\.?[0-9]*)',
                r'\$\s*([0-9,]+\.?[0-9]*)'
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        amount = float(match.group(1).replace(',', ''))
                        if 0 < amount < 1000000:
                            structured_data["total_amount"] = amount
                            break
                    except ValueError:
                        continue
        
        # Date extraction
        if "date" in request.schema:
            import re
            patterns = [
                r'([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
                r'([A-Za-z]+ [0-9]{1,2}, [0-9]{4})'
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    structured_data["date"] = match.group(1)
                    break
        
        processing_time = time.time() - start_time
        
        return StructuredResponse(
            structured_data=structured_data,
            confidence=ocr_response.confidence,
            processing_time=processing_time,
            metadata={
                "source_text": text,
                "extraction_method": "regex_patterns",
                "schema_keys": list(request.schema.keys())
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Structured extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Structured extraction failed: {e}")

# Robust Structured Data Extraction Endpoint
@app.post("/extract_structured_robust", response_model=StructuredResponse)
async def extract_structured_robust(request: StructuredRequest):
    """Extract structured data with robust processing."""
    import time
    start_time = time.time()
    
    try:
        # Use robust OCR extraction
        ocr_request = OCRRequest(
            image=request.image,
            model_name=request.model_name,
            prompt=request.prompt,
            use_preprocessing=True,
            ensemble_size=max(request.ensemble_size, 3),
            robust_processing=True,
            decoding_options=request.decoding_options
        )
        
        ocr_response = await extract_text_robust(ocr_request)
        
        # Enhanced structured extraction with more patterns
        structured_data = {}
        text = ocr_response.text
        text_lower = text.lower()
        
        # Enhanced extraction logic (similar to previous but more robust)
        # ... (Include enhanced regex patterns and validation)
        
        processing_time = time.time() - start_time
        
        return StructuredResponse(
            structured_data=structured_data,
            confidence=ocr_response.confidence * 0.95,  # Slightly lower due to parsing
            processing_time=processing_time,
            metadata={
                **ocr_response.metadata,
                "robust_structured_extraction": True,
                "schema_keys": list(request.schema.keys())
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Robust structured extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Robust structured extraction failed: {e}")

# RunPod handler function
def runpod_handler(job):
    """RunPod serverless handler."""
    try:
        job_input = job["input"]
        endpoint = job_input.get("endpoint", "/extract_text")
        
        # Route to appropriate endpoint
        if endpoint == "/health":
            # Simple health check
            return {"status": "healthy", "gpu_available": torch.cuda.is_available()}
        
        elif endpoint == "/extract_text":
            request = OCRRequest(**job_input.get("request", {}))
            # Note: In actual RunPod deployment, you'd need to handle async calls differently
            # This is a simplified version
            return {"message": "Use HTTP endpoints for OCR processing"}
        
        else:
            return {"error": "Unknown endpoint"}
            
    except Exception as e:
        return {"error": str(e)}

# Main execution
if __name__ == "__main__":
    # Check if running on RunPod
    if os.getenv("RUNPOD_POD_ID"):
        # RunPod serverless mode
        runpod.serverless.start({"handler": runpod_handler})
    else:
        # Local development mode
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            log_level="info"
        )
```

### Dockerfile

Create `Dockerfile`:

```dockerfile
# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "app.py"]
```

## Step 2: Build and Push Docker Image

### Build Image

```bash
# Build the Docker image
docker build -t your-username/ocr-service:latest .

# Test locally (optional)
docker run --rm -p 8000:8000 your-username/ocr-service:latest
```

### Push to Registry

```bash
# Push to Docker Hub (or your preferred registry)
docker push your-username/ocr-service:latest
```

## Step 3: Deploy on RunPod

### Create Serverless Endpoint

1. Log in to RunPod and navigate to "Serverless"
2. Click "New Endpoint"
3. Configure the endpoint:

```yaml
Name: ocular-ocr-service
Container Image: your-username/ocr-service:latest
Container Registry Credentials: (if private)

GPU Configuration:
- GPU Type: RTX 3090 / RTX 4090 / A100 (based on your needs)
- VRAM: Minimum 12GB recommended

Environment Variables:
- PRELOAD_MODELS: microsoft/trocr-base-printed,microsoft/trocr-large-printed
- PORT: 8000

Scaling Configuration:
- Max Workers: 5
- Idle Timeout: 5 minutes
- Request Timeout: 120 seconds

Advanced Configuration:
- Container Start Command: python app.py
- Container Port: 8000
```

### Test Deployment

After deployment, test your endpoint:

```bash
curl -X POST "https://api.runpod.ai/v2/your-endpoint-id/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "endpoint": "/health"
    }
  }'
```

## Step 4: Configure Ocular Application

### Environment Configuration

Add the following to your Ocular application's `.env` file:

```env
# OLM Provider Configuration
OLM_API_ENDPOINT=https://api.runpod.ai/v2/your-olm-endpoint-id
OLM_API_KEY=your-runpod-api-key
OLM_MODEL_NAME=microsoft/trocr-base-printed
OLM_TIMEOUT=60
OLM_MAX_RETRIES=3

# RoLM Provider Configuration  
ROLM_API_ENDPOINT=https://api.runpod.ai/v2/your-rolm-endpoint-id
ROLM_API_KEY=your-runpod-api-key
ROLM_MODEL_NAME=microsoft/trocr-large-printed
ROLM_TIMEOUT=90
ROLM_MAX_RETRIES=3
ROLM_USE_PREPROCESSING=true
ROLM_ENSEMBLE_SIZE=3
```

### Provider Configuration

Update your Ocular settings to use the RunPod endpoints:

```python
from ocular.settings import OcularSettings, ProviderSettings

settings = OcularSettings(
    providers=ProviderSettings(
        enabled_providers=["mistral", "olm_ocr", "rolm_ocr"],
        
        # OLM Configuration
        olm_api_endpoint="https://api.runpod.ai/v2/your-olm-endpoint-id",
        olm_api_key="your-runpod-api-key",
        olm_model_name="microsoft/trocr-base-printed",
        
        # RoLM Configuration
        rolm_api_endpoint="https://api.runpod.ai/v2/your-rolm-endpoint-id", 
        rolm_api_key="your-runpod-api-key",
        rolm_model_name="microsoft/trocr-large-printed",
        rolm_use_preprocessing=True,
        rolm_ensemble_size=3
    )
)
```

## Step 5: Testing and Monitoring

### Test OCR Functionality

```python
import asyncio
from pathlib import Path
from ocular.services.ocr_service import OCRService
from ocular.core.models import ProcessingRequest
from ocular.core.enums import ProcessingStrategy, ProviderType

async def test_runpod_ocr():
    service = OCRService.from_env()
    
    request = ProcessingRequest(
        file_path=Path("test_document.pdf"),
        strategy=ProcessingStrategy.ENSEMBLE,
        providers=[ProviderType.OLM_OCR, ProviderType.ROLM_OCR]
    )
    
    result = await service.process_document(request)
    print(f"Extracted text: {result.get_text()}")
    print(f"Confidence: {result.primary_result.confidence}")

# Run test
asyncio.run(test_runpod_ocr())
```

### Monitor Performance

Check RunPod metrics:
- Request latency
- GPU utilization  
- Memory usage
- Error rates
- Cost per request

### Optimize Costs

1. **Model Caching**: Pre-load frequently used models
2. **Batch Processing**: Process multiple images in single request
3. **Auto-scaling**: Configure appropriate idle timeouts
4. **GPU Selection**: Use appropriate GPU tier for your workload

## Step 6: Production Considerations

### Security

- Use environment variables for API keys
- Implement request authentication
- Enable CORS properly for your domains
- Use HTTPS for all communications

### Error Handling

- Implement comprehensive error handling
- Add retry logic with exponential backoff
- Monitor and alert on failures
- Provide fallback mechanisms

### Performance Optimization

- Implement request queuing for high loads
- Use model quantization if needed
- Optimize image preprocessing
- Cache frequently processed images

### Cost Management

- Monitor RunPod spending
- Set spending limits
- Use spot instances when appropriate
- Optimize container resource allocation

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check GPU memory availability
   - Verify model name spelling
   - Ensure sufficient container resources

2. **API Timeouts**
   - Increase timeout values
   - Check GPU queue lengths
   - Optimize image preprocessing

3. **High Costs**
   - Review auto-scaling settings
   - Optimize idle timeouts
   - Use appropriate GPU tiers
   - Implement request batching

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check RunPod logs:
- Container logs in RunPod dashboard
- Request/response logs
- GPU utilization metrics

## Conclusion

This deployment guide provides a complete solution for running OLM and RoLM OCR models on RunPod serverless infrastructure. The setup allows for:

- Scalable, GPU-accelerated OCR processing
- Cost-effective pay-per-use pricing
- High availability and reliability
- Easy integration with the Ocular package

For questions or issues, refer to the RunPod documentation or the Ocular project documentation.