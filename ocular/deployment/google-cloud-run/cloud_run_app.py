"""
Google Cloud Run FastAPI Application for OLMocr with vLLM
Optimized for GPU instances with OpenAI-compatible API
"""

import os
import base64
import io
import time
import logging
from typing import Dict, List, Any, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
tokenizer = None

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="allenai/olmOCR-7B-0225-preview")
    messages: List[ChatMessage]
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


async def load_model():
    """Load the vLLM model and tokenizer"""
    global model, tokenizer

    try:
        import vllm
        from transformers import AutoTokenizer

        model_name = os.getenv("MODEL_NAME", "allenai/olmOCR-7B-0225-preview")

        logger.info(f"Loading model: {model_name}")

        # Initialize vLLM engine
        model = vllm.LLM(
            model=model_name,
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
            dtype=os.getenv("DTYPE", "bfloat16"),
            trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true",
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
            tensor_parallel_size=1,
            disable_log_stats=False,
            download_dir="/app/cache/models"
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="/app/cache/huggingface",
            trust_remote_code=True
        )

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup"""
    logger.info("Starting up - loading model...")
    await load_model()
    logger.info("Startup complete")
    yield
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="OLMocr vLLM API",
    description="OLMocr document processing with vLLM on Google Cloud Run",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files and templates (if they exist)
if os.path.exists("/app/static"):
    app.mount("/static", StaticFiles(directory="/app/static"), name="static")

if os.path.exists("/app/templates"):
    templates = Jinja2Templates(directory="/app/templates")


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "OLMocr vLLM API",
        "version": "1.0.0",
        "model": os.getenv("MODEL_NAME", "allenai/olmOCR-7B-0225-preview"),
        "status": "ready" if model is not None else "loading",
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "process_file": "/process",
            "ui": "/ui"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": int(time.time()),
        "model_name": os.getenv("MODEL_NAME", "allenai/olmOCR-7B-0225-preview")
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    model_name = os.getenv("MODEL_NAME", "allenai/olmOCR-7B-0225-preview")
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 1677610602,
                "owned_by": "allenai",
                "permission": [],
                "root": model_name,
                "parent": None
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from vllm import SamplingParams

        # Process messages to extract text and images
        prompt_text = ""
        images = []

        for message in request.messages:
            if isinstance(message.content, list):
                for content_item in message.content:
                    if isinstance(content_item, dict):
                        if content_item.get("type") == "text":
                            prompt_text += content_item.get("text", "")
                        elif content_item.get("type") == "image_url":
                            # Handle base64 images
                            image_url = content_item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image"):
                                # Decode base64 image
                                try:
                                    image_data = image_url.split(",")[1]
                                    image_bytes = base64.b64decode(image_data)
                                    image = Image.open(io.BytesIO(image_bytes))
                                    images.append(image)
                                except Exception as e:
                                    logger.warning(f"Error processing image: {e}")
            elif isinstance(message.content, str):
                prompt_text += message.content

        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["<|im_end|>"]
        )

        # Format prompt for the model
        formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

        # Generate response
        start_time = time.time()
        outputs = model.generate([formatted_prompt], sampling_params)
        processing_time = time.time() - start_time

        # Format response
        response_text = outputs[0].outputs[0].text.strip()

        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
                "total_tokens": len(tokenizer.encode(formatted_prompt)) + len(outputs[0].outputs[0].token_ids)
            }
        )

        logger.info(f"Generated response in {processing_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    prompt: str = Form("Extract all text from this document and convert it to markdown format.")
):
    """Simple file processing endpoint"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and process the uploaded file
        file_content = await file.read()

        # Convert to PIL Image
        if file.filename.lower().endswith('.pdf'):
            # Handle PDF files
            try:
                import pdf2image
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file.flush()

                    # Convert first page to image
                    pages = pdf2image.convert_from_path(
                        tmp_file.name,
                        first_page=1,
                        last_page=1,
                        dpi=200
                    )

                    if pages:
                        image = pages[0]
                    else:
                        raise ValueError("Could not convert PDF to image")

                os.unlink(tmp_file.name)

            except ImportError:
                raise HTTPException(status_code=400, detail="PDF processing not available")
        else:
            # Handle image files
            image = Image.open(io.BytesIO(file_content))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if too large
        max_dimension = 1024
        if max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Create chat completion request
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            }
        ]

        # Process with chat completions
        chat_request = ChatCompletionRequest(messages=messages)
        result = await chat_completions(chat_request)

        return {
            "filename": file.filename,
            "extracted_text": result.choices[0]["message"]["content"],
            "usage": result.usage,
            "processing_time": time.time()
        }

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    """Simple UI for testing (if templates exist)"""
    if 'templates' in globals():
        return templates.TemplateResponse("olmocr.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
            <head><title>OLMocr API</title></head>
            <body>
                <h1>OLMocr vLLM API</h1>
                <p>API is running. Use the /v1/chat/completions endpoint for processing.</p>
                <h2>Test with curl:</h2>
                <pre>
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "allenai/olmOCR-7B-0225-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract text from this image"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  }'
                </pre>
            </body>
        </html>
        """)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))

    # For local development
    if os.getenv("ENVIRONMENT") == "development":
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="info"
        )
    else:
        # For production (Cloud Run)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
