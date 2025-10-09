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

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ocular import UnifiedDocumentProcessor, ProcessingStrategy, ProviderType
from ocular.providers.settings import OcularSettings
from ocular.exceptions import DocumentProcessingError, ConfigurationError
import base64
import httpx
from PIL import Image
import io

# Initialize Clerk client
clerk_secret_key = os.environ.get("CLERK_SECRET_KEY")
clerk = None

# Try to import and initialize Clerk
try:
    from clerk_backend_api import Clerk
    if clerk_secret_key:
        try:
            # Initialize Clerk client with the correct parameter for v3.3.0
            clerk = Clerk(bearer_auth=clerk_secret_key)
            print("✅ Clerk client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Clerk client: {e}")
            clerk = None
    else:
        print("⚠️ CLERK_SECRET_KEY not found - authentication will be disabled")
except ImportError:
    print("⚠️ clerk-backend-api not installed - authentication will be disabled")
    clerk = None
except Exception as e:
    print(f"❌ Failed to import/initialize Clerk: {e}")
    clerk = None

# Initialize Stripe
stripe_secret_key = os.environ.get("STRIPE_SECRET_KEY")
stripe_publishable_key = os.environ.get("STRIPE_PUBLISHABLE_KEY")
stripe_webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
stripe = None

try:
    import stripe
    if stripe_secret_key:
        try:
            stripe.api_key = stripe_secret_key
            print("✅ Stripe client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Stripe client: {e}")
            stripe.api_key = None
    else:
        print("⚠️ STRIPE_SECRET_KEY not found - payments will be disabled")
except ImportError:
    print("⚠️ stripe not installed - payments will be disabled")
    stripe = None
except Exception as e:
    print(f"❌ Failed to import/initialize Stripe: {e}")
    stripe = None

security = HTTPBearer()

async def get_session(request: Request):
    """
    Dependency to verify the Clerk session from the Authorization header.
    """
    if not clerk:
        raise HTTPException(
            status_code=503, 
            detail="Authentication service not available. CLERK_SECRET_KEY not configured."
        )
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.split(" ")[1]
    
    try:
        # Use authenticate_request method for token verification
        auth_result = clerk.authenticate_request(
            bearer_token=token
        )
        
        if not auth_result or not hasattr(auth_result, 'session_id'):
            raise HTTPException(status_code=401, detail="Invalid session")
            
        return auth_result
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

async def get_optional_session(request: Request):
    """
    Optional authentication dependency - returns session if available, None otherwise.
    """
    if not clerk:
        return None
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    
    try:
        auth_result = clerk.authenticate_request(
            bearer_token=token
        )
        return auth_result if auth_result and hasattr(auth_result, 'session_id') else None
    except Exception:
        return None

async def check_payment_status(user_id: str) -> dict:
    """Check if user has valid payment or subscription."""
    if not stripe.api_key:
        return {"has_payment": True, "type": "free"}  # Allow free usage when Stripe not configured
    
    try:
        # Check for active subscriptions
        customers = stripe.Customer.list(email=user_id, limit=1)
        if customers.data:
            customer = customers.data[0]
            subscriptions = stripe.Subscription.list(
                customer=customer.id,
                status='active',
                limit=10
            )
            
            if subscriptions.data:
                return {
                    "has_payment": True,
                    "type": "subscription",
                    "subscription_id": subscriptions.data[0].id,
                    "status": subscriptions.data[0].status
                }
        
        # Check for recent successful payment intents (last 24 hours)
        import time
        recent_payments = stripe.PaymentIntent.list(
            created={'gte': int(time.time()) - 86400},  # 24 hours ago
            limit=50
        )
        
        for payment in recent_payments.data:
            if (payment.status == 'succeeded' and 
                payment.metadata.get('user_id') == user_id):
                return {
                    "has_payment": True,
                    "type": "one_time",
                    "payment_intent_id": payment.id,
                    "amount": payment.amount
                }
        
        return {"has_payment": False, "type": "none"}
        
    except Exception as e:
        print(f"Payment status check failed: {e}")
        return {"has_payment": False, "type": "error"}

async def require_payment(auth_result = Depends(get_session)):
    """
    Dependency that requires either authentication + valid payment or subscription.
    """
    user_id = getattr(auth_result, 'user_id', None) if auth_result else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    payment_status = await check_payment_status(user_id)
    
    if not payment_status["has_payment"]:
        raise HTTPException(
            status_code=402, 
            detail="Payment required. Please purchase credits or subscribe to use this service."
        )
    
    return {"auth_result": auth_result, "payment_status": payment_status}

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

class PaymentIntentRequest(BaseModel):
    """Request model for creating a payment intent."""
    amount: int  # Amount in cents
    currency: str = "usd"
    description: Optional[str] = None
    metadata: Optional[dict] = None

class PaymentIntentResponse(BaseModel):
    """Response model for payment intent creation."""
    client_secret: str
    payment_intent_id: str
    publishable_key: Optional[str] = None

class SubscriptionRequest(BaseModel):
    """Request model for creating a subscription."""
    price_id: str
    payment_method_id: Optional[str] = None

class WebhookEvent(BaseModel):
    """Model for Stripe webhook events."""
    type: str
    data: dict

class OLMocrRequest(BaseModel):
    """Request model for OLMocr processing."""
    url: Optional[str] = None
    include_metadata: bool = False

class OLMocrResponse(BaseModel):
    """Response model for OLMocr processing."""
    success: bool
    markdown_content: str
    metadata: Optional[dict] = None
    error: Optional[str] = None


app = FastAPI(
    title="Ocular OCR Service",
    description="Extract text and data from PDFs and images using multiple OCR providers",
    version="0.1.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="ocular/templates")
# Note: Static files mounting disabled for Cloud Functions deployment
# Static assets should be served via CDN or external hosting
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Global processor instance
processor = None

# Configuration for OLMocr
class OLMocrConfig:
    RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
    RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
    TIMEOUT_SECONDS = 300  # 5 minutes

    @property
    def runpod_url(self) -> str:
        return f"https://api.runpod.ai/v2/{self.RUNPOD_ENDPOINT_ID}/run"

olmocr_config = OLMocrConfig()

class OLMocrClient:
    """Client for interacting with OLMocr via RunPod serverless (vLLM deployment)"""

    def __init__(self):
        self.timeout = httpx.Timeout(olmocr_config.TIMEOUT_SECONDS)

    async def process_document(self, image_data: bytes, filename: str) -> dict:
        """Process document using OLMocr via RunPod serverless with vLLM"""

        # Convert image to base64 for vLLM multimodal input
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Prepare the payload for RunPod serverless with vLLM
        payload = {
            "input": {
                "model": "allenai/olmOCR-7B-0225-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Convert this document to markdown format. Preserve the structure, tables, equations, and reading order. Output clean markdown without any additional commentary."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 8192,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            }
        }

        headers = {
            "Authorization": f"Bearer {olmocr_config.RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    olmocr_config.runpod_url,
                    json=payload,
                    headers=headers
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"RunPod API error: {response.text}"
                    )

                result = response.json()

                # Extract the OCR result from vLLM RunPod response
                if "output" in result:
                    output = result["output"]

                    markdown_content = ""
                    if isinstance(output, dict):
                        if "choices" in output and len(output["choices"]) > 0:
                            markdown_content = output["choices"][0]["message"]["content"]
                        elif "text" in output:
                            markdown_content = output["text"]
                        elif "generated_text" in output:
                            markdown_content = output["generated_text"]
                    elif isinstance(output, str):
                        markdown_content = output

                    if not markdown_content:
                        raise HTTPException(
                            status_code=500,
                            detail="No text content found in OCR response"
                        )

                    return {
                        "success": True,
                        "markdown_content": markdown_content.strip(),
                        "metadata": {
                            "filename": filename,
                            "processing_time": result.get("executionTime", 0),
                            "model_used": "olmocr-7b-vllm",
                            "response_format": "vllm_serverless"
                        }
                    }
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Unexpected response format from OCR service"
                    )

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=408,
                detail="OCR processing timeout. Please try with a smaller document."
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {str(e)}"
            )

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in olmocr_config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(olmocr_config.ALLOWED_EXTENSIONS)}"
        )

def prepare_image_for_ocr(file_content: bytes, filename: str) -> bytes:
    """Prepare image for OCR processing - resize if needed"""
    try:
        if filename.lower().endswith('.pdf'):
            return file_content

        image = Image.open(io.BytesIO(file_content))

        # Resize if largest dimension > 1024 pixels (OLMocr requirement)
        max_dimension = max(image.size)
        if max_dimension > 1024:
            ratio = 1024 / max_dimension
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            output = io.BytesIO()
            format_map = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG'}
            file_ext = Path(filename).suffix.lower()
            image_format = format_map.get(file_ext, 'PNG')
            image.save(output, format=image_format)
            return output.getvalue()

        return file_content

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )

# Initialize OLMocr client
olmocr_client = OLMocrClient()


@app.on_event("startup")
async def startup_event():
    """Initialize the OCR processor on startup."""
    global processor
    
    # Report Clerk authentication status
    if clerk:
        print("✅ Clerk authentication enabled")
    else:
        print("⚠️ Clerk authentication disabled - protected endpoints will return 503")
        
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


@app.get("/debug")
async def debug_info():
    """Debug endpoint to show app info."""
    available_providers = []
    if processor:
        try:
            await processor._ensure_providers_initialized()
            available_providers = processor.get_available_providers()
        except Exception as e:
            print(f"Error getting provider info: {e}")
    
    return {
        "message": "Ocular OCR Service",
        "status": "running",
        "processor_available": processor is not None,
        "authentication": {
            "clerk_enabled": clerk is not None,
            "protected_endpoints": ["/process", "/batch-process"]
        },
        "payments": {
            "stripe_enabled": stripe.api_key is not None,
            "webhook_enabled": stripe_webhook_secret is not None,
            "payment_required_endpoints": ["/process", "/batch-process"]
        },
        "available_providers": available_providers,
        "olmocr": {
            "runpod_configured": bool(olmocr_config.RUNPOD_ENDPOINT_ID and olmocr_config.RUNPOD_API_KEY),
            "max_file_size_mb": olmocr_config.MAX_FILE_SIZE // (1024*1024),
            "allowed_extensions": list(olmocr_config.ALLOWED_EXTENSIONS)
        },
        "endpoints": [
            "/debug - This debug info",
            "/health - Health check",
            "/process - OCR processing (requires auth + payment)",
            "/batch-process - Batch processing (requires auth + payment)",
            "/olmocr - OLMocr web interface",
            "/olmocr/file - OLMocr file processing",
            "/olmocr/url - OLMocr URL processing",
            "/olmocr/config - OLMocr configuration",
            "/payments/create-intent - Create payment intent",
            "/payments/create-subscription - Create subscription",
            "/payments/webhook - Stripe webhook handler",
            "/payments/config - Payment configuration",
            "/providers - Provider information"
        ]
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main upload form."""
    # For Cloud Functions, fall back to simple response if templates fail
    try:
        available_providers = []
        provider_stats = {}
        
        if processor:
            try:
                await processor._ensure_providers_initialized()
                available_providers = processor.get_available_providers()
                provider_stats = processor.get_provider_stats()
            except Exception as e:
                print(f"Error getting provider info: {e}")
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "available_providers": available_providers,
            "provider_stats": provider_stats,
            "strategies": [s.value for s in ProcessingStrategy],
            "active_tab": request.query_params.get("tab", "unified")
        })
    except Exception as e:
        print(f"Template error: {e}")
        # Fall back to simple HTML
        base_url = ""
        return HTMLResponse(content=f"""
        <html>
            <body>
                <h1>Ocular OCR Service</h1>
                <p>Service is running but templates are not available in this environment.</p>
                <p>Available endpoints:</p>
                <ul>
                    <li><a href="/health">/health</a> - Health check</li>
                    <li><a href="/debug">/debug</a> - Debug info</li>
                    <li><a href="/providers">/providers</a> - Provider info</li>
                </ul>
                <p>Available providers: {available_providers}</p>
            </body>
        </html>
        """, status_code=200)


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
    prompt: Optional[str] = Form(None),
    payment_validation: dict = Depends(require_payment)
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
    request_data: str = Form(...),
    payment_validation: dict = Depends(require_payment)
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

@app.get("/auth/status")
async def auth_status(session = Depends(get_optional_session)):
    """Check authentication status."""
    return {
        "authenticated": session is not None,
        "clerk_enabled": clerk is not None,
        "session": session if session else None
    }

@app.get("/auth/test")
async def test_auth(auth_result = Depends(get_session)):
    """Test protected endpoint that requires authentication."""
    user_id = None
    session_id = None
    
    if auth_result:
        user_id = getattr(auth_result, 'user_id', None)
        session_id = getattr(auth_result, 'session_id', None)
    
    return {
        "message": "Authentication successful",
        "user_id": user_id,
        "session_id": session_id,
        "auth_result": str(type(auth_result)) if auth_result else None
    }


@app.get("/download/{result_id}")
async def download_result(result_id: str, format: str = "json"):
    """Download processing results in various formats."""
    # Implementation for downloading results as JSON, CSV, or text
    return {"message": "Download endpoint - implementation pending"}

# Payment Endpoints

@app.post("/payments/create-intent", response_model=PaymentIntentResponse)
async def create_payment_intent(
    request: PaymentIntentRequest,
    auth_result = Depends(get_session)
):
    """Create a Stripe payment intent for OCR processing."""
    if not stripe.api_key:
        raise HTTPException(
            status_code=503, 
            detail="Payment service not available. Stripe not configured."
        )
    
    try:
        # Add user ID to metadata
        user_id = getattr(auth_result, 'user_id', None) if auth_result else None
        metadata = request.metadata or {}
        if user_id:
            metadata['user_id'] = user_id
        
        # Create payment intent
        intent = stripe.PaymentIntent.create(
            amount=request.amount,
            currency=request.currency,
            description=request.description or "OCR Processing Service",
            metadata=metadata,
            automatic_payment_methods={
                'enabled': True,
            },
        )
        
        return PaymentIntentResponse(
            client_secret=intent.client_secret,
            payment_intent_id=intent.id,
            publishable_key=stripe_publishable_key
        )
        
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Payment creation failed: {str(e)}")

@app.post("/payments/create-subscription")
async def create_subscription(
    request: SubscriptionRequest,
    auth_result = Depends(get_session)
):
    """Create a Stripe subscription for unlimited OCR processing."""
    if not stripe.api_key:
        raise HTTPException(
            status_code=503, 
            detail="Payment service not available. Stripe not configured."
        )
    
    user_id = getattr(auth_result, 'user_id', None) if auth_result else None
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required for subscriptions")
    
    try:
        # Check if customer already exists
        customers = stripe.Customer.list(
            email=user_id,  # Assuming user_id is email, adjust as needed
            limit=1
        )
        
        if customers.data:
            customer = customers.data[0]
        else:
            # Create new customer
            customer = stripe.Customer.create(
                email=user_id,
                metadata={'user_id': user_id}
            )
        
        # Create subscription
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{'price': request.price_id}],
            payment_behavior='default_incomplete',
            payment_settings={'payment_method_types': ['card']},
            expand=['latest_invoice.payment_intent'],
        )
        
        return {
            "subscription_id": subscription.id,
            "client_secret": subscription.latest_invoice.payment_intent.client_secret,
            "customer_id": customer.id,
            "publishable_key": stripe_publishable_key
        }
        
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription creation failed: {str(e)}")

@app.post("/payments/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    if not stripe.api_key or not stripe_webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook service not configured")
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")
    
    try:
        # Verify webhook signature
        event = stripe.Webhook.construct_event(
            payload, sig_header, stripe_webhook_secret
        )
    except ValueError:
        # Invalid payload
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    if event['type'] == 'payment_intent.succeeded':
        payment_intent = event['data']['object']
        print(f"Payment succeeded for intent: {payment_intent['id']}")
        # Add your payment success logic here
        
    elif event['type'] == 'payment_intent.payment_failed':
        payment_intent = event['data']['object']
        print(f"Payment failed for intent: {payment_intent['id']}")
        # Add your payment failure logic here
        
    elif event['type'] == 'invoice.payment_succeeded':
        invoice = event['data']['object']
        print(f"Subscription payment succeeded: {invoice['id']}")
        # Add your subscription success logic here
        
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        print(f"Subscription cancelled: {subscription['id']}")
        # Add your subscription cancellation logic here
    
    else:
        print(f"Unhandled event type: {event['type']}")
    
    return {"status": "success"}

@app.get("/payments/config")
async def get_payment_config():
    """Get payment configuration including publishable key."""
    return {
        "stripe_enabled": stripe.api_key is not None,
        "publishable_key": stripe_publishable_key,
        "webhook_enabled": stripe_webhook_secret is not None
    }

# OLMocr Endpoints

@app.get("/olmocr", response_class=HTMLResponse)
async def olmocr_home(request: Request):
    """OLMocr document processor interface."""
    try:
        return templates.TemplateResponse("olmocr.html", {"request": request})
    except Exception as e:
        print(f"Template error: {e}")
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>OLMocr Document Processor</h1>
                <p>Web interface template not available. Use the API endpoints directly:</p>
                <ul>
                    <li>POST /olmocr/file - Upload file for processing</li>
                    <li>POST /olmocr/url - Process document from URL</li>
                    <li>GET /olmocr/config - Configuration info</li>
                </ul>
            </body>
        </html>
        """, status_code=200)

@app.post("/olmocr/file", response_model=OLMocrResponse)
async def olmocr_process_file(
    file: UploadFile = File(...),
    include_metadata: bool = Form(False)
):
    """Process uploaded file with OLMocr."""
    # Validate file
    validate_file(file)

    # Check file size
    file_content = await file.read()
    if len(file_content) > olmocr_config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {olmocr_config.MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Prepare image for OCR
    processed_content = prepare_image_for_ocr(file_content, file.filename)

    # Process with OLMocr
    result = await olmocr_client.process_document(processed_content, file.filename)

    if include_metadata:
        return OLMocrResponse(**result)
    else:
        return OLMocrResponse(
            success=result["success"],
            markdown_content=result["markdown_content"]
        )

@app.post("/olmocr/url", response_model=OLMocrResponse)
async def olmocr_process_url(
    url: str = Form(...),
    include_metadata: bool = Form(False)
):
    """Process document from URL with OLMocr."""
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid URL format")

    try:
        # Download file from URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(ct in content_type for ct in ['pdf', 'image', 'jpeg', 'png']):
                raise HTTPException(
                    status_code=400,
                    detail="URL must point to a PDF or image file"
                )

            file_content = response.content

            # Check file size
            if len(file_content) > olmocr_config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {olmocr_config.MAX_FILE_SIZE // (1024*1024)}MB"
                )

            # Extract filename from URL
            filename = Path(url).name or "document"
            if not Path(filename).suffix:
                # Guess extension from content type
                if 'pdf' in content_type:
                    filename += '.pdf'
                elif 'png' in content_type:
                    filename += '.png'
                elif 'jpeg' in content_type or 'jpg' in content_type:
                    filename += '.jpg'

            # Prepare and process
            processed_content = prepare_image_for_ocr(file_content, filename)
            result = await olmocr_client.process_document(processed_content, filename)

            if include_metadata:
                return OLMocrResponse(**result)
            else:
                return OLMocrResponse(
                    success=result["success"],
                    markdown_content=result["markdown_content"]
                )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error downloading file: {e.response.status_code}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing URL: {str(e)}"
        )

@app.get("/olmocr/config")
async def olmocr_get_config():
    """Get OLMocr configuration information."""
    return {
        "runpod_configured": bool(olmocr_config.RUNPOD_ENDPOINT_ID and olmocr_config.RUNPOD_API_KEY),
        "max_file_size_mb": olmocr_config.MAX_FILE_SIZE // (1024*1024),
        "allowed_extensions": list(olmocr_config.ALLOWED_EXTENSIONS),
        "timeout_seconds": olmocr_config.TIMEOUT_SECONDS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.ocular_app:app",
        host="0.0.0.0", 
        port=8001,
        reload=True,
        log_level="info"
    )
