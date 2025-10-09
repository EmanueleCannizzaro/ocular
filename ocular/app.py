"""
Modularized FastAPI web application for Ocular OCR service.
"""

from dotenv import load_dotenv
# Core imports
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
# Create a custom TemplateResponse that adds translation context
from starlette.responses import Response
from starlette.background import BackgroundTask
from typing import Optional, Dict, Any

# Import database initialization
from ocular.database.session import init_db
from ocular.database.seed import seed_database

# Add i18n middleware for internationalization
from ocular.middleware import I18nMiddleware
# Import routers
from ocular.routers import templates
from ocular.routers.main import router as main_router
from ocular.routers.health import router as health_router
from ocular.routers.api import router as api_router
from ocular.routers.analytics import router as analytics_router
from ocular.routers.versions import router as versions_router
from ocular.routers.deployment_config import router as deployment_config_router
from ocular.routers.auth import router as auth_router
from ocular.routers.billing import router as billing_router
from ocular.routers.observability import router as observability_router
from ocular.routers.batch import router as batch_router
from ocular.routers.document_tracking import router as document_tracking_router
from ocular.routers.property_extraction import router as property_extraction_router
# Import new settings system
from ocular.settings import settings


env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")
else:
    load_dotenv()
    print("✅ Loaded .env from default location")

# Initialize external services
clerk = None
stripe = None
security = HTTPBearer()


def initialize_clerk():
    """Initialize Clerk authentication client"""
    global clerk

    if not settings.auth.clerk_secret_key:
        print("⚠️ CLERK_SECRET_KEY not found - authentication will be disabled")
        return None

    try:
        from clerk_backend_api import Clerk
        clerk = Clerk(bearer_auth=settings.auth.clerk_secret_key)
        print("✅ Clerk client initialized successfully")
        return clerk
    except ImportError:
        print("⚠️ clerk-backend-api not installed - authentication will be disabled")
    except Exception as e:
        print(f"❌ Failed to initialize Clerk client: {e}")

    return None


def initialize_stripe():
    """Initialize Stripe payment client"""
    global stripe

    if not settings.payments.stripe_secret_key:
        print("⚠️ STRIPE_SECRET_KEY not found - payments will be disabled")
        return None

    try:
        import stripe as stripe_module
        stripe_module.api_key = settings.payments.stripe_secret_key
        stripe = stripe_module
        print("✅ Stripe client initialized successfully")
        return stripe
    except ImportError:
        print("⚠️ stripe not installed - payments will be disabled")
    except Exception as e:
        print(f"❌ Failed to initialize Stripe client: {e}")

    return None


async def get_session(request: Request) -> Optional[Dict[str, Any]]:
    """
    Dependency to verify the Clerk session from the Authorization header.
    Returns None if authentication is disabled or session is invalid.
    """
    if not clerk:
        return None

    try:
        # Get the Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        # Extract the session token
        session_token = auth_header.replace("Bearer ", "")

        # Verify the session with Clerk
        session = clerk.sessions.verify_session(session_token, settings.auth.clerk_secret_key)

        if session and session.status == "active":
            return {"user_id": session.user_id, "session_id": session.id}

    except Exception as e:
        print(f"Session verification error: {e}")

    return None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    # Initialize external services
    initialize_clerk()
    initialize_stripe()

    # Create FastAPI app with settings
    app = FastAPI(
        title=settings.app.title,
        description=settings.app.description,
        version=settings.app.version,
        debug=settings.app.debug
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(I18nMiddleware, default_language='en')

    class TranslatedTemplateResponse(Response):
        def __init__(
            self,
            template: str,
            context: dict,
            status_code: int = 200,
            headers: dict = None,
            media_type: str = None,
            background: BackgroundTask = None,
        ):
            request = context.get('request')
            if request:
                # Add translation function and language to context
                context['_'] = getattr(request.state, '_', lambda x: x)
                context['current_lang'] = getattr(request.state, 'language', 'en')

            content = templates.get_template(template).render(context)
            super().__init__(content, status_code, headers, media_type, background)

    # Replace the templates.TemplateResponse with our custom one
    templates.TemplateResponse = TranslatedTemplateResponse

    print(f"✅ Templates directory: {settings.app.templates_path}")

    # Mount static files if enabled
    if settings.app.enable_static_files and settings.app.static_path.exists():
        app.mount("/static", StaticFiles(directory=str(settings.app.static_path)), name="static")
        print(f"✅ Static files mounted from {settings.app.static_path}")
    else:
        print("⚠️ Static files disabled or directory not found")


    # Include routers
    app.include_router(health_router, tags=["health"])
    app.include_router(main_router, tags=["ocr"])
    app.include_router(api_router, prefix="/api", tags=["api"])
    app.include_router(analytics_router, tags=["analytics"])
    app.include_router(versions_router, prefix="/api", tags=["versions"])
    app.include_router(deployment_config_router, prefix="/api", tags=["deployment-config"])
    app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
    app.include_router(billing_router, prefix="/api/billing", tags=["billing"])
    app.include_router(observability_router, prefix="/api/observability", tags=["observability"])
    app.include_router(batch_router, prefix="/api/batch", tags=["batch-processing"])
    app.include_router(document_tracking_router, prefix="/api/document-tracking", tags=["document-tracking"])
    app.include_router(property_extraction_router, tags=["property-extraction"])

    # Settings router
    from ocular.routers.settings import router as settings_router
    app.include_router(settings_router, prefix="/api", tags=["settings"])

    # Notifications router
    from ocular.routers.notifications import router as notifications_router
    app.include_router(notifications_router, prefix="/api", tags=["notifications"])

    # Add authentication and payment routes if services are available
    if clerk:
        setup_auth_routes(app)

    if stripe:
        setup_payment_routes(app)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors"""
        print(f"Unhandled exception: {exc}")

        if settings.app.debug:
            # In debug mode, show detailed error
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(exc),
                    "type": type(exc).__name__
                }
            )
        else:
            # In production, show generic error
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )

    return app


def setup_auth_routes(app: FastAPI):
    """Setup authentication-related routes"""

    @app.get("/auth/status")
    async def auth_status(session_info: Optional[Dict] = Depends(get_session)):
        """Get authentication status"""
        if session_info:
            return {
                "authenticated": True,
                "user_id": session_info.get("user_id"),
                "session_id": session_info.get("session_id")
            }
        else:
            return {"authenticated": False}

    @app.get("/auth/test")
    async def auth_test(session_info: Optional[Dict] = Depends(get_session)):
        """Test endpoint for authentication"""
        if not session_info:
            raise HTTPException(status_code=401, detail="Authentication required")

        return {
            "message": "Authentication successful",
            "user_id": session_info.get("user_id"),
            "timestamp": "2024-01-01T00:00:00Z"  # You might want to use actual timestamp
        }


def setup_payment_routes(app: FastAPI):
    """Setup payment-related routes"""

    @app.post("/payments/create-intent")
    async def create_payment_intent(
        amount: int,
        currency: str = "usd",
        session_info: Optional[Dict] = Depends(get_session)
    ):
        """Create a Stripe payment intent"""
        if not session_info:
            raise HTTPException(status_code=401, detail="Authentication required")

        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency=currency,
                metadata={'user_id': session_info.get("user_id")}
            )

            return {
                "client_secret": intent.client_secret,
                "intent_id": intent.id
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/payments/config")
    async def payment_config():
        """Get payment configuration"""
        return {
            "stripe_publishable_key": settings.payments.stripe_publishable_key,
            "currency": "usd",
            "payment_methods": ["card"]
        }

    @app.post("/payments/webhook")
    async def stripe_webhook(request: Request):
        """Handle Stripe webhooks"""
        payload = await request.body()
        sig_header = request.headers.get('stripe-signature')

        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.payments.stripe_webhook_secret
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid signature")

        # Handle the event
        if event['type'] == 'payment_intent.succeeded':
            payment_intent = event['data']['object']
            print(f"Payment succeeded: {payment_intent['id']}")

        return {"status": "success"}


# Create the app instance
app = create_app()


# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    print("🚀 Ocular OCR Service starting up...")
    print(f"📁 Templates directory: {settings.app.templates_dir}")
    print(f"📁 Static files directory: {settings.app.static_dir}")
    print(f"🌐 Environment: {settings.app.environment}")
    print(f"🔧 Debug mode: {settings.app.debug}")

    # Initialize database
    try:
        print("💾 Initializing database...")
        init_db()

        # Seed database with initial data if needed
        try:
            seed_database()
        except Exception as e:
            print(f"⚠️ Database seeding skipped or failed: {e}")

        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

    # Print configuration summary
    config_summary = settings.get_summary()
    print("⚙️ Configuration loaded:")
    for group, config in config_summary.items():
        # Handle lists (cloud_providers, ai_models)
        if isinstance(config, list):
            if config:
                print(f"  - {group}: {len(config)} items configured")
        # Handle dictionaries (normal settings groups)
        elif isinstance(config, dict):
            if any(v for v in config.values() if v and v != "***HIDDEN***"):
                print(f"  - {group}: configured")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    print("🛑 Ocular OCR Service shutting down...")


# For development server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
        workers=1 if settings.app.debug else settings.app.workers
    )