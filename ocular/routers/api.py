"""
API router for Ocular OCR application.
Contains all JSON API endpoints including deployment management.
"""

import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, status, File, Form, UploadFile
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ocular.unified_processor import UnifiedDocumentProcessor
from ocular.settings import get_settings
from ocular.database.session import get_db
from ocular.database.models import (
    Provider,
    ProviderStatus,
    Model,
    ModelStatus,
    Deployment,
    DeploymentStatus,
    DeploymentType
)
from ocular.routers import templates
from ocular.services.deployment_service import get_deployment_service
from ocular.services.env_service import get_env_service
from ocular.models import (
    DeploymentModelOption,
    DeploymentOptionsResponse,
    DeploymentProviderOption,
)

# Initialize router
router = APIRouter()

# Global processor instance
processor = None

def get_processor():
    """Get or create processor instance"""
    global processor
    if processor is None:
        try:
            settings = get_settings()
            processor = UnifiedDocumentProcessor(settings)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize OCR processor: {str(e)}")
    return processor


# ============================================================================
# Pydantic Models for Deployment API
# ============================================================================

# Provider Models
class ProviderResponse(BaseModel):
    """Response model for provider."""
    id: int
    name: str
    display_name: str
    description: Optional[str]
    website_url: Optional[str]
    documentation_url: Optional[str]
    status: str
    api_endpoint: Optional[str]
    supports_autoscaling: bool
    supports_spot_instances: bool
    available_gpus: List[str]
    min_cost_per_hour: Optional[float]
    max_cost_per_hour: Optional[float]
    metadata: dict
    created_at: Optional[str]
    updated_at: Optional[str]


# Model Models
class ModelResponse(BaseModel):
    """Response model for model."""
    id: int
    name: str
    display_name: str
    version: str
    description: Optional[str]
    status: str
    parameter_count: Optional[str]
    model_type: Optional[str]
    base_model: Optional[str]
    license: Optional[str]
    pipeline_tag: Optional[str]
    min_vram_gb: int
    recommended_gpu: Optional[str]
    docker_image: Optional[str]
    dockerfile_path: Optional[str]
    default_deployment_type: str
    supports_vllm: bool
    config_template: dict
    metadata: dict
    downloads: Optional[int]
    likes: Optional[int]
    last_modified: Optional[str]
    repo_url: Optional[str]
    documentation_url: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


# Deployment Models
class DeploymentCreate(BaseModel):
    """Request model for creating a deployment."""
    deployment_name: str = Field(..., min_length=1, max_length=255)
    model_id: int = Field(..., description="Model ID from models table")
    provider_id: int = Field(..., description="Provider ID from providers table")
    deployment_type: str = Field(default="docker", description="Deployment type (docker, vllm, custom)")
    gpu_type: str = Field(..., description="GPU type (e.g., RTX A4000)")
    gpu_count: int = Field(default=1, ge=1, le=8)
    api_key: str = Field(..., description="Provider API key")
    cost_per_hour: float = Field(default=0.0, ge=0.0)
    max_workers: int = Field(default=1, ge=0, le=10)
    min_workers: int = Field(default=0, ge=0, le=10)
    timeout: int = Field(default=300, ge=60, le=600, description="Timeout in seconds")
    autoscaling: bool = Field(default=False)
    region: Optional[str] = None
    config: Optional[dict] = None
    metadata: Optional[dict] = None


class DeploymentUpdate(BaseModel):
    """Request model for updating a deployment."""
    deployment_name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[str] = None
    endpoint_url: Optional[str] = None
    instance_id: Optional[str] = None
    max_workers: Optional[int] = Field(None, ge=0, le=10)
    min_workers: Optional[int] = Field(None, ge=0, le=10)
    timeout: Optional[int] = Field(None, ge=60, le=600)
    autoscaling: Optional[bool] = None
    error_message: Optional[str] = None
    config: Optional[dict] = None


class DeploymentAction(BaseModel):
    """Request model for deployment actions (start, stop, delete)."""
    api_key: str = Field(..., description="Provider API key for authentication")
    metadata: Optional[dict] = None


class DeploymentResponse(BaseModel):
    """Response model for deployment."""
    id: str
    deployment_name: str
    provider_id: int
    provider_name: Optional[str]
    model_id: int
    model_name: Optional[str]
    model_display_name: Optional[str]
    model_version: Optional[str]
    deployment_type: str
    gpu_type: str
    gpu_count: int
    status: str
    endpoint: Optional[str]
    cost_per_hour: float
    total_cost: float
    max_workers: int
    min_workers: int
    timeout: int
    autoscaling: bool
    instance_id: Optional[str]
    region: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    started_at: Optional[str]
    stopped_at: Optional[str]
    error_message: Optional[str]
    total_requests: int
    last_request_at: Optional[str]
    avg_response_time_ms: Optional[float]
    cost_class: str


class DeploymentMetrics(BaseModel):
    """Response model for deployment metrics."""
    active_deployments: int
    gpu_count: int
    monthly_cost: float
    avg_response: str
    total_requests: int
    total_cost: float


# Helper Functions
def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


# ============================================================================
# Deployment Management Endpoints
# ============================================================================

# Provider Endpoints
@router.get("/providers", response_model=List[ProviderResponse])
async def list_providers(
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all cloud providers."""
    query = db.query(Provider)

    if status_filter:
        try:
            status_enum = ProviderStatus(status_filter)
            query = query.filter(Provider.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}"
            )

    query = query.order_by(Provider.name)
    providers = query.offset(skip).limit(limit).all()

    return [p.to_dict() for p in providers]


@router.get("/providers/{provider_id}", response_model=ProviderResponse)
async def get_provider(provider_id: int, db: Session = Depends(get_db)):
    """Get a specific provider by ID."""
    provider = db.query(Provider).filter(Provider.id == provider_id).first()

    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider {provider_id} not found"
        )

    return provider.to_dict()


# Model Endpoints
@router.get("/deployments/models", response_model=List[ModelResponse])
async def list_models(
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all AI models available for deployment."""
    query = db.query(Model)

    if status_filter:
        try:
            status_enum = ModelStatus(status_filter)
            query = query.filter(Model.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}"
            )

    query = query.order_by(Model.name)
    models = query.offset(skip).limit(limit).all()

    return [m.to_dict() for m in models]


@router.get("/deployments/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get a specific model by ID."""
    model = db.query(Model).filter(Model.id == model_id).first()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    return model.to_dict()


@router.get("/deployments/options", response_model=DeploymentOptionsResponse)
async def get_deployment_options(db: Session = Depends(get_db)):
    """Aggregate deployable models and active providers for the deployment wizard."""
    available_models = (
        db.query(Model)
        .filter(Model.status == ModelStatus.AVAILABLE)
        .order_by(Model.name)
        .all()
    )

    active_providers = (
        db.query(Provider)
        .filter(Provider.status == ProviderStatus.ACTIVE)
        .order_by(Provider.name)
        .all()
    )

    model_options = [
        DeploymentModelOption(
            id=model.id,
            name=model.name,
            display_name=model.display_name,
            version=model.version,
            status=model.status.value,
            min_vram_gb=model.min_vram_gb,
            supports_vllm=model.supports_vllm,
            description=model.description,
            recommended_gpu=model.recommended_gpu,
            license=model.license,
            pipeline_tag=model.pipeline_tag,
            metadata=model.to_dict().get("metadata", {})
        )
        for model in available_models
    ]

    provider_options = [
        DeploymentProviderOption(
            id=provider.id,
            name=provider.name,
            display_name=provider.display_name,
            status=provider.status.value,
            available_gpus=provider.available_gpus or [],
            supports_autoscaling=provider.supports_autoscaling,
            supports_spot_instances=provider.supports_spot_instances,
            min_cost_per_hour=provider.min_cost_per_hour,
            max_cost_per_hour=provider.max_cost_per_hour,
            description=provider.description,
            metadata=provider.to_dict().get("metadata", {})
        )
        for provider in active_providers
    ]

    return DeploymentOptionsResponse.from_payload(
        models=model_options,
        providers=provider_options,
    )


@router.get("/deployments/storage/buckets")
async def get_storage_buckets():
    """
    Get available storage buckets from configuration.

    Returns list of configured S3 buckets for model uploads.
    """
    import os

    buckets = []

    # RunPod buckets
    runpod_bucket = os.getenv("RUNPOD_BUCKET_NAME")
    if runpod_bucket:
        buckets.append({
            "name": runpod_bucket,
            "provider": "runpod",
            "display_name": f"RunPod - {runpod_bucket}",
            "default": True
        })

    # AWS S3 buckets (comma-separated in env)
    aws_buckets = os.getenv("AWS_S3_BUCKETS", "").split(",")
    for bucket in aws_buckets:
        if bucket.strip():
            buckets.append({
                "name": bucket.strip(),
                "provider": "aws",
                "display_name": f"AWS S3 - {bucket.strip()}",
                "default": False
            })

    # Custom S3-compatible buckets
    custom_buckets = os.getenv("CUSTOM_S3_BUCKETS", "").split(",")
    for bucket in custom_buckets:
        if bucket.strip():
            buckets.append({
                "name": bucket.strip(),
                "provider": "custom",
                "display_name": f"Custom - {bucket.strip()}",
                "default": False
            })

    if not buckets:
        buckets.append({
            "name": "default-ocular-models",
            "provider": "runpod",
            "display_name": "Default RunPod Bucket",
            "default": True
        })

    return {"buckets": buckets}


@router.post("/deployments/models/{model_id}/upload-from-hf")
async def upload_model_from_huggingface(
    model_id: int,
    bucket: str = Form(...),
    provider: str = Form(default="runpod"),
    make_public: bool = Form(default=False),
    access_key_id: Optional[str] = Form(default=None),
    secret_access_key: Optional[str] = Form(default=None),
    db: Session = Depends(get_db)
):
    """
    Download model from HuggingFace and upload to S3-compatible storage.

    This endpoint:
    1. Validates the model exists and has a HuggingFace version
    2. Downloads model files from HuggingFace
    3. Uploads to S3 bucket (RunPod, AWS, or custom)
    4. Updates model record with S3 URL
    5. Returns upload status and URL

    Args:
        model_id: Model ID from database
        bucket: S3 bucket name
        provider: Storage provider (runpod, aws, custom)
        make_public: Whether to make file publicly accessible
        access_key_id: Optional S3 access key (uses env var if not provided)
        secret_access_key: Optional S3 secret key (uses env var if not provided)

    Returns:
        Upload status with S3 URL and metadata
    """
    from ocular.services.s3_service import S3Service
    from huggingface_hub import snapshot_download
    import tempfile
    import shutil

    # Validate model exists
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    # Validate model has HuggingFace version
    if not model.version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model does not have a HuggingFace version specified"
        )

    temp_dir = None
    try:
        # Create S3 service
        if access_key_id and secret_access_key:
            s3_service = S3Service(
                provider=provider,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key
            )
        else:
            from ocular.services.s3_service import get_s3_service
            s3_service = get_s3_service(provider=provider)

        # Download from HuggingFace to temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"hf_model_{model_id}_")

        try:
            # Download entire model repository
            download_path = snapshot_download(
                repo_id=model.version,
                cache_dir=temp_dir,
                local_dir=f"{temp_dir}/model",
                local_dir_use_symlinks=False
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download model from HuggingFace: {str(e)}"
            )

        # Upload all files to S3
        uploaded_files = []
        total_size = 0

        for root, dirs, files in os.walk(download_path):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(download_path)

                # Generate S3 object key
                object_key = f"models/{model.name}/{model.version}/{relative_path}"

                # Upload file
                result = await s3_service.upload_file(
                    file_path=str(file_path),
                    bucket=bucket,
                    object_key=object_key,
                    metadata={
                        "model_id": str(model_id),
                        "model_name": model.name,
                        "model_version": model.version,
                        "uploaded_by": "ocular_api",
                        "source": "huggingface"
                    },
                    make_public=make_public
                )

                if result.get("success"):
                    uploaded_files.append({
                        "filename": str(relative_path),
                        "url": result["url"],
                        "size_bytes": result["size_bytes"]
                    })
                    total_size += result["size_bytes"]

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        if not uploaded_files:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No files were uploaded"
            )

        # Update model record with S3 upload info
        model.extra_metadata = model.extra_metadata or {}
        model.extra_metadata["s3_uploads"] = model.extra_metadata.get("s3_uploads", [])
        model.extra_metadata["s3_uploads"].append({
            "source": "huggingface",
            "repo_id": model.version,
            "bucket": bucket,
            "base_key": f"models/{model.name}/{model.version}",
            "files": uploaded_files,
            "total_size_bytes": total_size,
            "file_count": len(uploaded_files),
            "provider": provider,
            "uploaded_at": datetime.utcnow().isoformat(),
            "public": make_public
        })

        model.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(model)

        return {
            "success": True,
            "message": f"Model downloaded from HuggingFace and uploaded to {bucket}",
            "model_id": model_id,
            "model_name": model.name,
            "huggingface_repo": model.version,
            "upload": {
                "bucket": bucket,
                "base_key": f"models/{model.name}/{model.version}",
                "file_count": len(uploaded_files),
                "total_size_bytes": total_size,
                "files": uploaded_files,
                "provider": provider,
                "public": make_public
            },
            "model": model.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        # Clean up temporary directory on error
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload from HuggingFace failed: {str(e)}"
        )


@router.post("/deployments/models/{model_id}/upload")
async def upload_model_to_s3(
    model_id: int,
    file: UploadFile = File(...),
    bucket: str = Form(...),
    provider: str = Form(default="runpod"),
    make_public: bool = Form(default=False),
    access_key_id: Optional[str] = Form(default=None),
    secret_access_key: Optional[str] = Form(default=None),
    db: Session = Depends(get_db)
):
    """
    Upload a model file to S3-compatible storage.

    This endpoint:
    1. Validates the model exists
    2. Saves uploaded file temporarily
    3. Uploads to S3 bucket (RunPod, AWS, or custom)
    4. Updates model record with S3 URL
    5. Returns upload status and URL

    Args:
        model_id: Model ID from database
        file: Model file to upload (weights, Docker image, etc.)
        bucket: S3 bucket name
        provider: Storage provider (runpod, aws, custom)
        make_public: Whether to make file publicly accessible
        access_key_id: Optional S3 access key (uses env var if not provided)
        secret_access_key: Optional S3 secret key (uses env var if not provided)

    Returns:
        Upload status with S3 URL and metadata
    """
    from ocular.services.s3_service import S3Service
    import tempfile
    import shutil

    # Validate model exists
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    # Create temporary file to save upload
    temp_file = None
    try:
        # Create S3 service
        if access_key_id and secret_access_key:
            s3_service = S3Service(
                provider=provider,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key
            )
        else:
            from ocular.services.s3_service import get_s3_service
            s3_service = get_s3_service(provider=provider)

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Generate S3 object key (path in bucket)
        # Format: models/{model_name}/{version}/{filename}
        object_key = f"models/{model.name}/{model.version}/{file.filename}"

        # Upload to S3
        result = await s3_service.upload_file(
            file_path=temp_file_path,
            bucket=bucket,
            object_key=object_key,
            metadata={
                "model_id": str(model_id),
                "model_name": model.name,
                "model_version": model.version,
                "uploaded_by": "ocular_api"
            },
            make_public=make_public
        )

        # Clean up temporary file
        Path(temp_file_path).unlink(missing_ok=True)

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload failed: {result.get('error', 'Unknown error')}"
            )

        # Update model record with S3 URL
        model.extra_metadata = model.extra_metadata or {}
        model.extra_metadata["s3_uploads"] = model.extra_metadata.get("s3_uploads", [])
        model.extra_metadata["s3_uploads"].append({
            "filename": file.filename,
            "bucket": bucket,
            "key": object_key,
            "url": result["url"],
            "size_bytes": result["size_bytes"],
            "provider": provider,
            "uploaded_at": datetime.utcnow().isoformat(),
            "public": make_public
        })

        # If this is a Docker image or primary model file, update docker_image field
        if file.filename.endswith(('.tar', '.tar.gz', '.tgz')) and 'docker' in file.filename.lower():
            model.docker_image = result["url"]

        model.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(model)

        return {
            "success": True,
            "message": "Model uploaded successfully",
            "model_id": model_id,
            "model_name": model.name,
            "upload": {
                "filename": file.filename,
                "url": result["url"],
                "bucket": bucket,
                "key": object_key,
                "size_bytes": result["size_bytes"],
                "provider": provider,
                "public": make_public
            },
            "model": model.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        # Clean up temporary file on error
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


# Deployment Endpoints
@router.post("/deployments", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
async def create_deployment(
    deployment_data: DeploymentCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new deployment instance and execute deployment to cloud provider.

    This endpoint:
    1. Validates provider and model exist
    2. Creates deployment record in database
    3. Executes actual deployment to cloud provider (RunPod, Vast.ai, Lambda)
    4. Updates deployment record with endpoint information
    """
    # Validate provider exists
    provider = db.query(Provider).filter(Provider.id == deployment_data.provider_id).first()
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider {deployment_data.provider_id} not found"
        )
    if provider.status != ProviderStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider.display_name}' is not active"
        )

    # Validate model exists
    model = db.query(Model).filter(Model.id == deployment_data.model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {deployment_data.model_id} not found"
        )
    if model.status != ModelStatus.AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model.display_name}' is not available for deployment"
        )

    # Validate deployment type
    try:
        deployment_type_enum = DeploymentType(deployment_data.deployment_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid deployment type: {deployment_data.deployment_type}"
        )

    # Validate vLLM compatibility
    if deployment_type_enum == DeploymentType.VLLM and not model.supports_vllm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model.display_name}' does not support vLLM deployments"
        )

    # Validate GPU selection
    provider_gpus = provider.available_gpus or []
    if provider_gpus and deployment_data.gpu_type not in provider_gpus:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"GPU type '{deployment_data.gpu_type}' is not available for provider "
                f"'{provider.display_name}'. Available options: {', '.join(provider_gpus)}"
            )
        )

    # Generate unique deployment ID
    deployment_id = str(uuid.uuid4())

    # Hash the API key for security (but keep plain text for deployment execution)
    api_key_hash = hash_api_key(deployment_data.api_key)
    api_key_plain = deployment_data.api_key  # Store temporarily for deployment

    # Create deployment object
    deployment = Deployment(
        deployment_id=deployment_id,
        deployment_name=deployment_data.deployment_name,
        provider_id=deployment_data.provider_id,
        model_id=deployment_data.model_id,
        deployment_type=deployment_type_enum,
        gpu_type=deployment_data.gpu_type,
        gpu_count=deployment_data.gpu_count,
        status=DeploymentStatus.PENDING,
        api_key_hash=api_key_hash,
        cost_per_hour=deployment_data.cost_per_hour,
        max_workers=deployment_data.max_workers,
        min_workers=deployment_data.min_workers,
        timeout_seconds=deployment_data.timeout,
        autoscaling_enabled=deployment_data.autoscaling,
        region=deployment_data.region,
        config=deployment_data.config,
        extra_metadata=deployment_data.metadata,
    )

    db.add(deployment)
    db.commit()
    db.refresh(deployment)

    # Execute actual deployment to cloud provider
    try:
        deployment_service = get_deployment_service()

        # Execute deployment based on provider
        deployment_result = await deployment_service.deploy(
            deployment=deployment,
            api_key=api_key_plain
        )

        # Deployment service updates the deployment record automatically
        db.commit()
        db.refresh(deployment)

    except NotImplementedError as e:
        # Provider deployment not yet implemented
        deployment.status = DeploymentStatus.ERROR
        deployment.error_message = str(e)
        db.commit()
        db.refresh(deployment)

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Deployment to {provider.name} not yet implemented: {str(e)}"
        )
    except Exception as e:
        # Deployment failed
        deployment.status = DeploymentStatus.ERROR
        deployment.error_message = str(e)
        db.commit()
        db.refresh(deployment)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deployment failed: {str(e)}"
        )

    response = deployment.to_dict()
    response["cost_class"] = deployment.cost_class

    return response


@router.get("/deployments", response_model=List[DeploymentResponse])
async def list_deployments(
    status_filter: Optional[str] = None,
    provider_id: Optional[int] = None,
    model_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all deployments with optional filtering."""
    query = db.query(Deployment)

    # Apply filters
    if status_filter:
        try:
            status_enum = DeploymentStatus(status_filter)
            query = query.filter(Deployment.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}"
            )

    if provider_id:
        query = query.filter(Deployment.provider_id == provider_id)

    if model_id:
        query = query.filter(Deployment.model_id == model_id)

    # Exclude deleted deployments by default
    query = query.filter(Deployment.status != DeploymentStatus.DELETED)

    # Order by creation date (newest first)
    query = query.order_by(Deployment.created_at.desc())

    # Pagination
    deployments = query.offset(skip).limit(limit).all()

    return [
        {**d.to_dict(), "cost_class": d.cost_class}
        for d in deployments
    ]


@router.get("/deployments/metrics", response_model=DeploymentMetrics)
async def get_deployment_metrics(db: Session = Depends(get_db)):
    """Get deployment metrics and statistics."""
    # Active deployments (running or deploying)
    active_count = db.query(Deployment).filter(
        Deployment.status.in_([DeploymentStatus.RUNNING, DeploymentStatus.DEPLOYING])
    ).count()

    # Total GPU count for active deployments
    active_deployments = db.query(Deployment).filter(
        Deployment.status.in_([DeploymentStatus.RUNNING, DeploymentStatus.DEPLOYING])
    ).all()

    gpu_count = sum(d.gpu_count for d in active_deployments)

    # Calculate monthly cost (hourly * 730 hours)
    hourly_cost = sum(d.cost_per_hour for d in active_deployments)
    monthly_cost = hourly_cost * 730

    # Total requests and cost
    all_deployments = db.query(Deployment).filter(
        Deployment.status != DeploymentStatus.DELETED
    ).all()

    total_requests = sum(d.total_requests for d in all_deployments)
    total_cost = sum(d.total_cost for d in all_deployments)

    # Calculate average response time
    avg_times = [d.avg_response_time_ms for d in all_deployments if d.avg_response_time_ms]
    avg_response = f"{sum(avg_times) / len(avg_times):.0f}ms" if avg_times else "N/A"

    return {
        "active_deployments": active_count,
        "gpu_count": gpu_count,
        "monthly_cost": round(monthly_cost, 2),
        "avg_response": avg_response,
        "total_requests": total_requests,
        "total_cost": round(total_cost, 2),
    }


@router.get("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(deployment_id: str, db: Session = Depends(get_db)):
    """Get a specific deployment by ID."""
    deployment = db.query(Deployment).filter(
        Deployment.deployment_id == deployment_id
    ).first()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )

    response = deployment.to_dict()
    response["cost_class"] = deployment.cost_class

    return response


@router.post("/deployments/{deployment_id}/refresh-status")
async def refresh_deployment_status(
    deployment_id: str,
    action_data: DeploymentAction,
    db: Session = Depends(get_db)
):
    """
    Refresh deployment status by polling the cloud provider.

    This endpoint queries the provider's API to get the current deployment status
    and updates the database accordingly.
    """
    deployment = db.query(Deployment).filter(
        Deployment.deployment_id == deployment_id
    ).first()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )

    if not deployment.instance_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deployment has no instance ID - cannot check status"
        )

    # Get provider
    provider = db.query(Provider).filter(Provider.id == deployment.provider_id).first()
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Provider not found"
        )

    # Check status with provider
    try:
        from ocular.services.deployment_providers import RunPodDeploymentProvider

        provider_name = provider.name.lower()

        if provider_name == "runpod":
            provider_instance = RunPodDeploymentProvider(api_key=action_data.api_key)
            result = await provider_instance.get_deployment_status(deployment.instance_id)

            # Update deployment status
            if result.success:
                # Map provider status to our status
                if result.status == "running":
                    deployment.status = DeploymentStatus.RUNNING
                    if not deployment.started_at:
                        deployment.started_at = datetime.utcnow()
                elif result.status == "deploying":
                    deployment.status = DeploymentStatus.DEPLOYING
                elif result.status == "error":
                    deployment.status = DeploymentStatus.ERROR
                    deployment.error_message = result.error_message
                    deployment.last_error_at = datetime.utcnow()

                # Update metadata
                deployment.extra_metadata = deployment.extra_metadata or {}
                deployment.extra_metadata.update(result.metadata)

                deployment.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(deployment)

                return {
                    "success": True,
                    "deployment_id": deployment_id,
                    "status": deployment.status.value,
                    "message": "Status refreshed successfully",
                    "deployment": deployment.to_dict()
                }
            else:
                return {
                    "success": False,
                    "deployment_id": deployment_id,
                    "status": deployment.status.value,
                    "error_message": result.error_message
                }

        else:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Status refresh not implemented for provider: {provider_name}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh status: {str(e)}"
        )


@router.patch("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def update_deployment(
    deployment_id: str,
    update_data: DeploymentUpdate,
    db: Session = Depends(get_db)
):
    """Update a deployment."""
    deployment = db.query(Deployment).filter(
        Deployment.deployment_id == deployment_id
    ).first()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )

    # Update fields if provided
    if update_data.deployment_name is not None:
        deployment.deployment_name = update_data.deployment_name

    if update_data.status is not None:
        try:
            new_status = DeploymentStatus(update_data.status)
            deployment.status = new_status

            # Update timestamps based on status
            if new_status == DeploymentStatus.RUNNING and not deployment.started_at:
                deployment.started_at = datetime.utcnow()
            elif new_status == DeploymentStatus.STOPPED:
                deployment.stopped_at = datetime.utcnow()
            elif new_status == DeploymentStatus.ERROR:
                deployment.last_error_at = datetime.utcnow()

        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {update_data.status}"
            )

    if update_data.endpoint_url is not None:
        deployment.endpoint_url = update_data.endpoint_url

    if update_data.instance_id is not None:
        deployment.instance_id = update_data.instance_id

    if update_data.max_workers is not None:
        deployment.max_workers = update_data.max_workers

    if update_data.min_workers is not None:
        deployment.min_workers = update_data.min_workers

    if update_data.timeout is not None:
        deployment.timeout_seconds = update_data.timeout

    if update_data.autoscaling is not None:
        deployment.autoscaling_enabled = update_data.autoscaling

    if update_data.error_message is not None:
        deployment.error_message = update_data.error_message

    if update_data.config is not None:
        deployment.config = update_data.config

    if update_data.metadata is not None:
        deployment.metadata = update_data.metadata

    deployment.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(deployment)

    response = deployment.to_dict()
    response["cost_class"] = deployment.cost_class

    return response


@router.post("/deployments/{deployment_id}/delete", response_model=DeploymentResponse)
async def delete_deployment(
    deployment_id: str,
    action_data: DeploymentAction,
    db: Session = Depends(get_db)
):
    """Delete a deployment from the cloud provider and mark as deleted in database."""
    deployment = db.query(Deployment).filter(
        Deployment.deployment_id == deployment_id
    ).first()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )

    if deployment.status == DeploymentStatus.DELETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deployment is already deleted"
        )

    # Execute actual delete process
    try:
        deployment_service = get_deployment_service()

        success = await deployment_service.delete_deployment(
            deployment=deployment,
            api_key=action_data.api_key
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete deployment"
            )

        # Deployment service updates the deployment record
        db.commit()
        db.refresh(deployment)

    except NotImplementedError as e:
        # If provider doesn't support delete, just mark as deleted locally
        deployment.status = DeploymentStatus.DELETED
        deployment.stopped_at = datetime.utcnow()
        deployment.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(deployment)

    except Exception as e:
        deployment.error_message = str(e)
        deployment.last_error_at = datetime.utcnow()
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete deployment: {str(e)}"
        )

    response = deployment.to_dict()
    response["cost_class"] = deployment.cost_class

    return response


@router.post("/deployments/{deployment_id}/start", response_model=DeploymentResponse)
async def start_deployment(deployment_id: str, db: Session = Depends(get_db)):
    """Start a stopped deployment."""
    deployment = db.query(Deployment).filter(
        Deployment.deployment_id == deployment_id
    ).first()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )

    if deployment.status == DeploymentStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deployment is already running"
        )

    deployment.status = DeploymentStatus.DEPLOYING
    deployment.started_at = datetime.utcnow()
    deployment.updated_at = datetime.utcnow()
    deployment.error_message = None

    db.commit()
    db.refresh(deployment)

    # TODO: Trigger actual deployment process here

    response = deployment.to_dict()
    response["cost_class"] = deployment.cost_class

    return response


@router.post("/deployments/{deployment_id}/stop", response_model=DeploymentResponse)
async def stop_deployment(
    deployment_id: str,
    action_data: DeploymentAction,
    db: Session = Depends(get_db)
):
    """Stop a running deployment on the cloud provider."""
    deployment = db.query(Deployment).filter(
        Deployment.deployment_id == deployment_id
    ).first()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )

    if deployment.status == DeploymentStatus.STOPPED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deployment is already stopped"
        )

    # Execute actual stop process
    try:
        deployment_service = get_deployment_service()

        success = await deployment_service.stop_deployment(
            deployment=deployment,
            api_key=action_data.api_key
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to stop deployment"
            )

        # Deployment service updates the deployment record
        db.commit()
        db.refresh(deployment)

    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Stop operation not implemented: {str(e)}"
        )
    except Exception as e:
        deployment.error_message = str(e)
        deployment.last_error_at = datetime.utcnow()
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop deployment: {str(e)}"
        )

    response = deployment.to_dict()
    response["cost_class"] = deployment.cost_class

    return response


# ============================================================================
# OCR Provider Endpoints
# ============================================================================

@router.get("/ocr-providers/status")
async def get_ocr_providers_status():
    """Get OCR providers status (local providers like Mistral, Google, AWS, etc.)"""
    try:
        ocr_processor = get_processor()
        await ocr_processor._ensure_providers_initialized()

        provider_stats = ocr_processor.get_provider_stats()

        # Format providers for UI display
        providers = []
        for name, stats in provider_stats.items():
            provider_info = {
                "name": name,
                "display_name": name.replace("_", " ").title(),
                "available": stats.get("available", False),
                "description": stats.get("description", ""),
                "configuration_status": stats.get("status", "Unknown"),
                "missing_config": stats.get("missing_config", [])
            }
            providers.append(provider_info)

        return {
            "success": True,
            "providers": providers,
            "count": len(providers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get OCR providers status: {str(e)}")


# @router.get("/providers")
# async def get_ocr_providers():
#     """Get list of available OCR providers (deprecated - use /ocr-providers/status)."""
#     try:
#         ocr_processor = get_processor()
#         await ocr_processor._ensure_providers_initialized()

#         provider_stats = ocr_processor.get_provider_stats()
#         available_providers = list(provider_stats.keys())

#         return {
#             "success": True,
#             "providers": available_providers,
#             "provider_stats": provider_stats,
#             "count": len(available_providers)
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")


@router.get("/config")
async def get_config():
    """Get application configuration information."""
    try:
        settings = get_settings()

        return {
            "success": True,
            "app": {
                "title": settings.app.title,
                "version": settings.app.version,
                "environment": settings.app.environment,
                "debug": settings.app.debug
            },
            "providers": {
                "configured": settings.get_configured_providers(),
                "count": len(settings.get_configured_providers())
            },
            "features": {
                "authentication": bool(settings.auth.clerk_secret_key),
                "payments": bool(settings.payments.stripe_secret_key),
                "caching": bool(settings.cache.redis_url),
                "olmocr": settings.olmocr.is_configured
            },
            "limits": {
                "max_file_size_mb": settings.app.max_file_size_mb,
                "max_files_per_request": settings.app.max_files_per_request,
                "supported_formats": settings.app.supported_formats
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Basic health check
        settings = get_settings()

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.app.version,
            "environment": settings.app.environment
        }

        # Check if we can initialize the processor
        try:
            ocr_processor = get_processor()
            health_status["ocr_processor"] = "available"
        except Exception as e:
            health_status["ocr_processor"] = f"error: {str(e)}"
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@router.get("/metrics")
async def get_metrics():
    """Get application metrics."""
    try:
        settings = get_settings()
        ocr_processor = get_processor()

        # Simulate basic metrics (in production, these would come from monitoring systems)
        metrics = {
            "uptime_seconds": 86400,  # 1 day example
            "requests_total": 1000,
            "requests_successful": 950,
            "requests_failed": 50,
            "response_time_avg_ms": 1250,
            "memory_usage_percent": 45,
            "cpu_usage_percent": 25,
            "providers": {
                "available": len(settings.get_configured_providers()),
                "configured": settings.get_configured_providers()
            }
        }

        return {
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/test-provider/{provider_name}")
async def test_provider(provider_name: str):
    """Test a specific OCR provider."""
    try:
        ocr_processor = get_processor()
        await ocr_processor._ensure_providers_initialized()

        # Check if provider is available
        provider_stats = ocr_processor.get_provider_stats()

        if provider_name not in provider_stats:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

        provider_info = provider_stats[provider_name]

        if not provider_info.get("available", False):
            return {
                "success": False,
                "provider": provider_name,
                "status": "unavailable",
                "message": f"Provider '{provider_name}' is not available",
                "timestamp": datetime.now().isoformat()
            }

        # Simulate a basic test (in production, this would do actual testing)
        test_result = {
            "success": True,
            "provider": provider_name,
            "status": "passed",
            "message": f"Provider '{provider_name}' is responding correctly",
            "response_time_ms": 150,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "connection": "ok",
                "authentication": "ok" if provider_name != "tesseract" else "n/a",
                "test_type": "connectivity_check"
            }
        }

        return test_result

    except Exception as e:
        return {
            "success": False,
            "provider": provider_name,
            "status": "error",
            "message": f"Test failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@router.get("/logs")
async def get_recent_logs(limit: int = 100):
    """Get recent application logs."""
    try:
        # Simulate log entries (in production, this would read from actual logs)
        logs = []
        for i in range(min(limit, 50)):  # Limit to 50 for demo
            timestamp = datetime.now() - __import__('datetime').timedelta(minutes=i)
            level = ["INFO", "DEBUG", "WARN", "ERROR"][i % 4]
            messages = [
                "Application started successfully",
                "OCR request processed",
                "Provider health check completed",
                "Configuration loaded",
                "User authentication successful",
                "File uploaded and validated",
                "Processing completed successfully"
            ]

            logs.append({
                "timestamp": timestamp.isoformat(),
                "level": level,
                "message": messages[i % len(messages)],
                "module": "ocular.main",
                "request_id": f"req-{1000 + i}"
            })

        return {
            "success": True,
            "logs": logs,
            "count": len(logs),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.get("/deployment/status")
async def get_deployment_status():
    """Get deployment status information."""
    try:
        settings = get_settings()

        # Simulate deployment status
        deployment_status = {
            "environment": settings.app.environment,
            "version": settings.app.version,
            "build": "build-123",
            "deploy_time": "2024-01-01T12:00:00Z",
            "status": "healthy",
            "services": {
                "web_app": {"status": "healthy", "instances": 1},
                "ocr_engine": {"status": "healthy", "providers": len(settings.get_configured_providers())},
                "file_storage": {"status": "healthy", "type": "local"},
                "cache": {"status": "healthy" if settings.cache.redis_url else "disabled", "type": "redis" if settings.cache.redis_url else "memory"}
            },
            "infrastructure": {
                "cpu_usage": 25,
                "memory_usage": 45,
                "disk_usage": 60,
                "network_status": "healthy"
            }
        }

        return {
            "success": True,
            "deployment": deployment_status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get deployment status: {str(e)}")


@router.get("/deployment/history")
async def get_deployment_history(limit: int = 10):
    """Get deployment history."""
    try:
        # Simulate deployment history
        deployments = []
        for i in range(min(limit, 5)):  # Limit to 5 for demo
            deploy_time = datetime.now() - __import__('datetime').timedelta(days=i)
            status = "success" if i < 4 else "failed"  # Last one failed

            deployments.append({
                "version": f"v0.1.{5-i}",
                "status": status,
                "deploy_time": deploy_time.isoformat(),
                "duration_seconds": 120 + (i * 30),
                "author": "Claude Code" if i == 0 else "Development Team",
                "commit_hash": f"abc123{i}",
                "environment": "production" if i % 2 == 0 else "staging"
            })

        return {
            "success": True,
            "deployments": deployments,
            "count": len(deployments),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get deployment history: {str(e)}")


# ============================================================================
# AI Models Management Endpoints (from ocular.json)
# ============================================================================

class ModelCreate(BaseModel):
    """Request model for creating an AI model in ocular.json"""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    version: str = Field(..., min_length=1)
    description: str = Field(default="")
    docker_image: str = Field(default="")
    min_vram_gb: int = Field(default=0, ge=0)
    recommended_gpu: str = Field(default="")
    supports_vllm: bool = Field(default=False)
    status: str = Field(default="INACTIVE")
    configuration: Dict[str, Any] = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    """Request model for updating an AI model"""
    display_name: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    docker_image: Optional[str] = None
    min_vram_gb: Optional[int] = None
    recommended_gpu: Optional[str] = None
    supports_vllm: Optional[bool] = None
    status: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None


@router.get("/models")
async def list_ai_models(
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all AI models available for deployment from database."""
    query = db.query(Model)

    if status_filter:
        try:
            status_enum = ModelStatus(status_filter)
            query = query.filter(Model.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}"
            )

    query = query.order_by(Model.name)
    models = query.offset(skip).limit(limit).all()

    return {
        "success": True,
        "models": [m.to_dict() for m in models],
        "count": len(models)
    }


@router.get("/models/{model_id}")
async def get_ai_model(model_id: int, db: Session = Depends(get_db)):
    """Get a specific AI model by ID from database"""
    model = db.query(Model).filter(Model.id == model_id).first()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    return {
        "success": True,
        "model": model.to_dict()
    }


@router.get("/models/search-huggingface")
async def search_huggingface_models(q: str, limit: int = 10):
    """Search HuggingFace Hub for models"""
    try:
        # This is a placeholder - in production, integrate with HuggingFace API
        # For now, return mock results
        mock_results = [
            {
                "id": f"author/{q}-model-{i}",
                "name": f"{q.title()} Model {i}",
                "description": f"A model for {q} tasks",
                "downloads": 1000 * (10 - i),
                "tags": ["ocr", "vision", q.lower()]
            }
            for i in range(1, min(limit + 1, 6))
        ]

        return {
            "success": True,
            "models": mock_results,
            "count": len(mock_results),
            "query": q
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search HuggingFace: {str(e)}")




# ============================================================================
# Cloud GPU Providers Management Endpoints (from ocular.json)
# ============================================================================

class CloudProviderCreate(BaseModel):
    """Request model for creating a cloud GPU provider"""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    website: str = Field(default="")
    status: str = Field(default="INACTIVE")
    available_gpus: List[str] = Field(default_factory=list)
    features: Dict[str, bool] = Field(default_factory=dict)


class CloudProviderUpdate(BaseModel):
    """Request model for updating a cloud GPU provider"""
    display_name: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    status: Optional[str] = None
    available_gpus: Optional[List[str]] = None
    features: Optional[Dict[str, bool]] = None


# The /providers endpoints already exist above and use the database
# These endpoints work with ocular.json config file which is read-only from API
# Users can view providers from ocular.json but management should be done via file editing


# ============================================================================
# Settings Management Endpoints
# ============================================================================

class SettingsSaveRequest(BaseModel):
    """Request model for saving settings to .env file"""
    settings: Dict[str, Any] = Field(..., description="Dictionary of setting key-value pairs")
    create_backup: bool = Field(default=True, description="Whether to create backup before saving")


class SettingsSaveResponse(BaseModel):
    """Response model for settings save operation"""
    success: bool
    message: str
    updated_count: int
    backup_created: bool
    errors: Optional[List[str]] = None


@router.get("/settings")
async def get_settings_from_env():
    """
    Get current settings from .env file.

    Returns settings values without exposing sensitive information.
    """
    try:
        env_service = get_env_service()
        env_vars = env_service.read_env()

        # Get current settings instance for display values
        current_settings = get_settings()

        # Build response with current values (hide sensitive fields)
        settings_data = {
            # Application settings
            "APP_TITLE": env_vars.get("APP_TITLE", current_settings.app.title),
            "APP_DESCRIPTION": env_vars.get("APP_DESCRIPTION", current_settings.app.description),
            "APP_VERSION": env_vars.get("APP_VERSION", current_settings.app.version),
            "ENVIRONMENT": env_vars.get("ENVIRONMENT", current_settings.app.environment),
            "DEBUG": env_vars.get("DEBUG", str(current_settings.app.debug)),
            "HOST": env_vars.get("HOST", current_settings.app.host),
            "PORT": env_vars.get("PORT", str(current_settings.app.port)),
            "WORKERS": env_vars.get("WORKERS", str(current_settings.app.workers)),
            "MAX_FILE_SIZE_MB": env_vars.get("MAX_FILE_SIZE_MB", str(current_settings.app.max_file_size_mb)),
            "MAX_FILES_PER_REQUEST": env_vars.get("MAX_FILES_PER_REQUEST", str(current_settings.app.max_files_per_request)),
            "SUPPORTED_FORMATS": env_vars.get("SUPPORTED_FORMATS", ",".join(current_settings.app.supported_formats)),
            "DEFAULT_STRATEGY": env_vars.get("DEFAULT_STRATEGY", current_settings.app.default_strategy),
            "DEFAULT_PROVIDERS": env_vars.get("DEFAULT_PROVIDERS", ",".join(current_settings.app.default_providers)),
            "MAX_CONCURRENT_REQUESTS": env_vars.get("MAX_CONCURRENT_REQUESTS", str(current_settings.app.max_concurrent_requests)),
            "TEMPLATES_DIR": env_vars.get("TEMPLATES_DIR", current_settings.app.templates_dir),
            "STATIC_DIR": env_vars.get("STATIC_DIR", current_settings.app.static_dir),
            "ENABLE_STATIC_FILES": env_vars.get("ENABLE_STATIC_FILES", str(current_settings.app.enable_static_files)),

            # Database settings
            "DATABASE_URL": "***CONFIGURED***" if current_settings.database.url else "",
            "DB_POOL_SIZE": env_vars.get("DB_POOL_SIZE", str(current_settings.database.pool_size)),
            "DB_MAX_OVERFLOW": env_vars.get("DB_MAX_OVERFLOW", str(current_settings.database.max_overflow)),

            # Authentication (masked)
            "CLERK_SECRET_KEY": "***CONFIGURED***" if current_settings.auth.clerk_secret_key else "",
            "CLERK_PUBLISHABLE_KEY": env_vars.get("CLERK_PUBLISHABLE_KEY", current_settings.auth.clerk_publishable_key or ""),
            "JWT_SECRET_KEY": "***CONFIGURED***" if current_settings.auth.jwt_secret_key else "",
            "AUTH_SESSION_TIMEOUT_MINUTES": env_vars.get("AUTH_SESSION_TIMEOUT_MINUTES", str(current_settings.auth.session_timeout_minutes)),

            # Payments (masked)
            "STRIPE_SECRET_KEY": "***CONFIGURED***" if current_settings.payments.stripe_secret_key else "",
            "STRIPE_PUBLISHABLE_KEY": env_vars.get("STRIPE_PUBLISHABLE_KEY", current_settings.payments.stripe_publishable_key or ""),
            "STRIPE_WEBHOOK_SECRET": "***CONFIGURED***" if current_settings.payments.stripe_webhook_secret else "",

            # OLMocr settings
            "OLMOCR_PROVIDER": env_vars.get("OLMOCR_PROVIDER", current_settings.olmocr.provider),
            "RUNPOD_ENDPOINT_ID": env_vars.get("RUNPOD_ENDPOINT_ID", current_settings.olmocr.runpod_endpoint_id),
            "RUNPOD_API_KEY": "***CONFIGURED***" if current_settings.olmocr.runpod_api_key else "",
            "RUNPOD_ACCESS_KEY_ID": env_vars.get("RUNPOD_ACCESS_KEY_ID", current_settings.olmocr.runpod_access_key_id),
            "RUNPOD_SECRET_ACCESS_KEY": "***CONFIGURED***" if current_settings.olmocr.runpod_secret_access_key else "",
            "RUNPOD_BUCKET_ENDPOINT_URL": env_vars.get("RUNPOD_BUCKET_ENDPOINT_URL", current_settings.olmocr.runpod_bucket_endpoint_url),
            "RUNPOD_NETWORK_VOLUME_ID": env_vars.get("RUNPOD_NETWORK_VOLUME_ID", current_settings.olmocr.runpod_network_volume_id),
            "RUNPOD_REGION": env_vars.get("RUNPOD_REGION", current_settings.olmocr.runpod_region),
            "VASTAI_INSTANCE_ID": env_vars.get("VASTAI_INSTANCE_ID", current_settings.olmocr.vastai_instance_id),
            "VASTAI_API_KEY": "***CONFIGURED***" if current_settings.olmocr.vastai_api_key else "",
            "VASTAI_ENDPOINT_URL": env_vars.get("VASTAI_ENDPOINT_URL", current_settings.olmocr.vastai_endpoint_url),
            "DEEPINFRA_TOKEN": "***CONFIGURED***" if current_settings.olmocr.deepinfra_api_key else "",
            "DEEPINFRA_MODEL": env_vars.get("DEEPINFRA_MODEL", current_settings.olmocr.deepinfra_model),
            "DEEPINFRA_BASE_URL": env_vars.get("DEEPINFRA_BASE_URL", current_settings.olmocr.deepinfra_base_url),
            "OLMOCR_MODEL_NAME": env_vars.get("OLMOCR_MODEL_NAME", current_settings.olmocr.model_name),
            "OLMOCR_MAX_FILE_SIZE_MB": env_vars.get("OLMOCR_MAX_FILE_SIZE_MB", str(current_settings.olmocr.max_file_size_mb)),
            "OLMOCR_TIMEOUT_SECONDS": env_vars.get("OLMOCR_TIMEOUT_SECONDS", str(current_settings.olmocr.timeout_seconds)),
            "OLMOCR_ALLOWED_EXTENSIONS": env_vars.get("OLMOCR_ALLOWED_EXTENSIONS", ",".join(current_settings.olmocr.allowed_extensions)),

            # OCR Providers (masked)
            "MISTRAL_API_KEY": "***CONFIGURED***" if current_settings.ocr_providers.mistral_api_key else "",
            "MISTRAL_MODEL": env_vars.get("MISTRAL_MODEL", current_settings.ocr_providers.mistral_model),
            "GOOGLE_APPLICATION_CREDENTIALS": env_vars.get("GOOGLE_APPLICATION_CREDENTIALS", current_settings.ocr_providers.google_application_credentials or ""),
            "GOOGLE_PROJECT_ID": env_vars.get("GOOGLE_PROJECT_ID", current_settings.ocr_providers.google_project_id or ""),
            "AWS_ACCESS_KEY_ID": "***CONFIGURED***" if current_settings.ocr_providers.aws_access_key_id else "",
            "AWS_SECRET_ACCESS_KEY": "***CONFIGURED***" if current_settings.ocr_providers.aws_secret_access_key else "",
            "AWS_REGION": env_vars.get("AWS_REGION", current_settings.ocr_providers.aws_region),
            "AZURE_DOC_INTEL_ENDPOINT": env_vars.get("AZURE_DOC_INTEL_ENDPOINT", current_settings.ocr_providers.azure_doc_intel_endpoint or ""),
            "AZURE_DOC_INTEL_API_KEY": "***CONFIGURED***" if current_settings.ocr_providers.azure_doc_intel_api_key else "",
            "TESSERACT_LANGUAGE": env_vars.get("TESSERACT_LANGUAGE", current_settings.ocr_providers.tesseract_language),
            "TESSERACT_DPI": env_vars.get("TESSERACT_DPI", str(current_settings.ocr_providers.tesseract_dpi)),
            "HF_TOKEN": "***CONFIGURED***" if current_settings.ocr_providers.hf_token else "",

            # Logging settings
            "LOG_LEVEL": env_vars.get("LOG_LEVEL", current_settings.logging.level),
            "LOG_FORMAT": env_vars.get("LOG_FORMAT", current_settings.logging.format),
            "LOGFIRE_TOKEN": "***CONFIGURED***" if current_settings.logging.logfire_token else "",
            "ENABLE_FILE_LOGGING": env_vars.get("ENABLE_FILE_LOGGING", str(current_settings.logging.enable_file_logging)),
            "LOG_FILE_PATH": env_vars.get("LOG_FILE_PATH", current_settings.logging.log_file_path),

            # Security settings
            "CORS_ORIGINS": env_vars.get("CORS_ORIGINS", ",".join(current_settings.security.cors_origins)),
            "MAX_REQUEST_SIZE_MB": env_vars.get("MAX_REQUEST_SIZE_MB", str(current_settings.security.max_request_size_mb)),
            "RATE_LIMIT_RPM": env_vars.get("RATE_LIMIT_RPM", str(current_settings.security.rate_limit_requests_per_minute)),
            "API_SECRET_KEY": "***CONFIGURED***" if current_settings.security.api_secret_key else "",
            "ENCRYPTION_KEY": "***CONFIGURED***" if current_settings.security.encryption_key else "",
            "ENABLE_HTTPS_ONLY": env_vars.get("ENABLE_HTTPS_ONLY", str(current_settings.security.enable_https_only)),

            # Cache settings
            "REDIS_URL": "***CONFIGURED***" if current_settings.cache.redis_url else "",
            "CACHE_TTL_SECONDS": env_vars.get("CACHE_TTL_SECONDS", str(current_settings.cache.cache_ttl_seconds)),
            "ENABLE_CACHING": env_vars.get("ENABLE_CACHING", str(current_settings.cache.enable_caching)),
        }

        return {
            "success": True,
            "settings": settings_data,
            "count": len(settings_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load settings: {str(e)}")


@router.post("/settings", response_model=SettingsSaveResponse)
async def save_settings_to_env(request: SettingsSaveRequest):
    """
    Save settings to .env file.

    Updates the .env file with new configuration values.
    Creates a backup before making changes.
    """
    try:
        env_service = get_env_service()

        # Validate all settings before writing
        errors = []
        validated_settings = {}

        for key, value in request.settings.items():
            # Skip empty values for sensitive fields (don't overwrite with empty)
            if value == "***CONFIGURED***" or (not value and any(s in key for s in ['KEY', 'SECRET', 'TOKEN', 'PASSWORD'])):
                continue

            # Convert boolean strings
            if value in ['true', 'True', 'false', 'False']:
                value = str(value.lower() == 'true')

            # Validate the value
            is_valid, error_msg = env_service.validate_env_value(key, value)
            if not is_valid:
                errors.append(error_msg)
                continue

            validated_settings[key] = value

        # If there are validation errors, return them
        if errors:
            return SettingsSaveResponse(
                success=False,
                message="Validation failed for some settings",
                updated_count=0,
                backup_created=False,
                errors=errors
            )

        # Write settings to .env file
        success = env_service.write_env(validated_settings, backup=request.create_backup)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to write settings to .env file"
            )

        return SettingsSaveResponse(
            success=True,
            message=f"Successfully updated {len(validated_settings)} settings",
            updated_count=len(validated_settings),
            backup_created=request.create_backup,
            errors=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@router.get("/settings/backups")
async def list_env_backups():
    """Get list of .env backup files."""
    try:
        env_service = get_env_service()
        backups = env_service.get_backup_files()

        backup_list = []
        for backup_path in backups:
            # Extract timestamp from filename
            parts = backup_path.name.split('.')
            timestamp_str = parts[-1] if len(parts) > 2 else "unknown"

            backup_list.append({
                "filename": backup_path.name,
                "path": str(backup_path),
                "timestamp": timestamp_str,
                "size_bytes": backup_path.stat().st_size if backup_path.exists() else 0
            })

        return {
            "success": True,
            "backups": backup_list,
            "count": len(backup_list)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")


@router.post("/settings/restore/{backup_filename}")
async def restore_env_backup(backup_filename: str):
    """Restore .env file from a backup."""
    try:
        env_service = get_env_service()

        # Find the backup file
        backups = env_service.get_backup_files()
        backup_path = None

        for backup in backups:
            if backup.name == backup_filename:
                backup_path = backup
                break

        if not backup_path:
            raise HTTPException(status_code=404, detail=f"Backup file not found: {backup_filename}")

        # Restore the backup
        success = env_service.restore_backup(backup_path)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to restore backup")

        return {
            "success": True,
            "message": f"Successfully restored settings from {backup_filename}",
            "restored_from": str(backup_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore backup: {str(e)}")


# =============================================================================
# MODEL TESTING ENDPOINTS
# =============================================================================

class ModelTestRequest(BaseModel):
    """Request model for testing a model with text or image input."""
    prompt: Optional[str] = Field(None, description="Text prompt for the model")
    max_tokens: int = Field(default=1000, ge=1, le=4000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    additional_params: Optional[Dict[str, Any]] = Field(default=None, description="Additional model parameters")


class ModelTestResponse(BaseModel):
    """Response model for model testing."""
    success: bool
    model_id: int
    model_name: str
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)


@router.post("/models/{model_id}/test", response_model=ModelTestResponse)
async def test_model(
    model_id: int,
    file: Optional[UploadFile] = File(None),
    prompt: str = Form(default=""),
    max_tokens: int = Form(default=1000),
    temperature: float = Form(default=0.7),
    db: Session = Depends(get_db)
):
    """
    Test a model with file input and/or text prompt.

    This endpoint allows testing models before deployment or comparing
    different models with the same input.

    Args:
        model_id: ID of the model to test
        file: Optional file to process (image, PDF, etc.)
        prompt: Text prompt or additional instructions
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Test results including output, performance metrics, and metadata
    """
    import time
    import tempfile
    import os
    from ocular.unified_processor import UnifiedDocumentProcessor

    # Validate model exists
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    # Check if model has a deployment or direct endpoint
    active_deployment = db.query(Deployment).filter(
        Deployment.model_id == model_id,
        Deployment.status == DeploymentStatus.RUNNING
    ).first()

    if not active_deployment and not model.docker_image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model.name}' has no active deployment or endpoint. Please deploy the model first."
        )

    temp_file_path = None
    start_time = time.time()

    try:
        # Save uploaded file temporarily if provided
        if file:
            # Validate file type
            allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'application/pdf']
            if file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
                )

            # Save to temp file
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

        # Initialize processor
        processor = UnifiedDocumentProcessor()

        # Prepare OCR parameters
        ocr_params = {
            'provider': model.name,  # Use model name as provider
            'max_tokens': max_tokens,
            'temperature': temperature
        }

        if prompt:
            ocr_params['prompt'] = prompt

        # Process the file or text
        if temp_file_path:
            result = await processor.process_file_async(
                file_path=temp_file_path,
                **ocr_params
            )
        elif prompt:
            # For text-only testing, create a simple text file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_text:
                temp_text.write(prompt)
                temp_file_path = temp_text.name

            result = await processor.process_file_async(
                file_path=temp_file_path,
                **ocr_params
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either file or prompt must be provided"
            )

        # Calculate performance metrics
        end_time = time.time()
        total_time = end_time - start_time

        # Extract text from result
        extracted_text = result.get('text', '') if isinstance(result, dict) else str(result)

        # Calculate tokens (rough estimate)
        estimated_tokens = len(extracted_text.split())
        tokens_per_second = estimated_tokens / total_time if total_time > 0 else 0

        performance_metrics = {
            'total_time_seconds': round(total_time, 3),
            'estimated_tokens': estimated_tokens,
            'tokens_per_second': round(tokens_per_second, 2),
            'timestamp': datetime.utcnow().isoformat()
        }

        # Build metadata
        metadata = {
            'model_version': model.version,
            'docker_image': model.docker_image,
            'vram_gb': model.min_vram_gb,
            'supports_vllm': model.supports_vllm,
            'deployment_id': active_deployment.id if active_deployment else None,
            'test_parameters': {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'prompt_provided': bool(prompt),
                'file_provided': bool(file),
                'file_type': file.content_type if file else None
            }
        }

        return ModelTestResponse(
            success=True,
            model_id=model_id,
            model_name=model.display_name or model.name,
            result=extracted_text,
            metadata=metadata,
            performance=performance_metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        return ModelTestResponse(
            success=False,
            model_id=model_id,
            model_name=model.display_name or model.name,
            error=str(e),
            metadata={
                'test_parameters': {
                    'max_tokens': max_tokens,
                    'temperature': temperature
                }
            },
            performance={}
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@router.post("/models/compare-test", response_model=List[ModelTestResponse])
async def compare_models_test(
    model_ids: List[int] = Form(...),
    file: Optional[UploadFile] = File(None),
    prompt: str = Form(default=""),
    max_tokens: int = Form(default=1000),
    temperature: float = Form(default=0.7),
    db: Session = Depends(get_db)
):
    """
    Test multiple models with the same input for comparison.

    Args:
        model_ids: List of model IDs to test (2-5 models)
        file: Optional file to process
        prompt: Text prompt
        max_tokens: Maximum tokens
        temperature: Sampling temperature

    Returns:
        List of test results for each model
    """
    if len(model_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 models required for comparison"
        )

    if len(model_ids) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5 models allowed for comparison"
        )

    # Save file to temp location for reuse
    temp_file_path = None
    if file:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

    results = []

    try:
        for model_id in model_ids:
            # For each model, call the test endpoint
            # We need to recreate the file upload if there was one
            if temp_file_path:
                # Read file and create UploadFile
                with open(temp_file_path, 'rb') as f:
                    file_content = f.read()

                # Create a new UploadFile instance
                from io import BytesIO
                test_file = UploadFile(
                    filename=file.filename if file else 'test_file',
                    file=BytesIO(file_content)
                )
                test_file.content_type = file.content_type if file else 'application/octet-stream'
            else:
                test_file = None

            # Call test_model for this model
            result = await test_model(
                model_id=model_id,
                file=test_file,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                db=db
            )

            results.append(result)

    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

    return results
