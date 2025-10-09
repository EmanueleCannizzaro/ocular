"""
API Router for Model Version Management.

Provides endpoints for version tracking, comparison, rollback, and A/B testing.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ocular.database.session import get_db
from ocular.services.version_service import VersionService
from ocular.database.models import VersionStatus


router = APIRouter()


# Pydantic Models for API

class VersionCreateRequest(BaseModel):
    """Request model for creating a new version."""
    model_id: int = Field(..., description="Model ID")
    version_number: str = Field(..., description="Version identifier (e.g., 'v1.0.0')")
    version_name: Optional[str] = Field(None, description="Human-readable version name")
    version_tag: Optional[str] = Field(None, description="Tag like 'latest', 'stable', 'beta'")
    changelog: Optional[str] = Field(None, description="Description of changes")
    docker_image: Optional[str] = Field(None, description="Docker image for this version")
    huggingface_repo: Optional[str] = Field(None, description="HuggingFace repository")
    config: Optional[Dict[str, Any]] = Field(None, description="Version-specific configuration")
    created_by: Optional[str] = Field(None, description="User who created this version")


class VersionUpdateRequest(BaseModel):
    """Request model for updating version information."""
    version_name: Optional[str] = None
    version_tag: Optional[str] = None
    changelog: Optional[str] = None
    status: Optional[VersionStatus] = None
    docker_image: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class VersionMetricsUpdate(BaseModel):
    """Request model for updating version metrics."""
    benchmark_scores: Optional[Dict[str, Any]] = None
    avg_inference_time_ms: Optional[float] = None
    avg_tokens_per_second: Optional[float] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=100.0)


class ABTestSetup(BaseModel):
    """Request model for setting up A/B test."""
    version_id_a: int = Field(..., description="First version ID (baseline)")
    version_id_b: int = Field(..., description="Second version ID (test)")
    traffic_split: float = Field(50.0, ge=0.0, le=100.0, description="% traffic for version B")
    duration_days: int = Field(7, ge=1, le=90, description="Duration in days")


class VersionResponse(BaseModel):
    """Response model for version data."""
    id: int
    model_id: int
    model_name: Optional[str]
    version_number: str
    version_tag: Optional[str]
    version_name: Optional[str]
    changelog: Optional[str]
    status: str
    is_active: bool
    docker_image: Optional[str]
    huggingface_repo: Optional[str]
    model_hash: Optional[str]
    model_size_gb: Optional[float]
    config: Dict[str, Any]
    deployment_config: Dict[str, Any]
    benchmark_scores: Dict[str, Any]
    avg_inference_time_ms: Optional[float]
    avg_tokens_per_second: Optional[float]
    quality_score: Optional[float]
    ab_test_enabled: bool
    ab_test_traffic_percentage: Optional[float]
    ab_test_start_date: Optional[str]
    ab_test_end_date: Optional[str]
    cloud_storage_path: Optional[str]
    created_by: Optional[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    deployed_at: Optional[str]
    deprecated_at: Optional[str]


# API Endpoints

@router.post("/versions", response_model=VersionResponse, status_code=status.HTTP_201_CREATED)
async def create_version(
    request: VersionCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new model version.

    This endpoint creates a new version entry for a model, allowing you to track
    different iterations with metadata, configuration, and performance metrics.
    """
    try:
        version = VersionService.create_version(
            db=db,
            model_id=request.model_id,
            version_number=request.version_number,
            version_name=request.version_name,
            version_tag=request.version_tag,
            changelog=request.changelog,
            docker_image=request.docker_image,
            huggingface_repo=request.huggingface_repo,
            config=request.config,
            created_by=request.created_by,
        )
        return version.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create version: {str(e)}")


@router.get("/versions/{version_id}", response_model=VersionResponse)
async def get_version(
    version_id: int,
    db: Session = Depends(get_db)
):
    """Get details of a specific version."""
    version = VersionService.get_version(db, version_id)
    if not version:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")

    return version.to_dict()


@router.get("/models/{model_id}/versions", response_model=List[VersionResponse])
async def get_model_versions(
    model_id: int,
    status: Optional[VersionStatus] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get all versions for a specific model.

    Returns versions in reverse chronological order (newest first).
    """
    try:
        versions = VersionService.get_model_versions(
            db=db,
            model_id=model_id,
            status=status,
            limit=limit
        )
        return [v.to_dict() for v in versions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch versions: {str(e)}")


@router.get("/models/{model_id}/versions/active", response_model=VersionResponse)
async def get_active_version(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get the currently active version for a model."""
    version = VersionService.get_active_version(db, model_id)
    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"No active version found for model {model_id}"
        )

    return version.to_dict()


@router.post("/versions/{version_id}/activate", response_model=VersionResponse)
async def activate_version(
    version_id: int,
    deactivate_others: bool = True,
    db: Session = Depends(get_db)
):
    """
    Activate a specific version.

    By default, this will deactivate all other versions of the same model.
    """
    try:
        version = VersionService.activate_version(
            db=db,
            version_id=version_id,
            deactivate_others=deactivate_others
        )
        return version.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate version: {str(e)}")


@router.post("/models/{model_id}/rollback/{version_id}", response_model=VersionResponse)
async def rollback_to_version(
    model_id: int,
    version_id: int,
    db: Session = Depends(get_db)
):
    """
    Rollback to a previous version.

    This activates the specified version and deactivates all others.
    """
    try:
        version = VersionService.rollback_to_version(
            db=db,
            model_id=model_id,
            version_id=version_id
        )
        return version.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rollback: {str(e)}")


@router.post("/versions/compare", response_model=List[Dict[str, Any]])
async def compare_versions(
    version_ids: List[int],
    db: Session = Depends(get_db)
):
    """
    Compare multiple versions side-by-side.

    Accepts 2-5 version IDs and returns detailed comparison data including
    performance metrics, deployment statistics, and configuration differences.
    """
    try:
        comparison_data = VersionService.compare_versions(db, version_ids)
        return comparison_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare versions: {str(e)}")


@router.put("/versions/{version_id}/metrics", response_model=VersionResponse)
async def update_version_metrics(
    version_id: int,
    metrics: VersionMetricsUpdate,
    db: Session = Depends(get_db)
):
    """
    Update performance metrics for a version.

    This endpoint is typically called after running benchmarks or tests on a version.
    """
    try:
        version = VersionService.update_version_metrics(
            db=db,
            version_id=version_id,
            benchmark_scores=metrics.benchmark_scores,
            avg_inference_time_ms=metrics.avg_inference_time_ms,
            avg_tokens_per_second=metrics.avg_tokens_per_second,
            quality_score=metrics.quality_score,
        )
        return version.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update metrics: {str(e)}")


@router.post("/versions/ab-test/setup", response_model=List[VersionResponse])
async def setup_ab_test(
    request: ABTestSetup,
    db: Session = Depends(get_db)
):
    """
    Setup A/B testing between two versions.

    Configures traffic splitting between a baseline version (A) and a test version (B).
    """
    try:
        version_a, version_b = VersionService.setup_ab_test(
            db=db,
            version_id_a=request.version_id_a,
            version_id_b=request.version_id_b,
            traffic_split=request.traffic_split,
            duration_days=request.duration_days,
        )
        return [version_a.to_dict(), version_b.to_dict()]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup A/B test: {str(e)}")


@router.post("/versions/{version_id}/ab-test/end", response_model=VersionResponse)
async def end_ab_test(
    version_id: int,
    promote_to_active: bool = False,
    db: Session = Depends(get_db)
):
    """
    End an A/B test for a version.

    Optionally promotes the version to active if it performed better.
    """
    try:
        version = VersionService.end_ab_test(
            db=db,
            version_id=version_id,
            promote_to_active=promote_to_active
        )
        return version.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end A/B test: {str(e)}")


@router.post("/versions/{version_id}/deprecate", response_model=VersionResponse)
async def deprecate_version(
    version_id: int,
    reason: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Deprecate a version.

    Marks the version as deprecated and optionally records the reason.
    """
    try:
        version = VersionService.deprecate_version(
            db=db,
            version_id=version_id,
            reason=reason
        )
        return version.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deprecate version: {str(e)}")


@router.get("/models/{model_id}/versions/timeline", response_model=List[Dict[str, Any]])
async def get_version_timeline(
    model_id: int,
    db: Session = Depends(get_db)
):
    """
    Get version history timeline for a model.

    Returns all version events (creation, deployment, deprecation) in chronological order.
    """
    try:
        timeline = VersionService.get_version_timeline(db, model_id)
        return timeline
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch timeline: {str(e)}")
