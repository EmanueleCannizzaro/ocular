"""
Analytics API Router for Ocular OCR Platform.

Provides endpoints for deployment analytics, cost tracking, and performance metrics.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ocular.database.session import get_db
from ocular.services.analytics_service import get_analytics_service


# Initialize router
router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


# Response Models
class DashboardSummary(BaseModel):
    """Dashboard summary response model."""
    active_deployments: int
    total_deployments: int
    total_models: int
    total_providers: int
    recent_deployments_24h: int
    recent_failures_24h: int
    estimated_daily_cost: float
    gpu_utilization_percent: float
    total_gpus: int
    active_gpus: int
    timestamp: str


@router.get("/dashboard", response_model=DashboardSummary)
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """
    Get comprehensive dashboard summary with key metrics.

    Returns:
        Dashboard summary including deployment counts, costs, and GPU utilization
    """
    analytics_service = get_analytics_service(db)
    return analytics_service.get_dashboard_summary()


@router.get("/deployment-timeline")
async def get_deployment_timeline(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    provider_id: Optional[int] = Query(default=None, description="Filter by provider ID"),
    model_id: Optional[int] = Query(default=None, description="Filter by model ID"),
    db: Session = Depends(get_db)
):
    """
    Get deployment timeline data.

    Args:
        days: Number of days to analyze (1-365)
        provider_id: Optional provider filter
        model_id: Optional model filter

    Returns:
        Timeline data with daily deployment counts and statuses
    """
    analytics_service = get_analytics_service(db)
    return analytics_service.get_deployment_timeline(
        days=days,
        provider_id=provider_id,
        model_id=model_id
    )


@router.get("/cost-analytics")
async def get_cost_analytics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    provider_id: Optional[int] = Query(default=None, description="Filter by provider ID"),
    db: Session = Depends(get_db)
):
    """
    Get cost analytics and breakdown.

    Args:
        days: Number of days to analyze (1-365)
        provider_id: Optional provider filter

    Returns:
        Cost breakdown by date, GPU type, and model
    """
    analytics_service = get_analytics_service(db)
    return analytics_service.get_cost_analytics(
        days=days,
        provider_id=provider_id
    )


@router.get("/gpu-utilization")
async def get_gpu_utilization(
    provider_id: Optional[int] = Query(default=None, description="Filter by provider ID"),
    db: Session = Depends(get_db)
):
    """
    Get GPU utilization metrics.

    Args:
        provider_id: Optional provider filter

    Returns:
        GPU utilization statistics by type
    """
    analytics_service = get_analytics_service(db)
    return analytics_service.get_gpu_utilization(provider_id=provider_id)


@router.get("/model-usage")
async def get_model_usage_stats(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get model usage statistics.

    Args:
        days: Number of days to analyze (1-365)

    Returns:
        Model usage statistics and deployment counts
    """
    analytics_service = get_analytics_service(db)
    return analytics_service.get_model_usage_stats(days=days)


@router.get("/performance-trends")
async def get_performance_trends(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get performance trends over time.

    Args:
        days: Number of days to analyze (1-365)

    Returns:
        Performance metrics including success rates and trends
    """
    analytics_service = get_analytics_service(db)
    return analytics_service.get_performance_trends(days=days)
