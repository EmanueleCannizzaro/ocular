"""
Observability API endpoints for logs, traces, metrics, and alerts.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query

from ocular.models.observability import (
    LogQueryRequest,
    LogQueryResponse,
    LogStatisticsResponse,
    TraceQueryResponse,
    PerformanceQueryResponse,
    PerformanceSummaryResponse,
    MetricRecordRequest,
    MetricQueryResponse,
    MetricSummaryResponse,
    AlertTriggerRequest,
    AlertQueryResponse,
    ErrorSummaryResponse,
    DashboardSummaryResponse,
)
from ocular.routers import templates
from ocular.services.observability_service import (
    get_observability_service,
    LogLevel,
    AlertSeverity,
    MetricType,
    LogEntry,
    TraceSpan,
    PerformanceProfile,
    CustomMetric,
    Alert,
)


router = APIRouter()


# ==================== Logging Endpoints ====================

@router.get("/logs", response_model=LogQueryResponse)
async def query_logs(
    level: Optional[LogLevel] = None,
    source: Optional[str] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[int] = None,
    since_hours: Optional[int] = Query(None, description="Only logs from last N hours"),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Query logs with filters.

    Args:
        level: Filter by log level
        source: Filter by source
        trace_id: Filter by trace ID
        user_id: Filter by user ID
        since_hours: Only logs from last N hours
        limit: Maximum number of logs

    Returns:
        Filtered logs
    """
    obs = get_observability_service()

    since = None
    if since_hours:
        since = datetime.utcnow() - timedelta(hours=since_hours)

    logs = obs.get_logs(
        level=level,
        source=source,
        trace_id=trace_id,
        user_id=user_id,
        since=since,
        limit=limit,
    )

    return LogQueryResponse(logs=logs, total=len(logs))


@router.get("/logs/statistics", response_model=LogStatisticsResponse)
async def get_log_statistics(
    hours: int = Query(24, description="Time period in hours"),
):
    """
    Get log statistics for a time period.

    Args:
        hours: Number of hours to analyze

    Returns:
        Log statistics
    """
    obs = get_observability_service()
    stats = obs.get_log_statistics(hours=hours)
    return LogStatisticsResponse(**stats)


@router.get("/logs/stream")
async def stream_logs():
    """
    Stream logs in real-time using Server-Sent Events (SSE).

    Returns:
        SSE stream of logs
    """
    # This would implement SSE streaming in production
    # For now, return a message
    return {"message": "Real-time log streaming endpoint (implement SSE)"}


# ==================== Tracing Endpoints ====================

@router.get("/traces/{trace_id}", response_model=TraceQueryResponse)
async def get_trace(trace_id: str):
    """
    Get all spans for a trace ID.

    Args:
        trace_id: Trace ID to query

    Returns:
        All spans for the trace
    """
    obs = get_observability_service()
    spans = obs.get_trace(trace_id)
    return TraceQueryResponse(spans=spans, total=len(spans))


@router.get("/traces", response_model=Dict[str, int])
async def get_active_traces():
    """
    Get count of active traces.

    Returns:
        Active trace count
    """
    obs = get_observability_service()
    return {"active_traces": len(obs.active_traces)}


# ==================== Performance Profiling Endpoints ====================

@router.get("/performance/profiles", response_model=PerformanceQueryResponse)
async def query_performance_profiles(
    operation: Optional[str] = None,
    since_hours: Optional[int] = Query(None, description="Only profiles from last N hours"),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Query performance profiles.

    Args:
        operation: Filter by operation name
        since_hours: Only profiles from last N hours
        limit: Maximum number of profiles

    Returns:
        Filtered performance profiles
    """
    obs = get_observability_service()

    since = None
    if since_hours:
        since = datetime.utcnow() - timedelta(hours=since_hours)

    profiles = obs.get_performance_profiles(
        operation=operation,
        since=since,
        limit=limit,
    )

    return PerformanceQueryResponse(profiles=profiles, total=len(profiles))


@router.get("/performance/summary/{operation}", response_model=PerformanceSummaryResponse)
async def get_performance_summary(
    operation: str,
    hours: int = Query(24, description="Time period in hours"),
):
    """
    Get performance summary for an operation.

    Args:
        operation: Operation name
        hours: Time period in hours

    Returns:
        Performance summary with percentiles
    """
    obs = get_observability_service()
    summary = obs.get_performance_summary(operation, hours=hours)
    return PerformanceSummaryResponse(**summary)


# ==================== Metrics Endpoints ====================

@router.post("/metrics")
async def record_metric(request: MetricRecordRequest):
    """
    Record a custom metric.

    Args:
        request: Metric data

    Returns:
        Success message
    """
    obs = get_observability_service()

    obs.record_metric(
        name=request.name,
        value=request.value,
        metric_type=request.metric_type,
        tags=request.tags,
        description=request.description,
    )

    return {"success": True, "message": "Metric recorded"}


@router.get("/metrics", response_model=MetricQueryResponse)
async def query_metrics(
    name: Optional[str] = None,
    metric_type: Optional[MetricType] = None,
    since_hours: Optional[int] = Query(None, description="Only metrics from last N hours"),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Query metrics.

    Args:
        name: Filter by metric name
        metric_type: Filter by metric type
        since_hours: Only metrics from last N hours
        limit: Maximum number of metrics

    Returns:
        Filtered metrics
    """
    obs = get_observability_service()

    since = None
    if since_hours:
        since = datetime.utcnow() - timedelta(hours=since_hours)

    metrics = obs.get_metrics(
        name=name,
        metric_type=metric_type,
        since=since,
        limit=limit,
    )

    return MetricQueryResponse(metrics=metrics, total=len(metrics))


@router.get("/metrics/summary/{metric_name}", response_model=MetricSummaryResponse)
async def get_metric_summary(metric_name: str):
    """
    Get aggregated metric summary.

    Args:
        metric_name: Metric name

    Returns:
        Metric summary with min/max/avg
    """
    obs = get_observability_service()
    summary = obs.get_metric_summary(metric_name)
    return MetricSummaryResponse(**summary)


@router.get("/metrics/names")
async def get_metric_names():
    """
    Get list of all metric names.

    Returns:
        List of metric names
    """
    obs = get_observability_service()
    return {"metric_names": list(obs.metric_aggregations.keys())}


# ==================== Error Tracking Endpoints ====================

@router.get("/errors/summary", response_model=ErrorSummaryResponse)
async def get_error_summary(
    hours: int = Query(24, description="Time period in hours"),
):
    """
    Get error summary.

    Args:
        hours: Time period in hours

    Returns:
        Error summary with counts by type
    """
    obs = get_observability_service()
    summary = obs.get_error_summary(hours=hours)
    return ErrorSummaryResponse(**summary)


@router.get("/errors/{error_type}")
async def get_error_samples(
    error_type: str,
    limit: int = Query(10, ge=1, le=50),
):
    """
    Get error samples for a specific error type.

    Args:
        error_type: Error type
        limit: Maximum number of samples

    Returns:
        Recent error samples
    """
    obs = get_observability_service()
    samples = list(obs.error_samples.get(error_type, []))[:limit]
    return {"error_type": error_type, "samples": samples, "total": len(samples)}


# ==================== Alerting Endpoints ====================

@router.post("/alerts")
async def trigger_alert(request: AlertTriggerRequest):
    """
    Manually trigger an alert.

    Args:
        request: Alert data

    Returns:
        Created alert
    """
    obs = get_observability_service()

    alert = obs.trigger_alert(
        severity=request.severity,
        title=request.title,
        description=request.description,
        source=request.source,
        triggered_by=request.triggered_by,
        metadata=request.metadata,
    )

    return alert


@router.get("/alerts", response_model=AlertQueryResponse)
async def query_alerts(
    severity: Optional[AlertSeverity] = None,
    resolved: Optional[bool] = None,
    since_hours: Optional[int] = Query(None, description="Only alerts from last N hours"),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Query alerts.

    Args:
        severity: Filter by severity
        resolved: Filter by resolved status
        since_hours: Only alerts from last N hours
        limit: Maximum number of alerts

    Returns:
        Filtered alerts
    """
    obs = get_observability_service()

    since = None
    if since_hours:
        since = datetime.utcnow() - timedelta(hours=since_hours)

    alerts = obs.get_alerts(
        severity=severity,
        resolved=resolved,
        since=since,
        limit=limit,
    )

    return AlertQueryResponse(alerts=alerts, total=len(alerts))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """
    Resolve an alert.

    Args:
        alert_id: Alert ID

    Returns:
        Resolved alert
    """
    obs = get_observability_service()
    alert = obs.resolve_alert(alert_id)

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return alert


# ==================== Dashboard Endpoint ====================

@router.get("/dashboard", response_model=DashboardSummaryResponse)
async def get_observability_dashboard(
    hours: int = Query(24, description="Time period in hours"),
):
    """
    Get comprehensive observability dashboard data.

    Args:
        hours: Time period for statistics

    Returns:
        Complete dashboard summary
    """
    obs = get_observability_service()

    log_stats = obs.get_log_statistics(hours=hours)
    error_summary = obs.get_error_summary(hours=hours)

    # Count active (unresolved) alerts
    active_alerts = len([a for a in obs.alerts if not a.resolved])

    # Count total metrics
    total_metrics = len(obs.metrics)

    # Count active traces
    active_traces = len(obs.active_traces)

    return DashboardSummaryResponse(
        log_stats=LogStatisticsResponse(**log_stats),
        error_summary=ErrorSummaryResponse(**error_summary),
        active_alerts=active_alerts,
        total_metrics=total_metrics,
        active_traces=active_traces,
    )


# ==================== Health Check ====================

@router.get("/health")
async def observability_health():
    """Check observability service health."""
    obs = get_observability_service()

    return {
        "status": "healthy",
        "stats": {
            "total_logs": len(obs.logs),
            "total_traces": len(obs.trace_spans),
            "total_metrics": len(obs.metrics),
            "total_alerts": len(obs.alerts),
            "active_alerts": len([a for a in obs.alerts if not a.resolved]),
        },
    }
