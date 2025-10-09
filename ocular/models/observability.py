from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from ocular.services.observability_service import (
    LogLevel,
    AlertSeverity,
    MetricType,
    LogEntry,
    TraceSpan,
    PerformanceProfile,
    CustomMetric,
    Alert,
)


# ==================== Request/Response Models ====================

class LogQueryRequest(BaseModel):
    """Log query request."""
    level: Optional[LogLevel] = None
    source: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[int] = None
    since_hours: Optional[int] = None
    limit: int = 100


class LogQueryResponse(BaseModel):
    """Log query response."""
    logs: List[LogEntry]
    total: int


class LogStatisticsResponse(BaseModel):
    """Log statistics response."""
    total_logs: int
    period_hours: int
    by_level: Dict[str, int]
    by_source: Dict[str, int]
    error_rate: float


class TraceQueryResponse(BaseModel):
    """Trace query response."""
    spans: List[TraceSpan]
    total: int


class PerformanceQueryResponse(BaseModel):
    """Performance query response."""
    profiles: List[PerformanceProfile]
    total: int


class PerformanceSummaryResponse(BaseModel):
    """Performance summary response."""
    operation: str
    count: int
    duration_ms: Optional[Dict[str, float]] = None


class MetricRecordRequest(BaseModel):
    """Metric recording request."""
    name: str
    value: float
    metric_type: MetricType
    tags: Optional[Dict[str, str]] = None
    description: Optional[str] = None


class MetricQueryResponse(BaseModel):
    """Metric query response."""
    metrics: List[CustomMetric]
    total: int


class MetricSummaryResponse(BaseModel):
    """Metric summary response."""
    name: str
    count: int
    min: Optional[float] = None
    max: Optional[float] = None
    avg: Optional[float] = None
    sum: Optional[float] = None


class AlertTriggerRequest(BaseModel):
    """Alert trigger request."""
    severity: AlertSeverity
    title: str
    description: str
    source: str
    triggered_by: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class AlertQueryResponse(BaseModel):
    """Alert query response."""
    alerts: List[Alert]
    total: int


class ErrorSummaryResponse(BaseModel):
    """Error summary response."""
    total_errors: int
    unique_errors: int
    by_type: Dict[str, int]
    all_time_counts: Dict[str, int]


class DashboardSummaryResponse(BaseModel):
    """Dashboard summary response."""
    log_stats: LogStatisticsResponse
    error_summary: ErrorSummaryResponse
    active_alerts: int
    total_metrics: int
    active_traces: int
