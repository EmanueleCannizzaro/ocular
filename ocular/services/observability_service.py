"""
Observability service for comprehensive monitoring, logging, and tracing.

Provides real-time logs, request tracing, performance profiling, error tracking,
custom metrics, and alerting capabilities.
"""

import time
import traceback
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class LogEntry(BaseModel):
    """Log entry model."""
    timestamp: datetime
    level: LogLevel
    message: str
    source: str
    trace_id: Optional[str] = None
    user_id: Optional[int] = None
    metadata: Dict[str, Any] = {}
    error_details: Optional[Dict[str, Any]] = None


class TraceSpan(BaseModel):
    """Trace span for distributed tracing."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "in_progress"  # in_progress, success, error
    tags: Dict[str, Any] = {}
    logs: List[Dict[str, Any]] = []
    error: Optional[str] = None


class PerformanceProfile(BaseModel):
    """Performance profiling data."""
    operation: str
    timestamp: datetime
    duration_ms: float
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    metadata: Dict[str, Any] = {}


class CustomMetric(BaseModel):
    """Custom metric data."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = {}
    description: Optional[str] = None


class Alert(BaseModel):
    """Alert model."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    source: str
    triggered_by: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class ObservabilityService:
    """
    Comprehensive observability service with logging, tracing, metrics, and alerting.
    """

    def __init__(
        self,
        max_log_entries: int = 10000,
        max_trace_spans: int = 5000,
        max_metrics: int = 10000,
        log_retention_hours: int = 24,
    ):
        """Initialize observability service."""
        self.max_log_entries = max_log_entries
        self.max_trace_spans = max_trace_spans
        self.max_metrics = max_metrics
        self.log_retention_hours = log_retention_hours

        # In-memory storage (use database for production)
        self.logs: deque = deque(maxlen=max_log_entries)
        self.trace_spans: Dict[str, TraceSpan] = {}
        self.active_traces: Dict[str, List[str]] = defaultdict(list)
        self.performance_profiles: deque = deque(maxlen=1000)
        self.metrics: deque = deque(maxlen=max_metrics)
        self.alerts: List[Alert] = []

        # Metric aggregations
        self.metric_aggregations: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "sum": 0, "min": float('inf'), "max": float('-inf'), "values": deque(maxlen=100)}
        )

        # Alert rules
        self.alert_rules: List[Dict[str, Any]] = []

        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=10))

    # ==================== Logging ====================

    def log(
        self,
        level: LogLevel,
        message: str,
        source: str,
        trace_id: Optional[str] = None,
        user_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """
        Create a log entry.

        Args:
            level: Log level
            message: Log message
            source: Source of the log (e.g., service name, function)
            trace_id: Optional trace ID for correlation
            user_id: Optional user ID
            metadata: Additional metadata
            error_details: Error details if applicable

        Returns:
            Created log entry
        """
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            source=source,
            trace_id=trace_id,
            user_id=user_id,
            metadata=metadata or {},
            error_details=error_details,
        )

        self.logs.append(entry)

        # Also log to standard logger
        log_method = getattr(logger, level.value)
        log_method(f"[{source}] {message}", extra={"trace_id": trace_id, "user_id": user_id})

        # Check alert rules
        self._check_alert_rules_for_log(entry)

        return entry

    def get_logs(
        self,
        level: Optional[LogLevel] = None,
        source: Optional[str] = None,
        trace_id: Optional[str] = None,
        user_id: Optional[int] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """
        Query logs with filters.

        Args:
            level: Filter by log level
            source: Filter by source
            trace_id: Filter by trace ID
            user_id: Filter by user ID
            since: Only logs after this timestamp
            limit: Maximum number of logs to return

        Returns:
            List of matching log entries
        """
        results = []

        for entry in reversed(self.logs):
            # Apply filters
            if level and entry.level != level:
                continue
            if source and entry.source != source:
                continue
            if trace_id and entry.trace_id != trace_id:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if since and entry.timestamp < since:
                continue

            results.append(entry)

            if len(results) >= limit:
                break

        return results

    def get_log_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get log statistics for the specified time period.

        Args:
            hours: Number of hours to analyze

        Returns:
            Log statistics
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        recent_logs = [log for log in self.logs if log.timestamp >= since]

        level_counts = defaultdict(int)
        source_counts = defaultdict(int)

        for log in recent_logs:
            level_counts[log.level.value] += 1
            source_counts[log.source] += 1

        return {
            "total_logs": len(recent_logs),
            "period_hours": hours,
            "by_level": dict(level_counts),
            "by_source": dict(source_counts),
            "error_rate": level_counts.get("error", 0) / max(len(recent_logs), 1),
        }

    # ==================== Distributed Tracing ====================

    def start_trace(
        self,
        trace_id: str,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """
        Start a new trace span.

        Args:
            trace_id: Unique trace identifier
            operation_name: Name of the operation
            parent_span_id: Parent span ID if nested
            tags: Additional tags

        Returns:
            Created trace span
        """
        import uuid

        span_id = str(uuid.uuid4())

        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            tags=tags or {},
        )

        self.trace_spans[span_id] = span
        self.active_traces[trace_id].append(span_id)

        return span

    def end_trace(
        self,
        span_id: str,
        status: str = "success",
        error: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Optional[TraceSpan]:
        """
        End a trace span.

        Args:
            span_id: Span ID to end
            status: Final status (success, error)
            error: Error message if failed
            tags: Additional tags to add

        Returns:
            Updated trace span or None if not found
        """
        span = self.trace_spans.get(span_id)
        if not span:
            return None

        span.end_time = datetime.utcnow()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        span.error = error

        if tags:
            span.tags.update(tags)

        return span

    def add_trace_log(self, span_id: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a log entry to a trace span."""
        span = self.trace_spans.get(span_id)
        if span:
            span.logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "message": message,
                "metadata": metadata or {},
            })

    @contextmanager
    def trace(
        self,
        trace_id: str,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracing operations.

        Usage:
            with obs.trace("trace-123", "database_query"):
                # Your code here
                pass
        """
        span = self.start_trace(trace_id, operation_name, parent_span_id, tags)
        try:
            yield span
            self.end_trace(span.span_id, status="success")
        except Exception as e:
            self.end_trace(span.span_id, status="error", error=str(e))
            raise

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace ID."""
        span_ids = self.active_traces.get(trace_id, [])
        return [self.trace_spans[sid] for sid in span_ids if sid in self.trace_spans]

    # ==================== Performance Profiling ====================

    @contextmanager
    def profile(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Profile an operation's performance.

        Usage:
            with obs.profile("expensive_operation"):
                # Code to profile
                pass
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())

        start_time = time.time()
        start_cpu = process.cpu_percent()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            end_cpu = process.cpu_percent()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB

            profile = PerformanceProfile(
                operation=operation,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                cpu_percent=end_cpu - start_cpu,
                memory_mb=end_memory - start_memory,
                metadata=metadata or {},
            )

            self.performance_profiles.append(profile)

            # Record as metric
            self.record_metric(
                name=f"performance.{operation}.duration_ms",
                value=duration_ms,
                metric_type=MetricType.HISTOGRAM,
                tags={"operation": operation},
            )

    def get_performance_profiles(
        self,
        operation: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[PerformanceProfile]:
        """Query performance profiles."""
        results = []

        for profile in reversed(self.performance_profiles):
            if operation and profile.operation != operation:
                continue
            if since and profile.timestamp < since:
                continue

            results.append(profile)

            if len(results) >= limit:
                break

        return results

    def get_performance_summary(self, operation: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for an operation."""
        since = datetime.utcnow() - timedelta(hours=hours)
        profiles = [
            p for p in self.performance_profiles
            if p.operation == operation and p.timestamp >= since
        ]

        if not profiles:
            return {"operation": operation, "count": 0}

        durations = [p.duration_ms for p in profiles]

        return {
            "operation": operation,
            "count": len(profiles),
            "duration_ms": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "p50": sorted(durations)[len(durations) // 2],
                "p95": sorted(durations)[int(len(durations) * 0.95)],
                "p99": sorted(durations)[int(len(durations) * 0.99)],
            },
        }

    # ==================== Custom Metrics ====================

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ):
        """
        Record a custom metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags
            description: Optional description
        """
        metric = CustomMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            description=description,
        )

        self.metrics.append(metric)

        # Update aggregations
        agg = self.metric_aggregations[name]
        agg["count"] += 1
        agg["sum"] += value
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)
        agg["values"].append(value)
        agg["avg"] = agg["sum"] / agg["count"]

        # Check alert rules for metrics
        self._check_alert_rules_for_metric(metric)

    def get_metrics(
        self,
        name: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[CustomMetric]:
        """Query metrics."""
        results = []

        for metric in reversed(self.metrics):
            if name and metric.name != name:
                continue
            if metric_type and metric.metric_type != metric_type:
                continue
            if since and metric.timestamp < since:
                continue

            results.append(metric)

            if len(results) >= limit:
                break

        return results

    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get aggregated metric summary."""
        agg = self.metric_aggregations.get(name)
        if not agg or agg["count"] == 0:
            return {"name": name, "count": 0}

        return {
            "name": name,
            "count": agg["count"],
            "min": agg["min"],
            "max": agg["max"],
            "avg": agg["avg"],
            "sum": agg["sum"],
        }

    # ==================== Error Tracking ====================

    def track_error(
        self,
        error: Exception,
        source: str,
        trace_id: Optional[str] = None,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Track an error with full context.

        Args:
            error: The exception
            source: Source of the error
            trace_id: Optional trace ID
            user_id: Optional user ID
            context: Additional context
        """
        error_type = type(error).__name__
        error_message = str(error)

        # Increment error count
        self.error_counts[error_type] += 1

        # Store error sample
        error_sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "message": error_message,
            "source": source,
            "trace_id": trace_id,
            "user_id": user_id,
            "context": context or {},
            "traceback": traceback.format_exc(),
        }
        self.error_samples[error_type].append(error_sample)

        # Log the error
        self.log(
            level=LogLevel.ERROR,
            message=f"{error_type}: {error_message}",
            source=source,
            trace_id=trace_id,
            user_id=user_id,
            metadata=context,
            error_details={
                "error_type": error_type,
                "message": error_message,
                "traceback": traceback.format_exc(),
            },
        )

        # Check if error rate exceeds threshold
        self._check_error_rate_alerts()

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary."""
        since = datetime.utcnow() - timedelta(hours=hours)

        recent_errors = defaultdict(int)
        for error_type, samples in self.error_samples.items():
            for sample in samples:
                timestamp = datetime.fromisoformat(sample["timestamp"])
                if timestamp >= since:
                    recent_errors[error_type] += 1

        return {
            "total_errors": sum(recent_errors.values()),
            "unique_errors": len(recent_errors),
            "by_type": dict(recent_errors),
            "all_time_counts": dict(self.error_counts),
        }

    # ==================== Alerting ====================

    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add an alert rule.

        Args:
            name: Rule name
            condition: Function that returns True when alert should trigger
            severity: Alert severity
            description: Alert description
            metadata: Additional metadata
        """
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "description": description,
            "metadata": metadata or {},
        })

    def trigger_alert(
        self,
        severity: AlertSeverity,
        title: str,
        description: str,
        source: str,
        triggered_by: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Trigger an alert."""
        import uuid

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.utcnow(),
            source=source,
            triggered_by=triggered_by,
            metadata=metadata or {},
        )

        self.alerts.append(alert)

        # Log critical alerts
        if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            self.log(
                level=LogLevel.ERROR if severity == AlertSeverity.ERROR else LogLevel.CRITICAL,
                message=f"ALERT: {title}",
                source=source,
                metadata={"alert_id": alert.alert_id, "description": description},
            )

        return alert

    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                return alert
        return None

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Query alerts."""
        results = []

        for alert in reversed(self.alerts):
            if severity and alert.severity != severity:
                continue
            if resolved is not None and alert.resolved != resolved:
                continue
            if since and alert.timestamp < since:
                continue

            results.append(alert)

            if len(results) >= limit:
                break

        return results

    # ==================== Private Helper Methods ====================

    def _check_alert_rules_for_log(self, entry: LogEntry):
        """Check if log entry triggers any alert rules."""
        for rule in self.alert_rules:
            if rule.get("type") == "log":
                try:
                    if rule["condition"](entry.dict()):
                        self.trigger_alert(
                            severity=rule["severity"],
                            title=rule["name"],
                            description=rule["description"],
                            source="alert_rule",
                            triggered_by={"log_entry": entry.dict()},
                            metadata=rule.get("metadata", {}),
                        )
                except Exception as e:
                    logger.error(f"Error checking alert rule {rule['name']}: {e}")

    def _check_alert_rules_for_metric(self, metric: CustomMetric):
        """Check if metric triggers any alert rules."""
        for rule in self.alert_rules:
            if rule.get("type") == "metric":
                try:
                    if rule["condition"](metric.dict()):
                        self.trigger_alert(
                            severity=rule["severity"],
                            title=rule["name"],
                            description=rule["description"],
                            source="alert_rule",
                            triggered_by={"metric": metric.dict()},
                            metadata=rule.get("metadata", {}),
                        )
                except Exception as e:
                    logger.error(f"Error checking alert rule {rule['name']}: {e}")

    def _check_error_rate_alerts(self):
        """Check if error rate exceeds thresholds."""
        # Get errors in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_error_count = sum(
            1 for log in self.logs
            if log.level == LogLevel.ERROR and log.timestamp >= one_hour_ago
        )

        # Trigger alert if error rate is high
        if recent_error_count > 100:  # More than 100 errors per hour
            self.trigger_alert(
                severity=AlertSeverity.ERROR,
                title="High Error Rate Detected",
                description=f"{recent_error_count} errors in the last hour",
                source="error_rate_monitor",
                triggered_by={"error_count": recent_error_count, "period_hours": 1},
            )


# Global observability service instance
_observability_service: Optional[ObservabilityService] = None


def get_observability_service() -> ObservabilityService:
    """Get or create the global observability service instance."""
    global _observability_service
    if _observability_service is None:
        _observability_service = ObservabilityService()
    return _observability_service
