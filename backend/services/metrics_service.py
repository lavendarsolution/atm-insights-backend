# backend/services/metrics_service.py
import logging
import time
from typing import Any, Dict

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

# Define Prometheus metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

TELEMETRY_MESSAGES_TOTAL = Counter(
    "telemetry_messages_total",
    "Total telemetry messages processed",
    ["atm_id", "status"],
)

ATM_STATUS_COUNT = Gauge("atm_status_count", "Number of ATMs by status", ["status"])

DATABASE_CONNECTIONS_ACTIVE = Gauge(
    "database_connections_active", "Number of active database connections"
)

REDIS_CONNECTIONS_ACTIVE = Gauge(
    "redis_connections_active", "Number of active Redis connections"
)

WEBSOCKET_CONNECTIONS_ACTIVE = Gauge(
    "websocket_connections_active",
    "Number of active WebSocket connections",
    ["connection_type"],
)

ALERT_TRIGGERED_TOTAL = Counter(
    "alerts_triggered_total", "Total alerts triggered", ["severity", "rule_type"]
)

BACKGROUND_TASKS_ACTIVE = Gauge(
    "background_tasks_active", "Number of active background tasks"
)

CACHE_OPERATIONS_TOTAL = Counter(
    "cache_operations_total", "Total cache operations", ["operation", "result"]
)

ATM_TELEMETRY_PROCESSING_TIME = Histogram(
    "atm_telemetry_processing_seconds", "Time spent processing telemetry data"
)


class MetricsService:
    """Service for collecting and exposing Prometheus metrics"""

    def __init__(self):
        self.start_time = time.time()

    async def record_http_request(
        self, request: Request, response: Response, process_time: float
    ):
        """Record HTTP request metrics"""
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=self._get_endpoint_name(request.url.path),
            status_code=response.status_code,
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method, endpoint=self._get_endpoint_name(request.url.path)
        ).observe(process_time)

    def record_telemetry_message(self, atm_id: str, status: str):
        """Record telemetry message processing"""
        TELEMETRY_MESSAGES_TOTAL.labels(atm_id=atm_id, status=status).inc()

    def update_atm_status_counts(self, status_counts: Dict[str, int]):
        """Update ATM status gauge metrics"""
        for status, count in status_counts.items():
            ATM_STATUS_COUNT.labels(status=status).set(count)

    def update_database_connections(self, count: int):
        """Update database connection count"""
        DATABASE_CONNECTIONS_ACTIVE.set(count)

    def update_redis_connections(self, count: int):
        """Update Redis connection count"""
        REDIS_CONNECTIONS_ACTIVE.set(count)

    def update_websocket_connections(self, connection_type: str, count: int):
        """Update WebSocket connection count"""
        WEBSOCKET_CONNECTIONS_ACTIVE.labels(connection_type=connection_type).set(count)

    def record_alert(self, severity: str, rule_type: str):
        """Record alert triggering"""
        ALERT_TRIGGERED_TOTAL.labels(severity=severity, rule_type=rule_type).inc()

    def update_background_tasks(self, count: int):
        """Update active background tasks count"""
        BACKGROUND_TASKS_ACTIVE.set(count)

    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation (hit/miss/error)"""
        CACHE_OPERATIONS_TOTAL.labels(operation=operation, result=result).inc()

    def record_telemetry_processing_time(self, process_time: float):
        """Record time spent processing telemetry data"""
        ATM_TELEMETRY_PROCESSING_TIME.observe(process_time)

    def _get_endpoint_name(self, path: str) -> str:
        """Normalize endpoint paths for metrics"""
        # Replace dynamic path segments with placeholders
        if path.startswith("/api/v1/atms/") and len(path.split("/")) > 4:
            return "/api/v1/atms/{atm_id}"
        elif path.startswith("/api/v1/alerts/") and len(path.split("/")) > 4:
            return "/api/v1/alerts/{alert_id}"
        elif path.startswith("/api/v1/ws/"):
            return "/api/v1/ws/*"
        else:
            return path

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest()

    def get_custom_metrics(self) -> Dict[str, Any]:
        """Get custom application metrics"""
        uptime = time.time() - self.start_time

        return {
            "application_uptime_seconds": uptime,
            "application_info": {"version": "1.0.0", "environment": "production"},
        }


# Global metrics service instance
metrics_service = MetricsService()


# Middleware for automatic metrics collection
async def metrics_middleware(request: Request, call_next):
    """Middleware to automatically collect HTTP metrics"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    await metrics_service.record_http_request(request, response, process_time)

    return response
