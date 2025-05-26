import logging
from datetime import datetime

from config import settings
from database import get_db
from fastapi import APIRouter
from services import BackgroundTaskService, CacheService, TelemetryService
from sqlalchemy import text

logger = logging.getLogger(__name__)
router = APIRouter()

# Service instances (will be set during startup)
cache_service: CacheService = None
telemetry_service: TelemetryService = None
background_service: BackgroundTaskService = None


def set_services(
    cache_svc: CacheService,
    telemetry_svc: TelemetryService,
    background_svc: BackgroundTaskService,
):
    """Set service instances"""
    global cache_service, telemetry_service, background_service
    cache_service = cache_svc
    telemetry_service = telemetry_svc
    background_service = background_svc


@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.env,
        "version": settings.api_version,
    }

    # Check database connectivity
    try:
        db = next(get_db())
        db.execute(text("SELECT 1"))
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # Check Redis connectivity
    if cache_service and cache_service._connection:
        try:
            await cache_service._connection.ping()
            health_status["redis"] = "connected"
        except Exception as e:
            health_status["redis"] = f"error: {str(e)}"
    else:
        health_status["redis"] = "not configured"

    # Get system health metrics
    try:
        system_health = await cache_service.get("system_health")
        if system_health:
            health_status["metrics"] = system_health
    except Exception:
        pass

    return health_status
