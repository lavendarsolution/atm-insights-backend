from fastapi import APIRouter, HTTPException
import logging

from config import settings
from services import TelemetryService, CacheService, BackgroundTaskService

logger = logging.getLogger(__name__)
router = APIRouter()

# Service instances (will be set during startup)
cache_service: CacheService = None
telemetry_service: TelemetryService = None
background_service: BackgroundTaskService = None

def set_services(cache_svc: CacheService, telemetry_svc: TelemetryService, background_svc: BackgroundTaskService):
    """Set service instances"""
    global cache_service, telemetry_service, background_service
    cache_service = cache_svc
    telemetry_service = telemetry_svc
    background_service = background_svc

@router.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint"""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    try:
        # Get cached metrics
        metrics = await cache_service.get("system_health")
        return metrics or {"message": "No metrics available"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))