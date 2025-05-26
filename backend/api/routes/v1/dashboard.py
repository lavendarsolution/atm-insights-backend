from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import logging

from database import get_db
from schemas import DashboardStats
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

@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get optimized dashboard statistics"""
    try:
        stats = await telemetry_service.get_dashboard_stats(db)
        return DashboardStats(**stats)
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))