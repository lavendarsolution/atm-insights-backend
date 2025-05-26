from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import logging

from database import get_db
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

@router.get("/atms/status")
async def get_atms_status(db: Session = Depends(get_db)):
    """Get optimized ATM status list"""
    try:
        atm_statuses = await telemetry_service.get_atm_status_list(db)
        return atm_statuses
        
    except Exception as e:
        logger.error(f"Error getting ATM status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))