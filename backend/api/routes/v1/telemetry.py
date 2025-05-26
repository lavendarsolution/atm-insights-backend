from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from database import get_db
from schemas import TelemetryData, TelemetryResponse
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

@router.post("/telemetry", response_model=TelemetryResponse)
async def receive_telemetry(
    telemetry: TelemetryData,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Optimized telemetry ingestion endpoint"""
    try:
        # Convert to dict for batch processing
        telemetry_data = telemetry.dict()
        
        # Process in background for better performance
        result = await telemetry_service.process_telemetry_batch(db, [telemetry_data])
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail="Failed to process telemetry")
        
        # Publish to cache for real-time updates
        await cache_service.publish(f"telemetry:{telemetry.atm_id}", telemetry_data)
        
        return TelemetryResponse(
            success=True,
            message=f"Telemetry processed for {telemetry.atm_id}",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Telemetry processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/telemetry/batch")
async def receive_telemetry_batch(
    telemetry_list: list[TelemetryData],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Optimized batch telemetry ingestion"""
    try:
        # Convert to dict list
        telemetry_data_list = [t.dict() for t in telemetry_list]
        
        # Process batch
        result = await telemetry_service.process_telemetry_batch(db, telemetry_data_list)
        
        return {
            "success": result["success"],
            "processed_count": result["processed"],
            "errors": result["errors"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))