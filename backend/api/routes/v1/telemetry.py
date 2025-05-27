import logging
from datetime import datetime

from database import get_db
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from schemas.telemetry import TelemetryData, TelemetryResponse
from services import BackgroundTaskService, CacheService, TelemetryService
from sqlalchemy.orm import Session

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


@router.post("/telemetry", response_model=TelemetryResponse)
async def receive_telemetry(
    telemetry: TelemetryData,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Optimized telemetry ingestion endpoint for essential ATM data"""
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
            atm_id=telemetry.atm_id,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Telemetry processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/telemetry/{atm_id}/history")
async def get_atm_telemetry_history(
    atm_id: str, hours: int = 24, db: Session = Depends(get_db)
):
    """Get telemetry history for specific ATM"""
    try:
        if hours > 168:  # Limit to 1 week
            hours = 168

        history = await telemetry_service.get_atm_telemetry_history(db, atm_id, hours)

        return {
            "atm_id": atm_id,
            "hours": hours,
            "data_points": len(history),
            "history": history,
        }

    except Exception as e:
        logger.error(f"Error getting telemetry history for {atm_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/telemetry/alerts/recent")
async def get_recent_alerts():
    """Get recent alerts generated from telemetry"""
    try:
        alerts = await telemetry_service.get_recent_alerts()

        return {
            "alert_count": len(alerts),
            "alerts": alerts,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting recent alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/telemetry/stats")
async def get_telemetry_stats(db: Session = Depends(get_db)):
    """Get telemetry ingestion statistics"""
    try:
        from sqlalchemy import text

        # Get basic telemetry stats
        stats_query = text(
            """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT atm_id) as unique_atms,
                MIN(time) as oldest_record,
                MAX(time) as newest_record,
                COUNT(*) FILTER (WHERE time >= NOW() - INTERVAL '1 hour') as last_hour,
                COUNT(*) FILTER (WHERE time >= NOW() - INTERVAL '24 hours') as last_24h
            FROM atm_telemetry
        """
        )

        result = db.execute(stats_query).fetchone()

        return {
            "total_records": result.total_records or 0,
            "unique_atms": result.unique_atms or 0,
            "oldest_record": (
                result.oldest_record.isoformat() if result.oldest_record else None
            ),
            "newest_record": (
                result.newest_record.isoformat() if result.newest_record else None
            ),
            "records_last_hour": result.last_hour or 0,
            "records_last_24h": result.last_24h or 0,
            "avg_per_hour": round((result.last_24h or 0) / 24, 2),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting telemetry stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
