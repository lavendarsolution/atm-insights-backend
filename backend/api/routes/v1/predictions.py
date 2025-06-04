import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from database import get_db
from dependencies.auth import get_current_user
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from models.user import User
from schemas.predictions import (
    BulkPredictionRequest,
    BulkPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from services.cache_service import CacheService
from services.ml_service import VectorizedMLPredictionService
from services.telemetry_service import TelemetryService
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter()

# Service instances (will be set during startup)
ml_service: VectorizedMLPredictionService = None
telemetry_service: TelemetryService = None
cache_service: CacheService = None


def set_services(
    ml_svc: VectorizedMLPredictionService, telemetry_svc: TelemetryService
):
    """Set service instances"""
    global ml_service, telemetry_service
    ml_service = ml_svc
    telemetry_service = telemetry_svc


@router.post("/predictions/failure", response_model=PredictionResponse)
async def predict_atm_failure(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Predict failure probability for a specific ATM with caching"""
    logger.info(
        f"Single prediction requested for ATM {request.atm_id} by user {current_user.email}"
    )

    try:
        start_time = time.time()

        # Use bulk prediction for single ATM for consistency and performance
        results = await ml_service.predict_failure_bulk_optimized(
            db=db,
            atm_ids=[request.atm_id],
            use_cache=True,
            cache_ttl=request.cache_ttl or 300,
        )

        if not results:
            raise HTTPException(status_code=500, detail="Failed to generate prediction")

        result = results[0]
        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Prediction completed for {request.atm_id} in {processing_time:.2f}ms"
        )

        return PredictionResponse(
            atm_id=result.atm_id,
            failure_probability=result.failure_probability,
            confidence=result.confidence,
            risk_level=result.risk_level,
            prediction_available=result.prediction_available,
            timestamp=result.timestamp,
            top_risk_factors=result.top_risk_factors or [],
            reason=result.reason,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(
            f"Error predicting failure for {request.atm_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predictions/bulk", response_model=BulkPredictionResponse)
async def bulk_predict_failures(
    request: BulkPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """High-performance bulk prediction for multiple ATMs with vectorized processing"""
    logger.info(
        f"Bulk prediction requested for {len(request.atm_ids)} ATMs by user {current_user.email}"
    )

    try:
        start_time = time.time()

        # Validate input
        if len(request.atm_ids) > 1000:
            raise HTTPException(
                status_code=400, detail="Maximum 1000 ATMs allowed per bulk request"
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_atm_ids = []
        for atm_id in request.atm_ids:
            if atm_id not in seen:
                seen.add(atm_id)
                unique_atm_ids.append(atm_id)

        # Perform bulk prediction with optimized processing
        prediction_results = await ml_service.predict_failure_bulk_optimized(
            db=db,
            atm_ids=unique_atm_ids,
            use_cache=request.use_cache,
            cache_ttl=request.cache_ttl or 300,
        )

        # Convert to response format
        predictions = []
        for result in prediction_results:
            predictions.append(
                PredictionResponse(
                    atm_id=result.atm_id,
                    failure_probability=result.failure_probability,
                    confidence=result.confidence,
                    risk_level=result.risk_level,
                    prediction_available=result.prediction_available,
                    timestamp=result.timestamp,
                    top_risk_factors=result.top_risk_factors or [],
                    reason=result.reason,
                )
            )

        # Calculate statistics
        processing_time_ms = (time.time() - start_time) * 1000
        successful_predictions = sum(1 for p in predictions if p.prediction_available)
        high_risk_atms = [p for p in predictions if p.failure_probability > 0.7]
        critical_risk_atms = [p for p in predictions if p.failure_probability > 0.8]

        # Performance metrics
        throughput = (
            len(unique_atm_ids) / (processing_time_ms / 1000)
            if processing_time_ms > 0
            else 0
        )

        logger.info(
            f"Bulk prediction completed: {len(unique_atm_ids)} ATMs, "
            f"{len(high_risk_atms)} high-risk, {processing_time_ms:.1f}ms, "
            f"{throughput:.1f} predictions/sec"
        )

        return BulkPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            successful_predictions=successful_predictions,
            failed_predictions=len(predictions) - successful_predictions,
            high_risk_count=len(high_risk_atms),
            critical_risk_count=len(critical_risk_atms),
            processing_time_ms=processing_time_ms,
            throughput_per_second=throughput,
            cache_hit_rate=None,
        )

    except Exception as e:
        logger.error(f"Error in bulk prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/high-risk")
async def get_high_risk_atms(
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Risk threshold"),
    limit: int = Query(
        20, ge=1, le=100, description="Maximum number of ATMs to return"
    ),
    use_cache: bool = Query(True, description="Use cached predictions"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get ATMs with high failure risk using bulk prediction optimization"""
    logger.info(
        f"High-risk ATMs requested by {current_user.email} (threshold={threshold})"
    )

    try:
        from models import ATM

        # Get all active ATMs
        active_atms = db.query(ATM).filter(ATM.status == "active").limit(500).all()

        if not active_atms:
            return {
                "high_risk_atms": [],
                "threshold": threshold,
                "total_checked": 0,
                "timestamp": datetime.now().isoformat(),
            }

        atm_ids = [atm.atm_id for atm in active_atms]

        # Use bulk prediction for efficiency
        predictions = await ml_service.predict_failure_bulk_optimized(
            db=db,
            atm_ids=atm_ids,
            use_cache=use_cache,
            cache_ttl=600,  # 10 minutes cache for high-risk queries
        )

        # Filter high-risk ATMs
        high_risk_atms = []
        atm_lookup = {atm.atm_id: atm for atm in active_atms}

        for prediction in predictions:
            if prediction.failure_probability >= threshold:
                atm = atm_lookup.get(prediction.atm_id)
                if atm:
                    high_risk_atms.append(
                        {
                            "atm_id": atm.atm_id,
                            "name": atm.name,
                            "location": atm.location_address,
                            "failure_probability": prediction.failure_probability,
                            "confidence": prediction.confidence,
                            "risk_level": prediction.risk_level,
                            "top_risk_factors": prediction.top_risk_factors or [],
                            "timestamp": prediction.timestamp,
                        }
                    )

        # Sort by risk and limit
        high_risk_atms.sort(key=lambda x: x["failure_probability"], reverse=True)
        high_risk_atms = high_risk_atms[:limit]

        logger.info(
            f"Found {len(high_risk_atms)} high-risk ATMs out of {len(active_atms)} checked"
        )

        return {
            "high_risk_atms": high_risk_atms,
            "threshold": threshold,
            "total_checked": len(active_atms),
            "processing_time_ms": None,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting high-risk ATMs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
