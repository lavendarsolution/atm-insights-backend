from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta
import logging
import time

from database import get_db
from schemas.predictions import (
    PredictionRequest,
    PredictionResponse,
    BulkPredictionRequest,
    BulkPredictionResponse,
)
from services.ml_service import VectorizedMLPredictionService
from services.telemetry_service import TelemetryService
from services.cache_service import CacheService
from dependencies.auth import get_current_user
from models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

# Service instances (will be set during startup)
ml_service: VectorizedMLPredictionService = None
telemetry_service: TelemetryService = None
cache_service: CacheService = None

def set_services(ml_svc: VectorizedMLPredictionService, telemetry_svc: TelemetryService):
    """Set service instances"""
    global ml_service, telemetry_service
    ml_service = ml_svc
    telemetry_service = telemetry_svc

@router.post("/predictions/failure", response_model=PredictionResponse)
async def predict_atm_failure(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Predict failure probability for a specific ATM with caching"""
    logger.info(f"Single prediction requested for ATM {request.atm_id} by user {current_user.email}")
    
    try:
        start_time = time.time()
        
        # Use bulk prediction for single ATM for consistency and performance
        results = await ml_service.predict_failure_bulk_optimized(
            db=db,
            atm_ids=[request.atm_id],
            use_cache=True,
            cache_ttl=request.cache_ttl or 300
        )
        
        if not results:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate prediction"
            )
        
        result = results[0]
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Prediction completed for {request.atm_id} in {processing_time:.2f}ms")
        
        return PredictionResponse(
            atm_id=result.atm_id,
            failure_probability=result.failure_probability,
            confidence=result.confidence,
            risk_level=result.risk_level,
            prediction_available=result.prediction_available,
            timestamp=result.timestamp,
            top_risk_factors=result.top_risk_factors or [],
            reason=result.reason,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error predicting failure for {request.atm_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predictions/bulk", response_model=BulkPredictionResponse)
async def bulk_predict_failures(
    request: BulkPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """High-performance bulk prediction for multiple ATMs with vectorized processing"""
    logger.info(f"Bulk prediction requested for {len(request.atm_ids)} ATMs by user {current_user.email}")
    
    try:
        start_time = time.time()
        
        # Validate input
        if len(request.atm_ids) > 1000:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 1000 ATMs allowed per bulk request"
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
            cache_ttl=request.cache_ttl or 300
        )
        
        # Convert to response format
        predictions = []
        for result in prediction_results:
            predictions.append(PredictionResponse(
                atm_id=result.atm_id,
                failure_probability=result.failure_probability,
                confidence=result.confidence,
                risk_level=result.risk_level,
                prediction_available=result.prediction_available,
                timestamp=result.timestamp,
                top_risk_factors=result.top_risk_factors or [],
                reason=result.reason
            ))
        
        # Calculate statistics
        processing_time_ms = (time.time() - start_time) * 1000
        successful_predictions = sum(1 for p in predictions if p.prediction_available)
        high_risk_atms = [p for p in predictions if p.failure_probability > 0.7]
        critical_risk_atms = [p for p in predictions if p.failure_probability > 0.8]
        
        # Performance metrics
        throughput = len(unique_atm_ids) / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
        
        logger.info(f"Bulk prediction completed: {len(unique_atm_ids)} ATMs, "
                   f"{len(high_risk_atms)} high-risk, {processing_time_ms:.1f}ms, "
                   f"{throughput:.1f} predictions/sec")
        
        return BulkPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            successful_predictions=successful_predictions,
            failed_predictions=len(predictions) - successful_predictions,
            high_risk_count=len(high_risk_atms),
            critical_risk_count=len(critical_risk_atms),
            processing_time_ms=processing_time_ms,
            throughput_per_second=throughput,
            cache_hit_rate=None  
        )
        
    except Exception as e:
        logger.error(f"Error in bulk prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/predictions/high-risk")
async def get_high_risk_atms(
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Risk threshold"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of ATMs to return"),
    use_cache: bool = Query(True, description="Use cached predictions"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get ATMs with high failure risk using bulk prediction optimization"""
    logger.info(f"High-risk ATMs requested by {current_user.email} (threshold={threshold})")
    
    try:
        from models import ATM
        
        # Get all active ATMs
        active_atms = db.query(ATM).filter(ATM.status == 'active').limit(500).all()
        
        if not active_atms:
            return {
                "high_risk_atms": [],
                "threshold": threshold,
                "total_checked": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        atm_ids = [atm.atm_id for atm in active_atms]
        
        # Use bulk prediction for efficiency
        predictions = await ml_service.predict_failure_bulk_optimized(
            db=db,
            atm_ids=atm_ids,
            use_cache=use_cache,
            cache_ttl=600  # 10 minutes cache for high-risk queries
        )
        
        # Filter high-risk ATMs
        high_risk_atms = []
        atm_lookup = {atm.atm_id: atm for atm in active_atms}
        
        for prediction in predictions:
            if prediction.failure_probability >= threshold:
                atm = atm_lookup.get(prediction.atm_id)
                if atm:
                    high_risk_atms.append({
                        "atm_id": atm.atm_id,
                        "name": atm.name,
                        "location": atm.location_address,
                        "failure_probability": prediction.failure_probability,
                        "confidence": prediction.confidence,
                        "risk_level": prediction.risk_level,
                        "top_risk_factors": prediction.top_risk_factors or [],
                        "timestamp": prediction.timestamp
                    })
        
        # Sort by risk and limit
        high_risk_atms.sort(key=lambda x: x["failure_probability"], reverse=True)
        high_risk_atms = high_risk_atms[:limit]
        
        logger.info(f"Found {len(high_risk_atms)} high-risk ATMs out of {len(active_atms)} checked")
        
        return {
            "high_risk_atms": high_risk_atms,
            "threshold": threshold,
            "total_checked": len(active_atms),
            "processing_time_ms": None,  
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting high-risk ATMs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# @router.get("/predictions/model-info", response_model=ModelInfoResponse)
# async def get_model_info(
#     current_user: User = Depends(get_current_user)
# ):
#     """Get comprehensive information about loaded ML models"""
#     try:
#         model_info = {
#             "failure_model_loaded": ml_service.failure_model is not None,
#             "anomaly_model_loaded": False,  # We don't use anomaly detection
#             "models_available": ml_service.model_loaded,
#             "last_updated": datetime.now().isoformat(),
#             "model_version": "v2.0",
#             "feature_count": 28,
#             "supports_batch_prediction": True,
#             "supports_vectorized_processing": True
#         }
        
#         if ml_service.failure_model:
#             model_info.update({
#                 "failure_model_type": type(ml_service.failure_model).__name__,
#                 "n_features": 28,
#                 "model_params": getattr(ml_service.failure_model, 'get_params', lambda: {})(),
#                 "feature_names": ml_service.feature_engineer.get_feature_names()
#             })
        
#         # Add performance statistics
#         perf_stats = ml_service.get_performance_stats()
#         model_info["performance_stats"] = perf_stats
            
#         logger.debug(f"Model info requested by {current_user.email}")
        
#         return ModelInfoResponse(**model_info)
        
#     except Exception as e:
#         logger.error(f"Error getting model info: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/predictions/performance", response_model=ModelPerformanceStats)
# async def get_model_performance(
#     current_user: User = Depends(get_current_user)
# ):
#     """Get detailed model performance statistics"""
#     try:
#         stats = ml_service.get_performance_stats()
        
#         return ModelPerformanceStats(
#             total_predictions=stats.get('total_predictions', 0),
#             total_processing_time_ms=stats.get('total_processing_time', 0),
#             average_prediction_time_ms=stats.get('average_prediction_time_ms', 0),
#             batch_predictions=stats.get('batch_predictions', 0),
#             cache_hits=stats.get('cache_hits', 0),
#             cache_misses=stats.get('cache_misses', 0),
#             cache_hit_rate=stats.get('cache_hit_rate', 0),
#             last_updated=datetime.now().isoformat()
#         )
        
#     except Exception as e:
#         logger.error(f"Error getting performance stats: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.post("/predictions/train", response_model=TrainingJobResponse)
# async def trigger_model_training(
#     background_tasks: BackgroundTasks,
#     request: Optional[HyperparameterOptimizationRequest] = None,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """Trigger model retraining with hyperparameter optimization"""
#     logger.info(f"Model training requested by {current_user.email}")
    
#     # Check user permissions (admin only)
#     if current_user.role != "admin":
#         raise HTTPException(
#             status_code=403, 
#             detail="Model training requires admin privileges"
#         )
    
#     try:
#         # Generate training job ID
#         job_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user.id}"
        
#         # Default training parameters
#         training_params = {
#             'n_trials': 100,
#             'cv_folds': 5,
#             'optimization_timeout': 3600,
#             'n_jobs': -1
#         }
        
#         # Override with request parameters if provided
#         if request:
#             if request.n_trials:
#                 training_params['n_trials'] = request.n_trials
#             if request.cv_folds:
#                 training_params['cv_folds'] = request.cv_folds
#             if request.optimization_timeout:
#                 training_params['optimization_timeout'] = request.optimization_timeout
        
#         # Add training job to background tasks
#         background_tasks.add_task(
#             _run_model_training_job,
#             job_id=job_id,
#             training_params=training_params,
#             user_id=current_user.id
#         )
        
#         return TrainingJobResponse(
#             job_id=job_id,
#             status="queued",
#             started_at=datetime.now().isoformat(),
#             estimated_duration_minutes=60,
#             training_params=training_params,
#             message="Training job has been queued and will start shortly"
#         )
        
#     except Exception as e:
#         logger.error(f"Error triggering model training: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/predictions/training/{job_id}")
# async def get_training_job_status(
#     job_id: str,
#     current_user: User = Depends(get_current_user)
# ):
#     """Get status of a training job"""
#     # This would typically query a job tracking system
#     # For now, return a simple response
#     return {
#         "job_id": job_id,
#         "status": "running",  # or "completed", "failed"
#         "progress": 50,  # percentage
#         "estimated_completion": datetime.now() + timedelta(minutes=30),
#         "message": "Training in progress..."
#     }

# async def _run_model_training_job(
#     job_id: str, 
#     training_params: dict, 
#     user_id: str
# ):
#     """Background task for model training"""
#     logger.info(f"Starting training job {job_id} with params: {training_params}")
    
#     try:
#         from ml.model_training import ATMFailureModelTrainer
#         from database.session import SessionLocal
        
#         # Create trainer with specified parameters
#         trainer = ATMFailureModelTrainer(**training_params)
        
#         # Get database session
#         db = SessionLocal()
        
#         try:
#             # Run training pipeline
#             results = trainer.full_training_pipeline(db)
            
#             logger.info(f"Training job {job_id} completed successfully")
#             logger.info(f"Final AUC: {results.get('auc_score', 'N/A')}")
            
#             # Here you would typically update job status in database
#             # and potentially reload the model in the ML service
            
#         finally:
#             db.close()
            
#     except Exception as e:
#         logger.error(f"Training job {job_id} failed: {e}")
#         # Update job status to failed

# Add these endpoints to your predictions.py routes for admin/monitoring

# @router.get("/predictions/system-status")
# async def get_prediction_system_status(
#     current_user: User = Depends(get_current_user)
# ):
#     """Get ML prediction system status (admin endpoint)"""
#     try:
#         # Get system status from background service
#         status = await background_service.get_system_status() if background_service else {}
        
#         # Get ML service performance stats
#         ml_performance = {}
#         if ml_service and hasattr(ml_service, 'prediction_stats'):
#             ml_performance = getattr(ml_service, 'prediction_stats', {})
        
#         # Get cached prediction summary
#         prediction_summary = await cache_service.get("ml_prediction_summary")
        
#         return {
#             "system_status": status,
#             "ml_performance": ml_performance,
#             "prediction_summary": prediction_summary,
#             "model_info": {
#                 "loaded": bool(ml_service and getattr(ml_service, 'model_loaded', False)),
#                 "model_type": type(ml_service.failure_model).__name__ if ml_service and hasattr(ml_service, 'failure_model') and ml_service.failure_model else None,
#                 "feature_count": 28 if ml_service and getattr(ml_service, 'model_loaded', False) else 0
#             },
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting prediction system status: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.post("/predictions/trigger-manual-run")
# async def trigger_manual_prediction_run(
#     current_user: User = Depends(get_current_user)
# ):
#     """Manually trigger a prediction run (admin endpoint)"""
#     try:
#         # Check if user has admin privileges (optional)
#         if hasattr(current_user, 'role') and current_user.role != 'admin':
#             raise HTTPException(status_code=403, detail="Admin privileges required")
        
#         # Trigger manual prediction run
#         success = await background_service.trigger_prediction_run() if background_service else False
        
#         if success:
#             return {
#                 "success": True,
#                 "message": "Manual prediction run triggered successfully",
#                 "timestamp": datetime.now().isoformat()
#             }
#         else:
#             raise HTTPException(status_code=503, detail="Prediction scheduler is not running")
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error triggering manual prediction run: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.post("/predictions/restart-scheduler")
# async def restart_prediction_scheduler(
#     current_user: User = Depends(get_current_user)
# ):
#     """Restart the prediction scheduler (admin endpoint)"""
#     try:
#         # Check if user has admin privileges (optional)
#         if hasattr(current_user, 'role') and current_user.role != 'admin':
#             raise HTTPException(status_code=403, detail="Admin privileges required")
        
#         # Restart prediction scheduler
#         success = await background_service.restart_prediction_scheduler() if background_service else False
        
#         if success:
#             return {
#                 "success": True,
#                 "message": "Prediction scheduler restarted successfully",
#                 "timestamp": datetime.now().isoformat()
#             }
#         else:
#             return {
#                 "success": False,
#                 "message": "Failed to restart prediction scheduler - ML service may not be available",
#                 "timestamp": datetime.now().isoformat()
#             }
            
#     except Exception as e:
#         logger.error(f"Error restarting prediction scheduler: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/predictions/performance-stats")
# async def get_prediction_performance_stats():
#     """Get detailed prediction performance statistics"""
#     try:
#         # Get ML service stats
#         ml_stats = {}
#         if ml_service and hasattr(ml_service, 'prediction_stats'):
#             stats = getattr(ml_service, 'prediction_stats', {})
            
#             # Calculate additional metrics
#             total_predictions = stats.get('total_predictions', 0)
#             total_time = stats.get('total_processing_time', 0)
#             cache_hits = stats.get('cache_hits', 0)
#             cache_misses = stats.get('cache_misses', 0)
            
#             ml_stats = {
#                 **stats,
#                 'average_prediction_time_ms': (
#                     total_time / total_predictions if total_predictions > 0 else 0
#                 ),
#                 'cache_hit_rate': (
#                     cache_hits / (cache_hits + cache_misses) 
#                     if (cache_hits + cache_misses) > 0 else 0
#                 ),
#                 'predictions_per_minute': (
#                     total_predictions / (total_time / 60000) if total_time > 0 else 0
#                 )
#             }
        
#         # Get system stats from cache
#         system_stats = await cache_service.get("ml_system_stats") or {}
        
#         return {
#             "ml_performance": ml_stats,
#             "system_stats": system_stats,
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting performance stats: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/predictions/cache-status")
# async def get_prediction_cache_status():
#     """Get prediction cache status and statistics"""
#     try:
#         cache_info = {
#             "connected": bool(cache_service and cache_service._connection),
#             "prediction_summary_cached": bool(await cache_service.get("ml_prediction_summary")),
#             "high_risk_cached": bool(await cache_service.get("ml_high_risk_atms")),
#             "critical_risk_cached": bool(await cache_service.get("ml_critical_risk_atms")),
#             "alert_summary_cached": bool(await cache_service.get("ml_alert_summary")),
#             "system_health_cached": bool(await cache_service.get("system_health")),
#         }
        
#         # Get prediction summary for additional info
#         summary = await cache_service.get("ml_prediction_summary")
#         if summary:
#             cache_info.update({
#                 "last_prediction_run": summary.get("last_updated"),
#                 "total_predictions_cached": summary.get("total_predictions", 0),
#                 "high_risk_count": summary.get("high_risk_atms", 0),
#                 "critical_risk_count": summary.get("critical_risk_atms", 0)
#             })
        
#         return {
#             "cache_status": cache_info,
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting cache status: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))