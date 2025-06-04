from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request for single ATM failure prediction"""
    atm_id: str = Field(..., description="ATM identifier", min_length=1, max_length=32)
    history_hours: Optional[int] = Field(24, ge=1, le=168, description="Hours of history to consider")
    use_cache: Optional[bool] = Field(True, description="Whether to use cached predictions")
    cache_ttl: Optional[int] = Field(300, ge=60, le=3600, description="Cache TTL in seconds")
    
    @validator('atm_id')
    def validate_atm_id(cls, v):
        if not v or not v.strip():
            raise ValueError('ATM ID cannot be empty')
        return v.strip().upper()

class PredictionResponse(BaseModel):
    """Response for failure prediction"""
    atm_id: str
    failure_probability: float = Field(..., ge=0, le=1, description="Probability of failure (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical, unknown")
    prediction_available: bool
    timestamp: str
    reason: Optional[str] = Field(None, description="Reason if prediction unavailable")
    top_risk_factors: Optional[List[Dict[str, Union[str, float]]]] = Field(
        None, description="Top contributing features"
    )
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

class BulkPredictionRequest(BaseModel):
    """Request for high-performance bulk ATM predictions"""
    atm_ids: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=1000, 
        description="List of ATM IDs (max 1000 per request)"
    )
    use_cache: Optional[bool] = Field(True, description="Whether to use cached predictions")
    cache_ttl: Optional[int] = Field(300, ge=60, le=3600, description="Cache TTL in seconds")
    parallel_processing: Optional[bool] = Field(True, description="Enable parallel processing")
    
    @validator('atm_ids')
    def validate_atm_ids(cls, v):
        if not v:
            raise ValueError('At least one ATM ID is required')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for atm_id in v:
            if atm_id and atm_id.strip():
                clean_id = atm_id.strip().upper()
                if clean_id not in seen:
                    seen.add(clean_id)
                    unique_ids.append(clean_id)
        
        if not unique_ids:
            raise ValueError('No valid ATM IDs provided')
        
        return unique_ids

class BulkPredictionResponse(BaseModel):
    """Response for bulk predictions with performance metrics"""
    predictions: List[PredictionResponse]
    total_predictions: int = Field(..., description="Total number of predictions")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    high_risk_count: int = Field(..., description="Number of high-risk ATMs (>70% probability)")
    critical_risk_count: int = Field(..., description="Number of critical-risk ATMs (>80% probability)")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    throughput_per_second: float = Field(..., description="Predictions processed per second")
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate (0-1)")

# class ModelInfoResponse(BaseModel):
#     """Comprehensive information about loaded ML models"""
#     failure_model_loaded: bool
#     anomaly_model_loaded: bool  # Always False since we don't use it
#     models_available: bool
#     failure_model_type: Optional[str] = None
#     n_features: Optional[int] = None
#     feature_names: Optional[List[str]] = None
#     model_params: Optional[Dict[str, Any]] = None
#     model_version: Optional[str] = None
#     last_updated: str
#     supports_batch_prediction: Optional[bool] = None
#     supports_vectorized_processing: Optional[bool] = None
#     performance_stats: Optional[Dict[str, Any]] = None

# class ModelPerformanceStats(BaseModel):
#     """Detailed model performance statistics"""
#     total_predictions: int = Field(..., description="Total predictions made")
#     total_processing_time_ms: float = Field(..., description="Total processing time in ms")
#     average_prediction_time_ms: float = Field(..., description="Average prediction time in ms")
#     batch_predictions: int = Field(..., description="Number of batch predictions")
#     cache_hits: int = Field(..., description="Number of cache hits")
#     cache_misses: int = Field(..., description="Number of cache misses")
#     cache_hit_rate: float = Field(..., ge=0, le=1, description="Cache hit rate (0-1)")
#     last_updated: str

# class HyperparameterOptimizationRequest(BaseModel):
#     """Request for hyperparameter optimization during training"""
#     n_trials: Optional[int] = Field(100, ge=10, le=500, description="Number of Optuna trials")
#     cv_folds: Optional[int] = Field(5, ge=3, le=10, description="Cross-validation folds")
#     optimization_timeout: Optional[int] = Field(
#         3600, ge=300, le=14400, description="Optimization timeout in seconds"
#     )
#     algorithm: Optional[str] = Field("TPE", description="Optimization algorithm")
#     early_stopping: Optional[bool] = Field(True, description="Enable early stopping")

# class TrainingJobResponse(BaseModel):
#     """Response for training job initiation"""
#     job_id: str = Field(..., description="Unique job identifier")
#     status: str = Field(..., description="Job status: queued, running, completed, failed")
#     started_at: str = Field(..., description="Job start timestamp")
#     estimated_duration_minutes: Optional[int] = Field(None, description="Estimated duration")
#     training_params: Dict[str, Any] = Field(..., description="Training parameters")
#     message: str = Field(..., description="Status message")

# class TrainingJobStatus(BaseModel):
#     """Status of a training job"""
#     job_id: str
#     status: str  # queued, running, completed, failed
#     progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
#     started_at: Optional[str] = None
#     completed_at: Optional[str] = None
#     estimated_completion: Optional[str] = None
#     current_trial: Optional[int] = None
#     best_score: Optional[float] = None
#     message: str
#     error_details: Optional[str] = None
#     results: Optional[Dict[str, Any]] = None

