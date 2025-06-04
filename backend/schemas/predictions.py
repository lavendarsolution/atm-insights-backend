from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request for single ATM failure prediction"""

    atm_id: str = Field(..., description="ATM identifier", min_length=1, max_length=32)
    history_hours: Optional[int] = Field(
        24, ge=1, le=168, description="Hours of history to consider"
    )
    use_cache: Optional[bool] = Field(
        True, description="Whether to use cached predictions"
    )
    cache_ttl: Optional[int] = Field(
        300, ge=60, le=3600, description="Cache TTL in seconds"
    )

    @validator("atm_id")
    def validate_atm_id(cls, v):
        if not v or not v.strip():
            raise ValueError("ATM ID cannot be empty")
        return v.strip().upper()


class PredictionResponse(BaseModel):
    """Response for failure prediction"""

    atm_id: str
    failure_probability: float = Field(
        ..., ge=0, le=1, description="Probability of failure (0-1)"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Prediction confidence (0-1)"
    )
    risk_level: str = Field(
        ..., description="Risk level: low, medium, high, critical, unknown"
    )
    prediction_available: bool
    timestamp: str
    reason: Optional[str] = Field(None, description="Reason if prediction unavailable")
    top_risk_factors: Optional[List[Dict[str, Union[str, float]]]] = Field(
        None, description="Top contributing features"
    )
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds"
    )


class BulkPredictionRequest(BaseModel):
    """Request for high-performance bulk ATM predictions"""

    atm_ids: List[str] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of ATM IDs (max 1000 per request)",
    )
    use_cache: Optional[bool] = Field(
        True, description="Whether to use cached predictions"
    )
    cache_ttl: Optional[int] = Field(
        300, ge=60, le=3600, description="Cache TTL in seconds"
    )
    parallel_processing: Optional[bool] = Field(
        True, description="Enable parallel processing"
    )

    @validator("atm_ids")
    def validate_atm_ids(cls, v):
        if not v:
            raise ValueError("At least one ATM ID is required")

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
            raise ValueError("No valid ATM IDs provided")

        return unique_ids


class BulkPredictionResponse(BaseModel):
    """Response for bulk predictions with performance metrics"""

    predictions: List[PredictionResponse]
    total_predictions: int = Field(..., description="Total number of predictions")
    successful_predictions: int = Field(
        ..., description="Number of successful predictions"
    )
    failed_predictions: int = Field(..., description="Number of failed predictions")
    high_risk_count: int = Field(
        ..., description="Number of high-risk ATMs (>70% probability)"
    )
    critical_risk_count: int = Field(
        ..., description="Number of critical-risk ATMs (>80% probability)"
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )
    throughput_per_second: float = Field(
        ..., description="Predictions processed per second"
    )
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate (0-1)")
