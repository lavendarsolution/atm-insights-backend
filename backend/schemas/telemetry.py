from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any

class TelemetryData(BaseModel):
    """Pydantic model for telemetry data input"""
    atm_id: str = Field(..., description="ATM identifier")
    timestamp: str = Field(..., description="Timestamp in ISO format")
    status: str = Field(..., description="ATM status")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    cash_level: Optional[float] = Field(None, description="Cash level percentage")
    transactions_count: Optional[int] = Field(0, description="Number of transactions")
    failed_transactions: Optional[int] = Field(0, description="Number of failed transactions")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")
    memory_usage: Optional[float] = Field(None, description="Memory usage percentage")
    disk_usage: Optional[float] = Field(None, description="Disk usage percentage")
    network_latency_ms: Optional[int] = Field(None, description="Network latency in milliseconds")
    uptime_seconds: Optional[int] = Field(None, description="Uptime in seconds")
    error_code: Optional[str] = Field(None, description="Error code if any")
    error_message: Optional[str] = Field(None, description="Error message if any")
    additional_metrics: Optional[Dict[str, Any]] = Field(None, description="Additional metrics")

class TelemetryResponse(BaseModel):
    """Response model for telemetry submission"""
    success: bool
    message: str
    timestamp: str