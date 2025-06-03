from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ATMStatus(str, Enum):
    """ATM operational status"""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class NetworkStatus(str, Enum):
    """Network connectivity status"""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNSTABLE = "unstable"


class TelemetryData(BaseModel):
    """Optimized ATM telemetry data with essential fields only"""

    # Required fields
    atm_id: str = Field(..., description="ATM identifier", max_length=20)
    timestamp: str = Field(..., description="Timestamp in ISO format")
    status: ATMStatus = Field(..., description="ATM operational status")

    # Optional monitoring fields
    uptime_seconds: Optional[int] = Field(
        None, description="System uptime in seconds", ge=0
    )
    cash_level_percent: Optional[float] = Field(
        None, description="Cash level percentage", ge=0, le=100
    )
    temperature_celsius: Optional[float] = Field(
        None, description="Temperature in Celsius", ge=-10, le=60
    )
    cpu_usage_percent: Optional[float] = Field(
        None, description="CPU usage percentage", ge=0, le=100
    )
    memory_usage_percent: Optional[float] = Field(
        None, description="Memory usage percentage", ge=0, le=100
    )
    disk_usage_percent: Optional[float] = Field(
        None, description="Disk usage percentage", ge=0, le=100
    )
    network_status: Optional[NetworkStatus] = Field(
        None, description="Network connectivity status"
    )
    network_latency_ms: Optional[int] = Field(
        None, description="Network latency in milliseconds", ge=0
    )
    error_code: Optional[str] = Field(None, description="Error code", max_length=10)
    error_message: Optional[str] = Field(
        None, description="Error message", max_length=500
    )

    class Config:
        """Pydantic configuration"""

        use_enum_values = True

    def dict(self, **kwargs):
        """Override dict method to handle enum values"""
        data = super().dict(**kwargs)
        # Convert enum values to strings
        if isinstance(data.get("status"), ATMStatus):
            data["status"] = data["status"].value
        if isinstance(data.get("network_status"), NetworkStatus):
            data["network_status"] = data["network_status"].value
        return data


class TelemetryResponse(BaseModel):
    """Response model for telemetry submission"""

    success: bool
    message: str
    atm_id: str
    timestamp: str


class TelemetryBatchResponse(BaseModel):
    """Response model for batch telemetry submission"""

    success: bool
    processed_count: int
    errors: list
    timestamp: str


# Alert thresholds for monitoring
ALERT_THRESHOLDS = {
    "cash_level_critical": 15,  # Below 15% cash
    "cash_level_low": 25,  # Below 25% cash
    "temperature_high": 35,  # Above 35°C
    "temperature_low": 5,  # Below 5°C
    "cpu_high": 80,  # Above 80% CPU
    "memory_high": 85,  # Above 85% memory
    "disk_high": 90,  # Above 90% disk
    "network_timeout": 5000,  # Above 5 second latency
    "error_frequency": 5,  # More than 5 errors per hour
}
