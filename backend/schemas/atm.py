from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .common import PaginatedResponse


class ATMBase(BaseModel):
    """Base ATM schema with common fields"""

    name: str = Field(..., description="ATM display name", max_length=100)
    location_address: Optional[str] = Field(None, description="ATM physical address")
    model: Optional[str] = Field(None, description="ATM model", max_length=50)
    manufacturer: Optional[str] = Field(
        None, description="ATM manufacturer", max_length=50
    )


class ATMCreate(ATMBase):
    """Schema for creating a new ATM"""

    atm_id: str = Field(..., description="Unique ATM identifier", max_length=20)
    status: Optional[str] = Field("active", description="ATM status")

    @field_validator("atm_id")
    def validate_atm_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("ATM ID cannot be empty")
        # Basic format validation (can be customized)
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "ATM ID must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v.strip().upper()

    @field_validator("status")
    def validate_status(cls, v):
        if v is None:
            return "active"
        allowed_statuses = ["active", "inactive", "maintenance", "decommissioned"]
        if v.lower() not in allowed_statuses:
            raise ValueError(f'Status must be one of: {", ".join(allowed_statuses)}')
        return v.lower()

    @field_validator("name")
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("ATM name cannot be empty")
        return v.strip()


class ATMUpdate(BaseModel):
    """Schema for updating an existing ATM"""

    name: Optional[str] = Field(None, description="ATM display name", max_length=100)
    location_address: Optional[str] = Field(None, description="ATM physical address")
    model: Optional[str] = Field(None, description="ATM model", max_length=50)
    manufacturer: Optional[str] = Field(
        None, description="ATM manufacturer", max_length=50
    )
    status: Optional[str] = Field(None, description="ATM status")

    @field_validator("status")
    def validate_status(cls, v):
        if v is None:
            return v
        allowed_statuses = ["active", "inactive", "maintenance", "decommissioned"]
        if v.lower() not in allowed_statuses:
            raise ValueError(f'Status must be one of: {", ".join(allowed_statuses)}')
        return v.lower()

    @field_validator("name")
    def validate_name(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError("ATM name cannot be empty")
        return v.strip() if v else v


class ATMResponse(ATMBase):
    """Schema for ATM response"""

    atm_id: str
    status: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ATMListResponse(PaginatedResponse[ATMResponse]):
    """Schema for paginated ATM list response"""

    pass


class ATMStatusUpdate(BaseModel):
    """Schema for status update response"""

    atm_id: str
    old_status: str
    new_status: str
    updated_at: str


class ATMStatus(BaseModel):
    """ATM status response model"""

    atm_id: str
    name: str
    region: str
    status: str
    last_update: Optional[str]
    temperature: Optional[float]
    cash_level: Optional[float]
    transactions_today: int
    error_code: Optional[str]
    error_message: Optional[str]


class ATMSummaryStats(BaseModel):
    """Schema for ATM summary statistics"""

    total_atms: int
    active_atms: int
    inactive_atms: int
    maintenance_atms: int
    decommissioned_atms: int
    last_updated: str


class ATMBulkCreateResponse(BaseModel):
    """Schema for bulk create response"""

    created_count: int
    created_atms: List[str]
    error_count: int
    errors: List[str]


class ATMDeleteResponse(BaseModel):
    """Schema for delete response"""

    message: str
    telemetry_data_preserved: bool
    telemetry_count: Optional[int] = None


# Additional schemas for extended functionality
class ATMWithTelemetry(ATMResponse):
    """ATM response with latest telemetry data"""

    last_telemetry: Optional[dict] = Field(None, description="Latest telemetry data")
    last_seen: Optional[str] = Field(None, description="Last telemetry timestamp")
    health_score: Optional[float] = Field(None, description="Health score 0-100")


class ATMRegionStats(BaseModel):
    """Regional ATM statistics"""

    region: str
    total_atms: int
    active_atms: int
    inactive_atms: int
    maintenance_atms: int
    avg_uptime: Optional[float] = None
    last_updated: str


class ATMSearchFilters(BaseModel):
    """Advanced search filters for ATMs"""

    status: Optional[List[str]] = Field(None, description="Filter by status")
    regions: Optional[List[str]] = Field(None, description="Filter by regions")
    manufacturers: Optional[List[str]] = Field(
        None, description="Filter by manufacturers"
    )
    models: Optional[List[str]] = Field(None, description="Filter by models")
    created_after: Optional[datetime] = Field(None, description="Created after date")
    created_before: Optional[datetime] = Field(None, description="Created before date")
    has_recent_telemetry: Optional[bool] = Field(
        None, description="Has telemetry in last hour"
    )


# Error response schemas
class ATMError(BaseModel):
    """Error response schema"""

    error: str
    detail: str
    atm_id: Optional[str] = None


class ATMValidationError(BaseModel):
    """Validation error response schema"""

    error: str
    field: str
    message: str
    value: Optional[str] = None
