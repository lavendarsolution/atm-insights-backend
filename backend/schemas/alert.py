from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AlertRuleType(str, Enum):
    """Pre-defined alert rule types"""

    LOW_CASH = "low_cash"
    CRITICAL_LOW_CASH = "critical_low_cash"
    HIGH_TRANSACTION_FAILURES = "high_transaction_failures"
    NETWORK_ISSUES = "network_issues"
    HARDWARE_MALFUNCTION = "hardware_malfunction"
    MAINTENANCE_DUE = "maintenance_due"
    UNUSUAL_ACTIVITY = "unusual_activity"


class AlertSeverity(str, Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status values"""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class NotificationChannel(BaseModel):
    """Schema for notification channel configuration"""

    type: str = Field(..., description="Channel type: 'email' or 'telegram'")
    enabled: bool = Field(True, description="Whether this channel is enabled")
    config: Optional[Dict] = Field(
        default_factory=dict, description="Channel-specific configuration"
    )


class AlertCreate(BaseModel):
    """Schema for creating alerts"""

    rule_type: AlertRuleType
    atm_id: str
    severity: AlertSeverity
    title: str
    message: str
    trigger_data: Optional[Dict] = None


class AlertUpdate(BaseModel):
    """Schema for updating alerts"""

    status: Optional[AlertStatus] = Field(
        None, description="Alert status: 'active', 'acknowledged', 'resolved'"
    )
    resolution_notes: Optional[str] = None


class AlertResponse(BaseModel):
    """Schema for alert responses"""

    alert_id: UUID
    rule_type: AlertRuleType
    atm_id: str
    severity: AlertSeverity
    title: str
    message: str
    status: AlertStatus
    triggered_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[UUID]
    trigger_data: Optional[Dict]
    resolution_notes: Optional[str]

    class Config:
        from_attributes = True


class AlertStats(BaseModel):
    """Schema for alert statistics"""

    total_alerts: int
    active_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    low_alerts: int


class AlertRuleConfig(BaseModel):
    """Schema for pre-defined alert rule configuration"""

    rule_type: AlertRuleType
    name: str
    description: str
    default_severity: AlertSeverity
    default_threshold: float
    condition_description: str
    notification_channels: List[str] = Field(default_factory=lambda: ["email"])
    cooldown_minutes: int = 10


# Add schemas for alert rule endpoints
class AlertRuleCreate(BaseModel):
    """Schema for creating/configuring alert rules (pre-defined)"""

    rule_type: AlertRuleType
    is_active: bool = True
    custom_threshold: Optional[float] = None
    notification_channels: List[str] = Field(default_factory=lambda: ["email"])
    cooldown_minutes: Optional[int] = None
    target_atms: Optional[List[str]] = None  # Specific ATM IDs to monitor


class AlertRuleResponse(BaseModel):
    """Schema for alert rule responses"""

    rule_type: AlertRuleType
    name: str
    description: str
    severity: AlertSeverity
    threshold: float
    condition_description: str
    is_active: bool = True
    notification_channels: List[str]
    cooldown_minutes: int
    target_atms: Optional[List[str]] = None


class AlertRuleTestRequest(BaseModel):
    """Schema for testing alert rules"""

    rule_type: AlertRuleType
    atm_id: str
    test_data: Optional[Dict] = None
    custom_threshold: Optional[float] = None


class AlertRuleTestResponse(BaseModel):
    """Schema for alert rule test results"""

    would_trigger: bool
    evaluation_details: Dict
    simulated_alert: Optional[Dict] = None


class NotificationTestRequest(BaseModel):
    """Schema for testing notifications"""

    channels: List[str] = Field(
        ..., description="Channels to test: 'email', 'telegram'"
    )
    test_message: Optional[str] = Field(
        "Test notification from ATM Insights", description="Test message content"
    )


class NotificationTestResponse(BaseModel):
    """Schema for notification test results"""

    results: Dict[str, bool] = Field(..., description="Results for each channel tested")
    details: Dict[str, str] = Field(
        default_factory=dict, description="Additional details or error messages"
    )
