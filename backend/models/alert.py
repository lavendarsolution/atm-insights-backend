import uuid

from database import Base
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class AlertRule(Base):
    """Configurable alert rules for automated monitoring"""

    __tablename__ = "alert_rules"

    rule_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)

    # Rule configuration
    condition_json = Column(JSONB, nullable=False)  # Flexible condition definition
    severity = Column(String(20), default="medium", index=True)
    is_active = Column(Boolean, default=True, index=True)

    # Notification settings
    notification_channels = Column(JSONB)  # email, sms, webhook, etc.
    cooldown_minutes = Column(Integer, default=10)  # Prevent spam

    # Targeting
    target_atms = Column(JSONB)  # Specific ATMs or regions

    # Metadata
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"))
    last_triggered = Column(DateTime)
    trigger_count = Column(Integer, default=0)

    # Relationships
    creator = relationship("User", back_populates="created_alerts")
    alerts = relationship("Alert", back_populates="rule")


class Alert(Base):
    """Alert instances/history"""

    __tablename__ = "alerts"

    alert_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id = Column(
        UUID(as_uuid=True), ForeignKey("alert_rules.rule_id"), nullable=False
    )
    atm_id = Column(String(20), ForeignKey("atms.atm_id"), nullable=False)

    # Alert details
    severity = Column(String(20), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)

    # Status tracking
    status = Column(
        String(20), default="active", index=True
    )  # active, acknowledged, resolved
    triggered_at = Column(DateTime, default=func.now(), nullable=False)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)

    # Assignment
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"))

    # Context data
    trigger_data = Column(JSONB)  # Snapshot of data that triggered the alert
    resolution_notes = Column(Text)

    # Relationships
    rule = relationship("AlertRule", back_populates="alerts")
    atm = relationship("ATM", back_populates="alerts")
    acknowledged_by_user = relationship("User", back_populates="acknowledged_alerts")
