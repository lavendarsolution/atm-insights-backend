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


class Alert(Base):
    """Alert instances/history"""

    __tablename__ = "alerts"

    alert_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_type = Column(String(50), nullable=False, index=True)  # Pre-defined rule type
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
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    # Context data
    trigger_data = Column(JSONB)  # Snapshot of data that triggered the alert
    resolution_notes = Column(Text)

    # Relationships
    atm = relationship("ATM", back_populates="alerts")
    acknowledged_by_user = relationship("User", back_populates="acknowledged_alerts")
