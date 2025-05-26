import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database import Base


class ATMTelemetry(Base):
    """
    ATM telemetry data - optimized for TimescaleDB
    This will be converted to a hypertable via migration
    """

    __tablename__ = "atm_telemetry"

    # Primary key and time dimension
    time = Column(DateTime, nullable=False, primary_key=True)
    atm_id = Column(
        String(20),
        ForeignKey("atms.atm_id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )

    # Operational metrics
    status = Column(String(20), nullable=False, index=True)
    uptime_seconds = Column(Integer)

    # Environmental metrics
    temperature_celsius = Column(Float)
    humidity_percent = Column(Float)

    # Transaction metrics
    transactions_count = Column(Integer, default=0)
    failed_transactions_count = Column(Integer, default=0)
    transaction_amount_sum = Column(Float)  # Total transaction amount

    # Hardware health metrics
    cash_level_percent = Column(Float)
    receipt_paper_level = Column(Float)
    ink_level_percent = Column(Float)

    # System performance metrics
    cpu_usage_percent = Column(Float)
    memory_usage_percent = Column(Float)
    disk_usage_percent = Column(Float)
    network_latency_ms = Column(Integer)
    network_bandwidth_mbps = Column(Float)

    # Error tracking
    error_code = Column(String(10), index=True)
    error_message = Column(Text)
    error_count = Column(Integer, default=0)

    # Additional flexible data
    additional_metrics = Column(JSONB)

    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    atm = relationship("ATM", back_populates="telemetries")

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "temperature_celsius >= -50 AND temperature_celsius <= 100",
            name="valid_temperature",
        ),
        CheckConstraint(
            "humidity_percent >= 0 AND humidity_percent <= 100", name="valid_humidity"
        ),
        CheckConstraint(
            "cash_level_percent >= 0 AND cash_level_percent <= 100",
            name="valid_cash_level",
        ),
        CheckConstraint(
            "cpu_usage_percent >= 0 AND cpu_usage_percent <= 100",
            name="valid_cpu_usage",
        ),
        CheckConstraint(
            "memory_usage_percent >= 0 AND memory_usage_percent <= 100",
            name="valid_memory_usage",
        ),
        CheckConstraint("transactions_count >= 0", name="non_negative_transactions"),
        CheckConstraint(
            "failed_transactions_count >= 0", name="non_negative_failed_transactions"
        ),
        # Composite indexes for common queries
        Index("idx_telemetry_atm_time", "atm_id", "time"),
        Index("idx_telemetry_status_time", "status", "time"),
        Index("idx_telemetry_error_time", "error_code", "time"),
        Index("idx_telemetry_time_only", "time"),  # For time-range queries
    )
