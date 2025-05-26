from datetime import datetime

from database import Base
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship


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
    created_at = Column(DateTime, default=datetime.now(), nullable=False)

    # Relationships
    atm = relationship("ATM", back_populates="telemetries")
