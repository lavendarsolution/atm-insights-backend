from datetime import datetime

from database import Base
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship


class ATMTelemetry(Base):
    """
    Optimized ATM telemetry data model for MVP
    Essential fields only for TimescaleDB hypertable
    """

    __tablename__ = "atm_telemetry"

    # Primary key and time dimension (TimescaleDB requirement)
    time = Column(DateTime, nullable=False, primary_key=True)
    atm_id = Column(
        String(20),
        ForeignKey("atms.atm_id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )

    # Essential operational metrics
    status = Column(
        String(20), nullable=False, index=True
    )  # online|offline|error|maintenance
    uptime_seconds = Column(Integer)  # System uptime

    # Critical business metric
    cash_level_percent = Column(Float)  # Most important for ATM operations

    # Environmental monitoring
    temperature_celsius = Column(Float)  # Hardware health indicator

    # System performance metrics
    cpu_usage_percent = Column(Float)  # System performance
    memory_usage_percent = Column(Float)  # System performance
    disk_usage_percent = Column(Float)  # Storage monitoring

    # Network connectivity
    network_status = Column(String(20), index=True)  # connected|disconnected|unstable
    network_latency_ms = Column(Integer)  # Connection quality

    # Error tracking for alerts
    error_code = Column(String(10), index=True)  # Standardized error codes
    error_message = Column(Text)  # Human-readable error description

    # Relationships
    atm = relationship("ATM", back_populates="telemetries")
