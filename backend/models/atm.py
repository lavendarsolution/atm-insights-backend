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


class ATM(Base):
    """ATM master registry with comprehensive information"""

    __tablename__ = "atms"

    atm_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    location_address = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    region = Column(String(50), index=True)
    bank_branch = Column(String(100))
    model = Column(String(50))
    manufacturer = Column(String(50))
    installation_date = Column(DateTime)
    last_maintenance_date = Column(DateTime)
    status = Column(String(20), default="active", index=True)

    # Configuration and metadata
    cash_capacity = Column(Integer)  # Maximum cash capacity
    software_version = Column(String(20))
    hardware_version = Column(String(20))
    network_type = Column(String(20))  # ethernet, wifi, cellular

    # Relationships
    telemetries = relationship(
        "ATMTelemetry", back_populates="atm", passive_deletes=True
    )
    alerts = relationship("Alert", back_populates="atm")

    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="valid_latitude"),
        CheckConstraint(
            "longitude >= -180 AND longitude <= 180", name="valid_longitude"
        ),
        CheckConstraint("cash_capacity > 0", name="positive_cash_capacity"),
        Index("idx_atm_location", "latitude", "longitude"),
        Index("idx_atm_region_status", "region", "status"),
    )
