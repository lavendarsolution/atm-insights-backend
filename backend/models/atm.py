import uuid
from datetime import datetime

from database import Base
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


class ATM(Base):
    """ATM master registry with comprehensive information"""

    __tablename__ = "atms"

    atm_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    location_address = Column(Text)
    model = Column(String(50))
    manufacturer = Column(String(50))
    status = Column(String(20), default="active", index=True)

    # Relationships
    telemetries = relationship(
        "ATMTelemetry", back_populates="atm", passive_deletes=True
    )
    alerts = relationship("Alert", back_populates="atm")

    created_at = Column(DateTime, default=datetime.now(), nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now(), onupdate=datetime.now(), nullable=False
    )
