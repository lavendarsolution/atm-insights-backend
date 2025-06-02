from datetime import datetime

from database import Base
from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.orm import relationship


class ATM(Base):
    """ATM master registry with comprehensive information"""

    __tablename__ = "atms"

    atm_id = Column(String(32), primary_key=True)
    name = Column(String(128), nullable=False)
    location_address = Column(Text)
    region = Column(String(32), nullable=False, index=True)
    model = Column(String(64))
    manufacturer = Column(String(64))
    status = Column(String(16), default="active", index=True)

    # Relationships
    telemetries = relationship(
        "ATMTelemetry", back_populates="atm", passive_deletes=True
    )
    alerts = relationship("Alert", back_populates="atm")

    created_at = Column(DateTime, default=datetime.now(), nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now(), onupdate=datetime.now(), nullable=False
    )
