import uuid
from datetime import datetime

from database import Base
from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship


class User(Base):
    """User accounts for system access"""

    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), default="operator", index=True)
    is_active = Column(Boolean, default=True, index=True)
    last_login = Column(DateTime)

    created_at = Column(DateTime, default=datetime.now(), nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now(), onupdate=datetime.now(), nullable=False
    )

    # Relationships
    created_alerts = relationship("AlertRule", back_populates="creator")
    acknowledged_alerts = relationship("Alert", back_populates="acknowledged_by_user")
