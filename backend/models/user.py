from sqlalchemy import (
    Column, String, DateTime, Integer, Float, Text, Boolean, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
from backend.database import Base

class User(Base, TimestampMixin):
    """User accounts for system access"""
    __tablename__ = "users"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), default="operator", index=True)
    is_active = Column(Boolean, default=True, index=True)
    last_login = Column(DateTime)
    
    # Profile information
    phone_number = Column(String(20))
    department = Column(String(50))
    permissions = Column(JSONB)  # Flexible permission system
    
    # Relationships
    created_alerts = relationship("AlertRule", back_populates="creator")
    acknowledged_alerts = relationship("Alert", back_populates="acknowledged_by_user")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'", 
                       name='valid_email'),
        Index('idx_user_role_active', 'role', 'is_active'),
    )