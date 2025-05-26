from sqlalchemy import (
    Column, String, DateTime, Integer, Float, Text, Boolean, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
from .connection import Base

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

class ATM(Base, TimestampMixin):
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
    telemetries = relationship("ATMTelemetry", back_populates="atm", passive_deletes=True)
    maintenance_records = relationship("MaintenanceRecord", back_populates="atm")
    alerts = relationship("Alert", back_populates="atm")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('latitude >= -90 AND latitude <= 90', name='valid_latitude'),
        CheckConstraint('longitude >= -180 AND longitude <= 180', name='valid_longitude'),
        CheckConstraint('cash_capacity > 0', name='positive_cash_capacity'),
        Index('idx_atm_location', 'latitude', 'longitude'),
        Index('idx_atm_region_status', 'region', 'status'),
    )

class ATMTelemetry(Base):
    """
    ATM telemetry data - optimized for TimescaleDB
    This will be converted to a hypertable via migration
    """
    __tablename__ = "atm_telemetry"
    
    # Primary key and time dimension
    time = Column(DateTime, nullable=False, primary_key=True)
    atm_id = Column(String(20), ForeignKey("atms.atm_id", ondelete="CASCADE"), 
                   nullable=False, primary_key=True)
    
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
        CheckConstraint('temperature_celsius >= -50 AND temperature_celsius <= 100', 
                       name='valid_temperature'),
        CheckConstraint('humidity_percent >= 0 AND humidity_percent <= 100', 
                       name='valid_humidity'),
        CheckConstraint('cash_level_percent >= 0 AND cash_level_percent <= 100', 
                       name='valid_cash_level'),
        CheckConstraint('cpu_usage_percent >= 0 AND cpu_usage_percent <= 100', 
                       name='valid_cpu_usage'),
        CheckConstraint('memory_usage_percent >= 0 AND memory_usage_percent <= 100', 
                       name='valid_memory_usage'),
        CheckConstraint('transactions_count >= 0', name='non_negative_transactions'),
        CheckConstraint('failed_transactions_count >= 0', name='non_negative_failed_transactions'),
        
        # Composite indexes for common queries
        Index('idx_telemetry_atm_time', 'atm_id', 'time'),
        Index('idx_telemetry_status_time', 'status', 'time'),
        Index('idx_telemetry_error_time', 'error_code', 'time'),
        Index('idx_telemetry_time_only', 'time'),  # For time-range queries
    )

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

class AlertRule(Base, TimestampMixin):
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
    
    # Constraints
    __table_args__ = (
        CheckConstraint('cooldown_minutes >= 0', name='non_negative_cooldown'),
        Index('idx_alert_rule_active_severity', 'is_active', 'severity'),
    )

class Alert(Base, TimestampMixin):
    """Alert instances/history"""
    __tablename__ = "alerts"
    
    alert_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id = Column(UUID(as_uuid=True), ForeignKey("alert_rules.rule_id"), nullable=False)
    atm_id = Column(String(20), ForeignKey("atms.atm_id"), nullable=False)
    
    # Alert details
    severity = Column(String(20), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Status tracking
    status = Column(String(20), default="active", index=True)  # active, acknowledged, resolved
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
    
    # Constraints
    __table_args__ = (
        Index('idx_alert_atm_status', 'atm_id', 'status'),
        Index('idx_alert_triggered_time', 'triggered_at'),
        Index('idx_alert_severity_status', 'severity', 'status'),
    )

class MaintenanceRecord(Base, TimestampMixin):
    """Maintenance and service records"""
    __tablename__ = "maintenance_records"
    
    maintenance_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    atm_id = Column(String(20), ForeignKey("atms.atm_id"), nullable=False)
    
    # Maintenance details
    maintenance_type = Column(String(50), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Scheduling
    scheduled_date = Column(DateTime, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    estimated_duration_minutes = Column(Integer)
    
    # Status and assignment
    status = Column(String(20), default="scheduled", index=True)
    technician_name = Column(String(100))
    technician_id = Column(String(50))
    
    # Cost and parts
    cost_amount = Column(Float)
    currency = Column(String(3), default="USD")
    parts_used = Column(JSONB)  # List of parts/components
    
    # Results
    work_performed = Column(Text)
    issues_found = Column(Text)
    recommendations = Column(Text)
    next_maintenance_date = Column(DateTime)
    
    # Relationships
    atm = relationship("ATM", back_populates="maintenance_records")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('cost_amount >= 0', name='non_negative_cost'),
        CheckConstraint('estimated_duration_minutes > 0', name='positive_duration'),
        Index('idx_maintenance_atm_date', 'atm_id', 'scheduled_date'),
        Index('idx_maintenance_status_date', 'status', 'scheduled_date'),
    )