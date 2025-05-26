"""Create initial tables

Revision ID: 001
Revises: 
Create Date: 2025-01-24 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create ATMs table
    op.create_table('atms',
        sa.Column('atm_id', sa.String(length=20), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('location_address', sa.Text(), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('region', sa.String(length=50), nullable=True),
        sa.Column('bank_branch', sa.String(length=100), nullable=True),
        sa.Column('model', sa.String(length=50), nullable=True),
        sa.Column('manufacturer', sa.String(length=50), nullable=True),
        sa.Column('installation_date', sa.DateTime(), nullable=True),
        sa.Column('last_maintenance_date', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('cash_capacity', sa.Integer(), nullable=True),
        sa.Column('software_version', sa.String(length=20), nullable=True),
        sa.Column('hardware_version', sa.String(length=20), nullable=True),
        sa.Column('network_type', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint('latitude >= -90 AND latitude <= 90', name='valid_latitude'),
        sa.CheckConstraint('longitude >= -180 AND longitude <= 180', name='valid_longitude'),
        sa.CheckConstraint('cash_capacity > 0', name='positive_cash_capacity'),
        sa.PrimaryKeyConstraint('atm_id')
    )
    
    # Create indexes for ATMs
    op.create_index('idx_atm_location', 'atms', ['latitude', 'longitude'])
    op.create_index('idx_atm_region_status', 'atms', ['region', 'status'])
    op.create_index(op.f('ix_atms_region'), 'atms', ['region'])
    op.create_index(op.f('ix_atms_status'), 'atms', ['status'])

    # Create Users table
    op.create_table('users',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=True),
        sa.Column('role', sa.String(length=20), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('phone_number', sa.String(length=20), nullable=True),
        sa.Column('department', sa.String(length=50), nullable=True),
        sa.Column('permissions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'", name='valid_email'),
        sa.PrimaryKeyConstraint('user_id')
    )
    
    # Create indexes for Users
    op.create_index('idx_user_role_active', 'users', ['role', 'is_active'])
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_is_active'), 'users', ['is_active'])
    op.create_index(op.f('ix_users_role'), 'users', ['role'])

    # Create ATM Telemetry table
    op.create_table('atm_telemetry',
        sa.Column('time', sa.DateTime(), nullable=False),
        sa.Column('atm_id', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('uptime_seconds', sa.Integer(), nullable=True),
        sa.Column('temperature_celsius', sa.Float(), nullable=True),
        sa.Column('humidity_percent', sa.Float(), nullable=True),
        sa.Column('transactions_count', sa.Integer(), nullable=True),
        sa.Column('failed_transactions_count', sa.Integer(), nullable=True),
        sa.Column('transaction_amount_sum', sa.Float(), nullable=True),
        sa.Column('cash_level_percent', sa.Float(), nullable=True),
        sa.Column('receipt_paper_level', sa.Float(), nullable=True),
        sa.Column('ink_level_percent', sa.Float(), nullable=True),
        sa.Column('cpu_usage_percent', sa.Float(), nullable=True),
        sa.Column('memory_usage_percent', sa.Float(), nullable=True),
        sa.Column('disk_usage_percent', sa.Float(), nullable=True),
        sa.Column('network_latency_ms', sa.Integer(), nullable=True),
        sa.Column('network_bandwidth_mbps', sa.Float(), nullable=True),
        sa.Column('error_code', sa.String(length=10), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_count', sa.Integer(), nullable=True),
        sa.Column('additional_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint('temperature_celsius >= -50 AND temperature_celsius <= 100', name='valid_temperature'),
        sa.CheckConstraint('humidity_percent >= 0 AND humidity_percent <= 100', name='valid_humidity'),
        sa.CheckConstraint('cash_level_percent >= 0 AND cash_level_percent <= 100', name='valid_cash_level'),
        sa.CheckConstraint('cpu_usage_percent >= 0 AND cpu_usage_percent <= 100', name='valid_cpu_usage'),
        sa.CheckConstraint('memory_usage_percent >= 0 AND memory_usage_percent <= 100', name='valid_memory_usage'),
        sa.CheckConstraint('transactions_count >= 0', name='non_negative_transactions'),
        sa.CheckConstraint('failed_transactions_count >= 0', name='non_negative_failed_transactions'),
        sa.ForeignKeyConstraint(['atm_id'], ['atms.atm_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('time', 'atm_id')
    )
    
    # Create indexes for ATM Telemetry
    op.create_index('idx_telemetry_atm_time', 'atm_telemetry', ['atm_id', 'time'])
    op.create_index('idx_telemetry_error_time', 'atm_telemetry', ['error_code', 'time'])
    op.create_index('idx_telemetry_status_time', 'atm_telemetry', ['status', 'time'])
    op.create_index('idx_telemetry_time_only', 'atm_telemetry', ['time'])
    op.create_index(op.f('ix_atm_telemetry_error_code'), 'atm_telemetry', ['error_code'])
    op.create_index(op.f('ix_atm_telemetry_status'), 'atm_telemetry', ['status'])

    # Create Alert Rules table
    op.create_table('alert_rules',
        sa.Column('rule_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rule_name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('condition_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('notification_channels', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cooldown_minutes', sa.Integer(), nullable=True),
        sa.Column('target_atms', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('last_triggered', sa.DateTime(), nullable=True),
        sa.Column('trigger_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
                sa.CheckConstraint('cooldown_minutes >= 0', name='non_negative_cooldown'),
        sa.ForeignKeyConstraint(['created_by'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('rule_id'),
        sa.UniqueConstraint('rule_name')
    )
    
    # Create indexes for Alert Rules
    op.create_index('idx_alert_rule_active_severity', 'alert_rules', ['is_active', 'severity'])
    op.create_index(op.f('ix_alert_rules_is_active'), 'alert_rules', ['is_active'])
    op.create_index(op.f('ix_alert_rules_severity'), 'alert_rules', ['severity'])

    # Create Alerts table
    op.create_table('alerts',
        sa.Column('alert_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rule_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('atm_id', sa.String(length=20), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('triggered_at', sa.DateTime(), nullable=False),
        sa.Column('acknowledged_at', sa.DateTime(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('acknowledged_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('trigger_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['acknowledged_by'], ['users.user_id'], ),
        sa.ForeignKeyConstraint(['atm_id'], ['atms.atm_id'], ),
        sa.ForeignKeyConstraint(['rule_id'], ['alert_rules.rule_id'], ),
        sa.PrimaryKeyConstraint('alert_id')
    )
    
    # Create indexes for Alerts
    op.create_index('idx_alert_atm_status', 'alerts', ['atm_id', 'status'])
    op.create_index('idx_alert_severity_status', 'alerts', ['severity', 'status'])
    op.create_index('idx_alert_triggered_time', 'alerts', ['triggered_at'])
    op.create_index(op.f('ix_alerts_severity'), 'alerts', ['severity'])
    op.create_index(op.f('ix_alerts_status'), 'alerts', ['status'])

    # Create Maintenance Records table
    op.create_table('maintenance_records',
        sa.Column('maintenance_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('atm_id', sa.String(length=20), nullable=False),
        sa.Column('maintenance_type', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('scheduled_date', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('estimated_duration_minutes', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('technician_name', sa.String(length=100), nullable=True),
        sa.Column('technician_id', sa.String(length=50), nullable=True),
        sa.Column('cost_amount', sa.Float(), nullable=True),
        sa.Column('currency', sa.String(length=3), nullable=True),
        sa.Column('parts_used', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('work_performed', sa.Text(), nullable=True),
        sa.Column('issues_found', sa.Text(), nullable=True),
        sa.Column('recommendations', sa.Text(), nullable=True),
        sa.Column('next_maintenance_date', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint('cost_amount >= 0', name='non_negative_cost'),
        sa.CheckConstraint('estimated_duration_minutes > 0', name='positive_duration'),
        sa.ForeignKeyConstraint(['atm_id'], ['atms.atm_id'], ),
        sa.PrimaryKeyConstraint('maintenance_id')
    )
    
    # Create indexes for Maintenance Records
    op.create_index('idx_maintenance_atm_date', 'maintenance_records', ['atm_id', 'scheduled_date'])
    op.create_index('idx_maintenance_status_date', 'maintenance_records', ['status', 'scheduled_date'])
    op.create_index(op.f('ix_maintenance_records_maintenance_type'), 'maintenance_records', ['maintenance_type'])
    op.create_index(op.f('ix_maintenance_records_status'), 'maintenance_records', ['status'])

def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('maintenance_records')
    op.drop_table('alerts')
    op.drop_table('alert_rules')
    op.drop_table('atm_telemetry')
    op.drop_table('users')
    op.drop_table('atms')