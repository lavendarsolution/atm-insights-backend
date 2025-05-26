"""Create TimescaleDB hypertable for telemetry

Revision ID: 002
Revises: 001
Create Date: 2025-01-24 13:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create TimescaleDB hypertable for ATM telemetry
    op.execute("""
        SELECT create_hypertable(
            'atm_telemetry', 
            'time',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
    """)
    
    # Enable compression after 1 day
    op.execute("""
        ALTER TABLE atm_telemetry SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'atm_id',
            timescaledb.compress_orderby = 'time DESC'
        );
    """)
    
    # Add compression policy
    op.execute("""
        SELECT add_compression_policy('atm_telemetry', INTERVAL '1 day');
    """)
    
    # Add retention policy (keep data for 2 years)
    op.execute("""
        SELECT add_retention_policy('atm_telemetry', INTERVAL '2 years');
    """)
    
    # Create continuous aggregates for common queries
    op.execute("""
        CREATE MATERIALIZED VIEW atm_telemetry_hourly
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', time) AS bucket,
            atm_id,
            COUNT(*) as data_points,
            AVG(temperature_celsius) as avg_temperature,
            AVG(cash_level_percent) as avg_cash_level,
            SUM(transactions_count) as total_transactions,
            SUM(failed_transactions_count) as total_failed_transactions,
            AVG(cpu_usage_percent) as avg_cpu_usage,
            AVG(memory_usage_percent) as avg_memory_usage,
            COUNT(*) FILTER (WHERE status = 'offline') as offline_count,
            COUNT(*) FILTER (WHERE error_code IS NOT NULL) as error_count
        FROM atm_telemetry
        GROUP BY bucket, atm_id;
    """)
    
    # Add refresh policy for continuous aggregate
    op.execute("""
        SELECT add_continuous_aggregate_policy('atm_telemetry_hourly',
            start_offset => INTERVAL '2 hours',
            end_offset => INTERVAL '10 minutes',
            schedule_interval => INTERVAL '10 minutes');
    """)
    
    # Create daily aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW atm_telemetry_daily
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 day', time) AS bucket,
            atm_id,
            COUNT(*) as data_points,
            AVG(temperature_celsius) as avg_temperature,
            MIN(temperature_celsius) as min_temperature,
            MAX(temperature_celsius) as max_temperature,
            AVG(cash_level_percent) as avg_cash_level,
            MIN(cash_level_percent) as min_cash_level,
            SUM(transactions_count) as total_transactions,
            SUM(failed_transactions_count) as total_failed_transactions,
            AVG(cpu_usage_percent) as avg_cpu_usage,
            MAX(cpu_usage_percent) as max_cpu_usage,
            COUNT(*) FILTER (WHERE status = 'offline') as offline_count,
            COUNT(*) FILTER (WHERE status = 'online') as online_count,
            COUNT(*) FILTER (WHERE error_code IS NOT NULL) as error_count
        FROM atm_telemetry
        GROUP BY bucket, atm_id;
    """)
    
    # Add refresh policy for daily aggregate
    op.execute("""
        SELECT add_continuous_aggregate_policy('atm_telemetry_daily',
            start_offset => INTERVAL '1 day',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """)

def downgrade() -> None:
    # Remove policies and aggregates
    op.execute("DROP MATERIALIZED VIEW IF EXISTS atm_telemetry_daily;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS atm_telemetry_hourly;")
    
    # Note: We don't revert the hypertable as it would require recreating the table
