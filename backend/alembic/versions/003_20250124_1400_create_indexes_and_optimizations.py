"""Create additional indexes and optimizations

Revision ID: 003
Revises: 002
Create Date: 2025-01-24 14:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create partial indexes for performance
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_telemetry_recent_24h 
        ON atm_telemetry (atm_id, time DESC) 
        WHERE time >= NOW() - INTERVAL '24 hours';
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_telemetry_errors_recent 
        ON atm_telemetry (atm_id, error_code, time DESC) 
        WHERE error_code IS NOT NULL AND time >= NOW() - INTERVAL '7 days';
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_telemetry_low_cash 
        ON atm_telemetry (atm_id, time DESC) 
        WHERE cash_level_percent < 20;
    """)
    
    # Create function for latest telemetry per ATM (for dashboard queries)
    op.execute("""
        CREATE OR REPLACE FUNCTION get_latest_atm_telemetry()
        RETURNS TABLE (
            atm_id VARCHAR(20),
            latest_time TIMESTAMP,
            status VARCHAR(20),
            temperature_celsius FLOAT,
            cash_level_percent FLOAT,
            transactions_count INTEGER,
            failed_transactions_count INTEGER,
            cpu_usage_percent FLOAT,
            memory_usage_percent FLOAT,
            error_code VARCHAR(10),
            error_message TEXT
        ) AS $
        BEGIN
            RETURN QUERY
            SELECT DISTINCT ON (t.atm_id)
                t.atm_id,
                t.time as latest_time,
                t.status,
                t.temperature_celsius,
                t.cash_level_percent,
                t.transactions_count,
                t.failed_transactions_count,
                t.cpu_usage_percent,
                t.memory_usage_percent,
                t.error_code,
                t.error_message
            FROM atm_telemetry t
            ORDER BY t.atm_id, t.time DESC;
        END;
        $ LANGUAGE plpgsql;
    """)
    
    # Create materialized view for real-time dashboard
    op.execute("""
        CREATE MATERIALIZED VIEW atm_status_summary AS
        SELECT 
            a.atm_id,
            a.name,
            a.region,
            a.location_address,
            a.status as atm_status,
            t.status as current_status,
            t.time as last_update,
            t.temperature_celsius,
            t.cash_level_percent,
            t.transactions_count,
            t.failed_transactions_count,
            t.error_code,
            t.error_message,
            CASE 
                WHEN t.time < NOW() - INTERVAL '5 minutes' THEN 'offline'
                WHEN t.status IS NULL THEN 'unknown'
                ELSE t.status 
            END as effective_status
        FROM atms a
        LEFT JOIN LATERAL (
            SELECT * FROM atm_telemetry 
            WHERE atm_id = a.atm_id 
            ORDER BY time DESC 
            LIMIT 1
        ) t ON true
        WHERE a.status = 'active';
    """)
    
    # Create unique index for materialized view
    op.execute("""
        CREATE UNIQUE INDEX idx_status_summary_atm_id 
        ON atm_status_summary (atm_id);
    """)
    
    # Create function to refresh status summary
    op.execute("""
        CREATE OR REPLACE FUNCTION refresh_atm_status_summary()
        RETURNS void AS $
        BEGIN
            REFRESH MATERIALIZED VIEW CONCURRENTLY atm_status_summary;
        END;
        $ LANGUAGE plpgsql;
    """)

def downgrade() -> None:
    # Drop functions and views
    op.execute("DROP FUNCTION IF EXISTS refresh_atm_status_summary();")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS atm_status_summary;")
    op.execute("DROP FUNCTION IF EXISTS get_latest_atm_telemetry();")
    
    # Drop indexes
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_telemetry_recent_24h;")
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_telemetry_errors_recent;")
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_telemetry_low_cash;")
