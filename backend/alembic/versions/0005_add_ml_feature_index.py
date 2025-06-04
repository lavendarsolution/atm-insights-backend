"""Add ML-optimized index for bulk feature extraction

Revision ID: 0005_add_ml_feature_index
Revises: 0004_create_alert_tables
Create Date: 2025-06-04 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005_add_ml_feature_index"
down_revision: Union[str, None] = "0004_create_alert_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ML-optimized composite index for bulk feature extraction"""
    
    # Create composite index with INCLUDE clause for ML feature extraction
    # # This index optimizes the bulk feature extraction queries used by ML service
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_atm_telemetry_ml_features
        ON atm_telemetry (atm_id, time DESC, status, error_code)
        INCLUDE (cash_level_percent, temperature_celsius, cpu_usage_percent,
                memory_usage_percent, network_status, network_latency_ms, uptime_seconds)
    """)
    
    print("✅ Created ML-optimized index for bulk feature extraction")


def downgrade() -> None:
    """Remove ML-optimized index"""
    
    op.execute("DROP INDEX IF EXISTS idx_atm_telemetry_ml_features")
    
    print("✅ Removed ML-optimized index")