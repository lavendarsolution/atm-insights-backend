"""Create optimized ATM telemetry tables

Revision ID: 0002_create_atm_telemetry_table
Revises: 0001_create_atm_tables
Create Date: 2025-05-26 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_create_atm_telemetry_table"
down_revision: Union[str, None] = "0001_create_atm_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create optimized ATM telemetry table
    op.create_table(
        "atm_telemetry",
        sa.Column("time", sa.DateTime(), nullable=False),
        sa.Column("atm_id", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("uptime_seconds", sa.Integer(), nullable=True),
        sa.Column("cash_level_percent", sa.Float(), nullable=True),
        sa.Column("temperature_celsius", sa.Float(), nullable=True),
        sa.Column("cpu_usage_percent", sa.Float(), nullable=True),
        sa.Column("memory_usage_percent", sa.Float(), nullable=True),
        sa.Column("disk_usage_percent", sa.Float(), nullable=True),
        sa.Column("network_status", sa.String(length=20), nullable=True),
        sa.Column("network_latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=10), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["atm_id"], ["atms.atm_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("time", "atm_id"),
    )

    # Create essential indexes for performance
    op.create_index(
        op.f("ix_atm_telemetry_status"), "atm_telemetry", ["status"], unique=False
    )
    op.create_index(
        op.f("ix_atm_telemetry_error_code"),
        "atm_telemetry",
        ["error_code"],
        unique=False,
    )
    op.create_index(
        op.f("ix_atm_telemetry_network_status"),
        "atm_telemetry",
        ["network_status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_atm_telemetry_atm_id_time"),
        "atm_telemetry",
        ["atm_id", "time"],
        unique=False,
    )

    # Convert to TimescaleDB hypertable (if TimescaleDB is available)
    try:
        op.execute("SELECT create_hypertable('atm_telemetry', 'time')")
        print("✅ Created TimescaleDB hypertable")
    except Exception as e:
        print(f"⚠️ TimescaleDB not available, using regular table: {e}")


def downgrade() -> None:
    # Drop indexes
    op.drop_index(op.f("ix_atm_telemetry_atm_id_time"), table_name="atm_telemetry")
    op.drop_index(op.f("ix_atm_telemetry_network_status"), table_name="atm_telemetry")
    op.drop_index(op.f("ix_atm_telemetry_error_code"), table_name="atm_telemetry")
    op.drop_index(op.f("ix_atm_telemetry_status"), table_name="atm_telemetry")

    # Drop tables
    op.drop_table("atm_telemetry")
