"""init_tables

Revision ID: 77734f0871eb
Revises:
Create Date: 2025-05-26 08:11:04.967958+00:00

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "77734f0871eb"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "atms",
        sa.Column("atm_id", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("location_address", sa.Text(), nullable=True),
        sa.Column("model", sa.String(length=50), nullable=True),
        sa.Column("manufacturer", sa.String(length=50), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("atm_id"),
    )
    op.create_index(op.f("ix_atms_status"), "atms", ["status"], unique=False)
    op.create_table(
        "users",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=100), nullable=True),
        sa.Column("role", sa.String(length=20), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("last_login", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_is_active"), "users", ["is_active"], unique=False)
    op.create_index(op.f("ix_users_role"), "users", ["role"], unique=False)
    op.create_table(
        "alert_rules",
        sa.Column("rule_id", sa.UUID(), nullable=False),
        sa.Column("rule_name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "condition_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("severity", sa.String(length=20), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column(
            "notification_channels",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("cooldown_minutes", sa.Integer(), nullable=True),
        sa.Column(
            "target_atms",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("created_by", sa.UUID(), nullable=True),
        sa.Column("last_triggered", sa.DateTime(), nullable=True),
        sa.Column("trigger_count", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.user_id"],
        ),
        sa.PrimaryKeyConstraint("rule_id"),
        sa.UniqueConstraint("rule_name"),
    )
    op.create_index(
        op.f("ix_alert_rules_is_active"),
        "alert_rules",
        ["is_active"],
        unique=False,
    )
    op.create_index(
        op.f("ix_alert_rules_severity"),
        "alert_rules",
        ["severity"],
        unique=False,
    )
    op.create_table(
        "atm_telemetry",
        sa.Column("time", sa.DateTime(), nullable=False),
        sa.Column("atm_id", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("uptime_seconds", sa.Integer(), nullable=True),
        sa.Column("temperature_celsius", sa.Float(), nullable=True),
        sa.Column("humidity_percent", sa.Float(), nullable=True),
        sa.Column("transactions_count", sa.Integer(), nullable=True),
        sa.Column("failed_transactions_count", sa.Integer(), nullable=True),
        sa.Column("transaction_amount_sum", sa.Float(), nullable=True),
        sa.Column("cash_level_percent", sa.Float(), nullable=True),
        sa.Column("receipt_paper_level", sa.Float(), nullable=True),
        sa.Column("ink_level_percent", sa.Float(), nullable=True),
        sa.Column("cpu_usage_percent", sa.Float(), nullable=True),
        sa.Column("memory_usage_percent", sa.Float(), nullable=True),
        sa.Column("disk_usage_percent", sa.Float(), nullable=True),
        sa.Column("network_latency_ms", sa.Integer(), nullable=True),
        sa.Column("network_bandwidth_mbps", sa.Float(), nullable=True),
        sa.Column("error_code", sa.String(length=10), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_count", sa.Integer(), nullable=True),
        sa.Column(
            "additional_metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["atm_id"], ["atms.atm_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("time", "atm_id"),
    )
    op.create_index(
        op.f("ix_atm_telemetry_error_code"),
        "atm_telemetry",
        ["error_code"],
        unique=False,
    )
    op.create_index(
        op.f("ix_atm_telemetry_status"),
        "atm_telemetry",
        ["status"],
        unique=False,
    )
    op.create_table(
        "alerts",
        sa.Column("alert_id", sa.UUID(), nullable=False),
        sa.Column("rule_id", sa.UUID(), nullable=False),
        sa.Column("atm_id", sa.String(length=20), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("triggered_at", sa.DateTime(), nullable=False),
        sa.Column("acknowledged_at", sa.DateTime(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("acknowledged_by", sa.UUID(), nullable=True),
        sa.Column(
            "trigger_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("resolution_notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["acknowledged_by"],
            ["users.user_id"],
        ),
        sa.ForeignKeyConstraint(
            ["atm_id"],
            ["atms.atm_id"],
        ),
        sa.ForeignKeyConstraint(
            ["rule_id"],
            ["alert_rules.rule_id"],
        ),
        sa.PrimaryKeyConstraint("alert_id"),
    )
    op.create_index(op.f("ix_alerts_severity"), "alerts", ["severity"], unique=False)
    op.create_index(op.f("ix_alerts_status"), "alerts", ["status"], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_alerts_status"), table_name="alerts")
    op.drop_index(op.f("ix_alerts_severity"), table_name="alerts")
    op.drop_table("alerts")
    op.drop_index(op.f("ix_atm_telemetry_status"), table_name="atm_telemetry")
    op.drop_index(op.f("ix_atm_telemetry_error_code"), table_name="atm_telemetry")
    op.drop_table("atm_telemetry")
    op.drop_index(op.f("ix_alert_rules_severity"), table_name="alert_rules")
    op.drop_index(op.f("ix_alert_rules_is_active"), table_name="alert_rules")
    op.drop_table("alert_rules")
    op.drop_index(op.f("ix_users_role"), table_name="users")
    op.drop_index(op.f("ix_users_is_active"), table_name="users")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
    op.drop_index(op.f("ix_atms_status"), table_name="atms")
    op.drop_table("atms")
    # ### end Alembic commands ###
