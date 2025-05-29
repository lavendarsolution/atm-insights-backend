"""Create atm tables

Revision ID: 0002_create_atm_tables
Revises: 0001_create_users_tables
Create Date: 2025-05-26 08:11:04.967958+00:00

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_create_atm_tables"
down_revision: Union[str, None] = "0001_create_users_tables"
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


def downgrade() -> None:
    op.drop_index(op.f("ix_atms_status"), table_name="atms")
    op.drop_table("atms")
