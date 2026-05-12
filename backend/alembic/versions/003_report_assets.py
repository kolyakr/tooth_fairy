"""Add report_assets table for generated PDF reports.

Revision ID: 003_report_assets
Revises: 002_tc_overlay
Create Date: 2026-05-09
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003_report_assets"
down_revision: Union[str, Sequence[str], None] = "002_tc_overlay"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create generated report storage table."""
    op.create_table(
        "report_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("reviewer", sa.String(length=255), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False, server_default="application/pdf"),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_report_assets_analysis_id", "report_assets", ["analysis_id"])


def downgrade() -> None:
    """Drop generated report storage table."""
    op.drop_index("ix_report_assets_analysis_id", table_name="report_assets")
    op.drop_table("report_assets")
