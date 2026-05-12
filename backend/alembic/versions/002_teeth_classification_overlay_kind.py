"""Add teeth_classification_overlay to image_asset_kind enum.

Revision ID: 002_tc_overlay
Revises: 001_initial
Create Date: 2026-05-08

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_tc_overlay"
down_revision: Union[str, Sequence[str], None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Append enum value for Postgres (cannot remove values on downgrade)."""
    op.execute("ALTER TYPE image_asset_kind ADD VALUE IF NOT EXISTS 'teeth_classification_overlay'")


def downgrade() -> None:
    """PostgreSQL cannot drop enum values safely; leave type unchanged."""
