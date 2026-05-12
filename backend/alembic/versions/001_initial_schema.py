"""Initial schema: patients, analyses, image assets, findings, audit entries.

Revision ID: 001_initial
Revises:
Create Date: 2026-05-08

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from backend.app.db.base import Base
from backend.app.db import models  # noqa: F401 — register mappers

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables from ORM metadata (bootstrap migration)."""
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    """Drop all application tables."""
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
