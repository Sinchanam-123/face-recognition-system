"""Add attendance.liveness — the presentation-attack outcome per record.

Why a column rather than a separate table: this is per-row provenance, exactly
like `multi_template_match` beside it. One attendance row is one person on one
day, and "was this record verified as a live human" is a property of that row,
not an event stream. A join table would buy history nobody queries at the cost
of making the common read — "show me today, and say which rows were verified" —
a join.

The default is `'unavailable'`, and that value is chosen so this migration is
truthful about the past. Every row that already exists was written with no
liveness check in the system at all, and `'unavailable'` says precisely that. A
default of `'pass'` would retroactively assert a verification that never
happened; a nullable column would make every reader decide what NULL meant.

`batch_alter_table` because the CHECK constraint needs SQLite to rebuild the
table — SQLite cannot add a named constraint to a live table in place. Alembic
reflects the existing definition, copies the data and swaps it, which is safe
here and is why the ADD COLUMN is inside the same batch.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-07
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("attendance") as batch:
        batch.add_column(
            sa.Column(
                "liveness",
                sa.String(length=20),
                nullable=False,
                server_default="unavailable",
            )
        )
        batch.create_check_constraint(
            "ck_attendance_liveness",
            "liveness IN ('unavailable', 'pass', 'fail', 'error')",
        )


def downgrade() -> None:
    with op.batch_alter_table("attendance") as batch:
        batch.drop_constraint("ck_attendance_liveness", type_="check")
        batch.drop_column("liveness")
