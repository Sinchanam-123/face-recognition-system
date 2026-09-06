"""Initial schema: person, face_template, attendance, user, audit_log.

This is a create-from-empty revision, not a conversion. Before it there was no
SQL schema at all — the gallery was a pickle (`face_db.pkl`) and the attendance
register was a Python dict that lived only as long as the process. So there is
nothing here to migrate *from*; the data import from the pickle is a separate
one-off script, `backend/migrate_pickle.py`, run after this.

Revision ID: 0001
Revises:
Create Date: 2026-09-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "person",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("active", sa.Boolean(), server_default="1", nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # One person, MANY templates. This is the shape the pickle could not express
    # and the enrolment augmentation requires.
    op.create_table(
        "face_template",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=False),
        # Raw little-endian float32 bytes, already L2-normalised. Read straight
        # into the gallery matrix with np.frombuffer: no parsing, no pickle, no
        # code execution.
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column("dim", sa.Integer(), server_default="512", nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["person_id"], ["person.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "source IN ('enrolment', 'augmented')", name="ck_face_template_source"
        ),
        sa.CheckConstraint("dim > 0", name="ck_face_template_dim_positive"),
    )
    op.create_index(
        "ix_face_template_person_id", "face_template", ["person_id"], unique=False
    )

    op.create_table(
        "attendance",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("session", sa.String(length=50), nullable=False),
        sa.Column("check_in", sa.DateTime(), nullable=True),
        sa.Column("check_out", sa.DateTime(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "multi_template_match", sa.Boolean(), server_default="0", nullable=False
        ),
        sa.ForeignKeyConstraint(["person_id"], ["person.id"]),
        sa.PrimaryKeyConstraint("id"),
        # The load-bearing constraint. It gives day rollover for free (tomorrow
        # is a different key) and lets the write path be a single ON CONFLICT
        # upsert instead of a racy read-modify-write.
        sa.UniqueConstraint(
            "person_id", "date", "session", name="uq_attendance_person_date_session"
        ),
    )
    op.create_index(
        "ix_attendance_date_session", "attendance", ["date", "session"], unique=False
    )

    op.create_table(
        "user",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        # bcrypt hash. Never a plaintext or reversibly-encoded credential.
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("active", sa.Boolean(), server_default="1", nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        sa.CheckConstraint("role IN ('admin', 'viewer')", name="ck_user_role"),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), nullable=False),
        # Nullable: a failed login may name no real account, and that attempt is
        # exactly the one worth recording.
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        # Denormalised so it survives the account being deactivated, and so a
        # failed login has something to record at all.
        sa.Column("actor_username", sa.String(length=100), nullable=True),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("target", sa.String(length=255), nullable=True),
        sa.Column(
            "timestamp", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("ip", sa.String(length=45), nullable=True),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_log_timestamp", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_table("user")
    op.drop_index("ix_attendance_date_session", table_name="attendance")
    op.drop_table("attendance")
    op.drop_index("ix_face_template_person_id", table_name="face_template")
    op.drop_table("face_template")
    op.drop_table("person")
