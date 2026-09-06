"""
SQLAlchemy models: the durable store that replaced the pickle and the dict.

Two problems motivated this schema, and each shaped one table.

**Attendance was a Python dict keyed by name.** It vanished on restart, and it
had no notion of a day at all — running the server across midnight silently
marked nobody on day two, because every name was still in the dict from day one
and `_mark` would not write twice. Making `date` part of the row and part of the
unique key fixes that structurally: day two is a different key, so it is a
different row, and nothing has to remember to reset anything.

**The gallery was a pickle holding one embedding per name.** It could not hold
a second template for the same person, which the planned enrolment augmentation
requires, and `pickle.load` on a file path is arbitrary code execution — not
acceptable in a system whose whole job is deciding who is allowed to be marked
present. `person` and `face_template` split identity from evidence, so a person
can accumulate templates from enrolment and from augmentation.

Types are kept deliberately narrow (no JSON columns, no pickled blobs beyond
the raw float32 embedding) so the schema stays readable from `sqlite3` and
portable off SQLite.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Embedding width. ArcFace `buffalo_sc` produces 512-d L2-normalised vectors;
# carried per row rather than assumed so that a model change is detectable at
# load time instead of silently reshaping a buffer.
EMBEDDING_DIM = 512

# `face_template.source` values.
SOURCE_ENROLMENT = "enrolment"   # a real captured frame an operator named
SOURCE_AUGMENTED = "augmented"   # synthesised from an enrolment template
TEMPLATE_SOURCES = (SOURCE_ENROLMENT, SOURCE_AUGMENTED)

# `user.role` values.
ROLE_ADMIN = "admin"    # enrol faces, delete people, export data, drive camera
ROLE_VIEWER = "viewer"  # read attendance and nothing else
ROLES = (ROLE_ADMIN, ROLE_VIEWER)


class Base(DeclarativeBase):
    """Declarative base. Alembic autogenerate reads metadata off this."""


class Person(Base):
    """One enrolled human being.

    Separated from `face_template` because a person is not their embedding: the
    augmentation work will give one person several, and an attendance record
    has to point at the person regardless of which template happened to win the
    argmax on a given frame.
    """

    __tablename__ = "person"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Unique because the operator names a face by typing a name, and the UI,
    # the CSV export and genai.py all key on it. Typing an existing name is
    # therefore a re-enrolment of that person — it attaches another template
    # rather than creating a second Person (see repo.enrol_face).
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Soft delete. Attendance rows reference a person forever, so a hard delete
    # would either orphan history or cascade it away; deactivating drops the
    # person out of the gallery matrix while leaving every record they appear in
    # intact and attributable.
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
    )

    templates: Mapped[list["FaceTemplate"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Person id={self.id} name={self.name!r} active={self.active}>"


class FaceTemplate(Base):
    """One 512-d embedding belonging to a person. Many per person.

    This is the table the whole migration exists for. The pickle stored
    ``{"embeddings": [...], "names": [...]}`` — two parallel lists, one entry
    each, with no way to express "these three vectors are the same person".

    The embedding is stored as raw little-endian float32 bytes rather than a
    JSON array or a pickled ndarray: it is 2048 bytes read straight into the
    gallery matrix with ``np.frombuffer``, no parsing and no code execution.
    Vectors are L2-normalised **before** they are stored, so matching stays a
    bare dot product and the load path does no arithmetic at all.
    """

    __tablename__ = "face_template"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("person.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Stored explicitly so a future model with a different width is caught at
    # load time rather than corrupting the matrix silently.
    dim: Mapped[int] = mapped_column(
        Integer, nullable=False, default=EMBEDDING_DIM,
        server_default=str(EMBEDDING_DIM),
    )

    # 'enrolment' | 'augmented'. The distinction matters for the ablation: an
    # augmented template must be identifiable so a run can be reproduced, and so
    # the augmentation can be rolled back without touching real enrolments.
    source: Mapped[str] = mapped_column(String(20), nullable=False)

    # Detector confidence / sharpness at capture, when known. Nullable because
    # the live registration path does not currently compute one and the pickle
    # migration has nothing to supply.
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    person: Mapped[Person] = relationship(back_populates="templates")

    __table_args__ = (
        CheckConstraint(
            "source IN ('enrolment', 'augmented')", name="ck_face_template_source"
        ),
        CheckConstraint("dim > 0", name="ck_face_template_dim_positive"),
    )

    def __repr__(self) -> str:
        return (
            f"<FaceTemplate id={self.id} person_id={self.person_id} "
            f"source={self.source!r}>"
        )


class Attendance(Base):
    """One person's presence in one session on one day.

    ``UNIQUE (person_id, date, session)`` is the load-bearing constraint. It is
    what makes the write path an upsert instead of a read-modify-write, so two
    threads racing on the same person resolve in the database rather than
    producing a duplicate row — and it is what gives day rollover for free,
    since tomorrow is simply a different key.
    """

    __tablename__ = "attendance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("person.id"), nullable=False
    )

    # Local calendar date of the sighting. Local rather than UTC on purpose: an
    # attendance day is the operator's day, and a UTC date would roll over
    # mid-afternoon in some timezones and mid-morning in others.
    date: Mapped[date_type] = mapped_column(Date, nullable=False)

    # See config.DEFAULT_SESSION. One value today; the column is what lets a day
    # be split into named periods later without a migration.
    session: Mapped[str] = mapped_column(String(50), nullable=False)

    # First confident sighting of the session.
    check_in: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Last confident sighting, refreshed at most every
    # config.CHECKOUT_REFRESH_S so the camera thread does not issue an UPDATE
    # per recognised frame.
    check_out: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Best cosine similarity seen for this row. Monotonically increasing — a
    # later, more confident frame raises it, a worse one does not lower it, so
    # the column answers "how sure were we, at best" rather than "how sure were
    # we on the last frame that happened to land".
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 'present' | 'registered'. Moves up engine._STATUS_RANK only. The
    # 'uncertain' band never reaches this table at all — measured open-set FAR
    # at WEAK_MATCH is 1.335%, which as an attendance record is proxy
    # attendance. See config.WEAK_MATCH.
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    # TRUE when the person matched here held more than one face template at the
    # time of any match contributing to this row (sticky: once set, it stays
    # set even if a template is later removed).
    #
    # This exists because STRONG_MATCH = 0.370 was derived at exactly one
    # template per person. Open-set false-accept rate scales with the total
    # number of enrolled templates, so the moment anyone has two, the threshold
    # that produced this row is no longer the threshold that was measured. The
    # flag makes "which records were written under an uncalibrated threshold"
    # a query rather than a reconstruction from log timestamps.
    multi_template_match: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )

    person: Mapped[Person] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "person_id", "date", "session", name="uq_attendance_person_date_session"
        ),
        Index("ix_attendance_date_session", "date", "session"),
    )

    def __repr__(self) -> str:
        return (
            f"<Attendance id={self.id} person_id={self.person_id} "
            f"date={self.date} session={self.session!r} status={self.status!r}>"
        )


class User(Base):
    """An operator account. Two roles: admin writes, viewer reads.

    `password_hash` holds a bcrypt hash and nothing else — see security.py. No
    column here ever holds a plaintext or reversibly-encoded credential.
    """

    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    # bcrypt output, ASCII, 60 chars for the current cost. Sized generously so a
    # future move to a longer format is not a migration.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    role: Mapped[str] = mapped_column(String(20), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Deactivation rather than deletion, so audit_log.actor_user_id keeps
    # resolving for everything the account ever did.
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
    )

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'viewer')", name="ck_user_role"),
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} role={self.role!r}>"


class AuditLog(Base):
    """Append-only record of every enrolment, deletion and failed login.

    Nothing in the app updates or deletes a row here.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Nullable: a failed login carries a username that may match no account at
    # all, and that attempt is exactly the one worth recording.
    actor_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id"), nullable=True
    )

    # Denormalised on purpose. It survives the user being deactivated, and it
    # captures the username as typed on a failed login, which has no id to
    # point at.
    actor_username: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Short stable verbs: 'enrol', 'person.delete', 'login.failed',
    # 'login.success', 'attendance.export', 'user.create'.
    action: Mapped[str] = mapped_column(String(50), nullable=False)

    # What was acted on, human-readable: "person:12 (Ada Lovelace)".
    target: Mapped[str | None] = mapped_column(String(255), nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )

    ip: Mapped[str | None] = mapped_column(String(45), nullable=True)  # IPv6-wide

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} action={self.action!r} "
            f"actor={self.actor_username!r}>"
        )


# Audit action verbs, as constants so a typo is an ImportError rather than a
# row nobody can find later.
ACTION_LOGIN_SUCCESS = "login.success"
ACTION_LOGIN_FAILED = "login.failed"
ACTION_ENROL = "enrol"
ACTION_PERSON_DELETE = "person.delete"
ACTION_EXPORT = "attendance.export"
ACTION_USER_CREATE = "user.create"
