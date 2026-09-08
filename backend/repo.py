"""
Query helpers. Every statement the app runs against the database lives here.

Kept separate from engine.py so the recognition code reads as recognition code,
and so the two paths that write attendance — the camera thread and the
registration endpoint — provably issue the *same* upsert rather than two
hand-rolled read-modify-writes that drift apart.

Nothing here imports numpy at module scope: engine.py is imported by
eval/evaluate.py, which must not pay for the recognition stack just to read a
threshold. The one function that needs it imports it locally, matching the
convention in the rest of the backend.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from typing import NamedTuple

from sqlalchemy import case, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models import (
    ACTION_ENROL,
    LIVENESS_PASS,
    LIVENESS_UNAVAILABLE,
    SOURCE_ENROLMENT,
    Attendance,
    AuditLog,
    FaceTemplate,
    Person,
    User,
)


class GalleryRow(NamedTuple):
    """One row of the in-memory matching matrix, as loaded from the database."""

    template_id: int
    person_id: int
    person_name: str
    embedding: bytes
    dim: int


class EnrolResult(NamedTuple):
    person_id: int
    person_name: str
    template_id: int
    created_person: bool     # False when a template was added to someone existing
    template_count: int      # this person's template count AFTER the insert


# --------------------------------------------------------------------- gallery
def load_gallery(session: Session) -> list[GalleryRow]:
    """Every template belonging to an active person, ordered by template id.

    One query, not one per person. Ordering by id keeps the matrix's row order
    stable across restarts, which makes a matrix row index reproducible when
    debugging a match.
    """
    stmt = (
        select(
            FaceTemplate.id,
            FaceTemplate.person_id,
            Person.name,
            FaceTemplate.embedding,
            FaceTemplate.dim,
        )
        .join(Person, Person.id == FaceTemplate.person_id)
        .where(Person.active.is_(True))
        .order_by(FaceTemplate.id)
    )
    return [GalleryRow(*row) for row in session.execute(stmt).all()]


def template_counts(session: Session) -> dict[int, int]:
    """person_id -> number of templates, for active people only."""
    stmt = (
        select(FaceTemplate.person_id, func.count(FaceTemplate.id))
        .join(Person, Person.id == FaceTemplate.person_id)
        .where(Person.active.is_(True))
        .group_by(FaceTemplate.person_id)
    )
    return {person_id: count for person_id, count in session.execute(stmt).all()}


def enrol_face(
    session: Session,
    name: str,
    embedding_bytes: bytes,
    dim: int,
    source: str = SOURCE_ENROLMENT,
    quality_score: float | None = None,
) -> EnrolResult:
    """Attach a template to `name`, creating the person if they are new.

    Typing a name that already exists is a **re-enrolment**: it adds a second
    template to that person rather than creating a duplicate identity. That is
    what the multi-template schema is for, and it is the path the enrolment
    augmentation needs — but it is also what invalidates STRONG_MATCH, so the
    returned `template_count` is what the caller uses to raise the calibration
    warning. See the comment at AttendanceEngine._match().

    Reactivates a soft-deleted person rather than colliding with their unique
    name: re-enrolling someone who was removed is a sensible operator action,
    and it keeps their attendance history attached.
    """
    person = session.execute(
        select(Person).where(Person.name == name)
    ).scalar_one_or_none()

    created_person = person is None
    if person is None:
        person = Person(name=name, created_at=datetime.now(), active=True)
        session.add(person)
        session.flush()  # assign person.id
    elif not person.active:
        person.active = True
        created_person = True  # re-entering the gallery; treat as a new arrival

    template = FaceTemplate(
        person_id=person.id,
        embedding=embedding_bytes,
        dim=dim,
        source=source,
        quality_score=quality_score,
        created_at=datetime.now(),
    )
    session.add(template)
    session.flush()  # assign template.id

    count = session.execute(
        select(func.count(FaceTemplate.id)).where(
            FaceTemplate.person_id == person.id
        )
    ).scalar_one()

    return EnrolResult(
        person_id=person.id,
        person_name=person.name,
        template_id=template.id,
        created_person=created_person,
        template_count=int(count),
    )


def deactivate_person(session: Session, person_id: int) -> Person | None:
    """Soft-delete: drop out of the gallery, keep every attendance row.

    A hard delete would either orphan history or cascade it away. Neither is
    acceptable for an attendance record, which is the artefact the whole system
    exists to produce.
    """
    person = session.get(Person, person_id)
    if person is None:
        return None
    person.active = False
    return person


def list_people(session: Session, include_inactive: bool = False) -> list[dict]:
    stmt = (
        select(
            Person.id,
            Person.name,
            Person.created_at,
            Person.active,
            func.count(FaceTemplate.id),
        )
        .outerjoin(FaceTemplate, FaceTemplate.person_id == Person.id)
        .group_by(Person.id)
        .order_by(Person.name)
    )
    if not include_inactive:
        stmt = stmt.where(Person.active.is_(True))
    return [
        {
            "id": pid,
            "name": name,
            "created_at": created.isoformat() if created else None,
            "active": bool(active),
            "template_count": int(count),
        }
        for pid, name, created, active, count in session.execute(stmt).all()
    ]


# ------------------------------------------------------------------ attendance
def upsert_attendance(
    session: Session,
    person_id: int,
    day: date_type,
    session_name: str,
    now: datetime,
    status: str,
    confidence: float | None,
    multi_template: bool,
    liveness: str = LIVENESS_UNAVAILABLE,
) -> int:
    """Create or update the row for (person, day, session). Returns its id.

    A single INSERT ... ON CONFLICT DO UPDATE, which is what makes the unique
    constraint load-bearing rather than decorative: the camera thread and the
    registration endpoint can both call this for the same person at the same
    moment and the database serialises them into one row. A read-then-write
    would race.

    The DO UPDATE half encodes the three rules that used to live in
    `engine._mark`:

      * **check_in is never overwritten.** It is the moment the person first
        arrived; a later frame is not new information about that.
      * **check_out moves forward.** It tracks the last confident sighting.
      * **confidence only rises**, so the column means "best we saw", not
        "whatever the last frame happened to score".

    Plus two provenance rules that work in opposite directions, deliberately:

    * **multi_template_match is sticky.** Once a row has been touched by a match
      against a person with several templates, it stays flagged — a later
      single-template match does not clear the fact that this record was partly
      written under an uncalibrated threshold. The pessimistic value wins,
      because the flag records a *doubt*.
    * **liveness promotes to 'pass' and never demotes.** A sighting that a
      liveness provider confirmed was a live human is a fact about this record;
      a later frame checked while the provider was unreachable does not undo it.
      The optimistic value wins, because the flag records a *verification*.

    Getting those two the same way round would be wrong in both cases: a sticky
    liveness would let one transient provider outage permanently mark a verified
    record unverified, and a promoting multi_template flag would erase the
    calibration doubt the moment a template was deleted.

    Note that 'fail' never reaches this method. A face that fails liveness is
    not marked present at all, so there is no row to write — see
    engine._decide_face. The value is in LIVENESS_STATES for the audit log and
    the eval harness, which speak the same vocabulary.

    Status upgrades are handled by the caller (engine._STATUS_RANK); by the time
    a call reaches here the status is the one that should win.
    """
    dialect = session.get_bind().dialect.name
    values = {
        "person_id": person_id,
        "date": day,
        "session": session_name,
        "check_in": now,
        "check_out": None,
        "confidence": confidence,
        "status": status,
        "multi_template_match": multi_template,
        "liveness": liveness,
    }

    if dialect == "sqlite":
        stmt = sqlite_insert(Attendance).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["person_id", "date", "session"],
            set_={
                # check_in: keep whatever is already there.
                "check_out": now,
                "confidence": func.max(
                    func.coalesce(Attendance.confidence, -1.0),
                    stmt.excluded.confidence,
                ),
                "status": stmt.excluded.status,
                "multi_template_match": (
                    Attendance.multi_template_match | stmt.excluded.multi_template_match
                ),
                # Promote to 'pass', never demote from it.
                "liveness": case(
                    (stmt.excluded.liveness == LIVENESS_PASS, LIVENESS_PASS),
                    else_=Attendance.liveness,
                ),
            },
        ).returning(Attendance.id)
        return int(session.execute(stmt).scalar_one())

    # Portable fallback for any other backend. Same rules, one extra round trip.
    existing = session.execute(
        select(Attendance).where(
            Attendance.person_id == person_id,
            Attendance.date == day,
            Attendance.session == session_name,
        )
    ).scalar_one_or_none()
    if existing is None:
        row = Attendance(**values)
        session.add(row)
        session.flush()
        return int(row.id)
    existing.check_out = now
    if confidence is not None:
        existing.confidence = max(existing.confidence or -1.0, confidence)
    existing.status = status
    existing.multi_template_match = existing.multi_template_match or multi_template
    if liveness == LIVENESS_PASS:
        existing.liveness = LIVENESS_PASS
    session.flush()
    return int(existing.id)


def attendance_for_day(
    session: Session, day: date_type, session_name: str | None = None
) -> list[dict]:
    """Attendance rows for one day, newest check-in first.

    The returned dicts keep the `name` / `time` / `date` / `status` keys the old
    in-memory records had, because genai.py reads exactly those and the frontend
    renders them. The new columns are added alongside rather than replacing
    them.
    """
    stmt = (
        select(Attendance, Person.name)
        .join(Person, Person.id == Attendance.person_id)
        .where(Attendance.date == day)
        .order_by(Attendance.check_in.desc())
    )
    if session_name is not None:
        stmt = stmt.where(Attendance.session == session_name)

    out = []
    for row, name in session.execute(stmt).all():
        check_in = row.check_in
        out.append(
            {
                "person_id": row.person_id,
                "name": name,
                # Legacy keys — genai.py and the CSV export read these.
                "time": check_in.strftime("%H:%M:%S") if check_in else "",
                "date": row.date.strftime("%d-%m-%Y"),
                "status": row.status,
                # New columns.
                "session": row.session,
                "check_in": check_in.isoformat() if check_in else None,
                "check_out": row.check_out.isoformat() if row.check_out else None,
                "confidence": (
                    round(row.confidence, 3) if row.confidence is not None else None
                ),
                "multi_template_match": bool(row.multi_template_match),
                "liveness": row.liveness,
            }
        )
    return out


# ----------------------------------------------------------------------- users
def find_user(session: Session, username: str) -> User | None:
    return session.execute(
        select(User).where(User.username == username)
    ).scalar_one_or_none()


def create_user(
    session: Session, username: str, password_hash: str, role: str
) -> User:
    user = User(
        username=username,
        password_hash=password_hash,
        role=role,
        created_at=datetime.now(),
        active=True,
    )
    session.add(user)
    session.flush()
    return user


def count_users(session: Session) -> int:
    return int(session.execute(select(func.count(User.id))).scalar_one())


# ------------------------------------------------------------------- audit log
def audit(
    session: Session,
    action: str,
    actor_user_id: int | None = None,
    actor_username: str | None = None,
    target: str | None = None,
    ip: str | None = None,
) -> None:
    """Append one audit row. Never raises into a caller's success path."""
    session.add(
        AuditLog(
            actor_user_id=actor_user_id,
            actor_username=actor_username,
            action=action,
            target=target,
            timestamp=datetime.now(),
            ip=ip,
        )
    )


def recent_audit(session: Session, limit: int = 100) -> list[dict]:
    stmt = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)
    return [
        {
            "id": row.id,
            "actor": row.actor_username,
            "action": row.action,
            "target": row.target,
            "timestamp": row.timestamp.isoformat(),
            "ip": row.ip,
        }
        for row in session.execute(stmt).scalars().all()
    ]


# --------------------------------------------------------------------- helpers
def embedding_to_bytes(vector) -> bytes:
    """L2-normalise and pack a 512-d vector as little-endian float32 bytes.

    Normalising here, once, on the way in, is what lets the load path be a bare
    `np.frombuffer` and matching a bare dot product. A vector that is not unit
    length would silently skew every similarity it takes part in.
    """
    import numpy as np

    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return np.ascontiguousarray(arr, dtype="<f4").tobytes()


def bytes_to_embedding(blob: bytes, dim: int):
    """Unpack stored bytes back into a (dim,) float32 array."""
    import numpy as np

    arr = np.frombuffer(blob, dtype="<f4")
    if arr.shape[0] != dim:
        raise ValueError(
            f"stored embedding has {arr.shape[0]} floats, expected {dim}"
        )
    return arr


__all__ = [
    "ACTION_ENROL",
    "EnrolResult",
    "GalleryRow",
    "attendance_for_day",
    "audit",
    "bytes_to_embedding",
    "count_users",
    "create_user",
    "deactivate_person",
    "embedding_to_bytes",
    "enrol_face",
    "find_user",
    "list_people",
    "load_gallery",
    "recent_audit",
    "template_counts",
    "upsert_attendance",
]
