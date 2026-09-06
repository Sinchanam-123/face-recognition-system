"""
Face-recognition attendance engine (InsightFace / Python 3.14 compatible).

Why InsightFace instead of `face_recognition`/dlib?
  dlib has no prebuilt wheel for Python 3.14, so the original stack can't run on
  this machine. InsightFace (ArcFace) installs cleanly on 3.14 via onnxruntime,
  is more accurate, and bundles both detection and recognition. It produces
  512-d L2-normalized embeddings, so identity matching is a simple cosine
  similarity (a dot product).

Web adaptations:
  * The webcam runs in a background thread; Flask streams annotated JPEG frames
    as MJPEG.
  * Unknown faces are queued with a cropped thumbnail so the frontend can name
    them, instead of the notebook's blocking `input()`.

Persistence (replaced the pickle and the in-memory dict)
--------------------------------------------------------
The gallery and the attendance register both live in SQLite now, reached through
`repo.py`. Two things drove that:

  * **Attendance was a dict keyed by name.** It was lost on every restart and it
    had no day rollover at all — running past midnight silently marked nobody on
    day two, because yesterday's names were still in the dict and the write was
    first-write-wins. The date is now part of the row's unique key, so day two
    is a different key and rollover needs no explicit reset.
  * **The gallery was a pickle**, holding exactly one embedding per name. It
    could not express "these three vectors are the same person", which the
    enrolment augmentation requires, and `pickle.load` on a file path is
    arbitrary code execution.

What did **not** change is the hot path. The gallery is still a persistent
normalized `(N, 512)` float32 matrix held in memory, loaded once at startup and
appended to in place on registration. The database is the durable store, not
something queried per frame.

Heavy dependencies (cv2, insightface, numpy, pandas) are imported lazily so the
Flask app can boot and serve the UI even before they're installed.
"""

import base64
import os
import threading
import time
from datetime import datetime

# Every tunable lives in config.py with the evaluation evidence for its value.
# They are re-exported here because eval/evaluate.py reads them off this module
# (`import engine; engine.STRONG_MATCH`) to keep the harness pinned to the app,
# and app.py imports PROJECT_ROOT from here.
from config import (  # noqa: F401  (re-exported for eval/evaluate.py + app.py)
    CHECKOUT_REFRESH_S,
    COLOR_PRESENT,
    COLOR_UNCERTAIN,
    COLOR_UNKNOWN,
    DB_PATH,
    DEFAULT_SESSION,
    DET_SIZE,
    MAX_PENDING,
    MODEL_NAME,
    PENDING_TTL_S,
    PROJECT_ROOT,
    RECOGNIZE_EVERY_N_FRAMES,
    STRONG_MATCH,
    UNKNOWN_COOLDOWN_S,
    UNKNOWN_DEDUP,
    UNKNOWN_MEMORY_S,
    WEAK_MATCH,
)

# Decision outcomes for a single detected face.
MATCH_PRESENT = "present"      # >= STRONG_MATCH: recorded
MATCH_UNCERTAIN = "uncertain"  # >= WEAK_MATCH:   displayed only, NOT recorded
MATCH_UNKNOWN = "unknown"      # below:           queued for naming

# Attendance statuses, ranked. A record may only ever move UP this ladder, which
# is what lets a later confident sighting correct an earlier weaker one. Both
# terminal statuses share a rank: "registered" is already a confirmed presence,
# so a later confident frame must not rewrite it.
_STATUS_RANK = {MATCH_UNCERTAIN: 0, "present": 1, "registered": 1}


class AttendanceEngine:
    """Owns the camera thread, the known-face gallery and all live state.

    Constructing one has **no side effects** — no database connection, no file
    read, no model load. `eval/evaluate.py` instantiates one purely to read the
    thresholds off the running app, and it has no database.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

        self._latest_jpeg = None
        self._camera_ok = False
        self._last_error = None

        self._app = None            # lazily-created InsightFace FaceAnalysis

        # The gallery, held as a persistent normalized (capacity, 512) float32
        # matrix rather than rebuilt from a Python list on every frame. Only the
        # first _known_count rows are live; the buffer doubles when full, so
        # registration appends in place instead of reallocating each time.
        #
        # A row is now a TEMPLATE, not a person: one person may own several.
        # Two parallel int64 arrays carry the mapping, grown alongside the
        # matrix, so a matrix row index resolves to an identity in O(1) without
        # a database round trip.
        self._known_matrix = None
        self._template_ids = None        # (capacity,) int64: matrix row -> template id
        self._template_persons = None    # (capacity,) int64: matrix row -> person id
        self._known_count = 0            # live rows = enrolled TEMPLATES
        self._person_names = {}          # person_id -> name
        self._person_template_counts = {}  # person_id -> template count
        self._gallery_loaded = False

        # Attendance write cache: (person_id, session) -> {id, status, last_write}
        # for the day named by _marked_date. Recognition runs every few frames,
        # so without this the camera thread would issue a database write per
        # recognised face per frame. On a date change the cache is cleared,
        # which is the whole of the day-rollover logic — everything else falls
        # out of `date` being part of the attendance unique key.
        self._marked = {}
        self._marked_date = None

        # Unknown faces awaiting a name:
        #   key -> {"id","thumb"(base64 jpeg),"embedding"(np array),
        #           "similarity","last_seen"}
        self._pending = {}
        self._pending_seq = 0
        # Recently-queued unknown clusters, [(embedding, when)], kept for
        # UNKNOWN_MEMORY_S so the same face is throttled per cluster even after
        # its card is dismissed. Replaces a single global cooldown that silently
        # dropped a second, different stranger arriving in the same window.
        self._recent_unknowns = []

    # ----------------------------------------------------------------- gallery
    def load_gallery(self, force=False):
        """Load every active person's templates into the matching matrix.

        One query at startup, not one per frame. Safe to call repeatedly; only
        the first call (or `force=True`) touches the database.
        """
        if self._gallery_loaded and not force:
            return

        import db
        import repo

        with db.session_scope() as session:
            rows = repo.load_gallery(session)
            counts = repo.template_counts(session)

        with self._lock:
            self._known_matrix = None
            self._template_ids = None
            self._template_persons = None
            self._known_count = 0
            self._person_names = {}
            self._person_template_counts = dict(counts)
            for row in rows:
                self._append_known_locked(
                    repo.bytes_to_embedding(row.embedding, row.dim),
                    row.template_id,
                    row.person_id,
                    row.person_name,
                )
            self._gallery_loaded = True

    def _append_known_locked(self, embedding, template_id, person_id, name):
        """Add one TEMPLATE to the gallery matrix, growing the buffer in place.

        Caller must hold the lock (or be in single-threaded setup). Embeddings
        are re-normalized defensively: matching is a bare dot product, so a
        vector that is not unit length would silently skew every similarity it
        takes part in. Rows stored via repo.embedding_to_bytes are already unit
        length, so this is a no-op for them.
        """
        import numpy as np

        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm

        if self._known_matrix is None:
            self._known_matrix = np.zeros((8, vec.shape[0]), dtype=np.float32)
            self._template_ids = np.zeros(8, dtype=np.int64)
            self._template_persons = np.zeros(8, dtype=np.int64)
        elif self._known_count >= self._known_matrix.shape[0]:
            capacity = self._known_matrix.shape[0] * 2
            grown = np.zeros((capacity, vec.shape[0]), dtype=np.float32)
            grown[:self._known_count] = self._known_matrix[:self._known_count]
            self._known_matrix = grown

            grown_ids = np.zeros(capacity, dtype=np.int64)
            grown_ids[:self._known_count] = self._template_ids[:self._known_count]
            self._template_ids = grown_ids

            grown_persons = np.zeros(capacity, dtype=np.int64)
            grown_persons[:self._known_count] = (
                self._template_persons[:self._known_count]
            )
            self._template_persons = grown_persons

        self._known_matrix[self._known_count] = vec
        self._template_ids[self._known_count] = template_id
        self._template_persons[self._known_count] = person_id
        self._known_count += 1
        self._person_names[person_id] = name

    def _gallery(self):
        """The live (N, 512) view of the gallery, or None when empty.

        A view, not a copy — callers only read from it, and copying 512 floats
        per enrolled template on every recognised frame is exactly the cost this
        matrix exists to avoid.
        """
        if self._known_matrix is None or self._known_count == 0:
            return None
        return self._known_matrix[:self._known_count]

    def _identity_count(self):
        """Distinct people currently in the matrix. Caller holds the lock."""
        return len(self._person_names)

    def _calibration_warning_locked(self):
        """A warning string when the deployed threshold is no longer calibrated.

        Returns None while every person holds exactly one template, which is the
        condition STRONG_MATCH was measured under. See _match() for why this
        matters and eval/README.md for the measurement.
        """
        templates = self._known_count
        identities = len(self._person_names)
        if templates <= identities:
            return None
        multi = sorted(
            name
            for pid, name in self._person_names.items()
            if self._person_template_counts.get(pid, 1) > 1
        )
        who = (
            f"{multi[0]} has more than one"
            if len(multi) == 1
            else f"{', '.join(multi)} have more than one"
        )
        return (
            f"Threshold not calibrated: {templates} face templates across "
            f"{identities} people ({who}). "
            f"STRONG_MATCH = {STRONG_MATCH:.3f} was derived at exactly one "
            "template per person. Open-set false-accept rate scales with the "
            "total number of templates, not the number of people — the "
            "evaluation measured N=5 templates needing 0.412 where N=1 needs "
            "0.370. Attendance is still being recorded, and rows written now "
            "are flagged multi_template_match. Re-derive the threshold with "
            "eval/evaluate.py before trusting these records."
        )

    # -------------------------------------------------------------- attendance
    def _mark(self, person_id, status, confidence=None, multi_template=False):
        """Record attendance for a person, in the database, at most once a day.

        The three rules this used to hold in a dict, now expressed against
        `attendance`:

        * **A day is a row.** (person_id, date, session) is unique, so a second
          sighting updates rather than duplicates, and tomorrow is a different
          key. The old dict had no date in the key at all, which is why running
          across midnight marked nobody on day two.
        * **Status moves up `_STATUS_RANK` only.** A later confident sighting
          corrects an earlier weaker one; the reverse never happens. In practice
          `uncertain` no longer reaches this method — only confident matches and
          registrations record anything (see `_decide_face`) — but the upgrade
          path stays because it is what makes that guarantee safe.
        * **check_in is written once, check_out tracks the last sighting.**

        The write is throttled by CHECKOUT_REFRESH_S. Without it, a person
        standing in front of the camera would produce an UPDATE several times a
        second, forever. A status upgrade always writes immediately regardless
        of the throttle, because that is real new information.

        The database round trip happens **outside** the lock, per the rule that
        no blocking call is made while holding it. Two calls racing on the same
        person both issue the upsert; that is harmless because the statement is
        ON CONFLICT DO UPDATE, so the unique constraint resolves them into one
        row rather than a duplicate.
        """
        import db
        import repo

        now = datetime.now()
        today = now.date()
        mono = time.monotonic()
        key = (person_id, DEFAULT_SESSION)

        with self._lock:
            # Day rollover: a new calendar day invalidates every cached row id.
            if self._marked_date != today:
                self._marked.clear()
                self._marked_date = today

            entry = self._marked.get(key)
            if entry is None:
                effective_status = status
            else:
                # Never let a weaker status overwrite a stronger one.
                upgraded = _STATUS_RANK.get(status, 0) > _STATUS_RANK.get(
                    entry["status"], 0
                )
                effective_status = status if upgraded else entry["status"]
                # A status upgrade is real new information and writes at once.
                # Anything else is just "still here", which the throttle bounds.
                if not upgraded and mono - entry["last_write"] < CHECKOUT_REFRESH_S:
                    return

        # --- outside the lock -------------------------------------------------
        # One upsert covers both cases. On a first sighting it inserts with
        # check_in set; on a later one the ON CONFLICT branch advances check_out,
        # raises confidence to its running maximum, applies the status and ORs in
        # the multi-template flag — all without a preceding read, so two threads
        # racing on the same person resolve in the database rather than here.
        try:
            with db.session_scope() as session:
                attendance_id = repo.upsert_attendance(
                    session,
                    person_id=person_id,
                    day=today,
                    session_name=DEFAULT_SESSION,
                    now=now,
                    status=effective_status,
                    confidence=confidence,
                    multi_template=multi_template,
                )
        except Exception as e:  # a database hiccup must not kill the camera loop
            with self._lock:
                self._last_error = f"Attendance write failed: {e}"
            return

        with self._lock:
            self._marked[key] = {
                "id": attendance_id,
                "status": effective_status,
                "last_write": mono,
            }
            self._marked_date = today

    # -------------------------------------------------------------- public API
    def status(self):
        self.load_gallery()  # cheap; lets the UI show the gallery size early
        with self._lock:
            deps_ok, deps_msg = _deps_available()
            warning = self._calibration_warning_locked()
            return {
                "running": self._running,
                "camera_ok": self._camera_ok,
                "deps_ok": deps_ok,
                "deps_message": deps_msg,
                # known_count keeps its old meaning for the UI: how many people
                # the system can recognise. template_count is the number that
                # governs the threshold.
                "known_count": self._identity_count(),
                "identity_count": self._identity_count(),
                "template_count": self._known_count,
                "calibration_warning": warning,
                "attendance_count": len(self._marked),
                "pending_count": len(self._pending),
                "session": DEFAULT_SESSION,
                "last_error": self._last_error,
            }

    def attendance(self, day=None):
        """Today's attendance rows, read from the database."""
        import db
        import repo

        target = day or datetime.now().date()
        with db.session_scope() as session:
            return repo.attendance_for_day(session, target)

    def people(self, include_inactive=False):
        import db
        import repo

        with db.session_scope() as session:
            return repo.list_people(session, include_inactive=include_inactive)

    def pending(self):
        # Pruned here too, not only when a new unknown arrives: the UI polls
        # this every couple of seconds, so stale cards disappear even once the
        # camera has stopped seeing anyone at all.
        with self._lock:
            self._prune_pending(time.time())
            return [
                {"id": p["id"], "thumb": p["thumb"], "similarity": round(p["similarity"], 3)}
                for p in self._pending.values()
            ]

    def register(self, pending_id, name):
        """Attach a name to a queued unknown face and persist it forever.

        Returns (ok, message, info) where `info` carries person_id, template_id
        and the person's template count so the caller can write an audit row and
        surface the calibration warning.

        Typing an existing name adds a SECOND TEMPLATE to that person rather
        than creating a duplicate identity — that is the point of the schema.
        It is also what invalidates the deployed threshold, which is why the
        template count comes back to the caller.
        """
        import db
        import models
        import repo

        name = (name or "").strip()
        if not name:
            return False, "Name is required.", None

        self.load_gallery()

        with self._lock:
            key = next(
                (k for k, p in self._pending.items() if p["id"] == pending_id), None
            )
            if key is None:
                return False, "That face is no longer pending.", None
            entry = self._pending.pop(key)

        embedding = entry["embedding"]
        try:
            with db.session_scope() as session:
                result = repo.enrol_face(
                    session,
                    name=name,
                    embedding_bytes=repo.embedding_to_bytes(embedding),
                    dim=models.EMBEDDING_DIM,
                    source=models.SOURCE_ENROLMENT,
                )
        except Exception as e:
            # Put the card back: the operator's action did not take effect, and
            # silently dropping the face would lose the only capture of it.
            with self._lock:
                self._pending[key] = entry
            return False, f"Could not save to the database: {e}", None

        with self._lock:
            # Only append to the matrix once the commit succeeded, so the hot
            # path can never contain a template the durable store does not.
            self._append_known_locked(
                embedding, result.template_id, result.person_id, result.person_name
            )
            self._person_template_counts[result.person_id] = result.template_count
            # Remember the cluster so the face still in front of the camera does
            # not immediately re-queue as a new unknown in the frames before the
            # gallery match takes over.
            self._remember_unknown(embedding, time.time())
            warning = self._calibration_warning_locked()

        self._mark(
            result.person_id,
            "registered",
            confidence=None,
            multi_template=result.template_count > 1,
        )

        if result.template_count > 1:
            message = (
                f"'{name}' now has {result.template_count} face templates. "
                "They will be recognized automatically — but the match "
                "threshold is no longer calibrated; see the warning banner."
            )
        else:
            message = f"'{name}' registered and will be recognized automatically."

        return True, message, {
            "person_id": result.person_id,
            "person_name": result.person_name,
            "template_id": result.template_id,
            "template_count": result.template_count,
            "created_person": result.created_person,
            "calibration_warning": warning,
        }

    def delete_person(self, person_id):
        """Soft-delete a person and drop their templates from the matcher.

        Their attendance history stays — it is the artefact the system exists to
        produce, and a record that cannot name who it refers to is worthless.
        The gallery is reloaded rather than surgically edited: deletion is rare,
        and a full reload cannot leave the parallel arrays inconsistent.
        """
        import db
        import repo

        with db.session_scope() as session:
            person = repo.deactivate_person(session, person_id)
            if person is None:
                return False, "No such person.", None
            name = person.name

        self.load_gallery(force=True)
        with self._lock:
            self._marked.pop((person_id, DEFAULT_SESSION), None)
        return True, f"'{name}' removed from the gallery.", {"name": name}

    def dismiss(self, pending_id):
        with self._lock:
            key = next(
                (k for k, p in self._pending.items() if p["id"] == pending_id), None
            )
            if key is not None:
                entry = self._pending.pop(key)
                # Same reason as register(): a dismissed face is usually still
                # in frame, and without this it would re-queue on the next frame.
                self._remember_unknown(entry["embedding"], time.time())
        return True

    def save_csv(self):
        import pandas as pd

        records = self.attendance()
        rows = [
            {
                "Name": r["name"],
                "Session": r["session"],
                "Date": r["date"],
                "CheckIn": r["check_in"] or "",
                "CheckOut": r["check_out"] or "",
                "Status": r["status"],
                "Confidence": r["confidence"] if r["confidence"] is not None else "",
                # Carried into the export so a record written under an
                # uncalibrated threshold stays identifiable outside the database.
                "MultiTemplateMatch": r["multi_template_match"],
            }
            for r in records
        ]
        filename = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
        path = os.path.join(PROJECT_ROOT, filename)
        pd.DataFrame(
            rows,
            columns=[
                "Name", "Session", "Date", "CheckIn", "CheckOut",
                "Status", "Confidence", "MultiTemplateMatch",
            ],
        ).to_csv(path, index=False)
        return filename, path

    def frames(self):
        """Generator of raw JPEG bytes for the MJPEG stream."""
        placeholder = _placeholder_jpeg("Camera stopped")
        while True:
            with self._lock:
                frame = self._latest_jpeg
                running = self._running
            yield frame if frame is not None else placeholder
            time.sleep(0.05 if running else 0.2)

    # ------------------------------------------------------------ camera thread
    def start(self):
        with self._lock:
            if self._running:
                return True, "Already running."
        deps_ok, deps_msg = _deps_available()
        if not deps_ok:
            with self._lock:
                self._last_error = deps_msg
            return False, deps_msg
        self.load_gallery()
        with self._lock:
            self._running = True
            self._last_error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True, "Camera started."

    def stop(self):
        with self._lock:
            self._running = False
        return True, "Camera stopped."

    def _ensure_model(self):
        """Create the InsightFace analyzer once (downloads the model if needed)."""
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=DET_SIZE)
        self._app = app

    def _loop(self):
        import cv2

        try:
            self._ensure_model()
        except Exception as e:  # model download / load failure
            with self._lock:
                self._running = False
                self._last_error = f"Failed to load face model: {e}"
            return

        cap = self._open_camera(cv2)
        if cap is None:
            with self._lock:
                self._running = False
                self._camera_ok = False
                self._last_error = "Could not open webcam (device 0)."
            return
        with self._lock:
            self._camera_ok = True
            self._last_error = None

        frame_count = 0
        consecutive_fail = 0
        annotations = []  # reused between recognition frames so boxes don't flicker
        try:
            while True:
                with self._lock:
                    if not self._running:
                        break

                ok, frame = cap.read()
                if not ok or frame is None:
                    # Cameras often drop a few frames while warming up; only give
                    # up after many consecutive failures.
                    consecutive_fail += 1
                    if consecutive_fail > 60:
                        with self._lock:
                            self._last_error = "Webcam stopped delivering frames."
                        break
                    time.sleep(0.03)
                    continue
                consecutive_fail = 0

                frame_count += 1
                if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:  # CPU-friendly
                    try:
                        annotations = self._recognize(frame, cv2)
                    except Exception as e:  # never let recognition kill the stream
                        annotations = []
                        with self._lock:
                            self._last_error = f"Recognition error: {e}"

                for (x1, y1, x2, y2), label, color in annotations:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
        finally:
            cap.release()
            with self._lock:
                self._running = False
                self._camera_ok = False
                self._latest_jpeg = None

    def _open_camera(self, cv2):
        """Open device 0, preferring DirectShow on Windows for a faster start."""
        backends = [getattr(cv2, "CAP_DSHOW", 0), 0]  # DSHOW, then default
        for backend in backends:
            cap = cv2.VideoCapture(0, backend) if backend else cv2.VideoCapture(0)
            if cap.isOpened():
                # Prime the pipeline — the first read or two are often empty.
                for _ in range(5):
                    cap.read()
                    time.sleep(0.02)
                return cap
            cap.release()
        return None

    def _match(self, embedding):
        """Best gallery match for one embedding: (row, person_id, similarity).

        Caller must hold the lock. Returns (-1, None, -1.0) when nothing is
        enrolled, which is the same sentinel the previous version used.

        A matrix row is a TEMPLATE and one person may own several, so the argmax
        is mapped through `_template_persons` before anything is decided:
        matching returns a PERSON, not a template.

        ----------------------------------------------------------------------
        THRESHOLD VALIDITY — READ BEFORE ADDING TEMPLATES
        ----------------------------------------------------------------------
        `STRONG_MATCH = 0.370` is only valid while **each person has exactly one
        template**. It was derived at N=1 enrolment image per person against a
        50-identity gallery (eval run 2026-09-05, seed 42, LFW; see
        eval/README.md and eval/results/history/).

        The reason is visible in the line below. `sims` is a max over every ROW
        of the matrix, and every row is one more chance for a stranger to score
        high. So open-set false-accept rate scales with the **total number of
        templates**, not with the number of distinct identities. Giving one
        person five templates costs the same FAR as enrolling four more people.

        The evaluation measured this directly rather than assuming it: holding
        FAR at the 0.1% budget, N=5 templates per person needed a threshold of
        **0.412** where N=1 needed **0.370** (eval/README.md, "What single-image
        enrolment costs"). At N=5 the stricter threshold ate the benefit
        entirely — degraded FRR was *worse* at N=5 than at N=3.

        The threshold is deliberately NOT changed here. The enrolment
        augmentation that will add templates has not landed, and a threshold
        moved ahead of the measurement that justifies it is exactly the
        hand-picked guess this codebase spent an evaluation harness removing.
        When templates are added, re-derive it in eval/ and update
        config.STRONG_MATCH from the run — do not interpolate the table.

        Until then the engine reports a calibration warning through /api/status
        the moment any person holds more than one template, and flags every
        attendance row written while that is true
        (`attendance.multi_template_match`).
        """
        gallery = self._gallery()
        if gallery is None:
            return -1, None, -1.0
        sims = gallery @ embedding  # cosine similarity (embeddings normalized)
        row = int(sims.argmax())
        return row, int(self._template_persons[row]), float(sims[row])

    def _decide_face(self, embedding):
        """Classify one embedding and record attendance if — and only if — confident.

        Returns (decision, name, similarity), where decision is one of
        MATCH_PRESENT / MATCH_UNCERTAIN / MATCH_UNKNOWN and name is None for an
        unknown face.

        The single rule that matters: **only MATCH_PRESENT writes anything.** An
        uncertain match is drawn on the video feed and nothing else. Measured
        open-set FAR at WEAK_MATCH is 1.335% (108 of 8090 impostor probes) — for
        an attendance system a false accept is proxy attendance, so the band that
        buys a lower FRR is exactly the band that must not be trusted with a
        record. See config.WEAK_MATCH.

        Split out of _recognize so the decision can be tested with synthetic
        embeddings and no camera, model or cv2 (see backend/test_engine.py).
        """
        with self._lock:
            _row, person_id, sim = self._match(embedding)
            if sim >= STRONG_MATCH:
                name = self._person_names[person_id]
                multi = self._person_template_counts.get(person_id, 1) > 1
                mark_args = (person_id, "present", sim, multi)
            elif sim >= WEAK_MATCH:
                # Displayed, deliberately NOT recorded.
                return MATCH_UNCERTAIN, self._person_names[person_id], sim
            else:
                return MATCH_UNKNOWN, None, sim

        # _mark does its own locking and a database round trip; calling it from
        # inside the block above would hold the engine lock across I/O.
        self._mark(*mark_args)
        return MATCH_PRESENT, name, sim

    def _recognize(self, frame, cv2):
        """Detect + identify every face; return a list of (bbox, label, color)."""
        with self._lock:
            self._prune_pending(time.time())

        out = []
        for face in self._app.get(frame):
            emb = face.normed_embedding
            bbox = tuple(int(v) for v in face.bbox)
            decision, name, sim = self._decide_face(emb)

            if decision == MATCH_PRESENT:
                out.append((bbox, name, COLOR_PRESENT))
            elif decision == MATCH_UNCERTAIN:
                # "?" as well as the colour: the label has to read as "not
                # recorded" on a monochrome screenshot too.
                out.append((bbox, f"~{name}?", COLOR_UNCERTAIN))
            else:
                self._queue_unknown(emb, frame, bbox, sim, cv2)
                out.append((bbox, "Unknown", COLOR_UNKNOWN))
        return out

    # ------------------------------------------------------------ unknown queue
    def _prune_pending(self, now):
        """Drop unknown cards nobody has seen lately, and stale throttle memory.

        Caller must hold the lock.
        """
        for key in [k for k, p in self._pending.items()
                    if now - p["last_seen"] > PENDING_TTL_S]:
            self._pending.pop(key, None)
        if self._recent_unknowns:
            self._recent_unknowns = [
                (emb, when) for emb, when in self._recent_unknowns
                if now - when <= UNKNOWN_MEMORY_S
            ]

    def _remember_unknown(self, embedding, now):
        """Note that this cluster was just queued/dismissed. Caller holds the lock."""
        self._recent_unknowns.append((embedding, now))

    def _queue_unknown(self, embedding, frame, bbox, similarity, cv2):
        """Queue an unrecognised face for naming, at most once per cluster.

        Threading note: the JPEG encode happens **outside** the mutex. Encoding a
        crop is milliseconds of CPU, and holding the engine lock across it stalls
        every Flask request thread polling /api/status and /api/pending. The
        check-then-insert that straddles the gap is safe because `_recognize` is
        only ever called from the single camera thread, so there is exactly one
        producer here; the Flask threads only ever remove entries.
        """
        import numpy as np

        now = time.time()
        with self._lock:
            self._prune_pending(now)

            # Already waiting? Compare by similarity — the same person's embedding
            # varies slightly frame to frame, so an exact key would never match.
            # Refresh last_seen so a person standing in view keeps their card.
            for p in self._pending.values():
                if float(np.dot(p["embedding"], embedding)) >= UNKNOWN_DEDUP:
                    p["last_seen"] = now
                    return

            # Per-cluster throttle: only this face is held back, so a different
            # stranger arriving in the same second still gets a card.
            for emb, when in self._recent_unknowns:
                if (now - when < UNKNOWN_COOLDOWN_S
                        and float(np.dot(emb, embedding)) >= UNKNOWN_DEDUP):
                    return

            if len(self._pending) >= MAX_PENDING:
                return

            self._pending_seq += 1
            entry_id = self._pending_seq
            self._remember_unknown(embedding, now)

        # --- outside the lock ---------------------------------------------
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        thumb_b64 = ""
        if crop.size:
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                thumb_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        with self._lock:
            self._pending[entry_id] = {
                "id": entry_id,
                "thumb": thumb_b64,
                "embedding": embedding,
                "similarity": similarity,
                "last_seen": now,
            }


# --------------------------------------------------------------------- module utils
def _deps_available():
    import importlib.util

    for mod in ("cv2", "insightface", "onnxruntime", "numpy", "pandas"):
        if importlib.util.find_spec(mod) is None:
            return False, (
                f"Missing Python package '{mod}'. Install backend requirements "
                "(pip install -r backend/requirements.txt) before starting the camera."
            )
    return True, "ok"


_PLACEHOLDER_CACHE = {}


def _placeholder_jpeg(text):
    if text in _PLACEHOLDER_CACHE:
        return _PLACEHOLDER_CACHE[text]
    try:
        from PIL import Image, ImageDraw
        import io

        img = Image.new("RGB", (640, 480), (24, 27, 34))
        ImageDraw.Draw(img).text((250, 230), text, fill=(140, 150, 165))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()
    except Exception:
        data = base64.b64decode(
            "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP//////////////////////////////"
            "////////////////////////////////////////////////////wgALCAABAAEB"
            "AREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        )
    _PLACEHOLDER_CACHE[text] = data
    return data


# A single shared engine instance for the whole process.
engine = AttendanceEngine()
