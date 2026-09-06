"""
Tunable constants for the recognition engine, and the evidence for each one.

Every threshold here is either measured or explicitly marked as a guess. The
measured ones cite the evaluation run that produced them so a number can always
be traced back to the experiment that justifies it:

    eval run      2026-09-05 22:00:57, seed 42
    dataset       LFW, 158 identities, 4324 images (embedding cache
                  b9cf72cad14bf2bb)
    protocol      open-set identification - max similarity over the whole
                  gallery, argmax for the name, accept above threshold, which
                  is exactly what AttendanceEngine._recognize does
    operating pt  N=1 enrolment image per person (what register() stores),
                  gallery of 50 enrolled identities, 5 trials, 80 impostor
                  identities held out of every gallery
    full detail   eval/results/metrics.json, eval/README.md

``eval/evaluate.py`` imports MODEL_NAME, DET_SIZE, STRONG_MATCH and WEAK_MATCH
from ``engine`` (which re-exports them from here) rather than restating them, so
the harness cannot silently drift from the app. If you change a threshold in
this file, re-running the harness reports the new operating point with no edits
on the eval side.

This module must stay import-cheap: no cv2, no numpy, no insightface. engine.py
imports it at module scope, and the Flask app has to boot without the
recognition stack installed.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LEGACY. The pickled gallery this system used to run on: 512-d ArcFace
# embeddings as {"embeddings": [...], "names": [...]}. It is no longer read by
# the running app — `pickle.load` on a file path is arbitrary code execution,
# which is not acceptable in a security-adjacent system. The durable store is
# now SQLite (see backend/models.py).
#
# The path survives for exactly one consumer: backend/migrate_pickle.py, the
# one-off import that reads it (through a restricted unpickler) into the
# person / face_template tables. The file itself is deliberately left on disk
# untouched as a backup; nothing deletes it.
DB_PATH = os.path.join(PROJECT_ROOT, "face_db.pkl")

# InsightFace model pack (auto-downloaded once to ~/.insightface on first use).
# buffalo_sc is small (~15 MB) and CPU-friendly. The evaluation embeds with this
# exact pack and det_size, so its numbers transfer directly.
MODEL_NAME = "buffalo_sc"
DET_SIZE = (640, 640)

# --------------------------------------------------------------- match thresholds
# Cosine similarity on L2-normalized embeddings, so a plain dot product in
# [-1, 1]; higher = more similar.

# Accept as a confident match and RECORD ATTENDANCE at or above this.
#
# Measured on the open-set protocol above (N=1 enrolment, gallery 50):
#     threshold 0.370 -> FAR 0.099% (8 of 8090 impostor probes)
#                        FRR 1.27%
#                        MISID 0.02% (1 of 6143)
#
# Chosen as the lowest threshold holding open-set FAR within a 0.1% budget. It
# replaces a hand-picked 0.50, which measured FAR 0.000% and **FRR 9.88%** at
# the same operating point: 0.50 rejected roughly one in ten genuine frames to
# buy 0.099 percentage points of FAR. That trade is not worth making for an
# attendance system, where a false reject is visible and self-correcting (the
# person is still standing there and recognition runs every few frames) while
# the FRR cost is paid on every user, every session.
#
# This number assumes a gallery of at most ~50 enrolled identities. Open-set FAR
# grows with gallery size, so re-run eval/evaluate.py before enrolling
# substantially more people than that; it is measured, not extrapolated.
#
# IT ALSO ASSUMES ONE TEMPLATE PER PERSON, which the schema no longer enforces.
# See the block comment at AttendanceEngine._match() in engine.py: open-set FAR
# scales with the TOTAL number of enrolled templates, not the number of
# identities, because the score is a max over every row of the gallery matrix.
# The eval measured this directly — N=5 templates per person needed 0.412 where
# N=1 needs 0.370 (eval/README.md, "What single-image enrolment costs"). The
# engine reports a calibration warning through /api/status the moment any person
# holds more than one template, and flags the attendance rows written while that
# is true.
STRONG_MATCH = 0.370

# DISPLAY ONLY. A match between WEAK_MATCH and STRONG_MATCH is drawn on the
# video feed in a distinct colour so an operator can see the system nearly
# recognised someone - but it MUST NOT write an attendance record.
#
# Measured open-set FAR at 0.32 is **1.335%** (108 of 8090 impostor probes),
# i.e. roughly one stranger frame in 75 accepted. For an attendance system a
# false accept is proxy attendance: it marks the wrong person present, silently,
# and `_mark` is first-write-wins so a single bad frame in a whole session
# sticks. Nothing in this band is confident enough to put in the record.
WEAK_MATCH = 0.32

# ------------------------------------------------------------- unknown-face queue
# Two unknown faces are treated as the same person (so we queue ONE card for
# them) when their embeddings are at least this similar. Embeddings drift a
# little every frame, so this must be a similarity check, not an exact match.
# Kept just above the measured impostor range - impostor pairs score 0.011 on
# average (sd 0.069) on the same eval run - so pose and expression changes of
# the SAME face still merge while two different strangers do not.
UNKNOWN_DEDUP = 0.35

# Per-cluster throttle: how long before the SAME unknown face may be queued
# again after it was last added, dismissed or registered. Deliberately per
# cluster rather than global - a single global cooldown silently dropped a
# second, different stranger who appeared within the window.
UNKNOWN_COOLDOWN_S = 1.5

# How long a recently-added unknown cluster is remembered for that throttle,
# including after its card is dismissed. Stops a dismissed face from
# immediately re-queueing while the person is still in frame.
UNKNOWN_MEMORY_S = 30.0

# Drop a pending unknown card that has not been seen for this long. Without it
# the queue only ever grew: someone who walked past once stayed in the operator's
# list until manually dismissed.
PENDING_TTL_S = 60.0

# Hard cap on the queue, belt-and-suspenders against flooding if a face's
# embedding is jittery enough to slip past the dedup check every frame.
MAX_PENDING = 25

# -------------------------------------------------------------------- attendance
# The `session` half of the attendance unique key (person_id, date, session).
# One value means one attendance row per person per day, which is exactly the
# behaviour the in-memory dict had — minus the missing day rollover, which the
# `date` half now provides for free. The column exists so that splitting a day
# into named periods later is a config change and a write-path rule, not a
# schema migration.
DEFAULT_SESSION = "default"

# How often a person who is still in front of the camera gets their `check_out`
# pushed forward. check_in is written once, on the first confident sighting of
# the session; check_out then tracks the last confident sighting, so it reads as
# "last seen". Recognition runs every RECOGNIZE_EVERY_N_FRAMES frames and would
# otherwise issue an UPDATE per recognised face per frame — several a second,
# per person, forever. 30 s bounds that to one write per person per half minute
# while keeping check_out accurate to well inside the granularity anyone reads
# an attendance record at.
CHECKOUT_REFRESH_S = 30.0

# ------------------------------------------------------------------- camera loop
# Run recognition on every Nth frame to keep CPU sane; the previous frame's
# boxes are reused in between so they don't flicker.
RECOGNIZE_EVERY_N_FRAMES = 3

# Annotation colours, BGR (OpenCV order). The uncertain band gets its own colour
# precisely because it means "shown but NOT recorded".
COLOR_PRESENT = (0, 200, 0)      # green  - confident, attendance recorded
COLOR_UNCERTAIN = (0, 165, 255)  # orange - near miss, nothing recorded
COLOR_UNKNOWN = (0, 0, 255)      # red    - unknown, queued for naming
