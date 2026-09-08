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

# --------------------------------------------------------------------- liveness
# Presentation-attack detection. See backend/liveness.py for why there is no
# heuristic fallback and why activation needs an explicit env flag.

# The outcome vocabulary. Defined HERE rather than in models.py, even though
# `attendance.liveness` is a database column, because engine.py needs these at
# module scope and models.py drags in SQLAlchemy. engine.py is imported by
# eval/evaluate.py purely to read thresholds off the running app, and that
# import must stay as cheap as it is today. models.py re-exports them so schema
# readers still find them where they would expect to.
#
# 'unavailable' is the DEFAULT and describes the pre-existing world: no provider
# configured, so no check was attempted. It is not a soft failure and must never
# be read as one.
LIVENESS_UNAVAILABLE = "unavailable"  # no provider; no check attempted
LIVENESS_PASS = "pass"                # a check ran and saw a live person
LIVENESS_FAIL = "fail"                # a check ran and saw a presentation attack
LIVENESS_ERROR = "error"              # a check ran and could not decide
LIVENESS_STATES = (
    LIVENESS_UNAVAILABLE, LIVENESS_PASS, LIVENESS_FAIL, LIVENESS_ERROR,
)
#
# NONE of these values are measured. Every threshold above this point cites the
# eval run that produced it; these are engineering defaults chosen from first
# principles, and they are marked as such rather than dressed up. eval/liveness/
# exists to replace the guesses with numbers once a test set has been captured.

# How long one tracked face's verdict is reused before it is re-checked.
#
# This is the cost dial and the security dial at once, and they pull in opposite
# directions. Long TTL: fewer API calls, but a spoof swapped in after a live
# check inherits the pass for up to this long. Short TTL: the reverse.
#
# 300 s is chosen for the attendance use case rather than for security in
# general: a person is marked present once per day, so the check that matters is
# the FIRST one, and re-checks exist to catch a substitution during a long
# session. At 300 s a 30-person hour-long class costs 30 x (1 + 60/5) = 390
# calls. Lower it if faces are marked more often than once a session.
LIVENESS_TTL_S = 300.0

# Two detections are the same tracked face when their embeddings are at least
# this similar. Higher than UNKNOWN_DEDUP (0.35) on purpose: that value only has
# to avoid merging two strangers into one queue card, whereas this one decides
# whether a face inherits somebody else's liveness verdict. Getting it wrong in
# the permissive direction lets an attacker inherit a real person's pass, so it
# sits near the measured genuine-pair mean (0.616) rather than near the impostor
# range (0.011 +/- 0.069).
LIVENESS_TRACK_SIM = 0.55

# How long a tracked face is remembered after it was last seen. Covers someone
# turning away and back without paying for a fresh check, while bounding the
# window in which a substitution inherits a verdict.
LIVENESS_TRACK_MEMORY_S = 60.0

# Face crop sent to the model. The margin keeps context AROUND the detection —
# a tight crop would cut off the phone bezel or paper edge that gives an attack
# away, which is the single most useful cue in the frame.
LIVENESS_CROP_MARGIN = 0.35

# Longest side of the crop, in pixels, after downscaling. Drives both cost and
# capability: Anthropic bills roughly (w x h) / 750 tokens, so 512 is about 350
# image tokens. Below ~320 the fine texture cues (moire, print dither, skin
# micro-texture) stop being resolvable and the check degrades to guessing at
# geometry.
LIVENESS_CROP_MAX_PX = 512

# JPEG quality for that crop. 85 rather than 95 because the difference is
# invisible to the model and visible on the wire; rather than 70 because
# compression artefacts are themselves one of the textures being judged, and
# over-compressing manufactures the evidence.
LIVENESS_JPEG_QUALITY = 85

# Response cap. The contract is four short fields; anything longer is a model
# ignoring the schema, and paying for it does not help.
LIVENESS_MAX_TOKENS = 300

# Per-call network timeout. Generous enough for a cold vision model on Ollama,
# short enough that a hung provider does not pin a worker. A timeout fails
# closed, so this bounds how long a genuine person waits to be marked, not how
# long an attacker gets through.
LIVENESS_TIMEOUT_S = 20.0

# Bound on the work queue between the camera thread and the liveness worker.
# Small on purpose: if checks cannot keep up, the useful thing to drop is the
# oldest request, because by the time a backlog clears the face it describes has
# usually left the frame. A large queue would spend money on stale crops.
LIVENESS_QUEUE_MAX = 8

# Flagged-attempt cards for the operator, mirroring the unknown-face queue.
FLAGGED_TTL_S = 300.0
MAX_FLAGGED = 25

# Annotation colour for a face that failed liveness, BGR. Magenta, deliberately
# unlike COLOR_UNCERTAIN (orange) and COLOR_UNKNOWN (red): a spoof is not a weak
# match and not a stranger, it is an attempt, and the feed has to say so.
COLOR_SPOOF = (255, 0, 255)

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
