"""
Bulk-enrol people into the face database from a folder of photos.

Scans a dataset laid out as one sub-folder per person:

    dataset/
      Alice/  img1.jpg  img2.jpg ...
      Bob/    img1.png  ...

and writes 512-d ArcFace embeddings to the **database** (person +
face_template), through the same `repo.enrol_face` the live registration path
uses. It no longer writes `face_db.pkl`: nothing reads that file any more, so a
script that produced it looked like it worked and silently had no effect on the
running app.

Usage:
    py backend/encode_faces.py path/to/dataset --dry-run
    py backend/encode_faces.py path/to/dataset --one-per-person
    py backend/encode_faces.py path/to/dataset --allow-multi-template

THE THRESHOLD PROBLEM THIS SCRIPT CREATES
-----------------------------------------
A dataset normally holds several photos per person, and enrolling all of them
gives that person several face templates. That invalidates the deployed match
threshold.

`STRONG_MATCH = 0.370` was derived at **exactly one template per person**
(eval run 2026-09-05, LFW, gallery 50). Matching takes a max over every ROW of
the gallery matrix, and every row is one more chance for a stranger to score
high — so open-set false-accept rate scales with the TOTAL number of templates,
not the number of people. The evaluation measured this directly: holding FAR at
the 0.1% budget, **N=5 templates per person needed 0.412 where N=1 needed
0.370** (eval/README.md, "What single-image enrolment costs").

So this script refuses by default to create a multi-template person. Either:

  * `--one-per-person` — enrol only the best photo of each person (highest
    detector score). Keeps N=1, stays inside the calibrated threshold.
  * `--allow-multi-template` — enrol everything, accepting that the threshold
    is no longer calibrated until it is re-derived in eval/.

With `--allow-multi-template`, the consequences show up on their own, because
this script writes through the same repository layer as the interactive path:
`/api/status` starts returning a `calibration_warning` that the dashboard
renders as a persistent banner, and every attendance row matched against a
multi-template person is flagged `multi_template_match` (carried into the CSV
export). Nothing extra is needed here to make that happen — it falls out of the
template counts being correct in the database.

Note: images must be reasonably sized, real face photos. The original
`known_faces/` folder in this repo contains all-black files and will yield
nothing — drop proper photos in a new folder instead.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from typing import Iterator, NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DET_SIZE, MODEL_NAME, PROJECT_ROOT  # noqa: E402
from models import (  # noqa: E402
    ACTION_ENROL,
    EMBEDDING_DIM,
    SOURCE_ENROLMENT,
)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EncodedFace(NamedTuple):
    """One detected face, ready to be enrolled."""

    person: str
    path: str          # the source photo, kept for the audit trail
    embedding: object  # (512,) float32 ndarray, L2-normalised by InsightFace
    det_score: float   # detector confidence; stored as face_template.quality_score


class PersonPlan(NamedTuple):
    """What would happen to one person."""

    name: str
    existing: int             # templates already in the database
    to_enrol: list[EncodedFace]
    duplicates: int           # photos whose embedding is already enrolled

    @property
    def final_count(self) -> int:
        return self.existing + len(self.to_enrol)


# --------------------------------------------------------------- dataset scan
def scan_dataset(dataset_path: str) -> list[tuple[str, list[str]]]:
    """(person, [photo paths]) for each sub-folder. No image decoding.

    Separated from embedding so the layout can be validated — and the CLI can
    fail with a useful message — before the model is loaded, which takes
    seconds and downloads ~15 MB on a cold machine.
    """
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(dataset_path)

    out = []
    for person in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        photos = sorted(
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in EXTS
        )
        if photos:
            out.append((person.strip(), photos))
    return out


def embed_dataset(
    dataset: list[tuple[str, list[str]]], verbose: bool = True
) -> tuple[list[EncodedFace], int]:
    """Detect and embed every photo. Returns (encoded, skipped).

    Heavy imports stay inside the function, per the convention in the rest of
    the backend: `--help` and a bad dataset path must not require the
    recognition stack to be installed.
    """
    import cv2
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    encoded: list[EncodedFace] = []
    skipped = 0

    for person, photos in dataset:
        count = 0
        for path in photos:
            img = cv2.imread(path)
            if img is None:
                skipped += 1
                continue
            faces = app.get(img)
            if not faces:
                skipped += 1
                continue
            # Keep the most confident face in the photo. (The eval harness uses
            # the *centred* face instead, because LFW labels the centred
            # subject — but that is a fix to the harness's ground truth, not to
            # this rule. Here the operator chose the folder, so the clearest
            # face is the right one. See eval/README.md, --face-select.)
            face = max(faces, key=lambda f: float(f.det_score))
            encoded.append(
                EncodedFace(
                    person=person,
                    path=path,
                    embedding=face.normed_embedding,
                    det_score=float(face.det_score),
                )
            )
            count += 1
        if verbose:
            print(f"  {person}: {count} of {len(photos)} photos encoded")

    return encoded, skipped


# ------------------------------------------------------------------- planning
def plan_enrolment(session, encoded: list[EncodedFace], one_per_person: bool):
    """Work out what would be written, without writing it.

    Idempotent by construction: a photo whose exact normalised embedding is
    already stored for that person is counted as a duplicate and skipped.
    Without this, re-running the script would silently double everyone's
    template count — which is the same class of quiet wrongness as writing a
    pickle nothing reads.
    """
    from sqlalchemy import select

    import repo
    from models import FaceTemplate, Person

    by_person: dict[str, list[EncodedFace]] = {}
    for face in encoded:
        by_person.setdefault(face.person, []).append(face)

    plans = []
    for name in sorted(by_person):
        faces = by_person[name]
        if one_per_person:
            # The clearest photo is the one most like a cooperative enrolment
            # frame, which is the distribution the threshold was measured on.
            faces = [max(faces, key=lambda f: f.det_score)]

        person = session.execute(
            select(Person).where(Person.name == name)
        ).scalar_one_or_none()

        existing = 0
        stored: set[bytes] = set()
        if person is not None:
            rows = session.execute(
                select(FaceTemplate.embedding).where(
                    FaceTemplate.person_id == person.id
                )
            ).scalars().all()
            existing = len(rows)
            stored = set(rows)

        to_enrol, duplicates = [], 0
        for face in faces:
            blob = repo.embedding_to_bytes(face.embedding)
            if blob in stored:
                duplicates += 1
                continue
            stored.add(blob)   # also dedups identical photos within one run
            to_enrol.append(face)

        plans.append(PersonPlan(name, existing, to_enrol, duplicates))

    return plans


def multi_template_people(plans: list[PersonPlan]) -> list[PersonPlan]:
    """Those who would end up with more than one template."""
    return [p for p in plans if p.final_count > 1]


REFUSAL = """\
REFUSED: this would give {count} {noun} more than one face template.

{listing}

STRONG_MATCH = 0.370 was derived at exactly ONE template per person (eval run
2026-09-05, LFW, gallery 50). Matching is a max over every row of the gallery
matrix, so open-set false-accept rate scales with the TOTAL number of enrolled
templates, not the number of people. The evaluation measured this directly:
holding FAR at the 0.1% budget, N=5 templates per person needed a threshold of
0.412 where N=1 needed 0.370 (eval/README.md, "What single-image enrolment
costs"). At N=5 the stricter threshold ate the benefit entirely — degraded
false-reject rate was worse at N=5 than at N=3.

Enrolling these anyway is a real option, but it is a decision, not a default.
Choose one:

  --one-per-person        Enrol only the best photo of each person (highest
                          detector score). Keeps one template each, so the
                          deployed threshold stays calibrated.

  --allow-multi-template  Enrol everything. The threshold is then no longer
                          calibrated: the dashboard shows a persistent warning,
                          and every attendance row matched against these people
                          is flagged multi_template_match. Re-derive the
                          threshold with eval/evaluate.py before trusting those
                          records.
"""


def format_refusal(offenders: list[PersonPlan]) -> str:
    shown = offenders[:10]
    listing = "\n".join(
        f"    {p.name}: {p.existing} existing + {len(p.to_enrol)} new "
        f"= {p.final_count} templates"
        for p in shown
    )
    if len(offenders) > len(shown):
        listing += f"\n    ... and {len(offenders) - len(shown)} more"
    return REFUSAL.format(
        count=len(offenders),
        noun="person" if len(offenders) == 1 else "people",
        listing=listing,
    )


# -------------------------------------------------------------------- writing
def apply_plan(session, plans: list[PersonPlan], actor: str) -> int:
    """Enrol everything in `plans`. Returns the number of templates written.

    Goes through `repo.enrol_face`, the same call `/api/register` makes, so a
    bulk enrolment and an interactive one produce identical rows — and an
    existing name attaches templates to that person rather than creating a
    duplicate identity.

    Writes one audit row per template, matching the single-enrolment path, with
    the source photo as provenance. `ip="cli"` follows the convention already
    set by backend/cli.py: this did not arrive over the network.
    """
    import repo

    written = 0
    for plan in plans:
        for face in plan.to_enrol:
            result = repo.enrol_face(
                session,
                name=plan.name,
                embedding_bytes=repo.embedding_to_bytes(face.embedding),
                dim=EMBEDDING_DIM,
                source=SOURCE_ENROLMENT,
                # The detector score is exactly what this column is for. The
                # interactive path has none to give and stores NULL.
                quality_score=face.det_score,
            )
            repo.audit(
                session,
                action=ACTION_ENROL,
                actor_user_id=None,          # a CLI run has no logged-in user
                actor_username=actor,
                target=(
                    f"person:{result.person_id} ({result.person_name}) "
                    f"template:{result.template_id} "
                    f"count:{result.template_count} "
                    f"source:{os.path.relpath(face.path, PROJECT_ROOT)} "
                    f"via:encode_faces"
                ),
                ip="cli",
            )
            written += 1
    return written


# ------------------------------------------------------------------------ CLI
def report(plans: list[PersonPlan], skipped: int, dry_run: bool) -> None:
    verb = "Would enrol" if dry_run else "Enrolled"
    total = sum(len(p.to_enrol) for p in plans)
    duplicates = sum(p.duplicates for p in plans)

    print()
    print(f"{'PERSON':<28} {'EXISTING':>8} {'NEW':>5} {'DUP':>5} {'FINAL':>6}")
    for p in plans:
        print(
            f"{p.name:<28} {p.existing:>8} {len(p.to_enrol):>5} "
            f"{p.duplicates:>5} {p.final_count:>6}"
        )
    print()
    print(f"{verb}: {total} template(s) across {len(plans)} person(s)")
    if duplicates:
        print(f"Skipped {duplicates} photo(s) already enrolled (re-run is a no-op)")
    if skipped:
        print(f"Skipped {skipped} image(s) with no detectable face")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-enrol people into the face database from a folder of photos, "
            "one sub-folder per person. Writes to the database, not to "
            "face_db.pkl."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=os.path.join(PROJECT_ROOT, "dataset"),
        help="Folder holding one sub-folder per person (default: ./dataset)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report exactly what would be enrolled, and write nothing.",
    )
    parser.add_argument(
        "--one-per-person",
        action="store_true",
        help=(
            "Enrol only the highest-scoring photo of each person, keeping one "
            "template each so STRONG_MATCH stays calibrated."
        ),
    )
    parser.add_argument(
        "--allow-multi-template",
        action="store_true",
        help=(
            "Permit people to end up with several templates. This invalidates "
            "STRONG_MATCH = 0.370 until it is re-derived in eval/."
        ),
    )
    parser.add_argument(
        "--actor",
        default=None,
        help="Name recorded in the audit log (default: the OS user).",
    )
    args = parser.parse_args(argv)

    if args.one_per_person and args.allow_multi_template:
        print(
            "--one-per-person and --allow-multi-template contradict each "
            "other. Pick one.",
            file=sys.stderr,
        )
        return 2

    # Validate the dataset before loading the model: preparing InsightFace takes
    # seconds and downloads ~15 MB on a cold machine.
    try:
        dataset = scan_dataset(args.dataset)
    except NotADirectoryError:
        print(f"Dataset folder not found: {args.dataset}", file=sys.stderr)
        print(
            "Expected one sub-folder per person, e.g. dataset/Alice/*.jpg",
            file=sys.stderr,
        )
        return 1

    if not dataset:
        print(
            f"No usable photos under {args.dataset}. Expected one sub-folder "
            f"per person containing {', '.join(sorted(EXTS))} files.",
            file=sys.stderr,
        )
        return 1

    print(f"Scanning {args.dataset} — {len(dataset)} person folder(s)")
    try:
        encoded, skipped = embed_dataset(dataset)
    except ImportError as e:
        print(
            f"The recognition stack is not installed ({e}). Install it with:\n"
            "    py -m pip install -r backend/requirements.txt",
            file=sys.stderr,
        )
        return 1

    if not encoded:
        print(
            "\nNo faces detected in any photo. If this is the repo's "
            "known_faces/ folder, note that those images are all black — see "
            "CLAUDE.md, Known issues.",
            file=sys.stderr,
        )
        return 1

    import db

    with db.session_scope() as session:
        plans = plan_enrolment(session, encoded, one_per_person=args.one_per_person)

        offenders = multi_template_people(plans)
        if offenders and not args.allow_multi_template:
            report(plans, skipped, dry_run=True)
            print()
            print(format_refusal(offenders), file=sys.stderr)
            session.rollback()
            return 2

        if args.dry_run:
            report(plans, skipped, dry_run=True)
            if offenders:
                print(
                    f"\nNOTE: --allow-multi-template is set. {len(offenders)} "
                    "person(s) would hold several templates, which leaves "
                    "STRONG_MATCH = 0.370 uncalibrated."
                )
            session.rollback()
            return 0

        actor = args.actor or f"{getpass.getuser()} (encode_faces)"
        apply_plan(session, plans, actor=actor)

    report(plans, skipped, dry_run=False)

    if offenders:
        print()
        print(
            "WARNING: the match threshold is no longer calibrated.\n"
            f"  {len(offenders)} person(s) now hold more than one face "
            "template. STRONG_MATCH = 0.370 was measured at one template each\n"
            "  (N=5 needed 0.412). The dashboard will show a persistent "
            "warning, and attendance rows matched against these people are\n"
            "  flagged multi_template_match. Re-derive the threshold:\n"
            "      py eval/evaluate.py",
            file=sys.stderr,
        )

    print("\nRestart the backend to load the new gallery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
