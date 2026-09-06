"""
One-off migration: face_db.pkl -> person + face_template.

Run once, after the schema exists:

    alembic upgrade head
    py backend/migrate_pickle.py            # --dry-run to preview

**The pickle is left on disk, untouched.** Nothing here writes to it and
nothing deletes it; it is the backup. After this runs, no code path in the
application reads it again.

Why this is not an Alembic revision
-----------------------------------
It needs numpy, and it needs a file that will not exist on every machine that
runs the migrations. A data step that fails for either reason inside a schema
revision would block the schema upgrade itself, which is a much worse failure
than "the import has not been run yet". Schema and data move separately.

Why the unpickler is restricted
-------------------------------
The whole point of this migration is that `pickle.load` on a file path is
arbitrary code execution — a pickle can name any importable callable and have
it invoked with attacker-chosen arguments during load. That is not acceptable
in a system whose job is deciding who gets marked present.

The one script that still has to read the old format therefore does not use
`pickle.load`. It uses an `Unpickler` whose `find_class` refuses everything
outside a small allowlist of numpy's array reconstructors and a few builtin
container types. A hostile pickle hits `UnpicklingError` at the offending
opcode instead of executing anything. Deserialising untrusted data safely is
the exception being made here, and it is made narrowly and once.
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db  # noqa: E402
import repo  # noqa: E402
from config import DB_PATH  # noqa: E402
from models import EMBEDDING_DIM, SOURCE_ENROLMENT  # noqa: E402

# Exactly what a `{"embeddings": [np.ndarray, ...], "names": [str, ...]}` pickle
# needs in order to reconstruct, and nothing else. numpy 2.x moved the
# reconstructor from `numpy.core` to `numpy._core`; both spellings appear in
# files written by different versions, so both are listed.
_ALLOWED = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    # A contiguous array pickles as `_frombuffer(bytes, dtype, shape, order)`,
    # which builds an ndarray over a bytes object and does nothing else. This is
    # what the repo's own face_db.pkl uses — confirmed by reading its opcodes
    # with `pickletools.genops`, which parses without executing.
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    # The general reconstruction path, for pickles written by other numpy
    # versions or from non-contiguous arrays.
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("builtins", "list"),
    ("builtins", "dict"),
    ("builtins", "str"),
    ("builtins", "bytes"),
    ("builtins", "int"),
    ("builtins", "float"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """An unpickler that can rebuild a numpy gallery and nothing else."""

    def __init__(self, file, source: str = "<pickle>"):
        super().__init__(file)
        self._source = source

    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"refusing to load {module}.{name} from {self._source!r}. Only "
            "numpy array reconstruction is permitted here — a pickle naming "
            "anything else is either corrupt or hostile."
        )


def load_legacy_gallery(path: str) -> tuple[list, list[str]]:
    """Read the legacy pickle through the restricted unpickler."""
    with open(path, "rb") as handle:
        data = RestrictedUnpickler(io.BytesIO(handle.read()), source=path).load()

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dict, got {type(data).__name__}")

    embeddings = list(data.get("embeddings", []))
    names = list(data.get("names", []))
    if len(embeddings) != len(names):
        raise ValueError(
            f"{path}: {len(embeddings)} embeddings but {len(names)} names — "
            "the two parallel lists have drifted apart and the file cannot be "
            "read unambiguously."
        )
    return embeddings, names


def migrate(path: str = DB_PATH, dry_run: bool = False) -> int:
    """Import every legacy entry. Idempotent: re-running adds nothing.

    Returns a process exit code.
    """
    import numpy as np
    from sqlalchemy import select

    from models import FaceTemplate, Person

    if not os.path.exists(path):
        print(f"No legacy gallery at {path} — nothing to migrate.")
        return 0

    try:
        embeddings, names = load_legacy_gallery(path)
    except (pickle.UnpicklingError, ValueError, EOFError) as e:
        print(f"FAILED to read {path}: {e}", file=sys.stderr)
        return 1

    print(f"Read {len(names)} entries from {path}")
    if not names:
        return 0

    imported = skipped = 0
    with db.session_scope() as session:
        for raw, name in zip(embeddings, names):
            name = str(name).strip()
            if not name:
                print("  skip: entry with an empty name")
                skipped += 1
                continue

            vector = np.asarray(raw, dtype=np.float32).reshape(-1)
            if vector.shape[0] != EMBEDDING_DIM:
                print(
                    f"  skip {name!r}: {vector.shape[0]}-d embedding, expected "
                    f"{EMBEDDING_DIM}. The old 128-d dlib encodings are not "
                    "compatible with ArcFace and must not be imported.",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            blob = repo.embedding_to_bytes(vector)

            # Idempotency: the same person with the same normalised embedding
            # is already here, so a second run is a no-op rather than a second
            # template (which would silently invalidate the threshold).
            duplicate = session.execute(
                select(FaceTemplate.id)
                .join(Person, Person.id == FaceTemplate.person_id)
                .where(Person.name == name, FaceTemplate.embedding == blob)
            ).first()
            if duplicate is not None:
                print(f"  skip {name!r}: already imported")
                skipped += 1
                continue

            if dry_run:
                print(f"  would import {name!r}")
                imported += 1
                continue

            result = repo.enrol_face(
                session,
                name=name,
                embedding_bytes=blob,
                dim=EMBEDDING_DIM,
                source=SOURCE_ENROLMENT,
                quality_score=None,  # the legacy format recorded none
            )
            note = "" if result.template_count == 1 else (
                f"  <-- now has {result.template_count} templates"
            )
            print(f"  imported {name!r} as person {result.person_id}{note}")
            imported += 1

        if dry_run:
            session.rollback()

    print(
        f"\n{'Would import' if dry_run else 'Imported'}: {imported}   "
        f"Skipped: {skipped}"
    )
    print(f"{path} left on disk untouched as a backup.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import the legacy face_db.pkl into the database. "
        "Idempotent; leaves the pickle in place."
    )
    parser.add_argument("--path", default=DB_PATH, help=f"default: {DB_PATH}")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would happen, write nothing."
    )
    args = parser.parse_args(argv)
    return migrate(args.path, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
