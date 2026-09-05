"""
Accuracy evaluation harness for the face-recognition engine.

Measures what the running app's matcher actually does, using the *same*
InsightFace model and detector settings as ``backend/engine.py`` (the constants
are imported from there, never duplicated), so the numbers transfer directly to
the deployed thresholds.

Two protocols are run, and they answer different questions:

  * **Verification** (pairwise, "are these two faces the same person?") —
    ROC + AUC, EER, FAR/FRR vs. threshold, score distributions. This is the
    standard way to summarise *matcher quality*, and it is what you compare
    across model changes. It does **not** set the app's threshold.
  * **Open-set identification** (1:N, "who is this, or nobody?") — exactly what
    ``engine._recognize()`` does: take the max similarity over the whole
    gallery, take the argmax, and accept only if it clears a threshold. False
    accepts get *more likely as the gallery grows*, because every extra
    enrolled person is another chance for a stranger to score high. So the
    operating threshold is picked from these curves, at a stated gallery size.

Produces:
  * verification ROC + AUC, with the EER marked
  * FAR / FRR against cosine threshold (FAR estimated on the FULL impostor
    pool, not the balanced subsample, so the low-FAR tail is resolved)
  * open-set FAR / FRR / misidentification rate vs. threshold, swept over
    gallery size, with a recommended operating threshold
  * rank-1 / rank-5 identification accuracy (+ CMC curve), with Wilson 95%
    confidence intervals and a minimum-detectable-difference analysis
  * genuine vs. impostor score distributions
  * top-2 margin distribution for correct vs. incorrect rank-1 predictions
  * image grids of every misclassified probe, under ``results/failures/``

Run ``python eval/evaluate.py --help`` for options. See eval/README.md.

This script only measures. It does not modify anything under backend/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

# Embeddings are L2-normalised, so cosine similarity lives in [-1, 1].
SCORE_MIN, SCORE_MAX = -1.0, 1.0


# --------------------------------------------------------------------- backend
def load_backend_constants() -> dict[str, Any]:
    """Import the live matcher settings from backend/engine.py.

    engine.py keeps its heavy dependencies (cv2, insightface, numpy, pandas)
    behind lazy function-level imports, so importing it here is cheap and has no
    side effects beyond constructing one idle AttendanceEngine instance.
    """
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    import engine as backend_engine  # noqa: PLC0415  (deliberate late import)

    return {
        "model_name": backend_engine.MODEL_NAME,
        "det_size": tuple(backend_engine.DET_SIZE),
        "strong_match": float(backend_engine.STRONG_MATCH),
        "weak_match": float(backend_engine.WEAK_MATCH),
        "source": str((BACKEND_DIR / "engine.py").relative_to(REPO_ROOT)),
    }


# ------------------------------------------------------------------------ data
def to_uint8_bgr(images: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Convert sklearn's LFW float images to uint8 BGR, and prove it worked.

    ``fetch_lfw_people`` returns float32 pixels already divided by 255, i.e. in
    [0, 1]. Calling ``.astype('uint8')`` on that truncates every pixel to 0 —
    this is the exact bug recorded in legacy/README.md that produced 3023
    all-black images in the old ``known_faces/`` folder. The multiply by 255
    below is therefore mandatory, and we assert loudly that it took effect.

    Converted in chunks: at full resolution the float32 source is already
    ~2.3 GB, and a whole-array ``arr * 255.0`` would transiently double that.
    """
    arr = np.asarray(images)

    if arr.dtype == np.uint8:
        scaled = arr
    else:
        peak = float(np.nanmax(arr[:chunk])) if arr.size else 0.0
        scaled = np.empty(arr.shape, dtype=np.uint8)
        for start in range(0, len(arr), chunk):
            block = arr[start:start + chunk]
            if peak <= 1.5:
                # The [0, 1] case: rescale BEFORE casting. Never reorder these.
                scaled[start:start + chunk] = (block * 255.0).astype(np.uint8)
            else:
                scaled[start:start + chunk] = np.clip(block, 0, 255).astype(np.uint8)

    # --- fail loudly if we produced black images -------------------------
    if not scaled.any():
        raise SystemExit(
            "FATAL: every pixel is zero after uint8 conversion. This is the "
            "legacy .astype('uint8')-without-x255 bug (see legacy/README.md). "
            "Refusing to evaluate on blank images."
        )
    per_image_max = scaled.reshape(len(scaled), -1).max(axis=1)
    n_black = int((per_image_max == 0).sum())
    if n_black:
        frac = n_black / len(scaled)
        if frac > 0.01:
            raise SystemExit(
                f"FATAL: {n_black}/{len(scaled)} ({frac:.1%}) images are "
                "entirely black after conversion — dataset is unusable."
            )
        print(f"  warning: {n_black} individually black image(s) present")
    mean = float(scaled.mean())
    if mean < 1.0:
        raise SystemExit(
            f"FATAL: mean pixel value is {mean:.4f}, effectively black. "
            "The x255 rescale did not take effect."
        )
    print(f"  uint8 conversion OK: mean={mean:.1f} max={int(scaled.max())}")

    if scaled.ndim == 4 and scaled.shape[-1] == 3:
        # insightface/cv2 expect BGR; LFW is RGB. backend/engine.py feeds
        # cv2 BGR frames, so match that convention.
        scaled = scaled[..., ::-1]
    return np.ascontiguousarray(scaled)


# Pose codes for datasets that carry one. LFW does not; CFP does.
POSE_FRONTAL, POSE_PROFILE = 0, 1

CFP_DOWNLOAD_HELP = """\
CFP-FP was not found.

Get it (82 MB, no registration or licence click-through):

    curl -L -o cfp-dataset.zip http://cfpw.io/cfp-dataset.zip
    unzip cfp-dataset.zip -d <somewhere>

then point the harness at the extracted folder:

    python eval/evaluate.py --dataset cfp-fp --cfp-root <somewhere>/cfp-dataset

The layout it expects (the zip's own layout - do not rearrange it):

    <cfp-root>/Data/Images/001/frontal/01.jpg ... 10.jpg
    <cfp-root>/Data/Images/001/profile/01.jpg ... 04.jpg
    ...          .../500/...
    <cfp-root>/Data/list_name.txt          (optional; supplies real names)

--cfp-root also accepts the Data/ or Data/Images/ directory directly, or any
parent containing them. 500 identities x (10 frontal + 4 profile) = 7000 images.

Cite: S. Sengupta et al., "Frontal to Profile Face Verification in the Wild",
IEEE WACV 2016.  http://www.cfpw.io/
"""


def _subsample_identities(images: Any, labels: np.ndarray, names: list[str],
                          args: argparse.Namespace,
                          pose: np.ndarray | None = None
                          ) -> tuple[Any, np.ndarray, list[str], np.ndarray | None]:
    """Apply --max-identities, shared by every dataset loader."""
    if not args.max_identities or args.max_identities >= len(names):
        return images, labels, names, pose
    rng = np.random.default_rng(args.seed)
    keep = np.sort(rng.choice(len(names), size=args.max_identities,
                              replace=False))
    mask = np.isin(labels, keep)
    remap = {old: new for new, old in enumerate(keep)}
    sel = np.flatnonzero(mask)
    images = (images[mask] if isinstance(images, np.ndarray)
              else [images[i] for i in sel])
    labels = np.array([remap[int(v)] for v in labels[mask]], dtype=np.int64)
    names = [names[i] for i in keep]
    if pose is not None:
        pose = pose[mask]
    print(f"  subsampled to {len(names)} identities / {len(labels)} images "
          f"(--max-identities, seed={args.seed})")
    return images, labels, names, pose


def _content_hash(paths: list[Path], root: Path) -> str:
    """Fingerprint a directory dataset by its file list and sizes.

    Cheap (no pixel reads) but enough that a different or partial copy of the
    dataset produces a different embedding cache key instead of silently
    reusing embeddings computed from other images.
    """
    digest = hashlib.sha256()
    for p in sorted(paths):
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        digest.update(f"{p.relative_to(root).as_posix()}:{size}\n".encode())
    return digest.hexdigest()[:16]


def load_cfp_fp(args: argparse.Namespace
                ) -> tuple[list[np.ndarray], np.ndarray, list[str], dict[str, Any]]:
    """Load the CFP dataset from disk, tagging each image frontal or profile.

    CFP is the benchmark LFW is not: 500 identities photographed both frontally
    and in profile, so it stresses exactly the axis a single enrolment frame
    fails on. ArcFace is nowhere near saturated on it, which is the whole
    reason for adding it.

    Unlike LFW, this is a directory of real JPEGs at varying sizes, so the
    return is a *list* of images rather than one stacked array. Everything
    downstream indexes rather than reshapes, so a list is enough.
    """
    import cv2  # noqa: PLC0415

    if not args.cfp_root:
        raise SystemExit("FATAL: --dataset cfp-fp needs --cfp-root.\n\n"
                         + CFP_DOWNLOAD_HELP)
    root = Path(args.cfp_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"FATAL: --cfp-root {root} is not a directory.\n\n"
                         + CFP_DOWNLOAD_HELP)

    # Accept the zip root, Data/, Data/Images/, or any parent of them.
    images_dir = None
    for candidate in (root / "Data" / "Images", root / "Images", root,
                      root / "cfp-dataset" / "Data" / "Images"):
        if candidate.is_dir() and any(
                (d / "frontal").is_dir() for d in candidate.iterdir()
                if d.is_dir()):
            images_dir = candidate
            break
    if images_dir is None:
        for found in root.rglob("frontal"):
            if found.is_dir() and found.parent.parent.is_dir():
                images_dir = found.parent.parent
                break
    if images_dir is None:
        raise SystemExit(
            f"FATAL: no CFP image tree under {root} (looked for "
            "*/frontal/*.jpg).\n\n" + CFP_DOWNLOAD_HELP)

    print(f"Loading CFP-FP from {images_dir} ...")
    ident_dirs = sorted((d for d in images_dir.iterdir() if d.is_dir()),
                        key=lambda d: d.name)
    if not ident_dirs:
        raise SystemExit(f"FATAL: {images_dir} has no identity folders.\n\n"
                         + CFP_DOWNLOAD_HELP)

    # Optional real names; the folders are just zero-padded indices.
    name_file = next((p for p in (images_dir.parent / "list_name.txt",
                                  images_dir / "list_name.txt",
                                  root / "Data" / "list_name.txt")
                      if p.is_file()), None)
    listed = (name_file.read_text(encoding="utf-8", errors="replace").split()
              if name_file else [])

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images: list[np.ndarray] = []
    labels: list[int] = []
    poses: list[int] = []
    names: list[str] = []
    all_paths: list[Path] = []
    unreadable = 0

    for ident, folder in enumerate(ident_dirs):
        try:
            index = int(folder.name)
        except ValueError:
            index = ident + 1
        names.append(listed[index - 1] if 0 < index <= len(listed)
                     else f"cfp_{folder.name}")
        for pose_name, pose_code in (("frontal", POSE_FRONTAL),
                                     ("profile", POSE_PROFILE)):
            pose_dir = folder / pose_name
            if not pose_dir.is_dir():
                continue
            for path in sorted(p for p in pose_dir.iterdir()
                               if p.suffix.lower() in exts):
                all_paths.append(path)
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)   # BGR uint8
                if img is None:
                    unreadable += 1
                    continue
                if args.cfp_max_side and max(img.shape[:2]) > args.cfp_max_side:
                    scale = args.cfp_max_side / max(img.shape[:2])
                    img = cv2.resize(
                        img, (int(round(img.shape[1] * scale)),
                              int(round(img.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA)
                images.append(np.ascontiguousarray(img))
                labels.append(ident)
                poses.append(pose_code)

    if not images:
        raise SystemExit(f"FATAL: read no images under {images_dir}.\n\n"
                         + CFP_DOWNLOAD_HELP)
    if unreadable:
        print(f"  warning: {unreadable} file(s) could not be decoded")

    labels_arr = np.asarray(labels, dtype=np.int64)
    pose_arr = np.asarray(poses, dtype=np.int8)
    n_front = int((pose_arr == POSE_FRONTAL).sum())
    n_prof = int((pose_arr == POSE_PROFILE).sum())

    # The same blank-image guard LFW gets: a silently-black dataset would
    # produce plausible-looking, meaningless metrics.
    sample = images[:256]
    mean = float(np.mean([float(im.mean()) for im in sample]))
    if mean < 1.0:
        raise SystemExit(f"FATAL: mean pixel value is {mean:.4f} over the "
                         "first images - the dataset is effectively black.")

    print(f"  {len(images)} images, {len(names)} identities "
          f"({n_front} frontal / {n_prof} profile), mean pixel {mean:.1f}")
    if n_prof == 0:
        raise SystemExit("FATAL: no profile images found; this is not CFP-FP.")

    images, labels_arr, names, pose_arr = _subsample_identities(
        images, labels_arr, names, args, pose_arr)

    meta = {
        "name": "cfp-fp",
        "source": str(images_dir),
        "pose": pose_arr,
        "content_hash": _content_hash(all_paths, images_dir),
        "images_frontal": n_front,
        "images_profile": n_prof,
        "reference": ("S. Sengupta et al., Frontal to Profile Face "
                      "Verification in the Wild, IEEE WACV 2016"),
    }
    return images, labels_arr, names, meta


def load_dataset(args: argparse.Namespace
                 ) -> tuple[Any, np.ndarray, list[str], dict[str, Any]]:
    """Dispatch to the requested benchmark loader."""
    if args.dataset == "cfp-fp":
        return load_cfp_fp(args)
    return load_lfw(args)


def load_lfw(args: argparse.Namespace
             ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Fetch LFW and return (images_bgr_uint8, labels, identity_names, meta)."""
    from sklearn.datasets import fetch_lfw_people  # noqa: PLC0415

    # sklearn's default slice_ crops LFW to a tight 125x94 face with no
    # surrounding context, and RetinaFace does not fire on a face that fills
    # the whole frame: measured 0/25 detections on the tight crop vs 25/25 on
    # the full 250x250 image. 'full' is therefore the default. See eval/README.
    slice_ = ((slice(0, 250), slice(0, 250)) if args.slice == "full" else None)

    print(f"Loading LFW (min_faces_per_person={args.min_faces}, "
          f"resize={args.resize}, color=True, slice={args.slice}) ...")
    kwargs: dict[str, Any] = {
        "min_faces_per_person": args.min_faces,
        "color": True,
        "resize": args.resize,
    }
    if slice_ is not None:
        kwargs["slice_"] = slice_
    lfw = fetch_lfw_people(**kwargs)
    images = to_uint8_bgr(lfw.images)
    labels = np.asarray(lfw.target, dtype=np.int64)
    names = [str(n) for n in lfw.target_names]
    del lfw  # free the float32 copy before embedding starts
    print(f"  {len(images)} images, {len(names)} identities, "
          f"image shape {images.shape[1:]}")

    images, labels, names, _ = _subsample_identities(images, labels, names, args)

    if args.max_images_per_identity:
        rng = np.random.default_rng(args.seed + 1)
        keep_idx = []
        for ident in range(len(names)):
            idx = np.flatnonzero(labels == ident)
            if len(idx) > args.max_images_per_identity:
                idx = np.sort(rng.choice(idx, args.max_images_per_identity,
                                         replace=False))
            keep_idx.append(idx)
        sel = np.sort(np.concatenate(keep_idx))
        images, labels = images[sel], labels[sel]
        print(f"  capped to <={args.max_images_per_identity} images/identity "
              f"-> {len(images)} images")

    # LFW has no pose annotation, so the pose-aware split does not apply.
    return images, labels, names, {"name": "lfw", "pose": None,
                                   "source": "sklearn.fetch_lfw_people",
                                   "content_hash": None}


# ------------------------------------------------------------------ embeddings
def cache_key(args: argparse.Namespace, backend: dict[str, Any],
              n_images: int, n_ids: int,
              variant: dict[str, Any] | None = None,
              meta: dict[str, Any] | None = None) -> str:
    """Hash of everything that changes the embeddings (not the split/metrics).

    The payload below is the *original* set of embedding-affecting fields and
    must not be reordered, renamed, or added to: any change to it re-hashes
    every previously cached ``.npz`` into uselessness. New knobs go in
    ``extras``, which is merged into the payload **only when it is non-empty**,
    so a configuration that predates a new knob still hashes to its old digest
    and keeps its cached embeddings. This is why nothing from the open-set
    protocol, the power analysis or the failure grids appears here — none of
    them change a single embedding; they only change how the embeddings are
    split and scored.
    """
    payload = {
        "model": backend["model_name"],
        "det_size": list(backend["det_size"]),
        "dataset": "sklearn.fetch_lfw_people",
        "min_faces": args.min_faces,
        "resize": args.resize,
        "color": True,
        "slice": args.slice,
        "max_identities": args.max_identities,
        "max_images_per_identity": args.max_images_per_identity,
        "subsample_seed": args.seed,
        "n_images": n_images,
        "n_identities": n_ids,
    }

    # Embedding-affecting options added after the first release. Each entry is
    # included only when it deviates from the behaviour that existed when the
    # cache format was frozen, so old digests survive.
    extras: dict[str, Any] = {}
    if getattr(args, "det_score_min", 0.0) > 0.0:
        extras["det_score_min"] = float(args.det_score_min)
    if getattr(args, "dataset", "lfw") != "lfw":
        # Dataset identity, including a fingerprint of the actual files, so a
        # different or partial copy cannot silently reuse another one's
        # embeddings. LFW stays out of `extras` entirely, which is what keeps
        # every previously cached LFW digest valid.
        # Only things that change a pixel belong here: --cfp-protocol picks
        # the split, not the images, so it is deliberately absent.
        extras["dataset"] = {
            "name": args.dataset,
            "content_hash": (meta or {}).get("content_hash"),
            "cfp_max_side": getattr(args, "cfp_max_side", 0),
        }
    if variant:
        # A different rendering of the same source images (currently: the
        # degraded arm). It gets its own cache file; the clean arm keeps its
        # digest because `variant` is None there and `extras` stays empty.
        extras["variant"] = variant
    if extras:
        payload["extras"] = extras

    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def compute_embeddings(images: np.ndarray, backend: dict[str, Any],
                       args: argparse.Namespace, key: str) -> dict[str, Any]:
    """Embed every image once, keeping BOTH face-selection rules.

    Many LFW frames contain more than one person — a head of state at a podium
    with an aide behind them. ``encode_faces.py`` keeps the highest-``det_score``
    face, which is the right rule for the app (it enrols whoever is most
    clearly visible) but the *wrong* rule for an evaluation, because LFW's label
    names the **centred** subject. When the two disagree, the harness would
    otherwise score a photo of Berlusconi against the label "Jose Maria Aznar"
    and call the resulting mismatch a recognition error.

    ``app.get()`` already returns an embedding for every detected face, so
    computing both selections costs nothing beyond the one detection pass. Both
    go in the cache and ``--face-select`` picks between them at scoring time —
    which is why the selection is deliberately *not* part of the cache key: the
    file contents do not depend on it.

    Returns {"best", "centred", "valid", "n_faces", "centred_is_best"}; the
    last three are None when read from a cache written before this change.
    """
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"emb_{key}.npz"

    if cache_file.exists() and not args.no_cache:
        blob = np.load(cache_file)
        print(f"Embeddings loaded from cache: {cache_file.name}")
        if "embeddings_centred" in set(blob.files):
            return {"best": blob["embeddings"], "centred": blob["embeddings_centred"],
                    "valid": blob["valid"], "n_faces": blob["n_faces"],
                    "centred_is_best": blob["centred_is_best"]}
        print("  note: this cache predates centred-face selection, so only "
              "--face-select best-score can be served from it. Re-run with "
              "--no-cache to add the centred embeddings.")
        return {"best": blob["embeddings"], "centred": None,
                "valid": blob["valid"], "n_faces": None, "centred_is_best": None}

    from insightface.app import FaceAnalysis  # noqa: PLC0415

    print(f"Preparing InsightFace '{backend['model_name']}' "
          f"det_size={backend['det_size']} on CPU ...")
    app = FaceAnalysis(name=backend["model_name"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=backend["det_size"])

    n = len(images)
    emb_best = np.zeros((n, 512), dtype=np.float32)
    emb_centred = np.zeros((n, 512), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    n_faces = np.zeros(n, dtype=np.int16)
    centred_is_best = np.ones(n, dtype=bool)
    started = time.time()

    for i, img in enumerate(images):
        faces = app.get(img)
        if args.det_score_min > 0.0:
            faces = [f for f in faces if float(f.det_score) >= args.det_score_min]
        n_faces[i] = len(faces)
        if faces:
            height, width = img.shape[:2]
            cx, cy = width / 2.0, height / 2.0

            def off_centre(f) -> float:
                x1, y1, x2, y2 = (float(v) for v in f.bbox)
                return ((x1 + x2) / 2.0 - cx) ** 2 + ((y1 + y2) / 2.0 - cy) ** 2

            # Same rule as backend/encode_faces.py: keep the best-scoring face.
            best = max(faces, key=lambda f: float(f.det_score))
            # LFW is funneled, so the labelled subject is the centred one.
            centred = min(faces, key=off_centre)
            emb_best[i] = best.normed_embedding
            emb_centred[i] = centred.normed_embedding
            centred_is_best[i] = best is centred
            valid[i] = True
        if (i + 1) % 200 == 0 or i + 1 == n:
            speed = (i + 1) / max(time.time() - started, 1e-9)
            print(f"  embedded {i + 1}/{n}  ({valid[:i + 1].sum()} detected, "
                  f"{speed:.1f} img/s)", flush=True)

    np.savez_compressed(cache_file, embeddings=emb_best,
                        embeddings_centred=emb_centred, valid=valid,
                        n_faces=n_faces, centred_is_best=centred_is_best)
    print(f"  cached -> {cache_file.name} "
          f"({time.time() - started:.1f}s total)")
    return {"best": emb_best, "centred": emb_centred, "valid": valid,
            "n_faces": n_faces, "centred_is_best": centred_is_best}


# ----------------------------------------------------------------- degradation
# Bumped whenever the corruption maths changes, so a stale degraded cache can
# never be mistaken for a current one. It is part of the cache variant.
#   v2: per-kind independent RNG streams (see _kind_rng) + pose_warp added.
DEGRADE_VERSION = 2

# The corruption families, split into two DISJOINT groups. This split is the
# whole point of the degraded arm: it is what makes it a legitimate metric for
# an enrolment-augmentation ablation rather than a circular one.
#
#   augmentable — effects a synthetic enrolment augmentation can plausibly
#                 imitate. Reserved FOR the augmentation. Never evaluate on
#                 these: an augmentation that applies them will trivially
#                 improve against them, and the improvement means nothing.
#   heldout     — effects the augmentation is not allowed to touch. These are
#                 what you EVALUATE on. Improving here is evidence that the
#                 augmentation bought general robustness rather than memorising
#                 the corruption it was trained against.
DEGRADE_FAMILIES: dict[str, tuple[str, ...]] = {
    "augmentable": ("illumination", "pose_warp"),
    "heldout": ("motion_blur", "downscale", "jpeg"),
}
# Pipeline order — the order a real capture applies them, independent of which
# group is selected.
DEGRADE_KINDS = ("illumination", "pose_warp", "motion_blur", "downscale", "jpeg")

assert not set(DEGRADE_FAMILIES["augmentable"]) & set(DEGRADE_FAMILIES["heldout"]), \
    "the corruption families must stay disjoint - that is their entire purpose"
assert set(DEGRADE_KINDS) == set(DEGRADE_FAMILIES["augmentable"]) | set(
    DEGRADE_FAMILIES["heldout"]), "every kind must belong to exactly one family"


def family_of(kind: str) -> str:
    """Which group a corruption belongs to."""
    for family, kinds in DEGRADE_FAMILIES.items():
        if kind in kinds:
            return family
    return "unknown"


def _kind_rng(seed: int, index: int, kind: str) -> np.random.Generator:
    """An independent parameter stream per (seed, image, corruption).

    Deliberately *not* one generator per image consumed in pipeline order: that
    made a corruption's parameters depend on which other corruptions were
    selected, so ``--degrade-kinds jpeg`` produced a different JPEG quality than
    the same image got inside the full composite. Hashing the kind name into the
    seed makes each corruption reproducible on its own terms, which is what
    lets the two family groups be compared as a controlled ablation.
    """
    digest = hashlib.sha256(f"{seed}:{index}:{kind}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "little"))


def _motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    """A normalised line kernel — a camera moving in a straight line."""
    k = np.zeros((size, size), dtype=np.float32)
    centre = (size - 1) / 2.0
    dx = math.cos(math.radians(angle_deg))
    dy = math.sin(math.radians(angle_deg))
    for t in np.linspace(-centre, centre, size * 2):
        x = int(round(centre + dx * t))
        y = int(round(centre + dy * t))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    total = k.sum()
    return k / total if total else k


def _apply_illumination(img: np.ndarray, rng: np.random.Generator,
                        cv2: Any) -> np.ndarray:
    """Underexposed room: gain 0.30-0.65 with a gamma lift, not a flat dim."""
    gain = float(rng.uniform(0.30, 0.65))
    gamma = float(rng.uniform(1.0, 1.6))
    norm = np.power(img.astype(np.float32) / 255.0, gamma)
    return np.clip(norm * gain * 255.0, 0, 255).astype(np.uint8)


def _apply_pose_warp(img: np.ndarray, rng: np.random.Generator,
                     cv2: Any) -> np.ndarray:
    """A yaw-like perspective warp — a crude 2D stand-in for turning the head.

    Honest about what this is: a planar perspective transform cannot rotate a
    head. It compresses one side of the frame and cannot reveal the structure
    an actual profile view would expose or hide. It is a *proxy* for pose, not
    pose.

    That limitation is exactly why it sits in the ``augmentable`` group. It is
    the kind of effect a synthetic augmentation can cheaply imitate, so it is
    reserved for the augmentation to use and is never evaluated on. Real pose
    generalisation needs CFP-FP or real captures, and no result from this
    corruption should be read as evidence about pose.
    """
    height, width = img.shape[:2]
    yaw = float(rng.uniform(-0.35, 0.35))
    shrink = abs(yaw) * width
    inset = abs(yaw) * height * 0.12
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    if yaw >= 0:                       # right edge recedes
        dst = np.float32([[0, 0], [width - shrink, inset],
                          [width - shrink, height - inset], [0, height]])
    else:                              # left edge recedes
        dst = np.float32([[shrink, inset], [width, 0],
                          [width, height], [shrink, height - inset]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (width, height),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _apply_motion_blur(img: np.ndarray, rng: np.random.Generator,
                       cv2: Any) -> np.ndarray:
    """A 5-13 px line kernel at a random angle: someone walking past."""
    size = int(rng.integers(5, 14))
    kernel = _motion_blur_kernel(size, float(rng.uniform(0.0, 180.0)))
    return cv2.filter2D(img, -1, kernel)


def _apply_downscale(img: np.ndarray, rng: np.random.Generator,
                     cv2: Any) -> np.ndarray:
    """Decimate 2.5-4x and scale back: a small face in a 640x480 frame."""
    factor = float(rng.uniform(2.5, 4.0))
    height, width = img.shape[:2]
    small = cv2.resize(
        img, (max(1, int(width / factor)), max(1, int(height / factor))),
        interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _apply_jpeg(img: np.ndarray, rng: np.random.Generator,
                cv2: Any) -> np.ndarray:
    """Quality 15-40, the compression an MJPEG stream applies."""
    quality = int(rng.integers(15, 41))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


DEGRADE_FUNCS = {
    "illumination": _apply_illumination,
    "pose_warp": _apply_pose_warp,
    "motion_blur": _apply_motion_blur,
    "downscale": _apply_downscale,
    "jpeg": _apply_jpeg,
}


def degrade_images(images: Any, args: argparse.Namespace) -> list[np.ndarray]:
    """Approximate webcam capture conditions, reproducibly.

    LFW is press photography: well-lit, sharp, high-bitrate. A webcam frame in
    a classroom is none of those, so a threshold calibrated on clean LFW is
    calibrated on the wrong distribution.

    Which corruptions run is chosen by ``--degrade-family`` (see
    ``DEGRADE_FAMILIES``); they are always applied in ``DEGRADE_KINDS`` order,
    which is the order a real capture pipeline applies them. Each corruption
    draws its parameters from its own ``_kind_rng(seed, image, kind)`` stream,
    so a given image gets the same JPEG quality whether JPEG runs alone or
    inside a composite — the family groups are therefore a controlled ablation
    rather than two unrelated distortions.

    Only *probes* are ever degraded. The gallery stays clean: enrolment in the
    app is a deliberate, usually cooperative act, while the frames it later has
    to match are whatever the camera produced.
    """
    import cv2  # noqa: PLC0415

    unknown = set(args.degrade_kinds) - set(DEGRADE_KINDS)
    if unknown:
        raise SystemExit(f"FATAL: unknown --degrade-kinds {sorted(unknown)}; "
                         f"choose from {list(DEGRADE_KINDS)}")
    ordered = [k for k in DEGRADE_KINDS if k in set(args.degrade_kinds)]

    out: list[np.ndarray] = []
    started = time.time()
    for i, img in enumerate(images):
        cur = img
        for kind in ordered:
            cur = DEGRADE_FUNCS[kind](cur, _kind_rng(args.seed, i, kind), cv2)
        out.append(np.ascontiguousarray(cur))

    print(f"  degraded {len(images)} images with {ordered} "
          f"({time.time() - started:.1f}s)")
    return out


def save_degrade_examples(clean: np.ndarray, dirty: np.ndarray,
                          out_dir: Path, seed: int, n: int = 8) -> str:
    """Clean-over-degraded strip, so the corruption can be eyeballed.

    A degradation nobody looked at is a degradation nobody can defend. If these
    look nothing like a webcam frame, the degraded numbers below them mean
    nothing either.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    pick = np.random.default_rng(seed + 11).choice(
        len(clean), size=min(n, len(clean)), replace=False)
    pick = np.sort(pick)

    fig, axes = plt.subplots(2, len(pick), squeeze=False,
                             figsize=(len(pick) * 1.7, 4.0))
    for col, idx in enumerate(pick):
        axes[0][col].imshow(clean[idx][..., ::-1])
        axes[0][col].axis("off")
        axes[1][col].imshow(dirty[idx][..., ::-1])
        axes[1][col].axis("off")
    axes[0][0].set_ylabel("clean")
    fig.suptitle("Probe degradation: clean (top) vs. degraded (bottom)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "degraded_examples.png", dpi=150)
    plt.close(fig)
    return "degraded_examples.png"


# ------------------------------------------------------------------ statistics
def wilson_ci(successes: int, n: int, conf: float = 0.95
              ) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal ("Wald") interval because every rate worth
    reporting here is either near 0 (FAR) or near 1 (rank-1 accuracy), exactly
    where Wald intervals go wrong — Wald gives 4/310 an interval that includes
    negative rates, and gives 0/1350 a width of zero.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(1.0 - (1.0 - conf) / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(successes: int, n: int, conf: float = 0.95) -> dict[str, Any]:
    """A proportion reported the only way it should be: with its interval."""
    successes, n = int(successes), int(n)
    lo, hi = wilson_ci(successes, n, conf)
    return {
        "count": successes,
        "n": n,
        "rate": (successes / n) if n else None,
        "ci_low": lo,
        "ci_high": hi,
        "ci_half_width": (hi - lo) / 2.0,
        "conf": conf,
    }


def fmt_rate(r: dict[str, Any], pct_fmt: str = ".2%") -> str:
    """'98.71% [96.7%, 99.5%] (n=310)' — the format used in every printout."""
    if r["rate"] is None:
        return "n/a (n=0)"
    return (f"{r['rate']:{pct_fmt}} "
            f"[{r['ci_low']:{pct_fmt}}, {r['ci_high']:{pct_fmt}}] "
            f"(n={r['n']})")


def mdd_unpaired(n: int, p_baseline: float, alpha: float = 0.05,
                 power: float = 0.80) -> float | None:
    """Smallest rank-1 difference detectable between two *independent* runs.

    Two-sided two-proportion z-test, equal sample size ``n`` per arm. Solved by
    bisection on the difference because the closed form runs the other way
    (n from delta) and we want delta from n.
    """
    if n <= 0 or not (0.0 < p_baseline < 1.0):
        return None
    z_a = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    def achieved_power(delta: float) -> float:
        p2 = min(1.0 - 1e-12, p_baseline + delta)
        pbar = (p_baseline + p2) / 2.0
        se_null = math.sqrt(2.0 * pbar * (1.0 - pbar) / n)
        se_alt = math.sqrt((p_baseline * (1.0 - p_baseline)
                            + p2 * (1.0 - p2)) / n)
        if se_alt <= 0.0:
            return 1.0
        return NormalDist().cdf((delta - z_a * se_null) / se_alt)

    lo, hi = 0.0, 1.0 - p_baseline
    if achieved_power(hi) < power:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if achieved_power(mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def mdd_paired_min_errors(alpha: float = 0.05) -> int:
    """Fewest net-fixed errors that reach significance on the *same* probes.

    Best case for a paired comparison: the new system fixes ``b`` of the old
    system's errors and breaks none. McNemar's test then reduces to an exact
    sign test on ``b`` discordant pairs, all pointing one way, whose two-sided
    p-value is ``2 * 0.5**b``. This is the floor: any real system also breaks
    some cases, which pushes the requirement up.
    """
    b = 1
    while 2.0 * (0.5 ** b) > alpha and b < 64:
        b += 1
    return b


def mdd_for(r: dict[str, Any], min_fixed: int, alpha: float, power: float
            ) -> dict[str, Any]:
    """Minimum detectable difference for one measured proportion.

    Two bars, because they answer different questions:

    * **paired** — the same probes re-scored after a change. The comparison is
      McNemar's; in the best case (the change fixes errors and breaks nothing)
      it reduces to an exact sign test needing ``min_fixed`` discordant pairs.
      This is the bar that applies to "I changed enrolment and re-ran".
    * **unpaired** — two independent evaluations, the standard two-proportion
      z-test at the requested power.

    ``measurable`` is the one that matters in practice: a metric with fewer
    observed errors than ``min_fixed`` cannot show a significant improvement no
    matter how the change performs, because there are not enough errors left to
    fix. Sample size does not help; only a harder dataset does.
    """
    n, k = r["n"], r["count"]
    unpaired = (mdd_unpaired(n, r["rate"], alpha, power)
                if n and r["rate"] is not None and 0.0 < r["rate"] < 1.0
                else None)
    return {
        "n": n,
        "errors_available": k,
        "mdd_paired_pp": (min_fixed / n * 100.0) if n else None,
        "mdd_paired_min_errors_fixed": min_fixed,
        "mdd_unpaired_pp": (unpaired * 100.0) if unpaired is not None else None,
        "measurable": bool(k >= min_fixed),
    }


# Public benchmarks worth adding when LFW runs out of errors to fix. Named
# rather than gestured at, so the next session does not have to re-derive them.
HARDER_BENCHMARKS = [
    ("CFP-FP", "Celebrities in Frontal-Profile. 500 identities, explicit "
               "frontal-vs-profile pairs. buffalo_sc scores far lower here "
               "than on LFW, so pose-driven errors are plentiful - the right "
               "choice if enrolment augmentation is meant to cover pose."),
    ("AgeDB-30", "Age-gap verification with a 30-year separation. Isolates "
                 "age drift, the other axis a single enrolment frame fails "
                 "on, and the one that matters for a gallery enrolled once "
                 "and used for years."),
    ("IJB-C", "Template-based 1:N protocol with a genuinely large gallery and "
              "published open-set (FPIR/FNIR) operating points. The closest "
              "public analogue to what this harness measures, and the one to "
              "use if the open-set FAR tail is the target."),
]


def augmentation_headroom(candidates: list[dict[str, Any]], min_fixed: int
                          ) -> dict[str, Any]:
    """Which metric can actually show an enrolment-augmentation improvement.

    Ranks the candidate metrics by how many errors are available to fix, since
    that - not the sample size - is what bounds a paired test. Returns a verdict
    that either names the metric to track or says plainly that none of them can
    carry the experiment on this dataset.
    """
    usable = [c for c in candidates
              if c["affected_by_enrolment"] and c["mdd"]["measurable"]]
    usable.sort(key=lambda c: c["mdd"]["errors_available"], reverse=True)
    best = usable[0] if usable else None
    clean_best = next((c for c in usable if not c.get("circularity_risk")), None)

    if best is not None:
        available = best["mdd"]["errors_available"]
        share = min_fixed / available
        room = ("ample headroom" if share <= 0.05 else
                "workable headroom" if share <= 0.25 else
                "tight headroom - a small regression would wipe out the signal")
        verdict = (
            f"Track '{best['name']}': {available:,} errors out of "
            f"{best['mdd']['n']:,} probes. A paired re-run must net-fix "
            f"{min_fixed} of them ({best['mdd']['mdd_paired_pp']:.2f} pp, "
            f"{share:.1%} of the errors that exist) to reach significance - "
            f"{room}."
        )
        if best.get("circularity_risk"):
            verdict += " " + best["circularity_risk"]
            if clean_best is not None and clean_best is not best:
                verdict += (
                    f" The largest metric with no such risk is "
                    f"'{clean_best['name']}' with "
                    f"{clean_best['mdd']['errors_available']:,} errors in "
                    f"{clean_best['mdd']['n']:,} probes "
                    f"({clean_best['mdd']['mdd_paired_pp']:.2f} pp) - report "
                    "that one as the headline and the degraded arm as "
                    "supporting evidence.")
    else:
        blocked = sorted((c for c in candidates if c["affected_by_enrolment"]),
                         key=lambda c: c["mdd"]["errors_available"],
                         reverse=True)
        top = blocked[0] if blocked else None
        verdict = (
            "NO METRIC HAS ENOUGH HEADROOM. Every candidate an enrolment "
            "change could move has fewer than "
            f"{min_fixed} errors"
            + (f" (the largest is '{top['name']}' with "
               f"{top['mdd']['errors_available']}), " if top else ", ")
            + "so no enrolment-augmentation improvement can be shown "
              "significant on this data at any sample size. LFW is too easy "
              "for this experiment: it is frontal, well-lit, adult-to-adult, "
              "and buffalo_sc has effectively solved it. Do not work around "
              "this by loosening the test - add a harder public benchmark."
        )

    return {
        "min_errors_for_paired_significance": min_fixed,
        "candidates": candidates,
        "recommended_metric": best["name"] if best else None,
        "recommended_metric_no_circularity": (
            clean_best["name"] if clean_best else None),
        "verdict": verdict,
        "harder_benchmarks": [{"name": n, "why": w} for n, w in
                              HARDER_BENCHMARKS],
    }


# ---------------------------------------------------------------------- splits
def build_splits(labels: np.ndarray, valid: np.ndarray, n_ids: int,
                 args: argparse.Namespace, pose: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, list[int],
                            dict[int, dict[str, np.ndarray]]]:
    """Split each identity's detected images into probe / gallery sets.

    Two modes, both disjoint by construction — an image index lands in exactly
    one of the two:

    ``gallery-first`` (default): reserve ``max(--enrol-counts)`` images per
    identity as enrolment candidates and make *everything else* a probe. This
    is both the higher-power split (it maximises probes) and the more faithful
    one: the app enrols a person from one click on one frame
    (``engine.register``) or from a handful of photos (``encode_faces.py``),
    then matches every later frame against that. A 5-probe / 45-gallery split
    flatters the matcher in a way the deployment never will.

    The *maximum* enrolment count is reserved regardless of which count is
    actually being evaluated, and each count uses a **prefix** of that reserved
    list. So N=1, N=3 and N=5 are scored on exactly the same probes against
    nested galleries, and any difference between them is caused by enrolment
    size alone rather than by a shifting probe set. It costs a handful of
    probes at N=1 and buys a properly paired comparison.

    ``probe-first``: the original behaviour — ``--probes-per-identity`` probes,
    everything else gallery. Kept so the earlier baseline can be reproduced.

    Returns (probe_idx, gallery_idx, dropped_identities, per_identity), where
    per_identity maps identity -> {"gallery": idx, "probes": idx} and is what
    the open-set protocol re-partitions by identity.
    """
    rng = np.random.default_rng(args.seed)
    probe_idx: list[int] = []
    gallery_idx: list[int] = []
    dropped: list[int] = []
    per_identity: dict[int, dict[str, np.ndarray]] = {}

    reserve = max([args.enrol_count, *args.enrol_counts])
    pose_split = pose is not None and args.cfp_protocol == "fp"

    for ident in range(n_ids):
        idx = np.flatnonzero((labels == ident) & valid)
        if pose_split:
            # The CFP-FP protocol: enrol FRONTAL, probe PROFILE. This is also
            # the deployment analogue — the operator registers a cooperative
            # face-on frame, and the camera then has to match whatever angle
            # the person happens to present. Nothing else in the harness has
            # to know; the rest of the pipeline just sees a gallery and probes.
            frontal = idx[pose[idx] == POSE_FRONTAL]
            profile = idx[pose[idx] == POSE_PROFILE]
            if len(frontal) < reserve or len(profile) < 1:
                dropped.append(ident)
                continue
            gal = rng.permutation(frontal)[:reserve]
            prb = np.sort(profile)
            if args.max_probes_per_identity:
                prb = prb[:args.max_probes_per_identity]
            per_identity[ident] = {"gallery": gal, "probes": prb}
            gallery_idx.extend(gal[:args.enrol_count].tolist())
            probe_idx.extend(prb.tolist())
            continue

        if args.split_mode == "gallery-first":
            if len(idx) < reserve + 1:
                dropped.append(ident)
                continue
            shuffled = rng.permutation(idx)
            gal = shuffled[:reserve]
            prb = shuffled[reserve:]
            if args.max_probes_per_identity:
                prb = prb[:args.max_probes_per_identity]
        else:
            if len(idx) < args.probes_per_identity + 1:
                dropped.append(ident)
                continue
            shuffled = rng.permutation(idx)
            prb = shuffled[:args.probes_per_identity]
            gal = shuffled[args.probes_per_identity:]

        # Gallery order is the draw order, NOT sorted: prefixes have to be a
        # random subset, and sorting would make N=1 always the earliest photo
        # of that person.
        prb = np.sort(prb)
        per_identity[ident] = {"gallery": gal, "probes": prb}
        # The closed-set / verification analyses use the headline enrolment
        # count; the reserved remainder exists only so the sweep can grow the
        # gallery without touching the probes.
        enrolled = (gal[:args.enrol_count]
                    if args.split_mode == "gallery-first" else gal)
        gallery_idx.extend(enrolled.tolist())
        probe_idx.extend(prb.tolist())

    return (np.sort(np.array(probe_idx, dtype=np.int64)),
            np.sort(np.array(gallery_idx, dtype=np.int64)),
            dropped, per_identity)


# ------------------------------------------------------- verification metrics
def far_at(threshold: np.ndarray, impostor_sorted: np.ndarray) -> np.ndarray:
    """False accept rate: fraction of impostor pairs scoring >= threshold."""
    n = len(impostor_sorted)
    above = n - np.searchsorted(impostor_sorted, threshold, side="left")
    return above / n


def count_at_or_above(threshold: np.ndarray | float,
                      sorted_scores: np.ndarray) -> np.ndarray:
    """How many scores are >= threshold — the *support* behind a FAR figure.

    A FAR of 0.0074% means nothing on its own; "1 pair out of 13530" says
    immediately that the figure is one outlier away from zero.
    """
    t = np.atleast_1d(np.asarray(threshold, dtype=np.float64))
    return len(sorted_scores) - np.searchsorted(sorted_scores, t, side="left")


def frr_at(threshold: np.ndarray, genuine_sorted: np.ndarray) -> np.ndarray:
    """False reject rate: fraction of genuine pairs scoring < threshold."""
    n = len(genuine_sorted)
    below = np.searchsorted(genuine_sorted, threshold, side="left")
    return below / n


# --------------------------------------------------- open-set identification
def identify(probe_emb: np.ndarray, gal_emb: np.ndarray, gal_lab: np.ndarray,
             enrolled: np.ndarray) -> dict[str, np.ndarray]:
    """Run the app's matcher: max over the gallery, then argmax.

    ``engine._recognize`` computes ``sims = known_matrix @ emb`` over every
    enrolled *embedding*, takes ``sims.argmax()``, and reads the name off that
    row. Taking the per-identity maximum first and then the argmax over
    identities is the same decision (an identity's score is its best entry),
    and it additionally gives the runner-up identity, which is what the top-2
    margin needs.

    Returns per-probe: top1_score, top1_ident, top2_margin, best_entry
    (a column into ``gal_emb``/the gallery index array, so a failure grid can
    show *which* enrolled photo was matched).
    """
    sims = probe_emb @ gal_emb.T                      # (P, G_entries)
    best_entry = np.argmax(sims, axis=1)
    n_probes = len(probe_emb)

    id_scores = np.full((n_probes, len(enrolled)), -np.inf, dtype=np.float32)
    for col, ident in enumerate(enrolled):
        member = gal_lab == ident
        if member.any():
            id_scores[:, col] = sims[:, member].max(axis=1)

    order = np.argsort(-id_scores, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(id_scores, order, axis=1)
    top1_score = sorted_scores[:, 0]
    top1_ident = np.asarray(enrolled)[order[:, 0]]
    if len(enrolled) >= 2:
        margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    else:
        margin = np.zeros(n_probes, dtype=np.float32)

    return {
        "sims": sims,
        "id_scores": id_scores,
        "order": order,
        "top1_score": top1_score,
        "top1_ident": top1_ident,
        "margin": margin,
        "best_entry": best_entry,
    }


def openset_curves(gen_score: np.ndarray, gen_correct: np.ndarray,
                   imp_score: np.ndarray, grid: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """FAR / FRR / misidentification rate as functions of the threshold.

    Three *different* failures, deliberately not collapsed into one number:

    * ``far``  — an impostor probe (identity not enrolled at all) whose top
      match clears the threshold. The system marks a stranger present.
    * ``frr``  — an enrolled person's probe whose top match does not clear the
      threshold. They stand there unrecognised; visible and self-correcting.
    * ``mir``  — an enrolled person's probe that clears the threshold on the
      *wrong* identity. Someone else is marked present, and the real person is
      not. This is the failure that a pairwise verification ROC cannot see at
      all, because it needs a gallery of competing identities to happen.

    Over genuine probes, correct-accept + frr + mir = 1 exactly.
    """
    n_gen, n_imp = len(gen_score), len(imp_score)
    imp_sorted = np.sort(imp_score)
    gen_sorted = np.sort(gen_score)
    wrong_sorted = np.sort(gen_score[~gen_correct])

    far_n = count_at_or_above(grid, imp_sorted)
    frr_n = np.searchsorted(gen_sorted, grid, side="left")
    mir_n = count_at_or_above(grid, wrong_sorted)

    return {
        "grid": grid,
        "far": far_n / n_imp if n_imp else np.zeros_like(grid),
        "frr": frr_n / n_gen if n_gen else np.zeros_like(grid),
        "mir": mir_n / n_gen if n_gen else np.zeros_like(grid),
        "far_n": far_n,
        "frr_n": frr_n,
        "mir_n": mir_n,
        "n_genuine": n_gen,
        "n_impostor": n_imp,
    }


def run_openset(gallery_emb: np.ndarray, probe_emb: np.ndarray,
                probe_valid: np.ndarray,
                per_identity: dict[int, dict[str, np.ndarray]],
                names: list[str], args: argparse.Namespace,
                enrol_count: int) -> dict[str, Any]:
    """The protocol that actually governs the app's threshold.

    Per trial: shuffle the identities, enrol the first ``max(gallery_sizes)``
    of them, and hold out the next ``--impostor-identities`` *entirely* — no
    image of theirs is ever in any gallery, so their probes must be rejected.
    Gallery sizes are nested prefixes of the same draw (10 ⊂ 25 ⊂ 50) so that
    the shift between curves is caused by gallery size and nothing else.

    ``enrol_count`` takes a prefix of each identity's reserved enrolment
    images, mirroring how many frames the app actually has for that person —
    one, for a face registered through the UI.

    The gallery and the probes come from **separate** embedding arrays so the
    degraded arm can match dirty probes against a clean enrolment, which is the
    deployed situation. ``probe_valid`` drops probes the detector missed in the
    probe arm; those are counted as detection failures and reported separately
    rather than being folded into the threshold's error rates (see
    ``end_to_end_miss_rate``).

    The identity draw is seeded per trial and does not depend on
    ``enrol_count`` or on which arm is running, so every cell of the sweep sees
    the identical identity partition and the comparisons are paired.
    """
    eligible = sorted(i for i, d in per_identity.items()
                      if len(d["gallery"]) and len(d["probes"]))
    sizes = sorted({s for s in args.gallery_sizes if s <= len(eligible)})
    if not sizes:
        raise SystemExit(
            f"FATAL: no requested gallery size fits {len(eligible)} usable "
            f"identities (asked for {args.gallery_sizes}).")
    if len(sizes) < len(args.gallery_sizes):
        print(f"  note: gallery sizes "
              f"{sorted(set(args.gallery_sizes) - set(sizes))} exceed the "
              f"{len(eligible)} available identities and were skipped")

    max_size = max(sizes)
    n_imp_ids = min(args.impostor_identities, len(eligible) - max_size)
    if n_imp_ids < 1:
        raise SystemExit(
            f"FATAL: {len(eligible)} identities cannot supply a {max_size}-"
            "identity gallery AND a disjoint impostor set. Lower "
            "--gallery-sizes or --min-faces.")
    if n_imp_ids < args.impostor_identities:
        print(f"  note: only {n_imp_ids} impostor identities available "
              f"(asked for {args.impostor_identities})")

    # Probes the probe-arm detector missed. They never reach the matcher, so
    # they are excluded from every rate here and counted as detection failures
    # instead; folding them into FRR would make the threshold answer for a
    # failure the threshold cannot fix.
    def probes_of(ident: int) -> np.ndarray:
        idx = per_identity[int(ident)]["probes"]
        return idx[probe_valid[idx]]

    per_size: dict[int, dict[str, Any]] = {
        s: {"gen_score": [], "gen_correct": [], "imp_score": [],
            "gen_per_trial": [], "imp_per_trial": [], "records": [],
            "gen_undetected": 0, "imp_undetected": 0}
        for s in sizes
    }

    for trial in range(args.openset_trials):
        rng = np.random.default_rng(args.seed * 1000 + trial)
        perm = rng.permutation(np.array(eligible, dtype=np.int64))
        gal_pool = perm[:max_size]
        imp_ids = perm[max_size:max_size + n_imp_ids]
        imp_probe_idx = np.concatenate([probes_of(i) for i in imp_ids])
        imp_missed = sum(int((~probe_valid[per_identity[int(i)]["probes"]]).sum())
                         for i in imp_ids)

        for size in sizes:
            enrolled = np.sort(gal_pool[:size])
            gal_idx = np.concatenate([per_identity[int(i)]["gallery"][:enrol_count]
                                      for i in enrolled])
            gal_lab = np.concatenate([
                np.full(len(per_identity[int(i)]["gallery"][:enrol_count]),
                        int(i))
                for i in enrolled])

            gen_lists = [probes_of(i) for i in enrolled]
            gen_probe_idx = np.concatenate(gen_lists)
            gen_missed = sum(
                int((~probe_valid[per_identity[int(i)]["probes"]]).sum())
                for i in enrolled)

            probe_idx = np.concatenate([gen_probe_idx, imp_probe_idx])
            probe_true = np.concatenate([
                np.concatenate([np.full(len(g), int(i))
                                for g, i in zip(gen_lists, enrolled)]),
                np.concatenate([np.full(len(probes_of(i)), int(i))
                                for i in imp_ids]),
            ])
            is_genuine = np.zeros(len(probe_idx), dtype=bool)
            is_genuine[:len(gen_probe_idx)] = True

            res = identify(probe_emb[probe_idx], gallery_emb[gal_idx],
                           gal_lab, enrolled)
            correct = res["top1_ident"] == probe_true

            bucket = per_size[size]
            bucket["gen_score"].append(res["top1_score"][is_genuine])
            bucket["gen_correct"].append(correct[is_genuine])
            bucket["imp_score"].append(res["top1_score"][~is_genuine])
            bucket["gen_per_trial"].append(int(is_genuine.sum()))
            bucket["imp_per_trial"].append(int((~is_genuine).sum()))
            bucket["gen_undetected"] += gen_missed
            bucket["imp_undetected"] += imp_missed
            if trial == 0:
                # Keep trial 0 whole so the failure grids have real images to
                # render; the other trials only contribute scores.
                bucket["records"] = {
                    "probe_index": probe_idx,
                    "true_ident": probe_true,
                    "pred_ident": res["top1_ident"],
                    "score": res["top1_score"],
                    "margin": res["margin"],
                    "match_index": gal_idx[res["best_entry"]],
                    "is_genuine": is_genuine,
                    "correct": correct,
                }

    # ---- pool the trials, build the curves --------------------------------
    curves: dict[int, dict[str, Any]] = {}
    all_scores = []
    for size in sizes:
        b = per_size[size]
        gen_score = np.concatenate(b["gen_score"])
        gen_correct = np.concatenate(b["gen_correct"])
        imp_score = np.concatenate(b["imp_score"])
        all_scores.append(gen_score)
        all_scores.append(imp_score)
        curves[size] = {"gen_score": gen_score, "gen_correct": gen_correct,
                        "imp_score": imp_score,
                        "gen_per_trial": b["gen_per_trial"],
                        "imp_per_trial": b["imp_per_trial"],
                        "gen_undetected": b["gen_undetected"],
                        "imp_undetected": b["imp_undetected"],
                        "records": b["records"]}

    grid = np.unique(np.concatenate(all_scores).astype(np.float64))
    for size in sizes:
        c = curves[size]
        c.update(openset_curves(c["gen_score"], c["gen_correct"],
                                c["imp_score"], grid))

    return {
        "sizes": sizes,
        "curves": curves,
        "grid": grid,
        "enrol_count": enrol_count,
        "n_impostor_identities": int(n_imp_ids),
        "n_eligible_identities": len(eligible),
        "trials": args.openset_trials,
        "names": names,
    }


def openset_operating_point(c: dict[str, Any], threshold: float,
                            conf: float) -> dict[str, Any]:
    """FAR / FRR / MIR at one threshold, each with its support and interval."""
    imp_sorted = np.sort(c["imp_score"])
    gen_sorted = np.sort(c["gen_score"])
    wrong_sorted = np.sort(c["gen_score"][~c["gen_correct"]])
    far_n = int(count_at_or_above(threshold, imp_sorted)[0])
    frr_n = int(np.searchsorted(gen_sorted, threshold, side="left"))
    mir_n = int(count_at_or_above(threshold, wrong_sorted)[0])
    n_gen, n_imp = len(gen_sorted), len(imp_sorted)
    return {
        "threshold": float(threshold),
        "far": rate(far_n, n_imp, conf),
        "frr": rate(frr_n, n_gen, conf),
        "misidentification": rate(mir_n, n_gen, conf),
        "correct_accept": rate(n_gen - frr_n - mir_n, n_gen, conf),
    }


def summarise_openset(os_res: dict[str, Any], args: argparse.Namespace,
                      backend: dict[str, Any], conf: float,
                      verif_threshold: float, deploy_size: int
                      ) -> dict[str, Any]:
    """Turn one (arm, enrolment count) run into the reportable block.

    Everything here is per gallery size: the threshold that meets the FAR
    budget, what that threshold costs, and what the *deploy-size* threshold
    costs at every other size — the "does it still hold when the gallery
    grows?" table.
    """
    sizes = os_res["sizes"]
    out: dict[str, Any] = {"enrol_count": os_res["enrol_count"],
                           "by_gallery_size": {}}

    for size in sizes:
        c = os_res["curves"][size]
        hit = np.flatnonzero(c["far"] <= args.openset_target_far)
        resolved = bool(len(hit))
        thr = (float(c["grid"][int(hit[0])]) if resolved
               else float(np.max(c["imp_score"])) + 1e-6)
        n_gen_seen = int(c["n_genuine"])
        out["by_gallery_size"][str(size)] = {
            "n_genuine_probes": n_gen_seen,
            "n_impostor_probes": int(c["n_impostor"]),
            "genuine_probes_per_trial": c["gen_per_trial"],
            "impostor_probes_per_trial": c["imp_per_trial"],
            "genuine_probes_undetected": int(c["gen_undetected"]),
            "impostor_probes_undetected": int(c["imp_undetected"]),
            "far_resolution_pooled": 1.0 / max(1, c["n_impostor"]),
            "far_resolution_per_trial": 1.0 / max(1, c["imp_per_trial"][0]),
            "threshold_for_target_far": thr,
            "target_far_resolved": resolved,
            "at_threshold_for_target_far": openset_operating_point(c, thr, conf),
            "at_backend_strong_match":
                openset_operating_point(c, backend["strong_match"], conf),
            "at_backend_weak_match":
                openset_operating_point(c, backend["weak_match"], conf),
            "at_verification_recommended":
                openset_operating_point(c, verif_threshold, conf),
        }

    deploy = out["by_gallery_size"][str(deploy_size)]
    threshold = deploy["threshold_for_target_far"]
    for size in sizes:
        out["by_gallery_size"][str(size)]["at_recommended"] = \
            openset_operating_point(os_res["curves"][size], threshold, conf)

    # A single FAR budget is a knife edge when the budget buys only a handful
    # of impostor probes, so show what the neighbouring budgets cost too.
    deploy_curve = os_res["curves"][deploy_size]
    out["far_budget_table"] = []
    for budget in sorted({0.05, 0.02, 0.01, 0.005, args.openset_target_far},
                         reverse=True):
        hit = np.flatnonzero(deploy_curve["far"] <= budget)
        thr = (float(deploy_curve["grid"][int(hit[0])]) if len(hit)
               else float(np.max(deploy_curve["imp_score"])) + 1e-6)
        out["far_budget_table"].append({
            "far_budget": budget,
            "threshold": thr,
            "impostor_probes_allowed": budget * deploy_curve["n_impostor"],
            **openset_operating_point(deploy_curve, thr, conf),
        })

    # A genuine probe the detector never saw is a person not marked present,
    # even though no threshold could have helped. Report it beside FRR so the
    # end-to-end miss rate is visible without conflating the two causes.
    gen_seen = deploy["n_genuine_probes"]
    gen_missed = deploy["genuine_probes_undetected"]
    frr_n = deploy["at_recommended"]["frr"]["count"]
    out["threshold"] = threshold
    out["deploy_gallery_size"] = deploy_size
    out["target_far_resolved"] = deploy["target_far_resolved"]
    out["under_resolved"] = (deploy["far_resolution_per_trial"]
                             > args.openset_target_far)
    out["impostor_probes_per_trial"] = deploy["impostor_probes_per_trial"][0]
    out["at_deploy_size"] = deploy["at_recommended"]
    out["detection_failure_on_probes"] = rate(
        gen_missed, gen_seen + gen_missed, conf)
    out["end_to_end_miss_rate"] = rate(
        frr_n + gen_missed, gen_seen + gen_missed, conf)
    return out


# ------------------------------------------------------------ failure grids
def save_failure_grid(images: np.ndarray, records: list[dict[str, Any]],
                      out_dir: Path, slug: str, title: str,
                      cols: int = 6, per_page: int = 24) -> list[str]:
    """Write misclassified probes as a labelled image grid, probe over match.

    Each cell is the probe on top and the gallery photo its top match came from
    underneath, so a wrong prediction can be read at a glance: 'these two do
    look alike' is a different problem from 'the probe is a blurry profile'.
    Records are ordered most-confident-first, because a confident error is the
    one that actually marks the wrong person present.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if not records:
        return written

    pages = [records[i:i + per_page] for i in range(0, len(records), per_page)]
    for page_no, page in enumerate(pages, start=1):
        rows = math.ceil(len(page) / cols)
        fig, axes = plt.subplots(rows * 2, cols, squeeze=False,
                                 figsize=(cols * 2.0, rows * 4.6))
        for ax in axes.ravel():
            ax.axis("off")
        for k, rec in enumerate(page):
            r, c = divmod(k, cols)
            ax_p, ax_g = axes[2 * r, c], axes[2 * r + 1, c]
            ax_p.imshow(images[rec["probe_index"]][..., ::-1])
            crowd = rec.get("faces_in_probe_frame", 1)
            ax_p.set_title(f"true: {rec['true_name']}"
                           + (f"\n[{crowd} faces in frame]" if crowd > 1 else ""),
                           fontsize=6, color="seagreen", pad=3)
            ax_g.imshow(images[rec["match_index"]][..., ::-1])
            ax_g.set_title(
                f"pred: {rec['pred_name']}\n"
                f"sim {rec['score']:.3f}   margin {rec['margin']:.3f}",
                fontsize=6, color="indianred", pad=3)
        suffix = (f"  (page {page_no}/{len(pages)})" if len(pages) > 1 else "")
        fig.suptitle(title + suffix, fontsize=10)
        # h_pad keeps the two-line 'pred:' caption off the probe image above it.
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.2)
        name = (f"{slug}.png" if len(pages) == 1
                else f"{slug}_page{page_no:02d}.png")
        fig.savefig(out_dir / name, dpi=140)
        plt.close(fig)
        written.append(name)
    return written


def collect_failures(probe_index: np.ndarray, true_ident: np.ndarray,
                     pred_ident: np.ndarray, score: np.ndarray,
                     margin: np.ndarray, match_index: np.ndarray,
                     mask: np.ndarray, names: list[str],
                     n_faces: np.ndarray | None = None,
                     impostor: bool = False) -> list[dict[str, Any]]:
    """Turn a boolean failure mask into sorted, human-readable records.

    ``n_faces`` is carried through so the grid can flag a probe whose frame
    held more than one person — the difference between "the matcher is wrong"
    and "the harness embedded the wrong person's face".
    """
    sel = np.flatnonzero(mask)
    sel = sel[np.argsort(-score[sel], kind="stable")]
    out = []
    for i in sel:
        pi = int(probe_index[i])
        rec = {
            "probe_index": pi,
            "match_index": int(match_index[i]),
            "true_name": ("IMPOSTOR / " + names[int(true_ident[i])]
                          if impostor else names[int(true_ident[i])]),
            "pred_name": names[int(pred_ident[i])],
            "score": float(score[i]),
            "margin": float(margin[i]),
        }
        if n_faces is not None:
            rec["faces_in_probe_frame"] = int(n_faces[pi])
        out.append(rec)
    return out


# --------------------------------------------------------------------- driver
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    conf = args.conf_level
    backend = load_backend_constants()
    print(f"Matcher settings imported from {backend['source']}: "
          f"model={backend['model_name']} det_size={backend['det_size']} "
          f"strong={backend['strong_match']} weak={backend['weak_match']}")

    images, labels, names, dmeta = load_dataset(args)
    n_ids = len(names)
    pose = dmeta.get("pose")
    key = cache_key(args, backend, len(images), n_ids, meta=dmeta)
    pack = compute_embeddings(images, backend, args, key)
    valid = pack["valid"]

    if args.face_select == "centred":
        if pack["centred"] is None:
            raise SystemExit(
                "FATAL: --face-select centred needs a cache written with the "
                "centred embeddings. Re-run once with --no-cache.")
        embeddings = pack["centred"]
    else:
        embeddings = pack["best"]

    # How much of the label set is at risk from multi-face frames. On LFW the
    # label names the centred subject, so every image where the highest-
    # det_score face is NOT the centred one is an image whose ground truth is
    # wrong under best-score selection.
    if pack["n_faces"] is not None:
        multi = int(((pack["n_faces"] > 1) & valid).sum())
        disagree = int(((~pack["centred_is_best"]) & valid).sum())
        face_selection = {
            "rule": args.face_select,
            "images_with_multiple_faces": rate(multi, int(valid.sum()), conf),
            "images_where_best_score_face_is_not_centred":
                rate(disagree, int(valid.sum()), conf),
            "note": (
                "LFW labels the centred subject. Under 'best-score' (the rule "
                "encode_faces.py uses) the images counted above are scored "
                "against the wrong person's face, and their errors are ground-"
                "truth defects, not matcher failures. 'centred' removes that "
                "contamination; it changes which detection the label is "
                "attached to, not how any embedding is computed."),
        }
    else:
        face_selection = {"rule": args.face_select,
                          "note": "cache predates multi-face accounting"}

    n_total = len(images)
    n_detected = int(valid.sum())
    n_failed = n_total - n_detected
    det_fail = rate(n_failed, n_total, conf)
    print(f"\nDetection: {n_detected}/{n_total} images produced a face "
          f"(failure rate {fmt_rate(det_fail)})")
    if n_detected < 2:
        raise SystemExit("FATAL: fewer than 2 usable embeddings; cannot evaluate.")

    # ---- degraded arm --------------------------------------------------------
    # A second embedding pass over corrupted copies of the same images. It is a
    # separate cache entry (the `variant` extras key), so the clean cache stays
    # valid and neither arm invalidates the other.
    degraded: dict[str, Any] | None = None
    if args.degrade:
        print(f"\nDegraded arm: {sorted(args.degrade_kinds)} (seed {args.seed})")
        dirty = degrade_images(images, args)
        example_file = save_degrade_examples(
            images, dirty, Path(args.results_dir), args.seed)
        deg_variant = {"degrade_kinds": sorted(args.degrade_kinds),
                       "degrade_version": DEGRADE_VERSION}
        print(f"  family: {args.degrade_family_resolved} "
              f"({', '.join(sorted(args.degrade_kinds))})")
        deg_key = cache_key(args, backend, len(images), n_ids, deg_variant,
                            meta=dmeta)
        deg_pack = compute_embeddings(dirty, backend, args, deg_key)
        del dirty            # ~800 MB at full resolution; the grid is saved
        deg_valid = deg_pack["valid"]
        deg_emb = (deg_pack["centred"] if args.face_select == "centred"
                   and deg_pack["centred"] is not None else deg_pack["best"])
        degraded = {
            "embeddings": deg_emb,
            "valid": deg_valid,
            "kinds": sorted(args.degrade_kinds),
            "family": args.degrade_family_resolved,
            # The full contract, written into metrics.json so a later ablation
            # can be checked against it without re-deriving anything.
            "families": {k: list(v) for k, v in DEGRADE_FAMILIES.items()},
            "reserved_for_augmentation":
                list(DEGRADE_FAMILIES["augmentable"]),
            "held_out_for_evaluation": list(DEGRADE_FAMILIES["heldout"]),
            "augmented_with_declared": sorted(args.augmented_with or []),
            "non_circular": bool(
                args.degrade_family_resolved == "heldout"
                and not (set(args.augmented_with or [])
                         & set(args.degrade_kinds))),
            "contract": (
                "An enrolment augmentation may imitate ONLY the "
                f"{DEGRADE_FAMILIES['augmentable']} family. Evaluation runs on "
                f"the disjoint {DEGRADE_FAMILIES['heldout']} family. Declare "
                "the augmentation's kinds with --augmented-with and the run "
                "aborts on any overlap."),
            "version": DEGRADE_VERSION,
            "cache_key": deg_key,
            "example_grid": example_file,
            "detection_failure_rate": rate(
                int((~deg_valid).sum()), len(deg_valid), conf),
            "detection_lost_vs_clean": rate(
                int((valid & ~deg_valid).sum()), n_detected, conf),
        }
        print(f"  degraded detection failure "
              f"{fmt_rate(degraded['detection_failure_rate'])}; "
              f"lost {fmt_rate(degraded['detection_lost_vs_clean'])} of the "
              "faces the clean arm found")

    probe_idx, gallery_idx, dropped, per_identity = build_splits(
        labels, valid, n_ids, args, pose)
    if len(probe_idx) == 0 or len(gallery_idx) == 0:
        raise SystemExit("FATAL: split produced an empty probe or gallery set.")
    assert not set(probe_idx.tolist()) & set(gallery_idx.tolist()), \
        "probe/gallery overlap - splits are not disjoint"

    probe_emb, probe_lab = embeddings[probe_idx], labels[probe_idx]
    gal_emb, gal_lab = embeddings[gallery_idx], labels[gallery_idx]
    kept_ids = np.array(sorted(per_identity.keys()), dtype=np.int64)
    split_name = ("frontal-enrol / profile-probe (CFP-FP)"
                  if pose is not None and args.cfp_protocol == "fp"
                  else args.split_mode)
    print(f"Split (seed={args.seed}, mode={split_name}): "
          f"{len(probe_idx)} probes / {len(gallery_idx)} gallery over "
          f"{len(kept_ids)} identities"
          + (f"; dropped {len(dropped)} identity(ies) with too few detections"
             if dropped else ""))

    # ================================================================ closed set
    # Every identity enrolled, every probe belongs to someone in the gallery.
    # This is the rank-1/rank-5/CMC view: it measures *ranking*, and asks no
    # accept/reject question at all.
    res = identify(probe_emb, gal_emb, gal_lab, kept_ids)
    sims = res["sims"]
    id_scores, order = res["id_scores"], res["order"]
    col_of = {int(v): i for i, v in enumerate(kept_ids)}
    truth_col = np.array([col_of[int(v)] for v in probe_lab])
    ranks = np.argmax(order == truth_col[:, None], axis=1)      # 0-based rank

    n_probes = len(probe_idx)
    rank1 = rate(int((ranks == 0).sum()), n_probes, conf)
    rank5 = rate(int((ranks < min(5, len(kept_ids))).sum()), n_probes, conf)
    max_rank = min(10, len(kept_ids))
    cmc = [rate(int((ranks < k).sum()), n_probes, conf)
           for k in range(1, max_rank + 1)]

    margins = res["margin"]
    correct = ranks == 0
    margin_correct, margin_wrong = margins[correct], margins[~correct]

    # Same gallery, same identities, degraded probes. Probes the degraded
    # detector missed are excluded rather than scored as errors — a detector
    # miss is reported on its own line, not smuggled into rank-1.
    if degraded is not None:
        keep = degraded["valid"][probe_idx]
        if keep.any():
            d_res = identify(degraded["embeddings"][probe_idx[keep]],
                             gal_emb, gal_lab, kept_ids)
            d_correct = d_res["top1_ident"] == probe_lab[keep]
            degraded["rank1"] = rate(int(d_correct.sum()), int(keep.sum()), conf)
            degraded["rank1_errors"] = rate(int((~d_correct).sum()),
                                            int(keep.sum()), conf)
            degraded["probes_scored"] = int(keep.sum())
            degraded["probes_undetected"] = int((~keep).sum())

    # The same split scored with the other face-selection rule. Re-uses the
    # embeddings already in memory, so it is one matrix product — cheap enough
    # to always report, and it turns "multi-face frames might matter" into a
    # number.
    alt_key = "centred" if args.face_select == "best-score" else "best"
    if pack[alt_key] is not None:
        alt = identify(pack[alt_key][probe_idx], pack[alt_key][gallery_idx],
                       gal_lab, kept_ids)
        alt_correct = alt["top1_ident"] == probe_lab
        face_selection["rank1_under_this_rule"] = rate(
            int(correct.sum()), n_probes, conf)
        face_selection["rank1_under_alternate_rule"] = rate(
            int(alt_correct.sum()), n_probes, conf)
        face_selection["alternate_rule"] = (
            "centred" if alt_key == "centred" else "best-score")

    # ---- statistical power --------------------------------------------------
    p_hat = rank1["rate"]
    n_errors = int((~correct).sum())
    mdd_indep = mdd_unpaired(n_probes, p_hat, args.alpha, args.power)
    min_fixed = mdd_paired_min_errors(args.alpha)

    # The bar can sit above the entire error budget. When it does, no rank-1
    # improvement is measurable on this data at ANY sample size, because there
    # are not enough errors left to fix - the fix is a harder dataset, not more
    # probes. Saying so is more useful than quoting a threshold nobody can meet.
    if n_errors < min_fixed:
        ceiling = (
            f"CEILING: only {n_errors} rank-1 errors exist, fewer than the "
            f"{min_fixed} a paired test needs. No rank-1 improvement is "
            "measurable on this data at any sample size. To measure one, make "
            "the task harder (lower --min-faces for more confusable "
            "identities, a lower --enrol-count, --degrade) rather "
            "than adding probes, or measure something with more headroom - "
            "open-set FAR/FRR at the operating threshold, or detection failure."
        )
    else:
        ceiling = None

    power_block = {
        "confidence_level": conf,
        "alpha": args.alpha,
        "target_power": args.power,
        "n_probes": n_probes,
        "rank1_errors": n_errors,
        "rank1_ci_half_width_pp": rank1["ci_half_width"] * 100.0,
        "mdd_independent_runs_pp": (mdd_indep * 100.0
                                    if mdd_indep is not None else None),
        "mdd_paired_same_probes_pp": min_fixed / n_probes * 100.0,
        "mdd_paired_min_errors_fixed": min_fixed,
        "at_ceiling": ceiling,
        "interpretation": (
            f"With {n_probes} probes and {n_errors} rank-1 errors, "
            f"a change re-run on THESE SAME probes must net-fix at least "
            f"{min_fixed} errors "
            f"({min_fixed / n_probes * 100.0:.2f} pp) to be significant at "
            f"alpha={args.alpha} (exact sign test, best case: it breaks "
            f"nothing). Compared as two independent runs the bar is "
            + (f"{mdd_indep * 100.0:.2f} pp" if mdd_indep is not None
               else "unreachable at this n")
            + f" at {args.power:.0%} power. Anything smaller than that is not "
              "measurable here, however real it is."
        ),
    }

    # ============================================================ verification
    # Balanced pairs for ROC/EER (a ROC over a 1:150 imbalanced set is dominated
    # by the impostor class and its AUC is not comparable to published ones),
    # but the FULL impostor pool for every FAR number, because the whole point
    # of a FAR figure is the tail and balancing throws the tail away.
    rng = np.random.default_rng(args.seed + 2)
    same = probe_lab[:, None] == gal_lab[None, :]
    gen_all = sims[same]
    imp_all = sims[~same]
    imp_full_sorted = np.sort(imp_all)
    gen_full_sorted = np.sort(gen_all)

    n_pairs = int(min(len(gen_all), len(imp_all), args.max_pairs))
    if n_pairs < 2:
        raise SystemExit("FATAL: not enough pairs to evaluate verification.")
    genuine = (rng.choice(gen_all, n_pairs, replace=False)
               if len(gen_all) > n_pairs else gen_all)
    impostor = (rng.choice(imp_all, n_pairs, replace=False)
                if len(imp_all) > n_pairs else imp_all)
    print(f"Verification pairs: {len(genuine)} genuine / {len(impostor)} "
          f"impostor balanced for ROC/EER; FAR estimated on all "
          f"{len(imp_all)} impostor pairs")

    gen_sorted = np.sort(genuine)
    imp_sorted = np.sort(impostor)
    far_res_balanced = 1.0 / len(imp_sorted)
    far_res_full = 1.0 / len(imp_full_sorted)
    if far_res_full > args.target_far:
        print(f"  WARNING: only {len(imp_full_sorted)} impostor pairs even at "
              f"full pool, so the smallest measurable FAR is "
              f"{far_res_full:.5%}, coarser than the {args.target_far:.2%} "
              "target. Use more identities.")

    # ---- ROC / AUC / EER (balanced) ----------------------------------------
    from sklearn.metrics import auc, roc_curve  # noqa: PLC0415

    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    grid = np.unique(np.concatenate([gen_sorted, imp_sorted]))
    far_grid = far_at(grid, imp_sorted)
    frr_grid = frr_at(grid, gen_sorted)
    eer_i = int(np.argmin(np.abs(far_grid - frr_grid)))
    eer = float((far_grid[eer_i] + frr_grid[eer_i]) / 2.0)
    eer_threshold = float(grid[eer_i])

    # ---- FAR/FRR curves on the full pool (the ones that resolve the tail) ---
    full_grid = np.unique(np.concatenate([gen_full_sorted, imp_full_sorted]))
    far_full_grid = far_at(full_grid, imp_full_sorted)
    frr_full_grid = frr_at(full_grid, gen_full_sorted)

    def verification_point(t: float) -> dict[str, Any]:
        arr = np.array([t], dtype=np.float64)
        n_above_full = int(count_at_or_above(arr, imp_full_sorted)[0])
        n_above_bal = int(count_at_or_above(arr, imp_sorted)[0])
        n_below_gen = int(np.searchsorted(gen_full_sorted, arr, side="left")[0])
        return {
            "threshold": float(t),
            "far_full_pool": rate(n_above_full, len(imp_full_sorted), conf),
            "far_balanced": rate(n_above_bal, len(imp_sorted), conf),
            "frr": rate(n_below_gen, len(gen_full_sorted), conf),
        }

    ok = np.flatnonzero(far_full_grid <= args.target_far)
    if len(ok):
        rec_threshold = float(full_grid[int(ok[0])])
    else:
        rec_threshold = float(imp_full_sorted[-1]) + 1e-6
    rec_point = verification_point(rec_threshold)

    current = {
        "strong_match": verification_point(backend["strong_match"]),
        "weak_match": verification_point(backend["weak_match"]),
    }

    # ================================================== open-set identification
    # Two arms (clean / degraded probes) x every enrolment count. The identity
    # partition is seeded per trial and independent of both, so every cell is
    # scored on the same probes against the same identity draw: differences are
    # attributable to the enrolment count or the corruption, and nothing else.
    print(f"\nOpen-set identification: {args.openset_trials} trial(s), "
          f"gallery sizes {args.gallery_sizes}, enrolment counts "
          f"{args.enrol_counts}, {args.impostor_identities} held-out impostor "
          f"identities")

    arm_sources: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "clean": (embeddings, valid)}
    if degraded is not None:
        arm_sources["degraded"] = (degraded["embeddings"], degraded["valid"])

    openset_block: dict[str, Any] = {
        "protocol": (
            "Impostor identities are absent from every gallery; their probes "
            "must all be rejected. For each probe the score is max over the "
            "whole gallery and the prediction is the argmax, exactly as "
            "engine._recognize does. Gallery sizes are nested prefixes of one "
            "identity draw per trial, and enrolment counts are nested prefixes "
            "of each identity's reserved images, so only the swept variable "
            "differs between curves. The gallery is always CLEAN; only probes "
            "are degraded, because enrolment is a deliberate act and matching "
            "is whatever the camera produced."
        ),
        "trials": args.openset_trials,
        "gallery_sizes": None,          # filled from the first run
        "enrol_counts": args.enrol_counts,
        "headline_enrol_count": args.enrol_count,
        "target_far": args.openset_target_far,
        "arms": {},
    }

    os_runs: dict[tuple[str, int], dict[str, Any]] = {}
    sizes: list[int] = []
    deploy_size = args.deploy_gallery_size

    for arm, (probe_src, probe_ok) in arm_sources.items():
        openset_block["arms"][arm] = {"by_enrol_count": {}}
        for n_enrol in args.enrol_counts:
            run = run_openset(embeddings, probe_src, probe_ok, per_identity,
                              names, args, n_enrol)
            os_runs[(arm, n_enrol)] = run
            if not sizes:
                sizes = run["sizes"]
                deploy_size = (args.deploy_gallery_size
                               if args.deploy_gallery_size in sizes
                               else max(sizes))
                openset_block["gallery_sizes"] = sizes
                openset_block["deploy_gallery_size"] = deploy_size
                openset_block["impostor_identities_per_trial"] = \
                    run["n_impostor_identities"]
                openset_block["eligible_identities"] = \
                    run["n_eligible_identities"]
            openset_block["arms"][arm]["by_enrol_count"][str(n_enrol)] = \
                summarise_openset(run, args, backend, conf, rec_threshold,
                                  deploy_size)
            print(f"  {arm:<9} enrol={n_enrol}: threshold "
                  f"{openset_block['arms'][arm]['by_enrol_count'][str(n_enrol)]['threshold']:.4f}"
                  f"  FRR "
                  f"{openset_block['arms'][arm]['by_enrol_count'][str(n_enrol)]['at_deploy_size']['frr']['rate']:.2%}"
                  f" @ gallery {deploy_size}")

    openset_block["note_frr_across_gallery_sizes"] = (
        "FAR is directly comparable across gallery sizes - the same impostor "
        "probes are scored against each gallery. FRR and misidentification are "
        "NOT: the genuine probe set is the probes of the enrolled identities, "
        "so it grows and changes composition with the gallery. Compare FRR "
        "across gallery sizes only at a FIXED threshold (the 'at_recommended' "
        "rows), where a larger gallery can only raise a genuine probe's score "
        "because the score is a max over a superset; a bigger gallery turns "
        "false rejects into correct accepts OR into misidentifications, never "
        "the reverse. The threshold_for_target_far rows move for two reasons "
        "at once and should not be read as a gallery-size effect on FRR."
    )

    openset_block["note_independence"] = (
        "Rates are pooled over trials. The trials redraw which identities are "
        "enrolled and which are impostors, so pooled decisions are mostly "
        "distinct probe images - but not fully independent, because an image "
        "can be scored in more than one trial against a different gallery. "
        "Read far_resolution_per_trial, not far_resolution_pooled, as the "
        "honest floor on a single FAR measurement; the Wilson intervals on "
        "pooled counts are correspondingly a little optimistic."
    )

    # ---- the number that goes into engine.py --------------------------------
    # Clean arm, headline enrolment count, deploy gallery size. Clean rather
    # than degraded because the degradation is a stress test with invented
    # severity, not a measured model of this deployment's camera; the degraded
    # arm bounds how much worse it gets, it does not set the threshold.
    headline = openset_block["arms"]["clean"]["by_enrol_count"][
        str(args.enrol_count)]
    os_threshold = headline["threshold"]
    os_res = os_runs[("clean", args.enrol_count)]

    openset_block["recommendation"] = {
        "threshold": os_threshold,
        "assumed_gallery_size": deploy_size,
        "assumed_enrolment_count": args.enrol_count,
        "assumed_arm": "clean",
        "assumed_note": (
            f"Chosen on the {deploy_size}-identity curve at "
            f"{args.enrol_count} enrolment image(s) per person - what "
            "engine.register actually stores. The deployed gallery starts far "
            "smaller, so this threshold is conservative today and stays valid "
            f"as enrolment grows to {deploy_size} identities. Above that, "
            "re-run the sweep: open-set FAR rises with gallery size and this "
            "threshold is not extrapolated. Enrolling MORE images per person "
            "moves it too - see arms.clean.by_enrol_count."
        ),
        "resolved_within_target": headline["target_far_resolved"],
        # A target finer than one impostor probe cannot be measured, only
        # reached by running out of impostors: the "FAR = 0%" it reports is
        # the absence of evidence, and the threshold is wherever the single
        # highest-scoring impostor happened to land. Flag it loudly rather
        # than shipping a number that looks like a measurement.
        "under_resolved": headline["under_resolved"],
        "impostor_probes_per_trial": headline["impostor_probes_per_trial"],
        "at_deploy_size": headline["at_deploy_size"],
        "cost_at_each_size": {
            str(s): headline["by_gallery_size"][str(s)]["at_recommended"]
            for s in sizes
        },
        "cost_under_degraded_probes": (
            openset_block["arms"]["degraded"]["by_enrol_count"][
                str(args.enrol_count)]["by_gallery_size"][str(deploy_size)][
                    "at_recommended"]
            if "degraded" in arm_sources else None),
    }

    # ---- enrolment-count sweep, flattened for reading ------------------------
    # The whole point of gap 1: what does enrolling one frame instead of three
    # actually cost? One row per (arm, enrolment count, gallery size).
    enrol_sweep = []
    for arm in arm_sources:
        for n_enrol in args.enrol_counts:
            blk = openset_block["arms"][arm]["by_enrol_count"][str(n_enrol)]
            for size in sizes:
                g = blk["by_gallery_size"][str(size)]
                op = g["at_threshold_for_target_far"]
                enrol_sweep.append({
                    "arm": arm,
                    "enrol_count": n_enrol,
                    "gallery_size": size,
                    "threshold_for_target_far": g["threshold_for_target_far"],
                    "far": op["far"],
                    "frr": op["frr"],
                    "misidentification": op["misidentification"],
                    "n_genuine_probes": g["n_genuine_probes"],
                    "n_impostor_probes": g["n_impostor_probes"],
                })
    openset_block["enrolment_sweep"] = enrol_sweep

    # ================================================= the augmentation target
    # Rank-1 is saturated, so the question is which OTHER metric still has
    # enough errors left to show a significant improvement. Every candidate is
    # measured at the deployed configuration: 1 enrolment image, gallery 50.
    min_fixed = mdd_paired_min_errors(args.alpha)
    primary: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []

    for arm in arm_sources:
        blk = openset_block["arms"][arm]["by_enrol_count"][str(args.enrol_count)]
        op = blk["at_deploy_size"]
        # Measuring a synthetic-augmentation change against a synthetic
        # corruption is circular only when both draw on the SAME effects. The
        # held-out family exists to break exactly that: the augmentation is
        # confined to `augmentable`, evaluation runs on the disjoint `heldout`
        # group, and resolve_defaults() aborts on any declared overlap. So the
        # warning applies to the mixed and augmentable arms, not to heldout.
        risk = None
        if arm == "degraded" and not degraded.get("non_circular"):
            risk = (
                "CIRCULARITY WARNING: probes were degraded with the "
                f"'{degraded.get('family')}' family "
                f"({', '.join(degraded.get('kinds', []))}), which is not the "
                "held-out evaluation group. An augmentation that imitates any "
                "of these will improve against them trivially. Re-run with "
                "--degrade-family heldout before using this as an ablation "
                "metric.")

        # NB: not `key` — that name holds the embedding cache key in this
        # scope, and shadowing it silently wrote "far" into metrics.json.
        for label, field in (
            (f"open-set FRR @ FAR<={args.openset_target_far:.2%} ({arm})",
             "frr"),
            (f"misidentification @ FAR<={args.openset_target_far:.2%} ({arm})",
             "misidentification"),
            (f"open-set FAR @ recommended threshold ({arm})", "far"),
        ):
            candidates.append({
                "name": label,
                "arm": arm,
                "rate": op[field],
                "affected_by_enrolment": True,
                "circularity_risk": risk,
                "mdd": mdd_for(op[field], min_fixed, args.alpha, args.power),
            })
        if arm == "degraded":
            candidates.append({
                "name": "probe detection failure (degraded)",
                "arm": arm,
                "rate": blk["detection_failure_on_probes"],
                # A detector miss happens before any gallery comparison, so a
                # better enrolment cannot fix it.
                "affected_by_enrolment": False,
                "mdd": mdd_for(blk["detection_failure_on_probes"], min_fixed,
                               args.alpha, args.power),
            })
        primary[arm] = {
            "metric": f"open-set FRR at FAR <= {args.openset_target_far:.2%}",
            "enrol_count": args.enrol_count,
            "gallery_size": deploy_size,
            "threshold": blk["threshold"],
            "frr": op["frr"],
            "far": op["far"],
            "misidentification": op["misidentification"],
            "detection_failure_on_probes": blk["detection_failure_on_probes"],
            "end_to_end_miss_rate": blk["end_to_end_miss_rate"],
            "mdd": mdd_for(op["frr"], min_fixed, args.alpha, args.power),
        }

    candidates.append({
        "name": "closed-set rank-1 error (clean)",
        "arm": "clean",
        "rate": rate(n_errors, n_probes, conf),
        "affected_by_enrolment": True,
        "mdd": mdd_for(rate(n_errors, n_probes, conf), min_fixed, args.alpha,
                       args.power),
    })
    if degraded is not None and "rank1" in degraded:
        candidates.append({
            "name": "closed-set rank-1 error (degraded)",
            "arm": "degraded",
            "rate": degraded["rank1_errors"],
            "affected_by_enrolment": True,
            "circularity_risk": next(
                (c["circularity_risk"] for c in candidates
                 if c["arm"] == "degraded" and c.get("circularity_risk")), None),
            "mdd": mdd_for(degraded["rank1_errors"], min_fixed, args.alpha,
                           args.power),
        })

    augmentation = augmentation_headroom(candidates, min_fixed)
    augmentation["primary_metric_by_arm"] = primary

    # ---- THE ablation metric ------------------------------------------------
    # Open-set FRR at fixed FAR, N=1 enrolment, deploy gallery size, measured on
    # the held-out corruption family. Named explicitly rather than left to be
    # inferred from the candidate table, so a future ablation reads one field.
    if degraded is not None and degraded.get("non_circular"):
        blk = openset_block["arms"]["degraded"]["by_enrol_count"][
            str(args.enrol_count)]
        op = blk["at_deploy_size"]
        augmentation["ablation_primary_metric"] = {
            "metric": (f"open-set FRR at FAR <= {args.openset_target_far:.2%}, "
                       "held-out corruption family"),
            "arm": "degraded",
            "degrade_family": degraded["family"],
            "degrade_kinds": degraded["kinds"],
            "reserved_for_augmentation": degraded["reserved_for_augmentation"],
            "enrol_count": args.enrol_count,
            "gallery_size": deploy_size,
            "threshold": blk["threshold"],
            "far_held_at": op["far"],
            "frr": op["frr"],
            "misidentification": op["misidentification"],
            "detection_failure_on_probes": blk["detection_failure_on_probes"],
            "end_to_end_miss_rate": blk["end_to_end_miss_rate"],
            "mdd": mdd_for(op["frr"], min_fixed, args.alpha, args.power),
            "supports": (
                "Evidence that an enrolment augmentation confined to "
                f"{degraded['reserved_for_augmentation']} bought robustness "
                "that generalises to corruptions it never saw."),
            "does_not_support": (
                "Any claim about real pose variation, real cameras, or this "
                "specific corruption being the one that matters. The "
                "corruptions are synthetic and their severity is invented; "
                "pose in particular is only a planar warp here. Use CFP-FP or "
                "real captures for those claims."),
        }
    else:
        augmentation["ablation_primary_metric"] = None
    augmentation["threshold_caveat"] = (
        "FRR is reported at a threshold re-derived to hold FAR at "
        f"{args.openset_target_far:.2%}. If a change moves the score "
        "distribution, that threshold moves too, so the comparison is not "
        "strictly paired at the decision level. The paired MDD above assumes a "
        "FIXED threshold; when re-deriving it, quote the unpaired bar instead, "
        "or hold the threshold at the current value and report FAR alongside "
        "FRR so both halves of the trade are visible."
    )

    # ---------------------------------------------------------------- failures
    failures_dir = Path(args.results_dir) / "failures"
    failure_files: dict[str, Any] = {}

    n_faces = pack["n_faces"]
    closed_records = collect_failures(
        probe_idx, probe_lab,
        kept_ids[order[:, 0]], id_scores[np.arange(n_probes), order[:, 0]],
        margins, gallery_idx[res["best_entry"]], ~correct, names, n_faces)
    failure_files["closed_set_rank1_errors"] = {
        "n": len(closed_records),
        "plotted": min(len(closed_records), args.max_failure_images),
        "files": save_failure_grid(
            images, closed_records[:args.max_failure_images], failures_dir,
            "closed_set_rank1_errors",
            f"Closed-set rank-1 errors ({len(closed_records)} of {n_probes} "
            f"probes, {len(kept_ids)}-identity gallery)"),
    }

    rec = os_res["curves"][deploy_size]["records"]
    if rec:
        fa_mask = (~rec["is_genuine"]) & (rec["score"] >= os_threshold)
        mi_mask = rec["is_genuine"] & (rec["score"] >= os_threshold) & ~rec["correct"]
        fr_mask = rec["is_genuine"] & (rec["score"] < os_threshold)

        fa_records = collect_failures(
            rec["probe_index"], rec["true_ident"], rec["pred_ident"],
            rec["score"], rec["margin"], rec["match_index"], fa_mask, names,
            n_faces, impostor=True)
        mi_records = collect_failures(
            rec["probe_index"], rec["true_ident"], rec["pred_ident"],
            rec["score"], rec["margin"], rec["match_index"], mi_mask, names,
            n_faces)
        fr_records = collect_failures(
            rec["probe_index"], rec["true_ident"], rec["pred_ident"],
            rec["score"], rec["margin"], rec["match_index"], fr_mask, names,
            n_faces)

        for slug, records, title in (
            ("openset_false_accepts", fa_records,
             f"Open-set FALSE ACCEPTS - impostor identities cleared "
             f"thr {os_threshold:.3f} (gallery {deploy_size}, trial 0)"),
            ("openset_misidentifications", mi_records,
             f"Open-set MISIDENTIFICATIONS - enrolled person, wrong name, "
             f"cleared thr {os_threshold:.3f} (gallery {deploy_size}, trial 0)"),
            ("openset_false_rejects", fr_records,
             f"Open-set FALSE REJECTS - enrolled person below "
             f"thr {os_threshold:.3f} (gallery {deploy_size}, trial 0)"),
        ):
            failure_files[slug] = {
                "n": len(records),
                "plotted": min(len(records), args.max_failure_images),
                "files": save_failure_grid(
                    images, records[:args.max_failure_images], failures_dir,
                    slug, title),
            }

        manifest = {
            "closed_set_rank1_errors": closed_records,
            "openset_false_accepts": fa_records,
            "openset_misidentifications": mi_records,
            "openset_false_rejects": fr_records,
        }
    else:
        manifest = {"closed_set_rank1_errors": closed_records}

    failures_dir.mkdir(parents=True, exist_ok=True)
    (failures_dir / "failures.json").write_text(
        json.dumps({"threshold": os_threshold,
                    "deploy_gallery_size": deploy_size,
                    "note": ("Every failure is listed here; the PNG grids show "
                             f"at most {args.max_failure_images} per category, "
                             "highest similarity first."),
                    "failures": manifest}, indent=2) + "\n",
        encoding="utf-8")
    print(f"Failure grids written to {failures_dir}/")

    # ------------------------------------------------------------- assemble
    metrics: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "config": {
            "dataset": args.dataset,
            "dataset_source": dmeta.get("source"),
            "dataset_content_hash": dmeta.get("content_hash"),
            "cfp_protocol": (args.cfp_protocol if args.dataset == "cfp-fp"
                             else None),
            "cfp_max_side": (args.cfp_max_side if args.dataset == "cfp-fp"
                             else None),
            "min_faces_per_person": args.min_faces,
            "resize": args.resize,
            "color": True,
            "slice": args.slice,
            "max_identities": args.max_identities,
            "max_images_per_identity": args.max_images_per_identity,
            "face_select": args.face_select,
            "split_mode": args.split_mode,
            "enrol_count": args.enrol_count,
            "enrol_counts": args.enrol_counts,
            "degrade": args.degrade,
            "degrade_family": args.degrade_family_resolved,
            "degrade_kinds": sorted(args.degrade_kinds),
            "augmented_with": sorted(args.augmented_with or []),
            "probes_per_identity": args.probes_per_identity,
            "max_probes_per_identity": args.max_probes_per_identity,
            "max_pairs": args.max_pairs,
            "target_far": args.target_far,
            "openset_target_far": args.openset_target_far,
            "gallery_sizes": args.gallery_sizes,
            "openset_trials": args.openset_trials,
            "impostor_identities": args.impostor_identities,
            "deploy_gallery_size": args.deploy_gallery_size,
            "conf_level": conf,
            "alpha": args.alpha,
            "power": args.power,
            "embedding_cache_key": key,
        },
        "backend_settings": backend,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "dataset": {
            "name": dmeta["name"],
            "source": dmeta.get("source"),
            "reference": dmeta.get("reference"),
            "content_hash": dmeta.get("content_hash"),
            "images_frontal": dmeta.get("images_frontal"),
            "images_profile": dmeta.get("images_profile"),
            "split": split_name,
            "identities_loaded": n_ids,
            "identities_evaluated": len(kept_ids),
            "identities_dropped_too_few_detections": len(dropped),
            "images_total": n_total,
            "images_with_face_detected": n_detected,
            "images_detection_failed": n_failed,
            "detection_failure_rate": det_fail,
            "probe_count": int(len(probe_idx)),
            "gallery_count": int(len(gallery_idx)),
        },
        "face_selection": face_selection,
        "verification": {
            "note": ("Pairwise same/different. Summarises matcher quality and "
                     "is the right thing to compare across model changes. It "
                     "does NOT set the app's threshold - see "
                     "openset_identification."),
            "genuine_pairs_balanced": int(len(genuine)),
            "impostor_pairs_balanced": int(len(impostor)),
            "genuine_pairs_full_pool": int(len(gen_all)),
            "impostor_pairs_full_pool": int(len(imp_all)),
            "impostor_pairs_discarded_by_balancing":
                int(len(imp_all) - len(impostor)),
            "far_resolution_balanced": far_res_balanced,
            "far_resolution_full_pool": far_res_full,
            "roc_auc": roc_auc,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "genuine_mean": float(gen_all.mean()),
            "genuine_std": float(gen_all.std()),
            "impostor_mean": float(imp_all.mean()),
            "impostor_std": float(imp_all.std()),
            "recommended_threshold": {
                "target_far": args.target_far,
                **rec_point,
            },
            "current_hardcoded_thresholds": current,
        },
        "identification_closed_set": {
            "note": ("Every identity enrolled; measures ranking only, no "
                     "accept/reject decision."),
            "rank1_accuracy": rank1,
            "rank5_accuracy": rank5,
            "cmc": cmc,
            "probes": n_probes,
            "gallery_identities": int(len(kept_ids)),
        },
        "openset_identification": openset_block,
        "degradation": (
            {k: v for k, v in degraded.items()
             if k not in ("embeddings", "valid")}
            if degraded is not None else None),
        "augmentation_target": augmentation,
        "statistical_power": power_block,
        "top2_margin": {
            "correct_rank1": {
                "n": int(len(margin_correct)),
                "mean": float(margin_correct.mean()) if len(margin_correct) else None,
                "median": float(np.median(margin_correct)) if len(margin_correct) else None,
                "p05": float(np.percentile(margin_correct, 5)) if len(margin_correct) else None,
            },
            "incorrect_rank1": {
                "n": int(len(margin_wrong)),
                "mean": float(margin_wrong.mean()) if len(margin_wrong) else None,
                "median": float(np.median(margin_wrong)) if len(margin_wrong) else None,
                "p95": float(np.percentile(margin_wrong, 95)) if len(margin_wrong) else None,
            },
        },
        "failure_grids": failure_files,
    }

    _make_plots(args, full_grid, far_full_grid, frr_full_grid, fpr, tpr,
                roc_auc, eer, eer_threshold, gen_all, imp_all, margin_correct,
                margin_wrong, [c["rate"] for c in cmc], rec_threshold,
                os_threshold, backend)
    _make_enrolment_plots(args, openset_block, sizes, deploy_size,
                          args.results_dir)
    _make_openset_plots(args, os_res, sizes, deploy_size, os_threshold,
                        backend, rec_threshold)
    return metrics


# ----------------------------------------------------------------------- plots
def _make_plots(args, grid, far_grid, frr_grid, fpr, tpr, roc_auc, eer,
                eer_threshold, genuine, impostor, margin_correct, margin_wrong,
                cmc, rec_threshold, os_threshold, backend) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    strong, weak = backend["strong_match"], backend["weak_match"]

    # 1. ROC with EER marked
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="chance")
    ax.plot([eer], [1 - eer], "o", ms=9, color="crimson",
            label=f"EER = {eer:.3%} @ thr {eer_threshold:.3f}")
    ax.set_xlabel("False accept rate (FAR)")
    ax.set_ylabel("True accept rate (1 - FRR)")
    ax.set_title("Verification ROC (balanced pairs)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=150)
    plt.close(fig)

    # 2. FAR / FRR vs threshold, FAR over the FULL impostor pool
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(grid, far_grid, lw=2,
            label=f"FAR, full pool (n={len(impostor):,} pairs)")
    ax.plot(grid, frr_grid, lw=2, label=f"FRR (n={len(genuine):,} pairs)")
    ax.axhline(1.0 / len(impostor), color="grey", ls=":", lw=1,
               label=f"1-pair floor = {1.0 / len(impostor):.2}")
    ax.axvline(eer_threshold, color="crimson", ls=":", lw=1.5,
               label=f"EER thr {eer_threshold:.3f}")
    ax.axvline(rec_threshold, color="green", ls="-.", lw=1.5,
               label=f"verification rec. {rec_threshold:.3f}")
    ax.axvline(os_threshold, color="black", ls="-", lw=1.5,
               label=f"OPEN-SET rec. {os_threshold:.3f}")
    ax.axvline(strong, color="darkorange", ls="--", lw=1.5,
               label=f"current strong {strong}")
    ax.axvline(weak, color="purple", ls="--", lw=1.5,
               label=f"current weak {weak}")
    ax.set_yscale("log")
    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("Error rate (log scale)")
    ax.set_title("Verification FAR / FRR vs. threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "far_frr_vs_threshold.png", dpi=150)
    plt.close(fig)

    # 3. Score distributions
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(min(impostor.min(), genuine.min()),
                       max(impostor.max(), genuine.max()), 80)
    ax.hist(impostor, bins=bins, alpha=.6, label=f"impostor (n={len(impostor):,})",
            color="steelblue", density=True)
    ax.hist(genuine, bins=bins, alpha=.6, label=f"genuine (n={len(genuine):,})",
            color="seagreen", density=True)
    ax.axvline(strong, color="darkorange", ls="--", lw=1.5,
               label=f"current strong {strong}")
    ax.axvline(weak, color="purple", ls="--", lw=1.5,
               label=f"current weak {weak}")
    ax.axvline(os_threshold, color="black", ls="-", lw=1.5,
               label=f"open-set rec. {os_threshold:.3f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Genuine vs. impostor score distributions (full pool)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "score_distributions.png", dpi=150)
    plt.close(fig)

    # 4. CMC curve
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ks = np.arange(1, len(cmc) + 1)
    ax.plot(ks, cmc, "o-", lw=2)
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Identification accuracy")
    ax.set_title("CMC curve (rank-1 = "
                 f"{cmc[0]:.2%}, rank-5 = {cmc[min(4, len(cmc) - 1)]:.2%})")
    ax.set_xticks(ks)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "cmc_curve.png", dpi=150)
    plt.close(fig)

    # 5. Top-2 margin, correct vs incorrect rank-1
    fig, ax = plt.subplots(figsize=(7, 5))
    allm = np.concatenate([m for m in (margin_correct, margin_wrong) if len(m)])
    bins = np.linspace(0, float(allm.max()) if len(allm) else 1.0, 60)
    if len(margin_correct):
        ax.hist(margin_correct, bins=bins, alpha=.6, density=True,
                color="seagreen", label=f"correct rank-1 (n={len(margin_correct)})")
    if len(margin_wrong):
        ax.hist(margin_wrong, bins=bins, alpha=.6, density=True,
                color="indianred", label=f"incorrect rank-1 (n={len(margin_wrong)})")
    ax.set_xlabel("Top-2 margin (best identity score - runner-up)")
    ax.set_ylabel("Density")
    ax.set_title("Top-2 margin: correct vs. incorrect rank-1")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "top2_margin.png", dpi=150)
    plt.close(fig)

    print(f"Plots written to {out}/")


def _make_openset_plots(args, os_res, sizes, deploy_size, os_threshold,
                        backend, verif_threshold) -> None:
    """The plots that actually justify the operating threshold."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out = Path(args.results_dir)
    strong, weak = backend["strong_match"], backend["weak_match"]
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))

    # 6. FAR / FRR / MIR vs threshold, one line per gallery size
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    panels = [
        ("far", "False accept rate\n(impostor identity accepted)", True),
        # symlog on FRR too: the operating region is 0-5%, which a linear
        # axis spanning 0-100% renders as a flat line on the floor.
        ("frr", "False reject rate\n(enrolled person rejected)", True),
        ("mir", "Misidentification rate\n(accepted as the WRONG person)", True),
    ]
    for ax, (field, ylabel, logy) in zip(axes, panels):
        for colour, size in zip(colours, sizes):
            c = os_res["curves"][size]
            ax.plot(c["grid"], c[field], lw=1.8, color=colour,
                    label=f"gallery = {size} identities "
                          f"(n={c['n_impostor'] if field == 'far' else c['n_genuine']:,})")
        if logy:
            # symlog, not log: these rates legitimately hit exactly zero.
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_ylim(bottom=0)
        ax.axvline(os_threshold, color="black", ls="-", lw=1.4,
                   label=f"recommended {os_threshold:.3f}")
        ax.axvline(strong, color="darkorange", ls="--", lw=1.2,
                   label=f"current strong {strong}")
        ax.axvline(weak, color="purple", ls="--", lw=1.2,
                   label=f"current weak {weak}")
        ax.axhline(args.openset_target_far, color="grey", ls=":", lw=1)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=.3, which="both")
        ax.legend(fontsize=7, loc="best")
    axes[0].set_title("Open-set identification: the protocol the app runs\n"
                      "(max over gallery, argmax, accept above threshold)")
    axes[-1].set_xlabel("Cosine similarity threshold")
    axes[-1].set_xlim(-0.05, 0.95)
    fig.tight_layout()
    fig.savefig(out / "openset_far_frr_mir.png", dpi=150)
    plt.close(fig)

    # 7. How the operating point moves as the gallery grows
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for label, thr, style in (
        (f"recommended {os_threshold:.3f}", os_threshold, "o-"),
        (f"current strong {strong}", strong, "s--"),
        (f"current weak {weak}", weak, "^--"),
        (f"verification rec. {verif_threshold:.3f}", verif_threshold, "d-."),
    ):
        fars = []
        for size in sizes:
            c = os_res["curves"][size]
            n_above = int(count_at_or_above(thr, np.sort(c["imp_score"]))[0])
            fars.append(n_above / max(1, c["n_impostor"]))
        ax1.plot(sizes, fars, style, lw=1.8, ms=6, label=label)
    ax1.set_xlabel("Gallery size (enrolled identities)")
    ax1.set_ylabel("Open-set FAR")
    ax1.set_yscale("symlog", linthresh=1e-4)
    ax1.set_ylim(bottom=0)          # a rate has no negative decades
    ax1.axhline(args.openset_target_far, color="grey", ls=":", lw=1,
                label=f"target FAR {args.openset_target_far:.2%}")
    ax1.set_title("Open-set FAR at a fixed threshold,\nvs. gallery size",
                  fontsize=11)
    ax1.set_xticks(sizes)
    ax1.grid(alpha=.3, which="both")
    ax1.legend(fontsize=7)

    need = []
    frrs = []
    for size in sizes:
        c = os_res["curves"][size]
        hit = np.flatnonzero(c["far"] <= args.openset_target_far)
        thr = (float(c["grid"][int(hit[0])]) if len(hit)
               else float(np.max(c["imp_score"])) + 1e-6)
        need.append(thr)
        frrs.append(float(np.mean(c["gen_score"] < thr)))
    ax2.plot(sizes, need, "o-", lw=2, color="black",
             label=f"threshold for FAR <= {args.openset_target_far:.2%}")
    ax2.set_xlabel("Gallery size (enrolled identities)")
    ax2.set_ylabel("Required threshold", color="black")
    ax2.set_xticks(sizes)
    ax2.grid(alpha=.3)
    ax2b = ax2.twinx()
    ax2b.plot(sizes, frrs, "s--", lw=1.6, color="crimson",
              label="FRR paid there")
    ax2b.set_ylabel("FRR at that threshold", color="crimson")
    ax2.set_title(f"Threshold needed to hold FAR <= "
                  f"{args.openset_target_far:.2%},\nand the FRR it costs",
                  fontsize=11)
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [ln.get_label() for ln in lines], fontsize=7,
               loc="best")
    fig.tight_layout()
    fig.savefig(out / "openset_gallery_size.png", dpi=150)
    plt.close(fig)

    print(f"Open-set plots written to {out}/")


def _make_enrolment_plots(args, openset_block, sizes, deploy_size,
                          results_dir) -> None:
    """What enrolling one frame instead of several actually costs.

    Left: the threshold each enrolment count needs to hold the FAR budget.
    Right: the FRR it pays there. One line per arm x gallery size, because the
    interesting question is whether the penalty for N=1 is uniform or gets
    worse as the gallery grows.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out = Path(results_dir)
    arms = list(openset_block["arms"])
    counts = openset_block["enrol_counts"]
    styles = {"clean": "-", "degraded": "--"}
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for arm in arms:
        by_n = openset_block["arms"][arm]["by_enrol_count"]
        for colour, size in zip(colours, sizes):
            thr = [by_n[str(n)]["by_gallery_size"][str(size)]
                   ["threshold_for_target_far"] for n in counts]
            frr = [by_n[str(n)]["by_gallery_size"][str(size)]
                   ["at_threshold_for_target_far"]["frr"]["rate"]
                   for n in counts]
            label = f"{arm}, gallery {size}"
            ax1.plot(counts, thr, styles[arm], marker="o", color=colour,
                     lw=1.8, ms=5, label=label)
            ax2.plot(counts, frr, styles[arm], marker="o", color=colour,
                     lw=1.8, ms=5, label=label)

    ax1.set_ylabel(f"Threshold holding FAR <= {args.openset_target_far:.2%}")
    ax2.set_ylabel("FRR paid at that threshold")
    ax2.set_yscale("symlog", linthresh=1e-3)
    ax2.set_ylim(bottom=0)
    for ax, title in ((ax1, "Threshold vs. enrolment images per person"),
                      (ax2, "False rejects vs. enrolment images per person")):
        ax.set_xlabel("Enrolment images per identity")
        ax.set_xticks(counts)
        ax.axvline(args.enrol_count, color="black", ls=":", lw=1.2,
                   label=f"app enrols {args.enrol_count}")
        ax.grid(alpha=.3, which="both")
        ax.set_title(title, fontsize=11)
    ax2.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out / "enrolment_count.png", dpi=150)
    plt.close(fig)

    # Clean vs degraded FRR/FAR curves at the deployed configuration.
    if len(arms) > 1:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for arm in arms:
            blk = openset_block["arms"][arm]["by_enrol_count"][
                str(args.enrol_count)]
            g = blk["by_gallery_size"][str(deploy_size)]
            ax.plot([g["at_threshold_for_target_far"]["far"]["rate"]],
                    [g["at_threshold_for_target_far"]["frr"]["rate"]],
                    "o", ms=11, label=f"{arm}: thr "
                    f"{g['threshold_for_target_far']:.3f}, FRR "
                    f"{g['at_threshold_for_target_far']['frr']['rate']:.2%}")
        for arm in arms:
            blk = openset_block["arms"][arm]["by_enrol_count"][
                str(args.enrol_count)]
            det = blk["detection_failure_on_probes"]["rate"]
            e2e = blk["end_to_end_miss_rate"]["rate"]
            ax.annotate(f"{arm}: detector misses {det:.2%}\n"
                        f"end-to-end miss {e2e:.2%}",
                        xy=(0.03, 0.9 - 0.12 * arms.index(arm)),
                        xycoords="axes fraction", fontsize=9)
        ax.set_xlabel("Open-set FAR")
        ax.set_ylabel("Open-set FRR")
        ax.set_title(f"Clean vs. degraded probes at {args.enrol_count} "
                     f"enrolment image(s), gallery {deploy_size}")
        ax.grid(alpha=.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "degradation_comparison.png", dpi=150)
        plt.close(fig)

    print(f"Enrolment / degradation plots written to {out}/")


# ------------------------------------------------------------------------ main
def _int_list(text: str) -> list[int]:
    """Comma-separated positive integers, deduplicated and sorted."""
    try:
        values = sorted({int(v) for v in text.split(",") if v.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of integers ({exc})") from exc
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("needs at least one value >= 1")
    return values


def _gallery_sizes(text: str) -> list[int]:
    sizes = _int_list(text)
    if min(sizes) < 2:
        raise argparse.ArgumentTypeError(
            "--gallery-sizes needs at least one value >= 2")
    return sizes


def _degrade_kinds(text: str) -> list[str]:
    kinds = [v.strip() for v in text.split(",") if v.strip()]
    unknown = sorted(set(kinds) - set(DEGRADE_KINDS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown degradation(s) {unknown}; choose from "
            f"{list(DEGRADE_KINDS)}")
    if not kinds:
        raise argparse.ArgumentTypeError("needs at least one degradation")
    return kinds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure face-recognition accuracy against the backend's "
                    "live model and thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- dataset ------------------------------------------------------------
    p.add_argument("--dataset", choices=("lfw", "cfp-fp"), default="lfw",
                   help="'lfw' is the saturated frontal baseline; 'cfp-fp' is "
                        "frontal-vs-profile and is where enrolment changes "
                        "have room to show an effect")
    p.add_argument("--cfp-root", default=None,
                   help="path to the extracted cfp-dataset folder (or its "
                        "Data/ or Data/Images/ subdirectory). Required for "
                        "--dataset cfp-fp")
    p.add_argument("--cfp-protocol", choices=("fp", "mixed"), default="fp",
                   help="'fp' enrols frontal images and probes with profile "
                        "ones - the CFP-FP protocol, and the deployment "
                        "analogue; 'mixed' ignores pose and splits like LFW")
    p.add_argument("--cfp-max-side", type=int, default=0,
                   help="downscale CFP images whose long side exceeds this "
                        "(0 = leave them alone). Part of the cache key")
    p.add_argument("--min-faces", type=int, default=10,
                   help="fetch_lfw_people(min_faces_per_person=...). 10 gives "
                        "158 identities / 4324 images; 20 gives 62 / 3023")
    p.add_argument("--resize", type=float, default=1.0,
                   help="fetch_lfw_people(resize=...)")
    p.add_argument("--slice", choices=("full", "tight"), default="full",
                   help="'full' uses the whole 250x250 LFW frame; 'tight' uses "
                        "sklearn's default face crop, on which the detector "
                        "finds nothing (0%% detection) - kept for reproducing "
                        "that result")
    p.add_argument("--max-identities", type=int, default=None,
                   help="subsample this many identities (default: all)")
    p.add_argument("--max-images-per-identity", type=int, default=None,
                   help="cap images per identity to speed up embedding")
    p.add_argument("--face-select", choices=("centred", "best-score"),
                   default="centred",
                   help="which detection carries the image's label when a "
                        "frame contains several faces. 'centred' matches LFW "
                        "ground truth, which names the centred subject; "
                        "'best-score' is encode_faces.py's rule and mislabels "
                        "2%% of images, which costs 2.1 pp of rank-1. Both are "
                        "computed in one pass and share a cache entry")
    p.add_argument("--det-score-min", type=float, default=0.0,
                   help="discard detections below this det_score (0 = keep "
                        "the best face whatever its score, as encode_faces "
                        "does). Non-zero values extend the cache key")

    # -- split --------------------------------------------------------------
    p.add_argument("--split-mode", choices=("gallery-first", "probe-first"),
                   default="gallery-first",
                   help="'gallery-first' enrols --enrol-count images "
                        "and probes with all the rest (mirrors the app, and "
                        "maximises probe count); 'probe-first' is the original "
                        "--probes-per-identity behaviour")
    p.add_argument("--enrol-count", "--gallery-per-identity", type=int,
                   default=1, dest="enrol_count",
                   help="gallery-first: images enrolled per identity for the "
                        "headline result. Default 1 because engine.register "
                        "stores exactly one webcam frame")
    p.add_argument("--enrol-counts", type=_int_list, default=[1, 3, 5],
                   help="comma-separated enrolment counts to sweep. The "
                        "largest is reserved per identity so all counts are "
                        "scored on the same probes")
    p.add_argument("--max-probes-per-identity", type=int, default=None,
                   help="gallery-first: cap probes per identity (default: all)")
    p.add_argument("--probes-per-identity", type=int, default=5,
                   help="probe-first: images held out per identity as probes")

    # -- verification -------------------------------------------------------
    p.add_argument("--max-pairs", type=int, default=25000,
                   help="cap per class for the balanced ROC/EER pairs. FAR is "
                        "always computed on the full impostor pool regardless")
    p.add_argument("--target-far", type=float, default=0.001,
                   help="FAR budget for the verification recommended threshold")

    # -- open-set identification -------------------------------------------
    p.add_argument("--gallery-sizes", type=_gallery_sizes, default=[10, 25, 50],
                   help="comma-separated gallery sizes to sweep, in identities")
    p.add_argument("--impostor-identities", type=int, default=None,
                   help="identities held out of every gallery; their probes "
                        "must all be rejected. This number, times their probe "
                        "count, is what resolves the open-set FAR tail. "
                        "Defaults to 80 on LFW and 400 on CFP-FP, which has "
                        "500 identities but only 4 profile probes each")
    p.add_argument("--openset-trials", type=int, default=5,
                   help="independent identity draws; results are pooled")
    p.add_argument("--openset-target-far", type=float, default=0.001,
                   help="open-set FAR budget for the recommended threshold")
    p.add_argument("--deploy-gallery-size", type=int, default=50,
                   help="gallery size the recommended threshold assumes; must "
                        "be one of --gallery-sizes (else the largest is used)")

    # -- degraded arm -------------------------------------------------------
    p.add_argument("--degrade", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="also score a degraded-probe arm approximating webcam "
                        "capture. Costs one extra embedding pass, cached "
                        "separately from the clean one")
    p.add_argument("--degrade-family",
                   choices=("heldout", "augmentable", "all"), default="heldout",
                   help="which corruption family degrades the probes. "
                        f"'heldout' = {','.join(DEGRADE_FAMILIES['heldout'])} "
                        "(the evaluation group - the default, and the only "
                        "valid choice for an augmentation ablation); "
                        f"'augmentable' = "
                        f"{','.join(DEGRADE_FAMILIES['augmentable'])} "
                        "(reserved for the augmentation to imitate); 'all' "
                        "mixes both and is circular for ablation purposes")
    p.add_argument("--degrade-kinds", type=_degrade_kinds, default=None,
                   help="explicit comma-separated subset of "
                        f"{','.join(DEGRADE_KINDS)}, overriding "
                        "--degrade-family. Applied in that order")
    p.add_argument("--augmented-with", type=_degrade_kinds, default=None,
                   help="declare which corruption kinds your enrolment "
                        "augmentation used. The run ABORTS if any of them is "
                        "also being evaluated on, so an ablation cannot "
                        "silently score itself against its own augmentation")

    # -- statistics ---------------------------------------------------------
    p.add_argument("--conf-level", type=float, default=0.95,
                   help="confidence level for every Wilson interval")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="significance level for the power analysis")
    p.add_argument("--power", type=float, default=0.80,
                   help="target power for the minimum detectable difference")

    # -- output -------------------------------------------------------------
    p.add_argument("--max-failure-images", type=int, default=48,
                   help="cap on misclassified probes rendered per category "
                        "(all of them are still listed in failures.json)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--results-dir", default=str(Path(__file__).parent / "results"))
    p.add_argument("--cache-dir", default=str(Path(__file__).parent / ".cache"))
    p.add_argument("--no-cache", action="store_true",
                   help="recompute embeddings even if cached")
    return p


def _wrap(text: str, width: int) -> list[str]:
    """Soft-wrap a paragraph for the terminal summary."""
    import textwrap  # noqa: PLC0415

    return textwrap.wrap(text, width=width)


def _print_openset_row(label: str, op: dict[str, Any]) -> None:
    far, frr, mir = op["far"], op["frr"], op["misidentification"]
    print(f"   {label:<26} FAR {far['rate']:>8.3%} "
          f"[{far['ci_low']:.3%},{far['ci_high']:.3%}] ({far['count']}/{far['n']})"
          f"   FRR {frr['rate']:>7.2%}   MISID {mir['rate']:>7.2%} "
          f"({mir['count']}/{mir['n']})")


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in the defaults that depend on which dataset was chosen.

    CFP-FP has 500 identities but only 4 profile probes each, so the impostor
    pool needs many more identities than LFW to resolve the same FAR budget:
    80 identities would give 320 impostor probes per trial, a 0.31% floor
    against a 0.1% target.
    """
    if args.impostor_identities is None:
        args.impostor_identities = 400 if args.dataset == "cfp-fp" else 80
        print(f"--impostor-identities defaulted to "
              f"{args.impostor_identities} for {args.dataset}")

    # Resolve the corruption family into an explicit kind list.
    if args.degrade_kinds is None:
        if args.degrade_family == "all":
            args.degrade_kinds = list(DEGRADE_KINDS)
        else:
            args.degrade_kinds = list(DEGRADE_FAMILIES[args.degrade_family])
        args.degrade_family_resolved = args.degrade_family
    else:
        # An explicit list may straddle the groups, so name the family honestly
        # rather than implying a guarantee the list does not provide.
        families = {family_of(k) for k in args.degrade_kinds}
        args.degrade_family_resolved = (
            families.pop() if len(families) == 1 else "custom-mixed")

    # The invariant that makes the degraded arm usable for an ablation: you may
    # never evaluate on a corruption you augmented with. Enforced, not merely
    # documented, because this is precisely the mistake that is invisible in
    # the results - the numbers look fine, they are just meaningless.
    augmented = set(args.augmented_with or [])
    overlap = sorted(augmented & set(args.degrade_kinds))
    if overlap:
        raise SystemExit(
            "FATAL: circular ablation refused.\n"
            f"  --augmented-with declares {sorted(augmented)}\n"
            f"  the probes are degraded with {sorted(args.degrade_kinds)}\n"
            f"  overlapping: {overlap}\n\n"
            "Evaluating on a corruption the augmentation was built to undo "
            "guarantees an improvement that transfers to nothing. Evaluate on "
            f"the held-out family instead:\n"
            f"    --degrade-family heldout   "
            f"({', '.join(DEGRADE_FAMILIES['heldout'])})\n"
            "and keep the augmentation to "
            f"{', '.join(DEGRADE_FAMILIES['augmentable'])}.")
    if augmented:
        print(f"augmentation declared {sorted(augmented)}; evaluating on "
              f"{sorted(args.degrade_kinds)} - disjoint, OK")
    return args


def main() -> int:
    args = resolve_defaults(build_parser().parse_args())
    metrics = evaluate(args)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                      encoding="utf-8")

    v = metrics["verification"]
    i = metrics["identification_closed_set"]
    o = metrics["openset_identification"]
    pw = metrics["statistical_power"]
    d = metrics["dataset"]
    m = metrics["top2_margin"]

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"dataset={d['name']}  seed={metrics['seed']}  "
          f"identities={d['identities_evaluated']}"
          f"  probes={d['probe_count']}  gallery={d['gallery_count']}")
    print(f"split={d['split']}"
          + (f"  ({d['images_frontal']} frontal / {d['images_profile']} "
             "profile images)" if d.get("images_profile") else ""))
    print(f"detection failure rate : {fmt_rate(d['detection_failure_rate'])}")

    fs = metrics["face_selection"]
    if "images_with_multiple_faces" in fs:
        print(f"face selection         : {fs['rule']}")
        print(f"  multi-face frames    : "
              f"{fmt_rate(fs['images_with_multiple_faces'])}")
        print(f"  best-score != centred: "
              f"{fmt_rate(fs['images_where_best_score_face_is_not_centred'])}"
              "  <- labels unreliable on these under 'best-score'")
        if "rank1_under_alternate_rule" in fs:
            for label, k in ((fs["rule"], "rank1_under_this_rule"),
                             (fs["alternate_rule"], "rank1_under_alternate_rule")):
                print(f"  rank-1 @ {label:<12}: {fmt_rate(fs[k])}")

    print("\n--- VERIFICATION (matcher quality; does NOT set the threshold) ---")
    print(f"pairs                  : {v['genuine_pairs_balanced']:,} genuine / "
          f"{v['impostor_pairs_balanced']:,} impostor balanced for ROC/EER")
    print(f"full impostor pool     : {v['impostor_pairs_full_pool']:,} pairs "
          f"(smallest measurable FAR {v['far_resolution_full_pool']:.6%}; "
          f"balancing would have discarded "
          f"{v['impostor_pairs_discarded_by_balancing']:,})")
    print(f"ROC AUC                : {v['roc_auc']:.4f}")
    print(f"EER                    : {v['eer']:.3%} at threshold "
          f"{v['eer_threshold']:.3f}")
    rt = v["recommended_threshold"]
    print(f"threshold for FAR<={rt['target_far']:.2%} : {rt['threshold']:.4f} -> "
          f"FAR {rt['far_full_pool']['rate']:.5%} "
          f"({rt['far_full_pool']['count']:,}/{rt['far_full_pool']['n']:,} pairs), "
          f"FRR {rt['frr']['rate']:.2%}")
    for label, k in (("strong", "strong_match"), ("weak", "weak_match")):
        op = v["current_hardcoded_thresholds"][k]
        f_full, f_bal = op["far_full_pool"], op["far_balanced"]
        print(f"   current {label:<6} {op['threshold']:.2f}: "
              f"FAR {f_full['rate']:.5%} ({f_full['count']:,}/{f_full['n']:,} pairs)"
              f"  [balanced subsample said {f_bal['rate']:.5%}, "
              f"{f_bal['count']}/{f_bal['n']:,}]"
              f"  FRR {op['frr']['rate']:.2%}")

    print("\n--- CLOSED-SET IDENTIFICATION (ranking only) ---")
    print(f"rank-1                 : {fmt_rate(i['rank1_accuracy'])}")
    print(f"rank-5                 : {fmt_rate(i['rank5_accuracy'])}")
    print(f"top-2 margin           : correct median "
          f"{m['correct_rank1']['median']}, wrong median "
          f"{m['incorrect_rank1']['median']} (n={m['incorrect_rank1']['n']})")

    r = o["recommendation"]
    head = o["arms"]["clean"]["by_enrol_count"][str(o["headline_enrol_count"])]

    print("\n--- OPEN-SET IDENTIFICATION (what the app runs) ---")
    print(f"{o['trials']} trial(s), {o['impostor_identities_per_trial']} "
          f"impostor identities per trial, never enrolled. Rows below are the "
          f"clean arm at {o['headline_enrol_count']} enrolment image(s).")
    for size in o["gallery_sizes"]:
        e = head["by_gallery_size"][str(size)]
        print(f"\n gallery = {size} identities  "
              f"({e['n_genuine_probes']:,} genuine / {e['n_impostor_probes']:,} "
              f"impostor probes pooled; {e['impostor_probes_per_trial'][0]:,} "
              f"per trial, so per-trial FAR resolution is "
              f"{e['far_resolution_per_trial']:.3%})")
        _print_openset_row(f"thr for FAR<={o['target_far']:.2%} "
                           f"= {e['threshold_for_target_far']:.3f}",
                           e["at_threshold_for_target_far"])
        _print_openset_row(f"at recommended {r['threshold']:.3f}",
                           e["at_recommended"])
        _print_openset_row(f"at verification rec. "
                           f"{v['recommended_threshold']['threshold']:.3f}",
                           e["at_verification_recommended"])
        _print_openset_row(f"at current strong "
                           f"{metrics['backend_settings']['strong_match']:.2f}",
                           e["at_backend_strong_match"])
        _print_openset_row(f"at current weak "
                           f"{metrics['backend_settings']['weak_match']:.2f}",
                           e["at_backend_weak_match"])

    # ---- enrolment-count sweep ---------------------------------------------
    print(f"\n--- ENROLMENT COUNT (the app enrols "
          f"{o['headline_enrol_count']}) ---")
    print(f"   {'arm':<9} {'N':>2} {'gal':>4} {'threshold':>10} {'FAR':>9} "
          f"{'FRR':>9} {'MISID':>8}")
    for row in o["enrolment_sweep"]:
        print(f"   {row['arm']:<9} {row['enrol_count']:>2} "
              f"{row['gallery_size']:>4} "
              f"{row['threshold_for_target_far']:>10.4f} "
              f"{row['far']['rate']:>9.3%} {row['frr']['rate']:>9.2%} "
              f"{row['misidentification']['rate']:>8.3%}")

    print(f"\n what each FAR budget costs at gallery = {r['assumed_gallery_size']}"
          f", enrol = {r['assumed_enrolment_count']}")
    print(f"   {'budget':>8}  {'threshold':>9}  {'FAR':>8}  {'FRR':>8}  "
          f"{'MISID':>8}  impostor probes allowed")
    for row in head["far_budget_table"]:
        print(f"   {row['far_budget']:>8.2%}  {row['threshold']:>9.3f}  "
              f"{row['far']['rate']:>8.3%}  {row['frr']['rate']:>8.2%}  "
              f"{row['misidentification']['rate']:>8.3%}  "
              f"{row['impostor_probes_allowed']:.1f} of "
              f"{row['far']['n']:,}")

    print("\n" + "-" * 78)
    print(f"RECOMMENDED OPERATING THRESHOLD: {r['threshold']:.4f}")
    print(f"  assumed enrolment    : {r['assumed_enrolment_count']} image(s) "
          "per person (what engine.register stores)")
    print(f"  assumed gallery size : {r['assumed_gallery_size']} enrolled "
          f"identities")
    print(f"  open-set FAR         : "
          f"{fmt_rate(r['at_deploy_size']['far'], '.3%')}")
    print(f"  false reject rate    : "
          f"{fmt_rate(r['at_deploy_size']['frr'])}")
    print(f"  misidentification    : "
          f"{fmt_rate(r['at_deploy_size']['misidentification'])}")
    print(f"  probe detection loss : "
          f"{fmt_rate(head['detection_failure_on_probes'])}")
    print(f"  end-to-end miss rate : "
          f"{fmt_rate(head['end_to_end_miss_rate'])}  (FRR + detector misses)")
    if r["cost_under_degraded_probes"]:
        dg = r["cost_under_degraded_probes"]
        print(f"  same threshold on DEGRADED probes: FAR "
              f"{dg['far']['rate']:.3%}, FRR {dg['frr']['rate']:.2%}, "
              f"MISID {dg['misidentification']['rate']:.3%}")
    print("  derived from the open-set curves, NOT the verification ROC")
    if not r["resolved_within_target"]:
        print("  WARNING: no threshold reached the target FAR within the "
              "measured impostor scores; this is the max impostor score + eps.")
    if r["under_resolved"]:
        print(f"  WARNING: a {o['target_far']:.2%} budget allows "
              f"{o['target_far'] * r['impostor_probes_per_trial']:.2f} of the "
              f"{r['impostor_probes_per_trial']:,} impostor probes per trial. "
              "The target is finer than\n           this evaluation can "
              "measure, so the threshold above is set by where the single "
              "highest-scoring impostor landed, not by a rate. Raise "
              "--impostor-identities,\n           lower --min-faces, or "
              "loosen --openset-target-far.")
    print("-" * 78)

    print("\n--- STATISTICAL POWER ---")
    print(f"probes                 : {pw['n_probes']:,} "
          f"({pw['rank1_errors']} rank-1 errors)")
    print(f"rank-1 95% CI half-width: +/- {pw['rank1_ci_half_width_pp']:.2f} pp")
    print(f"min detectable diff    : "
          f"{pw['mdd_paired_same_probes_pp']:.2f} pp paired (same probes, "
          f">= {pw['mdd_paired_min_errors_fixed']} net errors fixed)")
    if pw["mdd_independent_runs_pp"] is not None:
        print(f"                         "
              f"{pw['mdd_independent_runs_pp']:.2f} pp unpaired "
              f"(two independent runs, {pw['target_power']:.0%} power)")
    if pw["at_ceiling"]:
        print("\n".join(f"  {line}" for line in
                        _wrap(pw["at_ceiling"], 74)))

    dg = metrics["degradation"]
    if dg:
        print("\n--- DEGRADED PROBES ---")
        print(f"family                 : {dg['family']}"
              + ("  (held out from augmentation - non-circular)"
                 if dg["non_circular"] else "  (NOT held out - circular)"))
        print(f"corruptions            : {', '.join(dg['kinds'])} "
              f"(v{dg['version']}, seeded from --seed)")
        print(f"reserved for augment.  : "
              f"{', '.join(dg['reserved_for_augmentation'])}  (never evaluated)")
        if dg["augmented_with_declared"]:
            print(f"declared augmentation  : "
                  f"{', '.join(dg['augmented_with_declared'])}  (disjoint, checked)")
        print(f"detection failure      : "
              f"{fmt_rate(dg['detection_failure_rate'])}")
        print(f"faces lost vs clean    : "
              f"{fmt_rate(dg['detection_lost_vs_clean'])}")
        if "rank1" in dg:
            print(f"rank-1 on degraded     : {fmt_rate(dg['rank1'])} "
                  f"({dg['probes_scored']:,} scored, "
                  f"{dg['probes_undetected']:,} undetected)")
        print(f"examples               : results/{dg['example_grid']}")

    at = metrics["augmentation_target"]
    print("\n--- AUGMENTATION TARGET (a metric with headroom) ---")
    print(f"a paired re-run needs >= {at['min_errors_for_paired_significance']}"
          " net errors fixed to reach significance")
    print(f"   {'metric':<52} {'errors':>7} {'n':>8} {'MDD pp':>7}  ok")
    for c in at["candidates"]:
        mdd = c["mdd"]
        flag = ("yes" if mdd["measurable"] and c["affected_by_enrolment"]
                else ("n/a" if not c["affected_by_enrolment"] else "NO"))
        if c.get("circularity_risk") and flag == "yes":
            flag = "yes*"
        pp = (f"{mdd['mdd_paired_pp']:.3f}"
              if mdd["mdd_paired_pp"] is not None else "-")
        print(f"   {c['name']:<52} {mdd['errors_available']:>7,} "
              f"{mdd['n']:>8,} {pp:>7}  {flag}")
    print("   (* measurable, but see the circularity warning below)")
    print()
    for line in _wrap(at["verdict"], 74):
        print(f"  {line}")
    if at["recommended_metric"] is None:
        print("\n  Harder public benchmarks to add instead:")
        for b in at["harder_benchmarks"]:
            print(f"    * {b['name']}")
            for line in _wrap(b["why"], 68):
                print(f"        {line}")
    ap = at.get("ablation_primary_metric")
    if ap:
        print("\n  >>> PRIMARY METRIC FOR THE AUGMENTATION ABLATION <<<")
        print(f"  {ap['metric']}")
        print(f"  N={ap['enrol_count']} enrolment, gallery {ap['gallery_size']}, "
              f"threshold {ap['threshold']:.4f}, family "
              f"{ap['degrade_family']} ({', '.join(ap['degrade_kinds'])})")
        print(f"    FAR held at   : {fmt_rate(ap['far_held_at'], '.3%')}")
        print(f"    FRR           : {fmt_rate(ap['frr'])}")
        print(f"    MISID         : {fmt_rate(ap['misidentification'])}")
        print(f"    detector loss : "
              f"{fmt_rate(ap['detection_failure_on_probes'])}")
        print(f"    end-to-end    : {fmt_rate(ap['end_to_end_miss_rate'])}")
        print(f"    MDD (paired)  : {ap['mdd']['mdd_paired_pp']:.3f} pp "
              f"({ap['mdd']['mdd_paired_min_errors_fixed']} of "
              f"{ap['mdd']['errors_available']:,} errors)")
        for label, body in (("supports", ap["supports"]),
                            ("does NOT support", ap["does_not_support"])):
            print(f"    {label}:")
            for line in _wrap(body, 66):
                print(f"      {line}")

    print()
    for line in _wrap(at["threshold_caveat"], 74):
        print(f"  {line}")

    fg = metrics["failure_grids"]
    print("\n--- FAILURES ---")
    for slug, info in fg.items():
        print(f"   {slug:<30} {info['n']:>5} "
              f"(plotted {info['plotted']}: {', '.join(info['files']) or '-'})")
    print("=" * 78)
    print(f"metrics -> {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
