"""
Measure the presentation-attack detector against a captured test set.

The matcher has `eval/evaluate.py` and a threshold with a run behind it. Until
this script has been run, the liveness feature has neither — it is a plausible
prompt and an untested API path, and `README.md` must not claim otherwise. This
is the thing that turns "we added anti-spoofing" into a number with an interval.

What it measures
----------------
Per capture condition (live / printed photo / phone replay / laptop replay /
mask):

  * **detection rate** — attacks correctly refused. The headline for each attack.
  * **false-reject rate** — live faces wrongly refused. The number an enrolled
    person actually feels, because it is them standing there not being marked.
  * **attack-type accuracy** — when an attack was caught, was it named
    correctly. Secondary: refusing the mark is what protects the register;
    naming the attack only makes the audit row more useful.
  * **latency** — mean / median / p95 per decision, which is what a person waits
    before being marked present.
  * **errors** — checks that returned no usable verdict. Reported SEPARATELY and
    never folded into a detection rate. An ERROR fails closed, so counting it as
    a catch would let a broken provider score 100% against every attack.

Every rate carries a Wilson 95% interval, imported from the matcher harness
rather than restated, for the reason given in its docstring: these rates sit
near 0 or near 1, exactly where a Wald interval misbehaves.

What it does NOT do
-------------------
It does not pick a threshold. The model returns a verdict and a confidence, and
this script reports what that verdict costs; it deliberately does not sweep
`confidence` to find an operating point, because with the sample sizes a hand
capture can realistically produce, a threshold tuned on this set would be fitted
to it. If a confidence gate is wanted later, capture a second set.

Running it costs money
----------------------
One API call per image, per run. See eval/liveness/README.md for the arithmetic
before starting a full pass, and use --limit while iterating.

Usage:

    py eval/liveness/evaluate.py --dry-run     # inventory, no API calls
    py eval/liveness/evaluate.py
    py eval/liveness/evaluate.py --limit 5     # 5 per condition, a smoke test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
PROJECT_ROOT = EVAL_ROOT.parent

# The backend is imported the same way eval/evaluate.py imports it: flat module
# names off backend/, so this harness reads the SAME liveness code the app runs
# rather than a copy of it. A drift between the two would make every number here
# a measurement of something nobody is running.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(EVAL_ROOT))

import liveness  # noqa: E402
from config import DET_SIZE, MODEL_NAME  # noqa: E402
from evaluate import fmt_rate, rate  # noqa: E402  (one Wilson implementation)

# Capture conditions, and what a correct answer looks like for each.
#
# `expect_live` is the ground truth the operator supplies by putting a file in
# that folder. `expect_attack` is the label the model SHOULD choose when it
# correctly refuses — phone and laptop replays are both `screen_replay` to the
# model, but they are kept apart here because they are different captures with
# different tells (a phone is small, bright and hand-held; a laptop is large,
# matte and usually static) and averaging them would hide a real difference.
CONDITIONS: dict[str, dict[str, Any]] = {
    "live": {
        "expect_live": True,
        "expect_attack": "none",
        "label": "Live faces",
    },
    "printed_photo": {
        "expect_live": False,
        "expect_attack": "printed_photo",
        "label": "Printed photographs",
    },
    "screen_phone": {
        "expect_live": False,
        "expect_attack": "screen_replay",
        "label": "Phone screen replay",
    },
    "screen_laptop": {
        "expect_live": False,
        "expect_attack": "screen_replay",
        "label": "Laptop screen replay",
    },
    "mask": {
        "expect_live": False,
        "expect_attack": "mask",
        "label": "Masks (optional)",
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(dataset: Path, condition: str, limit: int | None) -> list[Path]:
    """Every image in one condition folder, sorted for reproducibility."""
    folder = dataset / condition
    if not folder.is_dir():
        return []
    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    return files[:limit] if limit else files


def detect_and_crop(path: Path, app: Any) -> bytes | None:
    """The app's own detector, then the app's own crop. None if no face.

    Going through `liveness.crop_for_liveness` rather than sending the whole
    file is what makes this an evaluation of the deployed pipeline: the model
    sees exactly the framing the camera path would give it, margin and
    downscaling included. Feeding it the full photograph would measure a
    different system and flatter it, because a full frame carries context
    (a whole phone, a hand, a desk) the live path deliberately crops away.
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None
    faces = app.get(image)
    if not faces:
        return None
    face = max(faces, key=lambda f: float(f.det_score))
    bbox = tuple(int(v) for v in face.bbox)
    return liveness.crop_for_liveness(image, bbox)


def evaluate_condition(condition: str, files: list[Path], app: Any,
                       verbose: bool) -> dict[str, Any]:
    """Run every image in one condition and tally the outcomes."""
    spec = CONDITIONS[condition]
    records: list[dict[str, Any]] = []
    undetected = 0

    for path in files:
        crop = detect_and_crop(path, app)
        if crop is None:
            # No face found: nothing was asked of the model, so this is not a
            # liveness outcome. Counted on its own, exactly as the matcher
            # harness separates detector misses from threshold failures.
            undetected += 1
            if verbose:
                print(f"    {path.name}: NO FACE DETECTED")
            continue

        started = time.perf_counter()
        result = liveness.check(crop)
        elapsed = time.perf_counter() - started

        records.append({
            "file": path.name,
            "state": result.state,
            "attack_type": result.attack_type,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "latency_s": elapsed,
            "retried": result.retried,
        })
        if verbose:
            print(f"    {path.name}: {result.state} "
                  f"({result.attack_type}, conf={result.confidence:.2f}, "
                  f"{elapsed:.2f}s)")

    answered = [r for r in records if r["state"] != liveness.LIVENESS_ERROR]
    errors = [r for r in records if r["state"] == liveness.LIVENESS_ERROR]
    n_answered = len(answered)

    block: dict[str, Any] = {
        "condition": condition,
        "label": spec["label"],
        "files_found": len(files),
        "faces_detected": len(records),
        "detection_failures": undetected,
        "checks_run": len(records),
        "errors": len(errors),
        # Errors are a property of the provider, not of the attack, so they get
        # their own rate over every check rather than being buried in one.
        "error_rate": rate(len(errors), len(records)) if records else None,
        "records": records,
    }

    if spec["expect_live"]:
        # Live faces: the failure that matters is a wrongly refused person.
        # Denominator is answered checks; errors are reported separately, and
        # the fail-closed consequence of an error is spelled out in the summary
        # rather than silently folded in here.
        rejected = sum(1 for r in answered if r["state"] != liveness.LIVENESS_PASS)
        block["false_reject_rate"] = rate(rejected, n_answered)
        # What an operator actually experiences: an ERROR also refuses the mark.
        block["effective_reject_rate"] = rate(
            rejected + len(errors), len(records)) if records else None
    else:
        caught = sum(1 for r in answered if r["state"] == liveness.LIVENESS_FAIL)
        block["detection_rate"] = rate(caught, n_answered)
        named = sum(
            1 for r in answered
            if r["state"] == liveness.LIVENESS_FAIL
            and r["attack_type"] == spec["expect_attack"]
        )
        block["attack_type_accuracy"] = rate(named, caught) if caught else None
        # Errors refuse the mark too, so an attack that errored was in practice
        # stopped. Reported as a separate, explicitly weaker claim: it was
        # stopped by a failure, not by a detection.
        block["effective_refusal_rate"] = rate(
            caught + len(errors), len(records)) if records else None

    latencies = [r["latency_s"] for r in records]
    if latencies:
        ordered = sorted(latencies)
        block["latency"] = {
            "n": len(ordered),
            "mean_s": statistics.fmean(ordered),
            "median_s": statistics.median(ordered),
            "p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "max_s": ordered[-1],
        }
    else:
        block["latency"] = None
    return block


def print_report(report: dict[str, Any]) -> None:
    """Human-readable summary. The JSON is the record; this is for reading."""
    print()
    print("=" * 78)
    print("LIVENESS EVALUATION")
    print("=" * 78)
    print(f"  provider    {report['provider']}")
    print(f"  model       {report['model']}")
    print(f"  detector    {report['detector']['model']} "
          f"det_size={tuple(report['detector']['det_size'])}")
    print(f"  dataset     {report['dataset']}")
    print()

    for block in report["conditions"]:
        if not block["files_found"]:
            continue
        print(f"  {block['label']}  ({block['condition']})")
        print(f"    images                {block['files_found']}")
        if block["detection_failures"]:
            print(f"    no face detected      {block['detection_failures']}"
                  "   (excluded: the model was never asked)")
        if block.get("detection_rate") is not None:
            print(f"    detection rate        "
                  f"{fmt_rate(block['detection_rate'])}")
            if block.get("attack_type_accuracy") is not None:
                print(f"    attack named right    "
                      f"{fmt_rate(block['attack_type_accuracy'])}")
            print(f"    refused in practice   "
                  f"{fmt_rate(block['effective_refusal_rate'])}"
                  "   (detections + errors)")
        if block.get("false_reject_rate") is not None:
            print(f"    FALSE REJECT RATE     "
                  f"{fmt_rate(block['false_reject_rate'])}")
            print(f"    refused in practice   "
                  f"{fmt_rate(block['effective_reject_rate'])}"
                  "   (rejects + errors)")
        if block["errors"]:
            print(f"    errors                {fmt_rate(block['error_rate'])}")
        if block["latency"]:
            lat = block["latency"]
            print(f"    latency               mean {lat['mean_s']:.2f}s  "
                  f"median {lat['median_s']:.2f}s  p95 {lat['p95_s']:.2f}s")
        print()

    total = report["totals"]
    print(f"  Total checks          {total['checks']}")
    print(f"  Total API calls       {total['checks']} "
          f"(+{total['retries']} retries)")
    if total["latency_mean_s"] is not None:
        print(f"  Overall latency       mean {total['latency_mean_s']:.2f}s  "
              f"p95 {total['latency_p95_s']:.2f}s")
    print()
    print("  Reminder: an ERROR fails closed. It refuses the mark, so it")
    print("  protects the register — but it is not a detection, and the two")
    print("  are reported separately above for that reason.")
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the liveness check against a captured test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", type=Path, default=HERE / "dataset",
        help="Capture root, containing one folder per condition.")
    parser.add_argument(
        "--out", type=Path, default=HERE / "results" / "liveness_metrics.json",
        help="Where to write the JSON record.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only the first N images per condition. Use while iterating — a "
             "full pass costs one API call per image.")
    parser.add_argument(
        "--conditions", type=str, default=None,
        help="Comma-separated subset, e.g. live,printed_photo.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Inventory the dataset and print the projected call count and "
             "cost. Makes NO API calls.")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every per-image verdict as it lands.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = args.dataset.resolve()

    wanted = list(CONDITIONS)
    if args.conditions:
        wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in CONDITIONS]
        if unknown:
            print(f"FATAL: unknown condition(s) {unknown}. "
                  f"Expected any of {list(CONDITIONS)}.", file=sys.stderr)
            return 2

    if not dataset.is_dir():
        print(f"FATAL: no dataset at {dataset}\n\n"
              "Nothing has been captured yet. eval/liveness/README.md says "
              "exactly what to capture and how many of each; the folder is "
              "gitignored because it holds real biometric data.",
              file=sys.stderr)
        return 2

    inventory = {c: find_images(dataset, c, args.limit) for c in wanted}
    total_images = sum(len(v) for v in inventory.values())

    print(f"Dataset: {dataset}")
    for condition in wanted:
        found = len(inventory[condition])
        note = "" if found else "   (empty — this condition will be skipped)"
        print(f"  {condition:<16} {found:>4} images{note}")
    print(f"  {'TOTAL':<16} {total_images:>4} images")

    if total_images == 0:
        print("\nFATAL: the dataset is empty. See eval/liveness/README.md.",
              file=sys.stderr)
        return 2

    if args.dry_run:
        print("\n--dry-run: no API calls made.")
        print(f"A full pass would make about {total_images} calls "
              "(one per detected face, plus a retry for any malformed reply).")
        print("Per-call cost depends on LIVENESS_MODEL — see "
              "eval/liveness/README.md for the arithmetic.")
        return 0

    # The provider check comes AFTER the inventory on purpose: knowing what you
    # captured is useful even on a machine with liveness switched off, which is
    # the machine most of this work happens on.
    if not liveness.is_active():
        status = liveness.status()
        print("\nFATAL: no liveness provider is active, so there is nothing to "
              "measure.\n", file=sys.stderr)
        print(f"  {status['reason']}\n", file=sys.stderr)
        print("  This harness deliberately has no offline mode. A liveness "
              "number\n  produced without a provider would be a number about "
              "nothing.\n", file=sys.stderr)
        print("  Use --dry-run to inventory the dataset without a provider.",
              file=sys.stderr)
        return 2

    from insightface.app import FaceAnalysis

    print(f"\nProvider: {liveness.provider_name()}")
    print(f"Loading detector {MODEL_NAME} …")
    app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    blocks = []
    for condition in wanted:
        files = inventory[condition]
        if not files:
            continue
        print(f"\n  {CONDITIONS[condition]['label']} ({len(files)} images) …")
        blocks.append(evaluate_condition(condition, files, app, args.verbose))

    all_records = [r for b in blocks for r in b["records"]]
    latencies = sorted(r["latency_s"] for r in all_records)
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": liveness.provider_name(),
        "model": liveness.status()["model"],
        "detector": {"model": MODEL_NAME, "det_size": list(DET_SIZE)},
        "dataset": str(dataset),
        "limit": args.limit,
        "crop": {
            "margin": liveness.LIVENESS_CROP_MARGIN,
            "max_px": liveness.LIVENESS_CROP_MAX_PX,
        },
        "conditions": blocks,
        "totals": {
            "checks": len(all_records),
            "retries": sum(1 for r in all_records if r["retried"]),
            "errors": sum(1 for b in blocks for _ in range(b["errors"])),
            "latency_mean_s": statistics.fmean(latencies) if latencies else None,
            "latency_p95_s": (
                latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
                if latencies else None
            ),
        },
    }

    print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print("NOTE: that file names real capture files and stays local — "
          "eval/liveness/results/ is gitignored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
