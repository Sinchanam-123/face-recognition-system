# Liveness evaluation

Measures the presentation-attack check in `backend/liveness.py` against a test
set you capture yourself.

**Nothing here has been run.** The feature ships inactive and unmeasured: the
provider code has never made an API call, and no test set exists on this
machine. Until this harness produces numbers, `README.md` must not claim the
system resists proxy attendance — it can only claim the check exists and is off.

That is the whole reason this directory is written before the capture: so the
first liveness claim anyone makes has an interval attached.

## Your test set is real biometric data

Everything you capture is a photograph of an identifiable person, plus
deliberate spoof artefacts made from their face. Two consequences:

* **It is gitignored already.** `eval/liveness/dataset/` and
  `eval/liveness/results/` are in `.gitignore` (added before these instructions,
  deliberately). Verify with `git check-ignore -v eval/liveness/dataset/x.jpg`
  before you capture anything.
* **It stays on this machine.** It is not committed, not uploaded, and not
  shared. The only thing that leaves is one cropped face per API call *during a
  run*, and only to the provider you configured.

Get consent from everyone you photograph, and tell them their face will be sent
to whichever provider you have set. If you are the only subject, this is simple;
if you use classmates, it is not optional.

## What to capture

Five folders under `eval/liveness/dataset/`. Folder name **is** the ground
truth, so a misfiled image is a wrong label:

```
eval/liveness/dataset/
    live/              a real person, at the camera
    printed_photo/     a printed photo held up to the camera
    screen_phone/      a face on a phone screen, held up to the camera
    screen_laptop/     a face on a laptop/monitor screen
    mask/              optional — see below
```

### How many, and why those numbers

| Folder | Capture | Why this count |
|---|---|---|
| `live/` | **150** | This is the false-reject rate, the number an enrolled person actually feels — they stand there not being marked. It needs the tightest interval, so it gets the most images. |
| `printed_photo/` | **60** | At 60/60 caught, the Wilson lower bound is 94.0%. That supports "at least 94% detection"; 50 images only supports 92.9%, and 30 only 88.6%. |
| `screen_phone/` | **60** | Same arithmetic. Kept separate from laptop because the tells differ — a phone is small, bright, hand-held and often tilted. |
| `screen_laptop/` | **60** | A laptop screen is large, matte, static, and usually fills more of the frame. Averaging the two would hide a real difference between them. |
| `mask/` | **0** (optional) | Only if you have a printed face mask or a mannequin. Skip it rather than fake it — an empty folder is honest, a folder of near-misses is not. |

**330 images total.** That is one API call each:

| `LIVENESS_MODEL` | Per call | Full pass (330) |
|---|---|---|
| `claude-opus-5` (default) | ~$0.012 | **~$4.00** |
| `claude-haiku-4-5` | ~$0.0012 | **~$0.40** |
| Ollama vision model | $0 | $0 (slower, local) |

Run `--dry-run` first; it inventories the folders and makes no calls.

### Capturing `live/` — 150 images

Vary the conditions, because a false-reject rate measured under one lighting
setup is a measurement of that setup. Aim for roughly even coverage:

* **Lighting** (the big one): bright overhead, dim room, side-lit from a window,
  backlit with a window behind you, warm lamp at night.
* **Distance**: face filling the frame, at arm's length, across a room.
* **Angle**: straight on, three-quarter left and right, slightly above and below.
* **Appearance**: with and without glasses; hair up and down; neutral and
  smiling; if applicable, with and without a head covering or a mask on the chin.
* **People**: as many different faces as you can consent. A 150-image set of one
  person measures that person.

Capture them the way the app sees faces — **from the same webcam**, at the
resolution the app runs at. A DSLR portrait is not what `engine._recognize`
gets, and a check that works on studio photographs and fails on your webcam is
a check that fails.

`py backend/check_camera.py` opens a preview window you can capture from.

### Capturing the attacks

The attacks must be made from the **same faces** in `live/`. An attack set built
from strangers' photos measures whether the model can spot a stranger, not
whether it can spot a spoof.

**`printed_photo/` (60)** — print face photos on ordinary paper. Vary:
* paper: matte and glossy, if you have both
* size: full page and small
* handling: held flat, slightly curved, with a visible hand vs. propped up
* lighting: some with glare across the print, some without
* framing: some with the paper edge visible, **some cropped so tightly the edge
  is not** — this is the hard case, and leaving it out inflates your score

**`screen_phone/` (60)** — display the photos full-screen on a phone. Vary:
* brightness: max, medium, low
* angle: straight on, tilted (moire is angle-dependent)
* distance: close enough to see pixels, far enough that you cannot
* framing: bezel visible, and bezel cropped out

**`screen_laptop/` (60)** — same, on a laptop or monitor. Vary brightness,
angle, and whether the screen edge is in frame.

For every attack folder: put roughly **a third with no give-away frame in
shot**. It is the easiest thing to get wrong. A set where every attack has a
visible bezel measures whether the model can see a rectangle.

## Running it

```bash
py eval/liveness/evaluate.py --dry-run
```

```bash
py eval/liveness/evaluate.py --limit 5
```

```bash
py eval/liveness/evaluate.py
```

`--limit 5` is a ~20-call smoke test that confirms the provider works and the
crops look right before you commit to the full pass. Start there.

Needs a provider — see the activation section in the root `README.md`. With
none configured the harness **refuses to run** rather than producing a number,
because a liveness figure produced with no provider is a figure about nothing.
`--dry-run` still works with liveness switched off.

## What it reports

Per condition, each with a Wilson 95% interval (imported from
`eval/evaluate.py`, not restated — these rates sit near 0 or 1, where a Wald
interval misbehaves):

| Metric | Where | Meaning |
|---|---|---|
| **detection rate** | attack folders | Attacks correctly refused. The headline. |
| **attack named right** | attack folders | Of those caught, how many got the label right. Secondary — refusing the mark is what protects the register; the label only enriches the audit row. |
| **false-reject rate** | `live/` | Live people wrongly refused. The cost the feature imposes on everybody. |
| **latency** | all | mean / median / p95 per decision — how long a person waits to be marked. |
| **errors** | all | Checks with no usable verdict. |

### Two things the report keeps separate on purpose

**Errors are not detections.** An `ERROR` fails closed, so it refuses the mark
and does protect the register — but it is not the model catching anything. If
errors were folded into the detection rate, a completely broken provider would
score 100% against every attack. The report gives detection rate (answered
checks only) and a separate "refused in practice" figure that includes errors,
so both readings are on the table.

**Detector misses are not liveness outcomes.** An image where InsightFace finds
no face never reaches the model, so it is excluded from every rate and counted
on its own — the same separation `eval/README.md` makes between
`detection_failure_on_probes` and FRR.

## What this can and cannot support

It **can** support: *"on 330 captures from this webcam, the check refused N% of
printed photos, phone replays and laptop replays, at a false-reject rate of M%
on live faces, with these intervals."*

It **cannot** support:

* **Anything about attacks you did not capture.** Video replay with motion,
  deepfakes, high-quality silicone masks, and paper masks with cut-out eyes are
  all real presentation attacks and none of them are in this protocol. A high
  score here is not a claim about them.
* **Anything about another camera or another room.** Same caveat the matcher
  harness carries: this measures your capture pipeline.
* **A security guarantee.** The check is a vision model reading a JPEG. It can
  be wrong, and an attacker who knows what it looks for can work against it. The
  honest claim is "raises the cost of proxy attendance from *holding up a
  phone*", not "prevents proxy attendance".
* **A threshold.** The harness reports the verdict the model returns; it does
  not sweep `confidence` for an operating point. With sample sizes a hand
  capture can produce, a threshold tuned here would be fitted to this set. If
  you want a confidence gate, capture a second, independent set for it.

## Writing up

Put the numbers in the root `RESULTS.md`, beside the matcher's. The liveness
section there is currently a stub that says it is unmeasured; replace it with
what this prints, intervals included, and keep the "cannot support" list with
it — the claim and its limits travel together.
