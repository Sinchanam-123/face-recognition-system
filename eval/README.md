# Evaluation harness

A defensible accuracy baseline for the recognition engine. Every threshold in
`backend/engine.py` was a hand-picked guess; this measures what those guesses
actually cost, and what to use instead.

**This directory only measures. It never modifies `backend/`.**

## The two benchmarks

| | **LFW** (`--dataset lfw`, default) | **CFP-FP** (`--dataset cfp-fp`) |
|---|---|---|
| What it is | 158 identities (at `--min-faces 10`), frontal press photography | 500 identities, each with 10 frontal + 4 profile images |
| Obtained | `sklearn.fetch_lfw_people`, automatic | Manual download, see below |
| Split | enrol `--enrol-count` random images, probe with the rest | enrol **frontal**, probe **profile** |
| Stresses | nothing much — `buffalo_sc` has effectively solved it | pose, the axis a single enrolment frame fails on |
| Used for | the deployed threshold, and regression-checking model changes | **the enrolment-augmentation ablation** |

Both run the *same* open-set identification protocol, the same gallery-size
sweep, the same Wilson intervals and the same MDD analysis. Only the data and
the split differ, so the two baselines are directly comparable.

**Which governs what.** The threshold that goes into `engine.py` is set on
**LFW**, because the app's enrolment path is a cooperative, roughly frontal
webcam frame and LFW is the closer match to that distribution. **CFP-FP governs
the augmentation ablation**, because LFW is saturated — see
[The two baselines, side by side](#the-two-baselines-side-by-side) for the
error counts behind that split of responsibilities.

### Getting CFP-FP

It is a plain public research download — no registration, no licence
click-through:

```bash
curl -L -o cfp-dataset.zip http://cfpw.io/cfp-dataset.zip
```

82 MB. Unzip it anywhere and point the harness at the extracted folder:

```bash
python eval/evaluate.py --dataset cfp-fp --cfp-root /path/to/cfp-dataset
```

The expected layout is the zip's own — do not rearrange it:

```
<cfp-root>/Data/Images/001/frontal/01.jpg ... 10.jpg
<cfp-root>/Data/Images/001/profile/01.jpg ... 04.jpg
...              .../500/...
<cfp-root>/Data/list_name.txt        (optional; supplies real names)
```

`--cfp-root` also accepts the `Data/` or `Data/Images/` directory directly, or
any parent containing them. If it cannot find the tree it prints these
instructions rather than guessing or falling back to another dataset.

Cite: S. Sengupta, J.C. Cheng, C.D. Castillo, V.M. Patel, R. Chellappa,
D.W. Jacobs, *Frontal to Profile Face Verification in the Wild*, IEEE WACV
2016. <http://www.cfpw.io/>

### The CFP-FP split

`--cfp-protocol fp` (default) enrols **frontal** images and probes with
**profile** ones. That is the CFP-FP protocol as published, and it is also the
deployment analogue: the operator registers a cooperative face-on frame, and
the camera then has to match whatever angle the person presents.
`--cfp-protocol mixed` ignores pose and splits like LFW, for comparison.

Because CFP gives only 4 profile probes per identity, `--impostor-identities`
defaults to **400** here rather than LFW's 80 — 80 identities would yield 320
impostor probes per trial, a 0.31% FAR floor against a 0.1% target. The harness
prints the resolved default.

CFP images are real JPEGs at varying sizes, so the pipeline carries them as a
list rather than one stacked array. `--cfp-max-side` downscales oversized
images and is part of the cache key; it defaults to 0 (leave them alone).

## The two protocols, and which one sets the threshold

This is the most important thing in this file. The harness runs two evaluations
that sound similar and answer completely different questions.

### Verification — 1:1, "are these two faces the same person?"

Take a pair of images, score them, accept if the score clears a threshold.
Every pair is independent; the answer for one pair does not depend on how many
people are enrolled. This is what produces the ROC, the AUC and the EER.

Verification is the right way to summarise **matcher quality**. AUC and EER are
threshold-free, so they are what you compare when you change model, detector
settings or preprocessing: "did the embedding get better?"

### Open-set identification — 1:N, "who is this, or nobody?"

This is what `engine._recognize()` actually does, every third frame:

```python
sims = known_matrix @ emb      # score against EVERY enrolled embedding
idx  = int(sims.argmax())      # take the best one
sim  = float(sims[idx])
if sim >= STRONG_MATCH: ...    # accept it as names[idx]
```

Three things follow from that code, and none of them are visible in a
verification ROC:

1. **The score is a maximum over the whole gallery.** A stranger gets one
   chance per enrolled person to score high, and only the luckiest one counts.
   Enrol twice as many people and you roughly double a stranger's chances. So
   the false-accept rate is a function of gallery size, and a threshold
   calibrated on a 10-person gallery does not hold at 50.
2. **There is a third failure mode.** A pair is either accepted or rejected,
   but an identification can also be accepted *as the wrong person* —
   the top match clears the threshold and it is someone else. That marks the
   wrong person present **and** leaves the real person unmarked. It cannot
   happen in a 1:1 protocol, because a 1:1 protocol has no competing
   identities to be confused with.
3. **Impostors must be strangers, not just non-matching pairs.** An impostor
   pair drawn from a closed identity set is still a pair of enrolled people.
   To measure what happens when someone who has never been enrolled walks in
   front of the camera, you need identities held out of the gallery entirely.

**The app's threshold is set from the open-set curves, at a stated gallery
size.** The verification ROC is reported for matcher comparison and nothing
else. A threshold read off the verification FAR curve will be too permissive in
deployment, by a factor that grows with the size of your gallery.

One deployment detail sharpens this further: `engine.py` runs recognition on
every third frame, and `_mark()` is first-write-wins. A false accept is not a
transient blip — one bad frame in a whole session permanently marks the wrong
person present. The per-decision FAR budget therefore has to be far tighter
than a per-session error rate would suggest.

### Enrolment count: the app enrols **one** frame

`engine.register()` appends exactly one embedding — the frame that was on
screen when the operator typed a name. `encode_faces.py` can enrol several
photos, but the live path cannot. An evaluation that enrols three images per
person is therefore measuring a system nobody is running, and it is optimistic
in a specific way: with three enrolled frames the gallery already covers some
of the pose and lighting variation a single frame does not, so the false-reject
rate is lower than deployment will see.

So `--enrol-count` defaults to **1**, and the open-set protocol is swept over
`--enrol-counts 1,3,5` to quantify what the extra frames are worth.

The sweep is properly paired: `max(--enrol-counts)` images per identity are
reserved as enrolment candidates up front, and each count uses a **prefix** of
that reserved list. N=1, N=3 and N=5 are therefore scored on identical probes
against nested galleries, so any difference between them is caused by
enrolment size and nothing else. It costs a handful of probes at N=1 and buys
a comparison that means something.

### How the open-set protocol is constructed

* Each identity contributes `--enrol-count` images (default 1) to the
  gallery; every other detected image of theirs is a probe.
* Per trial, the identities are shuffled. The first `max(--gallery-sizes)` are
  the gallery pool; the next `--impostor-identities` are held out **entirely**
  — no image of theirs is in any gallery, so every one of their probes must be
  rejected.
* Gallery sizes are **nested prefixes** of that one draw (10 ⊂ 25 ⊂ 50), so the
  only thing that differs between the curves is how many people are enrolled.
* The draw rotates each trial (`--openset-trials`, default 5) and results are
  pooled, so the numbers are not one lucky split.
* Every probe is scored exactly as `engine._recognize` scores a face: max over
  the gallery, argmax for the name, accept above the threshold.

Reported against threshold, for each gallery size:

| Rate | Denominator | What went wrong |
|---|---|---|
| **FAR** | impostor probes | A stranger's top match cleared the threshold. Someone who has never enrolled is marked present. |
| **FRR** | genuine probes | An enrolled person's top match did not clear the threshold. They stand there unrecognised — visible, and self-correcting. |
| **MISID** | genuine probes | An enrolled person's top match cleared the threshold **on the wrong identity**. Tracked separately from FAR because it is a different failure with a different cost: two people's records are wrong, not one. |

Over genuine probes, `correct-accept + FRR + MISID = 1` exactly.

A fourth failure sits outside that identity: a probe the **detector never
found**. No threshold can fix it, so it is excluded from the three rates above
and reported on its own as `detection_failure_on_probes`. Because it is still a
person not marked present, the harness also reports
`end_to_end_miss_rate = FRR + detector misses`, which is what an operator
actually experiences. Keep the two separate when tuning: one is a threshold
problem, the other is not.

## Degraded probes, and the two corruption families

LFW is press photography — well-lit, sharp, high-bitrate. A webcam frame in a
classroom is none of those, so a threshold calibrated purely on clean LFW is
calibrated on the wrong distribution. The `--degrade` arm embeds a corrupted
copy of every image and re-runs the whole open-set protocol against it.

The corruptions are split into **two disjoint families**, and that split is
what makes this arm usable as an ablation metric instead of a circular one.

### `augmentable` — reserved for the augmentation, never evaluated on

| Corruption | Range | Approximates |
|---|---|---|
| `illumination` | gain 0.30–0.65, gamma 1.0–1.6 | An underexposed room — not a uniform darkening. |
| `pose_warp` | yaw-like perspective, ±0.35 | Turning the head — *crudely*, see below. |

These are effects a synthetic enrolment augmentation can plausibly imitate.
They are reserved **for** the augmentation. Evaluating on them would be
measuring whether an augmentation can undo a distortion it was built to model,
which is guaranteed to succeed and transfers to nothing.

`pose_warp` deserves a caveat it carries in its own docstring too: a planar
perspective transform **cannot rotate a head**. It compresses one side of the
frame; it cannot reveal or hide the structure a real profile view does. It is a
proxy for pose, not pose — which is precisely why it belongs in the augmentable
group rather than the evaluation group.

### `heldout` — the evaluation group

| Corruption | Range | Approximates |
|---|---|---|
| `motion_blur` | 5–13 px line kernel, random angle | Someone walking past the camera. |
| `downscale` | decimate 2.5–4×, scale back up | A small face in a 640×480 frame. |
| `jpeg` | quality 15–40 | MJPEG stream compression. |

The augmentation is not allowed to touch these. An improvement measured here is
evidence the augmentation bought **general** robustness rather than memorising
the corruption it was trained against.

### The flags, and the check that enforces the split

```bash
python eval/evaluate.py                                     # heldout (default)
python eval/evaluate.py --degrade-family augmentable        # circular; diagnostic only
python eval/evaluate.py --degrade-family all                # circular; the old behaviour
python eval/evaluate.py --augmented-with illumination,pose_warp   # declare and check
```

`--degrade-family` selects the group. `--degrade-kinds` still takes an explicit
list and overrides it, in which case the family is recorded as the group the
list actually falls in, or `custom-mixed` if it straddles both.

`--augmented-with` is the important one. Declare what your augmentation used
and **the run aborts if any of it is also being evaluated on**:

```
FATAL: circular ablation refused.
  --augmented-with declares ['illumination', 'motion_blur']
  the probes are degraded with ['downscale', 'jpeg', 'motion_blur']
  overlapping: ['motion_blur']
```

This is enforced rather than merely documented because it is exactly the
mistake that leaves no trace in the results — the numbers look fine, they are
just meaningless. The choice is recorded in `metrics.json` under
`config.degrade_family`, `config.augmented_with`, and `degradation.non_circular`,
so a run can always be audited after the fact.

### Determinism

Each corruption draws its parameters from its own
`_kind_rng(seed, image index, kind)` stream. Deliberately *not* one generator
per image consumed in pipeline order: that made a corruption's parameters
depend on which other corruptions were selected, so `--degrade-kinds jpeg`
produced a different JPEG quality than the same image got inside the full
composite. Hashing the kind name into the seed makes each corruption
reproducible on its own terms, which is what lets the two family groups be
compared as a controlled ablation rather than two unrelated distortions.

Corruptions always run in `DEGRADE_KINDS` order — the order a real capture
pipeline applies them — regardless of which group is selected.

**Only probes are degraded; the gallery stays clean.** Enrolment in the app is
a deliberate, usually cooperative act — the operator is looking at the frame
when they name it — while the frames it later has to match are whatever the
camera produced.

Look at `results/degraded_examples.png` before trusting any degraded number. A
degradation nobody eyeballed is a degradation nobody can defend; if those do
not look like your camera, tune `--degrade-kinds` rather than believing the
table.

The degraded arm is a **stress test with invented severity**, not a measured
model of any particular camera.

## Why the numbers transfer

`evaluate.py` imports `MODEL_NAME`, `DET_SIZE`, `STRONG_MATCH` and
`WEAK_MATCH` directly from `backend/engine.py` rather than restating them, so
the harness cannot silently drift from the running app. If you change a
threshold in the backend, re-running this reports the new operating point with
no edits here.

Embeddings are produced exactly as the app produces them: the same InsightFace
pack, same `det_size`, same `normed_embedding`. Matching is the same cosine
similarity (a dot product on L2-normalised vectors), and the same
max-then-argmax decision.

The one place the harness deliberately differs is **which detection an image's
label belongs to** when a frame holds several people — a question about the
dataset's ground truth, which the app never has to answer because it labels
every detected face independently. See [`--face-select`](#multi-face-frames---face-select);
it is worth 2.1 points of rank-1, so it is not a detail.

## Install

```bash
pip install -r backend/requirements.txt
pip install -r eval/requirements.txt
```

## Run

```bash
python eval/evaluate.py
```

First run downloads LFW (~200 MB) into scikit-learn's data cache and embeds
every image on CPU. Embeddings are then cached under `eval/.cache/`, keyed by a
hash of (model, det_size, dataset config), so reruns with different splits,
seeds, gallery sizes or thresholds are fast. Pass `--no-cache` to force
recomputation.

For a quick pass while iterating:

```bash
python eval/evaluate.py --max-identities 15 --max-images-per-identity 20 --gallery-sizes 4,8,12 --impostor-identities 3
```

Any subsampling is recorded in `metrics.json` under `config`, so a number can
always be traced back to the run that produced it.

### Options that matter

| Flag | Default | Effect |
|---|---|---|
| `--dataset` | `lfw` | `lfw` or `cfp-fp`. |
| `--cfp-root` | — | Path to the extracted `cfp-dataset` folder. Required for `--dataset cfp-fp`. |
| `--cfp-protocol` | `fp` | `fp` enrols frontal / probes profile; `mixed` ignores pose. |
| `--cfp-max-side` | 0 | Downscale CFP images above this long side (0 = off). Part of the cache key. |
| `--min-faces` | 10 | `min_faces_per_person`. 10 gives 158 identities / 4324 images; 20 gives 62 / 3023. Lower = more identities = more probes and a better-resolved FAR tail. |
| `--split-mode` | `gallery-first` | `gallery-first` enrols a few images and probes with all the rest; `probe-first` is the original fixed-probe-count behaviour. |
| `--enrol-count` | **1** | Images enrolled per person for the headline result. 1 because `engine.register` stores exactly one webcam frame. (`--gallery-per-identity` is accepted as an alias.) |
| `--enrol-counts` | `1,3,5` | Enrolment counts swept by the open-set protocol. |
| `--degrade` / `--no-degrade` | on | Score a second arm with corrupted probes. Costs one extra embedding pass, cached separately. |
| `--degrade-family` | `heldout` | Which corruption group degrades the probes: `heldout` (evaluation), `augmentable` (reserved for the augmentation), `all` (circular). |
| `--degrade-kinds` | from family | Explicit override, e.g. `jpeg,downscale`. |
| `--augmented-with` | — | Declare the augmentation's corruption kinds. The run **aborts** on any overlap with what is being evaluated. |
| `--gallery-sizes` | `10,25,50` | Gallery sizes swept by the open-set protocol, in identities. |
| `--impostor-identities` | 80 | Identities held out of every gallery. This number × their probe count is what resolves the open-set FAR tail. |
| `--openset-trials` | 5 | Independent identity draws; pooled. |
| `--deploy-gallery-size` | 50 | Gallery size the recommended threshold assumes. |
| `--openset-target-far` | 0.001 | Open-set FAR budget for the recommendation. |
| `--face-select` | `centred` | Which detection carries the label in a multi-face frame. Worth 2.1 pp of rank-1 — see below. |
| `--max-pairs` | 25000 | Cap per class for the **balanced** ROC/EER pairs. FAR is always computed on the full impostor pool regardless. |
| `--target-far` | 0.001 | FAR budget for the *verification* recommended threshold (reported, not deployed). |
| `--seed` | 42 | Controls subsampling, the split, pair sampling and every open-set draw. Recorded in the output. |

## Determinism

Fixed seed everywhere (subsample, split, pair sampling, per-trial identity
draws derived as `seed * 1000 + trial`), stable sorts, and deterministic CPU
inference. The same command with the same seed produces the same
`metrics.json`. The seed is written into the output.

## The embedding cache

The cache key hashes only what changes an **embedding**: model, `det_size`,
and the dataset configuration. Nothing from the open-set protocol, the power
analysis or the failure grids is in it, because none of them change a single
embedding — they only change how the embeddings are split and scored. So all
of that is free to re-run.

The key is also **append-only by construction**. The original payload is
frozen; new embedding-affecting options go into an `extras` sub-dict that is
merged *only when non-empty*, so a configuration that predates a new option
still hashes to its old digest and keeps its cached `.npz`. A run with
`--min-faces 20 --slice full` still resolves to `emb_11cef27fa9003591.npz`, the
file the original baseline used.

Three things live in `extras` so far:

* `det_score_min` — only when non-zero.
* `variant` — the degraded arm, so it caches beside the clean one.
* `dataset` — **only when the dataset is not LFW**, which is exactly what keeps
  every LFW digest byte-identical. For CFP-FP it carries the dataset name, the
  `--cfp-max-side` setting, and a **content hash** over the file list and
  sizes, so a different or partial copy of the dataset gets its own cache entry
  instead of silently reusing embeddings computed from other images.
  `--cfp-protocol` is deliberately *absent*: it picks the split, not the
  pixels.

The `.npz` payload itself grew — it now also stores the centred-face
embeddings, the detected-face count per image, and whether the two selection
rules agreed. Caches written before that are still readable; they just cannot
serve `--face-select centred`, and the harness says so instead of guessing.

## The dataset trap this harness guards against

`fetch_lfw_people` returns float images already divided by 255, i.e. in
`[0, 1]`. Calling `.astype('uint8')` on that truncates every pixel to zero.
That is precisely the bug recorded in `legacy/README.md` which produced 3023
all-black JPEGs in the abandoned `known_faces/` folder.

`to_uint8_bgr()` multiplies by 255 *before* casting, then refuses to continue
unless the result is genuinely non-black: it fails if the whole array is zero,
if more than 1% of images are individually black, or if the mean pixel value is
below 1.0. A silent repeat of that bug would produce plausible-looking but
meaningless metrics, so it aborts instead.

The harness never reads `known_faces/`.

## Why `--slice full` is the default (a deliberate deviation)

The obvious call is
`fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0)`, using
sklearn's default `slice_`. **That configuration measures nothing**: it crops
each image to a tight 125×94 face with no surrounding context, and RetinaFace
does not fire on a face that fills the entire frame.

Measured over 25 images, same model and `det_size` throughout:

| Input framing | Faces detected |
|---|---|
| Tight crop (sklearn default `slice_`) | **0/25 (0%)** |
| Tight crop + 50% replicated border | 24/25 (96%) |
| Tight crop + 100% replicated border | 25/25 (100%) |
| **Full 250×250 LFW frame** | **25/25 (100%)** |

So the harness defaults to `--slice full`, which passes the whole LFW image
including background. This is the honest fix rather than the convenient one:
padding a crop with a replicated border also works, but it fabricates image
content, whereas the full frame is the real photograph and is closer to what
the app actually sees — a face occupying part of a webcam frame, not the whole
of it.

`--slice tight` reproduces the 0% result if you want to confirm it. Both
settings are recorded in `metrics.json` under `config.slice`, and the setting
is part of the embedding cache key.

Note that the full frame at `resize=1.0` makes the float32 array sklearn
allocates roughly 3.2 GB at `--min-faces 10`. The uint8 conversion is chunked
to avoid doubling that, but if memory is tight, lower `--resize` (at the cost
of a smaller face and weaker embeddings) or raise `--min-faces`.

## Multi-face frames: `--face-select`

Passing the full frame has a consequence the tight crop hid. Plenty of LFW
photographs contain **more than one person** — a head of state at a podium with
an aide over their shoulder. `encode_faces.py` keeps the highest-`det_score`
face, which is the right rule for the app, but LFW's *label* names the
**centred** subject. When the two disagree, the harness embeds one person's
face and scores it against another person's name, and every resulting mismatch
is counted as a recognition error when it is really a ground-truth defect.

This was not a hypothesis. It was the single most common pattern in the first
`results/failures/closed_set_rank1_errors.png` produced by this harness —
frame after frame of "true: Jose Maria Aznar / pred: Silvio Berlusconi" where
Berlusconi is simply the more prominent face in Aznar's photograph. Finding it
is exactly what error inspection is for.

Measured on the default configuration:

| | |
|---|---|
| Frames with more than one detected face | **14.68%** |
| …where the highest-`det_score` face is **not** the centred one | **2.04%** |
| Rank-1 under `best-score` | 97.76% (86 errors) |
| Rank-1 under `centred` | **99.87%** (5 errors) |

**2.04% of mislabelled images cost 2.11 percentage points of rank-1** — 81 of
the 86 apparent errors were ground-truth defects, not matcher failures. Left
uncorrected it also inflates EER twelvefold (2.78% → 0.226%) and pushes the
recommended threshold from 0.386 to 0.563, buying a 10% false-reject rate to
solve a problem the matcher did not have.

`app.get()` already returns an embedding for every detected face, so the
harness computes **both** selections in the one detection pass and stores both
in the cache:

* `--face-select centred` (default) — attaches the label to the detection
  nearest the frame centre, which is what LFW means.
* `--face-select best-score` — `encode_faces.py`'s rule. Reproduces the
  contaminated numbers above.

This changes *which detection the label refers to*, not how any embedding is
computed, so it is not a deviation from the app's matcher — it is a fix to the
harness's ground truth. Selecting between them costs nothing and needs no
re-embedding, which is why the selection is deliberately **not** part of the
cache key. Every run reports the multi-face rate, the disagreement rate, and
rank-1 under **both** rules, so the size of the effect stays on the table.

(For the app itself this is a non-issue: `engine._recognize` labels every
detected face independently, so it never has to choose. It matters only where a
dataset attaches one label to a frame holding several people.)

## Splits

Split by identity **and** by image — an image is either a probe or a gallery
entry, never both, and the script asserts the two index sets are disjoint.
Identities without enough successfully detected images are dropped and counted
in the output.

The default is `gallery-first`: enrol `--gallery-per-identity` images and probe
with everything else. This is both the higher-power split (it maximises probe
count — the point of gap 2 below) and the more faithful one. The app enrols a
person from **one** click on **one** frame (`engine.register`), or from a
handful of photos (`encode_faces.py`), and then matches every subsequent frame
against that. The original `probe-first` split held out 5 probes and put the
other ~45 images in the gallery, which flatters the matcher in a way the
deployment never will. `--split-mode probe-first` reproduces it.

## Statistical power

Every rate in `metrics.json` is reported as `{count, n, rate, ci_low, ci_high}`
with a **Wilson** 95% interval, not as a bare number. Wilson rather than the
normal/Wald interval because every rate that matters here sits near 0 (FAR) or
near 1 (rank-1), which is exactly where Wald misbehaves — it puts part of the
interval below zero for 4/310, and gives 0/1350 an interval of zero width.

Two numbers deliberately carry no interval: **AUC** and **EER**. Neither is a
binomial proportion — they are summaries over a whole curve — so a Wilson
interval would be wrong rather than merely approximate. Compare them between
runs at a fixed seed and split; do not read a confidence range into them.

The run also reports the **minimum detectable difference** in rank-1 accuracy,
which is the number to check before planning an improvement:

* **Paired** (the new system re-run on *these same probes*, the normal case):
  by McNemar / exact sign test, a change must net-fix at least 6 errors to
  reach p < 0.05 — and that is the *best* case, assuming it breaks nothing.
  Divided by the probe count, that is the smallest measurable improvement.
* **Unpaired** (two independent evaluations): the standard two-proportion
  z-test bar at 80% power, which is several times larger.

Anything smaller than these is not measurable at this sample size, however real
it may be. Under the old 310-probe configuration the paired bar was ~1.9
percentage points against a 1.3% error rate — i.e. the harness could only
detect a change that eliminated *most* of the remaining errors.

If the observed error count falls **below** the paired test's requirement, the
harness says so outright (`statistical_power.at_ceiling`) instead of quoting a
threshold nobody can meet: at that point the constraint is the dataset, not the
sample size. That is the situation on the current defaults — see the baseline
below.

### LFW rank-1 is saturated. Here is what replaces it.

**Rank-1 on LFW cannot measure anything any more.** At 1 enrolment image and a
158-identity gallery it sits at 99.91% with **3 errors in 3523 probes**. A
paired McNemar test needs at least 6 net errors fixed to reach p < 0.05, so
there are not enough errors in the benchmark to reach significance *even if a
change fixed every single one*. This is a ceiling, not a sampling problem: more
probes cannot help, because the errors do not exist to be found. The harness
says so itself in `statistical_power.at_ceiling` rather than quoting a
threshold nobody can meet.

The clean open-set FRR has more room — 78 errors — but a change must net-fix 6
of those 78, which is 7.7% of every error the benchmark contains. That is a
thin margin for anything short of a large effect.

**So the primary metric for the augmentation ablation is open-set FRR at a
fixed FAR ≤ 0.1%, N=1 enrolment, gallery 50, measured on the held-out
corruption family.** Corrupting the probes manufactures errors — that is the
point — and confining the corruption to a family the augmentation is forbidden
to imitate is what keeps those errors meaningful.

#### What this metric can and cannot support

It **can** support: *"an enrolment augmentation confined to `illumination` and
`pose_warp` improved open-set false-reject rate against `motion_blur`,
`downscale` and `jpeg` — corruptions it never saw — at a held-constant false
accept rate."* That is a real generalisation claim, machine-checked to be
non-circular, with a stated MDD.

It **cannot** support:

* **Anything about real pose.** `pose_warp` is a planar perspective transform,
  not a head rotation, and it is in the augmentable group anyway so it is never
  evaluated on. Pose claims need CFP-FP or real captures.
* **Anything about your camera.** The corruption severities are invented. The
  arm bounds how much worse things get under *these* distortions; it is not a
  measurement of any real capture pipeline.
* **That this corruption is the one that matters.** Nothing here establishes
  that motion blur and JPEG are the dominant real-world failure modes. They
  were chosen because they are disjoint from what the augmentation may imitate.
* **A threshold.** The deployed threshold is set on the clean arm. The degraded
  arm reports what that threshold costs under stress; it does not move it.

The disjointness is between *effect families*, not between causes — augmenting
with illumination could improve general robustness that also helps under JPEG.
That is fine, and is in fact the claim being made; it just means the result is
about transfer, not about JPEG specifically.

`augmentation_target.ablation_primary_metric` in `metrics.json` carries this
metric, its Wilson interval, its MDD, and both of these lists, so the claim and
its limits travel with the number.

#### Why *FRR at a fixed FAR*, and how the metric is picked

Hold the false-accept budget constant, re-derive the threshold that achieves
it, and report what fraction of enrolled people get rejected there. Pinning FAR
is what stops an "improvement" that is really just a looser threshold, and FRR
is the number an operator actually feels.

`augmentation_target` in `metrics.json` ranks every candidate metric by **how
many errors are available to fix** — that, not the sample size, is what bounds
a paired test — and returns a verdict naming the one to track, or states
plainly that none of them qualify. `ablation_primary_metric` is the single
field to read.

One caveat the harness states in its output and repeats here: because the
threshold is *re-derived* to hold FAR, a change that moves the score
distribution also moves the threshold, so the before/after comparison is not
strictly paired at the decision level. The paired MDD assumes a **fixed**
threshold. When re-deriving, quote the unpaired bar instead — or hold the
threshold at its current value and report FAR alongside FRR so both halves of
the trade are visible.

## The FAR tail

Balanced genuine/impostor pairs are the right input to a ROC (an ROC over a
1:150 imbalanced set is dominated by the impostor class and its AUC is not
comparable to published figures), but they are the **wrong** input to a FAR
estimate, because balancing throws away almost the entire impostor tail — and
the tail is the whole point of a FAR figure. The original baseline reported
FAR = 0.0074%, which was **one pair** out of 13530 after discarding 811,800
available impostor pairs.

So the two are now computed from different samples, deliberately:

* **ROC, AUC, EER** — the balanced subsample (`--max-pairs` per class).
* **Every FAR figure** — the *full* impostor pool, every non-matching
  probe×gallery pair.

Every FAR in `metrics.json` carries `count` and `n`, so the support behind it
is never in doubt, and the printout shows the balanced subsample's answer
alongside the full-pool one wherever they disagree.

For open-set FAR the equivalent honesty is `far_resolution_per_trial` versus
`far_resolution_pooled`. Pooling across trials pools mostly-distinct probe
images, but not fully independent ones — an image can be scored in more than
one trial against a different gallery. **Read the per-trial resolution as the
floor on a single FAR measurement**; the Wilson intervals on pooled counts are
correspondingly a little optimistic.

If `--openset-target-far` is finer than that floor — the budget allows less
than one impostor probe per trial — the run sets
`recommendation.under_resolved` and prints a warning. In that regime "FAR = 0%"
is the absence of evidence rather than a measurement, and the recommended
threshold is wherever the single highest-scoring impostor happened to land. The
fix is more impostor probes (`--impostor-identities`, a lower `--min-faces`),
not a tighter budget.

## Reading the output

Results land in `eval/results/`.

### `metrics.json`

Everything below, plus the seed, the full config, the resolved backend
settings, environment versions, and the embedding cache key.

### `openset_far_frr_mir.png` — **the plot that sets the threshold**

Three stacked panels — FAR, FRR and misidentification rate against threshold —
with one line per gallery size. Read it as: pick a FAR you can live with on the
top panel, read the threshold off the x-axis, then read what it costs you on
the two panels below. The spread between the gallery-size lines in the top
panel is the effect that makes the verification ROC unusable for this job.

### `enrolment_count.png`

Left: the threshold each enrolment count needs to hold the FAR budget. Right:
the FRR it pays there. One line per arm × gallery size, so you can see whether
the penalty for enrolling a single frame is uniform or gets worse as the
gallery grows. The dotted vertical line is what the app actually does.

### `degradation_comparison.png`

Clean vs. degraded probes at the deployed configuration, with the detector-miss
and end-to-end miss rates annotated. Pair it with `degraded_examples.png`.

### `degraded_examples.png`

Eight clean probes over their degraded counterparts. Look at this before
trusting the degraded numbers.

### `openset_gallery_size.png`

The same finding, collapsed. Left: open-set FAR at a *fixed* threshold as the
gallery grows — a flat line would mean gallery size does not matter, and it is
not flat. Right: the threshold needed to hold the FAR budget at each gallery
size, and the FRR that threshold costs.

### `roc_curve.png`

True accept rate against false accept rate across all thresholds, on the
balanced pairs. **AUC** in the legend summarises separability (1.0 = perfect,
0.5 = chance). The crimson dot is the **EER**. Use both to compare model
changes; do not use either to pick an operating point.

### `far_frr_vs_threshold.png`

The verification decision curve, with FAR computed on the full impostor pool.
Log y-axis, because the interesting FAR region is small. The dotted grey line
is the one-pair floor — anything touching it is a single-pair measurement.
Vertical lines mark the EER threshold, the verification-recommended threshold,
the **open-set** recommended threshold, and the two thresholds currently
hardcoded in the backend.

For attendance, FAR is the expensive error: a false accept marks the wrong
person present, silently, and permanently (`_mark` is first-write-wins). A
false reject is visible and self-correcting — the person stands there until it
recognises them.

### `score_distributions.png`

Genuine and impostor cosine similarities overlaid, full pool. The overlap
region *is* the error: no threshold can separate what overlaps.

### `cmc_curve.png`

Closed-set identification accuracy at rank k — every identity enrolled, so it
measures **ranking only** and asks no accept/reject question. Rank-1 is "the
top match is right". A large rank-1/rank-5 gap means the embedding knows the
identity but is not confident about ordering.

### `top2_margin.png`

For each probe, (best identity score − runner-up identity score), split by
whether rank-1 was correct. If correct predictions cluster at a clearly higher
margin than incorrect ones, then a rule like *"reject unless the best match
beats the runner-up by m"* would cut errors, and the histograms show what `m`
costs. If the two distributions sit on top of each other, a margin test cannot
help and the idea should be dropped — which is just as useful an answer.

### `failures/`

Every misclassified probe, as labelled image grids. Each cell is the **probe**
on top and the gallery photo its top match actually came from underneath, so
"these two genuinely look alike" can be told apart from "the probe is a blurry
profile" at a glance. Captions carry true identity, predicted identity, top-1
similarity and top-2 margin; a probe whose frame held more than one face is
flagged, because that is a ground-truth problem rather than a matcher problem
(see `--face-select`).

| File | Contents |
|---|---|
| `closed_set_rank1_errors.png` | Rank-1 errors with every identity enrolled. |
| `openset_false_accepts.png` | Impostor identities that cleared the recommended threshold. |
| `openset_misidentifications.png` | Enrolled people accepted as the wrong person. |
| `openset_false_rejects.png` | Enrolled people rejected. |
| `failures.json` | **Every** failure, not just the plotted ones. |

Grids are ordered most-confident-first, because a confident error is the one
that actually marks the wrong person present, and capped at
`--max-failure-images` per category (paged if needed).

## The two baselines, side by side

**LFW is measured. CFP-FP is implemented and tested but not yet run** — the
dataset is not on this machine. Everything below the LFW column is produced by
one command; the CFP-FP column fills in from another.

```bash
python eval/evaluate.py                                            # LFW
python eval/evaluate.py --dataset cfp-fp --cfp-root <path>         # CFP-FP
```

| | **LFW** (measured, seed 42) | **CFP-FP** (not yet run) |
|---|---|---|
| Identities | 158 | 500 |
| Images | 4324 | 7000 (5000 frontal + 2000 profile) |
| Enrolment | 1 random image | 1 **frontal** image |
| Probes | 3523 (all non-enrolled) | 2000 (all **profile**) |
| Detection failure | 0.25% [0.14%, 0.45%] | — |
| Rank-1 | 99.91% [99.75%, 99.97%] — **3 errors** | — |
| ROC AUC / EER | 0.9998 / 0.170% | — |
| Recommended threshold @ gallery 50 | **0.370** | — |
| Open-set FAR there | 0.099% [0.050%, 0.195%] (8/8090) | — |
| **Open-set FRR @ FAR ≤ 0.1%**, clean | **1.27%** [1.02%, 1.58%] — **78 errors / 6143** | — |
| **Open-set FRR @ FAR ≤ 0.1%**, held-out corruptions | **17.41%** [16.48%, 18.38%] — **1,065 errors / 6118** | — |
| Misidentification | 0.02% (1/6143) | — |
| Paired MDD on the held-out FRR | 0.098 pp (6 net errors, 0.6% of those available) | — |

### Which one governs what

* **LFW sets the deployed threshold.** The app enrols a cooperative, roughly
  frontal webcam frame, and LFW is the closer match to that distribution.
  0.370 at 1 enrolment image and a 50-identity gallery is the number for
  `engine.py`, and it does not change because a second benchmark exists.
* **The held-out corruption arm on LFW governs the augmentation ablation
  today.** LFW's *clean* metrics cannot carry that experiment: rank-1 has 3
  errors (below the 6-error floor for any paired test) and clean open-set FRR
  has 78, so a change must net-fix 7.7% of every error the benchmark contains.
  Degrading the probes with the **held-out** family manufactures errors without
  making the comparison circular — 1,065 of them — which is what turns the
  ablation into something measurable. See
  [the augmentation target](#the-augmentation-target-gap-3).

* **CFP-FP is what a *pose* claim would need.** The held-out arm shows transfer
  to unseen corruptions; it says nothing about real head rotation, because the
  only pose-like effect (`pose_warp`) is a planar warp and is reserved for the
  augmentation rather than evaluated on. Frontal-enrol / profile-probe on
  CFP-FP is precisely the failure a single enrolment frame causes, and
  `buffalo_sc` is not saturated on it. If you run it, compare the `errors`
  count for *open-set FRR @ FAR ≤ 0.1%* against the numbers above — the same
  `mdd_for()` computes the bar for all three.

### What to expect, and what to do about it

Published CFP-FP verification accuracy for small mobile ArcFace packs sits far
below their LFW figures, so the open-set FRR here should be substantially
higher than LFW's 1.27% and the error count correspondingly larger. That is the
point — but it is an expectation, not a measurement, and nothing in this file
reports a CFP-FP number until the run produces one.

Two things to check on the first CFP-FP run, because they are the ways this
protocol can go wrong rather than merely come out badly:

* **Detection failure on profile images.** RetinaFace can miss extreme profiles
  outright. That shows up as `detection_failure_on_probes`, is reported
  separately from FRR, and is *not* something enrolment augmentation can fix —
  it happens before any gallery comparison. If it is large, the honest headline
  is `end_to_end_miss_rate`, not FRR.
* **FAR resolution.** CFP gives 4 profile probes per identity, so the impostor
  pool is `--impostor-identities × 4`. At the default 400 that is 1600 probes
  per trial, a 0.0625% floor against the 0.1% target — adequate, but only just.
  If `recommendation.under_resolved` is true, raise `--impostor-identities`
  before trusting the threshold.

## LFW baseline (seed 42, defaults)

`python eval/evaluate.py` — 158 identities, 4324 images, **1 enrolment image
per person**, 3523 probes, 158 gallery entries, clean and degraded arms.

Degraded figures below are the **held-out** family (`motion_blur`, `downscale`,
`jpeg`) — the evaluation group.

| Metric | Value (95% Wilson CI) |
|---|---|
| Detection failure rate (clean) | 0.25% [0.14%, 0.45%] (11/4324) |
| Detection failure rate (degraded, held-out) | 0.53% [0.35%, 0.80%] |
| ROC AUC | 0.9998 |
| EER | 0.170% @ threshold 0.238 |
| Rank-1, clean (158-identity gallery) | 99.91% [99.75%, 99.97%] — **3 errors** |
| Rank-1, degraded (held-out) | 99.15% [98.78%, 99.40%] — 30 errors, plus 12 undetected |

### The threshold that goes into `engine.py`

**0.370**, assuming **1 enrolment image per person** and a gallery of
**50 enrolled identities**.

| At 0.370, gallery 50 | Clean probes | Degraded (held-out family) |
|---|---|---|
| Open-set FAR | 0.099% [0.050%, 0.195%] (8/8090) | 0.099% (8/8065) |
| FRR | **1.27%** [1.02%, 1.58%] | **17.41%** [16.48%, 18.38%] |
| Misidentification | 0.02% [0.00%, 0.09%] | 0.016% |
| Probe detection loss | 0.00% | 0.41% |
| End-to-end miss rate | 1.27% | 17.74% |

The clean arm sets the threshold; the degraded arm is a stress test with
invented severity and bounds how much worse it gets. **Both are worth
knowing**: a threshold calibrated on clean press photography rejects one frame
in six that is blurred, downscaled and JPEG-compressed — of people who *are*
enrolled. Attendance still
gets marked — the person stays in front of the camera and recognition runs
every third frame — but the first-frame hit rate is not what the clean number
suggests.

### What single-image enrolment costs (gap 1)

Threshold re-derived to hold FAR ≤ 0.1% at each enrolment count, shown as
*threshold / FRR*:

| Enrol N | Gallery 10 | Gallery 25 | Gallery 50 |
|---|---|---|---|
| **Clean** | | | |
| 1 | 0.279 / 0.25% | 0.370 / 1.93% | **0.370 / 1.27%** |
| 3 | 0.296 / 0.13% | 0.371 / 0.43% | 0.378 / 0.34% |
| 5 | 0.296 / 0.00% | 0.410 / 0.43% | 0.412 / 0.29% |
| **Degraded (held-out family)** | | | |
| 1 | 0.268 / 4.85% | 0.363 / 19.39% | **0.363 / 17.41%** |
| 3 | 0.283 / 2.42% | 0.369 / 9.26% | 0.372 / 8.86% |
| 5 | 0.284 / 1.91% | 0.392 / 10.08% | 0.395 / 9.30% |

**Single-image enrolment costs about 3.7× the false-reject rate** of three
images at gallery 50 (1.27% vs 0.34%), and about 2.0× on held-out degraded
probes (17.41% vs 8.86%). The earlier 0.386 / 0.42% was measured at three
enrolment images and was optimistic by exactly this factor.

Two things the sweep shows that are worth not mis-reading:

* **More enrolment images also raise the threshold.** N=5 needs 0.412 where
  N=1 needs 0.370, because every extra enrolled embedding is another chance for
  a stranger to score high — the same mechanism that makes FAR grow with
  gallery size. That is why degraded FRR at N=5 (9.30%) is *worse* than at
  N=3 (8.86%): the stricter threshold ate the benefit. Enrolling more frames
  is not monotonically good.
* **FRR is not comparable across gallery sizes.** FAR is — the same impostor
  probes are scored against every gallery. But the genuine probe set *is* the
  enrolled identities' probes, so its composition changes with the gallery.
  Compare FRR across sizes only at a fixed threshold (the `at_recommended`
  rows), never across the `threshold_for_target_far` rows, which move for two
  reasons at once. `metrics.json` carries this as
  `note_frr_across_gallery_sizes`.

### The complete threshold comparison (gap 2)

Open-set terms throughout — clean arm, 1 enrolment image, gallery 50. FRR and
misidentification are measured under *this* protocol, not carried over from the
retired verification run.

| Threshold | Open-set FAR | FRR | MISID | Verdict |
|---|---|---|---|---|
| 0.271 (verification's answer for FAR ≤ 0.1%) | **4.957%** (401/8090) | 0.11% | 0.02% | 50× its FAR budget. The verification ROC cannot set this threshold. |
| 0.32 (`WEAK_MATCH`) | **1.335%** (108/8090) | 0.36% | 0.02% | One stranger frame in 75 accepted. Dangerous. |
| **0.370 (recommended)** | **0.099%** (8/8090) | **1.27%** | 0.02% | Meets the budget at the cost the FRR column states. |
| 0.50 (`STRONG_MATCH`) | 0.000% (0/8090) | **9.88%** | 0.02% | Safe on FAR, rejects one in ten genuine frames. |

So the cost of the existing pair, stated properly:

* **`WEAK_MATCH = 0.32` accepts 1.34% of stranger frames.** With recognition
  running every third frame and `_mark()` first-write-wins, a stranger in view
  for a few seconds is very likely to be marked present as somebody else. This
  is the setting to change.
* **`STRONG_MATCH = 0.50` rejects 9.88% of genuine frames** to buy a FAR
  improvement of 0.099 pp over 0.370. On this data it is over-strict by a wide
  margin.
* A single threshold at **0.370** is simultaneously safer than 0.32 (13× less
  FAR) and far less strict than 0.50 (7.8× less FRR). The `uncertain` band
  between them is not buying anything these numbers can see.

### The augmentation target (gap 3)

Measured at the deployed configuration — 1 enrolment image, gallery 50. The
`errors` column is what bounds a paired test, not the sample size.

| Candidate metric | Errors | n | Paired MDD | Usable |
|---|---|---|---|---|
| **open-set FRR @ FAR ≤ 0.1% (degraded, held-out)** | **1,065** | 6,118 | **0.098 pp** | **yes — the primary metric** |
| open-set FRR @ FAR ≤ 0.1% (clean) | 78 | 6,143 | 0.098 pp | yes, but thin |
| closed-set rank-1 error (degraded, held-out) | 30 | 3,511 | 0.171 pp | yes |
| probe detection failure (degraded) | 25 | 6,143 | 0.098 pp | no — enrolment cannot fix a detector miss |
| open-set FAR @ recommended threshold (either arm) | 8 | ~8,080 | 0.074 pp | marginal |
| closed-set rank-1 error (clean) | 3 | 3,523 | 0.170 pp | no — below the 6-error floor |
| misidentification (either arm) | 1 | 6,118 | — | no |

**The primary metric for the augmentation ablation: open-set FRR at fixed
FAR ≤ 0.1%, N=1 enrolment, gallery 50, held-out corruption family.**

| | |
|---|---|
| Threshold | 0.3634 (re-derived to hold FAR at the budget) |
| FAR held at | 0.099% [0.050%, 0.196%] (8/8065) |
| **FRR** | **17.41% [16.48%, 18.38%]** — 1,065 errors in 6,118 probes |
| Misidentification | 0.016% |
| Probe detection loss | 0.41% [0.28%, 0.60%] |
| End-to-end miss rate | 17.74% [16.81%, 18.72%] |
| Paired MDD | **0.098 pp** — net-fix 6 of 1,065 errors (0.6%) |

That is **13.7× the headroom** of the clean arm's 78 errors, and unlike the
old composite arm it is not circular: the corruption family is disjoint from
what the augmentation is permitted to imitate, and `--augmented-with` makes the
harness abort if that is ever violated. `metrics.json` carries it as
`augmentation_target.ablation_primary_metric`, together with the `supports` /
`does_not_support` lists so the claim and its limits travel with the number.

Rank-1 on the clean arm remains unusable: 3 errors, below the 6-error floor for
any paired test. That has not changed and will not change by adding probes.

**What LFW still cannot do.** Manufacturing errors with held-out corruptions
gives the ablation a metric with room to move, but it does not make LFW a pose
benchmark. If the augmentation is meant to cover **pose**, this arm cannot
demonstrate it — `pose_warp` is a planar approximation and sits in the
augmentable group precisely because it is too easy to imitate. For that claim,
add a harder benchmark:

| Benchmark | Why it fits |
|---|---|
| **CFP-FP** | 500 identities with explicit frontal-vs-profile pairs. `buffalo_sc` scores far lower here than on LFW, so pose-driven errors are plentiful — the right choice if the augmentation is meant to cover pose. |
| **AgeDB-30** | Age-gap verification at a 30-year separation. Isolates age drift, the other axis a single enrolment frame fails on, and the one that matters for a gallery enrolled once and used for years. |
| **IJB-C** | Template-based 1:N protocol with a genuinely large gallery and published open-set (FPIR/FNIR) operating points. The closest public analogue to what this harness already measures, and the one to use if the FAR tail is the target. |

`augmentation_target.verdict` in `metrics.json` and this table come from the
same computation, so they cannot drift apart.

### Statistical power

3523 probes give a rank-1 CI half-width of ±0.11 pp. A paired McNemar
comparison needs 6 net errors fixed, which is 0.17 pp of rank-1 — but rank-1
has only 3 errors, so `statistical_power.at_ceiling` fires and no rank-1
improvement is measurable at any sample size. See the augmentation target above
for what to measure instead.

### Caveats

* **LFW is not your deployment.** Web photos of public figures, not your webcam
  under your lighting. The degraded arm is an *invented* stress test, not a
  model of your camera — look at `results/degraded_examples.png` and adjust
  `--degrade-kinds` if it does not match what you see.
* **8 false accepts is what the 0.1% figure rests on.** Per-trial impostor FAR
  resolution is 0.078% (1288 impostor probes), so the recommended threshold is
  resolved to about that granularity. The Wilson interval [0.050%, 0.195%] is
  the honest range.
* **The gallery sweep stops at 50 and the enrolment sweep at 5.** Past either,
  re-run; nothing here is extrapolated.
* **These are per-decision rates.** The app decides every third frame and
  `_mark()` is first-write-wins, so per-session false-accept risk is much
  higher than the per-decision FAR, and per-session false-reject risk is much
  lower than the per-decision FRR. Nothing here models that; it is a reason to
  treat the FAR budget as tight and the FRR figure as pessimistic for a
  cooperative user who stays in frame.
