# Measured results

What this system's numbers actually are, and which of them exist yet.

Two features make accuracy claims. One is measured; one is not, and this file
says so plainly rather than leaving the gap to be inferred.

| Feature | Status | Evidence |
|---|---|---|
| **Face matching** (threshold, FAR/FRR) | **Measured** | `eval/evaluate.py`, run 2026-09-05, seed 42 |
| **Liveness / anti-spoofing** | **Not measured** — code written, provider never called, no test set captured | `eval/liveness/` exists and is ready to run |

---

## 1. Face matching — measured

Source: `eval/results/metrics.json` (seed 42, LFW, 158 identities, 4324 images,
embedding cache `b9cf72cad14bf2bb`). Full method in
[`eval/README.md`](eval/README.md).

Protocol is open-set identification — max similarity over the whole gallery,
argmax for the name, accept above threshold — which is exactly what
`AttendanceEngine._recognize` does. Operating point: **1 enrolment image per
person** (what `engine.register` actually stores), **gallery of 50 identities**,
5 trials, 80 impostor identities held out of every gallery.

### The deployed threshold

**`STRONG_MATCH = 0.370`** — the lowest threshold holding open-set FAR inside a
0.1% budget.

| At 0.370, gallery 50 | Clean probes | Degraded (held-out family) |
|---|---|---|
| Open-set FAR | 0.099% [0.050%, 0.195%] (8/8090) | 0.099% (8/8065) |
| **FRR** | **1.27%** [1.02%, 1.58%] | **17.41%** [16.48%, 18.38%] |
| Misidentification | 0.02% [0.00%, 0.09%] | 0.016% |
| Probe detection loss | 0.00% | 0.41% |
| End-to-end miss rate | 1.27% | 17.74% |

Other headline figures:

| Metric | Value (95% Wilson CI) |
|---|---|
| Detection failure rate (clean) | 0.25% [0.14%, 0.45%] (11/4324) |
| ROC AUC | 0.9998 |
| EER | 0.170% @ threshold 0.238 |
| Rank-1, clean (158-identity gallery) | 99.91% [99.75%, 99.97%] — **3 errors** |
| Rank-1, degraded (held-out) | 99.15% [98.78%, 99.40%] — 30 errors |

### What changed, and what it bought

`STRONG_MATCH` was a hand-picked `0.50`. Measured at the same operating point,
that cost **9.88% FRR** — roughly one genuine frame in ten rejected — to buy
0.087 pp of FAR the measurement cannot even resolve (0 vs 7 impostor probes out
of 8090). Moving to 0.370 cut FRR **7.5×** to 1.32%.

Separately, `WEAK_MATCH = 0.32` no longer writes an attendance record. Its FAR
is 1.335% (108/8090) — about one stranger frame in 75 — and that was being
written straight into the register.

### Limits of the matcher numbers

* **LFW is not this deployment.** Web photos of public figures, not your webcam
  under your lighting. The degraded arm is an *invented* stress test.
* **Rank-1 on LFW is saturated** — 3 errors in 3523 probes, below the 6-error
  floor for any paired test. No rank-1 improvement is measurable at any sample
  size.
* **The threshold assumes one template per person.** Open-set FAR scales with
  total template count, not identity count; N=5 needed 0.412 where N=1 needs
  0.370. The engine warns through `/api/status` the moment anyone holds two.
* **These are per-decision rates.** Recognition runs every third frame and
  `_mark()` is first-write-wins, so per-session false-accept risk is much higher
  than the per-decision FAR.

---

## 2. Liveness / anti-spoofing — NOT measured

**No numbers exist for this feature, and none are estimated below.**

### What is true today

* `backend/liveness.py` is written: provider selection, prompt, strict JSON
  contract, one retry, fail-closed, tracked-face caching, off-thread execution.
* Enforcement is wired: a face failing the check is never marked, is flagged in
  the UI, and is written to `audit_log`. `attendance.liveness` records whether a
  check ran and passed.
* **The Claude path has never made an API call.** It is written against the
  documented SDK surface and is inactive by default.
* **No test set has been captured**, so `eval/liveness/evaluate.py` has never
  run against real data.
* `backend/test_liveness.py` passes 26 tests, all with a **stubbed** provider.
  They establish that the plumbing is correct — that an API key alone does not
  enable egress, that ERROR fails closed, that a spoof verdict never produces an
  attendance row, that the cache bounds the call count, that the worker survives
  a provider that raises. **None of them establish that the check can detect a
  printed photograph.** That is not a thing a stub can tell you.

### What the feature currently supports as a claim

> *"A liveness check exists, is off by default, and when switched on refuses to
> mark a face the provider judges to be a presentation attack."*

### What it does NOT support

* **Any detection rate.** Nobody has shown it catches a printed photo, a phone
  replay or a laptop replay, at any rate.
* **Any false-reject rate.** Nobody has shown what it costs a live person
  standing at the camera.
* **Any latency figure** beyond the design bound (`LIVENESS_TIMEOUT_S = 20 s`).
* **That `README.md`'s "reduce proxy attendance" objective is met.** With
  liveness off — the default — it is not: holding up a phone marks someone
  present. That is now stated in `README.md` rather than left implied.

### To replace this section with numbers

1. Read [`eval/liveness/README.md`](eval/liveness/README.md) — it specifies the
   capture protocol and the sample-size reasoning (150 live / 60 printed /
   60 phone / 60 laptop = 330 images).
2. Capture into `eval/liveness/dataset/`. It is gitignored; it holds real
   biometric data and stays on the machine.
3. Configure a provider (README.md § Liveness). A full pass is ~330 API calls:
   ~$4.00 on `claude-opus-5`, ~$0.40 on `claude-haiku-4-5`, free on Ollama.
4. `py eval/liveness/evaluate.py --dry-run`, then `--limit 5`, then the full run.
5. Replace everything under **2. Liveness** with what it prints — detection rate
   per attack type, false-reject rate on live faces, per-decision latency, each
   with its Wilson interval — and keep the harness's "cannot support" list
   attached to it.

Until step 5 is done, this section stays as it is. A liveness result is exactly
the kind of number that is tempting to assert from a plausible-looking prompt,
and asserting it is how a system ends up claiming a protection it does not have.
