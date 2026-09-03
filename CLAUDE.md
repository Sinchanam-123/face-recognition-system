# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Purpose

A **face-recognition attendance system**. A webcam feed is analysed in real time;
recognised people are automatically marked present with a timestamp, and faces
the system doesn't know are surfaced in the UI so an operator can name them in
one click. Attendance can be exported to CSV, and an "Ask AI" panel answers
plain-English questions about the day's records.

The repo grew out of three Jupyter notebooks (dlib / `face_recognition` based).
Those are kept for history but are **not** the running system — the app is the
Flask backend + React frontend described below.

## Architecture

```
Browser (React/Vite :5173)
   │  fetch /api/*        ──proxy──►  Flask (:5000)  backend/app.py
   │  <img src=/video_feed>              │
   │                                     ├─ engine.py  ── camera thread ──► OpenCV VideoCapture(0)
   │                                     │                     │
   │                                     │                     └─ InsightFace buffalo_sc (ONNX, CPU)
   │                                     │                            → 512-d L2-normalised ArcFace embedding
   │                                     │                            → cosine similarity vs. gallery
   │                                     │                            → face_db.pkl (pickled embeddings + names)
   │                                     └─ genai.py   ── Claude / Ollama / computed fallback
```

Key design points:

- **The camera lives server-side.** `backend/engine.py` owns a single daemon
  thread that reads frames, annotates them, and stores the latest JPEG.
  `/video_feed` streams those bytes as MJPEG (`multipart/x-mixed-replace`). The
  browser never touches `getUserMedia` — it just renders an `<img>`.
- **One shared engine instance** (`engine = AttendanceEngine()` at the bottom of
  `engine.py`) holds *all* live state: camera status, the known-face gallery,
  the attendance dict, and the pending-unknown queue. All mutation goes through
  `self._lock`.
- **Recognition is InsightFace `buffalo_sc`**, not dlib. dlib has no prebuilt
  wheel for Python 3.14, so the original stack can't run here. `buffalo_sc` is
  a small (~15 MB) CPU-friendly pack that bundles detection + recognition and is
  auto-downloaded to `~/.insightface/models` on first camera start (needs
  internet once).
- **Matching is a dot product.** `face.normed_embedding` is 512-d and
  L2-normalised, so cosine similarity is `known_matrix @ emb`. Thresholds in
  `engine.py`: `STRONG_MATCH = 0.50` (green, status `present`),
  `WEAK_MATCH = 0.32` (orange, status `uncertain`), below that → red `Unknown`
  and queued for naming.
- **Gallery persistence is a pickle.** `face_db.pkl` at the repo root holds
  `{"embeddings": [...], "names": [...]}`. Registering an unknown face appends
  to it and rewrites the file immediately.
- **Heavy deps are imported lazily** (`cv2`, `insightface`, `numpy`, `pandas`
  are imported inside functions) so Flask boots and serves the dashboard even
  when the recognition stack isn't installed. `_deps_available()` reports this
  to the UI, which shows a warning banner instead of failing.
- **Recognition runs every 3rd frame** (`frame_count % 3`) to keep CPU sane;
  the previous frame's boxes are reused in between so they don't flicker.
- **Attendance is in-memory only** — a `dict` keyed by name, first write wins.
  CSV export is a snapshot, not the source of truth.

## Key files

### Backend (`backend/`)

| File | What it does |
|---|---|
| `app.py` | Flask app + all HTTP routes. Thin — every route delegates to `engine` or `genai`. Runs with `threaded=True`, `use_reloader=False` (the reloader would kill the camera thread mid-session). |
| `engine.py` | The core. `AttendanceEngine` class: camera thread (`_loop`), camera open with DirectShow fallback (`_open_camera`), detection + matching (`_recognize`), unknown-face queueing with dedup/throttle (`_queue_unknown`), gallery load/save, attendance marking, CSV export, MJPEG frame generator. Also holds all tunable thresholds. |
| `genai.py` | Optional LLM layer for `/api/ask` and `/api/report`. Auto-selects a provider: Claude (if `ANTHROPIC_API_KEY` set and `anthropic` installed) → Ollama (if a server answers on `OLLAMA_HOST`) → deterministic computed fallback. Provider choice is cached 15 s. Never raises to the caller path without `app.py` catching it. |
| `encode_faces.py` | Offline CLI: builds `face_db.pkl` from `dataset/<Person>/*.jpg`. Keeps the highest-`det_score` face per photo. **Overwrites** `face_db.pkl` — it does not merge with live registrations. |
| `check_camera.py` | Standalone webcam sanity check with an OpenCV preview window and a mean/std overlay. Use this to tell "camera is broken" apart from "app is broken". |
| `requirements.txt` | Pinned by minimum version only. Comments explain why each dep is there. |

### Frontend (`frontend/`)

| File | What it does |
|---|---|
| `src/App.jsx` | The whole dashboard in one file: KPI row, live video panel, attendance table, unknown-face cards, and the Ask-AI panel, plus small presentational components (`Kpi`, `Icon`, `StatusPill`, `Avatar`, `StatusTag`, `Legend`, `Empty`). Polls `/api/status`, `/api/attendance`, `/api/pending` every 2 s. |
| `src/api.js` | Thin `fetch` wrapper. All URLs relative; throws `Error(data.message)` on non-2xx. |
| `src/main.jsx` | React root. |
| `src/styles.css` | All styling, hand-written (no framework). Dark theme. |
| `vite.config.js` | Dev server on :5173, proxies `/api` and `/video_feed` to `http://localhost:5000`. |

### Data / legacy at the repo root

| Path | Notes |
|---|---|
| `face_db.pkl` | Live gallery — 512-d ArcFace embeddings. **Currently contains only two junk entries: `ghost1`, `ghost2`.** |
| `encodings.pkl` | Dead. Old 128-d dlib encodings from the notebooks; incompatible with ArcFace and read by nothing. |
| `known_faces/` | Dead. 3023 all-black 125×94 JPEGs (62 people). See Known issues. |
| `attendance_2026-04-28.csv` | Sample export from the notebook era (`Name,Time,Date` — the app now also writes `Status`). |
| `*.ipynb` | Legacy dlib notebooks: `download_dataset.ipynb` (fetch LFW), `encode_faces.ipynb` (build `encodings.pkl`), `attendance_system.ipynb` (blocking-`input()` version of the app). Reference only. |
| `README.md` | User-facing overview. Keep in sync when architecture changes. |

## API surface

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| GET | `/api/status` | `running`, `camera_ok`, `deps_ok`, `deps_message`, counts, `last_error` |
| POST | `/api/start` / `/api/stop` | Control the camera thread |
| GET | `/video_feed` | MJPEG stream of annotated frames |
| GET | `/api/attendance` | Marked records |
| GET | `/api/pending` | Unknown faces awaiting a name (base64 JPEG thumb + best similarity) |
| POST | `/api/register` | `{id, name}` → append to gallery, persist, mark `registered` |
| POST | `/api/dismiss` | `{id}` → drop a pending face |
| POST | `/api/save` | Write `attendance_YYYY-MM-DD.csv` to the repo root |
| GET | `/api/download` | Write + return that CSV as an attachment |
| GET | `/api/ai_status` | `{enabled, provider}` |
| POST | `/api/ask` | `{question}` → NL answer over today's records |
| GET | `/api/report` | AI-written daily summary |

## Running it

Python here is **3.14**, invoked as `py` (there is no `python` on PATH — the
`python`/`python3` aliases hit the Microsoft Store shim and fail).

**Backend** — run it from the repo root so `PROJECT_ROOT` resolves and the
pickle/CSV land next to the source:

```bash
py -m pip install -r backend/requirements.txt
```

```bash
py backend/app.py
```

Serves `http://localhost:5000`. The first **Start camera** click downloads the
`buffalo_sc` model pack once. The server intentionally boots without the
recognition packages — only starting the camera requires them.

**Frontend** (needs Node):

```bash
cd frontend && npm install
```

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173`. There is also a preview config at
`.claude/launch.json` (`frontend-dev`, port 5173).

**Rebuild the gallery from photos** (optional, overwrites `face_db.pkl`):

```bash
py backend/encode_faces.py path/to/dataset
```

**Camera troubleshooting:**

```bash
py backend/check_camera.py
```

**Optional LLM** — set `ANTHROPIC_API_KEY` (and `py -m pip install anthropic`),
or run a local Ollama server (`OLLAMA_MODEL`, default `llama3.2`;
`OLLAMA_HOST`, default `http://localhost:11434`). With neither, the AI panel
falls back to deterministic computed answers and still works.

## Conventions

**Python**

- **Type hints on every function and method** — parameters and return type.
  Existing code has *none*; add them as you touch code, don't do a sweeping
  retrofit in an unrelated change.
- **Docstrings** on every module, class, and non-trivial function. Follow the
  existing voice: explain *why*, not just what (see the header of `engine.py`
  or the threshold comments) — the reasoning behind a constant or a workaround
  is the part that's expensive to rediscover.
- **No bare `except:`.** Catch the narrowest exception that can actually occur.
  Broad `except Exception` is acceptable only at a boundary that must not die
  (the camera loop, an LLM call, a Flask route) and must record the error —
  e.g. `self._last_error = f"Recognition error: {e}"`. The codebase is
  currently clean on this; keep it that way.
- Keep heavy imports (`cv2`, `insightface`, `numpy`, `pandas`) **inside**
  functions. Module-level imports of these would break the boots-without-deps
  behaviour the UI depends on.
- All shared-state access goes through `self._lock`. Don't hold the lock across
  a blocking call (model inference, file I/O to a slow disk, HTTP).
- Tunables (thresholds, timeouts, caps) live as module-level UPPER_CASE
  constants with a comment explaining the chosen value — not inline magic
  numbers.
- 4-space indent, ~88-col soft wrap, stdlib → third-party → local import order.
  No formatter or linter is configured; match surrounding style by hand.

**JavaScript / React**

- Function components with hooks. No class components, no state library —
  polling + `useState` is the deliberate pattern here.
- All API calls go through `src/api.js`; never `fetch` directly from a
  component.
- Keep URLs relative so the Vite proxy works.
- No CSS framework. Add styles to `styles.css` following the existing
  BEM-ish flat class naming (`panel-head`, `kpi-value`, `status-tag`).

**Commits**

- **Small, focused commits** — one logical change each. Don't mix a bug fix
  with a refactor, or backend and frontend changes that aren't part of the same
  feature.
- Imperative subject line, ~50 chars, no trailing period
  (`Fix unknown-face queue flooding`, not `fixed some stuff`).
- Body explains *why* when the change isn't self-evident.
- Never commit unless explicitly asked. The generated data files
  (`face_db.pkl`, `attendance_*.csv`) are local state — don't commit changes to
  them incidentally.

## Known issues

State of the repo as of this file's writing. Nothing here has been fixed.

**Data / setup**

- `known_faces/` is **all-black images** — 3023 files, all mean≈0, std≈0. Cause
  is a bug in `download_dataset.ipynb`: sklearn's LFW arrays are floats in
  `0.0–1.0` and were written with `.astype('uint8')` without multiplying by 255.
  The folder is useless for enrolment; `encode_faces.py` on it yields nothing.
- `face_db.pkl` holds only two test entries, `ghost1` and `ghost2`. There is no
  real enrolled gallery — the app effectively starts empty.
- `encodings.pkl` (2.8 MB of 128-d dlib vectors) is dead weight and read by
  nothing.
- `.gitignore` lists `known_faces/`, `encodings.pkl`, and `face_db.pkl`, but all
  three are **already tracked**, so the ignore rules have no effect. Removing
  them needs `git rm --cached`.
- `encode_faces.py` defaults to `./dataset`, which does not exist in this repo.
- It also **overwrites** `face_db.pkl` rather than merging, so running it wipes
  every face registered live through the UI.

**Correctness / behaviour**

- **No date rollover.** `_attendance` is an in-memory dict that never resets. If
  the server runs past midnight, yesterday's entries persist and a person marked
  yesterday will not be re-marked today (`_mark` is first-write-wins).
- **Status never upgrades.** Someone first seen at `uncertain` (0.32–0.50) stays
  `uncertain` for the session even after a confident match, for the same
  first-write-wins reason.
- **All attendance is lost on restart.** Nothing is persisted until someone
  clicks Save/Download.
- **start/stop race.** `stop()` only flips a flag and returns; it doesn't join
  the camera thread. A quick stop→start can spawn a second thread while the
  first still holds `VideoCapture(0)`.
- `start()` writes `self._last_error` outside the lock.
- The 404 branch in `/api/download` is unreachable — `save_csv()` writes the
  file immediately before the `os.path.exists` check. With no attendance it
  returns an empty CSV rather than an error (the UI hides this by disabling the
  button).
- Pending-face `similarity` is `-1.0` when the gallery is empty; the UI shows
  that raw value next to a ring clamped to 0%.
- `check_camera.py` imports `numpy` and never uses it.

**Missing infrastructure**

- **No tests at all** — no test framework, no fixtures, no CI.
- **No linter or formatter** configured for either language.
- Backend modules use flat imports (`import genai`, `from engine import ...`),
  so the app only runs as `py backend/app.py`. `py -m backend.app` or
  `flask --app` will fail with `ModuleNotFoundError`.
- **No authentication, and `host="0.0.0.0"`** — anyone on the LAN can start the
  camera, read attendance, and stream the video feed. `CORS(app)` allows every
  origin. Fine for local dev, not deployable as-is.
- `pickle.load` on `face_db.pkl` is arbitrary-code-execution-by-file; acceptable
  for local use, but it's why the file should never be fetched from elsewhere.
- No structured logging anywhere — errors surface only via `last_error` in
  `/api/status` or `print()`.
- `anthropic` is listed in `requirements.txt` but **not installed** in the
  current environment, so the Claude path is unavailable even if a key is set
  (it degrades silently to Ollama/fallback, which is by design).

**Docs / repo hygiene**

- `README.md` says registered encodings are saved to `encodings.pkl` — wrong,
  it's `face_db.pkl`. Its API table also omits `/api/save` and `/api/dismiss`.
- The repo root is nested one level down from the checkout directory
  (`face_recognition_attendence_system-main/face_recognition_attendence_system-main/`).
  The `.git` directory is in the inner folder.
- **The working tree is dirty**: `README.md`, `backend/app.py`,
  `backend/requirements.txt`, `face_db.pkl`, `frontend/src/App.jsx`,
  `frontend/src/api.js`, and `frontend/src/styles.css` are modified, and
  `backend/genai.py` is untracked. The entire generative-AI layer is
  uncommitted — history is a single `Initial commit`.
- `frontend/dist/` contains a stale build (gitignored, safe to delete).
