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
   │  fetch /api/*  + Bearer JWT ──proxy──►  Flask (:5000)  backend/app.py
   │  <img src=/video_feed?token=…>          │  before_request: default-deny auth
   │                                         ├─ engine.py ── camera thread ──► VideoCapture(0)
   │                                         │                   │
   │                                         │                   └─ InsightFace buffalo_sc (ONNX, CPU)
   │                                         │                          → 512-d L2-normalised ArcFace embedding
   │                                         │                          → cosine similarity vs. gallery matrix
   │                                         │                          → row → person_id → attendance upsert
   │                                         ├─ repo.py ──► SQLite (SQLAlchemy)
   │                                         │      person / face_template / attendance
   │                                         │      user / audit_log
   │                                         ├─ genai.py  ── Claude / Ollama / computed fallback
   │                                         └─ liveness.py ── Claude vision / Ollama vision / OFF
   │                                              (anti-spoofing; no heuristic fallback by design)
```

Key design points:

- **The camera lives server-side.** `backend/engine.py` owns a single daemon
  thread that reads frames, annotates them, and stores the latest JPEG.
  `/video_feed` streams those bytes as MJPEG (`multipart/x-mixed-replace`). The
  browser never touches `getUserMedia` — it just renders an `<img>`.
- **One shared engine instance** (`engine = AttendanceEngine()` at the bottom of
  `engine.py`) holds the live state: camera status, the gallery matrix, the
  per-day attendance write cache, and the pending-unknown queue. All mutation
  goes through `self._lock`, which is never held across a database round trip.
  Constructing one has no side effects — `eval/evaluate.py` builds an idle
  instance purely to read the thresholds off the running app.
- **Recognition is InsightFace `buffalo_sc`**, not dlib. dlib has no prebuilt
  wheel for Python 3.14, so the original stack can't run here. `buffalo_sc` is
  a small (~15 MB) CPU-friendly pack that bundles detection + recognition and is
  auto-downloaded to `~/.insightface/models` on first camera start (needs
  internet once).
- **Matching is a dot product.** `face.normed_embedding` is 512-d and
  L2-normalised, so cosine similarity is `known_matrix @ emb`. Thresholds live
  in `backend/config.py` with the eval run that justifies each:
  `STRONG_MATCH = 0.370` (green, status `present`), `WEAK_MATCH = 0.32`
  (orange, `uncertain` — **displayed only, never recorded**), below that → red
  `Unknown` and queued for naming.
- **A matrix row is a TEMPLATE, not a person.** One person may own several, so
  the argmax is mapped through `_template_persons` before anything is decided.
  `STRONG_MATCH` is only calibrated while each person has exactly one — open-set
  FAR scales with total template count. See the long comment at
  `AttendanceEngine._match()`; the engine raises a calibration warning through
  `/api/status` and flags `attendance.multi_template_match` when it is violated.
- **Persistence is SQLite via SQLAlchemy**, schema owned by Alembic
  (`backend/models.py`, `backend/migrations/`). The gallery matrix is loaded
  once at startup and appended in place on registration — the database is the
  durable store, not something queried per frame. Attendance writes are
  deduplicated by an in-memory per-day cache and throttled by
  `CHECKOUT_REFRESH_S`.
- **Auth is default-deny.** `security.enforce_authentication` runs
  `before_request` and refuses (500) any endpoint whose view function carries no
  `@public` / `@authenticated` / `@require_role` marker. Adding a route without
  one makes it unreachable rather than public.
- **Heavy deps are imported lazily** (`cv2`, `insightface`, `numpy`, `pandas`
  are imported inside functions) so Flask boots and serves the dashboard even
  when the recognition stack isn't installed. `_deps_available()` reports this
  to the UI, which shows a warning banner instead of failing.
- **Recognition runs every 3rd frame** (`frame_count % 3`) to keep CPU sane;
  the previous frame's boxes are reused in between so they don't flicker.
- **Attendance is a database table**, keyed `UNIQUE (person_id, date, session)`.
  The date being part of the key is what gives day rollover for free; the write
  is a single `INSERT ... ON CONFLICT DO UPDATE`, so the camera thread and the
  registration endpoint racing on one person resolve in the database rather than
  duplicating. CSV export is a snapshot, not the source of truth.

## Key files

### Backend (`backend/`)

| File | What it does |
|---|---|
| `app.py` | Flask app + all HTTP routes. Thin — every route delegates to `engine` or `genai`. Runs with `threaded=True`, `use_reloader=False` (the reloader would kill the camera thread mid-session). |
| `engine.py` | The core. `AttendanceEngine` class: camera thread (`_loop`), camera open with DirectShow fallback (`_open_camera`), detection + matching (`_recognize`), unknown-face queueing with dedup/throttle (`_queue_unknown`), gallery load/save, attendance marking, CSV export, MJPEG frame generator. Also holds all tunable thresholds. |
| `liveness.py` | Optional presentation-attack (liveness) check. Same provider pattern as `genai.py` — cached probe, ordered preference — with one deliberate difference: **no fallback**. With no provider the check is inactive and the engine behaves exactly as it did before the feature existed. Claude needs `ANTHROPIC_API_KEY` **and** `LIVENESS_ENABLED` (the key alone must not enable biometric egress); Ollama needs a served *vision* model. Never called from the camera thread. |
| `test_liveness.py` | Provider selection, the two-key rule, JSON parsing, fail-closed, the tracked-face cache, enforcement and the worker — all with a **stubbed** provider. No API calls, ever. |
| `genai.py` | Optional LLM layer for `/api/ask` and `/api/report`. Auto-selects a provider: Claude (if `ANTHROPIC_API_KEY` set and `anthropic` installed) → Ollama (if a server answers on `OLLAMA_HOST`) → deterministic computed fallback. Provider choice is cached 15 s. Never raises to the caller path without `app.py` catching it. |
| `encode_faces.py` | Bulk enrolment CLI: `dataset/<Person>/*.jpg` → `person` + `face_template`, through `repo.enrol_face` (the same call `/api/register` makes). Keeps the highest-`det_score` face per photo and stores that score as `quality_score`. Idempotent. **Refuses by default** to create a multi-template person — `--one-per-person` or `--allow-multi-template`. Also `--dry-run`, `--actor`. |
| `check_camera.py` | Standalone webcam sanity check with an OpenCV preview window and a mean/std overlay. Use this to tell "camera is broken" apart from "app is broken". |
| `models.py` | SQLAlchemy schema: `person`, `face_template` (many per person), `attendance` (unique on person/date/session), `user`, `audit_log`. Every column carries a comment explaining why it exists. |
| `db.py` | Engine + `session_scope()`. Sets WAL, `foreign_keys=ON`, `busy_timeout` per connection. Lazy — importing it opens nothing. |
| `repo.py` | Every SQL statement the app runs. Both attendance writers go through one `upsert_attendance`, so they cannot drift. |
| `security.py` | bcrypt hashing, JWT issue/decode, `@public`/`@authenticated`/`@require_role`, the default-deny `before_request` hook, and single-use stream tokens. |
| `settings.py` | Env-driven config. `require_jwt_secret()` raises at startup — deliberately NOT in `config.py`, which the eval harness and tests import without a key. |
| `cli.py` | `create-admin`, `create-user`, `list-users`, `init-db`. Prompts for passwords; seeds no defaults. |
| `migrate_pickle.py` | One-off `face_db.pkl` import via a restricted unpickler. Idempotent; leaves the pickle on disk. |
| `migrations/` | Alembic. `0001` creates all five tables (create-from-empty — there was no prior SQL schema). `0002` adds `attendance.liveness`, defaulting to `'unavailable'` so existing rows truthfully say no check ran. |
| `test_engine.py` | Matching decisions with synthetic embeddings. No DB, no camera, no cv2. |
| `test_persistence.py` | Day rollover, the unique constraint, check-in/out, multi-template matching, the pickle migration. Real SQLite in a temp file. |
| `test_auth.py` | Unauthenticated rejection, role enforcement, expired tokens, default-deny, rate limits, stream-token scope. Flask test client. |
| `test_encode_faces.py` | Bulk enrolment: dataset scanning, the multi-template refusal, `--one-per-person`, idempotency, audit rows, and that the status banner and row flag follow from it. `embed_dataset` is stubbed, so no cv2 or model needed. |
| `requirements.txt` | Pinned by minimum version only. Comments explain why each dep is there. |

### Frontend (`frontend/`)

| File | What it does |
|---|---|
| `src/App.jsx` | The whole dashboard in one file: KPI row, live video panel, attendance table, unknown-face cards, and the Ask-AI panel, plus small presentational components (`Kpi`, `Icon`, `StatusPill`, `Avatar`, `StatusTag`, `Legend`, `Empty`). Polls `/api/status`, `/api/attendance`, `/api/pending` every 2 s. |
| `src/api.js` | Thin `fetch` wrapper. Attaches the bearer token; a 401 clears the session so `App` swaps to the login screen. `downloadCsv()` fetches a blob because `window.open` cannot send a header. |
| `src/auth.js` | Token/session storage in `localStorage`, with a subscriber list so a 401 anywhere re-renders the app. |
| `src/Login.jsx` | Sign-in screen. No signup link by design — accounts come from an admin or `cli.py`. |
| `src/main.jsx` | React root. |
| `src/styles.css` | All styling, hand-written (no framework). Dark theme. |
| `vite.config.js` | Dev server on :5173, proxies `/api` and `/video_feed` to `http://localhost:5000`. |

### Data / legacy at the repo root

| Path | Notes |
|---|---|
| `attendance.db` | **The live store.** SQLite: gallery, attendance, users, audit log. Gitignored — it holds biometric data and password hashes. |
| `face_db.pkl` | **Legacy, read by nothing.** The old pickled gallery, kept as a backup after `migrate_pickle.py` imported it. Two junk entries: `ghost1`, `ghost2`. |
| `.env` / `.env.example` | Config and secrets. `.env` is gitignored; the example is the tracked template and leaves secrets blank. |
| `alembic.ini` | Migration config. Carries no database URL — `migrations/env.py` reads `DATABASE_URL`. |
| `encodings.pkl` | Dead. Old 128-d dlib encodings from the notebooks; incompatible with ArcFace and read by nothing. |
| `known_faces/` | Dead. 3023 all-black 125×94 JPEGs (62 people). See Known issues. |
| `attendance_2026-04-28.csv` | Sample export from the notebook era (`Name,Time,Date` — the app now also writes `Status`). |
| `*.ipynb` | Legacy dlib notebooks: `download_dataset.ipynb` (fetch LFW), `encode_faces.ipynb` (build `encodings.pkl`), `attendance_system.ipynb` (blocking-`input()` version of the app). Reference only. |
| `README.md` | User-facing overview. Keep in sync when architecture changes. |

## API surface

Every endpoint carries an explicit access level. `security.enforce_authentication`
returns **500** for any endpoint that declares none, so a route added without a
decorator is unreachable rather than public.

| Method | Endpoint | Role | Purpose |
|---|---|---|---|
| GET | `/` | public | Health check. Says nothing about the DB or gallery. |
| POST | `/api/auth/login` | public | `{username, password}` → JWT. Rate limited; every attempt audited. |
| GET | `/api/auth/me` | any | The token's user and role |
| POST | `/api/auth/users` | admin | `{username, password, role}` → create an account |
| GET | `/api/status` | viewer | `running`, `camera_ok`, `deps_ok`, counts, `identity_count`, `template_count`, `calibration_warning`, `liveness` (active/provider/reason/counts), `flagged_count`, `last_error` |
| POST | `/api/start` / `/api/stop` | admin | Control the camera thread |
| POST | `/api/stream_token` | viewer | Single-use, 60 s, feed-only token |
| GET | `/video_feed` | *stream token* | MJPEG stream of annotated frames |
| GET | `/api/attendance` | viewer | Today's records |
| GET | `/api/people` | viewer | Enrolled people + template counts |
| DELETE | `/api/people/<id>` | admin | Soft-delete (`active=0`); history is kept. Audited. |
| GET | `/api/pending` | admin | Unknown faces awaiting a name (base64 JPEG thumb + best similarity). Admin because it is unconsented imagery. |
| POST | `/api/register` | admin | `{id, name}` → enrol. An existing name adds a **template** to that person. Rate limited, audited. |
| POST | `/api/dismiss` | admin | `{id}` → drop a pending face |
| GET | `/api/flagged` | admin | Faces refused by the liveness check. Admin because it is unconsented imagery attached to an accusation. Always empty when liveness is inactive. |
| POST | `/api/flagged/dismiss` | admin | `{id}` → drop a flagged card. The `audit_log` row is untouched. |
| POST | `/api/save` | admin | Write `attendance_YYYY-MM-DD.csv` to the repo root. Audited. |
| GET | `/api/download` | admin | Write + return that CSV as an attachment. Audited. |
| GET | `/api/audit` | admin | Recent audit-log entries |
| GET | `/api/ai_status` | viewer | `{enabled, provider}` |
| POST | `/api/ask` | viewer | `{question}` → NL answer over today's records |
| GET | `/api/report` | viewer | AI-written daily summary |

`/api/attendance` rows keep the old `name` / `time` / `date` / `status` keys —
`genai.py` reads exactly those — and add `person_id`, `session`, `check_in`,
`check_out`, `confidence`, `multi_template_match`, `liveness`.

## Running it

Python here is **3.14**, invoked as `py` (there is no `python` on PATH — the
`python`/`python3` aliases hit the Microsoft Store shim and fail).

**Backend** — run it from the repo root so `PROJECT_ROOT` resolves and the
database/CSV land next to the source:

```bash
py -m pip install -r backend/requirements.txt
```

One-time setup. The server **refuses to start without `JWT_SECRET`** — there is
no fallback value, by design:

```bash
cp .env.example .env
```

```bash
py -c "import secrets; print(secrets.token_urlsafe(48))"
```

```bash
py -m alembic upgrade head
```

```bash
py backend/migrate_pickle.py
```

```bash
py backend/cli.py create-admin
```

Then:

```bash
py backend/app.py
```

`migrate_pickle.py` is idempotent and leaves `face_db.pkl` untouched.
`create-admin` prompts for a password and refuses to seed a default one.

**Tests:**

```bash
py -m unittest discover -s backend -p "test_*.py"
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

**Bulk-enrol from photos** (`dataset/<Person>/*.jpg`). Always preview first — it
writes to the live database and enrolment is not something you want to guess at:

```bash
py backend/encode_faces.py path/to/dataset --dry-run
```

A dataset normally holds several photos per person, which would give each of
them several face templates and invalidate `STRONG_MATCH = 0.370`. The script
**refuses by default** and names both ways forward:

```bash
py backend/encode_faces.py path/to/dataset --one-per-person
```

`--one-per-person` enrols only the highest-`det_score` photo each, keeping N=1
so the deployed threshold stays calibrated. `--allow-multi-template` enrols
everything and accepts that the threshold is stale until re-derived in `eval/`.
Both are idempotent — a re-run skips photos already enrolled rather than
doubling everyone's template count.

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

State of the repo. Items under **Fixed** were resolved by the persistence +
authentication change; everything below that section is still open.

**Fixed — persistence and auth (SQLite + SQLAlchemy + JWT)**

- Date rollover, attendance persistence and status upgrades: attendance is now
  a table keyed `UNIQUE (person_id, date, session)`, written with an upsert.
- The pickle gallery: replaced by `person` + `face_template` (many templates per
  person). `pickle.load` is gone from the running app; `migrate_pickle.py` is
  the only reader left and it uses a restricted unpickler. `face_db.pkl` stays
  on disk as a backup and is no longer read.
- No authentication, `host="0.0.0.0"`, `CORS(app)`: JWT with admin/viewer roles,
  a default-deny `before_request` hook, an origin allowlist from `CORS_ORIGINS`,
  and `HOST` defaulting to `127.0.0.1`. Rate limits on login and register; an
  `audit_log` table for enrolments, deletions and failed logins.
- No tests: 123 across `test_engine.py` (matching), `test_persistence.py`
  (schema, rollover, migration), `test_auth.py` (authn/authz) and
  `test_encode_faces.py` (bulk enrolment).
- `encode_faces.py` writing a pickle nothing reads: ported to write through
  `repo.enrol_face`. Because it uses the same repository layer as the
  interactive path, the `/api/status` calibration banner and the
  `multi_template_match` row flag follow from the data rather than being
  reimplemented — `ConsequencesTests` in `test_encode_faces.py` asserts that.

**Data / setup**

- `known_faces/` is **all-black images** — 3023 files, all mean≈0, std≈0. Cause
  is a bug in `download_dataset.ipynb`: sklearn's LFW arrays are floats in
  `0.0–1.0` and were written with `.astype('uint8')` without multiplying by 255.
  The folder is useless for enrolment; `encode_faces.py` on it yields nothing.
- The migrated gallery holds only the two test entries `ghost1` and `ghost2`.
  There is no real enrolled gallery — the app effectively starts empty.
- `encodings.pkl` (2.8 MB of 128-d dlib vectors) is dead weight and read by
  nothing.
- `.gitignore` lists `known_faces/`, `encodings.pkl`, and `face_db.pkl`, but all
  three are **already tracked**, so the ignore rules have no effect. Removing
  them needs `git rm --cached`.
- `encode_faces.py` defaults to `./dataset`, which does not exist in this repo.

**Correctness / behaviour**

- **`STRONG_MATCH = 0.370` is calibrated for one template per person.** The
  schema now permits many. The engine detects this and warns loudly rather than
  guessing a new threshold — re-derive it in `eval/` before relying on records
  flagged `multi_template_match`.
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

- **No CI.** There are tests (`py -m unittest discover -s backend -p "test_*.py"`)
  but nothing runs them automatically.
- **No linter or formatter** configured for either language.
- Backend modules use flat imports (`import genai`, `from engine import ...`),
  so the app only runs as `py backend/app.py`. `py -m backend.app` or
  `flask --app` will fail with `ModuleNotFoundError`. Alembic works around this
  with `prepend_sys_path = backend` in `alembic.ini`.
- No structured logging anywhere — errors surface only via `last_error` in
  `/api/status`, the `audit_log` table, or `print()`.
- **Tokens cannot be revoked before they expire.** JWTs are stateless and there
  is no denylist, so deactivating a user stops them logging in again but does
  not kill a token already issued (up to `JWT_TTL_SECONDS`, default 8 h).
  Rotating `JWT_SECRET` invalidates every token at once, which is the blunt
  instrument available today.
- The frontend keeps its token in `localStorage`, so the session is only as safe
  as the app's XSS posture. The alternative — an HttpOnly cookie — would need
  CSRF protection on every mutating endpoint.
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
