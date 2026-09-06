# face_recognition_attendence_system
This project is an automated Face Recognition-based Attendance System that uses computer vision to detect and recognize faces in real time and mark attendance accordingly.

Instead of traditional manual attendance methods, this system captures live video, identifies individuals using facial features, and records attendance with timestamps. This improves accuracy, efficiency, and security in attendance management.

Objective

The main goal of this project is to:

1) Eliminate manual attendance processes
2) Reduce proxy attendance and human errors
3) Automate attendance tracking using AI & Computer Vision
4) Maintain digital records in an organized format.

---

## Architecture

The project has two parts on top of the original notebooks:

```
backend/          Flask API + recognition engine (Python)
  app.py            REST endpoints + MJPEG /video_feed stream
  engine.py         camera thread, face matching, attendance & unknown-face state
  genai.py          generative-AI layer: NL Q&A + report over the records
  models.py         SQLAlchemy schema: person, face_template, attendance,
                    user, audit_log
  db.py             engine/session factory (WAL, foreign keys, busy timeout)
  repo.py           every SQL statement the app runs
  security.py       password hashing, JWTs, role gates, default-deny hook
  settings.py       environment-driven config (JWT, database, CORS, limits)
  cli.py            admin CLI: create-admin, create-user, list-users
  migrate_pickle.py one-off face_db.pkl -> database import
  migrations/       Alembic revisions
  encode_faces.py   bulk-enrol from dataset/<Person>/*.jpg into the database
  requirements.txt
frontend/         React + Vite dashboard
  src/App.jsx       live video, attendance table, register-unknown cards, Ask-AI panel
  src/Login.jsx     sign-in screen
  src/auth.js       token storage and session state
legacy/           the original dlib notebooks, archived (see legacy/README.md)
  attendance_system.ipynb
  encode_faces.ipynb
  download_dataset.ipynb
alembic.ini       migration config (URL comes from DATABASE_URL, not this file)
.env.example      every setting, with the secrets left blank
attendance.db     SQLite: gallery, attendance, users, audit log (gitignored)
face_db.pkl       LEGACY gallery, kept as a backup; no longer read by the app
```

### Persistence and authentication

Attendance and the face gallery live in **SQLite via SQLAlchemy**, with Alembic
owning the schema. Two problems drove that:

* **Attendance used to be an in-memory dict keyed by name.** It was lost on
  every restart, and it had no notion of a day — running past midnight marked
  nobody on day two, silently, because yesterday's names were still in the dict.
  `UNIQUE (person_id, date, session)` makes rollover structural rather than
  something that has to be remembered.
* **The gallery used to be a pickle** holding one embedding per name. It could
  not express "these vectors are the same person", which the planned enrolment
  augmentation needs, and `pickle.load` on a file path is arbitrary code
  execution. `person` and `face_template` split identity from evidence.

The API requires a **JWT** on every endpoint, with two roles — `admin` (enrol,
delete, export, drive the camera) and `viewer` (read attendance). A
`before_request` hook refuses to serve any endpoint that does not declare an
access level, so a route added without a decorator fails closed instead of
being exposed.

**First run:**

```bash
py -m pip install -r backend/requirements.txt
```

```bash
cp .env.example .env
```

Then put a signing key in `.env` — there is no default, and the server refuses
to start without one:

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

`migrate_pickle.py` is idempotent and leaves `face_db.pkl` on disk untouched as
a backup. `create-admin` prompts for a password; no default credentials are
ever seeded.

### Generative-AI layer — "Ask AI" (Claude)

On top of recognition (a *discriminative* task), the dashboard has an **Ask AI**
panel that layers a *generative* capability over the attendance records:

* **Chat with the data** — ask in plain English (*"who was marked uncertain?"*,
  *"how many are present?"*) and get an answer computed from today's records.
* **Daily report** — one click generates a short natural-language summary of
  attendance, uncertain matches, and unknown faces still pending.

`backend/genai.py` picks a provider **automatically**, in this order — no code
changes needed to switch:

1. **Claude** — if `ANTHROPIC_API_KEY` is set (uses `claude-haiku-4-5`, cheap/fast)
2. **Ollama** — if a local [Ollama](https://ollama.com) server is running (free, offline)
3. **Computed fallback** — deterministic answers, always available (nothing to install)

So the panel works out of the box (computed fallback), lights up automatically
when Ollama is running, and uses Claude if you add a key. This mirrors how the
server boots even without the recognition packages.

**Free local AI with Ollama (recommended, no API key/billing):**
```bash
# 1. install Ollama from https://ollama.com , then:
ollama pull llama3.2         # small (~2 GB) model; override with OLLAMA_MODEL
# 2. just start the backend — genai.py auto-detects the running server
```

**Or use Claude (needs an API key + billing credit):**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."        # bash
$env:ANTHROPIC_API_KEY = "sk-ant-..."        # Windows PowerShell
```

Env vars: `OLLAMA_MODEL` (default `llama3.2`), `OLLAMA_HOST` (default
`http://localhost:11434`). The **Ask AI** panel shows which provider is active.

### Recognition engine — InsightFace (Python 3.14 native)

The original notebooks used `face_recognition`/**dlib**, which has **no prebuilt
wheel for Python 3.14**, so it can't run on this machine. The backend instead
uses **InsightFace (ArcFace)** on `onnxruntime`, which installs cleanly on 3.14,
is more accurate, and bundles detection + recognition. It produces 512-d
L2-normalized embeddings, so identity matching is a cosine similarity (dot
product); thresholds are **`0.370`** (confident, green — records attendance) and
`0.32` (uncertain, orange — **displayed only, never recorded**). Both are
measured, not guessed: see `backend/config.py` for the evaluation run behind
each, and `eval/README.md` for how they were derived.

The webcam runs **server-side** in a background thread and streams annotated
frames to the browser as MJPEG. Unknown faces are queued as cards in the UI and
named with a click — replacing the notebook's blocking `input()`.

> Note: the old `known_faces/` images are empty/black files (the original
> notebook encoded from the sklearn LFW cache, not that folder), and 128-d dlib
> vectors aren't compatible with ArcFace — so the gallery starts essentially
> empty and is populated by registering faces live (or via `encode_faces.py` on
> a folder of real photos). Both `migrate_pickle.py` and `encode_faces.py`
> refuse 128-d vectors and undetectable images rather than importing junk.

## Running it

**1. Backend** (Python 3.14 — the stack you have). Run from the **project root**
so `PROJECT_ROOT` resolves and the database lands beside the source:

```bash
py -m pip install -r backend/requirements.txt
```

```bash
py backend/app.py
```

Serves `http://localhost:5000`. See **Persistence and authentication** above for
the one-time `.env`, `alembic upgrade head` and `create-admin` steps — the
server refuses to start until `JWT_SECRET` is set.

First camera start downloads the InsightFace `buffalo_sc` model (~15 MB) once.
The server boots even without the recognition packages installed — the
dashboard loads and shows a warning; only *starting the camera* needs them.

**2. Frontend** (needs Node.js):

```bash
cd frontend && npm install
```

```bash
cd frontend && npm run dev
```

Open **http://localhost:5173** and sign in. An admin can click **Start camera**,
export with **Download CSV**, and name people from the **Unknown faces** panel;
a viewer sees the attendance table and the AI panel only.

**3. Tests:**

```bash
py -m unittest discover -s backend -p "test_*.py"
```

### Bulk enrolment from photos

To enrol a folder of `dataset/<Person>/*.jpg` rather than one webcam frame at a
time. Preview first — it writes to the live database:

```bash
py backend/encode_faces.py path/to/dataset --dry-run
```

A dataset normally holds several photos per person, which gives each of them
several face templates and invalidates `STRONG_MATCH = 0.370` (see the note
below). So the script **refuses by default** and names both ways forward:

```bash
py backend/encode_faces.py path/to/dataset --one-per-person
```

`--one-per-person` enrols only the clearest photo of each person, keeping one
template each so the threshold stays calibrated. `--allow-multi-template`
enrols everything and accepts a stale threshold until it is re-derived. Both
are idempotent, both write audit-log entries, and neither touches
`face_db.pkl`.

### API

Every endpoint except `/` and `/api/auth/login` requires
`Authorization: Bearer <token>`. **Role** is the minimum needed.

| Method | Endpoint | Role | Purpose |
|---|---|---|---|
| GET | `/` | — | health check |
| POST | `/api/auth/login` | — | `{username, password}` → JWT. Rate limited. |
| GET | `/api/auth/me` | any | the token's user and role |
| POST | `/api/auth/users` | admin | create another account |
| GET | `/api/status` | viewer | camera / deps / counts / calibration warning |
| POST | `/api/start`,`/api/stop` | admin | control the camera thread |
| GET | `/video_feed` | *stream token* | MJPEG stream of the annotated webcam |
| POST | `/api/stream_token` | viewer | single-use 60 s token for `/video_feed` |
| GET | `/api/attendance` | viewer | today's records (JSON) |
| GET | `/api/people` | viewer | enrolled people and their template counts |
| DELETE | `/api/people/<id>` | admin | soft-delete; attendance history is kept |
| GET | `/api/pending` | admin | unknown faces awaiting a name |
| POST | `/api/register` | admin | `{id, name}` → enrol. Rate limited, audited. |
| POST | `/api/dismiss` | admin | `{id}` → drop a pending face |
| POST | `/api/save` | admin | write today's CSV |
| GET | `/api/download` | admin | download today's attendance CSV |
| GET | `/api/audit` | admin | recent audit-log entries |
| GET | `/api/ai_status` | viewer | whether the LLM path is configured |
| POST | `/api/ask` | viewer | `{question}` → NL answer over the records |
| GET | `/api/report` | viewer | AI-written daily attendance summary |

`/video_feed` is authenticated by a **single-use, 60-second, feed-only** token in
the query string, because an `<img>` cannot send an `Authorization` header. It
is not usable on any other endpoint. See `backend/app.py::stream_token` — the
mechanism should be deleted along with `/video_feed` if browser-side capture
over a WebSocket replaces it.

### A note on the match threshold

`STRONG_MATCH = 0.370` was measured at **one face template per person** (see
`eval/README.md`). Open-set false-accept rate scales with the *total* number of
templates, not the number of people, so enrolling a second template for anyone
invalidates it — the evaluation measured N=5 needing 0.412. The app does not
change the threshold on its own: it raises a persistent banner in the dashboard,
flags the affected attendance rows (`multi_template_match`, carried into the CSV
export), and leaves re-deriving the number to `eval/evaluate.py`.
