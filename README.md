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
  encode_faces.py   build face_db.pkl from a folder of real photos
  requirements.txt
frontend/         React + Vite dashboard
  src/App.jsx       live video, attendance table, register-unknown cards
face_db.pkl         saved 512-d face embeddings (created on first registration)
```

### Recognition engine — InsightFace (Python 3.14 native)

The original notebooks used `face_recognition`/**dlib**, which has **no prebuilt
wheel for Python 3.14**, so it can't run on this machine. The backend instead
uses **InsightFace (ArcFace)** on `onnxruntime`, which installs cleanly on 3.14,
is more accurate, and bundles detection + recognition. It produces 512-d
L2-normalized embeddings, so identity matching is a cosine similarity (dot
product); thresholds are `0.50` (confident, green) / `0.32` (uncertain, orange).

The webcam runs **server-side** in a background thread and streams annotated
frames to the browser as MJPEG. Unknown faces are queued as cards in the UI and
named with a click — replacing the notebook's blocking `input()`.

> Note: the old `known_faces/` images are empty/black files (the original
> notebook encoded from the sklearn LFW cache, not that folder), and 128-d dlib
> vectors aren't compatible with ArcFace — so the app starts with an empty
> `face_db.pkl` and is populated by registering faces live (or via
> `encode_faces.py` on a folder of real photos).

## Running it

**1. Backend** (Python 3.14 — the stack you have):

```bash
cd backend
pip install -r requirements.txt
python app.py            # serves http://localhost:5000
```

First camera start downloads the InsightFace `buffalo_sc` model (~15 MB) once.
The server boots even without the recognition packages installed — the
dashboard loads and shows a warning; only *starting the camera* needs them.

**2. Frontend** (needs Node.js):

```bash
cd frontend
npm install
npm run dev              # serves http://localhost:5173  (proxies /api to :5000)
```

Open **http://localhost:5173**, click **Start camera**, and attendance is marked
live. Use **Download CSV** to export, and the **Unknown faces** panel to register
new people (their encoding is saved to `encodings.pkl` for next time).

### API

| Method | Endpoint          | Purpose                                  |
|--------|-------------------|------------------------------------------|
| GET    | `/api/status`     | camera / deps / counts                   |
| POST   | `/api/start`,`/api/stop` | control the camera thread         |
| GET    | `/video_feed`     | MJPEG stream of the annotated webcam     |
| GET    | `/api/attendance` | marked-attendance records (JSON)         |
| GET    | `/api/pending`    | unknown faces awaiting a name            |
| POST   | `/api/register`   | `{id, name}` → save a face permanently   |
| GET    | `/api/download`   | download today's attendance CSV          |
