"""
Face-recognition attendance engine (InsightFace / Python 3.14 compatible).

Why InsightFace instead of `face_recognition`/dlib?
  dlib has no prebuilt wheel for Python 3.14, so the original stack can't run on
  this machine. InsightFace (ArcFace) installs cleanly on 3.14 via onnxruntime,
  is more accurate, and bundles both detection and recognition. It produces
  512-d L2-normalized embeddings, so identity matching is a simple cosine
  similarity (a dot product).

Web adaptations (unchanged from the previous version):
  * The webcam runs in a background thread; Flask streams annotated JPEG frames
    as MJPEG.
  * Attendance is kept in memory and exposed as JSON.
  * Unknown faces are queued with a cropped thumbnail so the frontend can name
    them, instead of the notebook's blocking `input()`.

Heavy dependencies (cv2, insightface, numpy, pandas) are imported lazily so the
Flask app can boot and serve the UI even before they're installed.
"""

import base64
import os
import pickle
import threading
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# New database file (512-d ArcFace embeddings). Kept separate from the old
# dlib `encodings.pkl`, whose 128-d vectors are not compatible with this model.
DB_PATH = os.path.join(PROJECT_ROOT, "face_db.pkl")

# InsightFace model pack (auto-downloaded once to ~/.insightface on first use).
# buffalo_sc is small (~15 MB) and CPU-friendly.
MODEL_NAME = "buffalo_sc"
DET_SIZE = (640, 640)

# Cosine-similarity thresholds (embeddings are L2-normalized, so similarity is a
# dot product in [-1, 1]; higher = more similar).
STRONG_MATCH = 0.50   # >= this -> confident match  (green)
WEAK_MATCH = 0.32     # >= this -> uncertain match  (orange); below -> unknown (red)

# Two unknown faces are treated as the same person (so we only queue ONE card
# for them) when their embeddings are at least this similar. Embeddings drift a
# little every frame, so this must be a similarity check, not an exact match.
# Kept just above the different-person range (~0.0-0.2) so pose/expression
# changes of the SAME face still merge.
UNKNOWN_DEDUP = 0.35
# Belt-and-suspenders against the queue flooding: never add unknown cards faster
# than this, and never keep more than this many at once.
UNKNOWN_COOLDOWN_S = 1.5
MAX_PENDING = 25


class AttendanceEngine:
    """Owns the camera thread, the known-face database and all live state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

        self._latest_jpeg = None
        self._camera_ok = False
        self._last_error = None

        self._app = None            # lazily-created InsightFace FaceAnalysis
        self._known_embeddings = []  # list of np.ndarray (512,)
        self._known_names = []
        self._db_loaded = False

        self._attendance = {}        # name -> {"Time","Date","Status"}

        # Unknown faces awaiting a name:
        #   key -> {"id","thumb"(base64 jpeg),"embedding"(np array),"similarity"}
        self._pending = {}
        self._pending_seq = 0
        self._last_unknown_add = 0.0

    # ----------------------------------------------------------------- database
    def _load_db(self):
        if self._db_loaded:
            return
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                data = pickle.load(f)
            self._known_embeddings = list(data.get("embeddings", []))
            self._known_names = list(data.get("names", []))
        self._db_loaded = True

    def _save_db(self):
        with open(DB_PATH, "wb") as f:
            pickle.dump(
                {"embeddings": self._known_embeddings, "names": self._known_names}, f
            )

    def _mark(self, name, status):
        if name not in self._attendance:
            now = datetime.now()
            self._attendance[name] = {
                "Time": now.strftime("%H:%M:%S"),
                "Date": now.strftime("%d-%m-%Y"),
                "Status": status,
            }

    # -------------------------------------------------------------- public API
    def status(self):
        self._load_db()  # cheap; lets the UI show the DB size before the camera runs
        with self._lock:
            deps_ok, deps_msg = _deps_available()
            return {
                "running": self._running,
                "camera_ok": self._camera_ok,
                "deps_ok": deps_ok,
                "deps_message": deps_msg,
                "known_count": len(self._known_names),
                "attendance_count": len(self._attendance),
                "pending_count": len(self._pending),
                "last_error": self._last_error,
            }

    def attendance(self):
        with self._lock:
            return [
                {"name": n, "time": i["Time"], "date": i["Date"], "status": i["Status"]}
                for n, i in self._attendance.items()
            ]

    def pending(self):
        with self._lock:
            return [
                {"id": p["id"], "thumb": p["thumb"], "similarity": round(p["similarity"], 3)}
                for p in self._pending.values()
            ]

    def register(self, pending_id, name):
        """Attach a name to a queued unknown face and persist it forever."""
        name = (name or "").strip()
        if not name:
            return False, "Name is required."
        with self._lock:
            key = next(
                (k for k, p in self._pending.items() if p["id"] == pending_id), None
            )
            if key is None:
                return False, "That face is no longer pending."
            entry = self._pending.pop(key)
            self._known_embeddings.append(entry["embedding"])
            self._known_names.append(name)
            self._mark(name, "registered")
            self._save_db()
        return True, f"'{name}' registered and will be recognized automatically."

    def dismiss(self, pending_id):
        with self._lock:
            key = next(
                (k for k, p in self._pending.items() if p["id"] == pending_id), None
            )
            if key is not None:
                self._pending.pop(key)
        return True

    def save_csv(self):
        import pandas as pd

        with self._lock:
            rows = [
                {"Name": n, "Time": i["Time"], "Date": i["Date"], "Status": i["Status"]}
                for n, i in self._attendance.items()
            ]
        filename = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
        path = os.path.join(PROJECT_ROOT, filename)
        pd.DataFrame(rows).to_csv(path, index=False)
        return filename, path

    def frames(self):
        """Generator of raw JPEG bytes for the MJPEG stream."""
        placeholder = _placeholder_jpeg("Camera stopped")
        while True:
            with self._lock:
                frame = self._latest_jpeg
                running = self._running
            yield frame if frame is not None else placeholder
            time.sleep(0.05 if running else 0.2)

    # ------------------------------------------------------------ camera thread
    def start(self):
        with self._lock:
            if self._running:
                return True, "Already running."
        deps_ok, deps_msg = _deps_available()
        if not deps_ok:
            self._last_error = deps_msg
            return False, deps_msg
        self._load_db()
        with self._lock:
            self._running = True
            self._last_error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True, "Camera started."

    def stop(self):
        with self._lock:
            self._running = False
        return True, "Camera stopped."

    def _ensure_model(self):
        """Create the InsightFace analyzer once (downloads the model if needed)."""
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=DET_SIZE)
        self._app = app

    def _loop(self):
        import cv2

        try:
            self._ensure_model()
        except Exception as e:  # model download / load failure
            with self._lock:
                self._running = False
                self._last_error = f"Failed to load face model: {e}"
            return

        cap = self._open_camera(cv2)
        if cap is None:
            with self._lock:
                self._running = False
                self._camera_ok = False
                self._last_error = "Could not open webcam (device 0)."
            return
        with self._lock:
            self._camera_ok = True
            self._last_error = None

        frame_count = 0
        consecutive_fail = 0
        annotations = []  # reused between recognition frames so boxes don't flicker
        try:
            while True:
                with self._lock:
                    if not self._running:
                        break

                ok, frame = cap.read()
                if not ok or frame is None:
                    # Cameras often drop a few frames while warming up; only give
                    # up after many consecutive failures.
                    consecutive_fail += 1
                    if consecutive_fail > 60:
                        with self._lock:
                            self._last_error = "Webcam stopped delivering frames."
                        break
                    time.sleep(0.03)
                    continue
                consecutive_fail = 0

                frame_count += 1
                if frame_count % 3 == 0:  # recognize every 3rd frame (CPU-friendly)
                    try:
                        annotations = self._recognize(frame, cv2)
                    except Exception as e:  # never let recognition kill the stream
                        annotations = []
                        with self._lock:
                            self._last_error = f"Recognition error: {e}"

                for (x1, y1, x2, y2), label, color in annotations:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
        finally:
            cap.release()
            with self._lock:
                self._running = False
                self._camera_ok = False
                self._latest_jpeg = None

    def _open_camera(self, cv2):
        """Open device 0, preferring DirectShow on Windows for a faster start."""
        backends = [getattr(cv2, "CAP_DSHOW", 0), 0]  # DSHOW, then default
        for backend in backends:
            cap = cv2.VideoCapture(0, backend) if backend else cv2.VideoCapture(0)
            if cap.isOpened():
                # Prime the pipeline — the first read or two are often empty.
                for _ in range(5):
                    cap.read()
                    time.sleep(0.02)
                return cap
            cap.release()
        return None

    def _recognize(self, frame, cv2):
        """Detect + identify every face; return a list of (bbox, label, color)."""
        import numpy as np

        faces = self._app.get(frame)
        with self._lock:
            known = list(self._known_embeddings)
            names = list(self._known_names)
        known_matrix = np.array(known) if known else None

        out = []
        for face in faces:
            emb = face.normed_embedding
            bbox = tuple(int(v) for v in face.bbox)

            if known_matrix is not None:
                sims = known_matrix @ emb  # cosine similarity (embeddings normalized)
                idx = int(sims.argmax())
                sim = float(sims[idx])
            else:
                idx, sim = -1, -1.0

            if sim >= STRONG_MATCH:
                name = names[idx]
                with self._lock:
                    self._mark(name, "present")
                out.append((bbox, name, (0, 200, 0)))          # green
            elif sim >= WEAK_MATCH:
                name = names[idx]
                with self._lock:
                    self._mark(name, "uncertain")
                out.append((bbox, f"~{name}", (0, 165, 255)))  # orange
            else:
                self._queue_unknown(emb, frame, bbox, sim, cv2)
                out.append((bbox, "Unknown", (0, 0, 255)))     # red
        return out

    def _queue_unknown(self, embedding, frame, bbox, similarity, cv2):
        import numpy as np

        with self._lock:
            # Already waiting? Compare by similarity — the same person's embedding
            # varies slightly frame to frame, so an exact key would never match.
            for p in self._pending.values():
                if float(np.dot(p["embedding"], embedding)) >= UNKNOWN_DEDUP:
                    return
            # Hard cap + throttle so the queue can never flood, even if a face's
            # embedding is jittery enough to slip past the dedup check.
            now = time.time()
            if len(self._pending) >= MAX_PENDING:
                return
            if now - self._last_unknown_add < UNKNOWN_COOLDOWN_S:
                return
            self._last_unknown_add = now
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            thumb_b64 = ""
            if crop.size:
                ok, buf = cv2.imencode(".jpg", crop)
                if ok:
                    thumb_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            self._pending_seq += 1
            self._pending[self._pending_seq] = {
                "id": self._pending_seq,
                "thumb": thumb_b64,
                "embedding": embedding,
                "similarity": similarity,
            }


# --------------------------------------------------------------------- module utils
def _deps_available():
    import importlib.util

    for mod in ("cv2", "insightface", "onnxruntime", "numpy", "pandas"):
        if importlib.util.find_spec(mod) is None:
            return False, (
                f"Missing Python package '{mod}'. Install backend requirements "
                "(pip install -r backend/requirements.txt) before starting the camera."
            )
    return True, "ok"


_PLACEHOLDER_CACHE = {}


def _placeholder_jpeg(text):
    if text in _PLACEHOLDER_CACHE:
        return _PLACEHOLDER_CACHE[text]
    try:
        from PIL import Image, ImageDraw
        import io

        img = Image.new("RGB", (640, 480), (24, 27, 34))
        ImageDraw.Draw(img).text((250, 230), text, fill=(140, 150, 165))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()
    except Exception:
        data = base64.b64decode(
            "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP//////////////////////////////"
            "////////////////////////////////////////////////////wgALCAABAAEB"
            "AREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        )
    _PLACEHOLDER_CACHE[text] = data
    return data


# A single shared engine instance for the whole process.
engine = AttendanceEngine()
