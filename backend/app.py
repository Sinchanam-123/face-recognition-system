"""
Flask backend for the Face Recognition Attendance System.

Run:  py backend/app.py     (from the project root)
API is served on http://localhost:5000
"""

import os

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

import genai
from engine import PROJECT_ROOT, engine

app = Flask(__name__)
CORS(app)  # allow the React dev server (localhost:5173) to call us


@app.get("/api/status")
def status():
    return jsonify(engine.status())


@app.post("/api/start")
def start():
    ok, msg = engine.start()
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 400)


@app.post("/api/stop")
def stop():
    ok, msg = engine.stop()
    return jsonify({"ok": ok, "message": msg})


@app.get("/api/attendance")
def attendance():
    return jsonify(engine.attendance())


@app.get("/api/pending")
def pending():
    return jsonify(engine.pending())


@app.post("/api/register")
def register():
    body = request.get_json(silent=True) or {}
    ok, msg = engine.register(body.get("id"), body.get("name"))
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 400)


@app.post("/api/dismiss")
def dismiss():
    body = request.get_json(silent=True) or {}
    engine.dismiss(body.get("id"))
    return jsonify({"ok": True})


@app.post("/api/save")
def save():
    filename, _ = engine.save_csv()
    return jsonify({"ok": True, "filename": filename})


@app.get("/api/download")
def download():
    filename, path = engine.save_csv()
    if not os.path.exists(path):
        return jsonify({"ok": False, "message": "No attendance to download."}), 404
    return send_from_directory(PROJECT_ROOT, filename, as_attachment=True)


@app.post("/api/ask")
def ask():
    """Answer a natural-language question about today's attendance."""
    body = request.get_json(silent=True) or {}
    records = engine.attendance()
    known = engine.status()["known_count"]
    try:
        text, source = genai.answer(body.get("question"), records, known)
    except Exception as e:  # never let an API hiccup 500 the dashboard
        return jsonify({"ok": False, "message": f"AI error: {e}"}), 502
    return jsonify({"ok": True, "answer": text, "source": source})


@app.get("/api/report")
def report():
    """Generate a short natural-language summary of today's attendance."""
    records = engine.attendance()
    st = engine.status()
    try:
        text, source = genai.report(records, st["known_count"], st["pending_count"])
    except Exception as e:
        return jsonify({"ok": False, "message": f"AI error: {e}"}), 502
    return jsonify({"ok": True, "report": text, "source": source})


@app.get("/api/ai_status")
def ai_status():
    """Which AI provider is active: Claude, Ollama, or the computed fallback."""
    return jsonify({"enabled": genai.is_enabled(), "provider": genai.provider_name()})


@app.get("/video_feed")
def video_feed():
    def stream():
        for jpeg in engine.frames():
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )

    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
def health():
    return jsonify({"service": "attendance-backend", "status": "ok"})


if __name__ == "__main__":
    # threaded=True so the MJPEG stream doesn't block other API calls.
    # use_reloader=False: the reloader restarts the process on file changes,
    # which drops the running camera thread and resets state mid-session.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
