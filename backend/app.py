"""
Flask backend for the Face Recognition Attendance System.

Run:  py backend/app.py     (from the project root)
API is served on http://localhost:5000
"""

import os

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

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
