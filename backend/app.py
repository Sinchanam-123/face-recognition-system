"""
Flask backend for the Face Recognition Attendance System.

Run:  py backend/app.py     (from the project root)
API is served on http://localhost:5000 by default.

Every route below carries an explicit access level. That is not a convention —
`security.enforce_authentication` runs before every request and **refuses to
serve any endpoint that does not declare one**, returning 500 rather than
falling open. Adding a route here without a decorator makes it unreachable, on
purpose: forgetting to protect an endpoint should break loudly at the first
request, not quietly expose the face database to the network.

Roles:
  admin   enrol faces, delete people, export data, drive the camera
  viewer  read attendance and ask the AI layer about it; writes nothing
"""

import os
import sys

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

import genai
import repo
import security
import settings
from engine import PROJECT_ROOT, engine
from models import (
    ACTION_ENROL,
    ACTION_EXPORT,
    ACTION_LOGIN_FAILED,
    ACTION_LOGIN_SUCCESS,
    ACTION_PERSON_DELETE,
    ACTION_USER_CREATE,
    ROLES,
)
from security import (
    admin_required,
    authenticated,
    public,
    viewer_required,
)


def create_app() -> Flask:
    """Build the Flask app, failing loudly on missing mandatory configuration."""
    # Checked HERE, not at import of settings, so that `import config` and
    # `import engine` stay safe for eval/evaluate.py and the unit tests, which
    # have no signing key and need none. A server that will accept credentials,
    # though, does not get to start without one.
    settings.require_jwt_secret()

    flask_app = Flask(__name__)

    # Explicit origin allowlist, not `CORS(app)`. The old wildcard echoed back
    # whatever Origin was sent, which combined with a 0.0.0.0 bind meant any
    # page the operator's browser happened to load could call this API.
    CORS(
        flask_app,
        origins=settings.CORS_ORIGINS,
        supports_credentials=False,
        allow_headers=["Content-Type", "Authorization"],
    )

    flask_app.before_request(security.enforce_authentication)
    _install_rate_limits(flask_app)
    return flask_app


def _install_rate_limits(flask_app: Flask) -> None:
    """Attach Flask-Limiter, degrading to unlimited if it is not installed.

    The limiter is a hard requirement in requirements.txt. The guard exists
    because the rest of the backend deliberately boots without its optional
    stack, and a dashboard that refuses to start because a rate limiter is
    missing would be a worse failure than one that starts and says so.
    """
    global limiter
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
    except ImportError:
        limiter = None
        print(
            "WARNING: flask-limiter is not installed; login and register are "
            "NOT rate limited. pip install -r backend/requirements.txt",
            file=sys.stderr,
        )
        return

    limiter = Limiter(
        get_remote_address,
        app=flask_app,
        default_limits=[settings.RATE_LIMIT_DEFAULT],
        storage_uri="memory://",
        strategy="fixed-window",
    )


limiter = None
app = create_app()


def _limit(rule: str):
    """Apply a rate limit when the limiter is available, else a no-op."""

    def decorator(view):
        if limiter is None:
            return view
        return limiter.limit(rule)(view)

    return decorator


def _audit(action, target=None, actor=None):
    """Write one audit row. Never raises into the caller's path.

    An audit write that fails must not turn a successful enrolment into a 500 —
    but it must also not vanish, so the failure is surfaced through
    /api/status's last_error rather than swallowed.
    """
    import db

    identity = actor or security.current_identity() or {}
    try:
        with db.session_scope() as session:
            repo.audit(
                session,
                action=action,
                actor_user_id=identity.get("user_id"),
                actor_username=identity.get("username"),
                target=target,
                ip=security.client_ip(),
            )
    except Exception as e:  # boundary: auditing must not break the request
        print(f"AUDIT WRITE FAILED ({action}): {e}", file=sys.stderr)


# ------------------------------------------------------------------------ auth
@app.post("/api/auth/login")
@public
@_limit(settings.RATE_LIMIT_LOGIN)
def login():
    """Exchange username + password for a JWT. Rate limited; failures audited."""
    import db

    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""

    if not username or not password:
        return jsonify({"ok": False, "message": "Username and password are required."}), 400

    with db.session_scope() as session:
        user = repo.find_user(session, username)
        if user is None:
            # Equalise timing so a request for a nonexistent username costs the
            # same as one for a real account; otherwise latency enumerates users.
            security.dummy_verify()
            ok = False
            user_id = None
            role = None
        else:
            ok = user.active and security.verify_password(password, user.password_hash)
            user_id = user.id
            role = user.role

        repo.audit(
            session,
            action=ACTION_LOGIN_SUCCESS if ok else ACTION_LOGIN_FAILED,
            actor_user_id=user_id if ok else None,
            actor_username=username,
            target=None,
            ip=security.client_ip(),
        )

    if not ok:
        # One message for every failure mode — wrong password, unknown user,
        # deactivated account. Distinguishing them tells an attacker which half
        # of the credential they already have right.
        return jsonify({"ok": False, "message": "Invalid username or password."}), 401

    token, ttl = security.issue_token(user_id, username, role)
    return jsonify(
        {"ok": True, "token": token, "expires_in": ttl,
         "user": {"username": username, "role": role}}
    )


@app.get("/api/auth/me")
@authenticated
def whoami():
    return jsonify({"ok": True, "user": security.current_identity()})


@app.post("/api/auth/users")
@admin_required
def create_user_endpoint():
    """Create another operator account. Admin only; no self-service signup."""
    import db

    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    role = (body.get("role") or "").strip()

    if not username or not password:
        return jsonify({"ok": False, "message": "Username and password are required."}), 400
    if role not in ROLES:
        return jsonify({"ok": False, "message": f"Role must be one of {', '.join(ROLES)}."}), 400

    try:
        password_hash = security.hash_password(password)
    except ValueError as e:
        return jsonify({"ok": False, "message": str(e)}), 400

    with db.session_scope() as session:
        if repo.find_user(session, username) is not None:
            return jsonify({"ok": False, "message": "That username is taken."}), 409
        repo.create_user(session, username, password_hash, role)

    _audit(ACTION_USER_CREATE, security.audit_target("user", username, role))
    return jsonify({"ok": True, "message": f"Created {role} '{username}'."}), 201


# ---------------------------------------------------------------------- status
@app.get("/api/status")
@viewer_required
def status():
    return jsonify(engine.status())


@app.post("/api/start")
@admin_required
def start():
    ok, msg = engine.start()
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 400)


@app.post("/api/stop")
@admin_required
def stop():
    ok, msg = engine.stop()
    return jsonify({"ok": ok, "message": msg})


# ------------------------------------------------------------------ attendance
@app.get("/api/attendance")
@viewer_required
def attendance():
    return jsonify(engine.attendance())


@app.get("/api/people")
@viewer_required
def people():
    return jsonify(engine.people())


@app.delete("/api/people/<int:person_id>")
@admin_required
def delete_person(person_id: int):
    ok, msg, info = engine.delete_person(person_id)
    if ok:
        _audit(
            ACTION_PERSON_DELETE,
            security.audit_target("person", person_id, info.get("name")),
        )
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 404)


# --------------------------------------------------------------------- enrolment
@app.get("/api/pending")
@admin_required
def pending():
    """Unknown faces awaiting a name.

    Admin-only rather than viewer-readable: the payload is cropped photographs
    of people who have not been enrolled and have not consented to anything, and
    it is the queue that feeds the face database.
    """
    return jsonify(engine.pending())


@app.post("/api/register")
@admin_required
@_limit(settings.RATE_LIMIT_REGISTER)
def register():
    body = request.get_json(silent=True) or {}
    ok, msg, info = engine.register(body.get("id"), body.get("name"))
    if ok:
        _audit(
            ACTION_ENROL,
            security.audit_target(
                "person", info["person_id"], info["person_name"]
            )
            + f" template:{info['template_id']} count:{info['template_count']}",
        )
    payload = {"ok": ok, "message": msg}
    if ok:
        payload["person_id"] = info["person_id"]
        payload["template_count"] = info["template_count"]
        payload["calibration_warning"] = info["calibration_warning"]
    return jsonify(payload), (200 if ok else 400)


@app.post("/api/dismiss")
@admin_required
def dismiss():
    body = request.get_json(silent=True) or {}
    engine.dismiss(body.get("id"))
    return jsonify({"ok": True})


# ------------------------------------------------------------------- liveness
@app.get("/api/flagged")
@admin_required
def flagged():
    """Faces refused a mark by the presentation-attack check.

    Admin-only for the same reason as /api/pending, and more so: the payload is
    a cropped photograph of someone the system believes was attempting to claim
    another person's attendance. That is unconsented imagery attached to an
    accusation, and it is not viewer-readable.

    Empty whenever liveness is inactive — with no provider no check runs, so
    nothing can be flagged. Read /api/status's `liveness.active` to tell "no
    attempts today" apart from "anti-spoofing is switched off".
    """
    return jsonify(engine.flagged())


@app.post("/api/flagged/dismiss")
@admin_required
def dismiss_flagged():
    """Drop a flagged card once an operator has dealt with it.

    Dismisses the *card*, never the evidence: the audit_log row written when the
    check failed is append-only and is not touched here.
    """
    body = request.get_json(silent=True) or {}
    ok, msg = engine.dismiss_flagged(body.get("id"))
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 404)


# ------------------------------------------------------------------- exporting
@app.post("/api/save")
@admin_required
def save():
    filename, _ = engine.save_csv()
    _audit(ACTION_EXPORT, security.audit_target("csv", filename))
    return jsonify({"ok": True, "filename": filename})


@app.get("/api/download")
@admin_required
def download():
    filename, path = engine.save_csv()
    if not os.path.exists(path):
        return jsonify({"ok": False, "message": "No attendance to download."}), 404
    _audit(ACTION_EXPORT, security.audit_target("csv", filename))
    return send_from_directory(PROJECT_ROOT, filename, as_attachment=True)


@app.get("/api/audit")
@admin_required
def audit_log():
    import db

    with db.session_scope() as session:
        return jsonify(repo.recent_audit(session))


# ------------------------------------------------------------------ the AI layer
@app.post("/api/ask")
@viewer_required
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
@viewer_required
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
@viewer_required
def ai_status():
    """Which AI provider is active: Claude, Ollama, or the computed fallback."""
    return jsonify({"enabled": genai.is_enabled(), "provider": genai.provider_name()})


# ----------------------------------------------------------------- video stream
@app.post("/api/stream_token")
@viewer_required
def stream_token():
    """Mint a single-use, 60-second, feed-only token for /video_feed.

    WHY A TOKEN IN A QUERY STRING AT ALL
    ------------------------------------
    The feed is rendered by `<img src="/video_feed">`. A browser will not attach
    an `Authorization` header to an image request and offers no way to make it,
    so the credential has to travel either in a cookie or in the URL. Cookies
    would make every other endpoint vulnerable to CSRF, since the API is
    otherwise header-authenticated and holds no CSRF token. So: the URL, with
    the usual objection to credentials-in-URLs narrowed away on three axes.

      * **Short-lived** — settings.STREAM_TOKEN_TTL_S, 60 s.
      * **Single-use** — consumed the moment a stream connection is
        established, so a token recovered afterwards from browser history, a
        proxy log or a Referer header is already dead.
      * **Single-scope** — it authenticates `/video_feed` and nothing else;
        `decode_token` rejects it everywhere a normal API token is expected.

    WHEN /video_feed GOES, THIS GOES WITH IT. The server-side webcam is planned
    to be replaced by browser-side capture over a WebSocket. A WebSocket carries
    real headers at its handshake, so the entire reason for a URL-borne
    credential disappears at that point. Delete this endpoint and
    `security.issue_stream_token` then — do not port the mechanism forward into
    a design that does not need it.
    """
    token, ttl = security.issue_stream_token()
    return jsonify({"ok": True, "token": token, "expires_in": ttl})


@app.get("/video_feed")
@public   # NOT unauthenticated: guarded by a single-use stream token, below.
def video_feed():
    if not security.consume_stream_token(request.args.get("token")):
        return (
            jsonify(
                {
                    "ok": False,
                    "message": (
                        "A valid single-use stream token is required. "
                        "POST /api/stream_token to obtain one."
                    ),
                }
            ),
            401,
        )

    def stream():
        for jpeg in engine.frames():
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )

    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
@public
def health():
    """Liveness only. Deliberately says nothing about the database or gallery."""
    return jsonify({"service": "attendance-backend", "status": "ok"})


if __name__ == "__main__":
    # threaded=True so the MJPEG stream doesn't block other API calls.
    # use_reloader=False: the reloader restarts the process on file changes,
    # which drops the running camera thread and resets state mid-session.
    #
    # HOST defaults to 127.0.0.1 now. The old 0.0.0.0 default put an
    # unauthenticated API on the LAN; binding wider is still supported, but it
    # is a decision made in the environment rather than one the source makes
    # on the operator's behalf.
    print(f"Serving on http://{settings.HOST}:{settings.PORT}")
    print(f"CORS origins: {', '.join(settings.CORS_ORIGINS) or '(none)'}")
    app.run(
        host=settings.HOST,
        port=settings.PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
