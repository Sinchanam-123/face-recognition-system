import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";

// Map a raw status string to a visual tone class (present / uncertain / default).
function statusTone(status) {
  const s = String(status || "").toLowerCase();
  if (s === "present" || s === "registered") return "present";
  if (s === "uncertain") return "uncertain";
  return "default";
}

function initials(name) {
  const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [attendance, setAttendance] = useState([]);
  const [pending, setPending] = useState([]);
  const [toast, setToast] = useState(null);
  const [busy, setBusy] = useState(false);
  const [ai, setAi] = useState(null);
  const imgRef = useRef(null);

  const flash = (msg, kind = "info") => {
    setToast({ msg, kind });
    setTimeout(() => setToast(null), 3500);
  };

  const refresh = useCallback(async () => {
    try {
      const [s, a, p] = await Promise.all([
        api.status(),
        api.attendance(),
        api.pending(),
      ]);
      setStatus(s);
      setAttendance(a);
      setPending(p);
    } catch (e) {
      // Backend probably not running yet — keep quiet, just mark offline.
      setStatus((prev) => ({ ...(prev || {}), offline: true }));
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    api
      .aiStatus()
      .then((r) => setAi(r))
      .catch(() => setAi({ enabled: false, provider: "fallback" }));
  }, []);

  const running = status?.running;

  const toggleCamera = async () => {
    setBusy(true);
    try {
      const res = running ? await api.stop() : await api.start();
      flash(res.message, res.ok ? "ok" : "err");
      // Bust the <img> cache so the MJPEG stream reconnects.
      if (imgRef.current) {
        imgRef.current.src = `/video_feed?t=${Date.now()}`;
      }
    } catch (e) {
      flash(e.message, "err");
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const register = async (id, name) => {
    try {
      const res = await api.register(id, name);
      flash(res.message, res.ok ? "ok" : "err");
    } catch (e) {
      flash(e.message, "err");
    }
    refresh();
  };

  const dismiss = async (id) => {
    await api.dismiss(id);
    refresh();
  };

  const download = () => {
    window.open("/api/download", "_blank");
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <div className="brand-text">
            <h1>Face Attendance</h1>
            <span className="brand-sub">Real-time recognition</span>
          </div>
        </div>
        <StatusPill status={status} />
      </header>

      {status && !status.deps_ok && !status.offline && (
        <div className="banner warn">
          ⚠️ {status.deps_message} The dashboard works, but the camera can’t start
          until the recognition packages are installed.
        </div>
      )}
      {status?.offline && (
        <div className="banner err">
          ⚠️ Can’t reach the backend on <code>localhost:5000</code>. Start it with{" "}
          <code>py backend/app.py</code>.
        </div>
      )}

      <section className="kpis">
        <Kpi label="Known faces" value={status?.known_count ?? "—"} icon="users" />
        <Kpi label="Marked today" value={attendance.length} icon="check" tone="present" />
        <Kpi
          label="Unknown waiting"
          value={pending.length}
          icon="alert"
          tone={pending.length ? "unknown" : "muted"}
          alert={pending.length > 0}
        />
      </section>

      <main className="grid">
        <section className="panel video-panel">
          <div className="panel-head">
            <h2><span className="head-ic video"><Icon name="camera" /></span>Live camera</h2>
            <div className="actions">
              <button
                className={running ? "btn danger" : "btn primary"}
                onClick={toggleCamera}
                disabled={busy}
              >
                {busy ? "…" : running ? "Stop camera" : "Start camera"}
              </button>
            </div>
          </div>
          <div className={`video-wrap ${running ? "live" : ""}`}>
            <img
              ref={imgRef}
              src="/video_feed"
              alt="Live feed"
              onError={(e) => {
                e.currentTarget.classList.add("broken");
              }}
            />
            {running && (
              <>
                <div className="cam-badge">
                  <span className="rec-dot" /> REC · CAM 01
                </div>
                <div className="scan-overlay" />
              </>
            )}
            {!running && (
              <div className="video-overlay">
                <div className="cam-icon" aria-hidden="true">
                  <Icon name="camera" />
                </div>
                <span className="video-overlay-text">Camera is off</span>
                <button className="btn primary" onClick={toggleCamera} disabled={busy}>
                  {busy ? "…" : "Start camera"}
                </button>
              </div>
            )}
          </div>
          <Legend />
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>
              <span className="head-ic present"><Icon name="check" /></span>
              Attendance
              {attendance.length > 0 && <span className="count-badge">{attendance.length}</span>}
            </h2>
            <div className="actions">
              <button className="btn ghost" onClick={download} disabled={!attendance.length}>
                Download CSV
              </button>
            </div>
          </div>
          {attendance.length === 0 ? (
            <Empty text="No one marked yet. Start the camera to begin." />
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Time</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {attendance.map((r) => (
                  <tr key={r.name}>
                    <td className="name">
                      <span className="cell-person">
                        <Avatar name={r.name} status={r.status} />
                        {r.name}
                      </span>
                    </td>
                    <td><StatusTag status={r.status} /></td>
                    <td className="mono-cell">{r.time}</td>
                    <td className="mono-cell">{r.date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="panel pending-panel">
          <div className="panel-head">
            <h2><span className="head-ic unknown"><Icon name="alert" /></span>Unknown faces</h2>
            <span className="muted">{pending.length} waiting</span>
          </div>
          {pending.length === 0 ? (
            <Empty text="No unknown faces. New people appear here to be named." />
          ) : (
            <div className="cards">
              {pending.map((p) => (
                <UnknownCard
                  key={p.id}
                  person={p}
                  onRegister={register}
                  onDismiss={dismiss}
                />
              ))}
            </div>
          )}
        </section>

        <AiPanel ai={ai} hasData={attendance.length > 0} onError={(m) => flash(m, "err")} />
      </main>

      {toast && <div className={`toast ${toast.kind}`}>{toast.msg}</div>}
    </div>
  );
}

const SUGGESTIONS = [
  "Who was marked uncertain?",
  "How many people are present?",
  "List everyone marked so far",
];

function AiPanel({ ai, hasData, onError }) {
  const enabled = ai?.enabled;
  const provider = ai?.provider;
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [thinking, setThinking] = useState(false);

  const ask = async (q) => {
    const query = (q ?? question).trim();
    if (!query || thinking) return;
    setThinking(true);
    setAnswer(null);
    try {
      const res = await api.ask(query);
      setAnswer({ text: res.answer, source: res.source, q: query });
    } catch (e) {
      onError(e.message);
    } finally {
      setThinking(false);
    }
  };

  const makeReport = async () => {
    if (thinking) return;
    setThinking(true);
    setAnswer(null);
    try {
      const res = await api.report();
      setAnswer({ text: res.report, source: res.source, q: "Daily summary" });
    } catch (e) {
      onError(e.message);
    } finally {
      setThinking(false);
    }
  };

  return (
    <section className="panel ai-panel">
      <div className="panel-head">
        <h2>✨ Ask AI</h2>
        <span className="muted small">{ai ? provider : "…"}</span>
      </div>

      {enabled === false && (
        <div className="ai-hint">
          No LLM active — answers are computed, not AI-written. Install{" "}
          <a href="https://ollama.com" target="_blank" rel="noreferrer">Ollama</a> (free,
          local) or set <code>ANTHROPIC_API_KEY</code> on the backend for natural-language
          answers.
        </div>
      )}

      <div className="ai-actions">
        <input
          value={question}
          placeholder="Ask about today's attendance…"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          disabled={thinking}
        />
        <button className="btn primary small" onClick={() => ask()} disabled={thinking || !question.trim()}>
          Ask
        </button>
        <button className="btn ghost small" onClick={makeReport} disabled={thinking || !hasData}>
          Daily report
        </button>
      </div>

      <div className="ai-suggestions">
        {SUGGESTIONS.map((s) => (
          <button key={s} className="chip" onClick={() => ask(s)} disabled={thinking}>
            {s}
          </button>
        ))}
      </div>

      {thinking && <div className="empty">Thinking…</div>}
      {answer && !thinking && (
        <div className="ai-answer">
          <div className="ai-q">{answer.q}</div>
          <p>{answer.text}</p>
        </div>
      )}
    </section>
  );
}

function UnknownCard({ person, onRegister, onDismiss }) {
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);

  // Similarity may arrive as 0–1 or 0–100; normalize to a 0–100 arc for the ring.
  const simNum = Number(person.similarity);
  const simPct = Number.isFinite(simNum)
    ? Math.max(0, Math.min(100, Math.round((simNum <= 1 ? simNum * 100 : simNum))))
    : null;

  const doSave = () => {
    if (!name.trim() || saving) return;
    setSaving(true);
    // Show the success check briefly before the list refresh removes this card.
    setTimeout(() => onRegister(person.id, name), 650);
  };

  return (
    <div className={`card ${saving ? "saved" : ""}`}>
      <div className="thumb-wrap">
        {person.thumb ? (
          <img src={`data:image/jpeg;base64,${person.thumb}`} alt="Unknown face" />
        ) : (
          <div className="thumb-fallback">?</div>
        )}
        {person.similarity != null && (
          <span
            className="sim-ring"
            title="Best match similarity"
            style={
              simPct != null
                ? { background: `conic-gradient(var(--accent) ${simPct * 3.6}deg, rgba(255,255,255,0.10) 0deg)` }
                : undefined
            }
          >
            <span className="sim-val">{person.similarity}</span>
          </span>
        )}
        {saving && (
          <div className="save-check" aria-hidden="true">
            <Icon name="check" />
          </div>
        )}
      </div>
      <div className="card-body">
        <input
          value={name}
          placeholder="Enter full name"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && name.trim() && doSave()}
          disabled={saving}
        />
        <div className="card-actions">
          <button
            className="btn primary small"
            disabled={!name.trim() || saving}
            onClick={doSave}
          >
            {saving ? "Saved ✓" : "Save"}
          </button>
          <button className="btn ghost small" onClick={() => onDismiss(person.id)} disabled={saving}>
            Ignore
          </button>
        </div>
      </div>
    </div>
  );
}

function Kpi({ label, value, icon, tone = "muted", alert = false }) {
  return (
    <div className={`kpi ${tone} ${alert ? "kpi-alert" : ""}`}>
      <span className="kpi-ic"><Icon name={icon} /></span>
      <div className="kpi-body">
        <span className="kpi-value">{value}</span>
        <span className="kpi-label">{label}</span>
      </div>
    </div>
  );
}

const ICONS = {
  users: (
    <>
      <circle cx="9" cy="8" r="3.2" />
      <path d="M3.5 19c0-3 2.5-4.6 5.5-4.6s5.5 1.6 5.5 4.6" />
      <path d="M16 5.2a3.2 3.2 0 0 1 0 6M17 14.6c2.4.4 4 1.9 4 4.4" />
    </>
  ),
  check: (
    <>
      <path d="M20 7L10 17l-5-5" />
    </>
  ),
  alert: (
    <>
      <path d="M12 4l9 15H3z" />
      <path d="M12 10v4" />
      <circle cx="12" cy="17" r="0.6" fill="currentColor" stroke="none" />
    </>
  ),
  camera: (
    <>
      <path d="M4 8h3l1.5-2h7L17 8h3v11H4z" />
      <circle cx="12" cy="13" r="3.4" />
    </>
  ),
};

function Icon({ name }) {
  return (
    <svg viewBox="0 0 24 24" width="1em" height="1em" fill="none"
      stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      {ICONS[name] || null}
    </svg>
  );
}

function StatusPill({ status }) {
  let text = "Offline";
  let cls = "off";
  if (status && !status.offline) {
    if (status.running) {
      text = "Running";
      cls = "on";
    } else {
      text = "Idle";
      cls = "idle";
    }
  }
  return <span className={`pill ${cls}`}>{text}</span>;
}

function Legend() {
  return (
    <div className="legend">
      <span><i style={{ background: "#2fd977" }} /> Confident match</span>
      <span><i style={{ background: "#fbbf24" }} /> Uncertain</span>
      <span><i style={{ background: "#f75d5d" }} /> Unknown</span>
    </div>
  );
}

function Avatar({ name, status }) {
  return <span className={`avatar tone-${statusTone(status)}`}>{initials(name)}</span>;
}

function StatusTag({ status }) {
  const tone = statusTone(status);
  return (
    <span className={`status-tag tone-${tone}`}>
      <span className="status-dot" />
      {status || "—"}
    </span>
  );
}

function Empty({ text }) {
  return <div className="empty">{text}</div>;
}
