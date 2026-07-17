import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";

const STATUS_COLORS = {
  present: "#22c55e",
  registered: "#22c55e",
  uncertain: "#f59e0b",
};

export default function App() {
  const [status, setStatus] = useState(null);
  const [attendance, setAttendance] = useState([]);
  const [pending, setPending] = useState([]);
  const [toast, setToast] = useState(null);
  const [busy, setBusy] = useState(false);
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
          <h1>Face Attendance</h1>
        </div>
        <div className="stat-strip">
          <Stat label="Known faces" value={status?.known_count ?? "—"} />
          <Stat label="Marked today" value={attendance.length} />
          <Stat label="Unknown" value={pending.length} />
          <StatusPill status={status} />
        </div>
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

      <main className="grid">
        <section className="panel video-panel">
          <div className="panel-head">
            <h2>Live camera</h2>
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
          <div className="video-wrap">
            <img
              ref={imgRef}
              src="/video_feed"
              alt="Live feed"
              onError={(e) => {
                e.currentTarget.classList.add("broken");
              }}
            />
            {!running && <div className="video-overlay">Camera is off</div>}
          </div>
          <Legend />
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Attendance</h2>
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
                    <td className="name">{r.name}</td>
                    <td>
                      <span
                        className="tag"
                        style={{ color: STATUS_COLORS[r.status] || "#94a3b8" }}
                      >
                        ● {r.status}
                      </span>
                    </td>
                    <td>{r.time}</td>
                    <td>{r.date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="panel pending-panel">
          <div className="panel-head">
            <h2>Unknown faces</h2>
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
      </main>

      {toast && <div className={`toast ${toast.kind}`}>{toast.msg}</div>}
    </div>
  );
}

function UnknownCard({ person, onRegister, onDismiss }) {
  const [name, setName] = useState("");
  return (
    <div className="card">
      {person.thumb ? (
        <img src={`data:image/jpeg;base64,${person.thumb}`} alt="Unknown face" />
      ) : (
        <div className="thumb-fallback">?</div>
      )}
      <div className="card-body">
        <span className="muted small">best match {person.similarity}</span>
        <input
          value={name}
          placeholder="Enter full name"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && name.trim() && onRegister(person.id, name)}
        />
        <div className="card-actions">
          <button
            className="btn primary small"
            disabled={!name.trim()}
            onClick={() => onRegister(person.id, name)}
          >
            Save
          </button>
          <button className="btn ghost small" onClick={() => onDismiss(person.id)}>
            Ignore
          </button>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
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
      <span><i style={{ background: "#22c55e" }} /> Confident match</span>
      <span><i style={{ background: "#f59e0b" }} /> Uncertain</span>
      <span><i style={{ background: "#ef4444" }} /> Unknown</span>
    </div>
  );
}

function Empty({ text }) {
  return <div className="empty">{text}</div>;
}
