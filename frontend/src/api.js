// Thin wrapper around the Flask API. All URLs are relative — Vite proxies them
// to http://localhost:5000 in dev (see vite.config.js).
//
// Every request carries the bearer token when one is held. A 401 means the
// token is missing, invalid or expired, so the session is cleared here rather
// than at each call site: expiry can happen on any poll, and every component
// would otherwise need the same handling.

import { clearSession, getToken } from "./auth";

async function json(url, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(url, { ...options, headers });
  const data = await res.json().catch(() => ({}));

  if (res.status === 401) {
    // Expired or rejected. Drop the session; App re-renders to the login screen.
    clearSession();
    throw new Error(data.message || "Your session has ended. Please sign in again.");
  }
  if (!res.ok) throw new Error(data.message || `Request failed: ${res.status}`);
  return data;
}

const post = (url, body) =>
  json(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });

export const api = {
  // Auth. login() deliberately does NOT go through json(): a failed login is a
  // 401 by design, and clearing the session on it would be circular.
  login: async (username, password) => {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.message || `Sign-in failed: ${res.status}`);
    return data;
  },
  me: () => json("/api/auth/me"),

  // Read.
  status: () => json("/api/status"),
  attendance: () => json("/api/attendance"),
  people: () => json("/api/people"),

  // Admin.
  pending: () => json("/api/pending"),
  start: () => post("/api/start"),
  stop: () => post("/api/stop"),
  save: () => post("/api/save"),
  register: (id, name) => post("/api/register", { id, name }),
  dismiss: (id) => post("/api/dismiss", { id }),
  // Faces the liveness check refused. Admin-only, and always empty when
  // anti-spoofing is inactive — read status.liveness.active to tell "no
  // attempts" apart from "not switched on".
  flagged: () => json("/api/flagged"),
  dismissFlagged: (id) => post("/api/flagged/dismiss", { id }),
  deletePerson: (id) => json(`/api/people/${id}`, { method: "DELETE" }),
  audit: () => json("/api/audit"),

  // A single-use, 60-second, feed-only token. The <img> that renders the MJPEG
  // stream cannot carry an Authorization header, so the credential goes in the
  // URL — see backend/app.py::stream_token for why that is acceptable here and
  // why the whole mechanism should disappear when /video_feed does.
  streamToken: () => post("/api/stream_token"),

  // AI layer.
  aiStatus: () => json("/api/ai_status"),
  ask: (question) => post("/api/ask", { question }),
  report: () => json("/api/report"),
};

// Download needs the token too, and window.open() cannot set a header. Fetch it
// as a blob and hand it to a temporary anchor instead.
export async function downloadCsv() {
  const token = getToken();
  const res = await fetch("/api/download", {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (res.status === 401) {
    clearSession();
    throw new Error("Your session has ended. Please sign in again.");
  }
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.message || `Download failed: ${res.status}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download =
    res.headers
      .get("Content-Disposition")
      ?.match(/filename="?([^";]+)"?/)?.[1] || "attendance.csv";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
