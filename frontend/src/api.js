// Thin wrapper around the Flask API. All URLs are relative — Vite proxies them
// to http://localhost:5000 in dev (see vite.config.js).

async function json(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
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
  status: () => json("/api/status"),
  attendance: () => json("/api/attendance"),
  pending: () => json("/api/pending"),
  start: () => post("/api/start"),
  stop: () => post("/api/stop"),
  save: () => post("/api/save"),
  register: (id, name) => post("/api/register", { id, name }),
  dismiss: (id) => post("/api/dismiss", { id }),
};
