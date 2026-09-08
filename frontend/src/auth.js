// Token storage and session state.
//
// The token lives in localStorage so a page reload does not force a re-login.
// That is a deliberate trade: localStorage is readable by any script running on
// this origin, so it is only as safe as the app's XSS posture. The alternative,
// an HttpOnly cookie, would defend against XSS but requires CSRF protection on
// every mutating endpoint, because a cookie is attached automatically and a
// Bearer header is not. For a single-origin operator dashboard the header is
// the simpler and more auditable of the two.

const TOKEN_KEY = "attendance.token";
const USER_KEY = "attendance.user";

// Subscribers re-render when the session changes (login, logout, a 401).
const listeners = new Set();

function notify() {
  for (const fn of listeners) fn(getSession());
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    // Private browsing, or storage disabled. The session then lasts one page
    // view, which is degraded but not broken.
    return null;
  }
}

export function getSession() {
  const token = getToken();
  if (!token) return null;
  try {
    return { token, user: JSON.parse(localStorage.getItem(USER_KEY) || "null") };
  } catch {
    return { token, user: null };
  }
}

export function setSession(token, user) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  } catch {
    /* storage unavailable; the in-memory listeners below still fire */
  }
  notify();
}

export function clearSession() {
  try {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  } catch {
    /* nothing to clear */
  }
  notify();
}

export function isAdmin(user) {
  return user?.role === "admin";
}
