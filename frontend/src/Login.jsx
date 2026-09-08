import { useState } from "react";
import { api } from "./api";
import { setSession } from "./auth";

// The sign-in screen. Shown whenever no valid token is held — on first load, on
// sign-out, and whenever a poll comes back 401 because the token expired.
//
// There is deliberately no "create account" link. Accounts are made by an admin
// (POST /api/auth/users) or, for the very first one, at the machine itself with
// `py backend/cli.py create-admin`. Self-service signup on a system that decides
// who is marked present would be a hole, not a feature.
export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (busy || !username.trim() || !password) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.login(username.trim(), password);
      setPassword("");
      setSession(res.token, res.user);
    } catch (err) {
      setError(err.message);
      setPassword("");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-shell">
      <form className="login-card" onSubmit={submit}>
        <div className="login-brand">
          <span className="dot" />
          <div>
            <h1>Face Attendance</h1>
            <span className="brand-sub">Sign in to continue</span>
          </div>
        </div>

        {error && <div className="login-error">{error}</div>}

        <label className="login-field">
          <span>Username</span>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            autoFocus
            disabled={busy}
          />
        </label>

        <label className="login-field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            disabled={busy}
          />
        </label>

        <button
          className="btn primary login-submit"
          type="submit"
          disabled={busy || !username.trim() || !password}
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>

        <p className="login-hint">
          No account yet? An administrator creates one. For the first admin, run{" "}
          <code>py backend/cli.py create-admin</code> on the server.
        </p>
      </form>
    </div>
  );
}
