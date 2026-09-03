"""
Generative-AI layer for the attendance system.

Two capabilities, both over the live attendance records:
  * answer(question)  -> natural-language Q&A ("who was late today?")
  * report()          -> a short daily summary paragraph

It picks a provider automatically, in this order:
  1. Anthropic Claude  — if ANTHROPIC_API_KEY is set and `anthropic` installed
  2. Ollama (local)    — if an Ollama server is reachable (free, offline)
  3. Computed fallback — deterministic answers, always available

So the feature works out of the box with nothing installed (computed fallback),
lights up automatically when you install Ollama, and uses Claude if you add a
key — no code changes needed to switch. This mirrors how the rest of the app
boots even without the recognition packages.

Enable a real LLM (either is optional):
  * Claude:  set ANTHROPIC_API_KEY   (PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-...")
  * Ollama:  install https://ollama.com , then `ollama pull llama3.2`
             (override with OLLAMA_MODEL / OLLAMA_HOST env vars)
"""

import json
import os
import time
import urllib.request
from datetime import datetime

# --- Claude (option 1) ---
CLAUDE_MODEL = "claude-haiku-4-5"

# --- Ollama (option 2) ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# A small ceiling — these answers are short, and it keeps latency/cost down.
MAX_TOKENS = 700

_SYSTEM = (
    "You are the assistant for a face-recognition attendance system. You are "
    "given today's attendance records as JSON and a question from a teacher or "
    "admin. Answer ONLY from the data provided. Be concise and direct — a "
    "sentence or two, or a short list. If the data doesn't contain the answer, "
    "say so plainly. Never invent names, times, or numbers.\n\n"
    "Record fields: name, status (present | uncertain | registered), time "
    "(HH:MM:SS, when they were first seen), date (DD-MM-YYYY)."
)


# --------------------------------------------------------------- provider pick
# Probing Ollama hits the network, so cache the resolved provider briefly to
# avoid re-probing on every request during a burst of questions.
_PROVIDER_CACHE = {"value": None, "at": 0.0}
_PROVIDER_TTL_S = 15.0


def _claude_available():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _ollama_available():
    """True if an Ollama server answers on OLLAMA_HOST (quick probe)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=0.6) as r:
            return r.status == 200
    except Exception:
        return False


def provider():
    """Return the active provider: 'claude', 'ollama', or None (fallback)."""
    now = time.time()
    if now - _PROVIDER_CACHE["at"] < _PROVIDER_TTL_S:
        return _PROVIDER_CACHE["value"]
    if _claude_available():
        value = "claude"
    elif _ollama_available():
        value = "ollama"
    else:
        value = None
    _PROVIDER_CACHE.update(value=value, at=now)
    return value


def is_enabled():
    """True when a real LLM (Claude or Ollama) is available."""
    return provider() is not None


def provider_name():
    """Human-readable label for the UI."""
    return {"claude": "Claude", "ollama": f"Ollama ({OLLAMA_MODEL})"}.get(
        provider(), "fallback"
    )


# --------------------------------------------------------------- LLM back-ends
def _ask_claude(system, user):
    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _ask_ollama(system, user):
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"num_predict": MAX_TOKENS},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return (data.get("message", {}).get("content") or "").strip()


def _run(system, user):
    """Route to the active provider. Returns (text, source)."""
    p = provider()
    if p == "claude":
        return _ask_claude(system, user), "claude"
    if p == "ollama":
        return _ask_ollama(system, user), "ollama"
    return None, "fallback"


def _context(records, known_count):
    return {
        "generated_at": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "known_faces_in_db": known_count,
        "marked_count": len(records),
        "records": records,
    }


# ------------------------------------------------------------------ public API
def answer(question, records, known_count):
    """Answer a natural-language question about the attendance records."""
    question = (question or "").strip()
    if not question:
        return "Ask a question about today's attendance.", "empty"

    ctx = _context(records, known_count)
    prompt = (
        f"Attendance data (JSON):\n{json.dumps(ctx, indent=2)}\n\n"
        f"Question: {question}"
    )
    text, source = _run(_SYSTEM, prompt)
    if text is None:
        return _fallback_answer(question, records), "fallback"
    return text, source


def report(records, known_count, pending_count):
    """Generate a short natural-language summary of today's attendance."""
    ctx = _context(records, known_count)
    ctx["unknown_faces_pending"] = pending_count
    prompt = (
        f"Attendance data (JSON):\n{json.dumps(ctx, indent=2)}\n\n"
        "Write a short (2-4 sentence) daily attendance summary for an admin. "
        "Include the number present, anyone marked 'uncertain', and note any "
        "unknown faces still waiting to be registered. Plain prose, no bullet "
        "points, no preamble like 'Here is'."
    )
    text, source = _run(_SYSTEM, prompt)
    if text is None:
        return _fallback_report(records, known_count, pending_count), "fallback"
    return text, source


# --------------------------------------------------------------- deterministic
# Used when no LLM is available, so the endpoints always return something
# useful. Not a natural-language model — just computed summaries.
def _fallback_report(records, known_count, pending_count):
    if not records:
        base = "No attendance has been marked yet today."
    else:
        present = [r for r in records if r.get("status") in ("present", "registered")]
        uncertain = [r for r in records if r.get("status") == "uncertain"]
        parts = [f"{len(present)} of {known_count} known people marked present today."]
        if uncertain:
            names = ", ".join(r["name"] for r in uncertain)
            parts.append(f"{len(uncertain)} uncertain match(es): {names}.")
        base = " ".join(parts)
    if pending_count:
        base += f" {pending_count} unknown face(s) are waiting to be registered."
    return base + "  (No LLM active — install Ollama or set ANTHROPIC_API_KEY for AI-written summaries.)"


def _fallback_answer(question, records):
    q = question.lower()
    if not records:
        return "No one has been marked present yet today."
    if any(w in q for w in ("how many", "count", "number", "total")):
        return f"{len(records)} people have been marked so far today."
    if "uncertain" in q:
        u = [r["name"] for r in records if r.get("status") == "uncertain"]
        return ("Uncertain matches: " + ", ".join(u)) if u else "No uncertain matches today."
    if any(w in q for w in ("who", "present", "list", "marked")):
        return "Marked present: " + ", ".join(
            f"{r['name']} ({r.get('time', '?')})" for r in records
        )
    return (
        "Free-form answers need an LLM. Install Ollama or set ANTHROPIC_API_KEY. "
        f"So far {len(records)} people are marked present."
    )
