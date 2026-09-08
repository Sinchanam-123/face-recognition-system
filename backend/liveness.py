"""
Presentation-attack detection (liveness) via a vision-language model.

Why this exists
---------------
`README.md` claims the system reduces proxy attendance. Until this module, that
claim was false: `engine._recognize` embeds whatever the camera shows it, and an
ArcFace embedding of a photograph of Ada is a perfectly good embedding of Ada.
Holding a phone up to the lens marked her present. Recognition answers "whose
face is this"; nothing answered "is this a face".

What this module is
-------------------
A second, independent opinion on a cropped face, asked of a vision model:
*is this a live human in front of the camera, or a photograph of one?* It
follows the provider pattern already established in `genai.py` — a cached probe,
an ordered preference, and graceful degradation — with one deliberate
difference, stated here because it is the most important design decision in the
file:

    **There is no fallback heuristic.**

`genai.py` degrades to computed summaries because a worse answer is still a
useful answer. Liveness has no such property. A blink detector or a texture
heuristic that has not been measured on this camera would produce a number that
*looks* like anti-spoofing, and the README would go on claiming protection that
does not exist — which is the exact failure this module was written to correct.
So when no provider is available, liveness is **inactive**: `engine` behaves
precisely as it did before, `/api/status` reports anti-spoofing as unavailable,
and nothing anywhere pretends a check happened.

Activation is deliberately two-key
----------------------------------
Face imagery is biometric data. It leaves this machine only when the operator
has said so twice: `ANTHROPIC_API_KEY` (a key may be present for the Ask-AI
panel, which sends only attendance text) **and** `LIVENESS_ENABLED=true`, which
is the one that authorises egress of images. The key alone does nothing here.
Ollama is the local, no-egress alternative and needs only a reachable server
running a vision model.

Threading
---------
Nothing in this module may be called from the camera thread. Every entry point
is a blocking network call of one to three seconds; `engine` runs it on its own
worker (see `AttendanceEngine._liveness_worker`) so a slow API cannot stall
recognition or the video stream.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Final, NamedTuple

from config import (
    LIVENESS_CROP_MARGIN,
    LIVENESS_CROP_MAX_PX,
    LIVENESS_ERROR,
    LIVENESS_FAIL,
    LIVENESS_JPEG_QUALITY,
    LIVENESS_MAX_TOKENS,
    LIVENESS_PASS,
    LIVENESS_TIMEOUT_S,
    LIVENESS_UNAVAILABLE,
)
from settings import (
    LIVENESS_ENABLED,
    LIVENESS_MODEL,
    OLLAMA_HOST,
    OLLAMA_VISION_MODEL,
)

# Attack labels the model may return. Kept as a tuple so a response carrying
# anything else is rejected by the parser rather than written into an audit row.
ATTACK_TYPES: Final[tuple[str, ...]] = (
    "none",
    "printed_photo",
    "screen_replay",
    "mask",
    "unclear",
)

# The response contract, as a JSON Schema. Passed to Claude through
# `output_config.format`, which constrains generation server-side, and restated
# in the prompt for Ollama, which has no such mechanism. `additionalProperties:
# False` and the enum are what make `_parse` able to reject a malformed answer
# instead of coercing one.
RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "is_live": {"type": "boolean"},
        "attack_type": {"type": "string", "enum": list(ATTACK_TYPES)},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["is_live", "attack_type", "confidence", "reasoning"],
    "additionalProperties": False,
}

_SYSTEM: Final[str] = (
    "You are a presentation-attack detector for a face-recognition attendance "
    "system. You are shown a cropped webcam frame containing one face, with a "
    "little of the surrounding scene.\n\n"
    "Decide whether this is a LIVE human physically present in front of the "
    "camera, or a presentation attack — a photograph, a face displayed on a "
    "screen, or a mask.\n\n"
    "Evidence that indicates an attack:\n"
    "  * a rectangular border, bezel, hand or frame around the face\n"
    "  * moire or scanline patterns, pixel grid, or backlit screen glow\n"
    "  * specular glare in a flat sheet-like patch, or paper texture and creases\n"
    "  * flat, uniform lighting on the face that does not match the surrounding "
    "scene, or a face whose perspective is inconsistent with the background\n"
    "  * over-smooth or posterised skin, print dithering, colour banding\n"
    "  * visible cut edges, or a face that is unnaturally planar\n\n"
    "Evidence that indicates a live subject:\n"
    "  * consistent three-dimensional shading across nose, eye sockets and jaw\n"
    "  * natural skin micro-texture, stray hair, asymmetric expression\n"
    "  * lighting and depth of field continuous with the background\n\n"
    "Be calibrated, not cautious. `confidence` is your probability that the "
    "verdict in `is_live` is correct, from 0.0 to 1.0. If the crop is too dark, "
    "too small or too blurred to judge, return is_live false with attack_type "
    "\"unclear\" and a LOW confidence — do not guess a specific attack.\n"
    "When is_live is true, attack_type must be \"none\".\n"
    "Keep `reasoning` to one short sentence naming the decisive evidence."
)

_USER_TEXT: Final[str] = (
    "Is the face in this image a live person present at the camera, or a "
    "presentation attack? Answer with the JSON object only."
)

# Appended on the single retry after an unparseable answer. Deliberately blunt:
# the first attempt already carried the schema, so the retry's job is to stop
# the model wrapping the object in prose or a code fence.
_STRICTER: Final[str] = (
    "\n\nYour previous answer could not be parsed. Reply with ONE JSON object "
    "and nothing else — no prose, no explanation, no markdown code fence. "
    "Exactly these four keys: "
    '{"is_live": <true|false>, "attack_type": '
    f'<one of {"|".join(ATTACK_TYPES)}>, '
    '"confidence": <number 0.0-1.0>, "reasoning": "<one short sentence>"}'
)


class LivenessResult(NamedTuple):
    """One verdict about one face crop.

    `state` is what the rest of the system branches on, and it separates two
    things that must not be conflated:

        PASS   the model saw a live person.
        FAIL   the model saw a presentation attack.
        ERROR  no usable answer — a timeout, a refusal, a malformed response
               that survived the retry, or a provider that vanished mid-run.

    ERROR is treated as not-live at the enforcement point (fail closed), but it
    is recorded distinctly, because "we caught a spoof" and "we could not tell"
    have very different meanings in an audit log and in the eval harness.
    """

    state: str            # LIVENESS_PASS | LIVENESS_FAIL | LIVENESS_ERROR
    attack_type: str
    confidence: float
    reasoning: str
    provider: str         # "claude" | "ollama" | "none"
    latency_s: float
    retried: bool

    @property
    def is_live(self) -> bool:
        """True only on an affirmative PASS. ERROR is not live — fail closed."""
        return self.state == LIVENESS_PASS


def _error(reason: str, provider: str, latency: float,
           retried: bool = False) -> LivenessResult:
    """A fail-closed result. The only way this module reports 'I do not know'."""
    return LivenessResult(
        state=LIVENESS_ERROR,
        attack_type="unclear",
        confidence=0.0,
        reasoning=reason,
        provider=provider,
        latency_s=latency,
        retried=retried,
    )


# --------------------------------------------------------------- provider pick
# Same shape as genai._PROVIDER_CACHE: probing Ollama hits the network, and the
# camera path may ask several times a second during a burst of new faces.
_PROVIDER_CACHE: dict[str, Any] = {"value": None, "at": 0.0}
_PROVIDER_TTL_S: Final[float] = 15.0


def _claude_available() -> bool:
    """Claude is usable only when BOTH keys are turned.

    `ANTHROPIC_API_KEY` alone is explicitly not enough. The key may already be
    set for the Ask-AI panel, which sends attendance text; this path sends
    photographs of people's faces to a third party, and that needs its own
    affirmative opt-in.
    """
    if not LIVENESS_ENABLED:
        return False
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _ollama_available() -> bool:
    """True if Ollama answers AND actually serves the configured vision model.

    genai.py's probe stops at "the server answered", which is right for a text
    model because any chat model can attempt any prompt. Here a text-only model
    would accept the request and fail on every image, so the probe checks the
    tag list. Reporting "available" for a provider that cannot see is worse than
    reporting unavailable — it would look like liveness was running.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=1.0) as r:
            if r.status != 200:
                return False
            tags = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False

    wanted = OLLAMA_VISION_MODEL.split(":")[0]
    for entry in tags.get("models") or []:
        name = (entry.get("name") or entry.get("model") or "")
        if name == OLLAMA_VISION_MODEL or name.split(":")[0] == wanted:
            return True
    return False


def provider() -> str | None:
    """The active provider: 'claude', 'ollama', or None when liveness is off."""
    now = time.time()
    if now - _PROVIDER_CACHE["at"] < _PROVIDER_TTL_S:
        return _PROVIDER_CACHE["value"]
    if _claude_available():
        value: str | None = "claude"
    elif _ollama_available():
        value = "ollama"
    else:
        value = None
    _PROVIDER_CACHE.update(value=value, at=now)
    return value


def is_active() -> bool:
    """True when a real liveness check can run.

    When this is False the engine must behave exactly as it did before this
    feature existed — no held-back marks, no flagged cards, no attendance
    column values other than 'unavailable'.
    """
    return provider() is not None


def provider_name() -> str:
    """Human-readable provider label for the UI."""
    return {
        "claude": f"Claude ({LIVENESS_MODEL})",
        "ollama": f"Ollama ({OLLAMA_VISION_MODEL})",
    }.get(provider(), "unavailable")


def status() -> dict[str, Any]:
    """What `/api/status` reports about anti-spoofing.

    `reason` exists so the UI can say *why* it is off. "Anti-spoofing: off" with
    no explanation is how a deployment ends up believing the feature is running.
    """
    active = is_active()
    if active:
        reason = None
    elif not LIVENESS_ENABLED and os.environ.get("ANTHROPIC_API_KEY"):
        reason = (
            "ANTHROPIC_API_KEY is set but LIVENESS_ENABLED is not true. Face "
            "images are not sent anywhere until you enable it explicitly."
        )
    elif LIVENESS_ENABLED and not os.environ.get("ANTHROPIC_API_KEY"):
        reason = (
            "LIVENESS_ENABLED is true but ANTHROPIC_API_KEY is unset and no "
            "Ollama vision model was found."
        )
    else:
        reason = (
            "No liveness provider configured. Set ANTHROPIC_API_KEY and "
            "LIVENESS_ENABLED=true, or run Ollama with a vision model. There "
            "is deliberately no heuristic fallback."
        )
    return {
        "active": active,
        "provider": provider_name() if active else None,
        "model": LIVENESS_MODEL if provider() == "claude" else (
            OLLAMA_VISION_MODEL if provider() == "ollama" else None
        ),
        "enabled_flag": LIVENESS_ENABLED,
        "reason": reason,
    }


# ------------------------------------------------------------------- the crop
def crop_for_liveness(frame: Any, bbox: tuple[int, int, int, int]) -> bytes | None:
    """JPEG of the face plus modest surrounding context, or None if unusable.

    Two reasons not to send the whole frame. The obvious one is that less data
    leaves the machine — a 512 px crop of one person rather than a room. The
    less obvious one is that it is a *better* input: the tells this check relies
    on (a phone bezel, a moire pattern, paper texture, glare on a flat sheet)
    live within a face-width of the face, and a downscaled full frame spends its
    resolution on the wall behind.

    The margin is what makes the difference between the two: a tight face crop
    would cut off the very bezel that gives a screen replay away, so
    LIVENESS_CROP_MARGIN deliberately keeps a border around the detection.

    cv2 is imported here rather than at module scope, per the backend
    convention — importing this module must not require the recognition stack.
    """
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    face_w, face_h = x2 - x1, y2 - y1
    if face_w <= 0 or face_h <= 0:
        return None

    pad_x = int(face_w * LIVENESS_CROP_MARGIN)
    pad_y = int(face_h * LIVENESS_CROP_MARGIN)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Downscale only. Upscaling a small face invents no detail and costs image
    # tokens for pixels that carry nothing.
    long_side = max(crop.shape[0], crop.shape[1])
    if long_side > LIVENESS_CROP_MAX_PX:
        scale = LIVENESS_CROP_MAX_PX / long_side
        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ok, buf = cv2.imencode(
        ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), LIVENESS_JPEG_QUALITY]
    )
    if not ok:
        return None
    return buf.tobytes()


# ----------------------------------------------------------------- the parser
def _parse(text: str) -> dict[str, Any] | None:
    """Validate a model response against the contract. None means malformed.

    Deliberately strict, and deliberately not "best effort". A liveness verdict
    that was salvaged by guessing what the model probably meant is a verdict
    nobody can defend, and this is the one place in the system where being
    wrong marks the wrong person present. Anything that does not parse cleanly
    becomes an ERROR, which fails closed.

    The one tolerance: a fenced or prose-wrapped object is unwrapped by taking
    the outermost braces. That is a formatting artefact, not an ambiguity about
    what the model decided.
    """
    if not text:
        return None

    candidate = text.strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = candidate[start:end + 1]

    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    if not isinstance(data.get("is_live"), bool):
        return None
    attack = data.get("attack_type")
    if attack not in ATTACK_TYPES:
        return None
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return None
    if not 0.0 <= float(confidence) <= 1.0:
        return None
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str):
        return None

    # A live verdict with a named attack is self-contradictory; the model got
    # the shape right and the semantics wrong, which is exactly the case the
    # retry is for.
    if data["is_live"] and attack != "none":
        return None

    return {
        "is_live": data["is_live"],
        "attack_type": attack,
        "confidence": float(confidence),
        # Bounded: this string lands in an audit row and in the UI.
        "reasoning": reasoning.strip()[:300],
    }


# ------------------------------------------------------------------ providers
def _ask_claude(jpeg: bytes, strict: bool) -> str:
    """One vision call to Claude. Raises on transport/API failure.

    `output_config.format` constrains generation to RESPONSE_SCHEMA server-side,
    so the retry path should effectively never fire here — it exists for Ollama,
    which has no equivalent, and as defence in depth.

    NOTE: this path has never been executed. It is written against the documented
    SDK surface and left inactive by default; the first operator to set
    LIVENESS_ENABLED is also the first to run it. See eval/liveness/README.md.
    """
    import anthropic

    text = _USER_TEXT + (_STRICTER if strict else "")
    resp = anthropic.Anthropic().messages.create(
        model=LIVENESS_MODEL,
        max_tokens=LIVENESS_MAX_TOKENS,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(jpeg).decode("ascii"),
                    },
                },
                {"type": "text", "text": text},
            ],
        }],
        output_config={
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            # This is a perception call, not a reasoning one, and it sits in
            # front of an attendance mark: latency is a user-visible cost. Low
            # effort also keeps the billed thinking tokens down, which is where
            # most of the per-check price goes on the Opus tier.
            "effort": "low",
        },
        timeout=LIVENESS_TIMEOUT_S,
    )
    # stop_reason "refusal" returns HTTP 200 with no usable content, so check it
    # before reading blocks rather than after failing to find text.
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError("model declined to answer")
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _ask_ollama(jpeg: bytes, strict: bool) -> str:
    """One vision call to a local Ollama model. Raises on transport failure.

    Ollama's `format` accepts a JSON Schema, which is the closest local
    equivalent to `output_config.format`; it is advisory in practice, so the
    parser and the retry carry the real weight here.
    """
    text = _USER_TEXT + (_STRICTER if strict else "")
    body = json.dumps({
        "model": OLLAMA_VISION_MODEL,
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "options": {"num_predict": LIVENESS_MAX_TOKENS, "temperature": 0.0},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": text,
                "images": [base64.standard_b64encode(jpeg).decode("ascii")],
            },
        ],
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=LIVENESS_TIMEOUT_S) as r:
        data = json.loads(r.read())
    return (data.get("message", {}).get("content") or "").strip()


def _call(active: str, jpeg: bytes, strict: bool) -> str:
    if active == "claude":
        return _ask_claude(jpeg, strict)
    return _ask_ollama(jpeg, strict)


# ---------------------------------------------------------------- public API
def check(jpeg: bytes) -> LivenessResult:
    """Ask the active provider whether this face crop is a live person.

    Never raises. Every failure — no provider, a timeout, an HTTP error, a
    response that will not parse twice — becomes an ERROR result, which the
    enforcement point treats as not-live.

    One retry, and only one. A malformed answer is usually a formatting slip
    that a blunter instruction fixes; a second malformed answer is a model that
    cannot do this task, and retrying further would just add latency and cost in
    front of an attendance decision.

    MUST NOT be called from the camera thread — it blocks for as long as
    LIVENESS_TIMEOUT_S.
    """
    active = provider()
    started = time.perf_counter()
    if active is None:
        return _error("no liveness provider configured", "none", 0.0)
    if not jpeg:
        return _error("empty crop", active, 0.0)

    retried = False
    for strict in (False, True):
        try:
            text = _call(active, jpeg, strict)
        except Exception as e:
            # Boundary: a provider failure must not propagate into the engine's
            # worker loop. The reason is preserved because it is the only place
            # a misconfiguration becomes visible.
            return _error(
                f"{active} call failed: {type(e).__name__}: {e}",
                active, time.perf_counter() - started, retried,
            )

        parsed = _parse(text)
        if parsed is not None:
            return LivenessResult(
                state=LIVENESS_PASS if parsed["is_live"] else LIVENESS_FAIL,
                attack_type=parsed["attack_type"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                provider=active,
                latency_s=time.perf_counter() - started,
                retried=retried,
            )
        retried = True

    return _error(
        "response did not match the required JSON schema after one retry",
        active, time.perf_counter() - started, True,
    )


__all__ = [
    "ATTACK_TYPES",
    "LIVENESS_ERROR",
    "LIVENESS_FAIL",
    "LIVENESS_PASS",
    "LIVENESS_UNAVAILABLE",
    "LivenessResult",
    "RESPONSE_SCHEMA",
    "check",
    "crop_for_liveness",
    "is_active",
    "provider",
    "provider_name",
    "status",
]
