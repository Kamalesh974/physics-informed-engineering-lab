"""
vision.py -- image import via the Gemini vision model (structure ONLY; all
physics is done by the deterministic engine afterwards).

Free-tier protections (the free quota is ~20 calls/day per Google project):
  * every image is hashed; a repeated image is served from disk cache, no call
  * an app-side daily counter (Pacific day, like Google's reset) is shown to the
    user and stops us calling once the allowance is used up
  * a daily-quota 429 is never retried (see image_to_bond_graph._send_with_backoff)
  * a bring-your-own key (header) bypasses the app counter and is never stored
"""
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timedelta, timezone

from PIL import Image

from .engine import IDENT, ROOT, derive_payload

log = logging.getLogger("bondlab")


def _data_dir():
    """Where the daily counter and the image cache live.
    BONDLAB_DATA_DIR wins; on Vercel only /tmp is writable (and it is per-instance and
    short-lived, so the counter/cache there are best-effort); otherwise app/data."""
    d = os.environ.get("BONDLAB_DATA_DIR")
    if d:
        return d
    if os.environ.get("VERCEL"):
        return os.path.join(tempfile.gettempdir(), "bondlab")
    return os.path.join(ROOT, "app", "data")


DATA_DIR = _data_dir()
CACHE_DIR = os.path.join(DATA_DIR, "cache")
USAGE_FILE = os.path.join(DATA_DIR, "usage.json")
DAILY_LIMIT = int(os.environ.get("GEMINI_DAILY_LIMIT", "20"))
MAX_BYTES = 8 * 1024 * 1024
MAX_SIDE = 1600
Image.MAX_IMAGE_PIXELS = 40_000_000

_ulock = threading.Lock()
_mem_usage = None  # in-memory counter, used only while the usage file cannot be written


class ImageError(ValueError):
    pass


def _pacific_day():
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
    except Exception:  # noqa: BLE001  (tzdata missing on some Windows installs)
        now = datetime.now(timezone.utc) - timedelta(hours=8)
    return now.strftime("%Y-%m-%d")


def _read_usage():
    day = _pacific_day()
    try:
        with open(USAGE_FILE, "r", encoding="utf-8") as fh:
            u = json.load(fh)
        if u.get("day") == day:
            return {"day": day, "calls": int(u.get("calls", 0))}
    except (OSError, ValueError):
        if _mem_usage and _mem_usage["day"] == day:
            return dict(_mem_usage)
    return {"day": day, "calls": 0}


def usage():
    with _ulock:
        u = _read_usage()
    return {"day": u["day"], "used": u["calls"], "limit": DAILY_LIMIT, "remaining": max(DAILY_LIMIT - u["calls"], 0),
            "resets": "midnight Pacific time"}


def _write_usage(calls):
    global _mem_usage
    rec = {"day": _pacific_day(), "calls": int(calls)}
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(USAGE_FILE, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        _mem_usage = None
    except OSError:
        # read-only or missing disk: keep counting in memory instead of failing the request
        _mem_usage = rec
        log.warning("usage counter could not be written to disk; counting in memory only")


def add_usage(n):
    with _ulock:
        _write_usage(_read_usage()["calls"] + n)


def mark_exhausted():
    with _ulock:
        _write_usage(max(_read_usage()["calls"], DAILY_LIMIT))


def prepare_image(raw):
    if len(raw) > MAX_BYTES:
        raise ImageError("Image too large (max 8 MB).")
    try:
        img = Image.open(io.BytesIO(raw))
        fmt = img.format
        img.load()
    except Exception:  # noqa: BLE001
        raise ImageError("That file is not a readable image (use PNG, JPG or WEBP).")
    if fmt not in ("PNG", "JPEG", "WEBP", "GIF", "BMP"):
        raise ImageError("Unsupported image type (use PNG, JPG or WEBP).")
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _cache_path(h):
    return os.path.join(CACHE_DIR, f"{h}.json")


def _load_cache(h):
    try:
        with open(_cache_path(h), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _save_cache(h, obj):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(h), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
    except OSError:
        log.warning("image cache could not be written; continuing without it")


def _clean(name, used):
    s = re.sub(r"\W+", "_", str(name)).strip("_") or "x"
    if not s[0].isalpha():
        s = "n" + s
    s = s[:30]
    base, k = s, 2
    while s in used:
        s, k = f"{base}_{k}", k + 1
    return s


def normalize_spec(spec):
    """AI output can use ids like 'R 1' or '1/k'; make everything valid identifiers."""
    nodes = [dict(n) for n in spec.get("nodes", []) if isinstance(n, dict)]
    idmap, used = {}, set()
    for n in nodes:
        new = n["id"] if isinstance(n.get("id"), str) and IDENT.match(n["id"]) and n["id"] not in used else _clean(n.get("id", "n"), used)
        idmap[n.get("id")] = new
        used.add(new)
        n["id"] = new
    pmap = {}
    for n in nodes:
        for k in ("param_symbol_name", "expr_param_name"):
            if isinstance(n.get(k), str) and not IDENT.match(n[k]):
                pmap.setdefault(n[k], _clean(n[k], set(pmap.values())))
                n[k] = pmap[n[k]]
        if n.get("signal_from") in idmap:
            n["signal_from"] = idmap[n["signal_from"]]
        n.setdefault("label", n["id"])
    edges = []
    for e in spec.get("edges", []):
        if isinstance(e, dict) and e.get("from") in idmap and e.get("to") in idmap:
            edges.append({"from": idmap[e["from"]], "to": idmap[e["to"]], "signal": bool(e.get("signal", False))})
    return {"nodes": nodes, "edges": edges}


def classify_error(msg):
    m = msg or ""
    if "RESOURCE_EXHAUSTED" in m or " 429" in m:
        return "quota", ("The free AI-reading allowance for today is used up (it resets at midnight Pacific time). "
                         "You can still build the graph by hand in the editor, or try again later.")
    if "UNAVAILABLE" in m or "503" in m:
        return "busy", "The AI service is busy right now. Wait a minute and try again."
    if any(x in m for x in ("API key", "API_KEY_INVALID", "PERMISSION_DENIED", "401", "403")):
        return "auth", "The AI service rejected the API key."
    return "read_failed", "The AI could not produce a valid bond graph from this image. " + (m[:280] if m else "")


def read_image(raw, api_key=None, byo=False):
    png = prepare_image(raw)
    h = hashlib.sha256(png).hexdigest()
    cached = _load_cache(h)
    if cached:
        cached["cached"] = True
        return cached

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {"ok": False, "error_code": "no_key", "error": "The server has no Gemini API key configured. Use the manual editor, or provide your own key."}
    if not byo and usage()["remaining"] <= 0:
        code, msg = classify_error("RESOURCE_EXHAUSTED")
        return {"ok": False, "error_code": code, "error": msg}

    from google import genai
    import image_to_bond_graph as itb

    client = genai.Client(api_key=key)
    # ONE retry only for a transient 'busy' reply: the free tier counts every request Google receives,
    # failed 503s included, so a long retry chain during an outage would burn the whole day's allowance.
    result = itb.run_pipeline(png, "image/png", client, max_retries=3, max_api_retries=2, verbose=False)
    if not byo:
        # count the requests actually sent (retries included), as Google does
        add_usage(max(1, int(result.get("api_calls") or result.get("attempts_used", 1))))
    if not result["success"]:
        code, msg = classify_error(result.get("error", ""))
        if code == "quota" and not byo:
            mark_exhausted()
        return {"ok": False, "error_code": code, "error": msg}

    spec = normalize_spec(result["spec"])
    payload = derive_payload(spec)
    checks = [{"name": "Causality (SCAP) resolved", "ok": bool(payload.get("ok"))},
              {"name": "Every declared parameter used", "ok": not payload.get("unused_parameters", [])}]
    out = {"ok": True, "cached": False, "hash": h, "spec": spec, "derive": payload,
           "confidence": {"checks": checks, "voting": None,
                          "verdict": ("Automated checks passed on a single read."
                                      if all(c["ok"] for c in checks) else "At least one automated check raised a flag."),
                          "warning": ("Passing every automated check is NOT proof the AI read the diagram correctly "
                                      "(a real motor example passed all checks and was still physically wrong). "
                                      "Review the graph and the equations before trusting them.")}}
    _save_cache(h, out)
    return out
