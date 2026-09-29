"""
BondLab API.  Run from the project root:
    python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
Serves the built PWA (app/frontend/dist) at "/" when it exists.
"""
import logging
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import library, vision
from .engine import ROOT, SpecError, derive_payload
from .sim import simulate

log = logging.getLogger("bondlab")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="BondLab API", version="0.1.0")
_origins = [o.strip() for o in os.environ.get("BONDLAB_CORS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


# ---- tiny in-memory per-IP rate limiter (single process; fine for MVP) -------
_hits = defaultdict(deque)


def client_ip(request: Request):
    # Behind Vercel every request arrives from its proxy; the real client is in X-Forwarded-For,
    # which Vercel overwrites itself so it cannot be spoofed. Elsewhere it CAN be spoofed, so only
    # trust it when running on Vercel.
    if os.environ.get("VERCEL"):
        fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if fwd:
            return fwd
    return request.client.host if request.client else "?"


def limit(request: Request, bucket, max_calls, window_s):
    ip = client_ip(request)
    q = _hits[(bucket, ip)]
    now = time.monotonic()
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= max_calls:
        raise HTTPException(429, f"Too many requests. Try again in a little while.")
    q.append(now)


class SpecBody(BaseModel):
    spec: dict


class SimBody(BaseModel):
    spec: dict
    values: dict = {}
    sources: dict = {}
    initial: dict = {}
    t_end: float = 1.0
    n: int = 500


@app.get("/api/health")
def health():
    return {"ok": True, "name": "BondLab", "version": app.version}


@app.get("/api/quota")
def quota():
    return vision.usage()


@app.get("/api/library")
def get_library():
    return library.components()


@app.get("/api/templates")
def get_templates():
    return library.list_templates()


@app.get("/api/templates/{tid}")
def get_template(tid: str):
    t = library.get_template(tid)
    if t is None:
        raise HTTPException(404, "Unknown template")
    return t


@app.post("/api/derive")
def derive(body: SpecBody, request: Request):
    limit(request, "derive", 240, 60)
    return derive_payload(body.spec)


@app.post("/api/simulate")
def simulate_route(body: SimBody, request: Request):
    limit(request, "sim", 120, 60)
    try:
        return simulate(body.spec, body.values, body.sources, body.initial, body.t_end, body.n)
    except SpecError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        log.exception("simulate failed")
        return {"ok": False, "error": f"Simulation failed ({type(e).__name__})."}


@app.post("/api/from-image")
def from_image(request: Request, file: UploadFile = File(...), x_gemini_key: str | None = Header(default=None)):
    limit(request, "image", 10, 3600)
    raw = file.file.read(vision.MAX_BYTES + 1)
    try:
        # the user's own key is used for this one request only; it is never logged or stored
        return vision.read_image(raw, api_key=x_gemini_key, byo=bool(x_gemini_key))
    except vision.ImageError as e:
        return JSONResponse({"ok": False, "error_code": "bad_image", "error": str(e)}, status_code=200)
    except Exception:  # noqa: BLE001
        log.exception("from-image failed")
        return {"ok": False, "error_code": "internal", "error": "Something went wrong reading that image."}


_dist = os.path.join(ROOT, "app", "frontend", "dist")
# The integrated Physics-Informed Engineering Lab (integrated/server.py) serves the web app under
# /bondlab/ itself and sets BONDLAB_SERVE_STATIC=0; standalone BondLab (and Vercel) serve it at "/".
if os.path.isdir(_dist) and os.environ.get("BONDLAB_SERVE_STATIC", "1") != "0":
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
