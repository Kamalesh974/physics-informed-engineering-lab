"""
server.py -- Physics-Informed Engineering Lab: ONE server for the landing page, BondLab and PINNlab.

    /                         landing page (integrated/static/index.html)
    /bondlab/                 BondLab web app (the React build in app/frontend/dist)
    /api/...                  BondLab API (derive, simulate, templates, ...)  -- app.backend.main
    /api/lab/status           integration status, active project, project list
    /api/lab/pinnsim   POST   PINNSIM: BondLab project -> equations derived by the BondLab engine ->
                              project folder (workspace/projects/<id>/) -> PINNlab
    /api/lab/projects/<id>/model   the stored BondLab document (to reopen it in BondLab)
    /pinnlab/                 PINNlab (Dash, mounted as a WSGI app in the same process)

Run from the repository root:
    python -m integrated.server               # http://127.0.0.1:8000
    python -m integrated.server --port 8080 --device cuda
"""
import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_lab"
STATIC = Path(__file__).resolve().parent / "static"
DIST = ROOT / "app" / "frontend" / "dist"
os.environ.setdefault("BONDLAB_SERVE_STATIC", "0")        # this server mounts BondLab under /bondlab/
for p in (str(ROOT), str(PINN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

faulthandler.enable()
import pandas  # noqa: E402,F401  -- load pyarrow's DLLs before torch (Windows access-violation otherwise)
from a2wsgi import WSGIMiddleware  # noqa: E402
from fastapi import Body  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.routing import Mount  # noqa: E402

from app.backend.main import app  # noqa: E402  -- the BondLab FastAPI app (all /api routes)
from bondgraph import bondlab_bridge as bridge  # noqa: E402
from bondgraph.equations import ModelError, parse_model  # noqa: E402
from storage import project_store as store  # noqa: E402
from ui.dashboard import create_app  # noqa: E402

MAX_DOC_BYTES = 2_000_000
LINKS = {"home": "/", "bondlab": "/bondlab/?launch=1", "bondlab_custom": "/bondlab/?launch=1&from=pinnlab"}

app.title = "Physics-Informed Engineering Lab"


@app.get("/api/lab/status")
def lab_status():
    active = store.get_active()
    return {"integrated": True, "name": "Physics-Informed Engineering Lab", "home_url": "/",
            "bondlab_url": "/bondlab/", "pinnlab_url": "/pinnlab/",
            "active_project": active.summary() if active else None,
            "projects": [p.summary() for p in store.list_projects()]}


@app.post("/api/lab/pinnsim")
def pinnsim(payload: dict = Body(...)):
    """Save the BondLab model and hand its equations to PINNlab as physics constraints."""
    doc = payload.get("project") if isinstance(payload, dict) else None
    if not isinstance(doc, dict) or not isinstance(doc.get("spec"), dict):
        return JSONResponse({"ok": False, "error": "expected {project: <BondLab project document>}"}, status_code=400)
    if len(json.dumps(doc)) > MAX_DOC_BYTES:
        return JSONResponse({"ok": False, "error": "project document too large"}, status_code=413)
    try:
        record = bridge.project_to_equations(doc)          # re-derived by the BondLab engine from the graph
        text = bridge.equations_to_model_text(record)
        system = parse_model(text)                          # proves the equations compile into residuals
    except (ValueError, KeyError, TypeError, ModelError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except Exception as e:  # noqa: BLE001  -- engine errors (bad graph) are the user's to fix, not a 500
        return JSONResponse({"ok": False, "error": f"BondLab could not derive the equations: {e}"}, status_code=400)
    if [s["name"] for s in record["states"]] != list(system.states):
        return JSONResponse({"ok": False, "error": "internal mismatch between derived states and compiled model"},
                            status_code=500)
    p = store.save_bondlab_model(doc, record, text)
    store.set_active(p.id)
    return {"ok": True, "project_id": p.id, "name": p.name, "redirect": f"/pinnlab/?project={p.id}",
            "states": list(system.states), "n_equations": len(record["equations"]),
            "n_parameters": len(record["parameters"]), "n_initial_conditions": len(record["initial_conditions"]),
            "n_boundary_conditions": len(record["boundary_conditions"]),
            "folder": str(p.root)}


@app.get("/api/lab/projects/{pid}/model")
def project_model(pid: str):
    try:
        p = store.get(pid)
    except (FileNotFoundError, ValueError):
        return JSONResponse({"ok": False, "error": "no such project"}, status_code=404)
    doc = (p.model() or {}).get("document")
    if not doc:
        return JSONResponse({"ok": False, "error": "this project has no BondLab model"}, status_code=404)
    return {"ok": True, "project_id": p.id, "document": doc}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/bondlab", include_in_schema=False)
def bondlab_slash():
    return RedirectResponse("/bondlab/")


@app.get("/pinnlab", include_in_schema=False)
def pinnlab_slash():
    return RedirectResponse("/pinnlab/")


def build(default_device="auto"):
    # If BondLab's module was imported before BONDLAB_SERVE_STATIC was set (e.g. by another import in the
    # same process), its catch-all "/" static mount would shadow every route below: remove it.
    app.router.routes = [r for r in app.router.routes if not (isinstance(r, Mount) and r.name == "web")]
    dash_app = create_app(prefix="/pinnlab/", links=LINKS, default_device=default_device)
    app.mount("/pinnlab", WSGIMiddleware(dash_app.server, workers=16))
    if DIST.is_dir():
        app.mount("/bondlab", StaticFiles(directory=str(DIST), html=True), name="bondlab")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="lab-static")
    return app


def main():
    ap = argparse.ArgumentParser(description="Physics-Informed Engineering Lab (BondLab + PINNlab)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    import torch
    import uvicorn
    torch.set_num_threads(max(1, args.threads))
    build(args.device)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"Physics-Informed Engineering Lab on http://{args.host}:{args.port}  (CUDA GPU: {gpu})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
