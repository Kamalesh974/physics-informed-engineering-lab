"""
checkpoint.py -- PyTorch checkpoints for the PINN.

Each .pt file holds everything needed to continue training exactly where it
stopped: model and optimizer state_dicts (Adam moments included), LR
scheduler state, the epoch counter, the latest loss terms, the best
validation loss, the training configuration, the governing-equation text,
the scaling, the dataset, the loss history and the sampling RNG state.

Only tensors and plain Python types are stored, so files are loaded with
torch.load(weights_only=True): loading a checkpoint never executes pickled
code, even for a file you did not create.

Layout:
    checkpoints/<run_id>/checkpoint_epoch_1000.pt
    checkpoints/<run_id>/checkpoint_epoch_2000.pt
    checkpoints/<run_id>/best_model.pt
    checkpoints/<run_id>/index.json      (small summary, for the UI list)
"""
import json
import os
import re
import time
from pathlib import Path

import torch

FORMAT = "pinn-lab-checkpoint/1"
CHECKPOINT_ROOT = Path(__file__).resolve().parents[1] / "checkpoints"


def new_run_dir(title, root=None):
    root = Path(root or CHECKPOINT_ROOT)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()[:40] or "run"
    d = root / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}"
    n = 1
    while d.exists():
        n += 1
        d = root / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}_{n}"
    d.mkdir(parents=True)
    return d


def save(payload, path):
    """Atomic save (write to a temp file, then rename) + update index.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, format=FORMAT, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    _update_index(path, payload)
    return path


def load(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or ck.get("format") != FORMAT:
        raise ValueError(f"{path.name} is not a PINN-lab checkpoint")
    return ck


def _summary(path, payload):
    return {
        "file": path.name, "epoch": int(payload.get("epoch", 0)), "kind": payload.get("kind", ""),
        "total_loss": payload.get("total_loss"), "physics_loss": payload.get("physics_loss"),
        "data_loss": payload.get("data_loss"), "best_val_loss": payload.get("best_val_loss"),
        "val_loss": payload.get("val_loss"),
        "saved_at": payload.get("saved_at"), "title": payload.get("title", ""),
    }


def _update_index(path, payload):
    idx_path = path.parent / "index.json"
    try:
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    except (OSError, ValueError):
        idx = {}
    idx[path.name] = _summary(path, payload)
    idx_path.write_text(json.dumps(idx, indent=1))


def delete(path):
    """Remove one checkpoint file and its index entry."""
    path = Path(path)
    if path.suffix != ".pt" or not path.is_file():
        raise FileNotFoundError(str(path))
    path.unlink()
    idx_path = path.parent / "index.json"
    try:
        idx = json.loads(idx_path.read_text())
        idx.pop(path.name, None)
        idx_path.write_text(json.dumps(idx, indent=1))
    except (OSError, ValueError):
        pass


def list_checkpoints(root=None):
    """All checkpoints under root, newest run first; uses index.json summaries."""
    root = Path(root or CHECKPOINT_ROOT)
    out = []
    if not root.exists():
        return out
    for run in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            idx = json.loads((run / "index.json").read_text())
        except (OSError, ValueError):
            idx = {}
        for f in run.glob("*.pt"):
            s = idx.get(f.name) or {"file": f.name, "epoch": None, "kind": "", "total_loss": None}
            out.append(dict(s, run=run.name, path=str(f)))
    out.sort(key=lambda s: (s["run"], s.get("epoch") or 0), reverse=True)
    return out
