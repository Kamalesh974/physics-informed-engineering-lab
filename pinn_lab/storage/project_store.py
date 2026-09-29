"""
project_store.py -- the shared project folder that connects BondLab and PINNlab.

    workspace/projects/<project-id>/
    ├── project.json                 identity, origin, status of every part
    ├── bondgraph/
    │   ├── model.json               the BondLab document: graph, layout, values, sources, ICs, BCs
    │   └── equations.json           structured equations derived by the BondLab engine + PINN model text
    ├── pinn/
    │   ├── configuration.json       PINN / training settings last used
    │   └── checkpoints/<run>/*.pt   PyTorch checkpoints
    ├── measurement_data/
    │   ├── <file>.csv               uploaded measurements
    │   └── dataset.json             how the CSV is used (time column, mapping, split)
    └── results/
        ├── metrics.json             RMSE / MAE / R2 of the last completed run
        └── prediction.csv           PINN prediction vs reference on the evaluation grid

Everything is plain JSON/CSV (+ .pt checkpoints) so a project can be copied, zipped or inspected.
Writes are atomic (temp file + rename). workspace/active_project.json remembers which project
PINNlab shows.
"""
import json
import os
import re
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("LAB_WORKSPACE") or (REPO_ROOT / "workspace"))
PROJECTS = WORKSPACE / "projects"
ACTIVE_FILE = WORKSPACE / "active_project.json"
DEMO_ID = "demo"
FORMAT = "physics-informed-engineering-lab/project/1"


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def slugify(name):
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(name or "")).strip("-").lower()[:48]
    return s or "project"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class Project:
    def __init__(self, pid):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", pid or ""):
            raise ValueError(f"invalid project id {pid!r}")
        self.id = pid
        self.root = PROJECTS / pid

    # paths ---------------------------------------------------------------
    project_json = property(lambda self: self.root / "project.json")
    model_json = property(lambda self: self.root / "bondgraph" / "model.json")
    equations_json = property(lambda self: self.root / "bondgraph" / "equations.json")
    configuration_json = property(lambda self: self.root / "pinn" / "configuration.json")
    checkpoints_dir = property(lambda self: self.root / "pinn" / "checkpoints")
    data_dir = property(lambda self: self.root / "measurement_data")
    dataset_json = property(lambda self: self.root / "measurement_data" / "dataset.json")
    results_dir = property(lambda self: self.root / "results")

    def exists(self):
        return self.project_json.is_file()

    def ensure_dirs(self):
        for d in (self.model_json.parent, self.checkpoints_dir, self.data_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    # project.json ----------------------------------------------------------
    def meta(self):
        return read_json(self.project_json, {}) or {}

    def update_meta(self, **changes):
        m = self.meta()
        for k, v in changes.items():
            if isinstance(v, dict) and isinstance(m.get(k), dict):
                m[k] = {**m[k], **v}
            else:
                m[k] = v
        m["updated"] = _now()
        write_json(self.project_json, m)
        return m

    @property
    def name(self):
        return self.meta().get("name", self.id)

    @property
    def source(self):
        return self.meta().get("source", "")

    # bond graph / equations --------------------------------------------------
    def model(self):
        return read_json(self.model_json)

    def equations(self):
        return read_json(self.equations_json)

    def model_text(self):
        eq = self.equations() or {}
        return eq.get("model_text")

    def save_model_text(self, text, note="edited in PINNlab"):
        """Store an edited model text (e.g. parameters changed in PINNlab) next to the BondLab record."""
        eq = self.equations() or {}
        eq["model_text"] = text
        eq.setdefault("history", []).append({"at": _now(), "change": note})
        write_json(self.equations_json, eq)
        self.update_meta(modified_in_pinnlab=True)

    # PINN configuration ------------------------------------------------------
    def configuration(self):
        return read_json(self.configuration_json)

    def save_configuration(self, cfg):
        write_json(self.configuration_json, {"saved": _now(), "config": cfg})
        self.update_meta(pinn={"configuration": "pinn/configuration.json"})

    # measurement data ---------------------------------------------------------
    def dataset_settings(self):
        return read_json(self.dataset_json)

    def save_dataset(self, settings, csv_bytes=None, filename=None):
        """settings: {source, time_column, mapping, val_fraction, ...}. csv_bytes: raw upload to keep."""
        self.ensure_dirs()
        if csv_bytes is not None:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "data.csv")
            if not safe.lower().endswith(".csv"):
                safe += ".csv"
            (self.data_dir / safe).write_bytes(csv_bytes)
            settings = dict(settings, file=safe)
        write_json(self.dataset_json, dict(settings, saved=_now()))
        self.update_meta(measurement_data={"source": settings.get("source"), "file": settings.get("file")})

    def dataset_csv(self):
        st = self.dataset_settings() or {}
        f = st.get("file")
        return (self.data_dir / f) if f and (self.data_dir / f).is_file() else None

    def remove_dataset(self):
        if self.data_dir.exists():
            for p in self.data_dir.iterdir():
                if p.is_file():
                    p.unlink()
        write_json(self.dataset_json, {"source": "none", "saved": _now()})
        self.update_meta(measurement_data={"source": "none", "file": None})

    # results -------------------------------------------------------------------
    def save_results(self, metrics, csv_text):
        self.ensure_dirs()
        write_json(self.results_dir / "metrics.json", metrics)
        (self.results_dir / "prediction.csv").write_text(csv_text, encoding="utf-8")
        self.update_meta(results={"metrics": "results/metrics.json", "prediction": "results/prediction.csv",
                                  "saved": _now()})

    def summary(self):
        m = self.meta()
        return {"id": self.id, "name": m.get("name", self.id), "source": m.get("source", ""),
                "updated": m.get("updated", ""), "n_states": (m.get("bondgraph") or {}).get("n_states")}


# ---------------------------------------------------------------------------
def get(pid):
    p = Project(pid)
    if not p.exists():
        raise FileNotFoundError(f"project '{pid}' not found")
    return p


def list_projects():
    if not PROJECTS.exists():
        return []
    out = [Project(d.name) for d in PROJECTS.iterdir() if d.is_dir() and (d / "project.json").is_file()
           and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", d.name)]
    return sorted(out, key=lambda p: p.meta().get("updated", ""), reverse=True)


def get_active():
    a = read_json(ACTIVE_FILE, {}) or {}
    pid = a.get("id")
    if pid:
        try:
            return get(pid)
        except (FileNotFoundError, ValueError):
            pass
    return None


def set_active(pid):
    write_json(ACTIVE_FILE, {"id": pid, "since": _now()})


def _create(pid, name, source, description=""):
    p = Project(pid)
    p.ensure_dirs()
    if not p.exists():
        write_json(p.project_json, {"format": FORMAT, "id": pid, "name": name, "source": source,
                                    "description": description, "created": _now(), "updated": _now(),
                                    "layout": {"bondgraph": "bondgraph/", "pinn": "pinn/",
                                               "measurement_data": "measurement_data/", "results": "results/"}})
    return p


def save_bondlab_model(document, equations_record, model_text, pid=None):
    """PINNSIM: store the BondLab model + its derived equations as a project (created or updated)."""
    name = (document.get("meta") or {}).get("name") or "BondLab model"
    pid = pid or slugify(name)
    p = _create(pid, name, "bondlab", (document.get("meta") or {}).get("description", ""))
    write_json(p.model_json, {"saved": _now(), "document": document})
    rec = dict(equations_record, model_text=model_text, transferred=_now())
    write_json(p.equations_json, rec)
    p.update_meta(name=name, source="bondlab", modified_in_pinnlab=False,
                  bondgraph={"model": "bondgraph/model.json", "equations": "bondgraph/equations.json",
                             "n_states": len(rec.get("states", [])), "transferred": rec["transferred"]})
    return p


def ensure_demo(demo_text, drawing_template="mass_spring_damper"):
    """The built-in 'Demo Physics Problem' project (created on first use)."""
    p = _create(DEMO_ID, "Demo Physics Problem — mass-spring-damper", "demo",
                "Built-in example: BondLab's mass-spring-damper equations.")
    if not p.equations_json.is_file():
        write_json(p.equations_json, {"format": "pinnlab-demo/1", "title": "Demo Physics Problem",
                                      "template_id": drawing_template, "model_text": demo_text,
                                      "transferred": _now()})
    return p


def delete_project(pid):
    if pid == DEMO_ID:
        raise ValueError("the demo project cannot be deleted")
    p = get(pid)
    shutil.rmtree(p.root)
