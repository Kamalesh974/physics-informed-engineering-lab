"""
state.py -- server-side state of PINNlab (one local user).

The lab always works on a PROJECT from the shared project store (storage/project_store.py): either
the built-in demo or a model sent from BondLab by PINNSIM. Loading a project compiles its equations
(bondgraph/equations.json -> model text -> ODESystem), solves the reference, restores the saved PINN
configuration and measurement-data settings, and draws the transferred bond graph.
"""
import io
import math
import threading

import pandas as pd

from bondgraph import bondlab_bridge as bridge
from bondgraph.equations import DEMO_TEXT, parse_model
from data import dataset_loader as dl
from pinn.trainer import TrainConfig
from storage import project_store as store
from visualization.comparison_plot import metrics

DEFAULT_DATA = {"source": "synthetic", "observables": None, "n_points": 30, "noise_pct": 1.0, "val_fraction": 0.2}


class Lab:
    def __init__(self):
        self.lock = threading.RLock()
        self.project = None
        self.system = None
        self.reference = None
        self.ref_error = None
        self.dataset = dl.empty_dataset()
        self.data_settings = dict(DEFAULT_DATA)
        self.drawing = None
        self.session = None
        self.config = TrainConfig()
        self.default_device = "auto"
        self.csv_df = None
        self.csv_name = None
        self.csv_bytes = None
        self.notice = ""

    # ------------------------------------------------------------------ view
    def view(self):
        """(system, reference, dataset) the dashboard shows: the session's own when one exists."""
        s = self.session
        if s is not None:
            return s.system, s.reference, s.dataset
        return self.system, self.reference, self.dataset

    @property
    def training_alive(self):
        return self.session is not None and self.session.is_alive

    def ensure_loaded(self):
        if self.project is not None:
            return
        p = store.get_active()
        try:
            self.load_project(p.id if p else store.DEMO_ID)
        except Exception:  # noqa: BLE001  -- a broken project must not lock the lab: fall back to the demo
            self.load_project(store.DEMO_ID)

    # --------------------------------------------------------------- projects
    def load_project(self, pid):
        with self.lock:
            if pid == store.DEMO_ID:
                p = store.ensure_demo(DEMO_TEXT)
            else:
                p = store.get(pid)
            if self.training_alive and self.session.project_id != p.id:
                raise RuntimeError(f"a training run on '{self.project.name}' is still active — pause and Reset "
                                   "it before switching project")
            text = p.model_text()
            if not text:
                raise ValueError(f"project '{p.id}' has no equations")
            system = parse_model(text)
            self.project = p
            self._set_system(system)
            self.drawing = self._drawing(p, system)
            cfg = (p.configuration() or {}).get("config")
            self.config = TrainConfig.from_dict(cfg) if cfg else TrainConfig(device=self.default_device)
            self.csv_df = self.csv_name = self.csv_bytes = None
            st = p.dataset_settings()
            if st and st.get("source") == "csv":
                path = p.dataset_csv()
                if path is not None:
                    self.csv_bytes = path.read_bytes()
                    self.csv_df = dl.read_csv_bytes(self.csv_bytes)
                    self.csv_name = path.name
            self.data_settings = dict(DEFAULT_DATA, **(st or {}))
            try:
                self.dataset = self.build_dataset(self.data_settings)
            except dl.DataError as e:
                self.dataset = dl.empty_dataset()
                self.notice = f"Saved measurement data could not be used ({e}); physics only."
            if self.session is not None and not self.session.is_alive and self.session.project_id != p.id:
                self.session = None
            store.set_active(p.id)
            return p

    def _set_system(self, system):
        self.system = system
        try:
            self.reference, self.ref_error = system.solve_reference(), None
        except Exception as e:  # noqa: BLE001
            self.reference, self.ref_error = None, str(e)

    def _drawing(self, p, system):
        if not bridge.available():
            return None
        try:
            if p.source == "bondlab":
                doc = (p.model() or {}).get("document") or {}
                if doc.get("spec"):
                    return bridge.drawing_from_spec(doc["spec"], doc.get("layout"), p.name)
            tid = system.meta.get("bondlab_template") or (p.equations() or {}).get("template_id")
            if tid:
                return bridge.bond_graph_drawing(tid)
        except Exception:  # noqa: BLE001
            return None
        return None

    def compile_text(self, text, note="edited in PINNlab"):
        """Compile edited model text, make it the project's equations (saved to equations.json)."""
        with self.lock:
            system = parse_model(text)
            self._set_system(system)
            if self.project is not None:
                self.project.save_model_text(text, note)
            try:
                self.dataset = self.build_dataset(self.data_settings)
            except dl.DataError:
                self.dataset = dl.empty_dataset()
            return system

    # ------------------------------------------------------------------ data
    def default_observables(self, system=None):
        system = system or self.system
        drawn = [o for w, o in system.visual if w != "gauge"]
        return [drawn[0] if drawn else system.states[0]]

    def build_dataset(self, st):
        system, ref = self.system, self.reference
        src = st.get("source", "synthetic")
        if src == "none" or system is None:
            return dl.empty_dataset()
        if src == "synthetic":
            if ref is None:
                raise dl.DataError("no reference solution to sample from")
            obs = [o for o in (st.get("observables") or self.default_observables()) if o in system.observables]
            return dl.synthetic_from_reference(ref, obs, int(st.get("n_points") or 30),
                                               float(st.get("noise_pct") or 0) / 100.0,
                                               float(st.get("val_fraction") or 0))
        if self.csv_df is None:
            raise dl.DataError("upload a CSV file first")
        return dl.from_dataframe(self.csv_df, st.get("time_column"), st.get("mapping") or {},
                                 float(st.get("val_fraction") or 0), t_range=(system.t_start, system.t_end))

    def apply_dataset(self, st):
        ds = self.build_dataset(st)
        self.dataset, self.data_settings = ds, dict(st)
        if self.project is not None:
            keep = {k: v for k, v in st.items() if k in ("source", "observables", "n_points", "noise_pct",
                                                          "val_fraction", "time_column", "mapping", "file")}
            if st.get("source") == "csv" and self.csv_bytes is not None:
                self.project.save_dataset(keep, self.csv_bytes, self.csv_name)
            else:
                self.project.save_dataset(keep)
        return ds

    def remove_dataset(self):
        self.dataset = dl.empty_dataset()
        self.data_settings = dict(DEFAULT_DATA, source="none")
        self.csv_df = self.csv_name = self.csv_bytes = None
        if self.project is not None:
            self.project.remove_dataset()

    # --------------------------------------------------------------- results
    def results_writer(self, project):
        """Called by the trainer when a run completes: metrics + prediction CSV into project/results/."""
        def write(session):
            ref = session.reference
            t = ref["t"] if ref is not None else session.t_eval.detach().cpu().numpy()
            pred = session.predict(t)
            cols = {"t": t}
            out = {"project": project.id, "epoch": session.epoch, "run": session.run_dir.name if session.run_dir else "",
                   "final_losses": dict(session.last), "best_val_loss": None if math.isinf(session.best_val)
                   else session.best_val, "config": dict(session.config.__dict__), "metrics": {}}
            for o in session.system.observables:
                cols[f"{o}_pinn"] = pred[o]
                if ref is not None:
                    cols[f"{o}_reference"] = ref["observables"][o]
                    out["metrics"][o] = metrics(ref["observables"][o], pred[o])
            buf = io.StringIO()
            pd.DataFrame(cols).to_csv(buf, index=False)
            project.save_results(out, buf.getvalue())
        return write


LAB = Lab()
