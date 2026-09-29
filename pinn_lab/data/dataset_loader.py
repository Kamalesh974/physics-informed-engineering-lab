"""
dataset_loader.py -- measurement data for the data term of the PINN loss.

Kept separate from the model: this module only produces arrays of
(time, observable value) pairs, a train/validation split, and the scaling
statistics used to non-dimensionalise the problem. It knows nothing about
PyTorch.

Three sources:
  * CSV upload        -- a time column plus one column per measured quantity
  * synthetic         -- points sampled from the reference ODE solution plus
                         Gaussian noise (clearly labelled as such in the UI)
  * none              -- physics-only training (L_data = 0)
"""
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_ROWS = 200_000


class DataError(ValueError):
    pass


@dataclass
class Dataset:
    source: str                       # "csv" | "synthetic" | "none"
    description: str
    t_train: np.ndarray
    y_train: dict                     # observable -> array (NaN = not measured)
    t_val: np.ndarray
    y_val: dict
    info: dict = field(default_factory=dict)

    @property
    def observables(self):
        return list(self.y_train)

    @property
    def n_train(self):
        return int(len(self.t_train))

    @property
    def n_val(self):
        return int(len(self.t_val))

    @property
    def has_data(self):
        return self.n_train > 0 and bool(self.y_train)

    @property
    def has_val(self):
        return self.n_val > 0 and bool(self.y_val)

    def to_dict(self):
        """Primitive-only form (safe for torch.save with weights_only loading)."""
        return {
            "source": self.source, "description": self.description,
            "t_train": [float(x) for x in self.t_train],
            "y_train": {k: [float(x) for x in v] for k, v in self.y_train.items()},
            "t_val": [float(x) for x in self.t_val],
            "y_val": {k: [float(x) for x in v] for k, v in self.y_val.items()},
            "info": {k: v for k, v in self.info.items() if isinstance(v, (str, int, float, bool))},
        }

    @classmethod
    def from_dict(cls, d):
        return cls(source=d["source"], description=d["description"],
                   t_train=np.asarray(d["t_train"], float),
                   y_train={k: np.asarray(v, float) for k, v in d["y_train"].items()},
                   t_val=np.asarray(d["t_val"], float),
                   y_val={k: np.asarray(v, float) for k, v in d["y_val"].items()},
                   info=dict(d.get("info") or {}))


def empty_dataset():
    e = np.zeros(0)
    return Dataset("none", "No measurement data: physics-only training (L_data = 0).", e, {}, e, {})


def _split(n, val_fraction, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * float(val_fraction)))
    if n >= 2:
        n_val = min(max(n_val, 1 if val_fraction > 0 else 0), n - 1)
    else:
        n_val = 0
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def read_csv_bytes(content):
    try:
        df = pd.read_csv(io.BytesIO(content), nrows=MAX_ROWS + 1)
    except Exception as e:  # noqa: BLE001
        raise DataError(f"could not read CSV: {e}") from e
    if len(df) > MAX_ROWS:
        raise DataError(f"CSV has more than {MAX_ROWS} rows")
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        raise DataError("CSV needs at least two numeric columns (time + one measured quantity)")
    return num


def from_dataframe(df, time_col, mapping, val_fraction=0.2, seed=0, t_range=None):
    """mapping: observable -> column name (None/'' = not measured)."""
    mapping = {o: c for o, c in (mapping or {}).items() if c}
    if time_col not in df.columns:
        raise DataError(f"time column '{time_col}' not found")
    if not mapping:
        raise DataError("map at least one CSV column to a state or output")
    missing = [c for c in mapping.values() if c not in df.columns]
    if missing:
        raise DataError(f"columns not found: {', '.join(missing)}")
    d = df[[time_col] + sorted(set(mapping.values()))].copy()
    d = d[np.isfinite(d[time_col])]
    if t_range is not None:
        lo, hi = t_range
        outside = int(((d[time_col] < lo) | (d[time_col] > hi)).sum())
        d = d[(d[time_col] >= lo) & (d[time_col] <= hi)]
    else:
        outside = 0
    if len(d) < 2:
        raise DataError("fewer than 2 usable rows inside the model's time range")
    t = d[time_col].to_numpy(float)
    ys = {o: d[c].to_numpy(float) for o, c in mapping.items()}
    tr, va = _split(len(t), val_fraction, seed)
    desc = (f"CSV: {len(t)} rows, time column '{time_col}', "
            + ", ".join(f"{c} -> {o}" for o, c in mapping.items()))
    if outside:
        desc += f" ({outside} rows outside the time range were dropped)"
    return Dataset("csv", desc, t[tr], {o: y[tr] for o, y in ys.items()}, t[va], {o: y[va] for o, y in ys.items()},
                   info={"rows": int(len(t)), "time_column": time_col})


def synthetic_from_reference(ref, observables, n_points=30, noise_frac=0.01, val_fraction=0.2, seed=0):
    """Sample 'measurements' from the reference ODE solution (+ Gaussian noise).

    noise_frac is relative to each observable's standard deviation over the
    time window. This is the honest stand-in for experimental data in demo mode."""
    if not observables:
        return empty_dataset()
    rng = np.random.default_rng(seed)
    t_ref = ref["t"]
    n = int(max(2, n_points))
    t = np.sort(rng.uniform(t_ref[0], t_ref[-1], n))
    ys = {}
    for o in observables:
        clean = np.interp(t, t_ref, ref["observables"][o])
        sd = float(np.std(ref["observables"][o])) or 1.0
        ys[o] = clean + rng.normal(0.0, noise_frac * sd, size=n)
    tr, va = _split(n, val_fraction, seed + 1)
    desc = (f"Synthetic measurements: {n} random times sampled from the reference ODE solution, "
            f"Gaussian noise {100 * noise_frac:g}% of each signal's std; measured: {', '.join(observables)}")
    return Dataset("synthetic", desc, t[tr], {o: y[tr] for o, y in ys.items()}, t[va], {o: y[va] for o, y in ys.items()},
                   info={"noise_frac": float(noise_frac), "n_points": n})


def reference_to_csv(ref):
    cols = {"t": ref["t"]}
    cols.update(ref["observables"])
    return pd.DataFrame(cols).to_csv(index=False)


# --------------------------------------------------------------------------
# Scaling (non-dimensionalisation)
# --------------------------------------------------------------------------
def _stats(arrays, mode):
    a = np.concatenate([np.asarray(x, float).ravel() for x in arrays if len(x)]) if arrays else np.zeros(0)
    a = a[np.isfinite(a)]
    if mode == "none" or a.size == 0:
        return 0.0, 1.0
    if mode == "minmax":
        lo, hi = float(a.min()), float(a.max())
        mu, sd = 0.5 * (lo + hi), 0.5 * (hi - lo)
    else:  # standard
        mu, sd = float(a.mean()), float(a.std())
    floor = max(1e-3 * float(np.max(np.abs(a))), 1e-12)
    return mu, max(sd, floor)


def compute_scaling(system, ref, dataset, mode="standard"):
    """Per-state (mu, sigma) for the network's output layer and a per-observable
    scale for the data/BC loss terms. Uses the reference solution's range when
    available, otherwise the measurement data, otherwise 1."""
    mu, sigma, obs_scale = [], [], {}
    for s in system.states:
        arrays = []
        if ref is not None:
            arrays.append(ref["states"][s])
        elif dataset is not None and s in dataset.y_train:
            arrays.append(dataset.y_train[s])
        arrays.append([system.initial.get(s, 0.0)])
        m, sd = _stats(arrays, mode)
        mu.append(m)
        sigma.append(sd)
    for o in system.observables:
        arrays = []
        if ref is not None:
            arrays.append(ref["observables"][o])
        if dataset is not None and o in dataset.y_train:
            arrays.append(dataset.y_train[o])
        _, sd = _stats(arrays, mode)
        obs_scale[o] = sd
    return {"mode": mode, "mu": mu, "sigma": sigma, "obs_scale": obs_scale,
            "source": "reference solution" if ref is not None else "measurement data"}
