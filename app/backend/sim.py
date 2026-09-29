"""
sim.py -- numerical simulation of the derived equations (SciPy), with
time-varying sources and initial conditions given in PHYSICAL units
(initial flow for I elements, initial effort for C elements).
"""
import math
import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from .engine import SpecError, compile_spec
from .library import series_labels

MAX_POINTS = 3000
WALL_LIMIT_S = 10.0


def _f(x, name):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise SpecError(f"'{name}' must be a number.")
    if not math.isfinite(v):
        raise SpecError(f"'{name}' must be finite.")
    return v


def make_waveform(name, w):
    """Waveforms are structured data, never evaluated strings."""
    if not isinstance(w, dict):
        raise SpecError(f"Source '{name}': waveform must be an object.")
    typ = w.get("type", "constant")
    off = _f(w.get("offset", 0.0), f"{name}.offset")
    if typ == "constant":
        val = _f(w.get("value", 0.0), f"{name}.value")
        return lambda t: val
    if typ == "step":
        amp, t0 = _f(w.get("amplitude", 1.0), f"{name}.amplitude"), _f(w.get("t0", 0.0), f"{name}.t0")
        return lambda t: off + (amp if t >= t0 else 0.0)
    if typ == "ramp":
        slope, t0 = _f(w.get("slope", 1.0), f"{name}.slope"), _f(w.get("t0", 0.0), f"{name}.t0")
        return lambda t: off + (slope * (t - t0) if t >= t0 else 0.0)
    if typ == "sine":
        amp, freq = _f(w.get("amplitude", 1.0), f"{name}.amplitude"), _f(w.get("freq", 1.0), f"{name}.freq")
        ph = _f(w.get("phase", 0.0), f"{name}.phase")
        return lambda t: off + amp * math.sin(2 * math.pi * freq * t + ph)
    if typ == "pulse":
        amp, t0 = _f(w.get("amplitude", 1.0), f"{name}.amplitude"), _f(w.get("t0", 0.0), f"{name}.t0")
        width, period = _f(w.get("width", 1.0), f"{name}.width"), _f(w.get("period", 0.0), f"{name}.period")

        def pulse(t):
            if t < t0:
                return off
            tt = (t - t0) % period if period > 0 else (t - t0)
            return off + (amp if tt < width else 0.0)
        return pulse
    if typ == "pwl":
        pts = w.get("points")
        if not isinstance(pts, list) or not (2 <= len(pts) <= 200):
            raise SpecError(f"Source '{name}': piecewise-linear needs 2 to 200 points.")
        xs = [_f(p[0], f"{name}.points.t") for p in pts]
        ys = [_f(p[1], f"{name}.points.value") for p in pts]
        if any(b <= a for a, b in zip(xs, xs[1:])):
            raise SpecError(f"Source '{name}': point times must be strictly increasing.")
        return lambda t: float(np.interp(t, xs, ys))
    raise SpecError(f"Source '{name}': unknown waveform type {typ!r}.")


def simulate(spec, values, sources, initial, t_end, n_points):
    c = compile_spec(spec)
    values = {k: _f(v, k) for k, v in (values or {}).items()}
    sources, initial = sources or {}, initial or {}
    t_end = _f(t_end, "t_end")
    if not (0 < t_end <= 1e7):
        raise SpecError("Simulation time must be between 0 and 1e7.")
    n_points = int(min(max(int(n_points or 500), 20), MAX_POINTS))

    G, t = c.G, c.t
    order = c.state_items
    ys = [sp.Symbol(f"y{i}") for i in range(len(order))]
    tt = sp.Symbol("tt")
    rep = {st: ys[i] for i, (_, st) in enumerate(order)}
    exprs = [c.odes[st].subs(rep) for _, st in order]

    free = set()
    for e in exprs:
        free |= e.free_symbols
    free -= set(ys)
    time_syms = {s for s in free if s.name == "t"}
    free -= time_syms
    free_sorted = sorted(free, key=lambda s: s.name)
    names = [s.name for s in free_sorted]

    source_nodes = {str(d["expr"]): nid for nid, d in G.nodes(data=True) if d["type"] in ("Se", "Sf")}
    getters, missing = [], []
    for nm in names:
        if nm in source_nodes:
            if nm in sources:
                getters.append(make_waveform(nm, sources[nm]))
            elif nm in values:
                v = values[nm]
                getters.append(lambda _t, v=v: v)
            else:
                missing.append(nm)
        elif nm in values:
            v = values[nm]
            getters.append(lambda _t, v=v: v)
        else:
            missing.append(nm)
    if missing:
        raise SpecError("Missing values for: " + ", ".join(sorted(missing)))

    xrep = {s: sp.Symbol(f"a{i}") for i, s in enumerate(free_sorted)}
    xrep.update({s: tt for s in time_syms})
    exprs2 = [e.xreplace(xrep) for e in exprs]
    f = sp.lambdify((tt, *ys, *[xrep[s] for s in free_sorted]), exprs2, modules="numpy")

    y0 = []
    for node, st in order:
        d = G.nodes[node]
        P = values.get(str(d["param_symbol"]))
        if P is None:
            raise SpecError(f"Missing value for parameter '{d['param_symbol']}'.")
        if P <= 0:
            raise SpecError(f"Parameter '{d['param_symbol']}' of '{node}' must be positive.")
        x0 = _f(initial.get(node, 0.0), f"initial.{node}")
        y0.append(P * x0)          # p0 = I*flow0 ; q0 = C*effort0

    deadline = time.monotonic() + WALL_LIMIT_S

    def rhs(tv, y):
        if time.monotonic() > deadline:
            raise TimeoutError("Simulation took too long (limit %.0f s). Try a shorter time span." % WALL_LIMIT_S)
        return np.asarray(f(tv, *y, *[g(tv) for g in getters]), dtype=float)

    t_eval = np.linspace(0.0, t_end, n_points)
    t0 = time.monotonic()
    try:
        sol = solve_ivp(rhs, (0.0, t_end), y0, t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-10,
                        max_step=t_end / 200.0)
    except TimeoutError as e:
        raise SpecError(str(e))
    if not sol.success:
        raise SpecError(f"The solver failed: {sol.message}")
    if not np.all(np.isfinite(sol.y)):
        raise SpecError("The solution diverged (non-finite values). Check parameter values.")

    series = []
    for i, (node, st) in enumerate(order):
        d = G.nodes[node]
        P = values[str(d["param_symbol"])]
        fl, fu, ef, eu, dn, du = series_labels(d["type"], d.get("domain", ""))
        label = d.get("label") or node
        if d["type"] == "I":
            series.append({"id": f"{node}.flow", "node": node, "kind": "flow", "name": f"{fl} ({label})", "unit": fu,
                           "data": (sol.y[i] / P).tolist()})
        else:
            series.append({"id": f"{node}.effort", "node": node, "kind": "effort", "name": f"{ef} ({label})", "unit": eu,
                           "data": (sol.y[i] / P).tolist()})
            series.append({"id": f"{node}.disp", "node": node, "kind": "disp", "name": f"{dn} ({label})", "unit": du,
                           "data": sol.y[i].tolist()})
    return {"ok": True, "t": sol.t.tolist(), "series": series,
            "stats": {"n_states": len(order), "solver_ms": round((time.monotonic() - t0) * 1000, 1), "nfev": int(sol.nfev)}}
