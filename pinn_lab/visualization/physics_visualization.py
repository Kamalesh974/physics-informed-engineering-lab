"""
physics_visualization.py -- data for the animated physical view.

The animation itself runs in the browser (ui/assets/physics_anim.js, a
60 fps canvas renderer), so playback is smooth and never waits for the
server. This module builds the only data it is allowed to use, all taken
from the trained network:

    pred    the PINN's prediction of the widget quantity over time
    rate    its time derivative, from torch.autograd on the network
            (net heat flow C*dT/dt, velocity dx/dt, net inflow dV/dt ...)
    driver  the input signal acting on it (source temperature, force ...),
            evaluated from the model's [inputs]
    ref     the reference ODE solution, drawn only as a dashed "ghost"

Widgets (chosen in the model's [visual] section):
    mass_spring  mass on a spring + damper, force and velocity arrows
    rotor        rotating disc; angle = integral of the predicted angular velocity
    thermal      heat reservoir -> conduction rod -> body, heat-flow particles, thermometer
    tank         tank level, inflow/outflow stream sized by the predicted net flow
    gauge        analog dial for any state or output
"""
import numpy as np

WIDGET_TITLES = {"mass_spring": "Mass–spring–damper", "rotor": "Rotor", "thermal": "Thermal body",
                 "tank": "Tank", "gauge": "Gauge"}
MAX_WIDGETS = 4


def widgets_for(system, limit=MAX_WIDGETS):
    items = list(system.visual) or [("gauge", o) for o in system.states]
    return items[:limit]


def _rate_source(system, obs):
    """Which quantity's derivative to show: for an effort readout e_X = q_X/C use the stored quantity
    q_X (its derivative is the net flow into the element: heat flow, volume flow ...)."""
    if obs.startswith("e_") and ("q_" + obs[2:]) in system.states:
        return "q_" + obs[2:]
    return obs


def _driver(system, kind, obs):
    """The input signal acting on this widget, if one can be identified from units."""
    unit = system.unit(obs)
    wanted = {"thermal": [unit], "mass_spring": ["N"]}.get(kind)
    for name in system.inputs:
        u = system.unit(name)
        if wanted is None or (u and u in wanted):
            return name
    return None


def _num(a):
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else float(f"{v:.6g}") for v in a]


def _range(*arrays):
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays if a is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12 * max(1.0, abs(hi)):
        pad = max(abs(hi) * 0.05, 1e-9)
        lo, hi = lo - pad, hi + pad
    return lo, hi


def animation_payload(system, pred, ref):
    if system is None:
        return {"ready": False, "message": "Compile the equations first."}
    if pred is None:
        return {"ready": False, "message": "The animation is driven by the PINN prediction — start training "
                                           "or load a checkpoint."}
    t = np.asarray(pred["t"], float)
    widgets = []
    for kind, obs in widgets_for(system):
        p = np.asarray(pred["observables"][obs], float)
        r = np.interp(t, ref["t"], ref["observables"][obs]) if ref is not None else None
        rsrc = _rate_source(system, obs)
        rates = pred.get("rates") or {}
        rate = np.asarray(rates[rsrc] if rsrc in rates else np.gradient(p, t), float)
        drv = _driver(system, kind, obs)
        dvals = system.input_numpy(drv, t) if drv else None
        lo, hi = _range(p, r, dvals if kind == "thermal" else None)
        runit = system.unit(rsrc)
        w = {
            "kind": kind, "obs": obs, "title": WIDGET_TITLES[kind], "label": system.label(obs),
            "unit": system.unit(obs), "pred": _num(p), "ref": _num(r) if r is not None else None,
            "lo": lo, "hi": hi,
            "rate": _num(rate), "rate_name": rsrc, "rate_unit": f"{runit}/s" if runit else "1/s",
            "rate_max": float(np.nanmax(np.abs(rate))) if rate.size else 0.0,
        }
        if drv:
            w["driver"] = {"name": drv, "label": system.labels.get(drv, drv), "unit": system.unit(drv),
                           "values": _num(dvals), "max": float(np.nanmax(np.abs(dvals))) if dvals.size else 0.0}
        widgets.append(w)
    return {"ready": True, "title": system.title, "epoch": int(pred["epoch"]), "t": _num(t),
            "t0": float(system.t_start), "t1": float(system.t_end), "widgets": widgets,
            "has_ref": ref is not None}
