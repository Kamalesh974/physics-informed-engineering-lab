"""
bondlab_bridge.py -- connects the PINN lab to the BondLab engine in this
repository (app/backend: SCAP causality + SymPy equation derivation).

It turns a BondLab project (bond graph + parameter values + source
waveforms + initial conditions) into the model text understood by
`equations.parse_model`, so the equations the PINN trains on are exactly the
equations BondLab derived. If the BondLab code is not importable (e.g. the
pinn_lab folder was copied elsewhere) everything here degrades gracefully and
the lab still works with typed/pasted equations.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_engine = _library = None
_import_error = None


def _load():
    global _engine, _library, _import_error
    if _engine is not None or _import_error is not None:
        return _engine is not None
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from app.backend import engine, library  # noqa: WPS433
        _engine, _library = engine, library
    except Exception as e:  # noqa: BLE001
        _import_error = f"{type(e).__name__}: {e}"
    return _engine is not None


def available():
    return _load()


def import_error():
    _load()
    return _import_error


def list_templates():
    if not _load():
        return []
    return [(t["id"], t["name"]) for t in _library.list_templates()]


def get_template(tid):
    if not _load():
        raise RuntimeError("BondLab engine not available: " + str(_import_error))
    return _library.get_template(tid)


def derive(spec):
    """Run BondLab's causality + equation derivation on a bond-graph spec."""
    if not _load():
        raise RuntimeError("BondLab engine not available: " + str(_import_error))
    return _engine.derive_payload(spec)


def _num(v):
    return format(float(v), ".10g")


def waveform_expr(w):
    """BondLab source waveform (structured data) -> expression of t."""
    typ = w.get("type", "constant")
    off = float(w.get("offset", 0.0))
    pre = f"{_num(off)} + " if off else ""
    if typ == "constant":
        return _num(float(w.get("value", 0.0)) + off)
    if typ == "step":
        return f"{pre}{_num(w.get('amplitude', 1.0))}*Heaviside(t - {_num(w.get('t0', 0.0))})"
    if typ == "ramp":
        t0 = _num(w.get("t0", 0.0))
        return f"{pre}{_num(w.get('slope', 1.0))}*(t - {t0})*Heaviside(t - {t0})"
    if typ == "sine":
        return (f"{pre}{_num(w.get('amplitude', 1.0))}*sin(2*pi*{_num(w.get('freq', 1.0))}*t"
                f" + {_num(w.get('phase', 0.0))})")
    if typ == "pulse":
        amp, t0 = _num(w.get("amplitude", 1.0)), _num(w.get("t0", 0.0))
        width, period = _num(w.get("width", 1.0)), float(w.get("period", 0.0))
        if period > 0:
            return f"{pre}{amp}*Heaviside(t - {t0})*(1 - Heaviside(Mod(t - {t0}, {_num(period)}) - {width}))"
        return f"{pre}{amp}*Heaviside(t - {t0})*(1 - Heaviside(t - {t0} - {width}))"
    if typ == "pwl":
        pts = [(float(a), float(b)) for a, b in w["points"]]
        terms = [_num(pts[0][1] + off)]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            s = (y1 - y0) / (x1 - x0)
            if s:
                terms.append(f"{_num(s)}*(Min(Max(t, {_num(x0)}), {_num(x1)}) - {_num(x0)})")
        return " + ".join(terms)
    raise ValueError(f"unknown waveform type {typ!r}")


def template_to_model_text(tid):
    """Derive a BondLab template's equations and return PINN-lab model text."""
    tpl = get_template(tid)
    return project_to_model_text(tpl)


def project_to_model_text(project):
    return equations_to_model_text(project_to_equations(project))


def _clean(text):
    return " ".join(str(text or "").replace("#", "").split())


def _unit(unit):
    return f" [{unit}]" if unit and unit != "-" else ""


def project_to_equations(project):
    """BondLab project document -> structured equations record (what equations.json stores).

    The equations are (re-)derived here by the BondLab engine from the bond graph itself, so what the
    PINN receives is always consistent with the graph, never a stale copy typed by hand."""
    spec = project["spec"]
    values = dict(project.get("values") or {})
    sources = dict(project.get("sources") or {})
    initial = dict(project.get("initial") or {})
    meta = project.get("meta") or {}
    sim = project.get("sim") or {}
    p = derive(spec)
    if not p.get("ok", True):
        raise ValueError(p.get("error") or "BondLab could not derive equations for this graph")
    nodes = {n["id"]: n for n in spec["nodes"]}
    t_end = float(sim.get("t_end", 10.0))
    if not (t_end > 0):
        raise ValueError("the simulation time t_end must be positive")

    rec = {
        "format": "bondlab-equations/1",
        "title": _clean(meta.get("name")) or "BondLab model",
        "description": _clean(meta.get("description")),
        "template_id": meta.get("id") or None,
        "derived_by": "BondLab engine: SCAP causality assignment + SymPy equation derivation",
        "states": [], "equations": [], "parameters": [], "constants": [], "inputs": [],
        "initial_conditions": [], "boundary_conditions": [], "outputs": [], "visual": [],
        "generalized": [{"text": e["text"], "latex": e["latex"]} for e in p.get("generalized", [])],
        "causality": p.get("causality", []),
        "unused_parameters": p.get("unused_parameters", []),
        "time": {"t_start": 0.0, "t_end": t_end},
    }
    missing = sorted({q["name"] for q in p["parameters"] if q["name"] not in values})
    if missing:
        raise ValueError("missing parameter values: " + ", ".join(missing))
    for prm in p["parameters"]:
        nm = prm["name"]
        if nm not in {q["name"] for q in rec["parameters"]}:
            rec["parameters"].append({"name": nm, "value": float(values[nm]), "unit": prm.get("unit") or "",
                                      "label": _clean(prm.get("label")) or prm["type"], "element": prm["type"],
                                      "node": prm.get("node")})
    for c in p.get("constants", []):
        nm = c["name"]
        if nm not in values:
            raise ValueError(f"missing value for the constant '{nm}'")
        rec["constants"].append({"name": nm, "value": float(values[nm])})
    for src in p["sources"]:
        nm = src["name"]
        wf = sources.get(nm)
        if wf is not None:
            expr = waveform_expr(wf)
        elif nm in values:
            expr, wf = _num(values[nm]), {"type": "constant", "value": float(values[nm])}
        else:
            expr, wf = "0", {"type": "constant", "value": 0.0}
        snode = nodes.get(src.get("node"), {})
        _fl, fu, _ef, eu, _dn, _du = _library.series_labels(src["type"].lstrip("M"), snode.get("domain", ""))
        unit = eu if src["type"].endswith("e") else fu      # Se/MSe set an effort, Sf/MSf a flow
        rec["inputs"].append({"name": nm, "expression": expr, "waveform": wf, "unit": unit if unit != "-" else "",
                              "label": _clean(src.get("label")) or src["type"], "element": src["type"],
                              "node": src.get("node")})

    # physical state variables: I -> flow f_<id>, C -> stored quantity q_<id>
    for phys in p["physical"]:
        nid, var = phys["node"], phys["variable"]
        nd = nodes[nid]
        dom, typ = nd.get("domain", ""), nd["type"]
        fl, fu, ef, eu, dn, du = _library.series_labels(typ, dom)
        label = _clean(nd.get("label")) or nid
        x0 = float(initial.get(nid, 0.0))
        rhs = phys["text"].split("=", 1)[1].strip()
        rec["equations"].append({"state": var, "text": phys["text"], "latex": phys["latex"], "rhs": rhs})
        if typ == "I":
            rec["states"].append({"name": var, "node": nid, "element": "I", "domain": dom, "unit": fu,
                                  "label": f"{fl} ({label})", "quantity": fl})
            rec["initial_conditions"].append({"state": var, "value": x0, "unit": fu,
                                              "physical": {"node": nid, "quantity": fl, "unit": fu, "value": x0}})
            rec["visual"].append(("rotor" if dom == "mech_rot" else "gauge", var))
        else:
            cpar = nd["param_symbol_name"]
            rec["states"].append({"name": var, "node": nid, "element": "C", "domain": dom, "unit": du,
                                  "label": f"{dn} ({label})", "quantity": dn})
            # BondLab's initial value for a C element is its EFFORT; the state is the stored quantity q = C*e
            rec["initial_conditions"].append({"state": var, "value": float(values.get(cpar, 1.0)) * x0, "unit": du,
                                              "physical": {"node": nid, "quantity": ef, "unit": eu, "value": x0}})
            eff = f"e_{nid}"
            rec["outputs"].append({"name": eff, "expression": f"{var}/{cpar}", "unit": eu,
                                   "label": f"{ef} ({label})"})
            widget = {"mech_trans": "mass_spring", "thermal": "thermal", "hydraulic": "tank"}.get(dom, "gauge")
            rec["visual"].append((widget, eff if widget in ("thermal", "gauge") else var))

    names = {x["name"] for x in rec["states"]} | {o["name"] for o in rec["outputs"]}
    for b in project.get("boundary") or []:
        var, tb, val = b.get("variable"), float(b.get("t")), float(b.get("value"))
        if var not in names:
            raise ValueError(f"boundary condition on unknown variable '{var}' "
                             f"(choose one of {', '.join(sorted(names))})")
        if not (0.0 <= tb <= t_end):
            raise ValueError(f"boundary condition time {tb} is outside [0, {t_end}]")
        unit = next((x["unit"] for x in rec["states"] + rec["outputs"] if x["name"] == var), "")
        rec["boundary_conditions"].append({"variable": var, "t": tb, "value": val, "unit": unit})
    rec["visual"] = [{"widget": w, "observable": o} for w, o in _pick_visual(rec["visual"])]
    return rec


def equations_to_model_text(rec):
    """Structured equations record -> the text format parsed by equations.parse_model."""
    lines = ["[meta]", f"title = {rec['title']} (BondLab)"]
    if rec.get("description"):
        lines.append(f"description = {rec['description']}")
    if rec.get("template_id"):
        lines.append(f"bondlab_template = {rec['template_id']}")
    lines += ["", "[equations]", "# Derived automatically by BondLab from the bond graph (causality -> SymPy):"]
    lines += [e["text"] for e in rec["equations"]]
    lines += ["", "[parameters]"]
    lines += [f"{q['name']}{_unit(q['unit'])} = {_num(q['value'])}  # {q['label']}" for q in rec["parameters"]]
    lines += [f"{c['name']} = {_num(c['value'])}  # constant in a modulation formula" for c in rec["constants"]]
    lines += ["", "[inputs]"]
    lines += [f"{i['name']}{_unit(i['unit'])} = {i['expression']}  # {i['label']} ({i['element']})"
              for i in rec["inputs"]]
    labels = {x["name"]: x["label"] for x in rec["states"]}
    lines += ["", "[initial]"]
    lines += [f"{ic['state']}{_unit(ic['unit'])} = {_num(ic['value'])}  # {labels.get(ic['state'], '')}"
              for ic in rec["initial_conditions"]]
    if rec["outputs"]:
        lines += ["", "[outputs]"]
        lines += [f"{o['name']}{_unit(o['unit'])} = {o['expression']}  # {o['label']}" for o in rec["outputs"]]
    if rec["boundary_conditions"]:
        lines += ["", "[boundary]"] + [f"{b['variable']}({_num(b['t'])}) = {_num(b['value'])}"
                                       for b in rec["boundary_conditions"]]
    lines += ["", "[visual]"] + [f"{v['widget']} = {v['observable']}" for v in rec["visual"]]
    lines += ["", "[time]", f"t_start = {_num(rec['time']['t_start'])}", f"t_end = {_num(rec['time']['t_end'])}", ""]
    return "\n".join(lines)


def _pick_visual(items, limit=4):
    """Prefer physical widgets over plain gauges; at most `limit`."""
    rank = {"mass_spring": 0, "rotor": 0, "thermal": 0, "tank": 0, "gauge": 1}
    items = sorted(items, key=lambda it: rank[it[0]])
    return items[:limit]


def bond_graph_drawing(tid):
    """Nodes, bonds, layout and causality for drawing a template's bond graph."""
    tpl = get_template(tid)
    return drawing_from_spec(tpl["spec"], tpl.get("layout"), tpl["meta"]["name"])


def drawing_from_spec(spec, layout=None, name=""):
    """Nodes, bonds, layout and causality for drawing ANY bond graph (e.g. one sent by PINNSIM)."""
    p = derive(spec)
    giver = {(c["from"], c["to"]): c.get("effort_giver") for c in p.get("causality", [])}
    return {
        "name": name,
        "nodes": spec["nodes"],
        "edges": [dict(e, effort_giver=giver.get((e["from"], e["to"]))) for e in spec["edges"]],
        "layout": layout or {},
    }
