"""
engine.py -- bridge between the web API and the shared engine that lives in
the project root (scap_causality.py, equation_gen.py, image_to_bond_graph.py,
param_usage_check.py, formula_safety.py). Nothing here re-implements physics:
it validates input, calls the root engine, and shapes the result for the UI.
"""
import hashlib
import json
import os
import re
import sys
import threading
from collections import OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sympy as sp  # noqa: E402

from equation_gen import derive_odes_generic  # noqa: E402
from formula_safety import UnsafeFormula  # noqa: E402
from image_to_bond_graph import build_graph_from_spec  # noqa: E402
from param_usage_check import find_unused_params  # noqa: E402

from .library import param_unit, series_labels  # noqa: E402

NODE_TYPES = {"0", "1", "R", "C", "I", "GY", "TF", "Se", "Sf", "MSe", "MSf"}
ONE_PORT = {"R", "C", "I", "Se", "Sf", "MSe", "MSf"}
IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
MAX_NODES, MAX_EDGES = 120, 240


class SpecError(ValueError):
    """The graph the user/AI supplied is not a valid bond graph (message is user-facing)."""


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_spec(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("nodes"), list) or not isinstance(spec.get("edges"), list):
        raise SpecError("Graph must contain a 'nodes' list and an 'edges' list.")
    nodes, edges = spec["nodes"], spec["edges"]
    if not nodes:
        raise SpecError("The graph is empty. Add some elements first.")
    if len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        raise SpecError(f"Graph too large (max {MAX_NODES} elements and {MAX_EDGES} bonds).")

    by_id = {}
    for n in nodes:
        if not isinstance(n, dict):
            raise SpecError("Every node must be an object.")
        nid, ntype = n.get("id"), n.get("type")
        if not isinstance(nid, str) or not IDENT.match(nid):
            raise SpecError(f"Invalid element id {nid!r}: use letters, digits and underscore, starting with a letter.")
        if nid in by_id:
            raise SpecError(f"Duplicate element id '{nid}'.")
        if ntype not in NODE_TYPES:
            raise SpecError(f"Element '{nid}' has unknown type {ntype!r}.")
        if ntype in ("R", "C", "I", "GY", "TF"):
            p = n.get("param_symbol_name")
            if not isinstance(p, str) or not IDENT.match(p):
                raise SpecError(f"Element '{nid}' ({ntype}) needs a parameter name (letters/digits/underscore).")
        if ntype in ("Se", "Sf"):
            p = n.get("expr_param_name")
            if not isinstance(p, str) or not IDENT.match(p):
                raise SpecError(f"Source '{nid}' needs a signal name (letters/digits/underscore).")
        if ntype in ("MSe", "MSf"):
            if not isinstance(n.get("modulation_formula"), str) or not n["modulation_formula"].strip():
                raise SpecError(f"Modulated source '{nid}' needs a formula.")
        by_id[nid] = n

    for nid, n in by_id.items():
        if n["type"] in ("MSe", "MSf"):
            src = n.get("signal_from")
            if src not in by_id or by_id[src]["type"] not in ("0", "1"):
                raise SpecError(f"Modulated source '{nid}' must read its signal from a junction (0 or 1).")
            if n.get("signal_kind", "effort") not in ("effort", "flow"):
                raise SpecError(f"Modulated source '{nid}': signal kind must be 'effort' or 'flow'.")

    seen = set()
    power = {nid: [] for nid in by_id}
    for e in edges:
        if not isinstance(e, dict):
            raise SpecError("Every bond must be an object.")
        a, b, sig = e.get("from"), e.get("to"), bool(e.get("signal", False))
        if a not in by_id or b not in by_id:
            raise SpecError(f"A bond refers to a missing element ({a!r} -> {b!r}).")
        if a == b:
            raise SpecError(f"Element '{a}' cannot be bonded to itself.")
        key = (frozenset((a, b)), sig)
        if key in seen:
            raise SpecError(f"Duplicate bond between '{a}' and '{b}'.")
        seen.add(key)
        if not sig:
            power[a].append(b)
            power[b].append(a)

    for nid, n in by_id.items():
        t, nb = n["type"], power[nid]
        if t in ONE_PORT and len(nb) != 1:
            raise SpecError(f"{t} element '{nid}' must have exactly 1 bond (it has {len(nb)}).")
        if t in ONE_PORT and by_id[nb[0]]["type"] not in ("0", "1"):
            raise SpecError(f"{t} element '{nid}' must connect to a junction (0 or 1), not directly to '{nb[0]}'.")
        if t in ("GY", "TF"):
            if len(nb) != 2:
                raise SpecError(f"{t} '{nid}' is a 2-port: it needs exactly 2 bonds (it has {len(nb)}).")
            if any(by_id[x]["type"] not in ("0", "1") for x in nb):
                raise SpecError(f"{t} '{nid}' must connect to junctions on both sides.")
        if t in ("0", "1") and len(nb) < 2:
            raise SpecError(f"Junction '{nid}' needs at least 2 bonds.")


_HINTS = (
    ("over-determined",
     "Two or more branches at this junction each try to fix its common effort/flow (for example a source and a "
     "capacitor on the same 0-junction). Put a resistor, on its own 1-junction, between them."),
    ("derivative causality",
     "Two energy-storage elements are locked together with no independent freedom (derivative causality). "
     "Add a resistor or compliance between them."),
    ("asked to give",
     "Two energy-storage elements are locked together with no independent freedom (derivative causality). "
     "Add a resistor or compliance between them."),
    ("causal loop",
     "Simultaneous equations are needed here (e.g. two resistors sharing a junction). Merge them into one resistor."),
    ("unresolved", "Some bonds could not be given a cause/effect direction. Check every junction has a valid mix of elements."),
)


def _friendly(msg):
    for key, hint in _HINTS:
        if key in msg.lower():
            return f"{msg}\n\nHint: {hint}"
    return msg


# ---------------------------------------------------------------------------
# compile (cached)
# ---------------------------------------------------------------------------
class Compiled:
    pass


_CACHE = OrderedDict()
_LOCK = threading.Lock()
_CACHE_MAX = 64


def spec_hash(spec):
    core = {"nodes": spec["nodes"], "edges": spec["edges"]}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


def compile_spec(spec):
    """validate -> build graph -> causality -> ODEs. Raises SpecError/UnsafeFormula/RuntimeError."""
    validate_spec(spec)
    key = spec_hash(spec)
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]

    G, symtab, t = build_graph_from_spec(spec)
    state_symbols, odes = derive_odes_generic(G)
    c = Compiled()
    c.key, c.G, c.symtab, c.t = key, G, symtab, t
    c.state_items = list(state_symbols.items())          # [(node_id, state Function)]
    c.odes = odes
    c.unused = find_unused_params(odes, symtab.keys())
    c.spec = spec

    with _LOCK:
        _CACHE[key] = c
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return c


# ---------------------------------------------------------------------------
# physical form + payload
# ---------------------------------------------------------------------------
def physical_form(c):
    """Rewrite generalized (p, q) equations in familiar variables:
    I element: flow f_<id> = p / param  (so  param * df/dt = effort);
    C element: displacement q_<id> is kept (it already is charge/position/volume)."""
    t = c.t
    subs, names = {}, {}
    for node, st in c.state_items:
        d = c.G.nodes[node]
        if d["type"] == "I":
            f = sp.Function(f"f_{node}")(t)
            subs[st] = d["param_symbol"] * f
            names[node] = f
        else:
            names[node] = st
    out = {}
    for node, st in c.state_items:
        d = c.G.nodes[node]
        rhs = c.odes[st].subs(subs)
        if d["type"] == "I":
            rhs = sp.simplify(rhs / d["param_symbol"])
        out[node] = (names[node], sp.simplify(rhs))
    return out


def _latex_eq(func, rhs, t):
    try:
        return sp.latex(sp.Eq(sp.Derivative(func, t), rhs))
    except Exception:  # noqa: BLE001
        return ""


def derive_payload(spec):
    try:
        c = compile_spec(spec)
    except SpecError as e:
        return {"ok": False, "kind": "spec", "error": str(e)}
    except UnsafeFormula as e:
        return {"ok": False, "kind": "formula", "error": f"Formula rejected: {e}"}
    except RuntimeError as e:
        return {"ok": False, "kind": "causality", "error": _friendly(str(e))}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "kind": "internal", "error": f"Could not derive equations ({type(e).__name__})."}

    G, t = c.G, c.t
    causality = []
    for u, v, d in G.edges(data=True):
        causality.append({"bond": d["bond_id"], "from": u, "to": v, "signal": bool(d["signal"]),
                          "effort_giver": d.get("effort_giver")})

    generalized, physical = [], []
    phys = physical_form(c)
    for node, st in c.state_items:
        generalized.append({"node": node, "state": str(st), "kind": "momentum p" if G.nodes[node]["type"] == "I" else "displacement q",
                            "text": f"d({st})/dt = {c.odes[st]}", "latex": _latex_eq(st, c.odes[st], t)})
        func, rhs = phys[node]
        physical.append({"node": node, "variable": str(func.func), "text": f"d({func})/dt = {rhs}", "latex": _latex_eq(func, rhs, t)})

    params, sources, seen = [], [], set()
    for nid, d in G.nodes(data=True):
        if d.get("param_key") and d["type"] in ("R", "C", "I", "GY", "TF"):
            if d["param_key"] in seen:
                continue
            seen.add(d["param_key"])
            params.append({"name": d["param_key"], "type": d["type"], "node": nid, "domain": d.get("domain", ""),
                           "unit": param_unit(d["type"], d.get("domain", "")), "label": d.get("label", "")})
        if d["type"] in ("Se", "Sf"):
            sources.append({"name": str(d["expr"]), "type": d["type"], "node": nid, "domain": d.get("domain", ""), "label": d.get("label", "")})
    src_names = {s["name"] for s in sources}
    constants = [{"name": n} for n in sorted(c.symtab) if n not in seen and n not in src_names and not n.startswith("SIGNAL")]

    initials = []
    for node, st in c.state_items:
        d = G.nodes[node]
        fl, fu, ef, eu, dn, du = series_labels(d["type"], d.get("domain", ""))
        initials.append({"node": node, "type": d["type"], "label": d.get("label", node),
                         "quantity": fl if d["type"] == "I" else ef, "unit": fu if d["type"] == "I" else eu})

    return {"ok": True, "hash": c.key, "n_states": len(c.state_items), "causality": causality,
            "generalized": generalized, "physical": physical, "parameters": params, "sources": sources,
            "constants": constants, "initials": initials,
            "unused_parameters": sorted(c.unused["unused"]),
            "notes": ["Thermal domain uses the pseudo bond graph convention (temperature and heat flow)."]
            if any(str(n.get("domain", "")).lower().startswith("therm") for n in spec["nodes"]) else []}
