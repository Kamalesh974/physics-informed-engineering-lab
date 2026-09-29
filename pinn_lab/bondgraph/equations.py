"""
equations.py -- the governing-equation model that the PINN trains on.

A model is plain text, organised in sections, so equations produced by
BondLab (or typed by hand) can be pasted in unchanged:

    [meta]
    title = Demo Physics Problem - mass-spring-damper
    bondlab_template = mass_spring_damper      # optional: draws the bond graph

    [equations]                                # one ODE per state, BondLab format
    d(f_M1(t))/dt = (C_k*(F - c*f_M1(t)) - q_K1(t))/(C_k*m)
    d(q_K1(t))/dt = f_M1(t)

    [parameters]                               # name [unit] = number  # label
    m [kg] = 1.0  # mass

    [inputs]                                   # sources, may depend on t
    F [N] = 1.0*Heaviside(t)

    [initial]                                  # state [unit] = value at t_start
    f_M1 [m/s] = 0  # velocity of the mass

    [outputs]                                  # optional readouts of the states
    F_spring [N] = q_K1/C_k  # spring force

    [boundary]                                 # optional point constraints
    q_K1(3.0) = 0.01

    [visual]                                   # optional: widget = observable
    mass_spring = q_K1

    [time]
    t_start = 0
    t_end = 3

Each equation d(x_i)/dt = f_i(x, t) becomes the residual
    r_i(t) = d(x_i)/dt - f_i(x(t), t)
which is evaluated with PyTorch autograd on the PINN's output and drives
L_physics. Nothing in this file is display-only.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from .parser import RESERVED, ParseError, compile_expr, parse_expression

SECTIONS = ("meta", "equations", "parameters", "inputs", "initial", "outputs", "boundary", "visual", "time")
WIDGETS = ("mass_spring", "rotor", "thermal", "tank", "gauge")

_LHS = re.compile(r"^\s*d\s*\(?\s*([A-Za-z_]\w*)\s*(?:\(\s*t\s*\))?\s*\)?\s*/\s*dt\s*$")
_DEF = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*=\s*(.+?)\s*$")
_BC = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*([^)]+)\)\s*=\s*(.+?)\s*$")
_IDENT = re.compile(r"^[A-Za-z_]\w*$")


class ModelError(ValueError):
    pass


@dataclass
class ODESystem:
    title: str
    states: list
    rhs: dict                      # state -> sympy expr (params/inputs still symbolic)
    params: dict                   # name -> float
    inputs: dict                   # name -> sympy expr in t
    initial: dict                  # state -> float
    outputs: dict                  # name -> sympy expr of states/params/inputs/t
    boundary: list                 # [(observable, t_b, value)]
    visual: list                   # [(widget, observable)]
    t_start: float
    t_end: float
    units: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    source_text: str = ""

    def __post_init__(self):
        self.t = sp.Symbol("t")
        self._compile()

    # ------------------------------------------------------------------ build
    def _substituted(self, expr):
        """Replace inputs by their t-expressions and parameters by numbers."""
        sub_inputs = {sp.Symbol(k): v for k, v in self.inputs.items()}
        e = expr.xreplace(sub_inputs)
        return e.xreplace({sp.Symbol(k): sp.Float(v) for k, v in self.params.items()})

    def _compile(self):
        self.rhs_numeric = {s: self._substituted(self.rhs[s]) for s in self.states}
        self.outputs_numeric = {o: self._substituted(e) for o, e in self.outputs.items()}
        self._rhs_torch = [compile_expr(self.rhs_numeric[s], "torch") for s in self.states]
        self._rhs_numpy = [compile_expr(self.rhs_numeric[s], "numpy") for s in self.states]
        self._out_torch = {o: compile_expr(e, "torch") for o, e in self.outputs_numeric.items()}
        self._out_numpy = {o: compile_expr(e, "numpy") for o, e in self.outputs_numeric.items()}
        self._in_numpy = {k: compile_expr(self._substituted(e), "numpy") for k, e in self.inputs.items()}

    # ----------------------------------------------------------- properties
    @property
    def observables(self):
        return list(self.states) + list(self.outputs)

    def unit(self, name):
        return self.units.get(name, "")

    def label(self, name):
        lab, u = self.labels.get(name) or name, self.unit(name)
        return f"{lab} [{u}]" if u else lab

    # ------------------------------------------------------------ evaluation
    def _env(self, t, xs):
        env = {"t": t}
        env.update(zip(self.states, xs))
        return env

    def rhs_torch(self, t, xs):
        """f(x, t) for every state; t and xs are tensors of shape (N,)."""
        env = self._env(t, xs)
        return [f(env) + 0.0 * t for f in self._rhs_torch]   # broadcast constants to (N,)

    def observable_torch(self, name, t, xs):
        if name in self.states:
            return xs[self.states.index(name)]
        return self._out_torch[name](self._env(t, xs)) + 0.0 * t

    def rhs_numpy(self, t, y):
        env = self._env(t, list(y))
        return np.array([float(f(env)) for f in self._rhs_numpy])

    def input_numpy(self, name, t):
        """Value of an input (source) signal over the time array t."""
        t = np.asarray(t, dtype=float)
        return np.broadcast_to(np.asarray(self._in_numpy[name]({"t": t}), dtype=float), t.shape).astype(float)

    def observable_numpy(self, name, t, Y):
        """Y: array (n_states, N)."""
        if name in self.states:
            return np.asarray(Y[self.states.index(name)], dtype=float)
        out = self._out_numpy[name](self._env(t, list(Y)))
        return np.broadcast_to(np.asarray(out, dtype=float), np.shape(t)).astype(float)

    # --------------------------------------------------------------- display
    def residual_latex(self):
        rows = []
        for s in self.states:
            lhs = sp.Derivative(sp.Function(s)(self.t), self.t)
            rhs = self.rhs[s].xreplace({sp.Symbol(x): sp.Function(x)(self.t) for x in self.states})
            rows.append((s, sp.latex(sp.Eq(lhs, rhs)),
                         rf"r_{{{sp.latex(sp.Symbol(s))}}}(t) = \frac{{d\,{sp.latex(sp.Symbol(s))}}}{{dt}} - \left({sp.latex(self.rhs[s])}\right)"))
        return rows

    # -------------------------------------------------------------- reference
    def solve_reference(self, n=1000):
        """Numerical solution of the SAME compiled equations with SciPy (LSODA)."""
        t_eval = np.linspace(self.t_start, self.t_end, n)
        y0 = [self.initial.get(s, 0.0) for s in self.states]
        span = self.t_end - self.t_start
        sol = solve_ivp(self.rhs_numpy, (self.t_start, self.t_end), y0, t_eval=t_eval, method="LSODA",
                        rtol=1e-9, atol=1e-12, max_step=span / 400.0)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            raise ModelError(f"reference solver failed: {sol.message}")
        obs = {o: self.observable_numpy(o, sol.t, sol.y) for o in self.observables}
        return {"t": sol.t, "states": {s: sol.y[i] for i, s in enumerate(self.states)}, "observables": obs,
                "method": "SciPy solve_ivp (LSODA, rtol=1e-9)"}


# --------------------------------------------------------------------------
# Text -> ODESystem
# --------------------------------------------------------------------------
def _split_comment(line):
    body, _, comment = line.partition("#")
    return body.rstrip(), comment.strip()


def _sections(text):
    out = {s: [] for s in SECTIONS}
    current = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        body, comment = _split_comment(raw)
        if not body.strip():
            continue
        m = re.match(r"^\s*\[\s*([A-Za-z_]+)\s*\]\s*$", body)
        if m:
            current = m.group(1).lower()
            if current not in out:
                raise ModelError(f"line {lineno}: unknown section [{current}] (use one of: "
                                 + ", ".join(f"[{s}]" for s in SECTIONS) + ")")
            continue
        if current is None:
            raise ModelError(f"line {lineno}: text before the first [section] header")
        out[current].append((lineno, body, comment))
    return out


def parse_model(text):
    """Parse model text into a compiled ODESystem, or raise ModelError."""
    if not isinstance(text, str) or not text.strip():
        raise ModelError("the model text is empty")
    if len(text) > 200_000:
        raise ModelError("the model text is too long")
    sec = _sections(text)
    t = sp.Symbol("t")
    units, labels, meta, warnings = {}, {}, {}, []

    for ln, body, _c in sec["meta"]:
        k, _, v = body.partition("=")
        if not _:
            raise ModelError(f"line {ln}: expected 'key = value' in [meta]")
        meta[k.strip()] = v.strip()

    # --- equations: find the states first (they define what x(t) means)
    eq_lines = []
    for ln, body, comment in sec["equations"]:
        if "=" not in body:
            raise ModelError(f"line {ln}: an equation needs '=' (expected d(x)/dt = ...)")
        lhs, rhs = body.split("=", 1)
        m = _LHS.match(lhs)
        if not m:
            raise ModelError(f"line {ln}: left side must be a time derivative like d(x(t))/dt or dx/dt, got '{lhs.strip()}'")
        name = m.group(1)
        if name in RESERVED:
            raise ModelError(f"line {ln}: '{name}' is reserved and cannot be a state name")
        if any(name == s for s, *_ in eq_lines):
            raise ModelError(f"line {ln}: state '{name}' has two equations")
        eq_lines.append((name, rhs, ln, comment))
    if not eq_lines:
        raise ModelError("no equations found in the [equations] section")
    states = [s for s, *_ in eq_lines]
    for s, _r, _l, comment in eq_lines:
        if comment:
            labels.setdefault(s, comment)
    state_call = re.compile(r"\b(" + "|".join(map(re.escape, states)) + r")\s*\(\s*t\s*\)")

    def clean(expr_text):
        return state_call.sub(r"\1", expr_text)

    symbols = {s: sp.Symbol(s) for s in states}
    symbols["t"] = t

    # --- parameters (numbers; may reference earlier parameters)
    params = {}
    for ln, body, comment in sec["parameters"]:
        m = _DEF.match(body)
        if not m:
            raise ModelError(f"line {ln}: expected 'name [unit] = value' in [parameters]")
        name, unit, val = m.groups()
        _check_new_name(name, ln, states, params)
        try:
            e = parse_expression(val, {})
            e = e.xreplace({sp.Symbol(k): sp.Float(v) for k, v in params.items()})
            if e.free_symbols:
                raise ModelError(f"line {ln}: parameter '{name}' must be a number "
                                 f"(unknown: {', '.join(sorted(map(str, e.free_symbols)))})")
            params[name] = float(e)
        except ParseError as ex:
            raise ModelError(f"line {ln}: {ex}") from ex
        if not np.isfinite(params[name]):
            raise ModelError(f"line {ln}: parameter '{name}' is not finite")
        if unit:
            units[name] = unit.strip()
        if comment:
            labels[name] = comment

    # --- inputs (functions of t)
    inputs = {}
    for ln, body, comment in sec["inputs"]:
        m = _DEF.match(body)
        if not m:
            raise ModelError(f"line {ln}: expected 'name [unit] = expression of t' in [inputs]")
        name, unit, val = m.groups()
        _check_new_name(name, ln, states, params, inputs)
        try:
            e = parse_expression(val, {"t": t})
        except ParseError as ex:
            raise ModelError(f"line {ln}: {ex}") from ex
        bad = {str(s) for s in e.free_symbols} - {"t"} - set(params)
        if bad:
            raise ModelError(f"line {ln}: input '{name}' may only use t and parameters (unknown: {', '.join(sorted(bad))})")
        inputs[name] = e
        if unit:
            units[name] = unit.strip()
        if comment:
            labels[name] = comment

    known = set(states) | set(params) | set(inputs) | {"t"}

    # --- the ODE right-hand sides
    rhs = {}
    for name, rtext, ln, _c in eq_lines:
        try:
            e = parse_expression(clean(rtext), symbols)
        except ParseError as ex:
            raise ModelError(f"line {ln}: {ex}") from ex
        unknown = {str(s) for s in e.free_symbols} - known
        if unknown:
            raise ModelError(f"line {ln}: equation for '{name}' uses undefined names: {', '.join(sorted(unknown))}. "
                             "Define them under [parameters] or [inputs].")
        rhs[name] = e

    # --- initial conditions
    initial = {}
    for ln, body, comment in sec["initial"]:
        m = _DEF.match(body)
        if not m:
            raise ModelError(f"line {ln}: expected 'state [unit] = value' in [initial]")
        name, unit, val = m.groups()
        name = name.strip()
        if name not in states:
            raise ModelError(f"line {ln}: '{name}' is not a state (states: {', '.join(states)})")
        try:
            e = parse_expression(val, {}).xreplace({sp.Symbol(k): sp.Float(v) for k, v in params.items()})
            initial[name] = float(e)
        except (ParseError, TypeError) as ex:
            raise ModelError(f"line {ln}: initial value for '{name}' must be a number") from ex
        if unit:
            units[name] = unit.strip()
        if comment:
            labels[name] = comment
    for s in states:
        if s not in initial:
            initial[s] = 0.0
            warnings.append(f"No initial value for '{s}'; using 0.")

    # --- outputs (readouts)
    outputs = {}
    for ln, body, comment in sec["outputs"]:
        m = _DEF.match(body)
        if not m:
            raise ModelError(f"line {ln}: expected 'name [unit] = expression' in [outputs]")
        name, unit, val = m.groups()
        _check_new_name(name, ln, states, params, inputs, outputs)
        try:
            e = parse_expression(clean(val), symbols)
        except ParseError as ex:
            raise ModelError(f"line {ln}: {ex}") from ex
        unknown = {str(s) for s in e.free_symbols} - known
        if unknown:
            raise ModelError(f"line {ln}: output '{name}' uses undefined names: {', '.join(sorted(unknown))}")
        outputs[name] = e
        if unit:
            units[name] = unit.strip()
        if comment:
            labels[name] = comment

    observables = set(states) | set(outputs)

    # --- time domain
    tvals = {"t_start": 0.0, "t_end": None}
    for ln, body, _c in sec["time"]:
        k, eq, v = body.partition("=")
        k = k.strip()
        if not eq or k not in tvals:
            raise ModelError(f"line {ln}: [time] accepts 't_start = ...' and 't_end = ...'")
        try:
            tvals[k] = float(v)
        except ValueError as ex:
            raise ModelError(f"line {ln}: {k} must be a number") from ex
    if tvals["t_end"] is None:
        raise ModelError("[time] needs t_end")
    t0, t1 = tvals["t_start"], tvals["t_end"]
    if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
        raise ModelError("t_end must be larger than t_start")

    # --- boundary / point constraints
    boundary = []
    for ln, body, _c in sec["boundary"]:
        m = _BC.match(body)
        if not m:
            raise ModelError(f"line {ln}: expected 'observable(time) = value' in [boundary]")
        name, tb, val = m.groups()
        if name not in observables:
            raise ModelError(f"line {ln}: '{name}' is not a state or output")
        try:
            tb, val = float(tb), float(val)
        except ValueError as ex:
            raise ModelError(f"line {ln}: time and value must be numbers") from ex
        if not (t0 <= tb <= t1):
            raise ModelError(f"line {ln}: time {tb} is outside [{t0}, {t1}]")
        boundary.append((name, tb, val))

    # --- visual hints
    visual = []
    for ln, body, _c in sec["visual"]:
        k, eq, v = body.partition("=")
        k, v = k.strip(), v.strip()
        if not eq or k not in WIDGETS:
            raise ModelError(f"line {ln}: [visual] lines look like 'widget = observable', widget one of {', '.join(WIDGETS)}")
        if v not in observables:
            raise ModelError(f"line {ln}: '{v}' is not a state or output")
        visual.append((k, v))

    for p in params:
        if not any(sp.Symbol(p) in e.free_symbols for e in list(rhs.values()) + list(inputs.values()) + list(outputs.values())):
            warnings.append(f"Parameter '{p}' is declared but not used by any equation.")

    try:
        system = ODESystem(title=meta.get("title", "Untitled model"), states=states, rhs=rhs, params=params,
                           inputs=inputs, initial=initial, outputs=outputs, boundary=boundary, visual=visual,
                           t_start=t0, t_end=t1, units=units, labels=labels, meta=meta, warnings=warnings,
                           source_text=text)
    except ParseError as ex:
        raise ModelError(str(ex)) from ex
    return system


def _check_new_name(name, ln, *taken):
    if not _IDENT.match(name) or name in RESERVED:
        raise ModelError(f"line {ln}: '{name}' is not a valid name")
    for group in taken:
        if name in group:
            raise ModelError(f"line {ln}: name '{name}' is defined twice")


# --------------------------------------------------------------------------
# Demo problem
# --------------------------------------------------------------------------
# The two equations below are, character for character, what BondLab's engine
# derives from the mass-spring-damper bond graph (Se -> 1-junction with I, C, R).
# tests/test_pinn_lab.py re-derives them with the engine and checks they match.
DEMO_TEXT = """\
[meta]
title = Demo Physics Problem - mass-spring-damper (BondLab bond graph)
description = Se (force) on a 1-junction with I (mass), C (spring compliance 1/k) and R (damper).
bondlab_template = mass_spring_damper

[equations]
# Derived automatically by BondLab from the bond graph (causality -> SymPy):
d(f_M1(t))/dt = (C_k*(F - c*f_M1(t)) - q_K1(t))/(C_k*m)
d(q_K1(t))/dt = f_M1(t)

[parameters]
m [kg] = 1.0          # mass m (I element)
C_k [m/N] = 0.01      # spring compliance C = 1/k (C element)
c [N*s/m] = 4.0       # damper c (R element)

[inputs]
F [N] = 1.0*Heaviside(t)   # force step at t = 0 (Se element)

[initial]
f_M1 [m/s] = 0        # velocity of the mass
q_K1 [m] = 0          # spring deflection = displacement of the mass

[outputs]
F_spring [N] = q_K1/C_k    # spring force (effort on the C element)

[visual]
mass_spring = q_K1
gauge = f_M1

[time]
t_start = 0
t_end = 3
"""


# --------------------------------------------------------------------------
# Editing a model from the UI (Physics settings) without retyping it
# --------------------------------------------------------------------------
def update_model_text(text, params=None, initial=None, boundary=None, t_end=None):
    """Return `text` with parameter values / initial values replaced (units and comments kept),
    the [boundary] section replaced by `boundary` = [(observable, t, value), ...], and t_end changed.
    The result is re-parsed by the caller, so any mistake is reported like a typed one."""
    params, initial = dict(params or {}), dict(initial or {})
    out, section, seen_boundary = [], None, False

    def fmt(v):
        return format(float(v), ".10g")

    for raw in text.splitlines():
        body, _, comment = raw.partition("#")
        m_sec = re.match(r"^\s*\[\s*([A-Za-z_]+)\s*\]\s*$", body)
        if m_sec:
            if section == "boundary" and boundary is not None:
                out += [f"{o}({fmt(t)}) = {fmt(v)}" for o, t, v in boundary]
            section = m_sec.group(1).lower()
            seen_boundary = seen_boundary or section == "boundary"
            out.append(raw)
            continue
        if section == "boundary" and boundary is not None:
            continue                                   # old constraints are replaced
        m = _DEF.match(body)
        tail = f"  # {comment.strip()}" if comment.strip() else ""
        if m and section == "parameters" and m.group(1) in params:
            unit = f" [{m.group(2)}]" if m.group(2) else ""
            out.append(f"{m.group(1)}{unit} = {fmt(params[m.group(1)])}{tail}")
            continue
        if m and section == "initial" and m.group(1) in initial:
            unit = f" [{m.group(2)}]" if m.group(2) else ""
            out.append(f"{m.group(1)}{unit} = {fmt(initial[m.group(1)])}{tail}")
            continue
        if section == "time" and t_end is not None and body.strip().startswith("t_end"):
            out.append(f"t_end = {fmt(t_end)}")
            continue
        out.append(raw)
    if section == "boundary" and boundary is not None:
        out += [f"{o}({fmt(t)}) = {fmt(v)}" for o, t, v in boundary]
    if boundary and not seen_boundary:
        out += ["", "[boundary]"] + [f"{o}({fmt(t)}) = {fmt(v)}" for o, t, v in boundary]
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")
