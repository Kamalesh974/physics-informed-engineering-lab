"""
equation_gen.py -- symbolically derive the governing ODEs from the
causally-assigned bond graph (bond_graph.py + scap_causality.py) using
SymPy. No equations are hardcoded here: this module only knows the generic
bond-graph rules (junction sum laws + R/C/I/source constitutive laws) and
walks whatever graph + causality it's given.

CORE IDEA
Every bond has exactly one EFFORT value and one FLOW value (shared by both
ends). Causality (scap_causality.py) tells us, for each bond, which end's
element law determines effort (the "giver") -- the other end's element law
then determines flow using that effort as input. Two mutually-recursive,
memoized functions walk the graph to resolve any bond's effort/flow:

    bond_effort(edge) -- dispatches on the EFFORT GIVER's element type:
        Se/MSe : returns the source's prescribed expression
        C      : returns state/param              (e = q/C, integral causality)
        R      : returns bond_flow(SAME edge)*R    (resistance causality)
        0-junc : this must be a "relay" bond -> returns bond_effort(that
                 junction's special bond)            (0-junction: effort is common)
        1-junc : this must be the junction's special bond -> returns the
                 SIGNED SUM of every other incident bond's effort
                 (1-junction: efforts sum to zero, so the special one is
                 minus the sum of the rest)

    bond_flow(edge) -- dispatches on the FLOW-DETERMINING side (whichever
    end is NOT the effort giver), by the dual rules:
        Sf/MSf : returns the source's prescribed expression
        I      : returns state/param              (f = p/I, integral causality)
        R      : returns bond_effort(SAME edge)/R  (conductance causality)
        1-junc : "relay" bond -> returns bond_flow(that junction's special bond)
        0-junc : special bond -> returns the SIGNED SUM of every other
                  incident bond's flow

A state element's (C or I) own ODE is then just the flow (for C) or effort
(for I) on its own bond, because that bond was drawn (in bond_graph.py)
pointing FROM the junction TOWARD the state element, so "flow/effort in
that reference direction" is already exactly "net flow/effort INTO the
element", which is by definition dq/dt or dp/dt.

This file was hand-verified against manual_derivation.md section 3 before
being trusted (see the walkthrough in the conversation/README) -- the
output below reproduces all four ODEs (Section 3.10) exactly.
"""
import sympy as sp

import params
from bond_graph import build_bond_graph
from scap_causality import assign_causality, _incident_power_bonds


# ---------------------------------------------------------------------
# Junction helpers
# ---------------------------------------------------------------------
def _incident_edges(G, node):
    return [ek for ek, _other in _incident_power_bonds(G, node)]


def _sign_at(G, node, edge_key):
    u, v = edge_key
    if v == node:
        return 1
    if u == node:
        return -1
    raise ValueError(f"{node} is not an endpoint of {edge_key}")


def _find_special_bond(G, junction):
    """The one incident bond that carries the junction's 'sum law' role
    (0-junction: the bond whose outside element supplies the common
    effort; 1-junction: the bond whose outside element supplies the
    common flow). SCAP already guarantees exactly one exists."""
    jtype = G.nodes[junction]["type"]
    specials = []
    for ek in _incident_edges(G, junction):
        giver = G.edges[ek]["effort_giver"]
        is_special = (giver != junction) if jtype == "0" else (giver == junction)
        if is_special:
            specials.append(ek)
    if len(specials) != 1:
        raise RuntimeError(f"{junction}: expected exactly 1 special bond, found {specials}")
    return specials[0]


# ---------------------------------------------------------------------
# Recursive, memoized bond effort/flow resolution
# ---------------------------------------------------------------------
def _resolve_source_expr(G, node, memo, resolving):
    """Se/MSe/Sf/MSf: return the node's expr, substituting the modulating
    signal (read off another node's resolved effort OR flow) if present.

    node['signal_kind'] selects which port variable of the signal_from
    junction is read: 'effort' (default, e.g. a temperature -- the
    original disc-brake model's T_d -> MSe_th coupling) or 'flow' (e.g. an
    angular velocity feeding a modulated heat source, as in the
    torque/inertia/friction front-end read from the second bond-graph
    image). Defaults to 'effort' so existing callers are unaffected."""
    expr = G.nodes[node]["expr"]
    signal_from = G.nodes[node].get("signal_from")
    if signal_from is not None:
        sig_sym = G.nodes[node]["signal_symbol"]
        signal_kind = G.nodes[node].get("signal_kind", "effort")
        special_bond = _find_special_bond(G, signal_from)
        if signal_kind == "effort":
            sig_val = bond_effort(G, special_bond, memo, resolving)
        elif signal_kind == "flow":
            sig_val = bond_flow(G, special_bond, memo, resolving)
        else:
            raise ValueError(f"{node}: unknown signal_kind '{signal_kind}'")
        expr = expr.subs(sig_sym, sig_val)
    return expr


def _gy_other_edge(G, gy_node, edge):
    """The gyrator's OTHER incident bond, given one of its two bonds --
    GY's law (e1=r*f2, e2=r*f1) always needs the opposite port's OTHER
    port variable, and since the law is symmetric under swapping port
    labels 1<->2, we never need to distinguish which port is "1" vs "2",
    just "this edge" vs "the other one"."""
    edges = _incident_edges(G, gy_node)
    others = [e for e in edges if e != edge]
    if len(others) != 1:
        raise RuntimeError(f"{gy_node}: expected exactly 2 incident bonds for GY, found {len(edges)}")
    return others[0]


def _tf_port_info(G, tf_node, edge):
    """Returns (is_port1, other_edge) for one of a TF's two incident bonds.
    UNLIKE GY, a transformer's law (e1 = n*e2, f2 = n*f1) is NOT symmetric
    under swapping the two ports (dividing by n vs multiplying by n), so
    which edge counts as "port 1" vs "port 2" matters -- the convention
    used here is simply "the order _incident_power_bonds finds them in"
    (first = port 1), matching the SAME convention scap_causality.py's TF
    propagation rule already relies on (both call the same function on the
    same graph, so they agree without needing an explicit port label
    stored on the graph)."""
    edges = _incident_edges(G, tf_node)
    if len(edges) != 2:
        raise RuntimeError(f"{tf_node}: expected exactly 2 incident bonds for TF, found {len(edges)}")
    is_port1 = (edges[0] == edge)
    other = edges[1] if is_port1 else edges[0]
    return is_port1, other


def bond_effort(G, edge, memo, resolving=frozenset()):
    if edge in memo["e"]:
        return memo["e"][edge]
    key = (edge, "e")
    if key in resolving:
        raise RuntimeError(f"Causal loop detected resolving effort of {edge}")
    resolving = resolving | {key}

    giver = G.edges[edge]["effort_giver"]
    gtype = G.nodes[giver]["type"]

    if gtype in ("Se", "MSe"):
        val = _resolve_source_expr(G, giver, memo, resolving)
    elif gtype == "C":
        val = G.nodes[giver]["state_symbol"] / G.nodes[giver]["param_symbol"]
    elif gtype == "R":
        val = bond_flow(G, edge, memo, resolving) * G.nodes[giver]["param_symbol"]
    elif gtype == "0":
        special = _find_special_bond(G, giver)
        if special == edge:
            raise RuntimeError(f"{giver}: 0-junction unexpectedly self-computing effort on its own special bond")
        val = bond_effort(G, special, memo, resolving)
    elif gtype == "1":
        special = _find_special_bond(G, giver)
        if special != edge:
            raise RuntimeError(f"{giver}: 1-junction giving effort on a non-special bond")
        others = [e for e in _incident_edges(G, giver) if e != edge]
        s = sum(_sign_at(G, giver, e) * bond_effort(G, e, memo, resolving) for e in others)
        val = -_sign_at(G, giver, edge) * s
    elif gtype == "I":
        raise RuntimeError(f"{giver}: I element asked to give effort (derivative causality, unsupported)")
    elif gtype == "GY":
        # GY gives effort on this bond using modulus*OTHER bond's flow
        # (e1 = r*f2) -- see scap_causality.py's GY propagation rule for
        # why both ports always share this same causal role.
        other_edge = _gy_other_edge(G, giver, edge)
        val = G.nodes[giver]["param_symbol"] * bond_flow(G, other_edge, memo, resolving)
    elif gtype == "TF":
        # TF gives effort on this bond from the OTHER port's effort (the
        # port causality PASSES THROUGH a transformer, unlike GY -- see
        # scap_causality.py's TF propagation rule): e1 = n*e2 if this edge
        # is port 1, or e2 = e1/n if this edge is port 2.
        is_port1, other_edge = _tf_port_info(G, giver, edge)
        n_sym = G.nodes[giver]["param_symbol"]
        other_effort = bond_effort(G, other_edge, memo, resolving)
        val = n_sym * other_effort if is_port1 else other_effort / n_sym
    else:
        raise RuntimeError(f"Unhandled giver type '{gtype}' at node {giver}")

    memo["e"][edge] = val
    return val


def bond_flow(G, edge, memo, resolving=frozenset()):
    if edge in memo["f"]:
        return memo["f"][edge]
    key = (edge, "f")
    if key in resolving:
        raise RuntimeError(f"Causal loop detected resolving flow of {edge}")
    resolving = resolving | {key}

    u, v = edge
    giver = G.edges[edge]["effort_giver"]
    receiver = v if giver == u else u
    rtype = G.nodes[receiver]["type"]

    if rtype in ("Sf", "MSf"):
        val = _resolve_source_expr(G, receiver, memo, resolving)
    elif rtype == "I":
        val = G.nodes[receiver]["state_symbol"] / G.nodes[receiver]["param_symbol"]
    elif rtype == "R":
        val = bond_effort(G, edge, memo, resolving) / G.nodes[receiver]["param_symbol"]
    elif rtype == "1":
        special = _find_special_bond(G, receiver)
        if special == edge:
            raise RuntimeError(f"{receiver}: 1-junction unexpectedly self-computing flow on its own special bond")
        val = bond_flow(G, special, memo, resolving)
    elif rtype == "0":
        special = _find_special_bond(G, receiver)
        if special != edge:
            raise RuntimeError(f"{receiver}: 0-junction giving flow on a non-special bond")
        others = [e for e in _incident_edges(G, receiver) if e != edge]
        s = sum(_sign_at(G, receiver, e) * bond_flow(G, e, memo, resolving) for e in others)
        val = -_sign_at(G, receiver, edge) * s
    elif rtype == "C":
        raise RuntimeError(f"{receiver}: C element asked to give flow (derivative causality, unsupported)")
    elif rtype == "GY":
        # GY gives flow on this bond using OTHER bond's effort / modulus
        # (f2 = e1/r) -- the dual of the bond_effort GY case above.
        other_edge = _gy_other_edge(G, receiver, edge)
        val = bond_effort(G, other_edge, memo, resolving) / G.nodes[receiver]["param_symbol"]
    elif rtype == "TF":
        # TF gives flow on this bond from the OTHER port's flow (dual of
        # the bond_effort TF case): f1 = f2/n if this edge is port 1
        # (power conservation e1*f1=e2*f2 requires the DUAL scaling of the
        # effort law), or f2 = n*f1 if this edge is port 2.
        is_port1, other_edge = _tf_port_info(G, receiver, edge)
        n_sym = G.nodes[receiver]["param_symbol"]
        other_flow = bond_flow(G, other_edge, memo, resolving)
        val = other_flow / n_sym if is_port1 else n_sym * other_flow
    else:
        raise RuntimeError(f"Unhandled receiver type '{rtype}' at node {receiver}")

    memo["f"][edge] = val
    return val


# ---------------------------------------------------------------------
# Top-level: assemble the ODE system
# ---------------------------------------------------------------------
def generate_odes():
    """Build the graph, run SCAP, and symbolically derive the ODE system.

    Returns a dict:
      't'        : sympy time symbol
      'states'   : dict name -> sympy Function(t), the 4 generalized
                    displacement/momentum state variables
      'odes'     : dict state Function -> sympy expr for its time derivative
      'readouts' : dict physical-quantity name -> sympy expr in terms of
                    the states (T_d, T_p, x, v_x, sigma_th), matching the
                    notation used in manual_derivation.md section 3.10
      'graph'    : the causally-assigned graph (for inspection/plotting)
    """
    G = build_bond_graph()
    scap_report = assign_causality(G)
    if scap_report["conflicts"]:
        raise RuntimeError(f"SCAP failed to resolve causality: {scap_report['conflicts']}")

    memo = {"e": {}, "f": {}}

    def state_edge(node):
        edges = _incident_edges(G, node)
        assert len(edges) == 1, f"{node}: expected exactly 1 incident bond, found {edges}"
        return edges[0]

    from bond_graph import q_Cd, q_Cp, q_Cs, p_Is, t, Cd_s, Cp_s, meff_s, Tref_s, E_s, alpha_s

    dq_Cd_dt = bond_flow(G, state_edge("C_d"), memo)
    dq_Cp_dt = bond_flow(G, state_edge("C_p"), memo)
    dq_Cs_dt = bond_flow(G, state_edge("C_struct"), memo)
    dp_Is_dt = bond_effort(G, state_edge("I_struct"), memo)

    odes = {
        q_Cd: sp.simplify(dq_Cd_dt),
        q_Cp: sp.simplify(dq_Cp_dt),
        q_Cs: sp.simplify(dq_Cs_dt),
        p_Is: sp.simplify(dp_Is_dt),
    }

    T_d_readout = q_Cd / Cd_s
    T_p_readout = q_Cp / Cp_s
    x_readout = q_Cs                     # C_struct's param is compliance -> q IS displacement
    v_x_readout = p_Is / meff_s
    sigma_th_readout = E_s * alpha_s * (T_d_readout - Tref_s)

    readouts = {
        "T_d": T_d_readout,
        "T_p": T_p_readout,
        "x": x_readout,
        "v_x": v_x_readout,
        "sigma_th": sigma_th_readout,
    }

    return {
        "t": t,
        "states": {"q_Cd": q_Cd, "q_Cp": q_Cp, "q_Cs": q_Cs, "p_Is": p_Is},
        "odes": odes,
        "readouts": readouts,
        "graph": G,
        "scap_report": scap_report,
    }


def derive_odes_generic(G):
    """Fully generic ODE derivation: auto-discovers every C and I element
    in G by node TYPE (not by hardcoded node name) and derives its state
    equation. Unlike generate_odes() (which is specific to the disc-brake
    model's particular node names), this works on ANY causally-resolvable
    graph following the bond_graph.py schema -- e.g. one built
    programmatically from an LLM's reading of an arbitrary bond graph
    image (see image_to_bond_graph.py).

    Mutates G in place (runs assign_causality on it). Raises RuntimeError
    if SCAP can't resolve it.

    Returns (state_symbols, odes):
      state_symbols -- dict node_id -> its state Function(t)
      odes          -- dict state Function -> its dq/dt or dp/dt expression
    """
    scap_report = assign_causality(G)
    if scap_report["conflicts"]:
        raise RuntimeError(f"SCAP failed to resolve causality: {scap_report['conflicts']}")

    memo = {"e": {}, "f": {}}
    state_symbols = {}
    odes = {}
    for node, d in G.nodes(data=True):
        if d["type"] not in ("C", "I"):
            continue
        edges = _incident_edges(G, node)
        if len(edges) != 1:
            raise RuntimeError(f"{node}: expected exactly 1 incident power bond, found {len(edges)}")
        edge = edges[0]
        state = d["state_symbol"]
        state_symbols[node] = state
        odes[state] = sp.simplify(bond_flow(G, edge, memo) if d["type"] == "C" else bond_effort(G, edge, memo))
    return state_symbols, odes


def to_physical_form(result):
    """Rewrite the q/p-form ODEs from generate_odes() into the physical
    variables T_d, T_p, x, v_x via the readout relations (same conversion
    validate.py uses for its symbolic/numeric checks, factored out here so
    pinn_model.py can reuse it without re-deriving anything by hand).

    Returns (state_funcs, odes):
      state_funcs -- dict name -> sympy Function(t): T_d, T_p, x, v_x
      odes        -- dict Function -> rhs expr, written purely in terms of
                      T_d(t), T_p(t), x(t), v_x(t), t, and parameter symbols
    """
    from bond_graph import Cd_s, Cp_s, meff_s, t as t_sym

    q_Cd_s, q_Cp_s = result["states"]["q_Cd"], result["states"]["q_Cp"]
    q_Cs_s, p_Is_s = result["states"]["q_Cs"], result["states"]["p_Is"]
    odes_qp = result["odes"]

    T_d = sp.Function("T_d")(t_sym)
    T_p = sp.Function("T_p")(t_sym)
    x = sp.Function("x")(t_sym)
    v_x = sp.Function("v_x")(t_sym)

    dTd_dt = sp.simplify(odes_qp[q_Cd_s].subs(q_Cd_s, Cd_s * T_d) / Cd_s)
    dTp_dt = sp.simplify(odes_qp[q_Cp_s].subs(q_Cp_s, Cp_s * T_p) / Cp_s)
    dx_dt = sp.simplify(odes_qp[q_Cs_s].subs(p_Is_s, meff_s * v_x))
    dvx_dt = sp.simplify(
        odes_qp[p_Is_s].subs([(q_Cd_s, Cd_s * T_d), (q_Cs_s, x), (p_Is_s, meff_s * v_x)]) / meff_s
    )

    state_funcs = {"T_d": T_d, "T_p": T_p, "x": x, "v_x": v_x}
    odes_physical = {T_d: dTd_dt, T_p: dTp_dt, x: dx_dt, v_x: dvx_dt}
    return state_funcs, odes_physical


def substitute_params(expr):
    """Replace every free symbol in `expr` whose name matches a key in
    params.PARAMS with its numeric value. (Symbol names were chosen in
    bond_graph.py to exactly match params.PARAMS keys, so this is a
    generic, no-special-casing substitution.)"""
    subs = {}
    for sym in expr.free_symbols:
        name = str(sym)
        if name in params.PARAMS:
            subs[sym] = params.PARAMS[name]
    return expr.subs(subs)


def pretty_print(result):
    lines = []
    lines.append("=" * 78)
    lines.append("AUTO-DERIVED ODE SYSTEM (generalized displacement/momentum form)")
    lines.append("=" * 78)
    for state, rhs in result["odes"].items():
        lines.append(f"\nd({state})/dt =")
        lines.append(f"    {rhs}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("READOUTS (physical quantities, matching manual_derivation.md notation)")
    lines.append("=" * 78)
    for name, expr in result["readouts"].items():
        lines.append(f"  {name:10s} = {expr}")
    return "\n".join(lines)


if __name__ == "__main__":
    result = generate_odes()
    print(pretty_print(result))
