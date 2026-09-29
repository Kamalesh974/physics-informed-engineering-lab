"""
bond_graph.py -- Hardcoded bond graph topology for the one-way coupled
thermal-structural disc brake model described in manual_derivation.md.

This is NOT a general-purpose bond graph parser: the topology (which nodes
exist, how they're wired) is fixed to this specific disc-brake model. What
IS generic here is the per-element metadata contract below, which
scap_causality.py and equation_gen.py consume without knowing anything
disc-brake-specific -- they only ever look at element `type` and the
constitutive-law metadata, so the automation in later steps is real
(equations are derived from graph structure, not hardcoded), even though the
graph itself is hand-built for this one system.

---------------------------------------------------------------------------
GRAPH CONTRACT (read this before touching scap_causality.py / equation_gen.py)
---------------------------------------------------------------------------
Graph type: networkx.DiGraph. Edge direction = reference direction of
positive power flow (the bond graph "half arrow"), NOT the causality stroke.

Node attribute `type`, one of:
    '0'   -- 0-junction (common EFFORT across all attached bonds,
             flows sum to zero with sign = edge direction convention)
    '1'   -- 1-junction (common FLOW across all attached bonds,
             efforts sum to zero with sign = edge direction convention)
    'R'   -- resistor,      constitutive law e = f * R_param
    'C'   -- capacitor,     integral-causality state q = generalized
             displacement (q = INTEGRAL of net flow dt); readout e = q / C_param
    'I'   -- inertia,       integral-causality state p = generalized
             momentum (p = INTEGRAL of net effort dt); readout f = p / I_param
    'Se'  -- effort source, prescribes e = node['expr'](t)
    'Sf'  -- flow source,   prescribes f = node['expr'](t)
    'MSe' -- modulated effort source, e = node['expr'], where 'expr' contains
             the symbol stored in node['signal_symbol'] which must be
             resolved from the node named node['signal_from'] (a SIGNAL
             bond, no power transfer -- see edge attribute 'signal' below)
    'MSf' -- modulated flow source (same idea, used for the two heat inputs,
             modulated by the prescribed kinematic profile v(t))

Node attributes for R / C / I:
    'param_symbol' -- sympy.Symbol used inside constitutive-law expressions
    'param_key'    -- string key into params.PARAMS for the numeric value
Node attributes for C additionally:
    'state_symbol' -- sympy Function q_xxx(t), the generalized displacement
                       this element integrates (its ODE state variable)
Node attributes for I additionally:
    'state_symbol' -- sympy Function p_xxx(t), the generalized momentum

Node attributes for Se / Sf / MSe / MSf:
    'expr'          -- sympy expression for the prescribed effort/flow
    'signal_from'   -- (MSe/MSf only) node id supplying the modulating
                        signal, or None if `expr` is a pure function of t
    'signal_symbol' -- (MSe/MSf only) the sympy symbol inside `expr` that
                        stands for the modulating signal's *readout* value

Edge attributes:
    'bond_id' -- string label ('b1', 'b2', ...)
    'signal'  -- bool, True for signal/activated bonds that carry
                  information but no power (only the T_d -> MSe_th bond).
                  All other bonds are power bonds.
    'causality' -- filled in later by scap_causality.py; None here.

Junction sign convention: for a '0' or '1' junction, an edge pointing
*into* the junction (u -> junction) is a positive contribution in the
junction's sum law; an edge pointing *out of* the junction (junction -> u)
is a negative contribution. This is just a bookkeeping convention (edges
were drawn in the "natural" power-flow direction from source to sink), and
equation_gen.py applies it consistently.
---------------------------------------------------------------------------
"""
import math
import networkx as nx
import sympy as sp

import params

# ---------------------------------------------------------------------
# Symbolic time base and dynamic (state / port) variables
# ---------------------------------------------------------------------
t = sp.symbols("t", nonnegative=True)

# Generalized displacement / momentum states (what equation_gen integrates)
q_Cd = sp.Function("q_Cd")(t)   # disc thermal C state      (T_d = q_Cd / C_d)
q_Cp = sp.Function("q_Cp")(t)   # pad  thermal C state      (T_p = q_Cp / C_p)
q_Cs = sp.Function("q_Cs")(t)   # structural spring state   (== displacement x)
p_Is = sp.Function("p_Is")(t)   # structural inertia state  (== momentum m_eff*v_x)

STATE_VARS = {"q_Cd": q_Cd, "q_Cp": q_Cp, "q_Cs": q_Cs, "p_Is": p_Is}

# ---------------------------------------------------------------------
# Parameter symbols (numeric values live in params.PARAMS)
# ---------------------------------------------------------------------
mu_s, Fcl_s, v0_s, a_s, tstop_s, gamma_s = sp.symbols(
    "mu F_cl v0 a t_stop gamma", real=True
)
Cd_s, Cp_s, R1_s, R2_s, R3_s = sp.symbols("C_d C_p R1 R2 R3", positive=True)
Tamb_s, Tref_s = sp.symbols("T_amb T_ref", real=True)
alpha_s, E_s, Aeff_s = sp.symbols("alpha E A_eff", positive=True)
meff_s, kstruct_s, cstruct_s = sp.symbols("m_eff k_struct c_struct", positive=True)

# ---------------------------------------------------------------------
# Prescribed kinematic input and friction heat source expressions.
# These encode the *physics of the source elements themselves* (how much
# heat this specific friction interface generates), which is legitimately
# part of describing what the Sf/MSf elements ARE -- equation_gen.py never
# needs to know about mu, F_cl or v(t); it just reads these expressions off
# the MSf_d / MSf_p nodes.
# ---------------------------------------------------------------------
v_expr = sp.Piecewise((v0_s - a_s * t, t <= tstop_s), (0, True))
q_expr = mu_s * Fcl_s * v_expr          # total friction heat rate q(t)
q_d_expr = gamma_s * q_expr             # heat rate into the disc
q_p_expr = (1 - gamma_s) * q_expr       # heat rate into the pad

# Modulated-effort-source law for the thermal->structural coupling
# (constrained thermal stress force). T_d_signal is a placeholder symbol
# standing for "the readout effort of node J0_d" (i.e. T_d = q_Cd/C_d),
# substituted in by equation_gen.py when it resolves the signal bond.
T_d_signal = sp.Symbol("T_d_signal", real=True)
Fth_expr = E_s * alpha_s * Aeff_s * (T_d_signal - Tref_s)


def build_bond_graph() -> nx.DiGraph:
    """Construct the hardcoded disc-brake bond graph.

    Returns a networkx.DiGraph. See module docstring for the node/edge
    attribute contract.
    """
    G = nx.DiGraph()

    # ---------------- Thermal subsystem ----------------
    G.add_node("Se_amb", type="Se", domain="thermal", expr=Tamb_s, label="T_amb (ambient)")

    G.add_node(
        "MSf_d", type="MSf", domain="thermal", expr=q_d_expr,
        signal_from=None, signal_symbol=None, label="q_d(t) = gamma*mu*F_cl*v(t)",
    )
    G.add_node(
        "MSf_p", type="MSf", domain="thermal", expr=q_p_expr,
        signal_from=None, signal_symbol=None, label="q_p(t) = (1-gamma)*mu*F_cl*v(t)",
    )

    G.add_node("J0_d", type="0", domain="thermal", label="0-junction: common T_d")
    G.add_node("J0_p", type="0", domain="thermal", label="0-junction: common T_p")

    G.add_node(
        "C_d", type="C", domain="thermal", param_symbol=Cd_s, param_key="C_d",
        state_symbol=q_Cd, label="disc thermal capacitance",
    )
    G.add_node(
        "C_p", type="C", domain="thermal", param_symbol=Cp_s, param_key="C_p",
        state_symbol=q_Cp, label="pad thermal capacitance",
    )

    G.add_node("J1_R1", type="1", domain="thermal", label="1-junction for R1 branch")
    G.add_node("J1_R2", type="1", domain="thermal", label="1-junction for R2 branch")
    G.add_node("J1_R3", type="1", domain="thermal", label="1-junction for R3 branch")

    G.add_node("R1", type="R", domain="thermal", param_symbol=R1_s, param_key="R1",
               label="disc -> hub conduction")
    G.add_node("R2", type="R", domain="thermal", param_symbol=R2_s, param_key="R2",
               label="disc -> ambient convection")
    G.add_node("R3", type="R", domain="thermal", param_symbol=R3_s, param_key="R3",
               label="pad -> ambient (cond.+conv.)")

    thermal_bonds = [
        ("MSf_d", "J0_d"),
        ("J0_d", "C_d"),
        ("J0_d", "J1_R1"), ("J1_R1", "R1"), ("J1_R1", "Se_amb"),
        ("J0_d", "J1_R2"), ("J1_R2", "R2"), ("J1_R2", "Se_amb"),
        ("MSf_p", "J0_p"),
        ("J0_p", "C_p"),
        ("J0_p", "J1_R3"), ("J1_R3", "R3"), ("J1_R3", "Se_amb"),
    ]

    # ---------------- Structural subsystem ----------------
    G.add_node(
        "MSe_th", type="MSe", domain="mechanical", expr=Fth_expr,
        signal_from="J0_d", signal_symbol=T_d_signal,
        label="F_th(t) = E*alpha*A_eff*(T_d - T_ref)  [modulated by signal from J0_d]",
    )
    G.add_node("J1_struct", type="1", domain="mechanical", label="1-junction: common v_x")
    G.add_node(
        "I_struct", type="I", domain="mechanical", param_symbol=meff_s, param_key="m_eff",
        state_symbol=p_Is, label="effective modal mass",
    )
    G.add_node(
        "C_struct", type="C", domain="mechanical",
        param_symbol=1 / kstruct_s, param_key="k_struct", param_is_inverse=True,
        state_symbol=q_Cs, label="effective modal stiffness (compliance = 1/k_struct)",
    )
    G.add_node("R_struct", type="R", domain="mechanical", param_symbol=cstruct_s,
               param_key="c_struct", label="effective modal damping")

    structural_power_bonds = [
        ("MSe_th", "J1_struct"),
        ("J1_struct", "I_struct"),
        ("J1_struct", "C_struct"),
        ("J1_struct", "R_struct"),
    ]
    signal_bonds = [("J0_d", "MSe_th")]  # one-way, no power

    for idx, (u, v) in enumerate(thermal_bonds + structural_power_bonds, start=1):
        G.add_edge(u, v, bond_id=f"b{idx}", signal=False, causality=None)
    for idx, (u, v) in enumerate(signal_bonds, start=len(thermal_bonds) + len(structural_power_bonds) + 1):
        G.add_edge(u, v, bond_id=f"b{idx}", signal=True, causality=None)

    return G


def summarize(G: nx.DiGraph) -> str:
    lines = []
    lines.append(f"Nodes: {G.number_of_nodes()}   Edges (bonds): {G.number_of_edges()}")
    element_types = ["Se", "Sf", "MSe", "MSf", "R", "C", "I"]
    n_elements = sum(1 for _, d in G.nodes(data=True) if d["type"] in element_types)
    n_junctions = sum(1 for _, d in G.nodes(data=True) if d["type"] in ("0", "1"))
    lines.append(f"R/C/I/Source elements: {n_elements}   Junctions (0/1): {n_junctions}")
    lines.append("")
    lines.append(f"{'node':12s} {'type':5s} {'domain':11s} label")
    lines.append("-" * 70)
    for n, d in G.nodes(data=True):
        lines.append(f"{n:12s} {d['type']:5s} {d.get('domain',''):11s} {d.get('label','')}")
    lines.append("")
    lines.append(f"{'bond':6s} {'from':12s} {'->':2s} {'to':12s} signal")
    lines.append("-" * 70)
    for u, v, d in G.edges(data=True):
        lines.append(f"{d['bond_id']:6s} {u:12s} -> {v:12s} {d['signal']}")
    return "\n".join(lines)


def draw(G: nx.DiGraph, path="figures/bond_graph_topology.png"):
    """Best-effort topology plot for a visual sanity check. Non-fatal if
    matplotlib isn't available or layout looks messy -- this is just a
    debugging aid, not a deliverable figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure.")
        return

    color_map = {
        "0": "#f4a261", "1": "#e9c46a", "R": "#2a9d8f", "C": "#264653",
        "I": "#e76f51", "Se": "#8ab17d", "Sf": "#8ab17d",
        "MSe": "#c084fc", "MSf": "#c084fc",
    }
    colors = [color_map[d["type"]] for _, d in G.nodes(data=True)]
    pos = nx.spring_layout(G, seed=7, k=1.2)
    plt.figure(figsize=(11, 8))
    power_edges = [(u, v) for u, v, d in G.edges(data=True) if not d["signal"]]
    signal_edges = [(u, v) for u, v, d in G.edges(data=True) if d["signal"]]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1400)
    nx.draw_networkx_labels(G, pos, font_size=7)
    nx.draw_networkx_edges(G, pos, edgelist=power_edges, arrows=True, width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=signal_edges, arrows=True, width=1.5,
                            style="dashed", edge_color="purple")
    plt.title("Disc brake bond graph (dashed purple = signal/activated bond)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved topology figure to {path}")


if __name__ == "__main__":
    G = build_bond_graph()
    print(summarize(G))
    draw(G)
