"""
library.py -- component catalog (per physical domain) and ready-made templates.

Every template is a "project document":
  {meta, spec:{nodes,edges}, layout:{id:{x,y}}, values, sources, initial, sim}
The multi-physics templates (DC motor, geared load, thermal->structural) use
topologies that were validated EXACTLY against textbook / hand-derived
equations earlier in this project (see MASTER_HANDOFF.md, section 4) and are
re-checked numerically by app/backend/tests.
"""
import re

# ---------------------------------------------------------------------------
# Domains: naming + units of the four generalized variables
# ---------------------------------------------------------------------------
DOMAINS = {
    "electrical": {
        "label": "Electrical",
        "effort": ("Voltage", "V"), "flow": ("Current", "A"), "disp": ("Charge", "C"),
        "R": ("Resistor", "R", "ohm", 5.0), "C": ("Capacitor", "C", "F", 1e-4),
        "I": ("Inductor", "L", "H", 0.1),
        "Se": ("Voltage source", "V"), "Sf": ("Current source", "I_in"),
    },
    "mech_trans": {
        "label": "Mechanical (translation)",
        "effort": ("Force", "N"), "flow": ("Velocity", "m/s"), "disp": ("Displacement", "m"),
        "R": ("Damper", "c", "N*s/m", 2.0),
        "C": ("Spring (compliance = 1/k)", "C_k", "m/N", 0.01),
        "I": ("Mass", "m", "kg", 1.0),
        "Se": ("Force source", "F"), "Sf": ("Velocity source", "v_in"),
    },
    "mech_rot": {
        "label": "Mechanical (rotation)",
        "effort": ("Torque", "N*m"), "flow": ("Angular velocity", "rad/s"), "disp": ("Angle", "rad"),
        "R": ("Rotary damper", "b", "N*m*s/rad", 0.05),
        "C": ("Torsional compliance", "C_t", "rad/(N*m)", 0.01),
        "I": ("Rotary inertia", "J", "kg*m^2", 0.01),
        "Se": ("Torque source", "T_in"), "Sf": ("Speed source", "w_in"),
    },
    "thermal": {
        "label": "Thermal (pseudo bond graph: T and heat flow)",
        "effort": ("Temperature", "K"), "flow": ("Heat flow", "W"), "disp": ("Stored heat", "J"),
        "R": ("Thermal resistance", "R_th", "K/W", 0.5),
        "C": ("Thermal capacitance", "C_th", "J/K", 500.0),
        "I": None,
        "Se": ("Temperature source", "T_src"), "Sf": ("Heat-flow source", "Q_heat"),
    },
    "hydraulic": {
        "label": "Hydraulic",
        "effort": ("Pressure", "Pa"), "flow": ("Flow rate", "m^3/s"), "disp": ("Volume", "m^3"),
        "R": ("Pipe / orifice resistance", "R_h", "Pa*s/m^3", 1e6),
        "C": ("Tank / accumulator", "C_h", "m^3/Pa", 1e-8),
        "I": ("Fluid inertance", "I_h", "Pa*s^2/m^3", 1e4),
        "Se": ("Pressure source", "P_in"), "Sf": ("Flow source", "Q_in"),
    },
    "generic": {
        "label": "Generic",
        "effort": ("Effort", "e"), "flow": ("Flow", "f"), "disp": ("Displacement", "q"),
        "R": ("Resistor", "R", "-", 1.0), "C": ("Capacitor", "C", "-", 1.0),
        "I": ("Inertance", "I", "-", 1.0),
        "Se": ("Effort source", "E"), "Sf": ("Flow source", "F"),
    },
}


def domain_key(text):
    """Normalize a free-text domain (e.g. from the vision model) to a key."""
    t = (text or "").lower()
    if any(w in t for w in ("elec", "circuit", "volt")):
        return "electrical"
    if any(w in t for w in ("rot", "torque", "angular", "shaft", "gear")):
        return "mech_rot"
    if any(w in t for w in ("mech", "trans", "linear", "force", "spring", "mass")):
        return "mech_trans"
    if any(w in t for w in ("therm", "heat", "temp")):
        return "thermal"
    if any(w in t for w in ("hyd", "fluid", "pneu", "tank", "pipe")):
        return "hydraulic"
    return t if t in DOMAINS else "generic"


def _dom(key):
    return DOMAINS.get(domain_key(key), DOMAINS["generic"])


def param_unit(ntype, domain):
    d = _dom(domain)
    if ntype in ("R", "C", "I") and d.get(ntype):
        return d[ntype][2]
    if ntype in ("GY", "TF"):
        return "-"
    return ""


def series_labels(node_type, domain):
    """(flow_name, flow_unit, effort_name, effort_unit, disp_name, disp_unit)"""
    d = _dom(domain)
    return d["flow"][0], d["flow"][1], d["effort"][0], d["effort"][1], d["disp"][0], d["disp"][1]


def components():
    """Palette entries, grouped for the UI."""
    items = [
        {"key": "J0", "type": "0", "domain": "any", "name": "0-junction (common effort)",
         "hint": "All attached bonds share the same effort (voltage / force / pressure / temperature); flows sum to zero."},
        {"key": "J1", "type": "1", "domain": "any", "name": "1-junction (common flow)",
         "hint": "All attached bonds share the same flow (current / velocity); efforts sum to zero."},
        {"key": "GY", "type": "GY", "domain": "any", "name": "Gyrator (GY)", "param": {"label": "k", "unit": "-", "default": 0.05},
         "hint": "2-port. Swaps effort and flow across domains (e.g. motor: voltage<->torque). Must join two junctions."},
        {"key": "TF", "type": "TF", "domain": "any", "name": "Transformer (TF)", "param": {"label": "n", "unit": "-", "default": 2.0},
         "hint": "2-port. Scales effort and flow (gear, lever, transformer). Convention here: e1 = n*e2. Must join two junctions."},
        {"key": "MSe", "type": "MSe", "domain": "any", "name": "Modulated effort source",
         "hint": "Effort set by a formula of a signal read from a junction (one-way coupling, e.g. temperature -> thermal-stress force)."},
        {"key": "MSf", "type": "MSf", "domain": "any", "name": "Modulated flow source",
         "hint": "Flow set by a formula of a signal read from a junction."},
    ]
    for dk, d in DOMAINS.items():
        if dk == "generic":
            continue
        for t in ("Se", "Sf", "R", "C", "I"):
            spec = d.get(t)
            if not spec:
                continue
            e = {"key": f"{dk}:{t}", "type": t, "domain": dk, "name": spec[0]}
            if t in ("Se", "Sf"):
                e["param"] = {"label": spec[1], "unit": d["effort"][1] if t == "Se" else d["flow"][1], "default": 1.0, "is_source": True}
            else:
                e["param"] = {"label": spec[1], "unit": spec[2], "default": spec[3]}
            items.append(e)
    return {"domains": {k: {"label": v["label"], "effort": v["effort"], "flow": v["flow"], "disp": v["disp"]}
                        for k, v in DOMAINS.items()},
            "items": items}


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
def _n(id_, type_, domain, label, **kw):
    d = {"id": id_, "type": type_, "domain": domain, "label": label}
    d.update(kw)
    return d


def _e(a, b, signal=False):
    return {"from": a, "to": b, "signal": signal}


def _pos(**kw):
    return {k: {"x": v[0], "y": v[1]} for k, v in kw.items()}


TEMPLATES = {}


def _add(tid, name, description, domains, nodes, edges, layout, values, sources, initial, sim):
    TEMPLATES[tid] = {
        "meta": {"id": tid, "name": name, "description": description, "domains": domains, "version": 1},
        "spec": {"nodes": nodes, "edges": edges},
        "layout": layout, "values": values, "sources": sources, "initial": initial, "sim": sim,
    }


_add("rlc_series", "Series RLC circuit", "Voltage step into R, L and C in series. Underdamped ringing.",
     ["electrical"],
     [_n("Se1", "Se", "electrical", "V(t)", expr_param_name="V"), _n("J1", "1", "electrical", "1"),
      _n("R1", "R", "electrical", "R", param_symbol_name="R"), _n("L1", "I", "electrical", "L", param_symbol_name="L"),
      _n("C1", "C", "electrical", "C", param_symbol_name="C")],
     [_e("Se1", "J1"), _e("J1", "R1"), _e("J1", "L1"), _e("J1", "C1")],
     _pos(Se1=(0, 130), J1=(170, 130), R1=(170, 0), L1=(340, 130), C1=(170, 260)),
     {"R": 5.0, "L": 0.1, "C": 1e-4},
     {"V": {"type": "step", "amplitude": 5.0, "t0": 0.0, "offset": 0.0}}, {}, {"t_end": 0.1, "n": 600})

_add("rlc_parallel", "Parallel RLC circuit", "Current step into R, C and L in parallel.",
     ["electrical"],
     [_n("Sf1", "Sf", "electrical", "I_in(t)", expr_param_name="I_in"), _n("J0", "0", "electrical", "0"),
      _n("R1", "R", "electrical", "R", param_symbol_name="R"), _n("C1", "C", "electrical", "C", param_symbol_name="C"),
      _n("L1", "I", "electrical", "L", param_symbol_name="L")],
     [_e("Sf1", "J0"), _e("J0", "R1"), _e("J0", "C1"), _e("J0", "L1")],
     _pos(Sf1=(0, 130), J0=(170, 130), R1=(170, 0), C1=(340, 130), L1=(170, 260)),
     {"R": 50.0, "C": 1e-4, "L": 0.1},
     {"I_in": {"type": "step", "amplitude": 1.0, "t0": 0.0, "offset": 0.0}}, {}, {"t_end": 0.05, "n": 600})

_add("mass_spring_damper", "Mass-spring-damper", "Force step on a mass with spring and damper (one degree of freedom).",
     ["mech_trans"],
     [_n("Se1", "Se", "mech_trans", "F(t)", expr_param_name="F"), _n("J1", "1", "mech_trans", "1"),
      _n("M1", "I", "mech_trans", "mass m", param_symbol_name="m"),
      _n("K1", "C", "mech_trans", "spring (C=1/k)", param_symbol_name="C_k"),
      _n("D1", "R", "mech_trans", "damper c", param_symbol_name="c")],
     [_e("Se1", "J1"), _e("J1", "M1"), _e("J1", "K1"), _e("J1", "D1")],
     _pos(Se1=(0, 130), J1=(170, 130), M1=(170, 0), K1=(340, 130), D1=(170, 260)),
     {"m": 1.0, "C_k": 0.01, "c": 2.0},
     {"F": {"type": "step", "amplitude": 1.0, "t0": 0.0, "offset": 0.0}}, {}, {"t_end": 4.0, "n": 600})

_add("thermal_rc", "Thermal RC (heating a mass)", "A hot reservoir heats a thermal capacitance through a thermal resistance.",
     ["thermal"],
     [_n("Se1", "Se", "thermal", "T_src", expr_param_name="T_src"), _n("J1", "1", "thermal", "1"),
      _n("R1", "R", "thermal", "R_th", param_symbol_name="R_th"), _n("C1", "C", "thermal", "C_th", param_symbol_name="C_th")],
     [_e("Se1", "J1"), _e("J1", "R1"), _e("J1", "C1")],
     _pos(Se1=(0, 120), J1=(170, 120), R1=(170, 0), C1=(340, 120)),
     {"R_th": 0.5, "C_th": 500.0},
     {"T_src": {"type": "constant", "value": 373.0}}, {"C1": 293.0}, {"t_end": 1000.0, "n": 500})

_add("dc_motor", "DC motor (electrical to mechanical)",
     "Multi-physics: armature circuit coupled to a rotor through a gyrator. Textbook: L di/dt = V - R i - k w ; J dw/dt = k i - b w.",
     ["electrical", "mech_rot"],
     [_n("Se1", "Se", "electrical", "V(t)", expr_param_name="V"), _n("Je", "1", "electrical", "1"),
      _n("Rw", "R", "electrical", "R_w", param_symbol_name="R_w"), _n("Lw", "I", "electrical", "L_w", param_symbol_name="L_w"),
      _n("GY1", "GY", "electrical", "motor k", param_symbol_name="k_m"),
      _n("Jm", "1", "mech_rot", "1"),
      _n("Jr", "I", "mech_rot", "rotor J", param_symbol_name="J_r"), _n("Bf", "R", "mech_rot", "friction b", param_symbol_name="b_f")],
     [_e("Se1", "Je"), _e("Je", "Rw"), _e("Je", "Lw"), _e("Je", "GY1"), _e("GY1", "Jm"), _e("Jm", "Jr"), _e("Jm", "Bf")],
     _pos(Se1=(0, 130), Je=(170, 130), Rw=(170, 0), Lw=(170, 260), GY1=(340, 130), Jm=(510, 130), Jr=(510, 0), Bf=(510, 260)),
     {"R_w": 1.0, "L_w": 5e-4, "k_m": 0.05, "J_r": 1e-4, "b_f": 1e-5},
     {"V": {"type": "step", "amplitude": 12.0, "t0": 0.0, "offset": 0.0}}, {}, {"t_end": 0.4, "n": 600})

_add("geared_load", "Geared load (transformer)",
     "Torque source drives a damped inertia through an ideal gear. Convention: input torque = n * load torque.",
     ["mech_rot"],
     [_n("Se1", "Se", "mech_rot", "T_in", expr_param_name="T_in"), _n("J1", "1", "mech_rot", "1"),
      _n("TF1", "TF", "mech_rot", "gear n", param_symbol_name="n_g"), _n("J2", "1", "mech_rot", "1"),
      _n("I2", "I", "mech_rot", "load J", param_symbol_name="J_load"), _n("R2", "R", "mech_rot", "load b", param_symbol_name="b_load")],
     [_e("Se1", "J1"), _e("J1", "TF1"), _e("TF1", "J2"), _e("J2", "I2"), _e("J2", "R2")],
     _pos(Se1=(0, 130), J1=(170, 130), TF1=(340, 130), J2=(510, 130), I2=(510, 0), R2=(510, 260)),
     {"n_g": 2.0, "J_load": 0.01, "b_load": 0.05},
     {"T_in": {"type": "step", "amplitude": 1.0, "t0": 0.0, "offset": 0.0}}, {}, {"t_end": 1.5, "n": 600})

_add("brake_thermal_structural", "Disc brake: heat -> thermal stress -> vibration",
     "Multi-physics, one-way coupled. Friction heat warms the disc (thermal RC); temperature drives a thermal-stress force on a mass-spring-damper. Reduced version of the course project's Path A model.",
     ["thermal", "mech_trans"],
     [_n("Q1", "Sf", "thermal", "friction heat", expr_param_name="Q_heat"), _n("Jd", "0", "thermal", "disc T"),
      _n("Cd", "C", "thermal", "C_d", param_symbol_name="C_d"), _n("Jr", "1", "thermal", "1"),
      _n("Rc", "R", "thermal", "R_c", param_symbol_name="R_c"), _n("Ta", "Se", "thermal", "T_amb", expr_param_name="T_amb"),
      _n("Fth", "MSe", "mech_trans", "thermal stress F", signal_from="Jd", signal_kind="effort",
         modulation_formula="k_th*(SIGNAL - T_ref)"),
      _n("Js", "1", "mech_trans", "1"), _n("Ms", "I", "mech_trans", "m_eff", param_symbol_name="m_eff"),
      _n("Ks", "C", "mech_trans", "C_k=1/k", param_symbol_name="C_k"), _n("Ds", "R", "mech_trans", "c_s", param_symbol_name="c_s")],
     [_e("Q1", "Jd"), _e("Jd", "Cd"), _e("Jd", "Jr"), _e("Jr", "Rc"), _e("Jr", "Ta"),
      _e("Fth", "Js"), _e("Js", "Ms"), _e("Js", "Ks"), _e("Js", "Ds"), _e("Jd", "Fth", signal=True)],
     _pos(Q1=(0, 130), Jd=(170, 130), Cd=(170, 0), Jr=(340, 130), Rc=(340, 0), Ta=(510, 130),
          Fth=(170, 330), Js=(340, 330), Ms=(340, 460), Ks=(510, 330), Ds=(170, 460)),
     {"C_d": 3000.0, "R_c": 0.108, "T_ref": 293.0, "k_th": 1.815, "m_eff": 1.5, "C_k": 2.533e-6, "c_s": 46.2},
     {"Q_heat": {"type": "pwl", "points": [[0, 85500.0], [5, 0.0], [60, 0.0]]},
      "T_amb": {"type": "constant", "value": 293.0}},
     {"Cd": 293.0}, {"t_end": 60.0, "n": 600})


_add("hydraulic_tank", "Hydraulic tank (RC charging)",
     "A pressure source charges a tank through a valve resistance. Same math as the thermal RC, in a new domain: "
     "p(t) = P_in*(1 - exp(-t/(R_h*C_h))).",
     ["hydraulic"],
     [_n("Se1", "Se", "hydraulic", "P_in(t)", expr_param_name="P_in"), _n("J1", "1", "hydraulic", "1"),
      _n("R1", "R", "hydraulic", "valve R_h", param_symbol_name="R_h"), _n("C1", "C", "hydraulic", "tank C_h", param_symbol_name="C_h")],
     [_e("Se1", "J1"), _e("J1", "R1"), _e("J1", "C1")],
     _pos(Se1=(0, 120), J1=(170, 120), R1=(170, 0), C1=(340, 120)),
     {"R_h": 1e5, "C_h": 1e-7},
     {"P_in": {"type": "step", "amplitude": 2e5, "t0": 0.0, "offset": 0.0}}, {"C1": 0.0}, {"t_end": 0.05, "n": 400})

_add("resistive_heater", "Resistive heater (electrical to thermal)",
     "Multi-physics, one-way coupled. A voltage source drives current through a heating resistor; the I^2*R "
     "power dissipated becomes a heat-flow input (a modulated flow source, MSf, reading the electrical current) "
     "into a thermal RC that loses heat to ambient. Textbook check: steady state T = T_amb + R_th*V^2/R_elec.",
     ["electrical", "thermal"],
     [_n("Se1", "Se", "electrical", "V(t)", expr_param_name="V"), _n("J1e", "1", "electrical", "1"),
      _n("Re1", "R", "electrical", "heater R_elec", param_symbol_name="R_elec"),
      _n("Qgen", "MSf", "thermal", "I^2 R heat", signal_from="J1e", signal_kind="flow", modulation_formula="R_elec*SIGNAL**2"),
      _n("Jd", "0", "thermal", "mass T"), _n("Cd", "C", "thermal", "thermal mass C_th", param_symbol_name="C_th"),
      _n("Jr", "1", "thermal", "1"), _n("Rc", "R", "thermal", "loss R_th", param_symbol_name="R_th"),
      _n("Ta", "Se", "thermal", "T_amb", expr_param_name="T_amb")],
     [_e("Se1", "J1e"), _e("J1e", "Re1"), _e("Qgen", "Jd"), _e("Jd", "Cd"), _e("Jd", "Jr"),
      _e("Jr", "Rc"), _e("Jr", "Ta"), _e("J1e", "Qgen", signal=True)],
     _pos(Se1=(0, 0), J1e=(170, 0), Re1=(170, -140), Qgen=(170, 250), Jd=(340, 250), Cd=(340, 120),
          Jr=(510, 250), Rc=(510, 120), Ta=(680, 250)),
     {"R_elec": 2.0, "C_th": 200.0, "R_th": 0.4},
     {"V": {"type": "step", "amplitude": 10.0, "t0": 0.0, "offset": 0.0}, "T_amb": {"type": "constant", "value": 293.0}},
     {"Cd": 293.0}, {"t_end": 400.0, "n": 500})

_add("rack_pinion", "Rack and pinion (rotation to translation)",
     "Multi-physics: a torque source drives a mass and damper through an ideal rack-and-pinion (a transformer, TF, "
     "between the mechanical-rotation and mechanical-translation domains: torque = r*force, velocity = r*angular velocity). "
     "Same structure as the geared-load template, but the output side changes domain rather than just scale.",
     ["mech_rot", "mech_trans"],
     [_n("Se1", "Se", "mech_rot", "T_in", expr_param_name="T_in"), _n("J1", "1", "mech_rot", "1"),
      _n("TF1", "TF", "mech_rot", "pinion r", param_symbol_name="r_p"), _n("J2", "1", "mech_trans", "1"),
      _n("M1", "I", "mech_trans", "load mass", param_symbol_name="m_load"),
      _n("D1", "R", "mech_trans", "friction", param_symbol_name="c_load")],
     [_e("Se1", "J1"), _e("J1", "TF1"), _e("TF1", "J2"), _e("J2", "M1"), _e("J2", "D1")],
     _pos(Se1=(0, 130), J1=(170, 130), TF1=(340, 130), J2=(510, 130), M1=(510, 0), D1=(510, 260)),
     {"r_p": 0.1, "m_load": 2.0, "c_load": 4.0},
     {"T_in": {"type": "step", "amplitude": 1.0, "t0": 0.0, "offset": 0.0}}, {}, {"t_end": 3.0, "n": 400})

_add("two_tank_hydraulic", "Two-tank hydraulic system",
     "Fluid flows into tank 1, through a connecting pipe resistance into tank 2, and out through an outlet "
     "resistance to atmosphere. A cascaded 2-state hydraulic network (two coupled RC stages), the hydraulic twin "
     "of two resistor-capacitor stages in series.",
     ["hydraulic"],
     [_n("Sf1", "Sf", "hydraulic", "Q_in(t)", expr_param_name="Q_in"), _n("J0_1", "0", "hydraulic", "tank 1"),
      _n("C1", "C", "hydraulic", "tank1 C1_h", param_symbol_name="C1_h"), _n("J1_12", "1", "hydraulic", "1"),
      _n("R12", "R", "hydraulic", "pipe R12_h", param_symbol_name="R12_h"), _n("J0_2", "0", "hydraulic", "tank 2"),
      _n("C2", "C", "hydraulic", "tank2 C2_h", param_symbol_name="C2_h"), _n("J1o", "1", "hydraulic", "1"),
      _n("R2", "R", "hydraulic", "outlet R2_h", param_symbol_name="R2_h"), _n("Se1", "Se", "hydraulic", "atmosphere", expr_param_name="P_atm")],
     [_e("Sf1", "J0_1"), _e("J0_1", "C1"), _e("J0_1", "J1_12"), _e("J1_12", "R12"), _e("J1_12", "J0_2"),
      _e("J0_2", "C2"), _e("J0_2", "J1o"), _e("J1o", "R2"), _e("J1o", "Se1")],
     _pos(Sf1=(0, 130), J0_1=(170, 130), C1=(170, 0), J1_12=(340, 130), R12=(340, 0),
          J0_2=(510, 130), C2=(510, 0), J1o=(680, 130), R2=(680, 0), Se1=(850, 130)),
     {"C1_h": 1e-6, "R12_h": 5e5, "C2_h": 1e-6, "R2_h": 1e6},
     {"Q_in": {"type": "step", "amplitude": 1e-4, "t0": 0.0, "offset": 0.0}, "P_atm": {"type": "constant", "value": 0.0}},
     {"C1": 0.0, "C2": 0.0}, {"t_end": 6.0, "n": 500})


def list_templates():
    return [t["meta"] for t in TEMPLATES.values()]


_ID = re.compile(r"^[a-z0-9_]+$")


def get_template(tid):
    if not _ID.match(tid or ""):
        return None
    return TEMPLATES.get(tid)
