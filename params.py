"""
Shared physical/numeric parameters for the disc-brake thermal-structural model.

All values are literature-typical order-of-magnitude figures for a
passenger-car front disc brake (cast-iron vented disc, semi-metallic pad).
See manual_derivation.md section 4 for the full derivation and citation
notes. These are illustrative defaults for a course-project demo -- replace
with a specifically-cited source if your report requires it.

Every other script (bond_graph.py, equation_gen.py, validate.py,
pinn_model.py, compare.py) imports numeric values from PARAMS so there is a
single source of truth.
"""
import math

# ---- Braking event (prescribed kinematic input, not solved for) ----
v0 = 25.0              # m/s, initial sliding speed (~90 km/h)
a = 5.0                # m/s^2, deceleration (~0.5 g, firm braking)
t_stop = v0 / a         # s, time to stop (= 5.0 s)

# ---- Friction / heat generation: q(t) = mu * F_cl * v(t) ----
mu = 0.38               # [-] friction coefficient, semi-metallic pad / cast iron
F_cl = 10000.0          # N, caliper clamp force
gamma = 0.90            # [-] fraction of frictional heat absorbed by the disc
                         # (disc thermal effusivity >> pad's -> most heat goes to disc)

# ---- Thermal capacitances: C = m * c ----
m_d = 6.0               # kg, disc mass (vented cast-iron front rotor)
c_d = 500.0             # J/(kg K), cast iron specific heat (~460-540 typical)
C_d = m_d * c_d          # J/K

m_p = 0.5               # kg, pad mass (both pads)
c_p = 1000.0            # J/(kg K), friction material specific heat
C_p = m_p * c_p          # J/K

# ---- Thermal resistances ----
h = 80.0                # W/(m^2 K), disc convection coeff. (forced, disc in motion)
A_d = 0.10               # m^2, disc convective area (both faces + vents, order-of-magnitude)
R2 = 1.0 / (h * A_d)      # K/W, disc -> ambient convection
R1 = 0.8                 # K/W, disc -> hub conduction (engineering estimate)
R3 = 3.0                 # K/W, pad -> caliper/ambient (conduction + convection lumped)

# ---- Ambient / reference ----
T_amb = 293.0            # K (20 C)
T_ref = T_amb             # K, stress-free reference temperature

# ---- Thermal -> structural coupling ----
alpha = 11e-6             # 1/K, cast iron thermal expansion coefficient
E = 110e9                 # Pa, cast iron Young's modulus
A_eff = 1.5e-6              # m^2, effective coupling area -- NOT a real geometric
                           # area. Tuned (not derived) so the single-DOF proxy's
                           # peak displacement lands in a physically plausible
                           # range (~O(100 um), typical order for disc thermal
                           # coning/judder) given E, alpha and k_struct above.
                           # See manual_derivation.md Section 6, simplification #4:
                           # this whole coupling is a lumped proxy, not a validated
                           # structural prediction.

# ---- Structural (lumped single-DOF mode, e.g. disc coning/warp) ----
m_eff = 1.5                # kg, effective modal mass
f_n = 100.0                 # Hz, assumed structural natural frequency
k_struct = m_eff * (2 * math.pi * f_n) ** 2   # N/m
zeta = 0.03                  # [-] damping ratio (lightly damped metal structure)
c_struct = 2 * zeta * math.sqrt(k_struct * m_eff)  # N.s/m

# Convenience dict for programmatic substitution (equation_gen / validate / pinn)
PARAMS = {
    "v0": v0, "a": a, "t_stop": t_stop,
    "mu": mu, "F_cl": F_cl, "gamma": gamma,
    "C_d": C_d, "C_p": C_p,
    "R1": R1, "R2": R2, "R3": R3,
    "T_amb": T_amb, "T_ref": T_ref,
    "alpha": alpha, "E": E, "A_eff": A_eff,
    "m_eff": m_eff, "k_struct": k_struct, "c_struct": c_struct,
}


def v_of_t(t_val):
    """Numeric prescribed sliding-velocity profile v(t) [m/s]."""
    v = v0 - a * t_val if t_val <= t_stop else 0.0
    return max(v, 0.0)


def q_total(t_val):
    """Numeric prescribed friction heat generation rate q(t) = mu*F_cl*v(t) [W]."""
    return mu * F_cl * v_of_t(t_val)


if __name__ == "__main__":
    print("Derived quantities:")
    print(f"  t_stop   = {t_stop:.3f} s")
    print(f"  C_d      = {C_d:.1f} J/K")
    print(f"  C_p      = {C_p:.1f} J/K")
    print(f"  R2       = {R2:.4f} K/W")
    print(f"  k_struct = {k_struct:.4g} N/m")
    print(f"  c_struct = {c_struct:.4g} N.s/m")
    print(f"  q(0)     = {q_total(0.0):.1f} W  (peak heat generation)")
    print()
    print("Full PARAMS dict:")
    for k, val in PARAMS.items():
        print(f"  {k:10s} = {val:.6g}")
