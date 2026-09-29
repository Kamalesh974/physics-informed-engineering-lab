"""
param_usage_check.py -- static check: does every declared physical
parameter actually appear somewhere in the final derived equations?

RATIONALE: a parameter that's declared (given a numeric value in a PARAMS
dict, or registered as a symbol name while parsing an LLM-read diagram) but
never appears in any state ODE or readout is a red flag -- it usually means
part of the topology that was SUPPOSED to use it never got wired into the
graph the resolver actually walks (e.g. an ambient-loss R+Se branch that
was declared but left disconnected). SCAP resolving with zero conflicts
does NOT check this: SCAP only verifies causal well-posedness (every C/I
gets integral causality, every R's causality is determined), which says
nothing about whether every declared element actually ended up mattering
to the final equations. This is a cheap, generic, no-API-cost check that
complements SCAP rather than replacing it -- see sanity_check_live_run.py
for a live example of a bug this class of check is aimed at (a missing
ambient heat-loss path that let T_d run away unbounded).
"""
import sympy as sp


def find_unused_params(odes, declared_names, extra_exprs=None, ignore=None):
    """
    odes: dict state_symbol -> sympy expr (the derived state-derivative RHS's)
    declared_names: iterable of parameter name strings declared somewhere
        in the model (e.g. params.PARAMS.keys(), or an LLM-built
        symbol_table's keys())
    extra_exprs: optional iterable of additional sympy exprs to also scan
        (e.g. readout quantities like sigma_th, which might use a
        parameter that never appears in a *state* ODE but does appear in a
        derived diagnostic quantity)
    ignore: names to exclude from the check outright

    Returns {"used": set, "unused": set} (both subsets of declared_names).
    """
    ignore = set(ignore or ())
    used = set()
    exprs = list(odes.values())
    if extra_exprs:
        exprs += list(extra_exprs)
    for expr in exprs:
        if not hasattr(expr, "free_symbols"):
            continue
        for s in expr.free_symbols:
            used.add(str(s))
    declared = set(declared_names) - ignore
    return {"used": declared & used, "unused": declared - used}


def report(result, label=""):
    lines = []
    header = f"Unused-parameter check{f' ({label})' if label else ''}"
    lines.append(header)
    lines.append("-" * len(header))
    if not result["unused"]:
        lines.append(f"  OK -- all {len(result['used'])} declared parameters appear in the final equations.")
    else:
        lines.append(f"  WARNING -- {len(result['unused'])} declared parameter(s) never appear in any "
                      f"final ODE or readout (possible dropped coupling / dead/disconnected element):")
        for name in sorted(result["unused"]):
            lines.append(f"    - {name}")
        lines.append(f"  ({len(result['used'])} parameter(s) confirmed used.)")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    import params
    from equation_gen import generate_odes

    result = generate_odes()
    res = find_unused_params(result["odes"], params.PARAMS.keys(),
                              extra_exprs=list(result["readouts"].values()))
    report(res, label="Path A (disc brake, original model)")
    print()

    import params_image_model as pim
    from equation_gen_from_image import generate_odes_from_image

    result_b = generate_odes_from_image()
    res_b = find_unused_params(result_b["odes"], pim.PARAMS.keys(),
                                extra_exprs=list(result_b["readouts"].values()))
    report(res_b, label="Path B (image-derived model)")
