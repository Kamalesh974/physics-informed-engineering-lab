"""
scap_causality.py -- Sequential Causality Assignment Procedure (SCAP)

This is the core automation step of the pipeline: causality (which port
variable, effort or flow, is "given" vs. "computed" on each bond) is
derived automatically from the graph structure via constraint propagation,
rather than being hand-assigned. equation_gen.py then just walks whatever
causality this script produces -- it never encodes disc-brake-specific
knowledge.

CAUSALITY REPRESENTATION
For every power bond (u, v) we store a single attribute:
    G.edges[u, v]['effort_giver'] = <node id>
meaning: that node supplies EFFORT onto the bond; the node at the other end
receives that effort as an input and must supply FLOW back. A bond has
exactly one effort-giver and one flow-giver (the two different ends), so
this single field fully determines causality -- no separate "stroke
position" bookkeeping is needed.

FIXED / PREFERRED CAUSALITY RULES (Step 1-2 of SCAP)
  Se / MSe (effort source): ALWAYS gives effort  -> effort_giver = source node
  Sf / MSf (flow source):   ALWAYS gives flow     -> effort_giver = neighbor
  C  (preferred: integral causality): effort_giver = the C node itself
       (C takes flow IN, integrates it to state q; outputs effort e = q/C)
  I  (preferred: integral causality): effort_giver = the neighbor
       (I takes effort IN, integrates it to state p; outputs flow f = p/I)
  R: no fixed preference -- its causality is whatever the junction
     constraints leave over.

JUNCTION CONSTRAINTS (Step 3 of SCAP, propagated to a fixed point)
  0-junction (common effort, flows sum to zero): exactly ONE incident bond
      may have effort_giver = the OUTSIDE neighbor (that one supplies the
      shared effort value into the junction); every other incident bond
      must have effort_giver = the junction itself (the junction hands the
      already-known effort back out to them).
  1-junction (common flow, efforts sum to zero): exactly ONE incident bond
      may have effort_giver = the JUNCTION itself (that bond's outside
      element is the one supplying the shared flow value, so the junction
      hands effort back to it); every other incident bond must have
      effort_giver = the outside neighbor.
  These are dual (0<->1, effort<->flow) forms of the same rule -- standard
  bond-graph junction causality (Karnopp, Margolis & Rosenberg, *System
  Dynamics: Modeling, Simulation, and Control of Mechatronic Systems*).

This module implements junction propagation generically (it only looks at
node `type` and the rules above), so the SAME code would correctly assign
causality on a different bond graph too -- only bond_graph.py is specific
to the disc brake.

LIMITATION (flagged): this implementation detects causality CONFLICTS
(over/under-determined junctions, which would mean some C or I element
cannot get its preferred integral causality -- "derivative causality") and
reports them loudly, but does not automatically resolve them by flipping a
C/I element's causality and retrying. For this project's specific topology
that never happens (verified: every bond resolves with zero conflicts, see
`if __name__ == "__main__"` output), but if you change the topology and hit
a reported conflict, that element's causality would need to be flipped and
re-derived by hand -- exactly the manual fallback the project brief already
anticipated for SCAP.
"""
from bond_graph import build_bond_graph


def _incident_power_bonds(G, node):
    """All incident power (non-signal) bonds of `node`, regardless of the
    edge's stored (u, v) direction. Returns list of (edge_key, other_node)."""
    out = []
    for u, v, d in G.in_edges(node, data=True):
        if not d["signal"]:
            out.append(((u, v), u))
    for u, v, d in G.out_edges(node, data=True):
        if not d["signal"]:
            out.append(((u, v), v))
    return out


def assign_causality(G):
    """Mutate G in place: set edge['effort_giver'] for every power bond.

    Returns a report dict: {'conflicts': [...], 'passes': int, 'unresolved': [...]}.
    An empty 'conflicts' + 'unresolved' list means SCAP fully and
    consistently resolved the graph.
    """
    conflicts = []

    def set_giver(edge_key, giver):
        u, v = edge_key
        existing = G.edges[u, v].get("effort_giver")
        if existing is not None and existing != giver:
            conflicts.append(
                f"Conflict on bond {G.edges[u, v]['bond_id']} ({u}-{v}): "
                f"tried effort_giver={giver}, already {existing}"
            )
            return
        G.edges[u, v]["effort_giver"] = giver

    # ---- Step 1: fixed source causality ----
    for n, d in G.nodes(data=True):
        if d["type"] in ("Se", "MSe"):
            for ek, _other in _incident_power_bonds(G, n):
                set_giver(ek, n)
        elif d["type"] in ("Sf", "MSf"):
            for ek, other in _incident_power_bonds(G, n):
                set_giver(ek, other)

    # ---- Step 2: preferred integral causality for C / I ----
    for n, d in G.nodes(data=True):
        if d["type"] == "C":
            for ek, _other in _incident_power_bonds(G, n):
                set_giver(ek, n)
        elif d["type"] == "I":
            for ek, other in _incident_power_bonds(G, n):
                set_giver(ek, other)

    # ---- Step 3: propagate junction constraints to a fixed point ----
    def propagate():
        """Runs the constraint-propagation passes until nothing changes.
        Returns (passes, stuck_conflicts) -- stuck_conflicts is populated
        only by genuine over/under-determination, not by plain unresolved
        bonds (those are reported by the caller after tie-breaking, if
        tie-breaking doesn't clear them either)."""
        local_conflicts = []
        changed = True
        n_passes = 0
        while changed and n_passes < 50:
            changed = False
            n_passes += 1
            for n, d in G.nodes(data=True):
                if d["type"] not in ("0", "1"):
                    continue
                incident = _incident_power_bonds(G, n)
                if not incident:
                    continue
                assigned = [(ek, o) for ek, o in incident if G.edges[ek].get("effort_giver") is not None]
                unassigned = [(ek, o) for ek, o in incident if G.edges[ek].get("effort_giver") is None]

                def is_special(giver, other, jtype=d["type"], jnode=n):
                    return (giver == other) if jtype == "0" else (giver == jnode)

                special_count = sum(1 for ek, o in assigned if is_special(G.edges[ek]["effort_giver"], o))

                if special_count > 1:
                    local_conflicts.append(f"Junction {n} ({d['type']}-junction) over-determined: "
                                            f"{special_count} bonds want the special role")
                    continue

                # GY/TF bonds are EXCLUDED from this junction's own generic
                # "default the rest to the outside neighbor" rule -- a 2-port
                # element's causality must be resolved by ITS OWN dedicated
                # propagation block below (coordinated across BOTH of its
                # bonds, which may sit on two DIFFERENT junctions), not by
                # each junction independently defaulting its own local view
                # of the bond. For GY this accidentally caused no visible
                # harm (its e1=r*f2 law naturally alternates effort/flow, so
                # two independently-defaulted ports don't create an infinite
                # loop) but for TF (e1=n*e2, effort-to-effort) an
                # independent default on BOTH ports directly created a
                # genuine unresolvable causal loop -- found by the TF
                # validation test. Excluding both here (not just TF) is the
                # principled fix rather than relying on GY's law happening
                # to tolerate the bug.
                default_unassigned = [(ek, o) for ek, o in unassigned if G.nodes[o]["type"] not in ("GY", "TF")]

                if special_count == 1 and default_unassigned:
                    for ek, other in default_unassigned:
                        giver = n if d["type"] == "0" else other
                        set_giver(ek, giver)
                        changed = True
                elif special_count == 0 and len(unassigned) == 1 and G.nodes[unassigned[0][1]]["type"] not in ("GY", "TF"):
                    ek, other = unassigned[0]
                    giver = other if d["type"] == "0" else n
                    set_giver(ek, giver)
                    changed = True

            # ---- GY (gyrator) 2-port causality propagation ----
            # A gyrator's constitutive law (e1 = r*f2, e2 = r*f1) forces
            # BOTH ports to have the SAME causal role -- unlike a junction,
            # there's no "special" bond; either GY gives effort on BOTH
            # bonds (each computed from the OTHER port's flow) or GY
            # receives effort on BOTH (each port's flow computed from the
            # OTHER port's effort). This is the standard bond-graph result
            # that a gyrator "swaps" causality type between its ports,
            # while a transformer (TF) would pass it through unchanged.
            for n, d in G.nodes(data=True):
                if d["type"] != "GY":
                    continue
                bonds = _incident_power_bonds(G, n)
                if len(bonds) != 2:
                    conflicts.append(f"{n}: GY must have exactly 2 incident power bonds, found {len(bonds)}")
                    continue
                (ek_a, _), (ek_b, _) = bonds
                giver_a = G.edges[ek_a].get("effort_giver")
                giver_b = G.edges[ek_b].get("effort_giver")
                if giver_a is not None and giver_b is None:
                    role_a_is_gy = (giver_a == n)
                    _, other_b = bonds[1]
                    set_giver(ek_b, n if role_a_is_gy else other_b)
                    changed = True
                elif giver_b is not None and giver_a is None:
                    role_b_is_gy = (giver_b == n)
                    _, other_a = bonds[0]
                    set_giver(ek_a, n if role_b_is_gy else other_a)
                    changed = True

            # ---- TF (transformer) 2-port causality propagation ----
            # A transformer's constitutive law (e1 = n*e2, f2 = n*f1) PASSES
            # causality straight through, the opposite of GY's "swap": exactly
            # ONE port has effort_giver = TF itself (TF computes that port's
            # effort from the OTHER port's effort, which must therefore be
            # externally given) -- there's no "special" bond in the
            # junction sense, just "whichever port isn't the TF-given one is
            # the externally-given one." bond_graph.py's port-order
            # convention (see equation_gen.py's TF dispatch) is simply the
            # order the two edges were added to the graph -- e1/f1 refers to
            # the first incident edge found by _incident_power_bonds, e2/f2
            # the second; this is an implementation convention, not a
            # physical port-1-vs-port-2 label, so it only matters that
            # causality assignment and equation derivation agree on it
            # (both call the same _incident_power_bonds on the same graph).
            for n, d in G.nodes(data=True):
                if d["type"] != "TF":
                    continue
                bonds = _incident_power_bonds(G, n)
                if len(bonds) != 2:
                    conflicts.append(f"{n}: TF must have exactly 2 incident power bonds, found {len(bonds)}")
                    continue
                (ek_a, _), (ek_b, _) = bonds
                giver_a = G.edges[ek_a].get("effort_giver")
                giver_b = G.edges[ek_b].get("effort_giver")
                if giver_a is not None and giver_b is None:
                    role_a_is_tf = (giver_a == n)
                    _, other_b = bonds[1]
                    set_giver(ek_b, other_b if role_a_is_tf else n)
                    changed = True
                elif giver_b is not None and giver_a is None:
                    role_b_is_tf = (giver_b == n)
                    _, other_a = bonds[0]
                    set_giver(ek_a, other_a if role_b_is_tf else n)
                    changed = True
        return n_passes, local_conflicts

    total_passes = 0
    for _ in range(20):  # outer loop: alternate propagation with R-indifference tie-breaks
        n_passes, prop_conflicts = propagate()
        total_passes += n_passes
        conflicts.extend(prop_conflicts)

        unresolved_bonds = [(u, v) for u, v, d in G.edges(data=True)
                             if not d["signal"] and d.get("effort_giver") is None]
        if not unresolved_bonds or prop_conflicts:
            break

        # ---- R-indifference tie-break (standard SCAP practice): a junction
        # can get stuck with special_count==0 and >1 unassigned bonds when
        # ALL of them are plain R's (no causal preference of their own) --
        # e.g. two resistors in series between two already-causal C's. Any
        # one of them can validly take the "special" role; the resulting
        # physics is identical (see conversation for the worked example).
        # Break the tie deterministically (first candidate) and let
        # propagation continue. This never fires for a junction that has a
        # genuine unique resolution, so it changes nothing for graphs that
        # already resolve cleanly (e.g. the original disc-brake model).
        tie_broken = False
        for n, d in G.nodes(data=True):
            if d["type"] not in ("0", "1"):
                continue
            incident = _incident_power_bonds(G, n)
            assigned = [(ek, o) for ek, o in incident if G.edges[ek].get("effort_giver") is not None]
            unassigned = [(ek, o) for ek, o in incident if G.edges[ek].get("effort_giver") is None]
            if len(unassigned) < 2:
                continue

            def is_special(giver, other, jtype=d["type"], jnode=n):
                return (giver == other) if jtype == "0" else (giver == jnode)

            special_count = sum(1 for ek, o in assigned if is_special(G.edges[ek]["effort_giver"], o))
            if special_count != 0:
                continue
            if not all(G.nodes[o]["type"] == "R" for _ek, o in unassigned):
                continue  # only break ties among plain, preference-free R's
            ek, other = unassigned[0]
            giver = n if d["type"] == "0" else other
            set_giver(ek, giver)
            tie_broken = True
            break  # re-run propagation after each single tie-break

        # ---- GY/TF seed tie-break: since both bonds of a 2-port element
        # are now EXCLUDED from each junction's own generic default (see
        # above), a GY/TF whose two bonds sit at junctions that never
        # independently force its causality can reach a stall with BOTH of
        # its bonds unassigned.
        #
        # IMPORTANT (found by the TF validation test): seeding ARBITRARILY
        # (e.g. always "this element gives on its first bond") is WRONG in
        # general -- if ONE of the two endpoint junctions has its OWN
        # single-remaining-bond constraint (special_count==0, this is its
        # only unassigned bond -- the same condition the generic per-
        # junction default would normally act on), that junction's role for
        # this bond is NOT a free choice, it's FORCED, and picking the
        # other role instead makes the OTHER port's causality (set next
        # pass by the dedicated block above) collide with whatever role its
        # OWN junction independently requires -- an over-determined-
        # junction conflict for a topology that actually has a perfectly
        # valid resolution (verified: this is exactly what happened testing
        # TF against a geared-inertia system with a single inertia on the
        # far side). So: check each side for a forcing constraint FIRST,
        # honor it if found, and only fall back to an arbitrary (genuinely
        # free, GY's law tolerates either choice, see the dedicated block's
        # docstring) seed when NEITHER side is forced.
        if not tie_broken:
            def _forced_role(edge_key, junction_node):
                jd = G.nodes[junction_node]
                if jd["type"] not in ("0", "1"):
                    return None
                incident = _incident_power_bonds(G, junction_node)
                assigned = [(ek, o) for ek, o in incident if G.edges[ek].get("effort_giver") is not None]
                unassigned_here = [(ek, o) for ek, o in incident if G.edges[ek].get("effort_giver") is None]

                def is_special(giver, other):
                    return (giver == other) if jd["type"] == "0" else (giver == junction_node)

                sc = sum(1 for ek, o in assigned if is_special(G.edges[ek]["effort_giver"], o))
                if sc == 0 and len(unassigned_here) == 1 and unassigned_here[0][0] == edge_key:
                    other = unassigned_here[0][1]
                    return other if jd["type"] == "0" else junction_node
                return None

            for n, d in G.nodes(data=True):
                if d["type"] not in ("GY", "TF"):
                    continue
                bonds = _incident_power_bonds(G, n)
                if len(bonds) != 2:
                    continue
                (ek_a, other_a), (ek_b, other_b) = bonds
                if G.edges[ek_a].get("effort_giver") is not None or G.edges[ek_b].get("effort_giver") is not None:
                    continue  # one side already resolved -- the dedicated block above handles it

                forced_a = _forced_role(ek_a, other_a)
                forced_b = _forced_role(ek_b, other_b)
                if forced_a is not None:
                    set_giver(ek_a, forced_a)
                elif forced_b is not None:
                    set_giver(ek_b, forced_b)
                else:
                    set_giver(ek_a, n)  # genuinely free choice -- either is physically valid
                tie_broken = True
                break

        if not tie_broken:
            break

    unresolved = [d["bond_id"] for u, v, d in G.edges(data=True)
                  if not d["signal"] and d.get("effort_giver") is None]
    if unresolved:
        conflicts.append(f"Bonds left unresolved after {total_passes} propagation passes: {unresolved}")

    return {"conflicts": conflicts, "passes": total_passes, "unresolved": unresolved}


def causality_report(G) -> str:
    """Human-readable per-bond and per-element causality summary."""
    lines = []
    lines.append(f"{'bond':6s} {'giver':11s} {'receiver':11s} interpretation")
    lines.append("-" * 90)
    for u, v, d in G.edges(data=True):
        if d["signal"]:
            lines.append(f"{d['bond_id']:6s} {'(signal)':11s} {'':11s} {u} -> {v} (informs, no power)")
            continue
        giver = d["effort_giver"]
        receiver = v if giver == u else u
        g_type = G.nodes[giver]["type"]
        r_type = G.nodes[receiver]["type"]
        interp = f"{giver}({g_type}) gives EFFORT -> {receiver}({r_type}) gives FLOW back"
        lines.append(f"{d['bond_id']:6s} {giver:11s} {receiver:11s} {interp}")

    lines.append("")
    lines.append("Element-level causality check:")
    for n, d in G.nodes(data=True):
        if d["type"] == "C":
            bonds = _incident_power_bonds(G, n)
            (ek, other), = bonds
            ok = G.edges[ek]["effort_giver"] == n
            lines.append(f"  C  {n:10s}: {'integral' if ok else 'DERIVATIVE (!)'} causality "
                         f"(C gives effort / takes flow in)")
        elif d["type"] == "I":
            bonds = _incident_power_bonds(G, n)
            (ek, other), = bonds
            ok = G.edges[ek]["effort_giver"] == other
            lines.append(f"  I  {n:10s}: {'integral' if ok else 'DERIVATIVE (!)'} causality "
                         f"(I takes effort in / gives flow)")
        elif d["type"] == "R":
            bonds = _incident_power_bonds(G, n)
            (ek, other), = bonds
            giver = G.edges[ek]["effort_giver"]
            causality = "resistance (flow in, effort out)" if giver == n else \
                        "conductance (effort in, flow out)"
            lines.append(f"  R  {n:10s}: {causality}")
    return "\n".join(lines)


if __name__ == "__main__":
    G = build_bond_graph()
    report = assign_causality(G)

    print(f"SCAP propagation finished in {report['passes']} pass(es).")
    if report["conflicts"]:
        print(f"\n{len(report['conflicts'])} CONFLICT(S) FOUND:")
        for c in report["conflicts"]:
            print(f"  - {c}")
    else:
        print("No conflicts. Every C/I element got its preferred integral causality;")
        print("every R's causality was fully determined by junction constraints.")

    print()
    print(causality_report(G))
