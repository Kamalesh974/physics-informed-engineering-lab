"""
image_to_bond_graph.py -- ask the user for a bond graph image (hand-drawn,
generated, or made in a diagramming tool), send it to a vision-capable LLM
(Google Gemini, via the free-tier Gemini API) to extract a structured graph
description, build the corresponding NetworkX bond graph, run it through
the EXISTING generic engine (scap_causality.assign_causality + equation_gen's
bond_effort/bond_flow via derive_odes_generic), and print the governing
ODEs -- all from a single terminal command, no manual code-writing step.

This is different in kind from bond_graph_from_image.py: that file is code
a human (well, an LLM in a chat session) wrote by hand after looking at one
specific image. This script is a program you run, that reads the image
itself, every time, for any image -- the automation the user actually
asked for.

REQUIREMENTS
    pip install google-genai
    export GEMINI_API_KEY=...      (or set it in your environment)
Get a FREE key (no credit card required) at https://aistudio.google.com/apikey
-- unlike many paid LLM APIs, Google AI Studio's free tier is genuinely
free, but rate-limited. VERIFIED LIVE (path_c_evaluation.py's batch run,
2026-08-13): the model this alias currently resolves to (gemini-3.6-flash)
has a free-tier cap of only 20 requests/DAY -- much lower than an earlier
generation's ~1500/day this docstring used to claim (that number is now
WRONG for whatever model "gemini-flash-latest" resolves to at any given
time; Google's free-tier limits are model-specific and change as the alias
rolls forward, so don't trust a cached number here -- check
https://ai.google.dev/gemini-api/docs/rate-limits before planning a batch
run). Concretely: this cap means a self-consistency batch job of N diagrams
x M samples each can only make ~20 calls/day total on a single free key --
plan batch runs accordingly (path_c_evaluation.py hit this cap mid-run on
its 6th benchmark; that benchmark could only be tested with a fresh key or
the next day). PRIVACY NOTE: Google's free tier may use your inputs/outputs
to improve their models -- keep that in mind before sending a diagram you'd
rather keep private.

ROBUSTNESS: the vision model's raw JSON output is not guaranteed to
describe a causally well-posed graph on the first try (see this project's
own history: even careful manual reading of two diagrams needed several
rounds of topology fixes). If SCAP fails to resolve the graph, this script
feeds the specific conflict back to the model and asks it to revise, up to
MAX_RETRIES times, before giving up and showing you the raw failure.

SCOPE LIMITS (flagged): the vision model can read topology and element
labels from an image, but it CANNOT read numeric parameter values off a
diagram that doesn't have them written on it. After the equations are
derived symbolically, this script optionally prompts you for numeric
values (or you can skip that and just get the symbolic ODEs).
"""
import json
import os
import re
import time
from collections import Counter

import sympy as sp
import networkx as nx

from equation_gen import derive_odes_generic
from param_usage_check import find_unused_params, report as report_unused_params
from formula_safety import safe_sympify

MAX_RETRIES = 5          # genuine "model output didn't resolve, try again with feedback" cycles
                          # -- now worth spending more on, since retries are real targeted
                          # corrections (same chat session) rather than blind regenerations
MAX_API_RETRIES = 4       # transient API-error retries (rate limit, 503, network) -- separate
API_RETRY_BACKOFF_S = 5   # seconds, doubles each transient retry
API_CALLS = 0             # every request actually sent to the API, retries included: the free tier
                          # counts ALL of them (a 503 'busy' reply still uses up allowance)
MODEL_NAME = "gemini-flash-latest"  # alias -> newest available Flash model, avoids hardcoding
                                     # a specific generation that gets deprecated for new
                                     # projects (hit exactly this with "gemini-2.5-flash")

SCHEMA_INSTRUCTIONS = """\
You are reading a hand-drawn or generated BOND GRAPH diagram (a standard \
engineering notation for multi-energy-domain dynamic systems) and must \
output ONLY a single JSON object (no prose, no markdown fences) describing \
its exact structure, using this schema:

{
  "nodes": [
    {
      "id": "<short unique identifier, e.g. 'C1', 'J_mech', 'Rf'>",
      "type": "<one of: 0, 1, R, C, I, GY, TF, Se, Sf, MSe, MSf>",
      "domain": "<free text, e.g. 'thermal', 'mechanical', 'electrical'>",
      "label": "<what the diagram calls this element>",

      // ONLY for type R, C, I, GY, or TF:
      "param_symbol_name": "<short symbol name for its parameter (resistance/capacitance/inertia value, or for GY the gyrator modulus, or for TF the transformer/gear turns ratio), e.g. 'R_f', 'C_d', 'I_disc', 'k_motor', 'n_gear'>",

      // ONLY for type Se or Sf that is a plain (unmodulated) constant/input:
      "expr_param_name": "<short symbol name for the prescribed value, e.g. 'T_amb', 'F_applied'>",

      // ONLY for type MSe or MSf (a source whose value is modulated by
      // another part of the graph via a signal/activated bond -- shown in
      // the diagram as a dashed line or explicitly noted as "modulated by X"):
      "signal_from": "<id of the JUNCTION node (a 0 or 1) that this source reads its modulating signal from>",
      "signal_kind": "<'effort' if the signal is the junction's common effort (e.g. a temperature or voltage), 'flow' if it's the junction's common flow (e.g. a velocity or current)>",
      "modulation_formula": "<a Python/sympy-syntax expression for this source's value. Use the literal token SIGNAL for the incoming signal's value, plus any short symbol names for physical constants this formula needs (they do NOT need their own separate node -- just use a clear name directly here, e.g. 'R_f*SIGNAL**2' or 'E*alpha*A_eff*(SIGNAL - T_ref)' is fine even if E/alpha/A_eff/T_ref weren't drawn as their own boxes).>"
    },
    ...
  ],
  "edges": [
    {"from": "<node id>", "to": "<node id>", "signal": false},
    ...
    // signal:true edges are the dashed/activated bonds feeding an MSe/MSf
    // node's modulating input -- there should be exactly one such edge
    // per MSe/MSf node, from its signal_from junction to itself.
  ]
}

RULES (read carefully -- these are common mistakes, from experience building bond graphs by hand):
1. EVERY R, C, I, Se, Sf, MSe, MSf element must attach to a 0-junction or
   1-junction via its own edge -- never connect two R/C/I/source elements
   directly to each other. GY (gyrator) and TF (transformer) are the TWO
   exceptions: both are 2-PORT elements with exactly TWO edges (one to a
   junction on each "side" of it, e.g. an electrical-side junction and a
   mechanical-side junction) -- do not give GY or TF only one edge, and do
   not treat either like a 1-port R/C/I.
     - GY converts between an effort/flow pair on one side and the
       OPPOSITE pair on the other (e.g. torque<->force, voltage<->angular-
       velocity) -- if the diagram shows a "GY" or "gyrator" symbol with a
       modulus/ratio label (like "r" or "k"), that's this.
     - TF converts an effort/flow pair into a SCALED version of the SAME
       pair on the other side (e.g. voltage<->voltage through an
       electrical transformer's turns ratio, or torque<->torque through a
       mechanical gear ratio -- NOT a domain change) -- if the diagram
       shows a "TF" symbol, a gear/pulley pair, or a transformer winding
       ratio label (like "n" or "1:n"), that's this.
2. Do NOT insert an intermediate 0- or 1-junction between two resistors (or
   any two elements) that have NOTHING else attached to them and no
   independent way to determine an effort or flow value -- e.g. do not put
   a bare 0-junction between two plain R's in series with nothing else
   branching off it. If two resistances are simply in series along one
   path with nothing else attached at the point between them, put them as
   TWO SEPARATE STUBS ON THE SAME 1-junction instead (do not invent an
   in-between junction node for them).
3. Every 0-junction needs at least one attached element that can
   independently supply a value (a C, an Se, or a chain leading to one) --
   a 0-junction surrounded only by resistors and other pass-through
   junctions is not resolvable.
4. Every dynamic element (C or I) needs EXACTLY ONE incident bond.
5. If the diagram shows a signal/activated bond (commonly drawn dashed, or
   with a note like "modulated by" / "driven by") crossing between energy
   domains (e.g. a temperature affecting a mechanical force, or an angular
   velocity affecting a heat generation rate), represent it with an MSe or
   MSf node plus a signal:true edge FROM the informing junction TO that
   MSe/MSf node -- do not represent it as a normal power bond.
6. Junction types: 0-junction = common EFFORT across all its bonds
   (efforts equal, flows sum to zero). 1-junction = common FLOW across all
   its bonds (flows equal, efforts sum to zero).
7. A 0-junction may have AT MOST ONE attached branch that can independently
   determine its effort value (a C, or a chain of bonds/junctions that
   eventually reaches a C or an Se) -- every OTHER branch at that
   0-junction must be a plain R (or a chain leading only to R's). If your
   diagram shows what looks like TWO capacitors, or a capacitor and a
   source, both connecting to the same point, they must NOT share one
   0-junction directly -- there must be a resistor (on its own 1-junction)
   between them representing the conduction/resistance path connecting
   those two points, even if the diagram doesn't draw a separate dot there.
   The dual rule applies to 1-junctions with I's/flow-sources.
   Getting this wrong is the single most common mistake -- double check
   every 0-junction and 1-junction against this rule before finalizing.

Read the image now and output ONLY the JSON object.
"""


def _check_setup():
    problems = []
    try:
        from google import genai  # noqa: F401
    except ImportError:
        problems.append("Package 'google-genai' is not installed. Run: pip install google-genai")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        problems.append(
            "GEMINI_API_KEY is not set. Get a FREE key (no credit card) at "
            "https://aistudio.google.com/apikey and set it, e.g.:\n"
            "    PowerShell:  $env:GEMINI_API_KEY = 'your-key-here'\n"
            "    (free tier: ~15 requests/min, 1500/day for gemini-2.5-flash -- "
            "plenty for this script)"
        )
    return problems


def _encode_image(path):
    ext = os.path.splitext(path)[1].lower()
    media_type = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
    }.get(ext)
    if media_type is None:
        raise ValueError(f"Unsupported image extension '{ext}'. Use png/jpg/jpeg/webp/gif.")
    with open(path, "rb") as f:
        data = f.read()
    return data, media_type


def _extract_json(text):
    """The model was told to output raw JSON, but be defensive: strip
    markdown fences if present, and grab the outermost {...} block."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text[:500]}")
    return json.loads(text[start:end + 1])


def _make_chat(client):
    """A persistent chat session (not a one-shot call) is essential here:
    when a retry is needed, the model must see its OWN previous JSON
    answer to make a targeted fix. Independent single-shot calls (the
    first version of this script) have no memory of the prior attempt, so
    "retries" were actually blind regenerations from scratch -- confirmed
    live: three attempts produced three totally different graph structures
    (different junction names each time) hitting different flavors of the
    same mistake, rather than converging on a fix."""
    from google.genai import types
    return client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )


def _first_turn_content(image_bytes, media_type):
    from google.genai import types
    return [types.Part.from_bytes(data=image_bytes, mime_type=media_type), SCHEMA_INSTRUCTIONS]


def _retry_turn_content(error_text):
    return [
        "Your previous JSON output (above) failed to resolve with this error:\n"
        f"{error_text}\n"
        "Look at what you output last time and make a TARGETED fix for this specific "
        "problem (see RULE 7 in particular if it mentions a junction being over- or "
        "under-determined). Output ONLY the corrected, complete JSON object."
    ]


def build_graph_from_spec(spec):
    """Convert the parsed JSON spec into a NetworkX graph following the
    bond_graph.py schema, creating sympy symbols/state functions on the
    fly. Returns (G, symbol_table, t_sym)."""
    t_sym = sp.symbols("t", nonnegative=True)
    symbol_table = {}

    def get_symbol(name, **assumptions):
        if name not in symbol_table:
            symbol_table[name] = sp.Symbol(name, **assumptions)
        return symbol_table[name]

    G = nx.DiGraph()
    node_specs = {n["id"]: n for n in spec["nodes"]}

    for nid, n in node_specs.items():
        ntype = n["type"]
        attrs = {"type": ntype, "domain": n.get("domain", ""), "label": n.get("label", "")}

        if ntype in ("R", "C", "I", "GY", "TF"):
            sym = get_symbol(n["param_symbol_name"], positive=True)
            attrs["param_symbol"] = sym
            attrs["param_key"] = n["param_symbol_name"]
            if ntype == "C":
                attrs["state_symbol"] = sp.Function(f"q_{nid}")(t_sym)
            elif ntype == "I":
                attrs["state_symbol"] = sp.Function(f"p_{nid}")(t_sym)
            # GY/TF get no state_symbol -- neither is a dynamic element,
            # just a 2-port converter (see equation_gen.py's bond_effort/
            # bond_flow GY/TF dispatch).

        elif ntype in ("Se", "Sf"):
            sym = get_symbol(n["expr_param_name"], real=True)
            attrs["expr"] = sym

        elif ntype in ("MSe", "MSf"):
            signal_sym = sp.Symbol(f"SIGNAL_{nid}", real=True)
            local_ns = dict(symbol_table)
            local_ns["SIGNAL"] = signal_sym
            expr = safe_sympify(n["modulation_formula"], local_ns)  # never raw sympify: see formula_safety.py
            # Any symbol here that ISN'T already declared elsewhere is treated
            # as a plain physical constant used only inside this formula (e.g.
            # a modulus like E or alpha that was never drawn as its own box in
            # the diagram, just written inline in a compound expression) --
            # auto-register it rather than requiring a separate node for it.
            # (Earlier version required pre-declaration and rejected these;
            # that was too strict and caused repeated real-world failures.)
            for s in expr.free_symbols:
                if s is not signal_sym and str(s) not in symbol_table:
                    symbol_table[str(s)] = s
            attrs["expr"] = expr
            attrs["signal_from"] = n["signal_from"]
            attrs["signal_symbol"] = signal_sym
            attrs["signal_kind"] = n.get("signal_kind", "effort")

        elif ntype in ("0", "1"):
            pass
        else:
            raise ValueError(f"Node {nid}: unknown type '{ntype}'")

        G.add_node(nid, **attrs)

    for idx, e in enumerate(spec["edges"], start=1):
        G.add_edge(e["from"], e["to"], bond_id=f"b{idx}", signal=bool(e.get("signal", False)), causality=None)

    return G, symbol_table, t_sym


def _send_with_backoff(chat, content):
    """Wraps chat.send_message with its OWN retry budget for TRANSIENT
    errors (rate limits, 'model overloaded' 503s, network blips). These
    have nothing to do with whether the model's output was usable, so they
    must not consume a slot from main()'s MAX_RETRIES loop (which is for
    genuine 'the model answered, but its answer didn't resolve' cycles) --
    conflating the two wastes real correction attempts on bad luck, which
    is exactly what happened the first time this ran live."""
    global API_CALLS
    delay = API_RETRY_BACKOFF_S
    for i in range(1, MAX_API_RETRIES + 1):
        try:
            API_CALLS += 1
            return chat.send_message(content).text
        except Exception as e:  # noqa: BLE001
            # A DAILY quota 429 will not clear in seconds: retrying only burns
            # more of the (tiny) free-tier allowance and blocks the caller.
            if "RESOURCE_EXHAUSTED" in str(e) and "PerDay" in str(e):
                raise
            if i == MAX_API_RETRIES:
                raise
            print(f"    (API call failed, transient -- retrying in {delay}s: {e})")
            time.sleep(delay)
            delay *= 2


def run_pipeline(image_bytes, media_type, client, max_retries=MAX_RETRIES,
                  max_api_retries=None, verbose=True):
    """Core image -> graph -> ODEs pipeline, factored out of main() so it
    can be reused both by the interactive CLI below AND by a batch
    evaluation harness (path_c_evaluation.py) that needs a programmatic
    return value instead of print-and-exit. Behavior is identical to the
    original main() loop -- same persistent-chat retry mechanism, same
    schema/prompt -- just returns a structured result instead of printing
    and returning early.

    Returns a dict:
      success        -- bool
      attempts_used   -- int
      G, symbol_table, state_symbols, odes -- present if success
      unused_params   -- param_usage_check result dict, present if success
      spec            -- last parsed JSON spec (may be present even on failure)
      error           -- last error string, present if not success
    """
    global MAX_API_RETRIES
    if max_api_retries is not None:
        MAX_API_RETRIES = max_api_retries
    calls_before = API_CALLS

    chat = _make_chat(client)
    content = _first_turn_content(image_bytes, media_type)
    last_error = None
    last_spec = None

    for attempt in range(1, max_retries + 1):
        if verbose:
            print(f"  [attempt {attempt}/{max_retries}] Sending {'image' if attempt == 1 else 'correction'} "
                  f"to {MODEL_NAME}...")
        try:
            raw = _send_with_backoff(chat, content)
        except Exception as e:  # noqa: BLE001 -- exhausted the API-level retry budget too
            last_error = f"API call kept failing: {e}"
            if verbose:
                print(f"    Giving up: {last_error}")
            return {"success": False, "attempts_used": attempt, "api_calls": API_CALLS - calls_before, "error": last_error, "spec": last_spec}

        try:
            spec = _extract_json(raw)
            last_spec = spec
            G, symbol_table, t_sym = build_graph_from_spec(spec)
            state_symbols, odes = derive_odes_generic(G)
        except Exception as e:  # noqa: BLE001 -- genuinely want to catch+retry on anything here
            last_error = str(e)
            if verbose:
                print(f"    Failed: {last_error}")
            content = _retry_turn_content(last_error)
            continue

        readouts = []  # no generic readout concept here (unlike the disc-brake-specific
                        # generate_odes()) -- unused-param check runs against the ODEs alone
        unused = find_unused_params(odes, symbol_table.keys(), extra_exprs=readouts)

        if verbose:
            print(f"  Resolved successfully on attempt {attempt}.")
            print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}  "
                  f"States found: {len(state_symbols)}")
            report_unused_params(unused, label="Path C live run")

        return {
            "success": True, "attempts_used": attempt, "api_calls": API_CALLS - calls_before, "G": G,
            "symbol_table": symbol_table, "state_symbols": state_symbols,
            "odes": odes, "unused_params": unused, "spec": spec,
        }

    return {"success": False, "attempts_used": max_retries, "api_calls": API_CALLS - calls_before, "error": last_error, "spec": last_spec}


def _structure_signature(result):
    """A coarse, symbol-naming-agnostic fingerprint of a successful run's
    graph -- used to score self-consistency across independent samples.
    Shared with path_c_evaluation.py (imports this rather than keeping its
    own copy) so the two never quietly drift apart."""
    G = result["G"]
    type_counts = Counter(d["type"] for _, d in G.nodes(data=True))
    return {
        "n_states": len(result["state_symbols"]),
        "type_counts": dict(sorted(type_counts.items())),
        "n_unused_params": len(result["unused_params"]["unused"]),
    }


def run_with_voting(image_bytes, media_type, client, n_samples=3, max_retries=MAX_RETRIES, verbose=True):
    """Run N independent FRESH-CHAT attempts at reading the SAME image (not
    corrections of each other -- each gets its own persistent chat session,
    same idea as path_c_evaluation.py's batch harness, now available in the
    interactive tool too) and report whether they agree on gross structure.

    IMPORTANT, PROVEN LIVE (see results/path_c_scoring.md): agreement across
    samples is NOT proof of correctness -- the DC-motor benchmark's 3
    samples were perfectly self-consistent (identical structure every time)
    and STILL had a real physics bug. Disagreement IS a reliable signal that
    something needs a closer look; agreement is only a weaker "no red flag
    raised," not a guarantee.
    """
    samples = []
    for i in range(n_samples):
        if verbose:
            print(f"\n-- independent read {i + 1}/{n_samples} --")
        result = run_pipeline(image_bytes, media_type, client, max_retries=max_retries, verbose=verbose)
        samples.append(result)

    succ = [s for s in samples if s["success"]]
    sigs = [json.dumps(_structure_signature(s), sort_keys=True) for s in succ]
    consistent = len(succ) == n_samples and len(set(sigs)) == 1
    return {"samples": samples, "n_success": len(succ), "n_samples": n_samples, "consistent": consistent}


def extract_numeric_values(client, image_bytes, media_type, param_names, verbose=True):
    """Best-effort: ask the vision model to read any NUMERIC values written
    on the diagram for the given parameter names (e.g. '10 ohm', '2 kg').
    Kept as a SEPARATE follow-up call (fresh chat, image re-sent) rather
    than folded into the main structural-extraction turn, because most
    diagrams (including all 6 of tonight's benchmarks) are purely symbolic
    -- a null result here just means "no numbers were drawn," not a
    failure, and shouldn't consume any of the main extraction's retry
    budget or complicate that prompt's already-carefully-tuned rules.

    Returns dict: param_name -> float, only for parameters where the model
    reported an actual visible number (never guessed/estimated -- the
    prompt explicitly forbids that).
    """
    from google.genai import types
    prompt = (
        "Look at this same bond graph image again. For EACH of the following "
        "parameter names, check whether the diagram has a NUMERIC value "
        "written next to it on the diagram itself (e.g. a resistance labeled "
        "'10 ohm', a mass labeled '2 kg', a source labeled '5V'). "
        f"Parameter names: {sorted(param_names)}\n"
        "Output ONLY a single JSON object (no prose) mapping each parameter "
        "name to either a plain number (just the numeric value, no units, no "
        "quotes) if one is ACTUALLY visibly written on the image for it, or "
        "null if no number is shown for it anywhere on the diagram. Do NOT "
        "guess, estimate, or fill in a typical/textbook value that isn't "
        "actually written on the image -- null is the correct answer for "
        "any parameter with no visible number."
    )
    content = [types.Part.from_bytes(data=image_bytes, mime_type=media_type), prompt]
    try:
        chat = _make_chat(client)
        raw = _send_with_backoff(chat, content)
        data = _extract_json(raw)
    except Exception as e:  # noqa: BLE001 -- non-fatal, equations just stay symbolic
        if verbose:
            print(f"  Numeric extraction failed (non-fatal, equations remain symbolic): {e}")
        return {}

    values = {}
    for name in param_names:
        v = data.get(name)
        if v is None:
            continue
        try:
            values[name] = float(v)
        except (TypeError, ValueError):
            pass
    return values


def confidence_report(result, voting_result=None):
    """One consolidated verdict combining every automated signal this
    pipeline has: SCAP well-posedness (implicit -- we only get here if it
    resolved), the unused-parameter check, and self-consistency voting (if
    it was run). Deliberately does NOT claim more than it can prove -- see
    the DC-motor finding in results/path_c_scoring.md, which passed EVERY
    one of these checks and was still wrong. This report is a triage tool
    (what to double-check by hand), not a correctness certificate."""
    lines = ["", "=" * 78, "CONFIDENCE REPORT", "=" * 78]
    lines.append("  [OK]   SCAP resolved with zero causality conflicts")

    unused = result["unused_params"]["unused"]
    if unused:
        lines.append(f"  [WARN] {len(unused)} declared parameter(s) never appear in the final "
                      f"equations: {sorted(unused)}")
    else:
        lines.append("  [OK]   all declared parameters appear in the final equations")

    if voting_result is not None:
        if voting_result["consistent"]:
            lines.append(f"  [OK]   {voting_result['n_success']}/{voting_result['n_samples']} independent "
                          f"reads agree on structure")
        else:
            lines.append(f"  [WARN] independent reads disagree or some failed "
                          f"({voting_result['n_success']}/{voting_result['n_samples']} succeeded) "
                          f"-- treat this result with real skepticism")
    else:
        lines.append("  [NOTE] ran only once -- no self-consistency check performed this time")

    all_clean = not unused and (voting_result is None or voting_result["consistent"])
    if all_clean and voting_result is not None and voting_result["consistent"]:
        verdict = "every automated check passed, including cross-sample agreement"
    elif all_clean:
        verdict = "every automated check passed on this single read (re-run with voting for a stronger signal)"
    else:
        verdict = "at least one check raised a flag -- verify this result by hand before trusting it"
    lines.append("")
    lines.append(f"  VERDICT: {verdict}")
    lines.append("  Reminder: passing every automated check here is NOT a correctness proof --")
    lines.append("  see results/path_c_scoring.md for a live case where all of them passed on a")
    lines.append("  result that was still physically wrong. Read the derived equations yourself.")
    lines.append("=" * 78)
    text = "\n".join(lines)
    print(text)
    return text


def main():
    problems = _check_setup()
    if problems:
        print("Setup needed before this can call the vision model:\n")
        for p in problems:
            print(f"  - {p}")
        print("\nOnce that's done, re-run this script.")
        return

    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    path = input("Path to your bond graph image: ").strip().strip('"')
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return

    image_bytes, media_type = _encode_image(path)

    vote_ans = input("Run with self-consistency voting (3 independent reads -- uses ~3x the API "
                      "calls, gives a real confidence signal instead of trusting one shot) [y/N]: ").strip().lower()
    use_voting = vote_ans in ("y", "yes")

    voting_result = None
    if use_voting:
        voting_result = run_with_voting(image_bytes, media_type, client)
        succ_samples = [s for s in voting_result["samples"] if s["success"]]
        if not succ_samples:
            print(f"\nAll {voting_result['n_samples']} independent reads failed. Last error:\n"
                  f"  {voting_result['samples'][-1]['error']}")
            return
        result = succ_samples[0]  # report on the first successful read; voting_result carries the agreement signal
    else:
        result = run_pipeline(image_bytes, media_type, client)
        if not result["success"]:
            print(f"\nGave up after {result['attempts_used']} attempts. Last error:\n  {result['error']}")
            print("You may need to redraw the ambiguous part of the diagram more clearly, "
                  "or build it by hand following bond_graph_from_image.py as a template.")
            return

    print("\n" + "=" * 78 + "\nAUTO-DERIVED ODE SYSTEM\n" + "=" * 78)
    for node_id, state in result["state_symbols"].items():
        print(f"\nd({state})/dt =\n    {result['odes'][state]}")

    print("\n" + "=" * 78)
    print(f"Parameters found (currently symbolic): {sorted(result['symbol_table'].keys())}")
    print("=" * 78)

    confidence_report(result, voting_result)

    numeric_ans = input("\nAlso try to read numeric parameter values off the diagram (best-effort, "
                         "another API call, only fills in values actually written on the image) [y/N]: ").strip().lower()
    if numeric_ans in ("y", "yes"):
        numeric_values = extract_numeric_values(client, image_bytes, media_type, result["symbol_table"].keys())
        if numeric_values:
            print(f"\nNumeric values found on the diagram: {numeric_values}")
            subs = {result["symbol_table"][name]: val for name, val in numeric_values.items()}
            print("\n" + "=" * 78 + "\nODE SYSTEM WITH NUMERIC VALUES SUBSTITUTED WHERE FOUND\n" + "=" * 78)
            for node_id, state in result["state_symbols"].items():
                print(f"\nd({state})/dt =\n    {result['odes'][state].subs(subs)}")
            missing = sorted(set(result["symbol_table"].keys()) - set(numeric_values.keys()))
            if missing:
                print(f"\n(Still symbolic -- no number was visible on the diagram for: {missing})")
        else:
            print("\nNo numeric values were found written on the diagram -- equations remain fully symbolic.")


if __name__ == "__main__":
    main()
