"""
formula_safety.py -- safe parsing of user/LLM-supplied formula strings.

WHY: image_to_bond_graph.build_graph_from_spec used sp.sympify() on the
`modulation_formula` string. That string can originate from a vision model
reading an UNTRUSTED image (text inside an image can steer the model), and
sympify() ends in eval() -- a remote-code-execution risk on any server.

It also had a silent correctness bug: with sympify's default namespace the
names E, I, S, N, Q, gamma, beta, zeta ... are NOT free symbols but Euler's
number, the imaginary unit, or sympy functions, so a formula such as
'E*alpha*A_eff*(SIGNAL - T_ref)' used 2.718... for the Young's modulus E.

Fix: every identifier is bound explicitly (whitelisted math functions or a
plain Symbol) and the parse runs with an empty __builtins__.
"""
import re

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

MAX_LEN = 300
MAX_POW_OPS = 3
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_\s+\-*/^().,]+$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_POW_CHAIN = re.compile(r"(\*\*|\^)\s*[A-Za-z0-9_.]+\s*(\*\*|\^)")
_LONG_NUMBER = re.compile(r"(?<![A-Za-z_0-9])\d{7,}")
_BIG_EXPONENT = re.compile(r"(\*\*|\^)\s*\(?\s*\d{4,}")

ALLOWED_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "tanh": sp.tanh, "Min": sp.Min, "Max": sp.Max,
    "Heaviside": sp.Heaviside,
}


class UnsafeFormula(ValueError):
    pass


def safe_sympify(formula, known_symbols=None):
    """Parse `formula` into a sympy expression, or raise UnsafeFormula.

    known_symbols: dict name -> sympy Symbol that must be reused (so the
    result shares symbols with the rest of the graph)."""
    known_symbols = known_symbols or {}
    if not isinstance(formula, str) or not formula.strip():
        raise UnsafeFormula("formula must be a non-empty string")
    if len(formula) > MAX_LEN:
        raise UnsafeFormula(f"formula longer than {MAX_LEN} characters")
    if not _ALLOWED_CHARS.match(formula):
        raise UnsafeFormula("formula contains characters outside [A-Za-z0-9_ +-*/^().,]")
    if "__" in formula:
        raise UnsafeFormula("double underscores are not allowed")
    if formula.count("**") + formula.count("^") > MAX_POW_OPS:
        raise UnsafeFormula("too many exponent operators")
    # a**b**c towers (e.g. 9**9**9**9) explode in size/time when evaluated
    if _POW_CHAIN.search(formula):
        raise UnsafeFormula("chained exponents are not allowed")
    if _BIG_EXPONENT.search(formula):
        raise UnsafeFormula("exponents of 1000 or more are not allowed")
    if _LONG_NUMBER.search(formula):
        raise UnsafeFormula("numeric literals longer than 6 digits are not allowed")

    ns = {}
    for name in set(_IDENT.findall(formula)):
        if name in ALLOWED_FUNCS:
            ns[name] = ALLOWED_FUNCS[name]
        elif name in known_symbols:
            ns[name] = known_symbols[name]
        else:
            ns[name] = sp.Symbol(name)
    try:
        return parse_expr(
            formula, local_dict=ns,
            global_dict={"__builtins__": {}, "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
                         "Mul": sp.Mul, "Add": sp.Add, "Pow": sp.Pow},
            transformations=standard_transformations + (convert_xor,), evaluate=False,
        )
    except Exception as e:  # noqa: BLE001
        raise UnsafeFormula(f"could not parse formula: {e}") from e
