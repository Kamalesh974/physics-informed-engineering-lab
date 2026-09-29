"""
parser.py -- safe parsing of governing-equation text and compilation of the
resulting SymPy expressions into evaluators.

Two things live here:

1. `parse_expression` turns a formula string into a SymPy expression without
   ever giving the parser access to Python builtins (same approach as the
   project's root `formula_safety.py`: every identifier is bound explicitly,
   either to a whitelisted math function or to a plain Symbol, so names such
   as E, I, S, N, gamma stay ordinary symbols instead of becoming Euler's
   number, the imaginary unit or SymPy functions).

2. `compile_expr` turns a SymPy expression into a closure that evaluates it
   with either PyTorch (autograd-friendly, used inside the PINN physics loss)
   or NumPy (used by the SciPy reference solver). Both backends walk the SAME
   expression tree, so the PINN and the reference solution are guaranteed to
   use identical physics.
"""
import re

import numpy as np
import sympy as sp
import torch
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

DTYPE = torch.float64          # default precision (CPU); the trainer can choose float32 for the GPU

MAX_EXPR_LEN = 5000
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_\s+\-*/^().,]*$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_POW_CHAIN = re.compile(r"(\*\*|\^)\s*[A-Za-z0-9_.]+\s*(\*\*|\^)")
_BIG_EXPONENT = re.compile(r"(\*\*|\^)\s*\(?\s*\d{4,}")

FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "tanh": sp.tanh, "sinh": sp.sinh, "cosh": sp.cosh,
    "atan": sp.atan, "asin": sp.asin, "acos": sp.acos, "sign": sp.sign,
    "Heaviside": sp.Heaviside, "Min": sp.Min, "Max": sp.Max, "Mod": sp.Mod,
}
CONSTS = {"pi": sp.pi}
RESERVED = set(FUNCS) | set(CONSTS) | {"t"}


class ParseError(ValueError):
    pass


def parse_expression(text, symbols):
    """Parse `text` into a SymPy expression.

    symbols: dict name -> Symbol; identifiers found there are reused, any
    other identifier becomes a fresh Symbol (the caller decides whether an
    unknown name is an error)."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty expression")
    if len(text) > MAX_EXPR_LEN:
        raise ParseError(f"expression longer than {MAX_EXPR_LEN} characters")
    if not _ALLOWED_CHARS.match(text):
        bad = sorted({c for c in text if not _ALLOWED_CHARS.match(c)})
        raise ParseError(f"characters not allowed in an expression: {' '.join(bad)}")
    if "__" in text:
        raise ParseError("double underscores are not allowed")
    if _POW_CHAIN.search(text) or _BIG_EXPONENT.search(text):
        raise ParseError("chained or very large exponents are not allowed")

    ns = {}
    for name in set(_IDENT.findall(text)):
        if name in FUNCS:
            ns[name] = FUNCS[name]
        elif name in CONSTS:
            ns[name] = CONSTS[name]
        elif name in symbols:
            ns[name] = symbols[name]
        else:
            ns[name] = sp.Symbol(name)
    try:
        expr = parse_expr(
            text, local_dict=ns,
            global_dict={"__builtins__": {}, "Integer": sp.Integer, "Float": sp.Float,
                         "Rational": sp.Rational, "Symbol": sp.Symbol},
            transformations=standard_transformations + (convert_xor,),
        )
    except Exception as e:  # noqa: BLE001
        raise ParseError(f"could not parse '{text.strip()}': {e}") from e
    if not isinstance(expr, sp.Expr):
        raise ParseError(f"'{text.strip()}' is not a mathematical expression")
    return expr


# --------------------------------------------------------------------------
# Backend-neutral evaluator
# --------------------------------------------------------------------------
def _like(x, ref):
    """x as a tensor with the dtype/device of `ref` (the tensor it is combined with).

    Constants in an expression are Python floats; they must follow the PINN's
    tensors onto the GPU and into float32/float64 instead of defaulting to CPU."""
    if torch.is_tensor(x):
        return x
    if torch.is_tensor(ref):
        return torch.as_tensor(float(x), dtype=ref.dtype, device=ref.device)
    return torch.tensor(float(x), dtype=DTYPE)


def _pair(a, b):
    ref = a if torch.is_tensor(a) else b
    return _like(a, ref), _like(b, ref)


class _Torch:
    unary = {
        "sin": torch.sin, "cos": torch.cos, "tan": torch.tan, "exp": torch.exp, "log": torch.log,
        "tanh": torch.tanh, "sinh": torch.sinh, "cosh": torch.cosh, "atan": torch.atan,
        "asin": torch.asin, "acos": torch.acos, "Abs": torch.abs, "sign": torch.sign,
    }

    @staticmethod
    def call(name, x):
        return _Torch.unary[name](_like(x, x))

    @staticmethod
    def heaviside(x):
        x = _like(x, x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def minimum(a, b):
        return torch.minimum(*_pair(a, b))

    @staticmethod
    def maximum(a, b):
        return torch.maximum(*_pair(a, b))

    @staticmethod
    def mod(a, b):
        return torch.remainder(*_pair(a, b))

    @staticmethod
    def sqrt(x):
        return torch.sqrt(_like(x, x))

    @staticmethod
    def pow(a, b):
        return torch.pow(*_pair(a, b))

    @staticmethod
    def where(c, a, b):
        a, b = _pair(a, b)
        if not torch.is_tensor(c):
            c = torch.tensor(bool(c), device=a.device)
        return torch.where(c, a, b)


class _Numpy:
    unary = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
        "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh, "atan": np.arctan,
        "asin": np.arcsin, "acos": np.arccos, "Abs": np.abs, "sign": np.sign,
    }

    @staticmethod
    def call(name, x):
        return _Numpy.unary[name](x)

    @staticmethod
    def heaviside(x):
        return np.heaviside(x, 1.0)

    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)
    mod = staticmethod(np.mod)
    sqrt = staticmethod(np.sqrt)
    pow = staticmethod(np.power)
    where = staticmethod(np.where)


_REL = {
    sp.StrictLessThan: lambda a, b: a < b, sp.LessThan: lambda a, b: a <= b,
    sp.StrictGreaterThan: lambda a, b: a > b, sp.GreaterThan: lambda a, b: a >= b,
}


def compile_expr(expr, backend="torch"):
    """Return f(env) evaluating `expr`; env maps symbol name -> value."""
    B = _Torch if backend == "torch" else _Numpy
    return _build(sp.sympify(expr), B)


def _build(e, B):
    if e is sp.true or e == sp.true:
        return lambda env: True
    if isinstance(e, sp.Symbol):
        name = e.name
        return lambda env: env[name]
    if not e.free_symbols and not isinstance(e, (sp.Piecewise, sp.Heaviside)):
        v = float(e)
        return lambda env: v
    if isinstance(e, sp.Add):
        fs = [_build(a, B) for a in e.args]

        def f_add(env):
            out = fs[0](env)
            for g in fs[1:]:
                out = out + g(env)
            return out
        return f_add
    if isinstance(e, sp.Mul):
        fs = [_build(a, B) for a in e.args]

        def f_mul(env):
            out = fs[0](env)
            for g in fs[1:]:
                out = out * g(env)
            return out
        return f_mul
    if isinstance(e, sp.Pow):
        base, ex = e.args
        fb = _build(base, B)
        if ex == sp.Rational(1, 2):
            return lambda env: B.sqrt(fb(env))
        if ex == sp.Rational(-1, 2):
            return lambda env: 1.0 / B.sqrt(fb(env))
        if ex.is_Integer:
            n = int(ex)
            if n == -1:
                return lambda env: 1.0 / fb(env)
            return lambda env: fb(env) ** n
        fe = _build(ex, B)
        return lambda env: B.pow(fb(env), fe(env))
    if isinstance(e, sp.Heaviside):
        fa = _build(e.args[0], B)
        return lambda env: B.heaviside(fa(env))
    if isinstance(e, (sp.Min, sp.Max)):
        fs = [_build(a, B) for a in e.args]
        op = B.minimum if isinstance(e, sp.Min) else B.maximum

        def f_mm(env):
            out = fs[0](env)
            for g in fs[1:]:
                out = op(out, g(env))
            return out
        return f_mm
    if isinstance(e, sp.Mod):
        fa, fb = _build(e.args[0], B), _build(e.args[1], B)
        return lambda env: B.mod(fa(env), fb(env))
    if isinstance(e, sp.Piecewise):
        pairs = [(_build(ex, B), _build(c, B)) for ex, c in e.args]

        def f_pw(env):
            out = pairs[-1][0](env)
            for fe, fc in reversed(pairs[:-1]):
                out = B.where(fc(env), fe(env), out)
            return out
        return f_pw
    if type(e) in _REL:
        fa, fb, op = _build(e.args[0], B), _build(e.args[1], B), _REL[type(e)]
        return lambda env: op(fa(env), fb(env))
    if isinstance(e, sp.Function):
        name = type(e).__name__
        if name in B.unary and len(e.args) == 1:
            fa = _build(e.args[0], B)
            return lambda env: B.call(name, fa(env))
    raise ParseError(f"unsupported construct in expression: {e}")
