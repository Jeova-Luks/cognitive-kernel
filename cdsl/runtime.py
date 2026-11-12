"""CDSL interpreter (component [4] RUNTIME, cru).

Implemented primitives (v0.1 cru): program, let, return, if, cond, assert,
compare, verify, call_tool, lookup.

Spec primitives NOT implemented yet: decompose, solve, loop, case, simulate,
prove, request_input. Using them raises a runtime error that names the
primitive, so the reviser can tell the compiler to avoid it.

Extension beyond the §5.2 primitive list: a small set of arithmetic and
comparison builtins (+ - * / // % ** min max abs round sum len ...). Without
them every multiplication would need a tool call; whether they belong in the
final DSL is a question for the Phase 5 ablation.

Leniencies added after the first dev run (they were the top two causes of
failed programs with an 8B stand-in compiler):
- Every variable bound with ``let`` is passed to tools automatically, so
  ``:env`` is optional. If given, ``:env [a b]``, ``:env {a a b b}`` and the
  malformed ``:env {a b c}`` are all accepted.
- ``total-cost`` and ``total_cost`` name the same variable (hyphens are
  normalised to underscores), so Lisp-style names work inside Python tools.

Semantics of the checks: ``assert`` and ``verify`` never abort the program.
They are recorded as ``Check`` entries and judged later by the verifier, so a
single run yields the complete list of failed checks.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable

from .parser import Keyword, MapLit, Symbol, Vector, to_source
from .tools import Tool, ToolError

UNSUPPORTED = {"decompose", "solve", "loop", "case", "simulate", "prove", "request_input"}
SUPPORTED_FORMS = ("program", "let", "return", "if", "cond", "assert", "compare",
                   "verify", "call_tool", "lookup")


class CDSLRuntimeError(RuntimeError):
    """Raised for any failure while executing a program."""


class _Return(Exception):
    def __init__(self, value: object):
        self.value = value


@dataclass
class Check:
    kind: str          # "assert" | "verify"
    source: str        # CDSL source of the check
    passed: bool
    detail: str = ""


@dataclass
class RunResult:
    ok: bool                      # finished without a runtime error
    value: object = None          # value given to (return ...)
    returned: bool = False
    error: str | None = None
    category: str | None = None   # the :type of the program
    checks: list[Check] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    steps: int = 0


def _numbers_equal(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _chain(op: Callable[[object, object], bool]) -> Callable[..., bool]:
    def fn(*xs: object) -> bool:
        if len(xs) < 2:
            raise CDSLRuntimeError("comparison needs at least two values")
        return all(op(a, b) for a, b in zip(xs, xs[1:]))
    return fn


def _minus(*xs):
    if len(xs) == 1:
        return -xs[0]
    out = xs[0]
    for x in xs[1:]:
        out -= x
    return out


def _div(*xs):
    out = xs[0]
    for x in xs[1:]:
        out /= x
    return out


def _mul(*xs):
    out = 1
    for x in xs:
        out *= x
    return out


def _flat(xs):
    return xs[0] if len(xs) == 1 and isinstance(xs[0], list) else list(xs)


COMPARE_OPS: dict[str, Callable[[object, object], bool]] = {
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    "=": _numbers_equal, "==": _numbers_equal,
    "!=": lambda a, b: not _numbers_equal(a, b),
}

BUILTINS: dict[str, Callable[..., object]] = {
    "+": lambda *xs: sum(xs), "-": _minus, "*": _mul, "/": _div,
    "//": lambda a, b: a // b, "%": lambda a, b: a % b, "**": lambda a, b: a ** b,
    "min": lambda *xs: min(_flat(xs)), "max": lambda *xs: max(_flat(xs)),
    "sum": lambda *xs: sum(_flat(xs)), "len": len, "abs": abs,
    "round": lambda x, n=None: round(x) if n is None else round(x, n),
    "floor": math.floor, "ceil": math.ceil, "sqrt": math.sqrt,
    "int": int, "float": float, "list": lambda *xs: list(xs),
    "range": lambda *xs: list(range(*xs)),
    "not": lambda x: not x,
    "and": lambda *xs: all(xs), "or": lambda *xs: any(xs),
    **{op: _chain(fn) for op, fn in COMPARE_OPS.items()},
}


def _split_kwargs(args: list) -> tuple[list, dict[str, object]]:
    """Separate ``:key value`` pairs from positional arguments."""
    positional, kwargs, i = [], {}, 0
    while i < len(args):
        a = args[i]
        if isinstance(a, Keyword):
            if i + 1 >= len(args):
                raise CDSLRuntimeError(f"keyword :{a.name} is missing its value")
            kwargs[a.name] = args[i + 1]
            i += 2
        else:
            positional.append(a)
            i += 1
    return positional, kwargs


def _norm(name: str) -> str:
    """Variable and form names: hyphens and underscores are interchangeable."""
    return name.replace("-", "_") if any(ch.isalpha() for ch in name) else name


_OPERATORS = {"+", "-", "*", "/", "//", "%", "**", "=", "==", "!=", "<", "<=", ">", ">="}
_PREFIX_HINT = ("Hint: CDSL is prefix-only, the operator comes first and every operation gets its "
                "own parentheses. Write (- (+ a b) 2), never (+ a b - 2) or (a + b).")
_LET_HINT = ("Hint: (let <name> <expr>) takes exactly ONE expression. Combine parts inside it, "
             "e.g. (let total (+ (* a b) (* c d))).")
_UNBOUND_HINT = ("Hint: every name must be bound with (let ...) before it is used. CDSL has no "
                 "unknowns and no equations: compute each value directly from earlier ones.")


def _undefined(name: str) -> str:
    """Error text for an unbound symbol, with a hint for the usual causes."""
    if name in _OPERATORS:
        return f"operator '{name}' used in infix position. {_PREFIX_HINT}"
    if any(ch.isdigit() for ch in name) and any(ch in "+-*/" for ch in name):
        return f"'{name}' is not a number. Write fractions and sums as forms, e.g. (/ 2 3). {_PREFIX_HINT}"
    return f"undefined variable '{name}'. {_UNBOUND_HINT}"


def _json_ok(value: object) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _short(value: object, limit: int = 80) -> str:
    s = repr(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."


class Runtime:
    """Executes a parsed ``(program ...)`` form."""

    def __init__(self, tools: dict[str, Tool] | None = None, memory=None,
                 max_steps: int = 20_000):
        self.tools = tools or {}
        self.memory = memory
        self.max_steps = max_steps

    # ---------------------------------------------------------------- entry point
    def run(self, program: list) -> RunResult:
        self._result = RunResult(ok=False)
        if not (isinstance(program, list) and program and program[0] == Symbol("program")):
            self._result.error = "top-level form must be (program ...)"
            return self._result
        body, opts = _split_kwargs(program[1:])
        cat = opts.get("type")
        self._result.category = cat.name if isinstance(cat, Symbol) else (str(cat) if cat else None)
        env: dict[str, object] = {}
        try:
            for form in body:
                try:
                    self._eval(form, env)
                except CDSLRuntimeError as e:
                    if "\n  in form:" not in str(e):
                        e = CDSLRuntimeError(f"{e}\n  in form: {to_source(form, 160)}")
                    raise e
            self._result.error = "program ended without (return ...)"
        except _Return as r:
            self._result.ok, self._result.returned, self._result.value = True, True, r.value
        except CDSLRuntimeError as e:
            self._result.error = str(e)
        except RecursionError:
            self._result.error = "program nested too deeply"
        return self._result

    # ------------------------------------------------------------------ evaluator
    def _eval(self, node: object, env: dict) -> object:
        self._result.steps += 1
        if self._result.steps > self.max_steps:
            raise CDSLRuntimeError(f"step limit exceeded ({self.max_steps})")
        if isinstance(node, Symbol):
            if _norm(node.name) in env:
                return env[_norm(node.name)]
            raise CDSLRuntimeError(_undefined(node.name))
        if isinstance(node, Keyword):
            return node
        if isinstance(node, MapLit):
            if node.dangling is not None:
                raise CDSLRuntimeError("map literal has an odd number of elements")
            return {self._key(k): self._eval(v, env) for k, v in node}
        if isinstance(node, Vector):
            return [self._eval(x, env) for x in node]
        if not isinstance(node, list):
            return node  # literal
        if not node:
            raise CDSLRuntimeError("empty form ()")
        head = node[0]
        if not isinstance(head, Symbol):
            raise CDSLRuntimeError(f"a form must start with a name, got {to_source(head, 40)}")
        name, args = head.name, node[1:]
        special = getattr(self, "_f_" + _norm(name), None) if _norm(name).isidentifier() else None
        if special is not None:
            return special(args, env)
        if name in UNSUPPORTED:
            raise CDSLRuntimeError(
                f"primitive '{name}' is not implemented in the v0.1 runtime; "
                f"use only: {', '.join(SUPPORTED_FORMS)} and arithmetic")
        if name == "program":
            raise CDSLRuntimeError("(program ...) can only appear at the top level")
        fn = BUILTINS.get(name)
        if fn is None:
            raise CDSLRuntimeError(
                f"unknown form or function '{name}'; valid forms: {', '.join(SUPPORTED_FORMS)}")
        values = [self._eval(a, env) for a in args]
        try:
            return fn(*values)
        except CDSLRuntimeError:
            raise
        except Exception as e:
            shown = " ".join(_short(v, 30) for v in values)
            raise CDSLRuntimeError(f"({name} {shown}) failed: {type(e).__name__}: {e}")

    def _key(self, k: object) -> str:
        if isinstance(k, (Symbol, Keyword)):
            return _norm(k.name)
        if isinstance(k, str):
            return k
        raise CDSLRuntimeError(f"map keys must be names, got {to_source(k, 40)}")

    # -------------------------------------------------------------- special forms
    def _f_let(self, args: list, env: dict) -> object:
        if len(args) != 2 or not isinstance(args[0], Symbol):
            raise CDSLRuntimeError(f"let expects (let <name> <expr>). {_LET_HINT}")
        value = self._eval(args[1], env)
        env[_norm(args[0].name)] = value
        self._result.trace.append(f"let {_norm(args[0].name)} = {_short(value)}")
        return value

    def _f_return(self, args: list, env: dict) -> object:
        if len(args) != 1:
            raise CDSLRuntimeError("return expects exactly one value: (return <expr>)")
        value = self._eval(args[0], env)
        self._result.trace.append(f"return {_short(value)}")
        raise _Return(value)

    def _f_if(self, args: list, env: dict) -> object:
        if len(args) not in (2, 3):
            raise CDSLRuntimeError("if expects (if <cond> <then> <else>)")
        if self._eval(args[0], env):
            return self._eval(args[1], env)
        return self._eval(args[2], env) if len(args) == 3 else None

    def _f_cond(self, args: list, env: dict) -> object:
        for clause in args:
            if not isinstance(clause, list) or len(clause) != 2:
                raise CDSLRuntimeError("cond clauses must be [<test> <branch>]")
            test, branch = clause
            is_else = test in (Symbol("else"), Keyword("else"))
            if is_else or self._eval(test, env):
                return self._eval(branch, env)
        return None

    def _f_compare(self, args: list, env: dict) -> bool:
        positional, kw = _split_kwargs(args)
        op = kw.get("op")
        if len(positional) == 3 and op is None:          # (compare a > b)
            positional, op = [positional[0], positional[2]], positional[1]
        op_name = op.name if isinstance(op, Symbol) else op
        if len(positional) != 2 or op_name not in COMPARE_OPS:
            raise CDSLRuntimeError(
                "compare expects (compare <a> :op <op> <b>) with op in > >= < <= = !=, where "
                f"<a> and <b> are each a single value or form. {_PREFIX_HINT}")
        a, b = (self._eval(x, env) for x in positional)
        try:
            return COMPARE_OPS[op_name](a, b)
        except TypeError as e:
            raise CDSLRuntimeError(f"cannot compare {_short(a, 30)} {op_name} {_short(b, 30)}: {e}")

    def _f_assert(self, args: list, env: dict) -> bool:
        if not args:
            raise CDSLRuntimeError("assert expects (assert <condition>)")
        passed = bool(self._eval(args[0], env))
        src = to_source(args[0], 120)
        self._result.checks.append(Check("assert", src, passed, "" if passed else f"{src} is false"))
        self._result.trace.append(f"assert {src} -> {'ok' if passed else 'FAILED'}")
        return passed

    def _f_verify(self, args: list, env: dict) -> bool:
        positional, kw = _split_kwargs(args)
        if len(positional) != 1 or "against" not in kw:
            raise CDSLRuntimeError("verify expects (verify <value> :against <expr>)")
        actual = self._eval(positional[0], env)
        expected = self._eval(kw["against"], env)
        passed = _numbers_equal(actual, expected)
        src = f"(verify {to_source(positional[0], 60)} :against {to_source(kw['against'], 80)})"
        detail = "" if passed else f"value {_short(actual, 40)} != recomputed {_short(expected, 40)}"
        self._result.checks.append(Check("verify", src, passed, detail))
        self._result.trace.append(f"verify {_short(actual, 40)} vs {_short(expected, 40)} -> "
                                  f"{'ok' if passed else 'FAILED'}")
        return passed

    def _f_call_tool(self, args: list, env: dict) -> object:
        positional, kw = _split_kwargs(args)
        if not positional:
            raise CDSLRuntimeError("call_tool expects (call_tool <tool> <args...> :env {...})")
        tool_ref = positional[0]
        name = tool_ref.name if isinstance(tool_ref, Symbol) else tool_ref
        tool = self.tools.get(name) if isinstance(name, str) else None
        if tool is None:
            raise CDSLRuntimeError(
                f"unknown tool {to_source(tool_ref, 40)}; available: {', '.join(sorted(self.tools))}")
        tool_args = [self._eval(a, env) for a in positional[1:]]
        tool_env = self._tool_env(kw.get("env"), env)
        try:
            value = tool(tool_args, tool_env)
        except ToolError as e:
            raise CDSLRuntimeError(f"tool {name} failed: {e}")
        self._result.trace.append(f"call_tool {name} {_short(tool_args[0] if tool_args else '', 60)} -> {_short(value)}")
        return value

    def _tool_env(self, spec: object, env: dict) -> dict:
        """Variables a tool can see: every bound variable, plus explicit renames."""
        tool_env = {k: v for k, v in env.items() if k.isidentifier() and _json_ok(v)}
        if spec is None:
            return tool_env
        if isinstance(spec, MapLit):
            for k, v in spec:
                key = self._key(k)
                if key not in env:                    # {n count}: expose `count` as `n`
                    tool_env[key] = self._eval(v, env)
            if isinstance(spec.dangling, Symbol):
                self._eval(spec.dangling, env)        # must at least be defined
        elif isinstance(spec, list):                  # [a b c]
            for item in spec:
                self._eval(item, env)
        else:
            raise CDSLRuntimeError(":env must be a list of names like [a b]")
        return tool_env

    def _f_lookup(self, args: list, env: dict) -> list[str]:
        if len(args) != 2:
            raise CDSLRuntimeError('lookup expects (lookup <factual|procedural|causal|episodic> "<query>")')
        bank = args[0].name if isinstance(args[0], (Symbol, Keyword)) else str(args[0])
        query = self._eval(args[1], env)
        if self.memory is None:
            hits = []
        else:
            try:
                hits = [h.text for h in self.memory.search(bank, str(query), k=3)]
            except KeyError as e:
                raise CDSLRuntimeError(str(e))
        self._result.trace.append(f"lookup {bank} {_short(query, 40)} -> {len(hits)} notes")
        return hits
