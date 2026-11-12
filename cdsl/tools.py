"""Tools reachable from CDSL via ``(call_tool <name> <args...> :env {...})``.

Every tool has the signature ``tool(args: list, env: dict) -> value`` and
raises ``ToolError`` on failure.

SECURITY NOTE - ``python_exec`` runs model-generated Python in a separate
``python -I`` process with a wall-clock timeout and a throwaway working
directory. That contains crashes and infinite loops; it is NOT a security
boundary (no memory cap on Windows, full filesystem access). It is meant for
a cooperative local model, the same threat model as cognitive_units/sandbox.py.
"""
from __future__ import annotations

import ast
import json
import math
import operator
import subprocess
import sys
import tempfile
from typing import Callable

Tool = Callable[[list, dict], object]


class ToolError(RuntimeError):
    """A tool call failed (bad input, exception inside the tool, timeout)."""


# --------------------------------------------------------------------- python_exec

_RESULT_MARKER = "__CDSL_RESULT__"

# Runs inside the child process. Reads {"code", "env"} as JSON on stdin.
# If the code is statements, the value is the last bare expression, or else
# the variable `result`.
_RUNNER = r'''
import ast, json, math, sys, traceback
payload = json.loads(sys.stdin.read())
ns = {"math": math}
ns.update(payload["env"])
try:
    tree = ast.parse(payload["code"], mode="exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<python_exec>", "exec"), ns)
    value = eval(compile(last, "<python_exec>", "eval"), ns) if last else ns.get("result")
except Exception as e:
    sys.stderr.write(traceback.format_exception_only(type(e), e)[-1].strip())
    sys.exit(1)
def plain(v):
    if isinstance(v, (set, frozenset, tuple, range)):
        return [plain(x) for x in v]
    if isinstance(v, list):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    return v
try:
    out = json.dumps({"value": plain(value)})
except TypeError:
    out = json.dumps({"value": repr(value)})
sys.stdout.write("\n__CDSL_RESULT__" + out)
'''


def python_exec(args: list, env: dict, timeout: float = 10.0) -> object:
    if len(args) != 1 or not isinstance(args[0], str):
        raise ToolError('python_exec expects exactly one string of Python code: (call_tool python_exec "<code>" :env {...})')
    try:
        payload = json.dumps({"code": args[0], "env": env})
    except TypeError as e:
        raise ToolError(f"python_exec :env values must be numbers, strings, lists or maps ({e})")
    with tempfile.TemporaryDirectory(prefix="cdsl_") as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER],
                input=payload, capture_output=True, text=True,
                timeout=timeout, cwd=cwd, encoding="utf-8",
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"python_exec timed out after {timeout:.0f}s")
    if proc.returncode != 0:
        raise ToolError(f"python_exec raised {proc.stderr.strip() or 'an error (exit code %d)' % proc.returncode}")
    _, sep, tail = proc.stdout.rpartition(_RESULT_MARKER)
    if not sep:
        raise ToolError("python_exec produced no result")
    return json.loads(tail)["value"]


# ---------------------------------------------------------------------- calculator

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max,
          "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil}


def _calc(node: ast.AST, env: dict) -> object:
    if isinstance(node, ast.Expression):
        return _calc(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ToolError(f"calculator: unknown variable '{node.id}' (pass it with :env)")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _calc(node.left, env), _calc(node.right, env)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ToolError("calculator: exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_calc(node.operand, env))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_calc(a, env) for a in node.args])
    raise ToolError(f"calculator: unsupported expression element {type(node).__name__}")


def calculator(args: list, env: dict) -> object:
    if len(args) != 1 or not isinstance(args[0], str):
        raise ToolError('calculator expects one arithmetic string: (call_tool calculator "<expr>" :env {...})')
    try:
        tree = ast.parse(args[0].replace("^", "**"), mode="eval")
        return _calc(tree, env)
    except ToolError:
        raise
    except SyntaxError as e:
        raise ToolError(f"calculator: syntax error in {args[0]!r}: {e.msg}")
    except Exception as e:  # ZeroDivisionError, TypeError, ...
        raise ToolError(f"calculator: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- sympy

def _sympy_tool(args: list, env: dict) -> object:
    import sympy  # optional dependency, checked in available_tools()

    if len(args) != 1 or not isinstance(args[0], str):
        raise ToolError('sympy expects one expression string: (call_tool sympy "<expr>" :env {...})')
    try:
        expr = sympy.sympify(args[0], locals={k: sympy.sympify(v) for k, v in env.items()})
        value = sympy.simplify(expr)
        if value.is_number:
            return int(value) if value.is_integer else float(value)
        return str(value)
    except Exception as e:
        raise ToolError(f"sympy: {type(e).__name__}: {e}")


def available_tools() -> dict[str, Tool]:
    tools: dict[str, Tool] = {"python_exec": python_exec, "calculator": calculator}
    try:
        import sympy  # noqa: F401
        tools["sympy"] = _sympy_tool
    except ImportError:
        pass
    return tools
