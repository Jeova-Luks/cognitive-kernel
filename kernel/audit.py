"""Audit of a chain-of-thought text: the machine redoes the arithmetic it wrote.

A CoT solution states its arithmetic as "expression = value". With large
numbers the model often reads the problem correctly and then writes a false
equation. This module recomputes every equation it can parse, carries the
exact value forward wherever the model reuses a value it stated, and returns
the final answer the same reasoning gives with exact arithmetic.

No LLM call. It only redoes arithmetic: a wrong reading stays wrong.
"""
from __future__ import annotations

import ast
import operator
import re

from .verifier import is_number

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}

_NUM = r"\d+(?:\.\d+)?"
_EQUATION = re.compile(rf"((?:\(?\s*{_NUM}\s*\)?\s*[-+*/]\s*)+\(?\s*{_NUM}\s*\)?)\s*=\s*({_NUM})")
_TOKEN = re.compile(_NUM)


def _calc(node):
    if isinstance(node, ast.Expression):
        return _calc(node.body)
    if isinstance(node, ast.Constant) and is_number(node.value):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.left), _calc(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.operand))
    raise ValueError("not plain arithmetic")


def _normalise(text: str) -> str:
    t = text.replace("$", "").replace("\\", "").replace("×", "*").replace("÷", "/")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)                 # 1,500 -> 1500
    return re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", t)     # 3 x 4 -> 3 * 4


def equations(text: str) -> list[tuple[str, float, float]]:
    """(expression, value the text states, exact value) for every equation that parses."""
    carried: dict[str, float] = {}      # a stated value, as written -> its exact value
    out = []
    for lhs, rhs in _EQUATION.findall(_normalise(text)):
        expr = _TOKEN.sub(lambda m: repr(carried.get(m.group(), float(m.group()))), lhs)
        try:
            exact = _calc(ast.parse(expr.strip(), mode="eval"))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        carried[rhs] = exact
        out.append((lhs.strip(), float(rhs), exact))
    return out


def audited_answer(text: str, final: object = None) -> float | None:
    """The final answer of a CoT text with its arithmetic redone, or None if no equation parses.

    ``final`` is the answer the text reports. The audited answer is the exact
    value of the last equation whose stated value is that answer; when no
    equation states it, the exact value of the last equation.
    """
    from .arbiter import numbers_match      # imported here: arbiter imports this module

    found = equations(text)
    if not found:
        return None
    if is_number(final):
        for _, stated, exact in reversed(found):
            if numbers_match(stated, final):
                return exact
    return found[-1][2]
