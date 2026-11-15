"""Chain of thought with a calculator: the simple rival of the program path.

The model reasons in free text, as in plain CoT, but writes every calculation
inside double angle brackets, the annotation GSM8K itself uses:

    The baker bakes 4 * 12 = <<4*12=48>>48 muffins.

Generation is stopped at each closing ">>". The machine evaluates the
expression exactly, overwrites whatever result the model wrote, and generation
continues from the corrected text. So the model reads the problem and decides
what to compute, and the machine computes. No program, no plan, no twin.

It needs a raw completion endpoint, because the half-written answer has to be
continued (``LocalLLM.complete`` with ``chat_prompt``).
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

from .arbiter import COT_PROMPT, CotResult, extract_final_number
from .llm import chat_prompt
from .verifier import is_number

_INSTRUCTION = (
    "You have a calculator. Every time you compute something, write the calculation inside "
    "double angle brackets, like <<12*4=48>>. The calculator checks it and writes the correct "
    "result, so always copy the result from the brackets and never work a number out in your "
    "head. Inside the brackets use only numbers and + - * / ( ).")

CALC_PROMPT = COT_PROMPT.replace("\n\nProblem: {q}", " " + _INSTRUCTION + "\n\nProblem: {q}")

_EXAMPLE = (
    "A baker bakes 4 trays of muffins with 12 muffins on each tray. She sells 30 muffins "
    "and gives 5 to her neighbor. How many muffins does she have left?",
    "The baker bakes 4 trays of 12 muffins: 4 * 12 = <<4*12=48>>48 muffins.\n"
    "She sells 30 and gives 5 away: 48 - 30 - 5 = <<48-30-5=13>>13 muffins are left.\n"
    "#### 13",
)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _calc(node):
    if isinstance(node, ast.Expression):
        return _calc(node.body)
    if isinstance(node, ast.Constant) and is_number(node.value):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.left), _calc(node.right))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        base, exp = _calc(node.left), _calc(node.right)
        if abs(exp) > 12 or abs(base) > 1e9:
            raise ValueError("power too large")
        return base ** exp
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.operand))
    raise ValueError("not plain arithmetic")


def evaluate(expression: str) -> float | None:
    """Exact value of an arithmetic expression as a model writes it, or None."""
    t = expression.replace("$", "").replace("\\", "").replace("×", "*").replace("÷", "/")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)                  # 1,500 -> 1500
    t = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", t)     # 3 x 4 -> 3*4
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", t)       # 40% -> (40/100)
    t = t.replace("^", "**").strip()
    if not t or len(t) > 200:
        return None
    try:
        value = _calc(ast.parse(t, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
        return None
    return value if is_number(value) else None


def show(value: float) -> str:
    """A result as the calculator writes it into the text."""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    if abs(value) < 1e-4:
        return f"{value:.10g}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


@dataclass
class CalcCotResult(CotResult):
    rounds: int = 0          # generation segments
    calculations: int = 0    # bracketed expressions the machine evaluated
    corrected: int = 0       # of those, how many results the model had written wrong


def solve_calc_cot(llm, problem: str, max_tokens: int = 900, max_rounds: int = 40,
                   seed: int | None = 0) -> CalcCotResult:
    messages = [{"role": "user", "content": CALC_PROMPT.format(q=_EXAMPLE[0])},
                {"role": "assistant", "content": _EXAMPLE[1]},
                {"role": "user", "content": CALC_PROMPT.format(q=problem)}]
    base = chat_prompt(llm.model, messages)
    text, tokens, seconds, rounds, calculations, corrected = "", 0, 0.0, 0, 0, 0
    while rounds < max_rounds and tokens < max_tokens:
        res = llm.complete(base + text, stop=[">>"], max_tokens=max_tokens - tokens,
                           temperature=0.0, seed=seed)
        rounds += 1
        tokens += res.completion_tokens
        seconds += res.seconds
        text += res.text
        start = text.rfind("<<")
        if start <= text.rfind(">>"):            # no open bracket: the model finished
            break
        expression, _, stated = text[start + 2:].partition("=")
        value = evaluate(expression)
        if value is None:                        # not arithmetic: leave it as written
            text += ">>"
            continue
        calculations += 1
        said = evaluate(stated) if stated.strip() else None
        corrected += said is not None and abs(said - value) > 1e-9 * max(1.0, abs(value))
        text = text[:start] + "<<" + expression.strip() + "=" + show(value) + ">>"
    return CalcCotResult(text, extract_final_number(text), seconds, tokens, rounds, calculations, corrected)
