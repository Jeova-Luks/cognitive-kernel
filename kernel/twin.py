"""Small-number twins of a problem with large numbers, fabricated by code.

Why: a program computes exactly at any number size, so whether it READ the
problem right can be tested where reading is easy to check. A twin is the
same text with every large number replaced by a small one. A chain of thought
solves the twin reliably. If the program, run with the twin's numbers, gives
the same answer as that chain of thought, the two derivations read the
problem the same way, and the program's answer for the real numbers can be
trusted.

How a replacement value is chosen (no LLM call, no access to any label):
- only numbers with ARITHMETIC_LOAD_THRESHOLD or more significant digits are
  replaced, the same test the rule referee uses;
- the new value keeps the place of the number among the other numbers of the
  text, so "more seats than reserved seats" stays true;
- it is not a number already in the text, nor a value problems use without
  saying so (7 days, 12 months, 24 hours, 60 minutes, 100 percent, ...);
- two twins are made. "round" uses a multiple of 60, so halves, thirds,
  quarters, fifths and tenths come out whole and the chain of thought has
  easy arithmetic. "odd" uses a prime, so a program that skips a rounding the
  problem needs (whole buses, full boxes) does not get the same answer.

A program is tested on a twin by replacing, in its text, the numeric literals
equal to the large numbers. If the program does not contain the large number
as a literal the test does not apply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .arbiter import ARITHMETIC_LOAD_THRESHOLD
from .verifier import is_number

IMPLICIT = frozenset({7, 12, 24, 30, 52, 60, 100, 365, 1000})
KINDS = ("round", "odd")
MARGIN = 2.0          # a replacement must exceed MARGIN times the largest smaller number of the text

_TOKEN = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")
_LITERAL = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w]|\.\d)")
_PERCENT = re.compile(r"\s*(%|percent\b)")
_PERCENT_VALUES = {"round": (20, 40, 25, 50, 75, 80, 10), "odd": (13, 17, 19, 23, 29, 31, 37, 41, 43, 47)}


@dataclass
class Twin:
    kind: str                       # "round" or "odd"
    text: str                       # the problem with the large numbers replaced
    mapping: dict[float, float]     # large value -> small value


def significant_digits(token: str) -> int:
    t = token.replace(",", "")
    t = t.rstrip("0") if "." in t else (t.rstrip("0") or "0")
    return len(t.replace(".", "").lstrip("0"))


def _numbers(text: str) -> list[tuple[int, int, float, str]]:
    """(start, end, value, token) of every number written in the text."""
    out = []
    for m in _TOKEN.finditer(text):
        token = m.group().rstrip(",")
        out.append((m.start(), m.start() + len(token), float(token.replace(",", "")), token))
    return out


def _sieve(limit: int) -> list[int]:
    flags = [True] * (limit + 1)
    flags[0:2] = [False, False]
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            flags[i * i::i] = [False] * len(flags[i * i::i])
    return [i for i, f in enumerate(flags) if f]


_PRIMES = [p for p in _sieve(997) if p >= 13]


def _candidates(kind: str):
    """Replacement values in ascending order.

    round: multiples of 60 with at most two significant digits (120, 180, 240, 300, ...).
    odd:   the primes from 13 to 997, then two-digit primes times a power of ten (1300, 1700, ...).
    """
    if kind == "odd":
        yield from _PRIMES
    scale = 1
    while scale <= 10 ** 9:
        for m in range(10, 100):
            value = m * scale
            if kind == "round" and value % 60 == 0:
                yield value
            if kind == "odd" and scale >= 100 and m in _PRIMES:
                yield value
        scale *= 10


def _choose(kind: str, lower: float, upper: float, taken: set[float], percent: bool) -> int | None:
    pool = _PERCENT_VALUES[kind] if percent else _candidates(kind)
    for value in pool:
        if not percent and value >= upper:
            return None
        if (percent or value > lower * MARGIN) and value not in taken and value not in IMPLICIT:
            return value
    return None


def fabricate(problem: str) -> list[Twin]:
    """The round and the odd twin of a problem; [] when it has no large number.

    A kind is left out when no value fits (for instance no room between two
    other numbers of the text).
    """
    numbers = _numbers(problem)
    large = sorted({v for _, _, v, token in numbers
                    if significant_digits(token) >= ARITHMETIC_LOAD_THRESHOLD})
    if not large:
        return []
    others = sorted({v for _, _, v, _ in numbers} - set(large))
    percent = {v: any(value == v and _PERCENT.match(problem, end) for _, end, value, _ in numbers)
               for v in large}
    twins = []
    for kind in KINDS:
        taken, mapping, floor = set(others), {}, 0.0
        for value in large:                                   # ascending, so their order is kept
            lower = max([o for o in others if o < value] + [floor])
            upper = min([o for o in others if o > value], default=float("inf"))
            small = _choose(kind, lower, upper, taken, percent[value])
            if small is None:
                break
            mapping[value] = float(small)
            taken.add(small)
            floor = max(floor, small)
        if len(mapping) == len(large):
            twins.append(Twin(kind, rewrite(problem, mapping), mapping))
    return twins


def _plain(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def rewrite(problem: str, mapping: dict[float, float]) -> str:
    """Replace whole number tokens (never substrings) whose value is a key of ``mapping``."""
    out, last = [], 0
    for start, end, value, _ in _numbers(problem):
        if value in mapping:
            out.append(problem[last:start] + _plain(mapping[value]))
            last = end
    return "".join(out) + problem[last:]


def substitute(program: str, mapping: dict[float, float]) -> tuple[str, int]:
    """Replace the numeric literals of a program equal to a key of ``mapping``.

    Returns the new program text and how many literals were replaced.
    """
    count = 0

    def swap(m: re.Match) -> str:
        nonlocal count
        value = float(m.group(1))
        for old, new in mapping.items():
            if abs(value - old) <= 1e-9 * max(1.0, abs(old)):
                count += 1
                return _plain(new)
        return m.group(0)

    return _LITERAL.sub(swap, program), count


def run_with(target, program: str | None, mapping: dict[float, float]) -> float | None:
    """The value a program returns when its large numbers are replaced, or None.

    None when the program is missing, does not contain every large number as a
    literal, or does not run to a numeric answer. ``target`` is the CDSLTarget
    or PythonTarget the program was written for.
    """
    if not program:
        return None
    for old in mapping:                       # every large number must be present
        if substitute(program, {old: mapping[old]})[1] == 0:
            return None
    text, _ = substitute(program, mapping)
    try:
        _, parsed = target.parse(text)
    except ValueError:
        return None
    run = target.run(parsed)
    return float(run.value) if run.returned and is_number(run.value) else None
