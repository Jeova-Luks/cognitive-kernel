"""Conditions of the 2026-11 experiment, computed from saved pieces. No LLM call.

``scripts/exp_twin_run.py`` saves, for every problem and model, what the model
wrote: a chain of thought, a plan and its program, a chain of thought with a
calculator, a chain of thought on each fabricated twin, a second reading, and
the value each program returns with the twin's numbers. Every function here
turns the records of one problem into an ``Outcome``: the answer a system
would give, whether that answer counts as confirmed, and what it cost.

Single model (``rec`` is that model's record):

    cot_only        1  plain chain of thought
    program_only    2  plan -> CDSL, executed, no check
    kernel_today    3  program checked against the chain of thought, rule referee
    kernel_audit    4  the same, and the chain of thought's arithmetic is redone by machine
    calc_only       5  chain of thought with a calculator
    rival           6  program checked against the calculator chain of thought
    verified(rec)   7  audit, then the twin test, then a second reading

Two models (``verified(base, other)``, condition 8): the same procedure as 7,
but the first program is checked by the OTHER model's chains of thought, and
the second reading is the other model's program, checked by the base model's
chains of thought. Whatever is left unconfirmed gets the base model's answer
from today's kernel.

"Confirmed" always means: two derivations that could fail in different ways
gave the same number after the machine did the arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass

from kernel.arbiter import ARITHMETIC_LOAD_THRESHOLD, arithmetic_load, numbers_match
from kernel.audit import audited_answer
from kernel.twin import KINDS
from kernel.verifier import is_number


@dataclass
class Outcome:
    answer: object
    confirmed: str | None      # None, or how the answer was confirmed
    tokens: int                # completion tokens the system needed for this problem


def _audited(cot: dict):
    return audited_answer(cot["text"], cot["answer"])


def agrees(value: object, cot: dict, audit: bool = True) -> str | None:
    """'agree' if a program value equals a chain of thought's answer, 'audit' if it equals
    that answer with the arithmetic redone, else None."""
    if not is_number(value):
        return None
    if numbers_match(value, cot["answer"]):
        return "agree"
    if audit and numbers_match(value, _audited(cot)):
        return "audit"
    return None


def rule(rec: dict):
    """Today's referee: trust the program when the problem has a large number, else the chain of thought."""
    program, cot = rec["first"]["answer"], rec["cot"]["answer"]
    if not is_number(program):
        return cot
    if not is_number(cot):
        return program
    return program if arithmetic_load(rec["question"]) >= ARITHMETIC_LOAD_THRESHOLD else cot


def _tok(piece: dict | None) -> int:
    return piece["completion_tokens"] if piece else 0


def _base_tokens(rec: dict) -> int:
    return _tok(rec["read"]) + _tok(rec["first"])


def cot_only(rec: dict) -> Outcome:
    return Outcome(rec["cot"]["answer"], None, _tok(rec["cot"]))


def program_only(rec: dict) -> Outcome:
    return Outcome(rec["first"]["answer"], None, _base_tokens(rec))


def kernel_today(rec: dict, audit: bool = False) -> Outcome:
    program = rec["first"]["answer"]
    how = agrees(program, rec["cot"], audit=audit)
    return Outcome(program if how else rule(rec), how, _base_tokens(rec) + _tok(rec["cot"]))


def kernel_audit(rec: dict) -> Outcome:
    return kernel_today(rec, audit=True)


def calc_only(rec: dict) -> Outcome:
    return Outcome(rec["calc"]["answer"], None, _tok(rec["calc"]))


def rival(rec: dict, prefer: str = "program") -> Outcome:
    """Program checked against the calculator chain of thought. On a disagreement ``prefer``
    says whose answer stands: 'program' or 'calc'."""
    program, calc = rec["first"]["answer"], rec["calc"]["answer"]
    how = agrees(program, rec["calc"])
    if how:
        answer = program
    elif prefer == "program":
        answer = program if is_number(program) else calc
    else:
        answer = calc if is_number(calc) else program
    return Outcome(answer, how, _base_tokens(rec) + _tok(rec["calc"]))


def _twins(rec: dict) -> dict[str, dict]:
    return {t["kind"]: t for t in rec["twins"]}


def twin_test(owner: dict, key: str, oracle: dict, kinds: tuple[str, ...] = KINDS) -> bool:
    """The program ``owner[key]``, run with each twin's numbers, matches the oracle model's
    chain of thought on that twin. Every kind in ``kinds`` must exist and match."""
    mine, theirs = _twins(owner), _twins(oracle)
    if not kinds:
        return False
    for kind in kinds:
        if kind not in mine or kind not in theirs:
            return False
        if not agrees(mine[kind][key], theirs[kind]["cot"]):
            return False
    return True


def verified(base: dict, other: dict | None = None, kinds: tuple[str, ...] = KINDS,
             programs_agree: bool = False) -> Outcome:
    """Conditions 7 (``other`` is None) and 8.

    Candidates, in order: the base model's program, then a second reading (the
    base model's own second reading, or the other model's program). A candidate
    is confirmed if it matches the checking model's chain of thought on the
    real numbers (as written, or with the arithmetic redone), or if it passes
    the twin test against the checking model's chains of thought. The checker
    of the base program is the other model when there is one; the checker of
    the other model's program is the base model.

    ``programs_agree`` (exploratory): also confirm when the two programs return
    the same value for the real numbers.
    """
    if other is None:
        candidates = [(base, "first", base), (base, "second", base)]
    else:
        candidates = [(base, "first", other), (other, "first", base)]
    tokens = _base_tokens(base) + _tok(candidates[0][2]["cot"])
    counted = {(id(candidates[0][2]), "cot")}

    def charge(rec: dict, what: str) -> int:
        if (id(rec), what) in counted:
            return 0
        counted.add((id(rec), what))
        if what == "cot":
            return _tok(rec["cot"])
        return sum(_tok(t["cot"]) for t in rec["twins"])

    for position, (owner, key, oracle) in enumerate(candidates):
        program = owner[key]
        if position == 1:                          # the second reading has to be produced
            tokens += _tok(program) if other is None else _base_tokens(owner)
            tokens += charge(oracle, "cot")
        if not program or not is_number(program["answer"]):
            continue
        how = agrees(program["answer"], oracle["cot"])
        if not how:
            tokens += charge(oracle, "twins")
            if twin_test(owner, key, oracle, kinds):
                how = "twin"
        if how:
            return Outcome(program["answer"], how if position == 0 else how + ":second", tokens)
    if programs_agree:
        a, b = (owner[key] for owner, key, _ in candidates)
        if a and b and numbers_match(a["answer"], b["answer"]):
            return Outcome(a["answer"], "programs", tokens)
    return Outcome(rule(base), None, tokens)
