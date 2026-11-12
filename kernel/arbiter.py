"""Cross-check and arbitration: verification by an independent derivation.

1. The kernel (plan -> program, executed) and a plain chain-of-thought (CoT)
   solve the problem independently; neither sees the other.
2. If the two answers agree, the answer is CONFIRMED (confidence "high"). On
   the dev set agreement was right 31 times out of 31, against 75% for
   programs that passed their own self-written checks.
3. If they disagree, a referee decides (confidence "low"). Three referees
   exist; all were measured on the dev set (GSM8K 50 / GSM-Hard 23 problems):

   - ``rule`` (default): no LLM call. Trust the program when the problem
     contains a number with ARITHMETIC_LOAD_THRESHOLD or more significant
     digits, i.e. when the arithmetic is too heavy to do reliably in the
     head; otherwise trust CoT, the better reader. Dev: 46/50 and 17/23.
   - ``choose``: the LLM is shown both solutions and must pick one. It sided
     with the program 12 times out of 13 on GSM8K and was right twice.
     Dev: 40/50 and 15/23.
   - ``resolve``: the LLM solves the problem again and may return any value.
     It invented a third value twice on GSM8K and was wrong both times.

   Why a rule beats the LLM referee here: which solver to trust depends on
   whether the disagreement is about reading or about arithmetic, and the 8B
   model cannot tell. Number size can. Caveat: number size is exactly what
   separates GSM8K from GSM-Hard, so the rule's robustness across those two
   benchmarks is partly by construction.

The two solvers fail differently, which is what makes a disagreement
informative: the program has exact arithmetic but may misread the problem;
CoT reads the problem more freely but can slip on arithmetic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .verifier import is_number

COT_PROMPT = ("Solve the following math problem step by step. At the end, write the final answer "
              "as a single number on its own line in the form '#### <number>'.\n\nProblem: {q}")

_SOLUTIONS = """Problem:
{problem}

Solver A wrote a plan and a program, and a machine executed the program. Its arithmetic is exact, but the plan may misread the problem.
{plan}{program}
Execution trace:
{trace}
Answer A: {a}

Solver B reasoned step by step. It may misread the problem or make an arithmetic slip.
{cot}
Answer B: {b}"""

_CHOOSE_PROMPT = """Two independent solvers gave different answers to the same problem. At least one of them is wrong. Your job is to pick the right one, not to solve the problem from scratch.

""" + _SOLUTIONS + """

Compare the two solutions:
1. Find the first step where A and B treat the problem differently.
2. Quote the sentence of the problem that step is about, and say which solver follows it.
3. If both read the problem the same way and only the arithmetic differs, A is right, because A's arithmetic was done by a machine.
Finish with exactly one line: 'CHOICE: A' or 'CHOICE: B'."""

_RESOLVE_PROMPT = """Two independent solvers gave different answers to the same problem. At least one of them is wrong.

""" + _SOLUTIONS + """

Solve the problem again yourself, step by step. For each quantity, check which solver read the problem text correctly. Then write the final answer as a single number on its own line in the form '#### <number>'."""


ARITHMETIC_LOAD_THRESHOLD = 5


def arithmetic_load(problem: str) -> int:
    """Most significant digits of any number in the problem text ("1,500" -> 2, "6375793" -> 7)."""
    best = 0
    for token in re.findall(r"\d[\d,]*\.?\d*", problem):
        t = token.rstrip(".,").replace(",", "")
        t = t.rstrip("0") if "." in t else (t.rstrip("0") or "0")
        best = max(best, len(t.replace(".", "").lstrip("0")))
    return best


@dataclass
class CotResult:
    text: str
    answer: float | None
    seconds: float
    completion_tokens: int


@dataclass
class Arbitration:
    answer: object
    # agree | rule:cdsl | rule:cot | referee:cdsl | referee:cot | referee:new | cot-default
    # | cot-only | cdsl-only | unchecked | none
    source: str
    referee: CotResult | None = None

    @property
    def confidence(self) -> str:
        """high: two independent derivations agree. low: one derivation, or a referee's pick."""
        if self.source == "agree":
            return "high"
        return "none" if self.source == "none" else "low"


def parse_number(text: str) -> float | None:
    m = re.search(r"-?\d[\d,]*\.?\d*|-?\.\d+", text.replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    """The number after the last '####', else the last number in the text."""
    marked = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+(?:[eE][+-]?\d+)?)", text)
    if marked:
        try:
            return float(marked[-1].replace(",", ""))
        except ValueError:
            return parse_number(marked[-1])
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return parse_number(nums[-1]) if nums else None


def numbers_match(a: object, b: object) -> bool:
    """Same number, allowing for float noise and for one side being rounded to cents."""
    if not (is_number(a) and is_number(b)):
        return False
    a, b = float(a), float(b)
    if abs(a - b) <= max(1e-4, 1e-6 * max(abs(a), abs(b))):
        return True
    return (not a.is_integer() or not b.is_integer()) and round(a, 2) == round(b, 2)


answers_agree = numbers_match


def solve_cot(llm, problem: str, temperature: float = 0.0, seed: int | None = 0) -> CotResult:
    res = llm.chat([{"role": "user", "content": COT_PROMPT.format(q=problem)}],
                   temperature=temperature, max_tokens=700, seed=seed)
    return CotResult(res.text, extract_final_number(res.text), res.seconds, res.completion_tokens)


class Arbiter:
    def __init__(self, llm, mode: str = "rule"):
        if mode not in ("rule", "choose", "resolve"):
            raise ValueError("arbiter mode must be 'rule', 'choose' or 'resolve'")
        self.llm = llm
        self.mode = mode

    def decide(self, problem: str, kernel_result, cot: CotResult, referee: bool = True) -> Arbitration:
        """``referee=False`` only cross-checks: on disagreement the program's answer stands."""
        a, b = kernel_result.program_answer, cot.answer
        if numbers_match(a, b):
            return Arbitration(a, "agree")
        if not is_number(a) and not is_number(b):
            return Arbitration(None, "none")
        if not is_number(a):
            return Arbitration(b, "cot-only")      # nothing to arbitrate
        if not is_number(b):
            return Arbitration(a, "cdsl-only")
        if not referee:
            return Arbitration(a, "unchecked")
        if self.mode == "rule":
            if arithmetic_load(problem) >= ARITHMETIC_LOAD_THRESHOLD:
                return Arbitration(a, "rule:cdsl")
            return Arbitration(b, "rule:cot")
        final = kernel_result.final
        trace = "\n".join(final.run.trace[-25:]) if final.run and final.run.trace else "(none)"
        plan = f"Plan:\n{kernel_result.plan[:1500]}\nProgram:\n" if kernel_result.plan else ""
        template = _CHOOSE_PROMPT if self.mode == "choose" else _RESOLVE_PROMPT
        prompt = template.format(
            problem=problem, plan=plan, program=(kernel_result.program or "(no program)")[:2500],
            trace=trace[:1500], a=a, cot=cot.text.strip()[:3000], b=f"{b:g}")
        res = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0,
                            max_tokens=800 if self.mode == "resolve" else 400, seed=0)
        if self.mode == "choose":
            picks = re.findall(r"CHOICE:\s*\**\s*([AB])\b", res.text)
            ref = CotResult(res.text, None, res.seconds, res.completion_tokens)
            if not picks:
                return Arbitration(b, "cot-default", ref)   # CoT is right more often on dev
            return Arbitration(a, "referee:cdsl", ref) if picks[-1] == "A" else \
                Arbitration(b, "referee:cot", ref)
        ref = CotResult(res.text, extract_final_number(res.text), res.seconds, res.completion_tokens)
        if ref.answer is None:
            return Arbitration(b, "cot-default", ref)
        if numbers_match(ref.answer, a):
            return Arbitration(a, "referee:cdsl", ref)
        if numbers_match(ref.answer, b):
            return Arbitration(b, "referee:cot", ref)
        return Arbitration(ref.answer, "referee:new", ref)
