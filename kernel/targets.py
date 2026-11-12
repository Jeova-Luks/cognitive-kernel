"""Compilation targets: the intermediate representation the compiler emits.

Two targets share one interface so the rest of the pipeline (memory, verifier,
reviser, episode log) is identical and only the representation changes:

- ``CDSLTarget``   the project's Cognitive DSL (the thesis under test).
- ``PythonTarget`` plain Python, i.e. Program-of-Thought. This is the control
  for the question "does CDSL buy anything over emitting Python directly?"
  (baseline 2 in spec §8.2).

The few-shot programs are hand-written for this file and are NOT from GSM8K.
Both targets get the same three problems, solved the same way.

Each target builds its prompt in four variants, from two switches:
- ``plan``         the compiler receives the reader's plan (planner.py) and is
                   told to follow it, instead of working from the problem alone;
- ``self_checks``  the program must contain its own assert + verify. Off by
                   default: on the dev set these self-written checks rejected a
                   correct answer in 12 of 37 failed CDSL attempts and 34 of 47
                   failed Python attempts, while an answer that passed them was
                   right only 75% / 61% of the time. Verification now comes from
                   an independent derivation (arbiter.py).
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile

from cdsl import CDSLSyntaxError, Check, RunResult, Runtime, extract_program_text, parse_program

EXAMPLE_PROBLEMS = [
    "A baker bakes 4 trays of muffins with 12 muffins on each tray. She sells 30 muffins "
    "and gives 5 to her neighbor. How many muffins does she have left?",
    "Maria earns $1,500 a month. She spends 40% of it on rent and $250 on food. "
    "How much money does she have left each month?",
    "How many multiples of 7 are there between 1 and 100?",
]

_HEADER = ("You are the COMPILER of a problem-solving system. You translate a natural-language "
           "problem into {what} that is executed by a machine. You never answer in prose: you "
           "output exactly one program and nothing else.")


class ProgramSyntaxError(ValueError):
    """The compiler output is not a well-formed program in the target language."""


# ============================================================================ CDSL

_CDSL_CARD = _HEADER.format(what="a program in CDSL, a small Lisp-like language") + """

CDSL syntax: s-expressions. Strings use double quotes. Numbers are plain: no commas, units, $ or % signs.

Forms:
(program :type <short-label> <form> ...)   the whole program
(let <name> <expr>)                        bind a variable; use snake_case names
(return <expr>)                            the final answer; must be the last form
{check_forms}(if <cond> <then> <else>)
(compare <a> :op <op> <b>)                 op is one of > >= < <= = !=
(call_tool <tool> "<expression>")          tools: {tools}
(lookup <factual|procedural|causal> "<query>")   returns notes (strings) from memory

Arithmetic functions: (+ a b ...) (- a b ...) (* a b ...) (/ a b) (// a b) (% a b) (** a b) (min a b) (max a b) (abs x) (round x) (sum list) (len list)

Tools:
- python_exec runs one Python expression. Every variable bound with let is available in it by name.
- calculator evaluates one arithmetic expression; let variables are available by name.

Rules:
{rules}"""

_CDSL_CHECK_FORMS = """(assert <condition>)                       a sanity condition that must hold
(verify <value> :against <expr>)           recompute the answer independently; both must be equal
"""

_CDSL_RULES = {
    "source_plan": "Follow the plan: one (let ...) for each GIVEN line and each STEPS line, in the "
                   "same order, with the same names. A step 'x = a * b' becomes (let x (* a b)) and "
                   "'x = a - b - c' becomes (let x (- a b c)). Do not add or drop steps.",
    "source_problem": "Bind every number from the problem with (let ...) before using it, and use "
                      "one (let ...) per intermediate quantity, with a meaningful name.",
    "let": "(let ...) takes exactly one name and one expression.",
    "prefix": "CDSL is prefix-only: the operator comes first and each operation has its own "
              "parentheses. Write (- (+ a b) 2), never (+ a b - 2).",
    "checks": "Add at least one (assert ...) and one (verify ...) that recomputes the answer a "
              "different way.",
    "return_plan": "End with (return <name>) where <name> is the quantity in the plan's ASK line.",
    "return_problem": "End with (return <answer>) where <answer> is a single number, in the unit "
                      "the question asks for.",
    "only": "Output only the program: no explanation, no markdown fences.",
}

_CDSL_EXAMPLES = [
    """(program :type remaining-after-sales
  (let trays 4)
  (let per_tray 12)
  (let sold 30)
  (let given 5)
  (let baked (* trays per_tray))
  (let left (- baked sold given))
  (assert (compare left :op >= 0))
  (verify left :against (call_tool python_exec "trays * per_tray - (sold + given)"))
  (return left))""",
    """(program :type percent-budget
  (let salary 1500)
  (let rent_percent 40)
  (let food 250)
  (let rent_fraction (/ rent_percent 100))
  (let rent (* salary rent_fraction))
  (let left (- salary rent food))
  (assert (compare left :op < salary))
  (verify left :against (call_tool calculator "salary * (1 - rent_percent / 100) - food"))
  (return left))""",
    """(program :type counting-with-constraint
  (let low 1)
  (let high 100)
  (let divisor 7)
  (let multiples (call_tool python_exec "[x for x in range(low, high + 1) if x % divisor == 0]"))
  (let count (len multiples))
  (assert (compare count :op > 0))
  (verify count :against (// high divisor))
  (return count))""",
]

_CDSL_FALLBACK = ("Write the simplest program that solves the problem: use only (let ...), the "
                  "arithmetic functions and (return ...). Do not use call_tool or lookup.")


def _numbered(rules: list[str]) -> str:
    return "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))


def _without_checks(program: str, markers: tuple[str, ...]) -> str:
    """Drop the self-check lines from a few-shot program."""
    return "\n".join(line for line in program.splitlines()
                     if not line.strip().startswith(markers))


class CDSLTarget:
    name = "cdsl"
    fallback_instruction = _CDSL_FALLBACK

    def __init__(self, tools: dict, memory=None, plan: bool = True, self_checks: bool = False):
        self.runtime = Runtime(tools=tools, memory=memory)
        r = _CDSL_RULES
        rules = [r["source_plan"] if plan else r["source_problem"], r["let"], r["prefix"]]
        if self_checks:
            rules.append(r["checks"])
        rules += [r["return_plan"] if plan else r["return_problem"], r["only"]]
        self.system = (_CDSL_CARD
                       .replace("{check_forms}", _CDSL_CHECK_FORMS if self_checks else "")
                       .replace("{tools}", ", ".join(sorted(tools)))
                       .replace("{rules}", _numbered(rules)))
        programs = _CDSL_EXAMPLES if self_checks else [
            _without_checks(p, ("(assert", "(verify")) for p in _CDSL_EXAMPLES]
        self.examples = list(zip(EXAMPLE_PROBLEMS, programs))

    def parse(self, raw: str) -> tuple[str, object]:
        try:
            text = extract_program_text(raw)
            return text, parse_program(text)
        except CDSLSyntaxError as e:
            raise ProgramSyntaxError(str(e))

    def run(self, parsed: object) -> RunResult:
        return self.runtime.run(parsed)


# ========================================================================== Python

_PY_CARD = _HEADER.format(what="a short Python program") + """

Rules:
{rules}"""

_PY_RULES = {
    "source_plan": "Follow the plan: one assignment for each GIVEN line and each STEPS line, in "
                   "the same order, with the same names and the same formulas. Do not add or drop steps.",
    "source_problem": "Assign every number from the problem to a variable before using it, and "
                      "use one assignment per intermediate quantity, with a meaningful snake_case name.",
    "checks": "Add at least one `assert` sanity condition, then recompute the answer a different "
              "way into `check` and add `assert abs(answer - check) < 1e-6`.",
    "answer_plan": "Store the quantity in the plan's ASK line in the variable `answer`.",
    "answer_problem": "Store the final result in the variable `answer`: a single number, in the "
                      "unit the question asks for.",
    "plain": "Plain Python only. The `math` module is already imported. No other imports, no "
             "input(), no files, no print().",
    "only": "Output only the program: no explanation, no markdown fences.",
}

_PY_EXAMPLES = [
    """trays = 4
per_tray = 12
sold = 30
given = 5
baked = trays * per_tray
left = baked - sold - given
answer = left
assert answer >= 0
check = trays * per_tray - (sold + given)
assert abs(answer - check) < 1e-6""",
    """salary = 1500
rent_percent = 40
food = 250
rent_fraction = rent_percent / 100
rent = salary * rent_fraction
left = salary - rent - food
answer = left
assert answer < salary
check = salary * (1 - rent_percent / 100) - food
assert abs(answer - check) < 1e-6""",
    """low = 1
high = 100
divisor = 7
multiples = [x for x in range(low, high + 1) if x % divisor == 0]
count = len(multiples)
answer = count
assert answer > 0
check = high // divisor
assert abs(answer - check) < 1e-6""",
]

_PY_FALLBACK = ("Write the simplest program that solves the problem: only variable assignments, "
                "arithmetic and `answer`. No functions, loops or imports.")

_PY_MARKER = "__PY_RESULT__"

# Runs in the child process: executes the program read from stdin and reports
# the `answer` variable, the failing line of any exception, and simple variables.
_PY_RUNNER = r'''
import json, math, numbers, sys, traceback
code = sys.stdin.read()
ns = {"math": math, "__name__": "__main__"}
out = {}
try:
    exec(compile(code, "<program>", "exec"), ns)
except BaseException as e:
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<program>"]
    lines = code.splitlines()
    line = lines[frames[-1].lineno - 1].strip() if frames and frames[-1].lineno <= len(lines) else ""
    out["kind"] = "assert" if isinstance(e, AssertionError) else "error"
    out["error"] = (type(e).__name__ + ": " + str(e)).rstrip(": ")
    out["line"] = line
def plain(v):
    if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
        return v
    if isinstance(v, numbers.Real):
        return float(v)
    return None
out["has_answer"] = "answer" in ns
value = ns.get("answer")
out["answer"] = plain(value) if plain(value) is not None or value is None else repr(value)
out["vars"] = [[k, plain(v)] for k, v in ns.items()
               if not k.startswith("_") and k != "math" and plain(v) is not None
               and not (isinstance(v, str) and len(v) > 60)][:40]
sys.stdout.write("\n__PY_RESULT__" + json.dumps(out))
'''


class PythonTarget:
    name = "python"
    fallback_instruction = _PY_FALLBACK

    def __init__(self, tools: dict | None = None, memory=None, plan: bool = True,
                 self_checks: bool = False, timeout: float = 10.0):
        r = _PY_RULES
        rules = [r["source_plan"] if plan else r["source_problem"]]
        if self_checks:
            rules.append(r["checks"])
        rules += [r["answer_plan"] if plan else r["answer_problem"], r["plain"], r["only"]]
        self.system = _PY_CARD.replace("{rules}", _numbered(rules))
        programs = _PY_EXAMPLES if self_checks else [
            _without_checks(p, ("assert", "check =")) for p in _PY_EXAMPLES]
        self.examples = list(zip(EXAMPLE_PROBLEMS, programs))
        self.timeout = timeout

    def parse(self, raw: str) -> tuple[str, object]:
        fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", raw, re.S)
        code = (fenced.group(1) if fenced else raw).strip()
        if not code:
            raise ProgramSyntaxError("empty program")
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ProgramSyntaxError(f"Python syntax error on line {e.lineno}: {e.msg}: "
                                     f"{(e.text or '').strip()[:80]}")
        return code, code

    def run(self, parsed: object) -> RunResult:
        result = RunResult(ok=False)
        with tempfile.TemporaryDirectory(prefix="pot_") as cwd:
            try:
                proc = subprocess.run([sys.executable, "-I", "-c", _PY_RUNNER], input=parsed,
                                      capture_output=True, text=True, timeout=self.timeout,
                                      cwd=cwd, encoding="utf-8")
            except subprocess.TimeoutExpired:
                result.error = f"program timed out after {self.timeout:.0f}s"
                return result
        _, sep, tail = proc.stdout.rpartition(_PY_MARKER)
        if not sep:
            result.error = f"program crashed: {proc.stderr.strip()[-300:] or 'no output'}"
            return result
        out = json.loads(tail)
        result.trace = [f"{k} = {v!r}" for k, v in out["vars"]]
        result.value, result.returned = out["answer"], out["has_answer"]
        kind = out.get("kind")
        if kind == "error":
            result.error = f"{out['error']}\n  in line: {out['line']}"
            return result
        if kind == "assert":
            values = dict(out["vars"])
            shown = ", ".join(f"{k}={values[k]!r}" for k in ("answer", "check") if k in values)
            detail = f"`{out['line']}` failed" + (f" ({shown})" if shown else "")
            result.checks.append(Check("verify" if "check" in out["line"] else "assert",
                                       out["line"], False, detail))
        if not result.returned:
            result.error = (f"AssertionError: `{out['line']}` failed before `answer` was defined"
                            if kind == "assert" else
                            "the program finished without defining the variable `answer`")
            return result
        result.ok = True
        return result


TARGETS = {"cdsl": CDSLTarget, "python": PythonTarget}


def make_target(name: str, tools: dict, memory=None, plan: bool = True,
                self_checks: bool = False):
    if name not in TARGETS:
        raise ValueError(f"unknown target '{name}'; choose from {sorted(TARGETS)}")
    return TARGETS[name](tools, memory, plan=plan, self_checks=self_checks)
