"""Component [2] COMPILER NL -> program (cru, prompt-only).

In the full project this is the deep component: the 100M base model
fine-tuned on (NL, CDSL) pairs (Phases 3-4). For the Phase 2 vertical slice
a stand-in instruction-tuned LLM emits the program from a grammar card plus
few-shot examples. The target language (CDSL or Python) is pluggable, see
``targets.py``.

When a plan is supplied (planner.py), every problem turn, including the
few-shot ones, carries the reader's plan and the compiler only has to render
it in the target language.

Three prompt modes:
- first     system + few-shot + problem
- repair    same, plus every earlier failed attempt with its error trace
- fallback  fresh start: earlier programs are NOT shown (they anchor the
            model), only the error that kept repeating and an instruction to
            write the simplest possible program
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .planner import EXAMPLE_PLANS
from .targets import ProgramSyntaxError

MAX_PROGRAM_CHARS = 1500
MAX_ERROR_CHARS = 700


@dataclass
class CompileOutput:
    raw: str                     # raw LLM text
    program_text: str | None     # extracted program source
    ast: object | None           # parsed program, None if parsing failed
    parse_error: str | None
    seconds: float
    completion_tokens: int
    mode: str = "first"          # first | repair | fallback
    temperature: float = 0.0


def _problem_turn(problem: str, plan: str | None) -> str:
    return f"Problem:\n{problem}" + (f"\n\nPlan:\n{plan}" if plan else "")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


class Compiler:
    def __init__(self, llm, target, max_tokens: int = 700):
        self.llm = llm
        self.target = target
        self.max_tokens = max_tokens

    def build_messages(self, problem: str, category: str, memory_notes: list[str],
                       episodes: list[tuple[str, str | None, str]],
                       failures: list[tuple[str, str]], fallback: bool = False,
                       plan: str | None = None) -> list[dict]:
        """``episodes``: (problem, plan, program) recalled from episodic memory.
        ``failures``: (program_or_raw_output, error_trace) of every earlier attempt.
        ``plan``: the reader's plan; when given, memory notes are not repeated here
        because the reader already had them."""
        msgs = [{"role": "system", "content": self.target.system}]
        shots = [(ex_problem, EXAMPLE_PLANS[i] if plan else None, ex_program)
                 for i, (ex_problem, ex_program) in enumerate(self.target.examples)]
        for ex_problem, ex_plan, ex_program in shots + list(episodes):
            if plan and not ex_plan:
                continue                     # an episode stored without a plan would mix formats
            msgs.append({"role": "user", "content": _problem_turn(ex_problem, ex_plan if plan else None)})
            msgs.append({"role": "assistant", "content": ex_program})
        parts = [f"Category: {category}"]
        if memory_notes and not plan:
            parts.append("Notes from memory (use only if relevant):\n" +
                         "\n".join(f"- {n}" for n in memory_notes))
        parts.append(_problem_turn(problem, plan))
        if failures and fallback:
            last_error = failures[-1][1].splitlines()[0] if failures[-1][1] else "unknown error"
            parts.append(
                f"Note: {len(failures)} earlier attempts failed, the last ones with the same "
                f"error:\n{_clip(last_error, 300)}\n\nStart over from scratch. "
                f"{self.target.fallback_instruction} Output only the program.")
        elif failures:
            blocks = [f"Attempt {i}:\n{_clip(program, MAX_PROGRAM_CHARS)}\nError:\n"
                      f"{_clip(error, MAX_ERROR_CHARS)}"
                      for i, (program, error) in enumerate(failures, 1)]
            parts.append(
                "Your previous attempts at this problem failed:\n\n" + "\n\n".join(blocks) +
                "\n\nWrite a new, corrected program for the same problem. Do not repeat any "
                "program above. Re-read the problem, keep what was right and fix what was "
                "wrong. Output only the program.")
        msgs.append({"role": "user", "content": "\n\n".join(parts)})
        return msgs

    def compile(self, problem: str, category: str, memory_notes: list[str] | None = None,
                episodes: list[tuple[str, str | None, str]] | None = None,
                failures: list[tuple[str, str]] | None = None, fallback: bool = False,
                temperature: float = 0.0, seed: int | None = 0,
                plan: str | None = None) -> CompileOutput:
        failures = failures or []
        msgs = self.build_messages(problem, category, memory_notes or [], episodes or [],
                                   failures, fallback, plan)
        t0 = time.perf_counter()
        res = self.llm.chat(msgs, temperature=temperature, max_tokens=self.max_tokens, seed=seed)
        dt = time.perf_counter() - t0
        program_text, parsed, err = None, None, None
        try:
            program_text, parsed = self.target.parse(res.text)
        except ProgramSyntaxError as e:
            err = str(e)
        mode = "first" if not failures else ("fallback" if fallback else "repair")
        return CompileOutput(res.text, program_text, parsed, err, dt, res.completion_tokens,
                             mode, temperature)
