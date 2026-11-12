"""Component [6] REVISER (cru).

Loop of at most N attempts: compile -> run -> verify. The spec (§6.5) uses 5;
the default is now 3 because on the dev set attempts 3-5 produced 7 passing
CDSL programs out of 41 tries, only 2 of them correct.

Repair policy (v2, tuned on the dev set after the first run showed the model
re-emitting the same broken program up to five times):
- every retry shows ALL earlier failed attempts with their error traces;
- temperature rises with each retry (0.0, 0.5, 0.7, 0.9, 0.9);
- when two consecutive attempts fail with the same error signature, the loop
  switches to FALLBACK for the remaining attempts: a fresh prompt without the
  failed programs, asking for the simplest possible program;
- if nothing verifies, the answer is the value most attempts returned.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass

from cdsl import RunResult

from .compiler import CompileOutput, Compiler
from .verifier import Verdict, Verifier, is_number

TEMPERATURES = (0.0, 0.5, 0.7, 0.9, 0.9)


@dataclass
class Attempt:
    index: int
    compiled: CompileOutput
    run: RunResult | None
    verdict: Verdict
    run_seconds: float
    verify_seconds: float


def error_signature(error_trace: str | None) -> str:
    """First line of an error trace with numbers masked: 'same kind of failure'."""
    first = (error_trace or "").strip().splitlines()[0] if (error_trace or "").strip() else ""
    return re.sub(r"\d+(\.\d+)?", "N", first)


class Reviser:
    def __init__(self, compiler: Compiler, verifier: Verifier, max_attempts: int = 3,
                 temperatures: tuple[float, ...] = TEMPERATURES, fallback_after: int = 2):
        self.compiler = compiler
        self.target = compiler.target
        self.verifier = verifier
        self.max_attempts = max_attempts
        self.temperatures = temperatures
        self.fallback_after = fallback_after      # identical consecutive errors that trigger fallback

    def solve(self, problem: str, category: str, memory_notes: list[str],
              episodes: list[tuple[str, str | None, str]],
              plan: str | None = None) -> tuple[list[Attempt], Attempt]:
        attempts: list[Attempt] = []
        failures: list[tuple[str, str]] = []
        fallback = False
        for i in range(1, self.max_attempts + 1):
            temperature = self.temperatures[min(i - 1, len(self.temperatures) - 1)]
            compiled = self.compiler.compile(problem, category, memory_notes, episodes, failures,
                                             fallback=fallback, temperature=temperature, seed=i,
                                             plan=plan)
            t0 = time.perf_counter()
            run = self.target.run(compiled.ast) if compiled.ast is not None else None
            t1 = time.perf_counter()
            verdict = self.verifier.verify(compiled.parse_error, run, category)
            t2 = time.perf_counter()
            attempt = Attempt(i, compiled, run, verdict, t1 - t0, t2 - t1)
            attempts.append(attempt)
            if verdict.passed:
                return attempts, attempt
            failures.append((compiled.program_text or compiled.raw.strip(), verdict.error_trace or ""))
            recent = [error_signature(e) for _, e in failures[-self.fallback_after:]]
            if len(recent) == self.fallback_after and len(set(recent)) == 1:
                fallback = True
        return attempts, self._best_effort(attempts)

    @staticmethod
    def _best_effort(attempts: list[Attempt]) -> Attempt:
        """No attempt verified: return the value most attempts agree on.

        Later attempts run at higher temperature and are noisier, so "the
        latest" is a poor default; ties go to the earliest attempt.
        """
        numeric = [a for a in attempts
                   if a.run is not None and a.run.returned and is_number(a.run.value)]
        if not numeric:
            return attempts[-1]
        votes = Counter(round(float(a.run.value), 6) for a in numeric)
        best = max(votes.values())
        return next(a for a in numeric if votes[round(float(a.run.value), 6)] == best)
