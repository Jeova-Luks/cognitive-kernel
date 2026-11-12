"""Component [5] VERIFIER (cru).

Deterministic checks over one compile + run attempt. The verifier never sees
the gold answer. Agreement with the gold answer is measured separately by the
eval harness.

Two levels of verification exist in the pipeline:
- STRUCTURAL (this file): the program parses, executes and returns a value of
  the right type. A failure here is something an error message can fix, so it
  drives the repair loop.
- CROSS-CHECK (arbiter.py): the program's answer matches an independently
  derived answer. This is what "verified" means for the final answer.

With ``self_checks=True`` the program's own assert / verify forms also count
as structural failures (the Phase 2 v1/v2 behaviour). It is off by default
because on the dev set those self-written checks rejected correct answers
more often than they caught wrong ones; failed checks are then only recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cdsl import RunResult

CHECK_ORDER = ("parses", "executes", "returns", "type_correct", "asserts_hold", "verifies_hold")


@dataclass
class Verdict:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    error_trace: str | None = None


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _trace_tail(run: RunResult, n: int = 6) -> str:
    if not run.trace:
        return ""
    return "\nExecution trace (last steps):\n" + "\n".join("  " + t for t in run.trace[-n:])


class Verifier:
    def __init__(self, self_checks: bool = False):
        self.self_checks = self_checks

    def verify(self, parse_error: str | None, run: RunResult | None, category: str) -> Verdict:
        checks: dict[str, bool] = {k: False for k in CHECK_ORDER}
        if parse_error is not None or run is None:
            return Verdict(False, checks, f"ParseError: {parse_error}")
        checks["parses"] = True
        if not run.ok:
            return Verdict(False, checks, f"RuntimeError: {run.error}{_trace_tail(run)}")
        checks["executes"] = True
        checks["returns"] = run.returned
        if category in ("math", "mixed"):
            checks["type_correct"] = is_number(run.value)
        else:
            checks["type_correct"] = run.value is not None
        failed_asserts = [c for c in run.checks if c.kind == "assert" and not c.passed]
        failed_verifies = [c for c in run.checks if c.kind == "verify" and not c.passed]
        checks["asserts_hold"] = not failed_asserts
        checks["verifies_hold"] = not failed_verifies

        errors = []
        if not checks["type_correct"]:
            errors.append(f"TypeError: a {category} problem must return a single number, "
                          f"got {run.value!r}")
        if not self.self_checks:          # recorded in `checks`, but not a failure
            failed_asserts, failed_verifies = [], []
        for c in failed_asserts:
            errors.append(f"AssertionFailed: {c.detail}")
        for c in failed_verifies:
            errors.append(f"VerifyFailed: {c.source}: {c.detail}. The two computations disagree, "
                          "so at least one of them misreads the problem.")
        if errors:
            return Verdict(False, checks, "\n".join(errors) + _trace_tail(run))
        return Verdict(True, checks, None)
