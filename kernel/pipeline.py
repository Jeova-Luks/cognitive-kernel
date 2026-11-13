"""The 7-component Cognitive Kernel vertical slice (Phase 2, cru).

    [1] classifier -> [3] memory recall -> [2a] reader/planner -> [2b] compiler
        -> [4] runtime -> [5a] structural verifier -> [6] reviser (back to [2b])
        -> [5b] cross-check against an independent derivation (+ referee)
        -> [7] episode log

Memory is consulted twice: before planning (semantic notes for the reader and
the most similar confirmed past episode as a worked example, spec §6.2
"joined into the compiler's context"), and at run time through (lookup ...).

Switches, all on the constructor, so each design decision can be ablated:
- ``target``       "cdsl" or "python": the intermediate representation;
- ``plan``         read the problem into a plan before compiling (planner.py);
- ``self_checks``  require the program to carry its own assert / verify;
- ``cross_check``  confirm the program's answer against plain CoT and
                   arbitrate on disagreement (arbiter.py);
- ``referee``      "rule", "choose" or "resolve": who decides a disagreement.

The stages are also exposed separately (``read``, ``solve_program``,
``cross_check``, ``commit``) so an eval harness can share one reading and one
CoT derivation between several kernels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from cdsl import available_tools

from .arbiter import Arbiter, Arbitration, CotResult, solve_cot
from .classifier import Classifier
from .compiler import Compiler
from .episode_log import EpisodeLog
from .memory import TypedMemory
from .planner import Planner
from .reviser import Attempt, Reviser
from .targets import EXAMPLE_PROBLEMS, make_target
from .verifier import Verifier, is_number

SEMANTIC_BANKS = ("procedural", "factual", "causal")


def normalize_answer(value: object) -> object:
    if is_number(value) and isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return int(value)
        return round(value, 6)
    return value


@dataclass
class Reading:
    """What is known about a problem before any program is written."""
    category: str
    notes: list[str]
    plan: str | None
    query_vector: list[float] | None = None
    seconds: dict[str, float] = field(default_factory=dict)
    completion_tokens: int = 0


@dataclass
class KernelResult:
    problem: str
    category: str
    plan: str | None
    program_answer: object              # what the executed program returned
    answer: object                      # final answer (after cross-check, if enabled)
    success: bool                       # the program passed the structural verifier
    confidence: str                     # high | low | none | unchecked
    attempts: list[Attempt]
    final: Attempt
    memory_notes: list[str]
    episodes_recalled: list[str]
    query_vector: list[float] | None = None
    arbitration: Arbitration | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)
    episode_id: int | None = None

    @property
    def program(self) -> str | None:
        return self.final.compiled.program_text

    @property
    def confirmed(self) -> bool:
        """Good enough to reuse as a worked example in episodic memory."""
        if self.arbitration is not None:
            return self.arbitration.source == "agree"
        return self.success


class CognitiveKernel:
    def __init__(self, llm, memory: TypedMemory | None = None,
                 episode_log: EpisodeLog | None = None, max_attempts: int = 3,
                 notes_k: int = 3, episodes_k: int = 1, tools: dict | None = None,
                 target: str = "cdsl", plan: bool = True, self_checks: bool = False,
                 cross_check: bool = True, referee: str = "rule", audit: bool = False):
        self.llm = llm
        self.memory = memory if memory is not None else TypedMemory()
        self.episode_log = episode_log if episode_log is not None else EpisodeLog()
        tools = tools if tools is not None else available_tools()
        self.classifier = Classifier(llm)
        self.planner = Planner(llm, EXAMPLE_PROBLEMS) if plan else None
        self.target = make_target(target, tools, self.memory, plan=plan, self_checks=self_checks)
        self.compiler = Compiler(llm, self.target)
        self.verifier = Verifier(self_checks=self_checks)
        self.reviser = Reviser(self.compiler, self.verifier, max_attempts)
        self.arbiter = Arbiter(llm, referee, audit) if cross_check else None
        self.notes_k = notes_k
        self.episodes_k = episodes_k

    # -------------------------------------------------------------- stage by stage
    def read(self, problem: str) -> Reading:
        """[1] classifier, [3] semantic memory, [2a] reader/planner."""
        sec: dict[str, float] = {}
        t = time.perf_counter()
        category = self.classifier.classify(problem)
        sec["classifier"] = time.perf_counter() - t

        t = time.perf_counter()
        query_vec = self.memory.embed_query(problem)
        notes = [h.text for h in self.memory.search_banks(problem, SEMANTIC_BANKS, k=self.notes_k)]
        sec["memory"] = time.perf_counter() - t

        plan, tokens = None, 0
        if self.planner is not None:
            p = self.planner.plan(problem, notes)
            plan, tokens, sec["planner"] = p.text, p.completion_tokens, p.seconds
        return Reading(category, notes, plan, query_vec, sec, tokens)

    def solve_program(self, problem: str, reading: Reading | None = None) -> KernelResult:
        """[2b] compiler, [4] runtime, [5a] structural verifier, [6] reviser. Nothing is logged."""
        reading = reading if reading is not None else self.read(problem)
        sec = dict(reading.seconds)

        t = time.perf_counter()
        recalled = self.memory.search("episodic", problem, k=self.episodes_k,
                                      query_vector=reading.query_vector) if self.episodes_k else []
        episodes = [(h.text, h.meta.get("plan"), h.meta["program"]) for h in recalled]
        sec["memory"] = sec.get("memory", 0.0) + time.perf_counter() - t

        attempts, final = self.reviser.solve(problem, reading.category, reading.notes, episodes,
                                             plan=reading.plan)
        sec["compiler"] = sum(a.compiled.seconds for a in attempts)
        sec["runtime"] = sum(a.run_seconds for a in attempts)
        sec["verifier"] = sum(a.verify_seconds for a in attempts)

        answer = normalize_answer(final.run.value) if final.run is not None and final.run.returned else None
        return KernelResult(problem, reading.category, reading.plan, answer, answer,
                            final.verdict.passed, "unchecked", attempts, final, reading.notes,
                            [h.text for h in recalled], reading.query_vector, None, sec)

    def cross_check(self, result: KernelResult, cot: CotResult | None = None,
                    referee: bool = True) -> Arbitration:
        """[5b] compare with an independent CoT derivation; arbitrate on disagreement."""
        arbiter = self.arbiter or Arbiter(self.llm, "rule")
        t = time.perf_counter()
        if cot is None:
            cot = solve_cot(self.llm, result.problem)
        arb = arbiter.decide(result.problem, result, cot, referee=referee)
        result.arbitration = arb
        result.answer = normalize_answer(arb.answer)
        result.confidence = arb.confidence
        result.stage_seconds["cross_check"] = time.perf_counter() - t
        return arb

    def commit(self, result: KernelResult) -> int:
        """[7] episode log; confirmed episodes also go to episodic memory."""
        t = time.perf_counter()
        result.episode_id = self.episode_log.log(
            result.problem, result.category, result.program, result.answer, len(result.attempts),
            result.confirmed, result.final.verdict.error_trace, result.query_vector,
            plan=result.plan, confidence=result.confidence)
        if result.confirmed and result.program:
            self.memory.add("episodic", result.problem,
                            meta={"program": result.program, "plan": result.plan,
                                  "episode_id": result.episode_id})
        result.stage_seconds["episode_log"] = time.perf_counter() - t
        return result.episode_id

    # --------------------------------------------------------------- whole pipeline
    def solve(self, problem: str) -> KernelResult:
        result = self.solve_program(problem)
        if self.arbiter is not None:
            self.cross_check(result)
        self.commit(result)
        return result


def format_trace(r: KernelResult) -> str:
    """Human-readable reasoning trace of one solve."""
    lines = [f"[1] classifier  -> {r.category}",
             f"[3] memory      -> {len(r.memory_notes)} notes, {len(r.episodes_recalled)} episode(s) recalled"]
    if r.plan:
        lines.append("[2a] planner    -> plan with " + str(len(r.plan.splitlines())) + " lines")
    for a in r.attempts:
        lines.append(f"[2b] compiler   -> attempt {a.index} ({a.compiled.mode}, "
                     f"T={a.compiled.temperature}): "
                     + ("parsed" if a.compiled.ast is not None else f"PARSE ERROR: {a.compiled.parse_error}"))
        if a.run is not None:
            status = f"returned {a.run.value!r}" if a.run.ok else f"ERROR: {a.run.error.splitlines()[0]}"
            lines.append(f"[4] runtime     -> {status}")
        lines.append(f"[5a] verifier   -> {'PASS' if a.verdict.passed else 'FAIL'} "
                     + " ".join(f"{k}={'y' if v else 'n'}" for k, v in a.verdict.checks.items()))
        if not a.verdict.passed and a.index < len(r.attempts):
            lines.append("[6] reviser     -> sending error trace back to the compiler")
    if r.arbitration is not None:
        lines.append(f"[5b] cross-check -> {r.arbitration.source} (program answer {r.program_answer!r})")
    lines.append(f"[7] episode log -> episode #{r.episode_id} saved "
                 f"({'confirmed' if r.confirmed else 'not confirmed'})")
    lines.append(f"    answer = {r.answer!r}   confidence = {r.confidence}")
    return "\n".join(lines)
