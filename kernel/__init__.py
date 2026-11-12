"""Cognitive Kernel - the 7-component pipeline (Phase 2 vertical slice, cru).

See §3 and §6 of docs/superpowers/specs/2026-05-24-cognitive-kernel-v0.1-design.md.
"""
from .arbiter import Arbiter, Arbitration, CotResult, answers_agree, numbers_match, solve_cot
from .classifier import CATEGORIES, Classifier
from .compiler import Compiler, CompileOutput
from .episode_log import EpisodeLog
from .llm import LLMError, LocalLLM
from .memory import BANKS, TypedMemory
from .pipeline import CognitiveKernel, KernelResult, Reading, format_trace, normalize_answer
from .planner import Planner
from .reviser import Attempt, Reviser
from .targets import CDSLTarget, PythonTarget
from .verifier import Verdict, Verifier

__all__ = [
    "Arbiter", "Arbitration", "CotResult", "answers_agree", "numbers_match", "solve_cot",
    "Planner", "Reading",
    "CDSLTarget", "PythonTarget",
    "CATEGORIES", "Classifier", "Compiler", "CompileOutput", "EpisodeLog",
    "LLMError", "LocalLLM", "BANKS", "TypedMemory", "CognitiveKernel", "KernelResult",
    "format_trace", "normalize_answer", "Attempt", "Reviser", "Verdict", "Verifier",
]
