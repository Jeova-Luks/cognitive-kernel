"""CDSL - Cognitive DSL (v0.1, cru).

The intermediate executable representation the compiler emits instead of a
natural-language chain of thought. See §5 of
docs/superpowers/specs/2026-05-24-cognitive-kernel-v0.1-design.md.
"""
from .parser import (
    CDSLSyntaxError,
    Keyword,
    MapLit,
    Symbol,
    Vector,
    extract_program_text,
    parse,
    parse_program,
    to_source,
)
from .runtime import CDSLRuntimeError, Check, RunResult, Runtime
from .tools import ToolError, available_tools

__all__ = [
    "CDSLSyntaxError", "Keyword", "MapLit", "Symbol", "Vector",
    "extract_program_text", "parse", "parse_program", "to_source",
    "CDSLRuntimeError", "Check", "RunResult", "Runtime",
    "ToolError", "available_tools",
]
