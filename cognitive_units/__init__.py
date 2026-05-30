"""Cognitive Units MVP — experimental package.

Goal: validate whether a single Cognitive Unit can LEARN rules autonomously
via LLM distillation, instead of having them hand-coded.

Atom of the thesis: if `LearnableCognitiveUnit` starting empty can reach
accuracy comparable to (or better than) the hand-coded U-05 security_scanner
on a 200-example security classification dataset, the broader Cognitive
Units vision is technically viable. If not, we pivot.

This package is independent from the Phase 1A data pipeline — it runs in
parallel without conflict. Integration with the broader Cognitive Kernel
is deferred until after this MVP either succeeds or fails.
"""
from .base import CognitiveUnit
from .learnable import LearnableCognitiveUnit
from .sandbox import safe_compile_lambda

__all__ = ["CognitiveUnit", "LearnableCognitiveUnit", "safe_compile_lambda"]
