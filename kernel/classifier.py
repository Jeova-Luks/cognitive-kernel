"""Component [1] CLASSIFIER (cru).

The spec (§6.1) calls for a linear probe on frozen base-model embeddings.
The 100M base does not exist yet, so this cru version asks the stand-in LLM
for a one-word label. The label biases the compiler prompt and sets the
verifier's type check.
"""
from __future__ import annotations

import re

CATEGORIES = ("math", "code", "logic", "mixed")

_SYSTEM = (
    "You label problems for a problem-solving system. Categories:\n"
    "math  - arithmetic, word problems with quantities, algebra, counting\n"
    "code  - write, fix or explain a computer program\n"
    "logic - puzzles, deduction, truth tables, constraints without heavy arithmetic\n"
    "mixed - needs two or more of the above\n"
    "Reply with exactly one word: math, code, logic or mixed."
)


class Classifier:
    def __init__(self, llm):
        self.llm = llm

    def classify(self, problem: str) -> str:
        out = self.llm.chat(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": f"Problem:\n{problem}\n\nCategory:"}],
            temperature=0.0, max_tokens=4,
        ).text.lower()
        for word in re.findall(r"[a-z]+", out):
            if word in CATEGORIES:
                return word
        return "mixed"
