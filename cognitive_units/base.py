"""Base CognitiveUnit — preserved from the user's original cognitive_20.py.

This is the user's original architecture. The MVP extends this via subclass
(LearnableCognitiveUnit) so the original API stays untouched.

A CognitiveUnit is a small symbolic agent with:
- a name (e.g., "U-05")
- a specialty (e.g., "security_scanner")
- a list of rules: each rule is {condition: callable, response: str, weight: float}
- internal state (idle/active/dormant), memory, confidence

When activated, it tries each rule's condition; among those that fire, the
highest-weight wins and its response is the unit's output.
"""
from __future__ import annotations
from typing import Any, Callable


class CognitiveUnit:
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.state = "idle"
        self.memory: list[dict[str, Any]] = []
        self.rules: list[dict[str, Any]] = []
        self.confidence = 0.0

    def add_rule(
        self,
        condition_fn: Callable[[str, dict], bool],
        response: str,
        weight: float = 1.0,
    ) -> None:
        self.rules.append({
            "condition": condition_fn,
            "response": response,
            "weight": weight,
        })

    def activate(self, signal: str, context: dict | None = None) -> dict | None:
        ctx = context or {}
        fired: list[tuple[str, float]] = []
        for rule in self.rules:
            try:
                if rule["condition"](signal, ctx):
                    fired.append((rule["response"], rule["weight"]))
            except Exception:
                # Defensive: a single buggy rule should not crash the unit.
                pass
        if not fired:
            self.state = "dormant"
            self.confidence = 0.0
            return None
        best = max(fired, key=lambda x: x[1])
        self.confidence = best[1]
        self.state = "active"
        self.memory.append({"input": signal[:60], "output": best[0]})
        return {
            "unit": self.name,
            "specialty": self.specialty,
            "output": best[0],
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:
        filled = int(self.confidence * 10)
        bar = "█" * filled + "░" * (10 - filled)
        status = "💤" if self.state == "dormant" else ("⚡" if self.confidence >= 0.8 else "·")
        return (
            f"  {status} {self.name} [{self.specialty:<26}] "
            f"{bar} {self.confidence:.1f} | mem={len(self.memory)}"
        )
