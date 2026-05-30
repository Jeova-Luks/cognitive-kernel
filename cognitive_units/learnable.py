"""LearnableCognitiveUnit — subclass of CognitiveUnit that can learn rules.

Preserves the entire base API. Adds:
- predict(text) — returns the unit's output label on an unlabelled input
- accuracy(dataset) — computes accuracy on a labelled (input, expected) dataset
- learn_from_dataset(...) — runs the LLM distillation loop until convergence

The distillation loop itself lives in distillation.py — this class only
exposes the entry point so callers don't need to wire the Claude client.
"""
from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

from .base import CognitiveUnit

if TYPE_CHECKING:
    from anthropic import Anthropic

# Type alias: a labelled example
LabelledExample = tuple[str, str]   # (text, expected_label)

# Fallback label used when no rule fires
DEFAULT_LABEL = "UNKNOWN"


class LearnableCognitiveUnit(CognitiveUnit):
    def predict(self, text: str, context: dict | None = None) -> str:
        """Return the unit's predicted label for `text`, or DEFAULT_LABEL."""
        result = self.activate(text, context)
        if result is None:
            return DEFAULT_LABEL
        return result["output"]

    def accuracy(
        self,
        dataset: Iterable[LabelledExample],
        context_fn=None,
    ) -> float:
        """Compute accuracy on a labelled dataset.

        `context_fn(text) -> dict` can supply per-example context (used in
        layered networks). Default: empty context.
        """
        examples = list(dataset)
        if not examples:
            return 0.0
        correct = 0
        for text, expected in examples:
            ctx = context_fn(text) if context_fn else {}
            if self.predict(text, ctx) == expected:
                correct += 1
        return correct / len(examples)

    def errors(
        self,
        dataset: Iterable[LabelledExample],
        context_fn=None,
    ) -> list[tuple[str, str, str]]:
        """Return list of (text, expected, predicted) for examples we got wrong."""
        out: list[tuple[str, str, str]] = []
        for text, expected in dataset:
            ctx = context_fn(text) if context_fn else {}
            pred = self.predict(text, ctx)
            if pred != expected:
                out.append((text, expected, pred))
        return out

    def learn_from_dataset(
        self,
        train: list[LabelledExample],
        val: list[LabelledExample],
        client: "Anthropic",
        model: str = "claude-haiku-4-5-20251001",
        max_iters: int = 20,
        errors_per_round: int = 5,
        target_accuracy: float = 0.85,
        verbose: bool = True,
    ) -> dict:
        """Run the LLM distillation loop.

        At each iteration:
        1. Identify examples in TRAIN the unit currently gets wrong.
        2. Sample a small batch of those errors.
        3. Ask Claude to propose a single rule (Python lambda + response label
           + weight) that distinguishes those errors correctly.
        4. Sandbox-compile the lambda.
        5. Temporarily add the rule; recompute TRAIN accuracy.
        6. Keep the rule iff it strictly improves TRAIN accuracy; otherwise
           discard.
        7. Stop when VAL accuracy reaches `target_accuracy` or iteration
           budget exhausted.

        Returns a report dict with metrics per iteration and final accuracies.
        """
        # Local import to keep the type-only dependency lazy
        from .distillation import propose_rule, RuleProposalError

        history: list[dict] = []
        baseline_train = self.accuracy(train)
        baseline_val = self.accuracy(val)

        if verbose:
            print(f"[init] train_acc={baseline_train:.3f} val_acc={baseline_val:.3f}")

        for it in range(max_iters):
            errs = self.errors(train)
            if not errs:
                if verbose:
                    print(f"[iter {it}] no train errors left; stopping")
                break

            sample = errs[:errors_per_round]
            try:
                rule = propose_rule(sample, self.specialty, client, model)
            except RuleProposalError as e:
                if verbose:
                    print(f"[iter {it}] proposal failed: {e}; retrying next iter")
                history.append({"iter": it, "skipped": True, "reason": str(e)})
                continue

            # Try the rule
            old_train_acc = self.accuracy(train)
            self.rules.append({
                "condition": rule["fn"],
                "response": rule["response"],
                "weight": rule["weight"],
            })
            new_train_acc = self.accuracy(train)

            if new_train_acc > old_train_acc:
                kept = True
            else:
                # Roll back
                self.rules.pop()
                kept = False

            val_acc = self.accuracy(val)
            history.append({
                "iter": it,
                "kept": kept,
                "rule_response": rule["response"],
                "rule_weight": rule["weight"],
                "rule_lambda": rule["lambda_str"],
                "train_acc": new_train_acc if kept else old_train_acc,
                "val_acc": val_acc,
                "n_rules": len(self.rules),
            })

            if verbose:
                kept_str = "+" if kept else "-"
                print(f"[iter {it}] {kept_str} {rule['response']:<30} "
                      f"train={new_train_acc if kept else old_train_acc:.3f} "
                      f"val={val_acc:.3f} rules={len(self.rules)}")

            if val_acc >= target_accuracy:
                if verbose:
                    print(f"[iter {it}] reached target val_acc={val_acc:.3f}; stopping")
                break

        final_train = self.accuracy(train)
        final_val = self.accuracy(val)

        return {
            "baseline_train_acc": baseline_train,
            "baseline_val_acc": baseline_val,
            "final_train_acc": final_train,
            "final_val_acc": final_val,
            "n_rules_learned": len(self.rules),
            "iterations": history,
        }
