"""LLM rule proposal: asks Claude for a single Python lambda that helps
distinguish a batch of misclassified examples.

The Claude response is constrained to a strict JSON schema so we can parse
it deterministically without regex on prose. The lambda string is then
passed through `safe_compile_lambda` before being returned.
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any

from .sandbox import safe_compile_lambda, UnsafeLambdaError

if TYPE_CHECKING:
    from anthropic import Anthropic


class RuleProposalError(Exception):
    """Raised when the LLM proposal cannot be parsed or sandbox-rejected."""


SYSTEM_PROMPT = """You are helping a small symbolic system LEARN classification rules \
from a dataset. The system is composed of "cognitive units" — each unit owns a \
list of Python lambdas as rules. A rule fires when its condition lambda \
returns True; the rule with the highest weight wins.

Your job: given a batch of examples the unit is currently misclassifying, \
propose ONE new rule that, when added, would correctly classify these \
examples (or most of them) WITHOUT breaking other examples.

CONSTRAINTS on the lambda you produce:
- Form: lambda s, c: <boolean expression>
- `s` is the input string (the text to classify).
- `c` is the context dict (you usually don't need it).
- Allowed string methods: lower, upper, strip, split, startswith, endswith,
  count, find, replace, isdigit, isalpha, isalnum, isspace.
- Allowed builtins: len, any, all, min, max, sum, sorted, set, list, str.
- Allowed dict methods on `c`: get, keys, values, items.
- NO imports, NO eval/exec, NO dunder access, NO file operations.
- Single line, no nested lambdas.
- Make the rule SPECIFIC enough to be useful but not match too much.

Respond with JSON ONLY, no preamble, no markdown fences. Exact schema:
{
  "response": "LABEL_TO_EMIT",
  "weight": 0.0 to 1.0,
  "lambda_str": "lambda s, c: <expression>",
  "reasoning": "1-2 sentences why this rule works"
}
"""


def _build_user_prompt(
    errors: list[tuple[str, str, str]],
    specialty: str,
) -> str:
    """Render the misclassified examples into a prompt for Claude."""
    lines = [
        f"Unit specialty: {specialty}",
        "",
        f"The unit got these {len(errors)} examples WRONG:",
        "",
    ]
    for i, (text, expected, predicted) in enumerate(errors, 1):
        # Truncate very long texts so the prompt stays small
        snippet = text if len(text) <= 400 else text[:400] + "..."
        lines.append(f"Example {i}:")
        lines.append(f"  Input: {snippet!r}")
        lines.append(f"  Expected label: {expected}")
        lines.append(f"  Unit emitted:   {predicted}")
        lines.append("")
    lines.append(
        "Propose ONE rule that helps classify these correctly. "
        "Return JSON only, matching the schema in the system prompt."
    )
    return "\n".join(lines)


def _parse_response(text: str) -> dict[str, Any]:
    """Extract the JSON payload from Claude's response."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        # remove first line (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuleProposalError(f"could not parse JSON: {e}\nraw: {text[:300]}")


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {"response", "weight", "lambda_str"}
    missing = required - set(payload.keys())
    if missing:
        raise RuleProposalError(f"missing keys: {missing}")
    if not isinstance(payload["response"], str) or not payload["response"]:
        raise RuleProposalError("`response` must be a non-empty string")
    weight = payload["weight"]
    if not isinstance(weight, (int, float)) or not (0.0 <= weight <= 1.0):
        raise RuleProposalError(f"`weight` must be in [0, 1], got {weight!r}")
    if not isinstance(payload["lambda_str"], str):
        raise RuleProposalError("`lambda_str` must be a string")


def propose_rule(
    errors: list[tuple[str, str, str]],
    specialty: str,
    client: "Anthropic",
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Ask Claude to propose ONE rule that classifies the given errors.

    Returns a dict with keys: response, weight, lambda_str, reasoning, fn
    where `fn` is the sandbox-compiled callable.
    Raises RuleProposalError on any failure (parse, validation, sandbox).
    """
    user_prompt = _build_user_prompt(errors, specialty)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Extract text from response (Anthropic SDK returns content blocks)
    text_blocks = [b.text for b in response.content if hasattr(b, "text")]
    if not text_blocks:
        raise RuleProposalError("Claude returned no text content")
    raw = "".join(text_blocks)

    payload = _parse_response(raw)
    _validate_payload(payload)

    # Sandbox-compile the lambda
    try:
        fn = safe_compile_lambda(payload["lambda_str"])
    except UnsafeLambdaError as e:
        raise RuleProposalError(f"sandbox rejected lambda: {e}")

    return {
        "response": payload["response"],
        "weight": float(payload["weight"]),
        "lambda_str": payload["lambda_str"],
        "reasoning": payload.get("reasoning", ""),
        "fn": fn,
    }
