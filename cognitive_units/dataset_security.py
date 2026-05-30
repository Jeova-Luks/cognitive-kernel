"""Generate (and load) a labelled dataset of 200 code snippets for the
security classification task — used as the test bed for the MVP experiment.

NOTE TO READERS / LINTERS: this file contains string literals that describe
dangerous Python constructs (eval, exec, os.system, shell=True, ...). Those
are PROMPT TEXT sent to an LLM so it can generate training examples of
vulnerable code patterns. Nothing in this module actually executes any of
those constructs — there is no eval/exec/os.system call site here.

Generation strategy: ask Claude Haiku to produce a balanced set with 6
labels (5 risks + 1 CLEAN). We then save to a JSON file so subsequent runs
of the experiment reuse the same dataset (reproducible).

Labels mirror U-05's `security_scanner` outputs:
- SQL_INJECTION_RISK
- CODE_INJECTION_RISK
- CREDENTIAL_LEAK_RISK
- AUTH_CONTEXT
- PATH_TRAVERSAL_RISK
- CLEAN
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic

LABELS = [
    "SQL_INJECTION_RISK",
    "CODE_INJECTION_RISK",
    "CREDENTIAL_LEAK_RISK",
    "AUTH_CONTEXT",
    "PATH_TRAVERSAL_RISK",
    "CLEAN",
]

TOTAL_EXAMPLES = 200
EXAMPLES_PER_LABEL = TOTAL_EXAMPLES // len(LABELS)   # ≈ 33 per label

DATASET_PATH = Path("cognitive_units/data/security_dataset.json")

GEN_SYSTEM_PROMPT = """You are generating a labelled dataset for a CODE \
CLASSIFIER training experiment. Your output will be used to train and \
evaluate a small symbolic system.

Produce realistic, diverse code snippets (Python, JavaScript, SQL, shell) \
and short natural-language descriptions of code situations. Each item \
must be 30-300 characters. Mix code with portuguese and english code \
review comments. Vary style: some are pure code, some are bug reports, \
some are PR descriptions.

CRITICAL: items labeled CLEAN must be GENUINELY safe code that LOOKS \
similar to vulnerable code but isn't. E.g., SQL with prepared statements, \
eval of a literal constant, env-var based secrets, validated paths. The \
classifier must learn to distinguish these from real vulnerabilities.

Respond ONLY with JSON array. No preamble, no markdown fences.
"""


def _build_user_prompt(label: str, n: int) -> str:
    descriptions = {
        "SQL_INJECTION_RISK": (
            "Code that builds SQL queries by string concatenation with user input, "
            "without parameterised queries."
        ),
        "CODE_INJECTION_RISK": (
            "Code that calls eval(), exec(), os.system(), subprocess with shell=True, "
            "or similar, on data that comes from user input."
        ),
        "CREDENTIAL_LEAK_RISK": (
            "Code that prints, logs, or otherwise exposes passwords, tokens, API keys, "
            "or secret credentials."
        ),
        "AUTH_CONTEXT": (
            "Code discussing authentication, login flows, JWT, sessions, CSRF, "
            "or related security primitives — not necessarily vulnerable, just AUTH-related."
        ),
        "PATH_TRAVERSAL_RISK": (
            "Code that opens or reads files using a path constructed from user input "
            "without validation against directory traversal."
        ),
        "CLEAN": (
            "Code that LOOKS like it could fall into one of the above risk categories "
            "but is actually safe. Examples: parameterised SQL, eval of literal constants, "
            "secrets from environment variables, paths validated against an allowlist."
        ),
    }
    return (
        f"Generate {n} realistic examples for label {label!r}.\n\n"
        f"Description of this label: {descriptions[label]}\n\n"
        f"Output JSON array of {n} strings, each 30-300 chars."
    )


def generate_dataset(
    client: "Anthropic",
    model: str = "claude-haiku-4-5-20251001",
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Call Claude to generate the full 200-example dataset, label-balanced."""
    all_examples: list[tuple[str, str]] = []
    for label in LABELS:
        prompt = _build_user_prompt(label, EXAMPLES_PER_LABEL)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=GEN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.rstrip().endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()
        try:
            items = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"could not parse generation for {label}: {e}\n{text[:300]}")
        if not isinstance(items, list):
            raise RuntimeError(f"expected list for {label}, got {type(items)}")
        for item in items:
            if isinstance(item, str) and 20 <= len(item) <= 600:
                all_examples.append((item, label))

    # Shuffle so train/val splits are not label-clustered
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    return all_examples


def save_dataset(examples: list[tuple[str, str]], path: Path = DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            [{"text": t, "label": l} for t, l in examples],
            f, indent=2, ensure_ascii=False,
        )


def load_dataset(path: Path = DATASET_PATH) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return [(r["text"], r["label"]) for r in rows]


def split_train_val(
    examples: list[tuple[str, str]],
    val_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deterministic train/val split."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]
