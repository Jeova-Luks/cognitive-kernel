"""Run the Cognitive Units MVP experiment.

End-to-end:
1. Generate (or load cached) 200-example security dataset via Claude Haiku.
2. Split 70/30 train/val.
3. Build a baseline LearnableCognitiveUnit (empty) and measure starting accuracy.
4. Run distillation for up to 20 iterations.
5. Compare against the hand-coded U-05 from cognitive_20.py (re-implemented).
6. Print a report and write metrics.json next to the dataset.

Usage (on Codespace, with `ANTHROPIC_API_KEY` env var or token file):

    export ANTHROPIC_API_KEY=$(cat ~/.anthropic_key)
    python -m cognitive_units.run_mvp
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from anthropic import Anthropic

from .base import CognitiveUnit
from .dataset_security import (
    DATASET_PATH, generate_dataset, save_dataset, load_dataset, split_train_val,
)
from .learnable import LearnableCognitiveUnit

METRICS_PATH = Path("cognitive_units/data/metrics.json")


def build_handcoded_u5() -> CognitiveUnit:
    """Re-implement U-05 from cognitive_20.py (the hand-coded baseline).

    NOTE: the strings "exec(", "eval(", "os.system" etc. below are LITERAL
    KEYWORDS that the rules SEARCH FOR inside the input text. They are NOT
    executed. The point of these rules is precisely to flag input strings
    that mention these dangerous patterns.
    """
    u5 = CognitiveUnit("U-05-handcoded", "security_scanner")
    u5.add_rule(
        lambda s, c: any(w in s.lower() for w in ["sql", "select", "insert", "delete", "where", "input", "usuário", "user input"])
                     and "prepared" not in s.lower(),
        "SQL_INJECTION_RISK", 1.0,
    )
    u5.add_rule(
        lambda s, c: any(w in s.lower() for w in ["exec(", "eval(", "os.system", "subprocess", "shell=true", "__import__"]),
        "CODE_INJECTION_RISK", 1.0,
    )
    u5.add_rule(
        lambda s, c: any(w in s.lower() for w in ["senha", "password", "token", "api_key", "secret", "credencial"])
                     and "print" in s.lower(),
        "CREDENTIAL_LEAK_RISK", 1.0,
    )
    u5.add_rule(
        lambda s, c: any(w in s.lower() for w in ["jwt", "autenticação", "auth", "login", "sessão", "cookie", "csrf", "xss"]),
        "AUTH_CONTEXT", 0.85,
    )
    u5.add_rule(
        lambda s, c: any(w in s.lower() for w in ["open(", "file", "ler arquivo", "write", "caminho", "path"])
                     and "input" in s.lower(),
        "PATH_TRAVERSAL_RISK", 0.9,
    )
    u5.add_rule(lambda s, c: True, "CLEAN", 0.3)
    return u5


def accuracy(unit: CognitiveUnit, dataset: list[tuple[str, str]]) -> float:
    if not dataset:
        return 0.0
    correct = 0
    for text, expected in dataset:
        result = unit.activate(text)
        pred = result["output"] if result else "UNKNOWN"
        if pred == expected:
            correct += 1
    return correct / len(dataset)


def load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    home_key = Path.home() / ".anthropic_key"
    if home_key.exists():
        return home_key.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "Anthropic API key not found. Set ANTHROPIC_API_KEY env var or "
        "create ~/.anthropic_key containing the key."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-dataset", action="store_true",
                        help="Force re-generate the dataset (default: use cached if exists)")
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--target-acc", type=float, default=0.85)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    client = Anthropic(api_key=load_api_key())

    # === Step 1: dataset ===
    if args.regenerate_dataset or not DATASET_PATH.exists():
        print(f"[1] generating dataset via {args.model} ...")
        examples = generate_dataset(client, model=args.model)
        save_dataset(examples)
        print(f"    saved {len(examples)} examples to {DATASET_PATH}")
    else:
        print(f"[1] loading cached dataset from {DATASET_PATH}")
        examples = load_dataset()
        print(f"    {len(examples)} examples loaded")

    train, val = split_train_val(examples, val_ratio=0.3, seed=42)
    print(f"    split: train={len(train)} val={len(val)}")

    # Print label distribution
    from collections import Counter
    label_dist = Counter(label for _, label in examples)
    print(f"    label distribution: {dict(label_dist)}")

    # === Step 2: baseline (empty) ===
    print("\n[2] empty unit (no rules) baseline:")
    empty = LearnableCognitiveUnit("U-05-empty", "security_scanner")
    baseline_train = accuracy(empty, train)
    baseline_val = accuracy(empty, val)
    print(f"    train_acc={baseline_train:.3f}  val_acc={baseline_val:.3f}")

    # === Step 3: hand-coded baseline ===
    print("\n[3] hand-coded U-05 baseline:")
    handcoded = build_handcoded_u5()
    hand_train = accuracy(handcoded, train)
    hand_val = accuracy(handcoded, val)
    print(f"    train_acc={hand_train:.3f}  val_acc={hand_val:.3f}")
    print(f"    rules: {len(handcoded.rules)}")

    # === Step 4: distillation ===
    print(f"\n[4] running distillation (max_iters={args.max_iters}, target_val_acc={args.target_acc}) ...")
    report = empty.learn_from_dataset(
        train=train,
        val=val,
        client=client,
        model=args.model,
        max_iters=args.max_iters,
        target_accuracy=args.target_acc,
    )

    # === Step 5: final report ===
    print("\n" + "=" * 60)
    print(" FINAL REPORT")
    print("=" * 60)
    print(f" empty start       : train={report['baseline_train_acc']:.3f} val={report['baseline_val_acc']:.3f}")
    print(f" distilled         : train={report['final_train_acc']:.3f} val={report['final_val_acc']:.3f}  ({report['n_rules_learned']} rules)")
    print(f" hand-coded U-05   : train={hand_train:.3f} val={hand_val:.3f}  ({len(handcoded.rules)} rules)")
    print()
    if report["final_val_acc"] >= hand_val:
        verdict = "✅ DISTILLED >= HAND-CODED  →  thesis viable"
    elif report["final_val_acc"] >= 0.3:
        verdict = "🟡 DISTILLED LEARNED SOMETHING  →  investigate scaling"
    else:
        verdict = "🔴 DISTILLED FAILED TO LEARN  →  pivot recommended"
    print(f" verdict           : {verdict}")
    print("=" * 60)

    # === Step 6: persist metrics ===
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": args.model,
        "n_examples": len(examples),
        "n_train": len(train),
        "n_val": len(val),
        "label_distribution": dict(label_dist),
        "baseline_empty": {
            "train_acc": baseline_train,
            "val_acc": baseline_val,
        },
        "baseline_handcoded": {
            "train_acc": hand_train,
            "val_acc": hand_val,
            "n_rules": len(handcoded.rules),
        },
        "distilled": report,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nmetrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
