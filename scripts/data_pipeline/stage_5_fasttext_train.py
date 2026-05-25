"""Stage 5a: Train a FastText quality classifier.

Positives = Wikipedia FA + top-starred GitHub Python + curated math.
Negatives = random Common Crawl + zero-star GitHub + spam patterns.

Output: quality_classifier.bin (committed to artifacts, NOT git).
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import fasttext
from datasets import load_dataset

from .common import get_logger

log = get_logger("stage5_train")

TRAIN_FILE = "fasttext_train.txt"
MODEL_FILE = "quality_classifier.bin"
N_POS = 50_000
N_NEG = 50_000


def format_label(label: str, text: str) -> str:
    """Convert (label, text) to FastText supervised format."""
    clean = text.replace("\n", " ").replace("\r", " ").strip()
    return f"__label__{label} {clean}"


def write_training_lines(samples: list[tuple[str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for label, text in samples:
            f.write(format_label(label, text) + "\n")


def gather_positives(n: int) -> list[tuple[str, str]]:
    """Pull high-quality samples from upstream sources."""
    pos: list[tuple[str, str]] = []

    log.info("positives: Wikipedia FA-like (heuristic: long en wiki articles) ...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        split="train", streaming=True,
    )
    for rec in ds:
        if len(pos) >= n // 2:
            break
        if len(rec["text"]) > 5000:    # proxy for FA/featured-article length
            pos.append(("high_quality", rec["text"][:2000]))

    log.info("positives: top-starred Python from the_stack ...")
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train", streaming=True,
    )
    for rec in ds:
        if len(pos) >= n:
            break
        if rec.get("language") != "Python":
            continue
        if (rec.get("revision_stars") or 0) < 1000:
            continue
        pos.append(("high_quality", (rec.get("content") or "")[:2000]))
    return pos


def gather_negatives(n: int) -> list[tuple[str, str]]:
    """Pull low-quality / noisy samples."""
    neg: list[tuple[str, str]] = []

    log.info("negatives: cc100 random (treated as low quality by default) ...")
    ds = load_dataset("cc100", lang="en", split="train", streaming=True)
    for rec in ds:
        if len(neg) >= n // 2:
            break
        text = rec.get("text") or ""
        if len(text) < 100:
            continue
        neg.append(("low_quality", text[:2000]))

    log.info("negatives: zero-star Python from the_stack ...")
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train", streaming=True,
    )
    for rec in ds:
        if len(neg) >= n:
            break
        if rec.get("language") != "Python":
            continue
        if (rec.get("revision_stars") or 0) > 0:
            continue
        neg.append(("low_quality", (rec.get("content") or "")[:2000]))
    return neg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./artifacts")
    parser.add_argument("--n-pos", type=int, default=N_POS)
    parser.add_argument("--n-neg", type=int, default=N_NEG)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("gathering positives ...")
    pos = gather_positives(args.n_pos)
    log.info(f"positives: {len(pos):,}")

    log.info("gathering negatives ...")
    neg = gather_negatives(args.n_neg)
    log.info(f"negatives: {len(neg):,}")

    samples = pos + neg
    random.shuffle(samples)

    train_path = out_dir / TRAIN_FILE
    write_training_lines(samples, train_path)
    log.info(f"wrote {len(samples):,} lines to {train_path}")

    log.info("training fasttext supervised ...")
    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=50,
        lr=0.5,
        dim=100,
        wordNgrams=2,
        loss="softmax",
    )

    model_path = out_dir / MODEL_FILE
    model.save_model(str(model_path))
    log.info(f"saved model to {model_path}")

    # Quick self-eval on the training set itself
    n_correct = 0
    for label, text in samples[:1000]:
        pred = model.predict(text.replace("\n", " "))[0][0]
        if pred == f"__label__{label}":
            n_correct += 1
    log.info(f"self-accuracy on 1000 train samples: {n_correct / 1000:.2%}")


if __name__ == "__main__":
    main()
