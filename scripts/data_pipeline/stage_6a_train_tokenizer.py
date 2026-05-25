"""Stage 6a: Train BPE tokenizer on a 1 GB representative sample.

Output: tokenizer_fast.json (committed to git)."""
from __future__ import annotations
import argparse
import random
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

from .common import pull_dataset, stage_repo, get_logger, TARGETS

log = get_logger("stage6a")

SAMPLE_BYTES_PER_CATEGORY = {
    "python":        300_000_000,
    "math":          200_000_000,
    "english_prose": 300_000_000,
    "sexp":          100_000_000,
    "pt_br":         100_000_000,
}
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|cdsl_start|>",
    "<|cdsl_end|>",
    "<|tool_call|>",
    "<|tool_result|>",
]


def sample_corpus(input_repo: str, out_path: Path) -> None:
    """Sample ~1 GB of representative text proportional to mix C."""
    ds = pull_dataset(input_repo)
    log.info(f"loaded {len(ds):,} docs for sampling")
    rng = random.Random(42)

    by_cat: dict[str, list[int]] = {c: [] for c in TARGETS}
    for i, rec in enumerate(ds):
        by_cat[rec["category"]].append(i)
    log.info({c: len(v) for c, v in by_cat.items()})

    out = open(out_path, "w", encoding="utf-8")
    for cat, budget in SAMPLE_BYTES_PER_CATEGORY.items():
        rng.shuffle(by_cat[cat])
        consumed = 0
        for idx in by_cat[cat]:
            if consumed >= budget:
                break
            text = ds[idx]["text"]
            out.write(text + "\n")
            consumed += len(text)
        log.info(f"sampled {consumed:,} bytes from {cat}")
    out.close()


def train_tokenizer(corpus_path: Path, out_path: Path) -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )

    log.info(f"training BPE on {corpus_path} ...")
    tokenizer.train([str(corpus_path)], trainer)
    log.info(f"trained, vocab size = {tokenizer.get_vocab_size()}")
    tokenizer.save(str(out_path))
    log.info(f"saved tokenizer to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", default=stage_repo(5, "quality"))
    parser.add_argument("--workdir", default="./artifacts")
    parser.add_argument("--out", default="tokenizer_fast.json")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    corpus_path = workdir / "tokenizer_train_sample.txt"

    sample_corpus(args.input_repo, corpus_path)
    train_tokenizer(corpus_path, Path(args.out))


if __name__ == "__main__":
    main()
