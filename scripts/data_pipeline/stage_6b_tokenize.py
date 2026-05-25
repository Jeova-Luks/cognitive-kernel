"""Stage 6b: Tokenize all surviving docs using tokenizer_fast.json.

Output: HF dataset with a single `tokens` column (List[int]) per doc.
"""
from __future__ import annotations
import argparse

from .common import pull_dataset, push_dataset, stage_repo, get_logger
from tokenizer_fast import FastBPETokenizer

log = get_logger("stage6b")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", default=stage_repo(5, "quality"))
    parser.add_argument("--output-repo", default=stage_repo(6, "tokenized"))
    parser.add_argument("--tokenizer", default="tokenizer_fast.json")
    parser.add_argument("--num-proc", type=int, default=4)
    args = parser.parse_args()

    tok = FastBPETokenizer(args.tokenizer)
    log.info(f"loaded tokenizer (vocab {tok.vocab_size})")

    ds = pull_dataset(args.input_repo)
    log.info(f"loaded {len(ds):,} docs")

    def encode(rec):
        rec["tokens"] = tok.encode(rec["text"])
        rec["n_tokens"] = len(rec["tokens"])
        return rec

    log.info("tokenizing ...")
    ds = ds.map(encode, num_proc=args.num_proc)

    # Remove the big raw text column; keep only what Phase 1B needs
    keep_cols = {"tokens", "n_tokens", "category", "source", "doc_id"}
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(drop_cols)

    total_tokens = sum(ds["n_tokens"])
    log.info(f"total tokens: {total_tokens:,}")
    log.info(f"pushing to {args.output_repo}")
    push_dataset(ds, args.output_repo)


if __name__ == "__main__":
    main()
