"""Stage 4: L2 near-dedup via MinHash LSH.

Pure helpers (shingles, signature) are unit-tested. main() is the heavy
pipeline: build signatures for all docs, find clusters, keep longest per
cluster.

Memory: ~5-8 GB for ~100M docs. Run on Kaggle (30 GB RAM).
"""
from __future__ import annotations
import argparse
from typing import Iterator

from datasketch import MinHash, MinHashLSH

from .common import pull_dataset, push_dataset, stage_repo, get_logger

log = get_logger("stage4")

SHINGLE_K = 5
NUM_PERM = 256
JACCARD_THRESHOLD = 0.85


def shingles_of(text: str, k: int = SHINGLE_K) -> Iterator[str]:
    """Yield k-grams of whitespace-tokenized text."""
    tokens = text.lower().split()
    for i in range(len(tokens) - k + 1):
        yield " ".join(tokens[i:i + k])


def build_signature(text: str, num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles_of(text):
        m.update(sh.encode("utf-8"))
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", default=stage_repo(3, "filtered"))
    parser.add_argument("--output-repo", default=stage_repo(4, "deduped"))
    parser.add_argument("--threshold", type=float, default=JACCARD_THRESHOLD)
    parser.add_argument("--num-perm", type=int, default=NUM_PERM)
    args = parser.parse_args()

    log.info(f"pulling {args.input_repo}")
    ds = pull_dataset(args.input_repo)
    log.info(f"loaded {len(ds):,} docs; building signatures ...")

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    # Insert all signatures
    for idx, rec in enumerate(ds):
        sig = build_signature(rec["text"], num_perm=args.num_perm)
        lsh.insert(str(idx), sig, check_duplication=False)
        if idx % 10_000 == 0:
            log.info(f"  signed {idx:,} / {len(ds):,}")

    log.info("finding clusters ...")
    keep_idx: set[int] = set()
    dropped: set[int] = set()
    for idx in range(len(ds)):
        if idx in dropped:
            continue
        sig = build_signature(ds[idx]["text"], num_perm=args.num_perm)
        neighbors = [int(n) for n in lsh.query(sig)]
        if len(neighbors) == 1:
            keep_idx.add(idx)
            continue
        # Cluster: keep the longest doc, drop the rest
        longest = max(neighbors, key=lambda i: len(ds[i]["text"]))
        keep_idx.add(longest)
        for n in neighbors:
            if n != longest:
                dropped.add(n)

    log.info(f"clusters resolved: keeping {len(keep_idx):,}, dropping {len(dropped):,}")
    survived = ds.select(sorted(keep_idx))
    log.info(f"pushing {len(survived):,} to {args.output_repo}")
    push_dataset(survived, args.output_repo)


if __name__ == "__main__":
    main()
