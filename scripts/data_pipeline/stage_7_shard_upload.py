"""Stage 7: Shard the tokenized corpus into nanoGPT-style .bin files and
upload to Google Drive.

Run on Google Colab (drive.mount works natively there).
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np

from .common import pull_dataset, stage_repo, get_logger, TARGETS

log = get_logger("stage7")

TOKENS_PER_SHARD = 100_000_000   # 100M tokens × 2 bytes = ~200 MB shards
VAL_FRACTION = 0.05               # 5% reserved for validation


def shard_arrays(
    all_tokens: np.ndarray,
    shard_size: int,
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(all_tokens) + shard_size - 1) // shard_size
    for i in range(n_shards):
        chunk = all_tokens[i * shard_size : (i + 1) * shard_size]
        path = out_dir / f"{prefix}_{i:03d}.bin"
        chunk.astype(np.uint16).tofile(path)
        paths.append(path)
        log.info(f"  wrote {path.name} ({len(chunk):,} tokens, {path.stat().st_size:,} bytes)")
    return paths


def category_breakdown(ds) -> dict[str, int]:
    """Count tokens per category in a HF Dataset."""
    counts: dict[str, int] = {c: 0 for c in TARGETS}
    for rec in ds:
        counts[rec["category"]] += rec["n_tokens"]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", default=stage_repo(6, "tokenized"))
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/cognitive-kernel/data",
        help="Target Drive folder for shards + manifest",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = pull_dataset(args.input_repo)
    log.info(f"loaded {len(ds):,} docs")

    cat_breakdown = category_breakdown(ds)
    log.info(f"per-category token counts: {cat_breakdown}")

    log.info("concatenating all tokens ...")
    all_tokens: list[int] = []
    rng = random.Random(args.seed)
    # Shuffle doc order for better mix
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for i in indices:
        all_tokens.extend(ds[i]["tokens"])
    all_tokens_np = np.asarray(all_tokens, dtype=np.uint16)
    log.info(f"total tokens: {len(all_tokens_np):,}")

    n_val = int(len(all_tokens_np) * VAL_FRACTION)
    n_train = len(all_tokens_np) - n_val
    train_arr = all_tokens_np[:n_train]
    val_arr = all_tokens_np[n_train:]
    log.info(f"split: train={n_train:,}, val={n_val:,}")

    drive_root = Path(args.drive_root)
    shards_dir = drive_root / "shards"
    train_paths = shard_arrays(train_arr, TOKENS_PER_SHARD, shards_dir, "train")
    val_paths = shard_arrays(val_arr, TOKENS_PER_SHARD, shards_dir, "val")

    manifest = {
        "total_tokens": int(len(all_tokens_np)),
        "train_tokens": int(n_train),
        "val_tokens":   int(n_val),
        "n_train_shards": len(train_paths),
        "n_val_shards": len(val_paths),
        "tokens_per_shard": TOKENS_PER_SHARD,
        "category_breakdown": cat_breakdown,
        "seed": args.seed,
        "produced_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    manifest_path = drive_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"wrote manifest to {manifest_path}")
    log.info("done")


if __name__ == "__main__":
    main()
