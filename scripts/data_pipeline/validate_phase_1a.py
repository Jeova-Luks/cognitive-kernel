"""Run the Phase 1A Definition of Done checks (DoD #4 sanity + #5 loader)."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np

from .common import get_logger
from tokenizer_fast import FastBPETokenizer

log = get_logger("validate")


def check_files_present(drive_root: Path) -> bool:
    shards = list((drive_root / "shards").glob("*.bin"))
    train = [p for p in shards if p.name.startswith("train_")]
    val = [p for p in shards if p.name.startswith("val_")]
    log.info(f"shards: train={len(train)}, val={len(val)}")
    manifest = drive_root / "manifest.json"
    log.info(f"manifest present: {manifest.exists()}")
    return len(train) > 0 and len(val) > 0 and manifest.exists()


def sanity_decode_samples(drive_root: Path, tokenizer_path: Path) -> None:
    tok = FastBPETokenizer(tokenizer_path)
    shards = list((drive_root / "shards").glob("train_*.bin"))
    random.shuffle(shards)
    for shard in shards[:3]:
        data = np.fromfile(shard, dtype=np.uint16)
        for _ in range(3):
            offset = random.randint(0, len(data) - 200)
            sample = data[offset:offset + 200].tolist()
            text = tok.decode(sample)
            log.info(f"  [{shard.name} @ {offset}]: {text[:120]}...")


def check_dataset_loader(drive_root: Path) -> None:
    """Use Phase 0's ShardedTokenDataset to verify the loader works."""
    from dataset import ShardedTokenDataset
    glob = str(drive_root / "shards" / "train_*.bin")
    ds = ShardedTokenDataset(glob_pattern=glob, block_size=512, seed=42)
    x, y = ds.get_batch(batch_size=4)
    assert x.shape == (4, 512), f"unexpected shape: {x.shape}"
    assert y.shape == (4, 512)
    log.info(f"loader OK; x.shape={x.shape}, x[0,:10]={x[0,:10].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/cognitive-kernel/data",
    )
    parser.add_argument("--tokenizer", default="tokenizer_fast.json")
    args = parser.parse_args()

    drive_root = Path(args.drive_root)

    log.info("=== DoD #3: files present ===")
    assert check_files_present(drive_root), "shards or manifest missing"

    log.info("=== DoD #2: manifest content ===")
    manifest = json.loads((drive_root / "manifest.json").read_text())
    log.info(json.dumps(manifest, indent=2))

    log.info("=== DoD #4: sanity decode samples ===")
    sanity_decode_samples(drive_root, Path(args.tokenizer))

    log.info("=== DoD #5: dataset loader ===")
    check_dataset_loader(drive_root)

    log.info("All Phase 1A validation checks passed (DoD #6 signal test runs separately).")


if __name__ == "__main__":
    main()
