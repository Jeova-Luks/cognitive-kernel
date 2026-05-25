"""Generate tiny synthetic training shards for smoke-testing the full pipeline.

These shards are random tokens — not meaningful for actual training. Their sole
purpose is to verify the trainer pipeline runs end-to-end (data → forward →
loss → backward → checkpoint → resume). For real training data, see Phase 2.
"""
from pathlib import Path
import numpy as np

OUT = Path("data/shards")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
# 2 training shards of 1M tokens each (~2MB each as uint16)
for i in range(2):
    arr = rng.integers(0, 32000, size=1_000_000, dtype=np.uint16)
    arr.tofile(OUT / f"train_{i:03d}.bin")

# 1 validation shard of 200k tokens
arr_val = rng.integers(0, 32000, size=200_000, dtype=np.uint16)
arr_val.tofile(OUT / "val_000.bin")

print(f"Wrote 3 smoke shards to {OUT}/")
print(f"  train_*.bin: 2 files, ~2MB each (1M uint16 tokens)")
print(f"  val_*.bin:   1 file,  ~400KB (200K uint16 tokens)")
