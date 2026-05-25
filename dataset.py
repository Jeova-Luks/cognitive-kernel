"""Streaming token dataset for pre-tokenized .bin shards (uint16 nanoGPT format)."""
import glob as glob_module
from typing import Tuple
import numpy as np
import torch


class ShardedTokenDataset:
    """Reads .bin shards (uint16 tokens) and yields random (x, y) batches.

    Designed for Google-Drive-hosted data:
    - Uses np.memmap so the OS handles paging from disk; no small random reads.
    - Random offsets are within a single mmap'd shard chosen per batch
      (locality of reference keeps Drive happy).
    """

    def __init__(self, glob_pattern: str, block_size: int, seed: int):
        self.glob_pattern = glob_pattern
        self.block_size = block_size
        self.shard_paths = sorted(glob_module.glob(glob_pattern))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards matched: {glob_pattern}")
        self.rng = np.random.default_rng(seed)
        # Lazy-open shards on first access
        self._mmaps = {}

    def _shard(self, path: str) -> np.memmap:
        if path not in self._mmaps:
            self._mmaps[path] = np.memmap(path, dtype=np.uint16, mode="r")
        return self._mmaps[path]

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a (batch_size, block_size) pair of input/target token tensors."""
        # Pick one shard for the entire batch (locality of reference)
        path = self.shard_paths[self.rng.integers(0, len(self.shard_paths))]
        data = self._shard(path)
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"Shard {path} too small ({len(data)} tokens) for block_size={self.block_size}"
            )
        offsets = self.rng.integers(0, max_start, size=batch_size)
        xs = np.stack([np.asarray(data[i : i + self.block_size], dtype=np.int64)
                       for i in offsets])
        ys = np.stack([np.asarray(data[i + 1 : i + 1 + self.block_size], dtype=np.int64)
                       for i in offsets])
        return torch.from_numpy(xs), torch.from_numpy(ys)

    def state_dict(self) -> dict:
        """Capture RNG state for bit-identical resume."""
        return {"rng_state": self.rng.bit_generator.state,
                "glob_pattern": self.glob_pattern,
                "block_size": self.block_size}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng_state"]
