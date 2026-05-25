"""Tests for dataset.py streaming loader."""
import numpy as np
import pytest
import torch
from dataset import ShardedTokenDataset


@pytest.fixture(scope="module")
def toy_shards(tmp_path_factory):
    """Generate three small training shards and one validation shard."""
    fixtures_dir = tmp_path_factory.mktemp("shards")
    rng = np.random.default_rng(0)
    for i in range(3):
        arr = rng.integers(0, 256, size=10_000, dtype=np.uint16)
        arr.tofile(fixtures_dir / f"toy_train_{i:03d}.bin")
    arr_val = rng.integers(0, 256, size=2_000, dtype=np.uint16)
    arr_val.tofile(fixtures_dir / "toy_val_000.bin")
    return fixtures_dir


def test_dataset_yields_correct_shapes(toy_shards):
    ds = ShardedTokenDataset(
        glob_pattern=str(toy_shards / "toy_train_*.bin"),
        block_size=16,
        seed=42,
    )
    x, y = ds.get_batch(batch_size=4)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_dataset_deterministic_given_seed(toy_shards):
    ds_a = ShardedTokenDataset(
        glob_pattern=str(toy_shards / "toy_train_*.bin"),
        block_size=16, seed=42)
    ds_b = ShardedTokenDataset(
        glob_pattern=str(toy_shards / "toy_train_*.bin"),
        block_size=16, seed=42)
    xa, ya = ds_a.get_batch(batch_size=4)
    xb, yb = ds_b.get_batch(batch_size=4)
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)


def test_dataset_different_seeds_differ(toy_shards):
    ds_a = ShardedTokenDataset(glob_pattern=str(toy_shards / "toy_train_*.bin"),
                               block_size=16, seed=42)
    ds_b = ShardedTokenDataset(glob_pattern=str(toy_shards / "toy_train_*.bin"),
                               block_size=16, seed=43)
    xa, _ = ds_a.get_batch(batch_size=4)
    xb, _ = ds_b.get_batch(batch_size=4)
    assert not torch.equal(xa, xb)


def test_dataset_raises_on_empty_glob():
    with pytest.raises(FileNotFoundError):
        ShardedTokenDataset(glob_pattern="nonexistent_*.bin", block_size=16, seed=0)
