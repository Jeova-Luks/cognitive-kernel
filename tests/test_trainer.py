"""Tests for trainer.py."""
import numpy as np
import pytest
import torch
from pathlib import Path
from config import load_config
from trainer import Trainer


@pytest.fixture
def toy_shards():
    """Generate fixtures that match configs/test_toy.yaml's glob.

    Writes directly into tests/fixtures/ (gitignored as *.bin)."""
    fixtures = Path("tests/fixtures")
    fixtures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(2):
        rng.integers(0, 256, size=5_000, dtype=np.uint16).tofile(
            fixtures / f"toy_train_{i:03d}.bin")
    rng.integers(0, 256, size=1_000, dtype=np.uint16).tofile(
        fixtures / "toy_val_000.bin")
    yield fixtures


def test_trainer_runs_two_steps_on_toy(toy_shards, tmp_path):
    cfg = load_config(Path("configs/test_toy.yaml"))
    cfg.train.max_iters = 2  # override for speed
    trainer = Trainer(cfg, output_dir=tmp_path / "out")
    initial_loss = trainer.train_step()
    second_loss = trainer.train_step()
    assert isinstance(initial_loss, float)
    assert isinstance(second_loss, float)
    assert trainer.step == 2


def test_autocast_used_in_train_step(toy_shards, tmp_path, monkeypatch):
    """Verify torch.amp.autocast is entered during train_step."""
    cfg = load_config(Path("configs/test_toy.yaml"))
    trainer = Trainer(cfg, output_dir=tmp_path / "out")

    autocast_calls = {"count": 0}
    real_autocast = torch.amp.autocast

    def tracking_autocast(*args, **kwargs):
        autocast_calls["count"] += 1
        return real_autocast(*args, **kwargs)

    monkeypatch.setattr(torch.amp, "autocast", tracking_autocast)
    trainer.train_step()
    assert autocast_calls["count"] >= 1, "torch.amp.autocast was not called"
