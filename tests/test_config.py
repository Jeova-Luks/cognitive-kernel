"""Tests for config.py."""
import pytest
from pathlib import Path
from config import load_config, Config


def test_load_toy_config():
    cfg = load_config(Path("configs/test_toy.yaml"))
    assert isinstance(cfg, Config)
    assert cfg.model.n_embd == 32
    assert cfg.model.n_layer == 2
    assert cfg.train.batch_size > 0
    assert cfg.train.max_iters > 0


def test_load_base_100m_config():
    cfg = load_config(Path("configs/base_100m.yaml"))
    assert cfg.model.n_embd == 768
    assert cfg.model.n_head == 12
    assert cfg.model.n_kv_head == 4
    assert cfg.model.n_layer == 12
    assert cfg.model.max_seq_len == 2048
    assert cfg.model.vocab_size == 32000


def test_missing_field_raises():
    """Dataclass attribute access raises AttributeError for unknown fields."""
    cfg = load_config(Path("configs/test_toy.yaml"))
    with pytest.raises(AttributeError):
        cfg.model.nonexistent_field
