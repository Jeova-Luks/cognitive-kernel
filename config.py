"""YAML-backed configuration loader for trainer and model."""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    n_embd: int
    n_head: int
    n_kv_head: Optional[int]
    n_layer: int
    max_seq_len: int
    vocab_size: int
    grad_checkpoint: bool = True


@dataclass
class TrainConfig:
    batch_size: int            # physical batch (per device)
    grad_accum_steps: int      # effective batch = batch_size * grad_accum_steps
    block_size: int            # sequence length per sample
    max_iters: int
    learning_rate: float
    warmup_iters: int
    min_lr: float
    weight_decay: float
    grad_clip: float
    eval_interval: int
    eval_iters: int
    checkpoint_interval: int
    optimizer: str = "adamw_8bit"   # adamw | adamw_8bit | muon


@dataclass
class DataConfig:
    train_shards_glob: str      # e.g. "MyDrive/cognitive-kernel/data/train_*.bin"
    val_shards_glob: str
    seed: int


@dataclass
class LogConfig:
    project: str = "cognitive-kernel"
    run_name: str = ""
    log_interval: int = 10
    wandb_mode: str = "online"  # online | offline | disabled


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    log: LogConfig = field(default_factory=LogConfig)


def load_config(path: Path) -> Config:
    """Load a YAML config into nested dataclasses. Strict — missing required
    fields raise TypeError from the dataclass constructor."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        data=DataConfig(**raw["data"]),
        log=LogConfig(**raw.get("log", {})),
    )
