"""Cognitive Kernel trainer. Loads a Config, builds model + optimizer + dataset,
runs the train/eval/checkpoint loop. Designed to survive Colab disconnects."""
from __future__ import annotations
import math
import random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch

from model import GPTModel
from config import Config
from dataset import ShardedTokenDataset


class Trainer:
    def __init__(self, cfg: Config, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (torch.bfloat16
                      if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                      else torch.float32)

        self._seed_everything(cfg.data.seed)
        self.model = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer()
        self.train_ds = ShardedTokenDataset(
            cfg.data.train_shards_glob, cfg.train.block_size, cfg.data.seed)
        self.val_ds = ShardedTokenDataset(
            cfg.data.val_shards_glob, cfg.train.block_size, cfg.data.seed + 1)
        self.step = 0
        self.best_val_loss = float("inf")

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self) -> GPTModel:
        m = self.cfg.model
        return GPTModel(
            vocab_size=m.vocab_size,
            n_embd=m.n_embd,
            n_head=m.n_head,
            n_kv_head=m.n_kv_head,
            n_layer=m.n_layer,
            max_seq_len=m.max_seq_len,
            grad_checkpoint=m.grad_checkpoint,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Optimizer selection. Mixed precision and 8-bit added in later tasks."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.cfg.train.weight_decay,
        )

    def _get_lr(self, it: int) -> float:
        cfg = self.cfg.train
        if it < cfg.warmup_iters:
            return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
        if it >= cfg.max_iters:
            return cfg.min_lr
        decay_ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def train_step(self) -> float:
        """Run one optimizer step (with gradient accumulation). Returns mean loss."""
        cfg = self.cfg.train
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = self.train_ds.get_batch(cfg.batch_size)
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            total_loss += loss.item() * cfg.grad_accum_steps
        lr = self._get_lr(self.step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.step += 1
        return total_loss / cfg.grad_accum_steps

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        cfg = self.cfg.train
        self.model.eval()
        out = {}
        for split, ds in [("train", self.train_ds), ("val", self.val_ds)]:
            losses = []
            for _ in range(cfg.eval_iters):
                x, y = ds.get_batch(cfg.batch_size)
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
        return out

    def fit(self) -> None:
        """Main loop. Checkpoint/resume and wandb added in Tasks 10-12."""
        while self.step < self.cfg.train.max_iters:
            loss = self.train_step()
            if self.step % self.cfg.train.eval_interval == 0:
                evals = self.evaluate()
                print(f"[{self.step}] train_loss={loss:.4f} "
                      f"val_loss={evals['val']:.4f}")
