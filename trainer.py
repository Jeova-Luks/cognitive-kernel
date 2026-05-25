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
        """Dispatch on cfg.train.optimizer: adamw | adamw_8bit | muon."""
        cfg = self.cfg.train
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adamw_8bit":
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=cfg.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "muon":
            # Implemented in a later task; not yet wired in.
            raise NotImplementedError(
                "Muon optimizer is opt-in and not yet implemented in Phase 0. "
                "Set optimizer: adamw_8bit for now."
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")

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
        # BF16 autocast on supported GPUs; no-op on CPU/FP32 (enabled=False)
        amp_enabled = self.dtype != torch.float32
        for _ in range(cfg.grad_accum_steps):
            x, y = self.train_ds.get_batch(cfg.batch_size)
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(device_type=self.device, dtype=self.dtype,
                                    enabled=amp_enabled):
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
        amp_enabled = self.dtype != torch.float32
        out = {}
        for split, ds in [("train", self.train_ds), ("val", self.val_ds)]:
            losses = []
            for _ in range(cfg.eval_iters):
                x, y = ds.get_batch(cfg.batch_size)
                x, y = x.to(self.device), y.to(self.device)
                with torch.amp.autocast(device_type=self.device, dtype=self.dtype,
                                        enabled=amp_enabled):
                    _, loss = self.model(x, y)
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
        return out

    def load_checkpoint(self, path: Path) -> None:
        """Restore complete state from a checkpoint (bit-identical resume)."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.step = payload["step"]
        self.best_val_loss = payload["best_val_loss"]
        random.setstate(payload["rng_python"])
        np.random.set_state(payload["rng_numpy"])
        torch.set_rng_state(payload["rng_torch"])
        if payload["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
        self.train_ds.load_state_dict(payload["train_ds_state"])
        self.val_ds.load_state_dict(payload["val_ds_state"])

    def _prune_old_checkpoints(self, keep: int = 3) -> None:
        """Keep only the `keep` most recent ckpt_step_*.pt files."""
        ckpts = sorted(self.output_dir.glob("ckpt_step_*.pt"))
        for old in ckpts[:-keep]:
            old.unlink()

    def save_checkpoint(self) -> Path:
        """Save a complete, resumable checkpoint.

        Captures: weights, optimizer state, step counter, best val loss,
        Python/NumPy/Torch/CUDA RNG states, dataset RNG states, and full Config.
        Writes ckpt_step_NNNNNNN.pt and a latest.txt pointer for resume detection.
        """
        ckpt_path = self.output_dir / f"ckpt_step_{self.step:07d}.pt"
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            "train_ds_state": self.train_ds.state_dict(),
            "val_ds_state": self.val_ds.state_dict(),
            "config": asdict(self.cfg),
        }
        torch.save(payload, ckpt_path)
        # Pointer to "latest" for resume detection in the next session
        latest = self.output_dir / "latest.txt"
        latest.write_text(ckpt_path.name)
        return ckpt_path

    def fit(self) -> None:
        """Main loop with auto-resume from latest checkpoint. Wandb added in Task 12."""
        from resume import find_latest
        latest = find_latest(self.output_dir)
        if latest is not None:
            print(f"Resuming from {latest}")
            self.load_checkpoint(latest)

        while self.step < self.cfg.train.max_iters:
            loss = self.train_step()
            if self.step % self.cfg.train.eval_interval == 0:
                evals = self.evaluate()
                print(f"[{self.step}] train_loss={loss:.4f} "
                      f"val_loss={evals['val']:.4f}")
                if evals["val"] < self.best_val_loss:
                    self.best_val_loss = evals["val"]
            if self.step % self.cfg.train.checkpoint_interval == 0:
                self.save_checkpoint()
                self._prune_old_checkpoints(keep=3)
