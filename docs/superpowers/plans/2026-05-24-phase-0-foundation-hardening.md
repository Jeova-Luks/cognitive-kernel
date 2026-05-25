# Phase 0 — Foundation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing trainer survive Colab Free's session disconnects, enable training a ~100M-parameter model on a single T4 GPU via mixed precision + gradient checkpointing + 8-bit AdamW, and prove bit-identical checkpoint/resume across simulated session breaks.

**Architecture:** Extend the existing `model.py`, `trainer.py`, and `tokenizer.py` (do not rewrite). Add three new modules: streaming `dataset.py` for Drive-hosted shards, `optim_muon.py` (optional), and `resume.py` for cross-session state detection. Tests run in any Python environment with PyTorch (Colab, Codespaces, or local CPU); the bit-identical resume test uses CPU and a 2 M-parameter toy model so it's reproducible anywhere.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, `torch.amp`, `torch.utils.checkpoint`, `bitsandbytes` (8-bit AdamW), `wandb`, `PyYAML`, `pytest`, `numpy`.

**Spec reference:** [docs/superpowers/specs/2026-05-24-cognitive-kernel-v0.1-design.md](../specs/2026-05-24-cognitive-kernel-v0.1-design.md) — Phase 0 in §9.

---

## File Structure (Phase 0)

| File | Status | Responsibility |
|---|---|---|
| `model.py` | modify | Flash Attention + GQA + gradient-checkpointing hook + new special tokens |
| `tokenizer.py` | modify | Reserve special token IDs for CDSL boundary markers |
| `trainer.py` | refactor | Mixed precision, gradient accumulation, 8-bit AdamW, checkpoint save/load, wandb |
| `dataset.py` | **create** | Streaming `DataLoader` reading `.bin` shards from Google Drive |
| `resume.py` | **create** | Detect latest checkpoint in a directory and restore full state |
| `optim_muon.py` | **create** | Muon optimizer for 2D matrices (opt-in via config) |
| `configs/base_100m.yaml` | **create** | Hyperparameters for the 100M base model |
| `configs/test_toy.yaml` | **create** | Tiny config for unit tests (~2M params, runs on CPU) |
| `scripts/train_colab.ipynb` | **create** | Notebook that mounts Drive, loads config, runs trainer with resume |
| `tests/test_model.py` | **create** | Flash attention parity, GQA correctness, grad checkpoint equivalence |
| `tests/test_trainer.py` | **create** | Mixed precision, gradient accumulation, 8-bit AdamW smoke tests |
| `tests/test_resume.py` | **create** | The headline test: train N steps → kill → resume → identical loss |
| `tests/test_dataset.py` | **create** | Streaming loader determinism, shard boundary correctness |
| `requirements.txt` | modify | Add `bitsandbytes`, `wandb`, `pyyaml`, `pytest` |
| `.gitignore` | **create** | Ignore `__pycache__/`, `*.bin`, `checkpoints/`, `results/`, `wandb/` |
| `pytest.ini` | **create** | Pytest config (testpaths, markers) |

---

## Task 0: Initialize repository and dev environment

**Files:**
- Create: `.gitignore`
- Create: `pytest.ini`
- Modify: `requirements.txt`

This is the only task without TDD steps because it sets up the environment in which tests will run.

- [ ] **Step 0.1: Decide where Python runs**

Local C: drive has ~228 MB free — not enough for PyTorch (~500 MB installed). Pick one option and proceed:
- **Option A (recommended): GitHub Codespaces.** 60 free hours/month, full Linux + VS Code in browser, pre-installed Python. After Step 0.2 pushes the repo, create a Codespace from it.
- **Option B: Google Colab.** Run all tests inside Colab notebooks (slower iteration; mount the repo via `git clone` per session).
- **Option C: External drive.** If you have a D: or USB drive with space, install Python there and use `pip install --target=D:\python-libs ...`.

The remaining steps assume `pytest` and `python` are runnable somewhere with the right packages installed.

- [ ] **Step 0.2: Init git repo and make first commit of existing code**

Run:
```bash
cd c:/Users/jeude/.gemini/antigravity/scratch/LLMPessoal
git init
git add model.py tokenizer.py trainer.py server.py run.py requirements.txt static/ docs/
git commit -m "chore: import existing LLMPessoal codebase as Phase 0 starting point"
```

Expected: `[main (root-commit) <hash>] chore: import existing LLMPessoal codebase as Phase 0 starting point`

- [ ] **Step 0.3: Create `.gitignore`**

Write file `.gitignore`:
```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
*.bin
*.pt
checkpoints/
results/
wandb/
.env
.venv/
venv/
*.egg-info/
.ipynb_checkpoints/
```

- [ ] **Step 0.4: Create `pytest.ini`**

Write file `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests that require a CUDA GPU
addopts = -v --tb=short
```

- [ ] **Step 0.5: Update `requirements.txt`**

Replace the contents of `requirements.txt` (currently `torch / fastapi / uvicorn / websockets`) with:
```
# Core
torch>=2.1.0
numpy>=1.24.0

# Training infrastructure
bitsandbytes>=0.41.0
wandb>=0.16.0
pyyaml>=6.0.0

# Existing web server
fastapi
uvicorn
websockets

# Testing
pytest>=7.4.0
```

- [ ] **Step 0.6: Install and verify**

Run (in your chosen environment):
```bash
pip install -r requirements.txt
python -c "import torch, bitsandbytes, wandb, yaml, pytest; print('ok')"
```

Expected: `ok` (and no ImportError).

- [ ] **Step 0.7: Commit**

```bash
git add .gitignore pytest.ini requirements.txt
git commit -m "chore: add dev environment scaffolding (gitignore, pytest config, deps)"
```

---

## Task 1: Add Flash Attention to `model.py`

Replace the manual softmax-attention block in [model.py:112-125](../../../model.py#L112-L125) with `F.scaled_dot_product_attention`. Mathematically identical, ~40% VRAM savings on supported GPUs, falls back to manual math on CPU.

**Files:**
- Modify: `model.py:67-125` (the `CausalSelfAttention` class)
- Create: `tests/test_model.py`

- [ ] **Step 1.1: Write the failing test (numerical parity)**

Create `tests/test_model.py`:
```python
"""Tests for model.py architecture changes."""
import torch
import pytest
from model import CausalSelfAttention


def test_flash_attention_parity_with_manual():
    """SDPA-based attention must produce numerically identical output to the
    manual implementation, given the same weights and inputs."""
    torch.manual_seed(42)
    B, T, n_embd, n_head = 2, 16, 64, 4
    attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, max_seq_len=32)
    attn.eval()
    x = torch.randn(B, T, n_embd)

    # Reference: manual computation re-done locally (independent of model.py)
    import math
    import torch.nn.functional as F
    q = attn.q_proj(x).view(B, T, n_head, n_embd // n_head).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, n_head, n_embd // n_head).transpose(1, 2)
    v = attn.v_proj(x).view(B, T, n_head, n_embd // n_head).transpose(1, 2)
    q = attn.rope(q, T)
    k = attn.rope(k, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(n_embd // n_head))
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    att = att.masked_fill(mask == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    expected = (att @ v).transpose(1, 2).contiguous().view(B, T, n_embd)
    expected = attn.out_proj(expected)

    # Model's forward (post-change uses SDPA)
    with torch.no_grad():
        actual = attn(x)

    assert torch.allclose(actual, expected, atol=1e-5), \
        f"Max diff: {(actual - expected).abs().max().item()}"
```

- [ ] **Step 1.2: Run test, verify it fails as written**

Run: `pytest tests/test_model.py::test_flash_attention_parity_with_manual -v`

Expected: PASS (the existing implementation IS the manual implementation, so the test passes against it). This is the baseline — we want the test to still pass *after* swapping to SDPA. Record this as the regression-prevention test.

- [ ] **Step 1.3: Modify `CausalSelfAttention.forward` to use SDPA**

In `model.py`, replace lines 98-125 (the entire `forward` method of `CausalSelfAttention`) with:

```python
    def forward(self, x):
        B, T, C = x.size()

        # Q, K, V projections; reshape to [B, n_head, T, head_dim]
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Rotary position embedding on Q and K
        q = self.rope(q, T)
        k = self.rope(k, T)

        # Flash Attention 2 when available; falls back to math on CPU.
        # is_causal=True applies the causal mask internally (no allocation).
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        # Concatenate heads and project out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)
```

The pre-computed `self.bias` causal mask buffer (registered in `__init__` at line 92-96) is no longer needed but leave it for now to keep the diff focused; remove in a later cleanup pass.

- [ ] **Step 1.4: Run test, verify it still passes**

Run: `pytest tests/test_model.py::test_flash_attention_parity_with_manual -v`

Expected: PASS with `Max diff` < 1e-5. If it fails with a larger diff, investigate (most likely cause: RoPE dimension ordering doesn't match SDPA's expectations on your PyTorch version).

- [ ] **Step 1.5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): use scaled_dot_product_attention for ~40% VRAM savings"
```

---

## Task 2: Add Grouped-Query Attention (GQA) support to `model.py`

Allow `n_kv_head < n_head` so K/V projections are smaller. Reduces KV cache memory in inference (helps the verifier loop's many rollouts in later phases).

**Files:**
- Modify: `model.py` (CausalSelfAttention.__init__ and forward, plus GPTModel.__init__)
- Modify: `tests/test_model.py` (add GQA test)

- [ ] **Step 2.1: Write the failing test**

Append to `tests/test_model.py`:
```python
def test_gqa_param_count_reduction():
    """With n_kv_head=2 and n_head=8, K and V projections should be 1/4
    the size of their MHA counterparts. Total CausalSelfAttention params
    should drop accordingly."""
    n_embd = 64
    mha = CausalSelfAttention(n_embd=n_embd, n_head=8, max_seq_len=16, n_kv_head=8)
    gqa = CausalSelfAttention(n_embd=n_embd, n_head=8, max_seq_len=16, n_kv_head=2)

    mha_params = sum(p.numel() for p in mha.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())

    # K and V projections shrink from n_embd*n_embd to n_embd*(n_embd*n_kv_head/n_head)
    # = n_embd*n_embd*(2/8) each; that's a 6/8 reduction across K+V (the two of four).
    # Total reduction = 2*(n_embd^2)*(1 - 2/8) = 6144 for n_embd=64.
    expected_diff = 2 * (n_embd * n_embd) * (1 - 2/8)
    assert (mha_params - gqa_params) == expected_diff, \
        f"Expected diff {expected_diff}, got {mha_params - gqa_params}"


def test_gqa_forward_shape():
    """GQA forward must produce the same output shape as MHA."""
    B, T, n_embd = 2, 16, 64
    gqa = CausalSelfAttention(n_embd=n_embd, n_head=8, max_seq_len=32, n_kv_head=2)
    x = torch.randn(B, T, n_embd)
    with torch.no_grad():
        y = gqa(x)
    assert y.shape == (B, T, n_embd), f"Expected {(B, T, n_embd)}, got {y.shape}"
```

- [ ] **Step 2.2: Run tests, verify both fail**

Run: `pytest tests/test_model.py -v -k gqa`

Expected: FAIL (`__init__` doesn't accept `n_kv_head`).

- [ ] **Step 2.3: Modify `CausalSelfAttention.__init__` to accept `n_kv_head`**

In `model.py`, replace the `__init__` of `CausalSelfAttention` (around line 72-96) with:

```python
    def __init__(self, n_embd, n_head, max_seq_len, n_kv_head=None):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        assert n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.head_dim = n_embd // n_head
        self.max_seq_len = max_seq_len
        self.kv_dim = self.head_dim * self.n_kv_head  # smaller K/V projection

        # Q is full-size; K and V are reduced when n_kv_head < n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
```

(Remove the `self.register_buffer("bias", ...)` line — the SDPA call handles causal masking.)

- [ ] **Step 2.4: Modify `CausalSelfAttention.forward` to repeat K/V across query groups**

Replace the forward (modified in Task 1) with:

```python
    def forward(self, x):
        B, T, C = x.size()

        # Q is [B, n_head, T, head_dim]; K/V are [B, n_kv_head, T, head_dim]
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self.rope(q, T)
        k = self.rope(k, T)

        # Expand K/V so each query group sees the right K/V head
        if self.n_kv_head < self.n_head:
            n_repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)
```

- [ ] **Step 2.5: Run all model tests, verify they pass**

Run: `pytest tests/test_model.py -v`

Expected: 3 PASS (`test_flash_attention_parity_with_manual`, `test_gqa_param_count_reduction`, `test_gqa_forward_shape`).

- [ ] **Step 2.6: Propagate `n_kv_head` through `GPTModel`**

In `model.py`, modify `GPTModel.__init__` signature and the `TransformerBlock` instantiation. Replace:

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, max_seq_len=128):
```

with:

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_kv_head=None,
                 n_layer=6, max_seq_len=128):
```

And in the same `__init__`, change:

```python
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, max_seq_len) for _ in range(n_layer)
        ])
```

to:

```python
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, max_seq_len, n_kv_head=n_kv_head)
            for _ in range(n_layer)
        ])
```

Also modify `TransformerBlock.__init__` (around line 157-163) signature:

```python
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, max_seq_len, n_kv_head=None):
        super().__init__()
        self.attn_norm = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, n_kv_head=n_kv_head)
        self.mlp_norm = RMSNorm(n_embd)
        self.mlp = SwiGLUMLP(n_embd)
```

- [ ] **Step 2.7: Run the in-file self-test of model.py to ensure backward compatibility**

Run: `python model.py`

Expected: prints `Modelo LLM Pessoal criado com sucesso! Parâmetros totais: ...` and `Logits shape: torch.Size([4, 32, 256])` and a generation example. No exceptions. (The default `n_kv_head=None` falls back to MHA, preserving old behavior.)

- [ ] **Step 2.8: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): support Grouped-Query Attention (n_kv_head parameter)"
```

---

## Task 3: Add gradient-checkpointing hook to `model.py`

Wrap each `TransformerBlock`'s forward in `torch.utils.checkpoint.checkpoint` when `grad_checkpoint=True`. Trades ~30% compute for ~60% activation memory savings — required for fitting the 100M model in a T4's 16 GB.

**Files:**
- Modify: `model.py` (`GPTModel.__init__`, `GPTModel.forward`)
- Modify: `tests/test_model.py`

- [ ] **Step 3.1: Write the failing test**

Append to `tests/test_model.py`:
```python
def test_gradient_checkpointing_loss_equivalence():
    """Forward with and without grad checkpointing must produce identical loss."""
    from model import GPTModel
    torch.manual_seed(123)
    V, B, T = 100, 2, 16
    cfg = dict(vocab_size=V, n_embd=32, n_head=4, n_kv_head=2,
               n_layer=3, max_seq_len=32)

    model_no_ckpt = GPTModel(**cfg, grad_checkpoint=False)
    model_ckpt = GPTModel(**cfg, grad_checkpoint=True)
    # Copy weights so they're identical
    model_ckpt.load_state_dict(model_no_ckpt.state_dict())

    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))

    _, loss_no_ckpt = model_no_ckpt(x, y)
    _, loss_ckpt = model_ckpt(x, y)

    assert torch.allclose(loss_no_ckpt, loss_ckpt, atol=1e-5), \
        f"Loss differs: no_ckpt={loss_no_ckpt.item()}, ckpt={loss_ckpt.item()}"
```

- [ ] **Step 3.2: Run test, verify it fails**

Run: `pytest tests/test_model.py::test_gradient_checkpointing_loss_equivalence -v`

Expected: FAIL (`GPTModel.__init__` does not accept `grad_checkpoint`).

- [ ] **Step 3.3: Add `grad_checkpoint` flag to `GPTModel`**

In `model.py`, modify `GPTModel.__init__` to accept the flag and store it:

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_kv_head=None,
                 n_layer=6, max_seq_len=128, grad_checkpoint=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.grad_checkpoint = grad_checkpoint
        # ... rest unchanged ...
```

Modify `GPTModel.forward` (around line 218-238) to use checkpointing conditionally:

```python
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.max_seq_len

        x = self.token_embeddings(idx)

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
```

- [ ] **Step 3.4: Run test, verify it passes**

Run: `pytest tests/test_model.py::test_gradient_checkpointing_loss_equivalence -v`

Expected: PASS.

- [ ] **Step 3.5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): add gradient checkpointing hook (~60% activation memory savings)"
```

---

## Task 4: Reserve CDSL special tokens in `tokenizer.py`

The base tokenizer currently produces token IDs 0–255 from raw bytes plus BPE merges starting at 256. Reserve a fixed band of token IDs for CDSL boundary and tool-call markers so the model can emit them as single tokens later.

**Files:**
- Modify: `tokenizer.py` (`BPETokenizer.__init__`, add `SPECIAL_TOKENS`)
- Create: `tests/test_tokenizer.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_tokenizer.py`:
```python
"""Tests for tokenizer.py."""
from tokenizer import BPETokenizer, SPECIAL_TOKENS


def test_special_tokens_registered():
    """All required special tokens have stable IDs in the 60000-block."""
    expected = ["<|endoftext|>", "<|cdsl_start|>", "<|cdsl_end|>",
                "<|tool_call|>", "<|tool_result|>"]
    for tok in expected:
        assert tok in SPECIAL_TOKENS
        assert 60000 <= SPECIAL_TOKENS[tok] < 60100


def test_special_token_ids_are_unique():
    ids = list(SPECIAL_TOKENS.values())
    assert len(ids) == len(set(ids))


def test_special_tokens_present_in_vocab_after_init():
    tok = BPETokenizer(vocab_size=300)
    for special, special_id in SPECIAL_TOKENS.items():
        assert special_id in tok.vocab, f"{special} (id {special_id}) missing from vocab"
        assert tok.vocab[special_id] == special.encode("utf-8")
```

- [ ] **Step 4.2: Run test, verify it fails**

Run: `pytest tests/test_tokenizer.py -v`

Expected: FAIL (`SPECIAL_TOKENS` not importable from tokenizer).

- [ ] **Step 4.3: Add `SPECIAL_TOKENS` constant and register them in `__init__`**

In `tokenizer.py`, at the top of the file (just after the existing imports), add:

```python
SPECIAL_TOKENS = {
    "<|endoftext|>":   60000,
    "<|cdsl_start|>":  60001,
    "<|cdsl_end|>":    60002,
    "<|tool_call|>":   60003,
    "<|tool_result|>": 60004,
}
```

Then modify `BPETokenizer.__init__` (around line 12-17) to register them:

```python
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        for special, special_id in SPECIAL_TOKENS.items():
            self.vocab[special_id] = special.encode("utf-8")
        self.merges = {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
```

- [ ] **Step 4.4: Run test, verify it passes**

Run: `pytest tests/test_tokenizer.py -v`

Expected: 3 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add tokenizer.py tests/test_tokenizer.py
git commit -m "feat(tokenizer): reserve special-token band 60000-60099 for CDSL markers"
```

---

## Task 5: Create YAML config loader and configs

Move all hyperparameters out of trainer call sites into versioned YAML files. The trainer in Task 8+ will load configs via this loader.

**Files:**
- Create: `config.py`
- Create: `configs/base_100m.yaml`
- Create: `configs/test_toy.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_config.py`:
```python
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
    with pytest.raises(KeyError):
        load_config(Path("configs/test_toy.yaml")).model.nonexistent_field
```

- [ ] **Step 5.2: Run tests, verify they fail**

Run: `pytest tests/test_config.py -v`

Expected: FAIL (`config` module doesn't exist).

- [ ] **Step 5.3: Create `config.py`**

Write file `config.py`:
```python
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
    """Load a YAML config into nested dataclasses. Strict — missing required fields raise."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        data=DataConfig(**raw["data"]),
        log=LogConfig(**raw.get("log", {})),
    )
```

- [ ] **Step 5.4: Create `configs/base_100m.yaml`**

```bash
mkdir -p configs
```

Write file `configs/base_100m.yaml`:
```yaml
model:
  n_embd: 768
  n_head: 12
  n_kv_head: 4
  n_layer: 12
  max_seq_len: 2048
  vocab_size: 32000
  grad_checkpoint: true

train:
  batch_size: 16
  grad_accum_steps: 16     # effective batch = 256
  block_size: 2048
  max_iters: 20000
  learning_rate: 3.0e-4
  warmup_iters: 200
  min_lr: 3.0e-5
  weight_decay: 0.1
  grad_clip: 1.0
  eval_interval: 500
  eval_iters: 50
  checkpoint_interval: 500
  optimizer: adamw_8bit

data:
  train_shards_glob: "data/shards/train_*.bin"
  val_shards_glob: "data/shards/val_*.bin"
  seed: 42

log:
  project: cognitive-kernel
  run_name: base_100m
  log_interval: 10
  wandb_mode: online
```

- [ ] **Step 5.5: Create `configs/test_toy.yaml`**

Write file `configs/test_toy.yaml`:
```yaml
model:
  n_embd: 32
  n_head: 4
  n_kv_head: 2
  n_layer: 2
  max_seq_len: 64
  vocab_size: 256
  grad_checkpoint: false

train:
  batch_size: 4
  grad_accum_steps: 2
  block_size: 32
  max_iters: 50
  learning_rate: 1.0e-3
  warmup_iters: 5
  min_lr: 1.0e-4
  weight_decay: 0.0
  grad_clip: 1.0
  eval_interval: 10
  eval_iters: 5
  checkpoint_interval: 25
  optimizer: adamw

data:
  train_shards_glob: "tests/fixtures/toy_train_*.bin"
  val_shards_glob: "tests/fixtures/toy_val_*.bin"
  seed: 0

log:
  project: cognitive-kernel
  run_name: toy_test
  log_interval: 5
  wandb_mode: disabled
```

- [ ] **Step 5.6: Run tests, verify they pass**

Run: `pytest tests/test_config.py -v`

Expected: 3 PASS.

- [ ] **Step 5.7: Commit**

```bash
git add config.py configs/ tests/test_config.py
git commit -m "feat(config): YAML-backed configuration for trainer and model"
```

---

## Task 6: Create streaming `dataset.py`

Reads pre-tokenized `.bin` shards (uint16, 2 bytes/token — nanoGPT format) from a glob, yields random training batches with deterministic seeding. Designed for Drive: streams sequentially in large chunks, never does small random reads against Drive.

**Files:**
- Create: `dataset.py`
- Create: `tests/test_dataset.py`
- Create: `tests/fixtures/` (small `.bin` fixtures generated by a helper test)

- [ ] **Step 6.1: Write the failing test**

Create `tests/test_dataset.py`:
```python
"""Tests for dataset.py streaming loader."""
import os
import numpy as np
import pytest
from pathlib import Path
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
    arr_val.tofile(fixtures_dir / f"toy_val_000.bin")
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
    # y is x shifted by 1 — verify with a known seed
    assert x.dtype == np.dtype("int64") or hasattr(x, "long")  # torch tensor


def test_dataset_deterministic_given_seed(toy_shards):
    ds_a = ShardedTokenDataset(
        glob_pattern=str(toy_shards / "toy_train_*.bin"),
        block_size=16, seed=42)
    ds_b = ShardedTokenDataset(
        glob_pattern=str(toy_shards / "toy_train_*.bin"),
        block_size=16, seed=42)
    xa, ya = ds_a.get_batch(batch_size=4)
    xb, yb = ds_b.get_batch(batch_size=4)
    import torch
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)


def test_dataset_different_seeds_differ(toy_shards):
    import torch
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
```

- [ ] **Step 6.2: Run tests, verify they fail**

Run: `pytest tests/test_dataset.py -v`

Expected: FAIL (`dataset` module does not exist).

- [ ] **Step 6.3: Create `dataset.py`**

Write file `dataset.py`:
```python
"""Streaming token dataset for pre-tokenized .bin shards (uint16 nanoGPT format)."""
import glob as glob_module
from pathlib import Path
from typing import Tuple
import numpy as np
import torch


class ShardedTokenDataset:
    """Reads .bin shards (uint16 tokens) and yields random (x, y) batches.

    Designed for Google-Drive-hosted data:
    - Uses np.memmap so the OS handles paging from disk; no small random reads.
    - Random offsets are within a single mmap'd shard chosen per batch.
    """

    def __init__(self, glob_pattern: str, block_size: int, seed: int):
        self.glob_pattern = glob_pattern
        self.block_size = block_size
        self.shard_paths = sorted(glob_module.glob(glob_pattern))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards matched: {glob_pattern}")
        self.rng = np.random.default_rng(seed)
        # Lazy-open shards
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
```

- [ ] **Step 6.4: Run tests, verify all pass**

Run: `pytest tests/test_dataset.py -v`

Expected: 4 PASS.

- [ ] **Step 6.5: Commit**

```bash
git add dataset.py tests/test_dataset.py
git commit -m "feat(dataset): streaming sharded token dataset with deterministic RNG"
```

---

## Task 7: Refactor `trainer.py` to use the new model/config/dataset

The old trainer hardcodes things and reads a text file. Replace its data path with `ShardedTokenDataset`, take a `Config` instead of kwargs, and prepare it for the mixed-precision / 8-bit-optimizer additions in Tasks 8 and 9. **No test gating in this task** — it's a structural refactor; the tests in Tasks 8-12 will cover the new behaviors.

**Files:**
- Modify: `trainer.py`

- [ ] **Step 7.1: Move the existing `trainer.py` out of the way as a reference**

```bash
mv trainer.py trainer_legacy.py
```

This is kept temporarily for cross-referencing; deleted in Task 14 once all behaviors are reproduced.

- [ ] **Step 7.2: Create the new `trainer.py` skeleton**

Write file `trainer.py`:
```python
"""Cognitive Kernel trainer. Loads a Config, builds model + optimizer + dataset,
runs the train/eval/checkpoint loop. Designed to survive Colab disconnects."""
from __future__ import annotations
import os
import math
import time
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
        self.dtype = (torch.bfloat16 if torch.cuda.is_available()
                      and torch.cuda.is_bf16_supported() else torch.float32)

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
```

- [ ] **Step 7.3: Smoke-test against the toy config (no real shards yet)**

Create a tiny fixture generator. Add to `tests/test_trainer.py` (create file):

```python
"""Tests for trainer.py."""
import os
import numpy as np
import pytest
import torch
from pathlib import Path
from config import load_config
from trainer import Trainer


@pytest.fixture
def toy_shards(tmp_path):
    """Generate fixtures that match configs/test_toy.yaml's glob."""
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
```

Run: `pytest tests/test_trainer.py::test_trainer_runs_two_steps_on_toy -v`

Expected: PASS. Loss values printed are not checked for magnitude — just that the loop runs and completes two steps.

- [ ] **Step 7.4: Commit**

```bash
git add trainer.py trainer_legacy.py tests/test_trainer.py
git commit -m "refactor(trainer): rebuild trainer on Config + ShardedTokenDataset; legacy kept temporarily"
```

---

## Task 8: Add BF16 mixed precision to `trainer.py`

Wrap forward/backward in `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`. BF16 (not FP16) — no GradScaler needed; BF16 has the same exponent range as FP32 and is stable on Ampere+ GPUs (T4 has it via emulation, A100 native).

**Files:**
- Modify: `trainer.py` (`Trainer.train_step` and `Trainer.evaluate`)
- Modify: `tests/test_trainer.py`

- [ ] **Step 8.1: Write the failing test (no-op on CPU, real on GPU)**

Append to `tests/test_trainer.py`:
```python
def test_autocast_used_in_train_step(toy_shards, tmp_path, monkeypatch):
    """Verify torch.amp.autocast is entered during train_step (instrumented)."""
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
```

- [ ] **Step 8.2: Run test, verify it fails**

Run: `pytest tests/test_trainer.py::test_autocast_used_in_train_step -v`

Expected: FAIL (autocast not called yet).

- [ ] **Step 8.3: Wrap forward in autocast**

In `trainer.py`, modify `train_step`:

```python
    def train_step(self) -> float:
        cfg = self.cfg.train
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = self.train_ds.get_batch(cfg.batch_size)
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(device_type=self.device, dtype=self.dtype):
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
```

And modify `evaluate` similarly to wrap in autocast.

- [ ] **Step 8.4: Run test, verify it passes**

Run: `pytest tests/test_trainer.py::test_autocast_used_in_train_step -v`

Expected: PASS.

- [ ] **Step 8.5: Commit**

```bash
git add trainer.py tests/test_trainer.py
git commit -m "feat(trainer): wrap forward in torch.amp.autocast (BF16 on GPU)"
```

---

## Task 9: Add 8-bit AdamW via `bitsandbytes`

Optimizer states for AdamW are 2× model size in FP32 — for 100 M that's 800 MB just for optimizer. 8-bit AdamW shrinks this 4×.

**Files:**
- Modify: `trainer.py` (`_build_optimizer`)
- Modify: `tests/test_trainer.py`

- [ ] **Step 9.1: Write the failing test**

Append to `tests/test_trainer.py`:
```python
def test_optimizer_dispatch_adamw(tmp_path, toy_shards):
    cfg = load_config(Path("configs/test_toy.yaml"))
    cfg.train.optimizer = "adamw"
    trainer = Trainer(cfg, output_dir=tmp_path / "out")
    assert trainer.optimizer.__class__.__name__ == "AdamW"


def test_optimizer_dispatch_adamw_8bit(tmp_path, toy_shards):
    pytest.importorskip("bitsandbytes")
    cfg = load_config(Path("configs/test_toy.yaml"))
    cfg.train.optimizer = "adamw_8bit"
    trainer = Trainer(cfg, output_dir=tmp_path / "out")
    assert "8bit" in trainer.optimizer.__class__.__name__.lower() or \
           "8bit" in repr(trainer.optimizer).lower()


def test_optimizer_dispatch_unknown_raises(tmp_path, toy_shards):
    cfg = load_config(Path("configs/test_toy.yaml"))
    cfg.train.optimizer = "made_up"
    with pytest.raises(ValueError, match="Unknown optimizer"):
        Trainer(cfg, output_dir=tmp_path / "out")
```

- [ ] **Step 9.2: Run tests, verify they fail**

Run: `pytest tests/test_trainer.py -v -k optimizer_dispatch`

Expected: FAIL (current `_build_optimizer` always returns AdamW).

- [ ] **Step 9.3: Implement optimizer dispatch**

In `trainer.py`, replace `_build_optimizer`:

```python
    def _build_optimizer(self) -> torch.optim.Optimizer:
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
```

- [ ] **Step 9.4: Run tests, verify they pass**

Run: `pytest tests/test_trainer.py -v -k optimizer_dispatch`

Expected: 3 PASS.

- [ ] **Step 9.5: Commit**

```bash
git add trainer.py tests/test_trainer.py
git commit -m "feat(trainer): support adamw_8bit via bitsandbytes (saves ~3x optimizer memory)"
```

---

## Task 10: Add checkpoint **save** (full state)

A complete checkpoint must capture: model weights, optimizer state, step, best val loss, Python `random` state, numpy `random` state, torch global RNG state, CUDA RNG state, dataset RNG states, and the original Config. Without ALL of these, "resume" is not bit-identical.

**Files:**
- Create: `resume.py`
- Modify: `trainer.py` (add `save_checkpoint` method)

- [ ] **Step 10.1: Write the failing test**

Append to `tests/test_trainer.py`:
```python
def test_save_checkpoint_writes_complete_state(tmp_path, toy_shards):
    cfg = load_config(Path("configs/test_toy.yaml"))
    trainer = Trainer(cfg, output_dir=tmp_path / "out")
    trainer.train_step()
    trainer.train_step()

    ckpt_path = trainer.save_checkpoint()
    assert ckpt_path.exists()

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required = {"model_state", "optimizer_state", "step", "best_val_loss",
                "rng_python", "rng_numpy", "rng_torch", "rng_cuda",
                "train_ds_state", "val_ds_state", "config"}
    missing = required - set(payload.keys())
    assert not missing, f"Checkpoint missing keys: {missing}"
    assert payload["step"] == 2
```

- [ ] **Step 10.2: Run test, verify it fails**

Run: `pytest tests/test_trainer.py::test_save_checkpoint_writes_complete_state -v`

Expected: FAIL (`save_checkpoint` doesn't exist).

- [ ] **Step 10.3: Add `save_checkpoint` method to `Trainer`**

In `trainer.py`, add at the end of the `Trainer` class:

```python
    def save_checkpoint(self) -> Path:
        """Save a complete, resumable checkpoint."""
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
        # Also write a pointer to "latest" for resume detection
        latest = self.output_dir / "latest.txt"
        latest.write_text(ckpt_path.name)
        return ckpt_path
```

- [ ] **Step 10.4: Run test, verify it passes**

Run: `pytest tests/test_trainer.py::test_save_checkpoint_writes_complete_state -v`

Expected: PASS.

- [ ] **Step 10.5: Commit**

```bash
git add trainer.py tests/test_trainer.py
git commit -m "feat(trainer): save_checkpoint captures full state (model, opt, RNGs, ds)"
```

---

## Task 11: Add checkpoint **load** / resume

Build `resume.py` with `find_latest(dir)` and add `Trainer.load_checkpoint(path)` that restores every state captured in Task 10.

**Files:**
- Create: `resume.py`
- Modify: `trainer.py` (`Trainer.load_checkpoint`)
- Modify: `tests/test_trainer.py`

- [ ] **Step 11.1: Write the failing test (the headline test)**

Append to `tests/test_trainer.py`:
```python
def test_resume_is_bit_identical(tmp_path, toy_shards):
    """Train 5 steps; save; resume; train 5 more steps. Result must equal
    training 10 steps without interruption (bit-identical loss trajectory)."""
    cfg = load_config(Path("configs/test_toy.yaml"))
    cfg.train.max_iters = 10

    # Run A: 10 uninterrupted steps
    trainer_a = Trainer(cfg, output_dir=tmp_path / "a")
    losses_a = []
    for _ in range(10):
        losses_a.append(trainer_a.train_step())

    # Run B: 5 steps, checkpoint, NEW Trainer instance loads it, 5 more steps
    trainer_b = Trainer(cfg, output_dir=tmp_path / "b")
    losses_b = []
    for _ in range(5):
        losses_b.append(trainer_b.train_step())
    ckpt_path = trainer_b.save_checkpoint()
    del trainer_b

    trainer_c = Trainer(cfg, output_dir=tmp_path / "b")
    trainer_c.load_checkpoint(ckpt_path)
    for _ in range(5):
        losses_b.append(trainer_c.train_step())

    # Every loss should match to floating-point equality
    for i, (la, lb) in enumerate(zip(losses_a, losses_b)):
        assert abs(la - lb) < 1e-6, \
            f"Step {i}: uninterrupted={la}, resumed={lb}, diff={abs(la - lb)}"
```

- [ ] **Step 11.2: Run test, verify it fails**

Run: `pytest tests/test_trainer.py::test_resume_is_bit_identical -v`

Expected: FAIL (`load_checkpoint` doesn't exist).

- [ ] **Step 11.3: Create `resume.py`**

Write file `resume.py`:
```python
"""Detect the latest checkpoint in an output directory."""
from pathlib import Path
from typing import Optional


def find_latest(output_dir: Path) -> Optional[Path]:
    """Return path to the latest checkpoint in output_dir, or None if none exists.

    Prefers the explicit `latest.txt` pointer; falls back to the highest-step
    ckpt_step_*.pt file if the pointer is missing.
    """
    output_dir = Path(output_dir)
    pointer = output_dir / "latest.txt"
    if pointer.exists():
        name = pointer.read_text().strip()
        candidate = output_dir / name
        if candidate.exists():
            return candidate

    ckpts = sorted(output_dir.glob("ckpt_step_*.pt"))
    return ckpts[-1] if ckpts else None
```

- [ ] **Step 11.4: Add `load_checkpoint` to `Trainer`**

In `trainer.py`, add to the `Trainer` class:

```python
    def load_checkpoint(self, path: Path) -> None:
        """Restore complete state from a checkpoint."""
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
```

- [ ] **Step 11.5: Run the headline test, verify it passes**

Run: `pytest tests/test_trainer.py::test_resume_is_bit_identical -v`

Expected: PASS. **If this passes, the central definition-of-done for Phase 0 is met.**

- [ ] **Step 11.6: Add a small auto-resume helper to `Trainer.fit`**

In `trainer.py`, modify `fit` to auto-detect existing checkpoints on start:

```python
    def fit(self) -> None:
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

    def _prune_old_checkpoints(self, keep: int = 3) -> None:
        """Keep only the `keep` most recent ckpt_step_*.pt files."""
        ckpts = sorted(self.output_dir.glob("ckpt_step_*.pt"))
        for old in ckpts[:-keep]:
            old.unlink()
```

- [ ] **Step 11.7: Commit**

```bash
git add trainer.py resume.py tests/test_trainer.py
git commit -m "feat(trainer): bit-identical checkpoint resume across sessions"
```

---

## Task 12: Add Weights & Biases logging

Wandb is **opt-out** (default `online`); for tests, the toy config sets `wandb_mode: disabled`.

**Files:**
- Modify: `trainer.py`
- Modify: `tests/test_trainer.py`

- [ ] **Step 12.1: Write the failing test**

Append to `tests/test_trainer.py`:
```python
def test_wandb_disabled_mode_does_not_call_init(tmp_path, toy_shards, monkeypatch):
    """When cfg.log.wandb_mode == 'disabled', wandb.init must not be called."""
    import wandb
    init_calls = {"count": 0}
    monkeypatch.setattr(wandb, "init", lambda *a, **k: (
        init_calls.__setitem__("count", init_calls["count"] + 1), None)[1])

    cfg = load_config(Path("configs/test_toy.yaml"))
    assert cfg.log.wandb_mode == "disabled"
    trainer = Trainer(cfg, output_dir=tmp_path / "out")
    trainer.fit()  # max_iters=50 from toy config
    assert init_calls["count"] == 0
```

- [ ] **Step 12.2: Run test, verify it fails**

Run: `pytest tests/test_trainer.py::test_wandb_disabled_mode_does_not_call_init -v`

Expected: FAIL (Trainer is silent and never tried to log). The test "passes" trivially right now — we need a *different* failure mode to trigger development. Adjust:

Replace the test with one that confirms logging IS attempted in online mode:

```python
def test_wandb_logs_loss_metrics(tmp_path, toy_shards, monkeypatch):
    """When wandb_mode != 'disabled', trainer must call wandb.log with loss."""
    import wandb
    logged = []
    monkeypatch.setattr(wandb, "init", lambda *a, **k: None)
    monkeypatch.setattr(wandb, "log", lambda d, **k: logged.append(d))
    monkeypatch.setattr(wandb, "finish", lambda *a, **k: None)

    cfg = load_config(Path("configs/test_toy.yaml"))
    cfg.log.wandb_mode = "online"
    cfg.train.max_iters = 5
    trainer = Trainer(cfg, output_dir=tmp_path / "out")
    trainer.fit()
    assert any("train_loss" in d for d in logged), \
        f"No train_loss in logged dicts: {logged}"
```

Run: FAIL (no wandb integration in trainer).

- [ ] **Step 12.3: Add wandb integration**

In `trainer.py`, modify `fit` to initialize, log, and finish wandb:

```python
    def fit(self) -> None:
        import wandb
        from resume import find_latest

        latest = find_latest(self.output_dir)
        if latest is not None:
            print(f"Resuming from {latest}")
            self.load_checkpoint(latest)

        if self.cfg.log.wandb_mode != "disabled":
            wandb.init(
                project=self.cfg.log.project,
                name=self.cfg.log.run_name or None,
                config=asdict(self.cfg),
                mode=self.cfg.log.wandb_mode,
                resume="allow",
            )

        try:
            while self.step < self.cfg.train.max_iters:
                loss = self.train_step()

                if self.step % self.cfg.log.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    metrics = {"train_loss": loss, "lr": lr, "step": self.step}
                    if self.cfg.log.wandb_mode != "disabled":
                        wandb.log(metrics, step=self.step)

                if self.step % self.cfg.train.eval_interval == 0:
                    evals = self.evaluate()
                    print(f"[{self.step}] train_loss={loss:.4f} "
                          f"val_loss={evals['val']:.4f}")
                    if evals["val"] < self.best_val_loss:
                        self.best_val_loss = evals["val"]
                    if self.cfg.log.wandb_mode != "disabled":
                        wandb.log({"val_loss": evals["val"],
                                   "train_eval_loss": evals["train"]},
                                  step=self.step)

                if self.step % self.cfg.train.checkpoint_interval == 0:
                    self.save_checkpoint()
                    self._prune_old_checkpoints(keep=3)
        finally:
            if self.cfg.log.wandb_mode != "disabled":
                wandb.finish()
```

- [ ] **Step 12.4: Run test, verify it passes**

Run: `pytest tests/test_trainer.py::test_wandb_logs_loss_metrics -v`

Expected: PASS.

- [ ] **Step 12.5: Commit**

```bash
git add trainer.py tests/test_trainer.py
git commit -m "feat(trainer): integrate Weights & Biases logging (opt-out via wandb_mode)"
```

---

## Task 13: Create the Colab training notebook

The notebook is the actual entry point used by the user. It must: mount Drive, clone or pull this repo from GitHub, install requirements, login to wandb (optional), pick a config, and call `trainer.fit()`. Resume is automatic via `find_latest`.

**Files:**
- Create: `scripts/train_colab.ipynb`

- [ ] **Step 13.1: Generate the notebook**

This is generated via a tiny Python script so it's reproducible and reviewable. Create `scripts/_gen_notebook.py`:

```python
"""One-time script: generate scripts/train_colab.ipynb from cells defined below."""
import json
from pathlib import Path

CELLS = [
    ("md", [
        "# Cognitive Kernel v0.1 — Training Notebook\n",
        "\n",
        "Run this in Google Colab. Resumes automatically if a checkpoint exists in `OUTPUT_DIR`.\n"
    ]),
    ("code", [
        "# Cell 1: mount Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n"
    ]),
    ("code", [
        "# Cell 2: clone repo (or pull if already cloned)\n",
        "import os, subprocess\n",
        "REPO_DIR = '/content/cognitive-kernel'\n",
        "if not os.path.exists(REPO_DIR):\n",
        "    subprocess.run(['git', 'clone', 'https://github.com/YOUR_USERNAME/cognitive-kernel.git', REPO_DIR], check=True)\n",
        "else:\n",
        "    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)\n",
        "os.chdir(REPO_DIR)\n"
    ]),
    ("code", [
        "# Cell 3: install dependencies\n",
        "!pip install -q -r requirements.txt\n"
    ]),
    ("code", [
        "# Cell 4: wandb login (skip with empty key for offline mode)\n",
        "import wandb\n",
        "# wandb.login(key='YOUR_KEY')  # uncomment and paste\n"
    ]),
    ("code", [
        "# Cell 5: pick config and output dir\n",
        "from pathlib import Path\n",
        "from config import load_config\n",
        "from trainer import Trainer\n",
        "\n",
        "CONFIG = 'configs/base_100m.yaml'\n",
        "OUTPUT_DIR = Path('/content/drive/MyDrive/cognitive-kernel/checkpoints/base_100m')\n",
        "\n",
        "cfg = load_config(Path(CONFIG))\n",
        "print(cfg)\n"
    ]),
    ("code", [
        "# Cell 6: train (resumes automatically)\n",
        "trainer = Trainer(cfg, output_dir=OUTPUT_DIR)\n",
        "print(f'Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}')\n",
        "trainer.fit()\n"
    ]),
]

def make_cell(kind, source):
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": source
    }

nb = {
    "cells": [make_cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "train_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out}")
```

- [ ] **Step 13.2: Run the generator**

Run:
```bash
python scripts/_gen_notebook.py
```

Expected: `Wrote scripts/train_colab.ipynb`. Open it locally to sanity-check (or in Colab directly).

- [ ] **Step 13.3: Commit**

```bash
git add scripts/_gen_notebook.py scripts/train_colab.ipynb
git commit -m "feat(scripts): Colab training notebook with auto-resume from Drive"
```

---

## Task 14: Smoke-test the full Colab pipeline and clean up

This is the Phase 0 definition-of-done validation. The bit-identical test (Task 11) already proved correctness on CPU; this confirms the GPU pipeline.

**Files:**
- Delete: `trainer_legacy.py`
- Modify: `README.md` (or create if missing)

- [ ] **Step 14.1: Generate a small set of training shards locally for smoke testing**

Create `scripts/make_smoke_shards.py`:
```python
"""Generate tiny training shards for smoke-testing trainer in Colab."""
from pathlib import Path
import numpy as np

OUT = Path("data/shards")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for i in range(2):
    arr = rng.integers(0, 32000, size=1_000_000, dtype=np.uint16)
    arr.tofile(OUT / f"train_{i:03d}.bin")
arr_val = rng.integers(0, 32000, size=200_000, dtype=np.uint16)
arr_val.tofile(OUT / "val_000.bin")
print(f"Wrote 3 smoke shards to {OUT}")
```

Run: `python scripts/make_smoke_shards.py`. Verify files appear in `data/shards/`.

- [ ] **Step 14.2: Push the repo to GitHub**

```bash
# If you haven't yet, create an EMPTY repo on github.com named cognitive-kernel
git remote add origin https://github.com/YOUR_USERNAME/cognitive-kernel.git
git branch -M main
git push -u origin main
```

- [ ] **Step 14.3a: Create smoke config**

Write file `configs/smoke_100m.yaml`:
```yaml
model:
  n_embd: 768
  n_head: 12
  n_kv_head: 4
  n_layer: 12
  max_seq_len: 2048
  vocab_size: 32000
  grad_checkpoint: true

train:
  batch_size: 4
  grad_accum_steps: 4
  block_size: 2048
  max_iters: 100
  learning_rate: 3.0e-4
  warmup_iters: 10
  min_lr: 3.0e-5
  weight_decay: 0.1
  grad_clip: 1.0
  eval_interval: 25
  eval_iters: 5
  checkpoint_interval: 25
  optimizer: adamw_8bit

data:
  train_shards_glob: "data/shards/train_*.bin"
  val_shards_glob: "data/shards/val_*.bin"
  seed: 42

log:
  project: cognitive-kernel
  run_name: smoke_100m
  log_interval: 5
  wandb_mode: offline
```

Commit: `git add configs/smoke_100m.yaml && git commit -m "feat(configs): add smoke_100m config for end-to-end Colab validation"`

- [ ] **Step 14.3b: Open `scripts/train_colab.ipynb` in Colab**

In Colab:
1. Edit Cell 2's `YOUR_USERNAME` to your actual GitHub username.
2. In Cell 5, change `CONFIG` to `'configs/smoke_100m.yaml'`.
3. Run cells 1-5. Verify Drive mounts, repo clones, packages install, config loads.
4. Run cell 6. Verify training begins and the first checkpoint appears in `/content/drive/MyDrive/cognitive-kernel/checkpoints/smoke_100m/`.

- [ ] **Step 14.4: Force-disconnect and resume**

In Colab:
1. After ~50 steps complete and a checkpoint is saved, **manually disconnect** the runtime (Runtime → Disconnect and delete runtime).
2. Reconnect. Re-run cells 1-6.
3. Verify the log line `Resuming from ckpt_step_0000050.pt` appears.
4. Verify training continues to step 100 without error.

Pass criterion: smoke training completes 100 steps across one forced disconnect, loss is finite and decreasing on average.

- [ ] **Step 14.5: Delete `trainer_legacy.py`**

```bash
git rm trainer_legacy.py
git commit -m "chore: remove trainer_legacy.py after successful refactor"
```

- [ ] **Step 14.6: Write a brief README**

Replace or create `README.md`:
```markdown
# Cognitive Kernel — Phase 0

First subproject of a multi-year research program (see [design spec](docs/superpowers/specs/2026-05-24-cognitive-kernel-v0.1-design.md)).

Phase 0 status: Foundation hardened. Trainer survives Colab disconnects with bit-identical resume. Mixed precision + GQA + gradient checkpointing + 8-bit AdamW + wandb logging in place.

## Quickstart

1. Generate smoke shards locally: `python scripts/make_smoke_shards.py`
2. Push to GitHub.
3. Open `scripts/train_colab.ipynb` in Colab. Run all cells.

## Run tests

```bash
pip install -r requirements.txt
pytest -v
```

## Layout

See [docs/superpowers/specs/](docs/superpowers/specs/) for the design document.
See [docs/superpowers/plans/](docs/superpowers/plans/) for phase plans.
```

- [ ] **Step 14.7: Commit**

```bash
git add README.md
git commit -m "docs: add Phase 0 README pointing to spec + quickstart"
```

- [ ] **Step 14.8: Push final state**

```bash
git push
```

---

## Task 15: Run the full test suite end-to-end

Final gate.

- [ ] **Step 15.1: Run all tests locally (or in Codespaces)**

```bash
pytest -v
```

Expected: All tests in `tests/test_model.py`, `tests/test_tokenizer.py`, `tests/test_config.py`, `tests/test_dataset.py`, `tests/test_trainer.py` pass. **In particular, `test_resume_is_bit_identical` MUST pass — that's the Phase 0 definition of done.**

- [ ] **Step 15.2: Tag the Phase 0 release**

```bash
git tag -a phase-0-complete -m "Phase 0: Foundation Hardening complete"
git push --tags
```

---

## Phase 0 Definition of Done (recap)

When all 15 tasks are complete and all tests pass, you have:

1. ✅ Modern `model.py` with Flash Attention + GQA + gradient checkpointing
2. ✅ Tokenizer with CDSL special tokens reserved
3. ✅ YAML-backed configuration
4. ✅ Streaming Drive-friendly dataset loader
5. ✅ Trainer with BF16 mixed precision, gradient accumulation, 8-bit AdamW
6. ✅ Complete-state checkpoint save and bit-identical resume
7. ✅ Weights & Biases logging
8. ✅ Colab notebook entry point with auto-resume from Drive
9. ✅ Smoke-tested end-to-end on Colab with simulated disconnect
10. ✅ Foundation for Phase 1 (pre-train base 100 M) to plug into

**Next:** Phase 1 plan — pre-training the 100 M base model on the curated data mix. To be written when Phase 0 is complete.

---

**END OF PHASE 0 PLAN**
