"""Tests for model.py architecture changes."""
import math
import torch
import torch.nn.functional as F
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
