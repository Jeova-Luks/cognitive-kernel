"""Tests for model.py architecture changes."""
import math
import torch
import torch.nn.functional as F
from model import CausalSelfAttention, GPTModel


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


def test_gqa_param_count_reduction():
    """With n_kv_head=2 and n_head=8, K and V projections should be 1/4 the
    size of their MHA counterparts. Total CausalSelfAttention params should
    drop accordingly."""
    n_embd = 64
    mha = CausalSelfAttention(n_embd=n_embd, n_head=8, max_seq_len=16, n_kv_head=8)
    gqa = CausalSelfAttention(n_embd=n_embd, n_head=8, max_seq_len=16, n_kv_head=2)

    mha_params = sum(p.numel() for p in mha.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())

    # K and V projections shrink from n_embd*n_embd to n_embd*(n_embd*n_kv_head/n_head)
    # = n_embd*n_embd*(2/8) each; reduction across K+V = 2*(n_embd^2)*(1 - 2/8) = 6144.
    expected_diff = 2 * (n_embd * n_embd) * (1 - 2 / 8)
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


def test_gradient_checkpointing_loss_equivalence():
    """Forward with and without grad checkpointing must produce identical loss."""
    torch.manual_seed(123)
    V, B, T = 100, 2, 16
    cfg = dict(vocab_size=V, n_embd=32, n_head=4, n_kv_head=2,
               n_layer=3, max_seq_len=32)

    model_no_ckpt = GPTModel(**cfg, grad_checkpoint=False)
    model_ckpt = GPTModel(**cfg, grad_checkpoint=True)
    # Copy weights so they're identical
    model_ckpt.load_state_dict(model_no_ckpt.state_dict())

    # Both in train mode so the checkpointing branch is exercised
    model_no_ckpt.train()
    model_ckpt.train()

    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))

    _, loss_no_ckpt = model_no_ckpt(x, y)
    _, loss_ckpt = model_ckpt(x, y)

    assert torch.allclose(loss_no_ckpt, loss_ckpt, atol=1e-5), \
        f"Loss differs: no_ckpt={loss_no_ckpt.item()}, ckpt={loss_ckpt.item()}"
