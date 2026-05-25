"""Tests for tokenizer_fast.py. Skipped if tokenizer_fast.json does not exist
(it is produced by stage_6a)."""
import pytest
from pathlib import Path

TOKENIZER_PATH = Path("tokenizer_fast.json")

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(),
    reason="tokenizer_fast.json not yet trained (run stage_6a first)",
)


def test_roundtrip_basic():
    from tokenizer_fast import FastBPETokenizer
    tok = FastBPETokenizer()
    samples = [
        "def hello(): return 'world'",
        "The quick brown fox jumps over the lazy dog.",
        "Resolva: 2 + 2 = ?",
        "(let x 1) (+ x 2)",
    ]
    for s in samples:
        ids = tok.encode(s)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        decoded = tok.decode(ids)
        # Allow whitespace normalization differences
        assert decoded.strip() == s.strip() or decoded.replace(" ", "") == s.replace(" ", "")


def test_special_tokens_present():
    from tokenizer_fast import FastBPETokenizer
    tok = FastBPETokenizer()
    vocab = tok.tokenizer.get_vocab()
    for sp in [
        "<|endoftext|>",
        "<|cdsl_start|>",
        "<|cdsl_end|>",
        "<|tool_call|>",
        "<|tool_result|>",
    ]:
        assert sp in vocab, f"missing special token: {sp}"


def test_compression_ratio_reasonable():
    """On a representative sample, bytes/token should be in range [2.5, 5.0]."""
    from tokenizer_fast import FastBPETokenizer
    tok = FastBPETokenizer()
    sample = (
        "def quicksort(arr):\n"
        "    if len(arr) <= 1: return arr\n"
        "    p = arr[0]\n"
        "    return quicksort([x for x in arr[1:] if x < p]) + [p] + "
        "quicksort([x for x in arr[1:] if x >= p])\n"
    )
    n_bytes = len(sample.encode("utf-8"))
    n_tokens = len(tok.encode(sample))
    ratio = n_bytes / n_tokens
    assert 2.5 <= ratio <= 5.0, f"bytes/token = {ratio}"
