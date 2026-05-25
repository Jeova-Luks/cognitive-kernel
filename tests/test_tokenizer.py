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
        assert special_id in tok.vocab, \
            f"{special} (id {special_id}) missing from vocab"
        assert tok.vocab[special_id] == special.encode("utf-8")
