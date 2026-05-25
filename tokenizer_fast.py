"""Production tokenizer wrapper around HuggingFace tokenizers.

For the pedagogical pure-Python BPE implementation, see tokenizer.py."""
from __future__ import annotations
from pathlib import Path

from tokenizers import Tokenizer


class FastBPETokenizer:
    def __init__(self, path: Path | str = "tokenizer_fast.json"):
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
