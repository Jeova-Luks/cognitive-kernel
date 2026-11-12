"""Component [3] MEMORY - typed (cru).

Four banks, as in §6.2: factual, procedural, causal, episodic. The semantic
banks are seeded from ``memory_seed.json``; the episodic bank is filled by the
pipeline with self-verified episodes as the system runs.

Deviation from the spec: instead of FAISS + all-MiniLM-L6-v2, retrieval uses
any ``embed(texts) -> vectors`` callable (the demo uses the nomic embedding
model already installed in LM Studio) with brute-force cosine similarity in
pure Python. With no embedder, it falls back to bag-of-words cosine. At a few
hundred entries brute force is instant, so FAISS buys nothing yet.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BANKS = ("factual", "procedural", "causal", "episodic")
SEED_PATH = Path(__file__).with_name("memory_seed.json")

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class MemoryEntry:
    bank: str
    text: str
    meta: dict = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass
class Hit:
    bank: str
    text: str
    score: float
    meta: dict


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _bow(text: str) -> Counter:
    return Counter(re.findall(r"[a-z]+", text.lower()))


def _bow_cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class TypedMemory:
    def __init__(self, embedder: Embedder | None = None, seed_path: Path | None = SEED_PATH,
                 query_prefix: str = "", doc_prefix: str = ""):
        self.embedder = embedder
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.entries: dict[str, list[MemoryEntry]] = {b: [] for b in BANKS}
        if seed_path is not None and Path(seed_path).exists():
            seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
            for bank, texts in seed.items():
                for text in texts:
                    self.add(bank, text, embed=False)

    def _check_bank(self, bank: str) -> None:
        if bank not in self.entries:
            raise KeyError(f"unknown memory bank '{bank}'; banks: {', '.join(BANKS)}")

    def add(self, bank: str, text: str, meta: dict | None = None,
            vector: list[float] | None = None, embed: bool = True) -> MemoryEntry:
        self._check_bank(bank)
        entry = MemoryEntry(bank, text, meta or {}, vector)
        if entry.vector is None and embed and self.embedder is not None:
            entry.vector = self.embedder([self.doc_prefix + text])[0]
        self.entries[bank].append(entry)
        return entry

    def _ensure_vectors(self, entries: list[MemoryEntry]) -> None:
        missing = [e for e in entries if e.vector is None]
        if missing and self.embedder is not None:
            vectors = self.embedder([self.doc_prefix + e.text for e in missing])
            for e, v in zip(missing, vectors):
                e.vector = v

    def embed_query(self, query: str) -> list[float] | None:
        return self.embedder([self.query_prefix + query])[0] if self.embedder else None

    def search(self, bank: str, query: str, k: int = 3,
               query_vector: list[float] | None = None) -> list[Hit]:
        self._check_bank(bank)
        entries = self.entries[bank]
        if not entries:
            return []
        if self.embedder is not None:
            self._ensure_vectors(entries)
            qv = query_vector if query_vector is not None else self.embed_query(query)
            scored = [(_cosine(qv, e.vector), e) for e in entries]
        else:
            qb = _bow(query)
            scored = [(_bow_cosine(qb, _bow(e.text)), e) for e in entries]
        scored.sort(key=lambda se: se[0], reverse=True)
        return [Hit(e.bank, e.text, round(s, 4), e.meta) for s, e in scored[:k]]

    def search_banks(self, query: str, banks: tuple[str, ...], k: int = 3,
                     min_score: float = 0.0) -> list[Hit]:
        """Top-k hits pooled across several banks (one query embedding)."""
        qv = self.embed_query(query)
        hits = [h for b in banks for h in self.search(b, query, k=k, query_vector=qv)]
        hits = [h for h in hits if h.score >= min_score]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

    def size(self) -> dict[str, int]:
        return {b: len(v) for b, v in self.entries.items()}
