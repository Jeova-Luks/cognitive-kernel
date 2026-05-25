"""Shared utilities for the Phase 1A data pipeline.

Every stage uses these:
- HuggingFace dataset push/pull for checkpointing between stages
- Standard logger
- The token-budget table per category
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

# Mix C — Generalist with focus. Tokens per category (target, billions).
TARGETS = {
    "python":        3.0e9,
    "math":          2.0e9,
    "english_prose": 3.0e9,
    "sexp":          1.0e9,
    "pt_br":         1.0e9,
}
TOTAL_TARGET = sum(TARGETS.values())   # 10.0e9

HF_USER = os.environ.get("HF_USER", "Jeova-Luks")
STAGE_REPO_PREFIX = "ck-stage"

def stage_repo(stage: int, suffix: str) -> str:
    """e.g. stage_repo(3, 'filtered') -> 'Jeova-Luks/ck-stage-3-filtered'."""
    return f"{HF_USER}/{STAGE_REPO_PREFIX}-{stage}-{suffix}"

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

def push_dataset(ds: Dataset, repo: str, private: bool = True) -> None:
    """Push a Dataset to HF Hub under our naming convention."""
    ds.push_to_hub(repo, private=private)

def pull_dataset(repo: str, split: str = "train") -> Dataset:
    """Pull a Dataset from HF Hub by repo name."""
    return load_dataset(repo, split=split)

@dataclass
class DocRecord:
    """Normalized record across all stages."""
    text: str
    category: str       # one of TARGETS.keys()
    source: str         # original dataset id, e.g. 'bigcode/the-stack-v2-dedup'
    doc_id: str         # unique id per doc; we generate sequential ones
    n_chars: int        # convenience: len(text)
    metadata: dict | None = None   # optional source-specific fields

def docs_to_dataset(docs: Iterable[DocRecord]) -> Dataset:
    """Materialize an iterable of DocRecord into a HF Dataset."""
    records = [d.__dict__ for d in docs]
    return Dataset.from_list(records)
