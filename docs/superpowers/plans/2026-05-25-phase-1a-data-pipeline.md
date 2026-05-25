# Phase 1A — Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Tasks 11-13 require manual execution outside subagent reach (HuggingFace auth, Kaggle session, Drive mount). Subagent-doable tasks are 0-10.**

**Goal:** Build the 7-stage data pipeline (scripts + tests) and then execute it across Kaggle/Codespaces/Colab to produce ~10 B clean tokens in nanoGPT-format `.bin` shards on Google Drive, ready for Phase 1B pre-training.

**Architecture:** Each stage is a standalone Python script in `scripts/data_pipeline/`. Stages exchange data through HuggingFace Datasets (private) for checkpoint-able resumption. Pure-Python logic (filtering, signature building, tokenization wrappers) is unit-tested in `tests/`. Heavy compute runs on Kaggle Notebooks (30 GB RAM, 73 GB disk). Final shards land on Google Drive via Colab.

**Tech Stack:** Python 3.10+, `datasets`, `huggingface_hub`, `tokenizers` (Rust BPE), `datasketch` (MinHash LSH), `fasttext`, `langdetect`, `beautifulsoup4`, `pyarrow` (parquet), `numpy`, `pytest`.

**Spec reference:** [docs/superpowers/specs/2026-05-25-phase-1a-data-pipeline-design.md](../specs/2026-05-25-phase-1a-data-pipeline-design.md).

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `requirements.txt` | modify | Add pipeline deps |
| `tokenizer_fast.py` | **create** | Wrapper around HF tokenizers library |
| `tokenizer_fast.json` | **create** (by Task 7 script) | Trained BPE vocab + merges |
| `scripts/data_pipeline/common.py` | **create** | Shared utilities: HF auth, dataset push/pull, logging, category dispatch |
| `scripts/data_pipeline/stage_1_download.py` | **create** | Download raw datasets from HF |
| `scripts/data_pipeline/stage_2_normalize.py` | **create** | Unicode NFC + strip HTML + lang detect |
| `scripts/data_pipeline/stage_3_heuristics.py` | **create** | L1 quality filters |
| `scripts/data_pipeline/stage_4_minhash.py` | **create** | L2 near-dedup via MinHash LSH |
| `scripts/data_pipeline/stage_5_fasttext_train.py` | **create** | Train quality classifier |
| `scripts/data_pipeline/stage_5_fasttext_apply.py` | **create** | Apply classifier |
| `scripts/data_pipeline/stage_6a_train_tokenizer.py` | **create** | Train BPE on 1 GB sample |
| `scripts/data_pipeline/stage_6b_tokenize.py` | **create** | Tokenize full corpus |
| `scripts/data_pipeline/stage_7_shard_upload.py` | **create** | Shard + push to Drive |
| `scripts/data_pipeline/validate_phase_1a.py` | **create** | Run DoD checks |
| `configs/tiny_signal_test.yaml` | **create** | Tiny 10M model config for signal validation |
| `tests/test_tokenizer_fast.py` | **create** | 3 tests (roundtrip, specials, compression) |
| `tests/test_data_pipeline.py` | **create** | Unit tests for pure functions in stages 2-6 |
| `tests/fixtures/data_pipeline/` | **create** (directory) | Tiny sample documents for testing filters |
| `README.md` | modify | Phase 1A status |

---

## Task 0: Dependencies and module scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/data_pipeline/__init__.py`
- Create: `scripts/data_pipeline/common.py`
- Create: `tests/fixtures/data_pipeline/.gitkeep`

- [ ] **Step 0.1: Add new dependencies to requirements.txt**

Append to `requirements.txt` (keep existing entries unchanged):

```
# Phase 1A data pipeline
datasets>=2.18.0
huggingface_hub>=0.21.0
tokenizers>=0.15.0
datasketch>=1.6.0
fasttext>=0.9.2
langdetect>=1.0.9
beautifulsoup4>=4.12.0
pyarrow>=14.0.0
tqdm>=4.66.0
```

- [ ] **Step 0.2: Verify installation in Codespaces**

Run:
```bash
pip install -r requirements.txt
python -c "import datasets, tokenizers, datasketch, fasttext, langdetect, bs4, pyarrow; print('ok')"
```

Expected: `ok`. (fasttext may need a compiler; if it fails on first install, `pip install fasttext-wheel` as a fallback works.)

- [ ] **Step 0.3: Create the package directory**

```bash
mkdir -p scripts/data_pipeline
touch scripts/data_pipeline/__init__.py
mkdir -p tests/fixtures/data_pipeline
touch tests/fixtures/data_pipeline/.gitkeep
```

- [ ] **Step 0.4: Create `scripts/data_pipeline/common.py`**

Write:

```python
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
```

- [ ] **Step 0.5: Commit**

```bash
git add requirements.txt scripts/data_pipeline/ tests/fixtures/data_pipeline/
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "chore(phase-1a): add pipeline deps + common.py scaffolding"
git push
```

---

## Task 1: Stage 1 — Download raw datasets

**Files:**
- Create: `scripts/data_pipeline/stage_1_download.py`
- Create: `tests/test_data_pipeline.py` (with a single test for `target_count` arithmetic)

This task writes the DOWNLOAD ORCHESTRATOR. Actual execution against HuggingFace happens in Task 11 (manual). What we test here are the small helper functions; the network calls are not unit-testable without mocks.

- [ ] **Step 1.1: Write the test for target-token arithmetic**

Create `tests/test_data_pipeline.py`:

```python
"""Unit tests for pure functions in the data pipeline."""
import pytest
from scripts.data_pipeline.common import TARGETS, TOTAL_TARGET


def test_targets_sum_to_10b():
    assert TOTAL_TARGET == pytest.approx(1.0e10)


def test_targets_proportions_match_mix_c():
    assert TARGETS["python"]        / TOTAL_TARGET == pytest.approx(0.30)
    assert TARGETS["math"]          / TOTAL_TARGET == pytest.approx(0.20)
    assert TARGETS["english_prose"] / TOTAL_TARGET == pytest.approx(0.30)
    assert TARGETS["sexp"]          / TOTAL_TARGET == pytest.approx(0.10)
    assert TARGETS["pt_br"]         / TOTAL_TARGET == pytest.approx(0.10)
```

- [ ] **Step 1.2: Run test, verify it passes**

Run: `python -m pytest tests/test_data_pipeline.py -v`

Expected: 2 PASSED.

- [ ] **Step 1.3: Write `scripts/data_pipeline/stage_1_download.py`**

```python
"""Stage 1: Download raw documents from upstream HuggingFace datasets.

Pulls each source listed in the spec, samples enough docs to hit ~6x the
target token count (so subsequent filter stages have headroom), and pushes
the merged result to HF as ck-stage-1-raw.

Run on Kaggle with HF_TOKEN exported. Idempotent: re-running re-uploads.
"""
from __future__ import annotations
import argparse
from typing import Iterator

from datasets import load_dataset

from .common import (
    DocRecord, docs_to_dataset, push_dataset, stage_repo, get_logger, TARGETS,
)

log = get_logger("stage1")

# How much raw text to ingest per category, given a ~60-70% expected loss
# through filters: take ~6x the final target.
RAW_MULTIPLIER = 6.0

# Average bytes per token across our domain (empirical estimate; refined later)
BYTES_PER_TOKEN = 4.0


def target_bytes(category: str) -> int:
    return int(TARGETS[category] * RAW_MULTIPLIER * BYTES_PER_TOKEN)


def iter_python() -> Iterator[DocRecord]:
    """Pull from the_stack_v2_dedup (python), codeparrot_clean, python_edu."""
    quota = target_bytes("python")
    consumed = 0
    counter = 0

    # Source 1: The Stack v2 dedup, python only, stars >= 5
    log.info("python: streaming bigcode/the-stack-v2-dedup ...")
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train",
        streaming=True,
    )
    for r in ds:
        if r.get("language") != "Python":
            continue
        stars = r.get("revision_stars") or 0
        if stars < 5:
            continue
        text = r.get("content") or ""
        if not text:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text,
            category="python",
            source="bigcode/the-stack-v2-dedup",
            doc_id=f"python-stack-{counter:09d}",
            n_chars=len(text),
            metadata={"stars": stars, "repo": r.get("repo_name")},
        )
        if consumed >= quota * 0.85:    # 85% of quota from this source
            break

    # Source 2: CodeParrot clean — fills the gap
    log.info("python: streaming codeparrot/codeparrot-clean ...")
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    for r in ds:
        if consumed >= quota:
            break
        text = r.get("content") or ""
        if not text:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text,
            category="python",
            source="codeparrot/codeparrot-clean",
            doc_id=f"python-codeparrot-{counter:09d}",
            n_chars=len(text),
        )

    log.info(f"python: total {consumed:,} bytes in {counter:,} docs")


def iter_math() -> Iterator[DocRecord]:
    quota = target_bytes("math")
    consumed = 0
    counter = 0

    log.info("math: streaming EleutherAI/proof-pile-2 ...")
    ds = load_dataset("EleutherAI/proof-pile-2", split="train", streaming=True)
    for r in ds:
        if consumed >= quota * 0.6:
            break
        text = r.get("text") or ""
        if not text:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="math",
            source="EleutherAI/proof-pile-2",
            doc_id=f"math-proofpile-{counter:09d}",
            n_chars=len(text),
        )

    log.info("math: loading nvidia/OpenMathInstruct-1 ...")
    ds = load_dataset("nvidia/OpenMathInstruct-1", split="train")
    for r in ds:
        if consumed >= quota * 0.9:
            break
        q = r.get("question") or ""
        a = r.get("answer") or ""
        text = f"Question: {q}\n\nSolution: {a}"
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="math",
            source="nvidia/OpenMathInstruct-1",
            doc_id=f"math-openmath-{counter:09d}",
            n_chars=len(text),
        )

    log.info("math: loading gsm8k ...")
    ds = load_dataset("gsm8k", "main", split="train")
    for r in ds:
        if consumed >= quota:
            break
        text = f"Problem: {r['question']}\n\nSolution: {r['answer']}"
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="math",
            source="gsm8k",
            doc_id=f"math-gsm8k-{counter:09d}",
            n_chars=len(text),
        )

    log.info(f"math: total {consumed:,} bytes in {counter:,} docs")


def iter_english_prose() -> Iterator[DocRecord]:
    quota = target_bytes("english_prose")
    consumed = 0
    counter = 0

    log.info("english: streaming HuggingFaceFW/fineweb-edu (sample-10BT) ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True,
    )
    for r in ds:
        if consumed >= quota * 0.85:
            break
        text = r.get("text") or ""
        if not text:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="english_prose",
            source="HuggingFaceFW/fineweb-edu",
            doc_id=f"prose-fineweb-{counter:09d}",
            n_chars=len(text),
        )

    log.info("english: streaming wikimedia/wikipedia en ...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        split="train", streaming=True,
    )
    for r in ds:
        if consumed >= quota:
            break
        text = r.get("text") or ""
        if len(text) < 500:    # skip stubs
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="english_prose",
            source="wikimedia/wikipedia-en",
            doc_id=f"prose-wiki-{counter:09d}",
            n_chars=len(text),
            metadata={"title": r.get("title")},
        )
    log.info(f"english_prose: total {consumed:,} bytes in {counter:,} docs")


def iter_sexp() -> Iterator[DocRecord]:
    quota = target_bytes("sexp")
    consumed = 0
    counter = 0
    LISP_LANGS = {"Common Lisp", "Scheme", "Racket", "Clojure"}

    log.info("sexp: streaming bigcode/the-stack-v2-dedup (lisp variants) ...")
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train", streaming=True,
    )
    for r in ds:
        if consumed >= quota * 0.35:
            break
        if r.get("language") not in LISP_LANGS:
            continue
        text = r.get("content") or ""
        if not text:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="sexp",
            source="bigcode/the-stack-v2-dedup",
            doc_id=f"sexp-lisp-{counter:09d}",
            n_chars=len(text),
            metadata={"lang": r.get("language")},
        )

    log.info("sexp: streaming proof-pile-2 latex ...")
    ds = load_dataset(
        "EleutherAI/proof-pile-2", "arxiv",
        split="train", streaming=True,
    )
    for r in ds:
        if consumed >= quota * 0.75:
            break
        text = r.get("text") or ""
        if not text:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="sexp",
            source="EleutherAI/proof-pile-2-arxiv",
            doc_id=f"sexp-latex-{counter:09d}",
            n_chars=len(text),
        )

    log.info("sexp: loading internlm/Lean-Workbook ...")
    ds = load_dataset("internlm/Lean-Workbook", split="train")
    for r in ds:
        if consumed >= quota * 0.8:
            break
        text = r.get("formal_proof") or r.get("informal_proof") or ""
        if not text:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="sexp",
            source="internlm/Lean-Workbook",
            doc_id=f"sexp-lean-{counter:09d}",
            n_chars=len(text),
        )

    log.info("sexp: streaming the-stack-v2-dedup json ...")
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train", streaming=True,
    )
    for r in ds:
        if consumed >= quota:
            break
        if r.get("language") != "JSON":
            continue
        text = r.get("content") or ""
        if not text or len(text) < 200 or len(text) > 50_000:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="sexp",
            source="bigcode/the-stack-v2-dedup-json",
            doc_id=f"sexp-json-{counter:09d}",
            n_chars=len(text),
        )
    log.info(f"sexp: total {consumed:,} bytes in {counter:,} docs")


def iter_pt_br() -> Iterator[DocRecord]:
    quota = target_bytes("pt_br")
    consumed = 0
    counter = 0

    log.info("pt_br: streaming wikimedia/wikipedia pt ...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.pt",
        split="train", streaming=True,
    )
    for r in ds:
        if consumed >= quota * 0.7:
            break
        text = r.get("text") or ""
        if len(text) < 500:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="pt_br",
            source="wikimedia/wikipedia-pt",
            doc_id=f"ptbr-wiki-{counter:09d}",
            n_chars=len(text),
            metadata={"title": r.get("title")},
        )

    log.info("pt_br: loading nilc-nlp/BrWac ...")
    try:
        ds = load_dataset("nilc-nlp/BrWac", split="train", streaming=True)
        for r in ds:
            if consumed >= quota * 0.9:
                break
            text = " ".join(r.get("text", {}).get("paragraphs", [])) or ""
            if len(text) < 500:
                continue
            consumed += len(text)
            counter += 1
            yield DocRecord(
                text=text, category="pt_br",
                source="nilc-nlp/BrWac",
                doc_id=f"ptbr-brwac-{counter:09d}",
                n_chars=len(text),
            )
    except Exception as e:
        log.warning(f"BrWac unavailable ({e}); falling back to cc100.")

    log.info("pt_br: streaming cc100 pt ...")
    ds = load_dataset("cc100", lang="pt", split="train", streaming=True)
    for r in ds:
        if consumed >= quota:
            break
        text = r.get("text") or ""
        if len(text) < 300:
            continue
        consumed += len(text)
        counter += 1
        yield DocRecord(
            text=text, category="pt_br",
            source="cc100-pt",
            doc_id=f"ptbr-cc100-{counter:09d}",
            n_chars=len(text),
        )
    log.info(f"pt_br: total {consumed:,} bytes in {counter:,} docs")


CATEGORY_ITERATORS = {
    "python":        iter_python,
    "math":          iter_math,
    "english_prose": iter_english_prose,
    "sexp":          iter_sexp,
    "pt_br":         iter_pt_br,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories", nargs="+",
        default=list(TARGETS.keys()),
        help="Subset of categories to download (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't push to HF; just iterate to verify access")
    args = parser.parse_args()

    all_docs: list[DocRecord] = []
    for cat in args.categories:
        if cat not in CATEGORY_ITERATORS:
            log.error(f"unknown category: {cat}")
            continue
        log.info(f"=== Downloading category: {cat} ===")
        for doc in CATEGORY_ITERATORS[cat]():
            all_docs.append(doc)

    log.info(f"total docs collected: {len(all_docs):,}")
    if args.dry_run:
        log.info("dry-run: skipping HF push")
        return

    ds = docs_to_dataset(all_docs)
    repo = stage_repo(1, "raw")
    log.info(f"pushing to {repo} ...")
    push_dataset(ds, repo)
    log.info("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.4: Smoke-import test (no network)**

Run:
```bash
python -c "from scripts.data_pipeline.stage_1_download import CATEGORY_ITERATORS, target_bytes; print(list(CATEGORY_ITERATORS), target_bytes('python'))"
```

Expected:
```
['python', 'math', 'english_prose', 'sexp', 'pt_br'] 72000000000
```

(72 billion bytes = 72 GB raw target for python; will shrink through filters.)

- [ ] **Step 1.5: Commit**

```bash
git add scripts/data_pipeline/stage_1_download.py tests/test_data_pipeline.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): stage 1 download script — pulls all 5 categories from HF"
git push
```

---

## Task 2: Stage 2 — Normalize

**Files:**
- Create: `scripts/data_pipeline/stage_2_normalize.py`
- Modify: `tests/test_data_pipeline.py` (add normalization tests)

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/test_data_pipeline.py`:

```python
from scripts.data_pipeline.stage_2_normalize import (
    normalize_unicode,
    strip_html,
    is_valid_length,
)


def test_normalize_unicode_nfc():
    # 'é' as two-codepoint NFD vs one-codepoint NFC
    s_nfd = "café"
    s_nfc = normalize_unicode(s_nfd)
    assert s_nfc == "café"
    assert len(s_nfc) == 4


def test_strip_html_basic():
    html = "<p>hello <b>world</b></p><script>alert(1)</script>"
    assert strip_html(html) == "hello world"


def test_strip_html_preserves_code():
    # Code blocks shouldn't have their content stripped to plain text
    # (we treat them as plain text — no special preservation here)
    text = "plain text without html"
    assert strip_html(text) == text


def test_is_valid_length():
    assert is_valid_length("x" * 50) is True
    assert is_valid_length("x" * 49) is False
    assert is_valid_length("x" * 1_000_000) is True
    assert is_valid_length("x" * 1_000_001) is False
```

- [ ] **Step 2.2: Run tests, verify failure**

Run: `python -m pytest tests/test_data_pipeline.py -v`

Expected: 4 new tests FAIL with ImportError (the module doesn't exist yet).

- [ ] **Step 2.3: Implement `scripts/data_pipeline/stage_2_normalize.py`**

```python
"""Stage 2: Normalize Unicode, strip HTML, drop docs outside length bounds.

Reads ck-stage-1-raw, writes ck-stage-2-normalized.

Pure functions are unit-tested. The main() orchestration runs on Kaggle.
"""
from __future__ import annotations
import argparse
import unicodedata

from bs4 import BeautifulSoup
from datasets import Dataset

from .common import (
    DocRecord, docs_to_dataset, pull_dataset, push_dataset, stage_repo, get_logger,
)

log = get_logger("stage2")

MIN_CHARS = 50
MAX_CHARS = 1_000_000


def normalize_unicode(text: str) -> str:
    """NFC normalize so visually-identical strings compare equal."""
    return unicodedata.normalize("NFC", text)


def strip_html(text: str) -> str:
    """Strip HTML tags using BeautifulSoup. Returns plain text.

    For non-HTML inputs (most of our data), this is approximately a no-op.
    """
    if "<" not in text or ">" not in text:
        return text
    soup = BeautifulSoup(text, "html.parser")
    # remove script/style tags entirely
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def is_valid_length(text: str) -> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS


def normalize_doc(rec: dict) -> dict | None:
    """Normalize one HF record. Returns None if the doc should be dropped."""
    text = rec.get("text", "")
    text = normalize_unicode(text)
    text = strip_html(text)
    text = text.strip()
    if not is_valid_length(text):
        return None
    rec["text"] = text
    rec["n_chars"] = len(text)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo",
                        default=stage_repo(1, "raw"))
    parser.add_argument("--output-repo",
                        default=stage_repo(2, "normalized"))
    parser.add_argument("--num-proc", type=int, default=4)
    args = parser.parse_args()

    log.info(f"pulling {args.input_repo} ...")
    ds = pull_dataset(args.input_repo)
    log.info(f"loaded {len(ds):,} docs")

    log.info("normalizing ...")
    normalized = ds.map(
        normalize_doc,
        num_proc=args.num_proc,
    ).filter(lambda r: r is not None)
    log.info(f"survived: {len(normalized):,} / {len(ds):,}")

    log.info(f"pushing to {args.output_repo} ...")
    push_dataset(normalized, args.output_repo)
    log.info("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.4: Run tests, verify pass**

Run: `python -m pytest tests/test_data_pipeline.py -v`

Expected: 6 PASSED (2 from Task 1 + 4 from Task 2).

- [ ] **Step 2.5: Commit**

```bash
git add scripts/data_pipeline/stage_2_normalize.py tests/test_data_pipeline.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): stage 2 normalize — Unicode NFC + HTML strip + length filter"
git push
```

---

## Task 3: Stage 3 — L1 Heuristic filters

**Files:**
- Create: `scripts/data_pipeline/stage_3_heuristics.py`
- Modify: `tests/test_data_pipeline.py`

This is the most filter-logic-heavy stage. Pure functions get extensive unit tests.

- [ ] **Step 3.1: Write failing tests**

Append to `tests/test_data_pipeline.py`:

```python
from scripts.data_pipeline.stage_3_heuristics import (
    mean_line_length,
    max_line_length,
    ratio_special_chars,
    ratio_whitespace,
    unique_lines_ratio,
    is_likely_python,
    passes_heuristics,
)


def test_mean_line_length():
    assert mean_line_length("aa\nbb\nccc") == pytest.approx(7 / 3)
    assert mean_line_length("") == 0.0


def test_max_line_length():
    assert max_line_length("a\naaa\naa") == 3
    assert max_line_length("") == 0


def test_ratio_special_chars():
    s = "abc!@#"
    assert ratio_special_chars(s) == pytest.approx(0.5)
    assert ratio_special_chars("aaaa") == 0.0


def test_ratio_whitespace():
    assert ratio_whitespace("a b c") == pytest.approx(2 / 5)
    assert ratio_whitespace("abc") == 0.0


def test_unique_lines_ratio():
    text = "a\nb\na\nb\na"
    assert unique_lines_ratio(text) == pytest.approx(2 / 5)


def test_is_likely_python_positive():
    code = "def foo():\n    return 42"
    assert is_likely_python(code) is True


def test_is_likely_python_negative():
    prose = "The quick brown fox jumps over the lazy dog."
    assert is_likely_python(prose) is False


def test_passes_heuristics_python_pass():
    doc = "def square(x):\n    return x * x\n\nprint(square(5))"
    assert passes_heuristics(doc, "python") is True


def test_passes_heuristics_too_short():
    assert passes_heuristics("hi", "english_prose") is False


def test_passes_heuristics_too_repetitive():
    repeated = "spam\n" * 100
    assert passes_heuristics(repeated, "english_prose") is False


def test_passes_heuristics_python_not_python():
    # A doc tagged "python" but with no Python markers should fail
    assert passes_heuristics("just some prose here without any code keywords", "python") is False
```

- [ ] **Step 3.2: Run tests, verify failure**

Run: `python -m pytest tests/test_data_pipeline.py -v -k 'heuristic or python or line or ratio'`

Expected: 11 new tests FAIL (ImportError).

- [ ] **Step 3.3: Implement `scripts/data_pipeline/stage_3_heuristics.py`**

```python
"""Stage 3: L1 heuristic quality filters.

All decision logic is pure functions, unit-tested. main() pulls
ck-stage-2-normalized, applies filters, pushes ck-stage-3-filtered.
"""
from __future__ import annotations
import argparse
import hashlib

from .common import pull_dataset, push_dataset, stage_repo, get_logger

log = get_logger("stage3")


def mean_line_length(text: str) -> float:
    lines = text.split("\n")
    if not lines:
        return 0.0
    return sum(len(l) for l in lines) / len(lines)


def max_line_length(text: str) -> int:
    lines = text.split("\n")
    if not lines:
        return 0
    return max(len(l) for l in lines)


def ratio_special_chars(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return n / len(text)


def ratio_whitespace(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for c in text if c.isspace())
    return n / len(text)


def unique_lines_ratio(text: str) -> float:
    lines = text.split("\n")
    if not lines:
        return 1.0
    return len(set(lines)) / len(lines)


PYTHON_MARKERS = ("def ", "class ", "import ", "    ", "return ", "= ", "if ", "for ")


def is_likely_python(text: str) -> bool:
    return any(m in text for m in PYTHON_MARKERS)


def passes_heuristics(text: str, category: str) -> bool:
    # Length
    if not (50 <= len(text) <= 1_000_000):
        return False

    # Line structure
    mll = mean_line_length(text)
    if not (1.0 < mll <= 1000.0):
        return False
    if max_line_length(text) > 100_000:
        return False

    # Composition
    if ratio_special_chars(text) > 0.5:
        return False
    if ratio_whitespace(text) > 0.5:
        return False

    # Repetition (only meaningful when there are enough lines)
    n_lines = len(text.split("\n"))
    if n_lines > 10 and unique_lines_ratio(text) < 0.7:
        return False

    # Category-specific
    if category == "python":
        if not is_likely_python(text):
            return False
    elif category in ("english_prose", "pt_br"):
        try:
            from langdetect import detect_langs
            target = "en" if category == "english_prose" else "pt"
            langs = detect_langs(text[:5000])
            if not any(l.lang == target and l.prob > 0.95 for l in langs):
                return False
        except Exception:
            return False

    return True


def doc_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def filter_record(rec: dict) -> bool:
    return passes_heuristics(rec["text"], rec["category"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo",
                        default=stage_repo(2, "normalized"))
    parser.add_argument("--output-repo",
                        default=stage_repo(3, "filtered"))
    parser.add_argument("--num-proc", type=int, default=4)
    args = parser.parse_args()

    log.info(f"pulling {args.input_repo}")
    ds = pull_dataset(args.input_repo)
    log.info(f"loaded {len(ds):,} docs")

    # Exact dedup via MD5
    seen: set[str] = set()
    def keep_if_unique_hash(rec):
        h = doc_md5(rec["text"])
        if h in seen:
            return False
        seen.add(h)
        return True

    log.info("applying heuristic filters ...")
    survived = ds.filter(filter_record, num_proc=args.num_proc)
    log.info(f"heuristic survived: {len(survived):,} / {len(ds):,}")

    log.info("exact dedup (md5) ...")
    survived = survived.filter(keep_if_unique_hash)
    log.info(f"after exact dedup: {len(survived):,}")

    log.info(f"pushing to {args.output_repo}")
    push_dataset(survived, args.output_repo)
    log.info("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4: Run tests, verify pass**

Run: `python -m pytest tests/test_data_pipeline.py -v`

Expected: 17 PASSED (cumulative).

- [ ] **Step 3.5: Commit**

```bash
git add scripts/data_pipeline/stage_3_heuristics.py tests/test_data_pipeline.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): stage 3 L1 heuristic filters + exact dedup"
git push
```

---

## Task 4: Stage 4 — L2 MinHash near-deduplication

**Files:**
- Create: `scripts/data_pipeline/stage_4_minhash.py`
- Modify: `tests/test_data_pipeline.py`

- [ ] **Step 4.1: Write failing tests**

Append:

```python
from scripts.data_pipeline.stage_4_minhash import (
    build_signature,
    shingles_of,
)


def test_shingles_of_basic():
    text = "the quick brown fox jumps over"
    shing = list(shingles_of(text, k=3))
    assert shing[0] == "the quick brown"
    assert shing[-1] == "fox jumps over"
    assert len(shing) == 4


def test_shingles_short_text():
    text = "two words"
    shing = list(shingles_of(text, k=3))
    assert shing == []


def test_build_signature_deterministic():
    text = "the quick brown fox jumps over the lazy dog"
    s1 = build_signature(text, num_perm=64)
    s2 = build_signature(text, num_perm=64)
    assert s1.digest().tolist() == s2.digest().tolist()


def test_build_signature_similar_texts_have_close_jaccard():
    text_a = "the quick brown fox jumps over the lazy dog"
    text_b = "the quick brown fox jumps over the lazy cat"
    s_a = build_signature(text_a, num_perm=128)
    s_b = build_signature(text_b, num_perm=128)
    j = s_a.jaccard(s_b)
    assert j > 0.7   # strong overlap because only one word changed
```

- [ ] **Step 4.2: Run, verify failure**

Run: `python -m pytest tests/test_data_pipeline.py -v -k 'shingle or signature'`

Expected: 4 FAIL.

- [ ] **Step 4.3: Implement `scripts/data_pipeline/stage_4_minhash.py`**

```python
"""Stage 4: L2 near-dedup via MinHash LSH.

Pure helpers (shingles, signature) are unit-tested. main() is the heavy
pipeline: build signatures for all docs, find clusters, keep longest per
cluster.

Memory: ~5-8 GB for ~100M docs. Run on Kaggle (30 GB RAM).
"""
from __future__ import annotations
import argparse
from typing import Iterator

from datasketch import MinHash, MinHashLSH

from .common import pull_dataset, push_dataset, stage_repo, get_logger

log = get_logger("stage4")

SHINGLE_K = 5
NUM_PERM = 256
JACCARD_THRESHOLD = 0.85


def shingles_of(text: str, k: int = SHINGLE_K) -> Iterator[str]:
    """Yield k-grams of whitespace-tokenized text."""
    tokens = text.lower().split()
    for i in range(len(tokens) - k + 1):
        yield " ".join(tokens[i:i + k])


def build_signature(text: str, num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles_of(text):
        m.update(sh.encode("utf-8"))
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", default=stage_repo(3, "filtered"))
    parser.add_argument("--output-repo", default=stage_repo(4, "deduped"))
    parser.add_argument("--threshold", type=float, default=JACCARD_THRESHOLD)
    parser.add_argument("--num-perm", type=int, default=NUM_PERM)
    args = parser.parse_args()

    log.info(f"pulling {args.input_repo}")
    ds = pull_dataset(args.input_repo)
    log.info(f"loaded {len(ds):,} docs; building signatures ...")

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    # Insert all signatures
    for idx, rec in enumerate(ds):
        sig = build_signature(rec["text"], num_perm=args.num_perm)
        lsh.insert(str(idx), sig, check_duplication=False)
        if idx % 10_000 == 0:
            log.info(f"  signed {idx:,} / {len(ds):,}")

    log.info("finding clusters ...")
    keep_idx: set[int] = set()
    dropped: set[int] = set()
    for idx in range(len(ds)):
        if idx in dropped:
            continue
        sig = build_signature(ds[idx]["text"], num_perm=args.num_perm)
        neighbors = [int(n) for n in lsh.query(sig)]
        if len(neighbors) == 1:
            keep_idx.add(idx)
            continue
        # Cluster: keep the longest doc, drop the rest
        longest = max(neighbors, key=lambda i: len(ds[i]["text"]))
        keep_idx.add(longest)
        for n in neighbors:
            if n != longest:
                dropped.add(n)

    log.info(f"clusters resolved: keeping {len(keep_idx):,}, dropping {len(dropped):,}")
    survived = ds.select(sorted(keep_idx))
    log.info(f"pushing {len(survived):,} to {args.output_repo}")
    push_dataset(survived, args.output_repo)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4: Run tests, verify pass**

Run: `python -m pytest tests/test_data_pipeline.py -v`

Expected: 21 PASSED.

- [ ] **Step 4.5: Commit**

```bash
git add scripts/data_pipeline/stage_4_minhash.py tests/test_data_pipeline.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): stage 4 MinHash LSH near-dedup (threshold 0.85, num_perm 256)"
git push
```

---

## Task 5: Stage 5a — FastText classifier training

**Files:**
- Create: `scripts/data_pipeline/stage_5_fasttext_train.py`
- Modify: `tests/test_data_pipeline.py`

- [ ] **Step 5.1: Write the failing test**

Append:

```python
from scripts.data_pipeline.stage_5_fasttext_train import (
    format_label,
    write_training_lines,
)


def test_format_label():
    assert format_label("high_quality", "hello world") == "__label__high_quality hello world"


def test_format_label_strips_newlines():
    assert format_label("low_quality", "line1\nline2") == "__label__low_quality line1 line2"


def test_write_training_lines(tmp_path):
    path = tmp_path / "out.txt"
    samples = [
        ("high_quality", "good text"),
        ("low_quality", "bad text"),
    ]
    write_training_lines(samples, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "__label__high_quality good text"
    assert lines[1] == "__label__low_quality bad text"
```

- [ ] **Step 5.2: Run, verify failure**

Run: `python -m pytest tests/test_data_pipeline.py -v -k 'format_label or write_training'`

Expected: 3 FAIL.

- [ ] **Step 5.3: Implement `scripts/data_pipeline/stage_5_fasttext_train.py`**

```python
"""Stage 5a: Train a FastText quality classifier.

Positives = Wikipedia FA + top-starred GitHub Python + curated math.
Negatives = random Common Crawl + zero-star GitHub + spam patterns.

Output: quality_classifier.bin (committed to artifacts, NOT git).
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import fasttext
from datasets import load_dataset

from .common import get_logger

log = get_logger("stage5_train")

TRAIN_FILE = "fasttext_train.txt"
MODEL_FILE = "quality_classifier.bin"
N_POS = 50_000
N_NEG = 50_000


def format_label(label: str, text: str) -> str:
    """Convert (label, text) to FastText supervised format."""
    clean = text.replace("\n", " ").replace("\r", " ").strip()
    return f"__label__{label} {clean}"


def write_training_lines(samples: list[tuple[str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for label, text in samples:
            f.write(format_label(label, text) + "\n")


def gather_positives(n: int) -> list[tuple[str, str]]:
    """Pull high-quality samples from upstream sources."""
    pos: list[tuple[str, str]] = []

    log.info("positives: Wikipedia FA-like (heuristic: long en wiki articles) ...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        split="train", streaming=True,
    )
    for rec in ds:
        if len(pos) >= n // 2:
            break
        if len(rec["text"]) > 5000:    # proxy for FA/featured-article length
            pos.append(("high_quality", rec["text"][:2000]))

    log.info("positives: top-starred Python from the_stack ...")
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train", streaming=True,
    )
    for rec in ds:
        if len(pos) >= n:
            break
        if rec.get("language") != "Python":
            continue
        if (rec.get("revision_stars") or 0) < 1000:
            continue
        pos.append(("high_quality", (rec.get("content") or "")[:2000]))
    return pos


def gather_negatives(n: int) -> list[tuple[str, str]]:
    """Pull low-quality / noisy samples."""
    neg: list[tuple[str, str]] = []

    log.info("negatives: cc100 random (treated as low quality by default) ...")
    ds = load_dataset("cc100", lang="en", split="train", streaming=True)
    for rec in ds:
        if len(neg) >= n // 2:
            break
        text = rec.get("text") or ""
        if len(text) < 100:
            continue
        neg.append(("low_quality", text[:2000]))

    log.info("negatives: zero-star Python from the_stack ...")
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train", streaming=True,
    )
    for rec in ds:
        if len(neg) >= n:
            break
        if rec.get("language") != "Python":
            continue
        if (rec.get("revision_stars") or 0) > 0:
            continue
        neg.append(("low_quality", (rec.get("content") or "")[:2000]))
    return neg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./artifacts")
    parser.add_argument("--n-pos", type=int, default=N_POS)
    parser.add_argument("--n-neg", type=int, default=N_NEG)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("gathering positives ...")
    pos = gather_positives(args.n_pos)
    log.info(f"positives: {len(pos):,}")

    log.info("gathering negatives ...")
    neg = gather_negatives(args.n_neg)
    log.info(f"negatives: {len(neg):,}")

    samples = pos + neg
    random.shuffle(samples)

    train_path = out_dir / TRAIN_FILE
    write_training_lines(samples, train_path)
    log.info(f"wrote {len(samples):,} lines to {train_path}")

    log.info("training fasttext supervised ...")
    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=50,
        lr=0.5,
        dim=100,
        wordNgrams=2,
        loss="softmax",
    )

    model_path = out_dir / MODEL_FILE
    model.save_model(str(model_path))
    log.info(f"saved model to {model_path}")

    # Quick self-eval on the training set itself
    n_correct = 0
    for label, text in samples[:1000]:
        pred = model.predict(text.replace("\n", " "))[0][0]
        if pred == f"__label__{label}":
            n_correct += 1
    log.info(f"self-accuracy on 1000 train samples: {n_correct / 1000:.2%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.4: Run tests, verify pass**

Run: `python -m pytest tests/test_data_pipeline.py -v`

Expected: 24 PASSED.

- [ ] **Step 5.5: Commit**

```bash
git add scripts/data_pipeline/stage_5_fasttext_train.py tests/test_data_pipeline.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): stage 5a FastText classifier training"
git push
```

---

## Task 6: Stage 5b — Apply classifier to docs

**Files:**
- Create: `scripts/data_pipeline/stage_5_fasttext_apply.py`
- Modify: `tests/test_data_pipeline.py`

- [ ] **Step 6.1: Write the failing test**

Append:

```python
from scripts.data_pipeline.stage_5_fasttext_apply import needs_classifier


def test_needs_classifier_pre_filtered():
    # Sources that are pre-filtered upstream should NOT be re-classified
    assert needs_classifier("HuggingFaceFW/fineweb-edu") is False
    assert needs_classifier("EleutherAI/proof-pile-2") is False
    assert needs_classifier("wikimedia/wikipedia-en") is False
    assert needs_classifier("wikimedia/wikipedia-pt") is False
    assert needs_classifier("bigcode/python-edu") is False


def test_needs_classifier_raw():
    # Raw sources need our classifier
    assert needs_classifier("cc100-pt") is True
    assert needs_classifier("nilc-nlp/BrWac") is True
    assert needs_classifier("bigcode/the-stack-v2-dedup-json") is True
```

- [ ] **Step 6.2: Run, verify failure**

Run: `python -m pytest tests/test_data_pipeline.py -v -k needs_classifier`

Expected: 2 FAIL.

- [ ] **Step 6.3: Implement `scripts/data_pipeline/stage_5_fasttext_apply.py`**

```python
"""Stage 5b: Apply the trained FastText classifier to docs that need it.

Hybrid policy: pre-filtered upstream sources pass through unchanged; raw
sources get classified and dropped below threshold.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import fasttext

from .common import pull_dataset, push_dataset, stage_repo, get_logger

log = get_logger("stage5_apply")

PRE_FILTERED_SOURCES = {
    "HuggingFaceFW/fineweb-edu",
    "EleutherAI/proof-pile-2",
    "EleutherAI/proof-pile-2-arxiv",
    "wikimedia/wikipedia-en",
    "wikimedia/wikipedia-pt",
    "bigcode/python-edu",
    "nvidia/OpenMathInstruct-1",
    "gsm8k",
    "meta-math/MetaMathQA",
    "internlm/Lean-Workbook",
}


def needs_classifier(source: str) -> bool:
    return source not in PRE_FILTERED_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo",  default=stage_repo(4, "deduped"))
    parser.add_argument("--output-repo", default=stage_repo(5, "quality"))
    parser.add_argument("--model", default="./artifacts/quality_classifier.bin")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Min P(high_quality) to keep")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"missing classifier model at {model_path}")
    log.info(f"loading classifier from {model_path}")
    clf = fasttext.load_model(str(model_path))

    log.info(f"pulling {args.input_repo}")
    ds = pull_dataset(args.input_repo)
    log.info(f"loaded {len(ds):,} docs")

    def keep(rec: dict) -> bool:
        if not needs_classifier(rec["source"]):
            return True
        text = rec["text"][:2000].replace("\n", " ")
        labels, probs = clf.predict(text, k=2)
        # find probability of high_quality label
        for lbl, p in zip(labels, probs):
            if lbl == "__label__high_quality":
                return p >= args.threshold
        return False

    survived = ds.filter(keep)
    log.info(f"survived: {len(survived):,} / {len(ds):,}")
    push_dataset(survived, args.output_repo)
    log.info("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run tests, verify pass**

Run: `python -m pytest tests/test_data_pipeline.py -v`

Expected: 26 PASSED.

- [ ] **Step 6.5: Commit**

```bash
git add scripts/data_pipeline/stage_5_fasttext_apply.py tests/test_data_pipeline.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): stage 5b apply quality classifier (hybrid policy)"
git push
```

---

## Task 7: Tokenizer fast wrapper + training script + tests

**Files:**
- Create: `tokenizer_fast.py`
- Create: `scripts/data_pipeline/stage_6a_train_tokenizer.py`
- Create: `tests/test_tokenizer_fast.py`

- [ ] **Step 7.1: Write tests for `tokenizer_fast.py`**

Create `tests/test_tokenizer_fast.py`:

```python
"""Tests for tokenizer_fast.py. Skipped if tokenizer_fast.json does not exist
(it is produced by stage_6a)."""
import os
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
```

- [ ] **Step 7.2: Implement `tokenizer_fast.py`**

```python
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
```

- [ ] **Step 7.3: Implement `scripts/data_pipeline/stage_6a_train_tokenizer.py`**

```python
"""Stage 6a: Train BPE tokenizer on a 1 GB representative sample.

Output: tokenizer_fast.json (committed to git)."""
from __future__ import annotations
import argparse
import random
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

from .common import pull_dataset, stage_repo, get_logger, TARGETS

log = get_logger("stage6a")

SAMPLE_BYTES_PER_CATEGORY = {
    "python":        300_000_000,
    "math":          200_000_000,
    "english_prose": 300_000_000,
    "sexp":          100_000_000,
    "pt_br":         100_000_000,
}
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|cdsl_start|>",
    "<|cdsl_end|>",
    "<|tool_call|>",
    "<|tool_result|>",
]


def sample_corpus(input_repo: str, out_path: Path) -> None:
    """Sample ~1 GB of representative text proportional to mix C."""
    ds = pull_dataset(input_repo)
    log.info(f"loaded {len(ds):,} docs for sampling")
    rng = random.Random(42)

    by_cat: dict[str, list[int]] = {c: [] for c in TARGETS}
    for i, rec in enumerate(ds):
        by_cat[rec["category"]].append(i)
    log.info({c: len(v) for c, v in by_cat.items()})

    out = open(out_path, "w", encoding="utf-8")
    for cat, budget in SAMPLE_BYTES_PER_CATEGORY.items():
        rng.shuffle(by_cat[cat])
        consumed = 0
        for idx in by_cat[cat]:
            if consumed >= budget:
                break
            text = ds[idx]["text"]
            out.write(text + "\n")
            consumed += len(text)
        log.info(f"sampled {consumed:,} bytes from {cat}")
    out.close()


def train_tokenizer(corpus_path: Path, out_path: Path) -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )

    log.info(f"training BPE on {corpus_path} ...")
    tokenizer.train([str(corpus_path)], trainer)
    log.info(f"trained, vocab size = {tokenizer.get_vocab_size()}")
    tokenizer.save(str(out_path))
    log.info(f"saved tokenizer to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", default=stage_repo(5, "quality"))
    parser.add_argument("--workdir", default="./artifacts")
    parser.add_argument("--out", default="tokenizer_fast.json")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    corpus_path = workdir / "tokenizer_train_sample.txt"

    sample_corpus(args.input_repo, corpus_path)
    train_tokenizer(corpus_path, Path(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.4: Confirm tokenizer tests SKIP (because no tokenizer_fast.json yet)**

Run: `python -m pytest tests/test_tokenizer_fast.py -v`

Expected: 3 SKIPPED with reason "tokenizer_fast.json not yet trained". This confirms the test file is correctly structured.

- [ ] **Step 7.5: Commit**

```bash
git add tokenizer_fast.py scripts/data_pipeline/stage_6a_train_tokenizer.py tests/test_tokenizer_fast.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): tokenizer_fast.py + stage 6a BPE training script + tests"
git push
```

---

## Task 8: Stage 6b — Tokenize the corpus

**Files:**
- Create: `scripts/data_pipeline/stage_6b_tokenize.py`

This stage has no pure functions worth unit-testing on its own (it just calls our wrapper from Task 7 and pushes a parquet). Unit-test the wrapper, integration-test by running.

- [ ] **Step 8.1: Implement `scripts/data_pipeline/stage_6b_tokenize.py`**

```python
"""Stage 6b: Tokenize all surviving docs using tokenizer_fast.json.

Output: HF dataset with a single `tokens` column (List[int]) per doc.
"""
from __future__ import annotations
import argparse

from .common import pull_dataset, push_dataset, stage_repo, get_logger
from tokenizer_fast import FastBPETokenizer

log = get_logger("stage6b")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", default=stage_repo(5, "quality"))
    parser.add_argument("--output-repo", default=stage_repo(6, "tokenized"))
    parser.add_argument("--tokenizer", default="tokenizer_fast.json")
    parser.add_argument("--num-proc", type=int, default=4)
    args = parser.parse_args()

    tok = FastBPETokenizer(args.tokenizer)
    log.info(f"loaded tokenizer (vocab {tok.vocab_size})")

    ds = pull_dataset(args.input_repo)
    log.info(f"loaded {len(ds):,} docs")

    def encode(rec):
        rec["tokens"] = tok.encode(rec["text"])
        rec["n_tokens"] = len(rec["tokens"])
        return rec

    log.info("tokenizing ...")
    ds = ds.map(encode, num_proc=args.num_proc)

    # Remove the big raw text column; keep only what Phase 1B needs
    keep_cols = {"tokens", "n_tokens", "category", "source", "doc_id"}
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(drop_cols)

    total_tokens = sum(ds["n_tokens"])
    log.info(f"total tokens: {total_tokens:,}")
    log.info(f"pushing to {args.output_repo}")
    push_dataset(ds, args.output_repo)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Commit**

```bash
git add scripts/data_pipeline/stage_6b_tokenize.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): stage 6b tokenize full corpus"
git push
```

---

## Task 9: Stage 7 — Shard + upload to Drive

**Files:**
- Create: `scripts/data_pipeline/stage_7_shard_upload.py`

This script runs in Colab so `google.colab.drive` is available. We import it lazily so the script remains import-clean on Codespaces.

- [ ] **Step 9.1: Implement `scripts/data_pipeline/stage_7_shard_upload.py`**

```python
"""Stage 7: Shard the tokenized corpus into nanoGPT-style .bin files and
upload to Google Drive.

Run on Google Colab (drive.mount works natively there).
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np

from .common import pull_dataset, stage_repo, get_logger, TARGETS

log = get_logger("stage7")

TOKENS_PER_SHARD = 100_000_000   # 100M tokens × 2 bytes = ~200 MB shards
VAL_FRACTION = 0.05               # 5% reserved for validation


def shard_arrays(
    all_tokens: np.ndarray,
    shard_size: int,
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(all_tokens) + shard_size - 1) // shard_size
    for i in range(n_shards):
        chunk = all_tokens[i * shard_size : (i + 1) * shard_size]
        path = out_dir / f"{prefix}_{i:03d}.bin"
        chunk.astype(np.uint16).tofile(path)
        paths.append(path)
        log.info(f"  wrote {path.name} ({len(chunk):,} tokens, {path.stat().st_size:,} bytes)")
    return paths


def category_breakdown(ds) -> dict[str, int]:
    """Count tokens per category in a HF Dataset."""
    counts: dict[str, int] = {c: 0 for c in TARGETS}
    for rec in ds:
        counts[rec["category"]] += rec["n_tokens"]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", default=stage_repo(6, "tokenized"))
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/cognitive-kernel/data",
        help="Target Drive folder for shards + manifest",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = pull_dataset(args.input_repo)
    log.info(f"loaded {len(ds):,} docs")

    cat_breakdown = category_breakdown(ds)
    log.info(f"per-category token counts: {cat_breakdown}")

    log.info("concatenating all tokens ...")
    all_tokens: list[int] = []
    rng = random.Random(args.seed)
    # Shuffle doc order for better mix
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for i in indices:
        all_tokens.extend(ds[i]["tokens"])
    all_tokens_np = np.asarray(all_tokens, dtype=np.uint16)
    log.info(f"total tokens: {len(all_tokens_np):,}")

    n_val = int(len(all_tokens_np) * VAL_FRACTION)
    n_train = len(all_tokens_np) - n_val
    train_arr = all_tokens_np[:n_train]
    val_arr = all_tokens_np[n_train:]
    log.info(f"split: train={n_train:,}, val={n_val:,}")

    drive_root = Path(args.drive_root)
    shards_dir = drive_root / "shards"
    train_paths = shard_arrays(train_arr, TOKENS_PER_SHARD, shards_dir, "train")
    val_paths = shard_arrays(val_arr, TOKENS_PER_SHARD, shards_dir, "val")

    manifest = {
        "total_tokens": int(len(all_tokens_np)),
        "train_tokens": int(n_train),
        "val_tokens":   int(n_val),
        "n_train_shards": len(train_paths),
        "n_val_shards": len(val_paths),
        "tokens_per_shard": TOKENS_PER_SHARD,
        "category_breakdown": cat_breakdown,
        "seed": args.seed,
        "produced_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    manifest_path = drive_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"wrote manifest to {manifest_path}")
    log.info("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Commit**

```bash
git add scripts/data_pipeline/stage_7_shard_upload.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): stage 7 shard + upload to Drive"
git push
```

---

## Task 10: `validate_phase_1a.py` + tiny config

**Files:**
- Create: `configs/tiny_signal_test.yaml`
- Create: `scripts/data_pipeline/validate_phase_1a.py`

- [ ] **Step 10.1: Create `configs/tiny_signal_test.yaml`**

This config is used by DoD criterion #6 to train a 10M-param model briefly on real data and verify loss decreases.

```yaml
model:
  n_embd: 256
  n_head: 8
  n_kv_head: 2
  n_layer: 4
  max_seq_len: 512
  vocab_size: 32000
  grad_checkpoint: false

train:
  batch_size: 8
  grad_accum_steps: 4
  block_size: 512
  max_iters: 2000
  learning_rate: 1.0e-3
  warmup_iters: 100
  min_lr: 1.0e-4
  weight_decay: 0.1
  grad_clip: 1.0
  eval_interval: 200
  eval_iters: 10
  checkpoint_interval: 500
  optimizer: adamw

data:
  train_shards_glob: "/content/drive/MyDrive/cognitive-kernel/data/shards/train_*.bin"
  val_shards_glob:   "/content/drive/MyDrive/cognitive-kernel/data/shards/val_*.bin"
  seed: 42

log:
  project: cognitive-kernel
  run_name: phase_1a_signal_test
  log_interval: 50
  wandb_mode: offline
```

- [ ] **Step 10.2: Implement `scripts/data_pipeline/validate_phase_1a.py`**

```python
"""Run the Phase 1A Definition of Done checks (DoD #4 sanity + #5 loader)."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np

from .common import get_logger
from tokenizer_fast import FastBPETokenizer

log = get_logger("validate")


def check_files_present(drive_root: Path) -> bool:
    shards = list((drive_root / "shards").glob("*.bin"))
    train = [p for p in shards if p.name.startswith("train_")]
    val = [p for p in shards if p.name.startswith("val_")]
    log.info(f"shards: train={len(train)}, val={len(val)}")
    manifest = drive_root / "manifest.json"
    log.info(f"manifest present: {manifest.exists()}")
    return len(train) > 0 and len(val) > 0 and manifest.exists()


def sanity_decode_samples(drive_root: Path, tokenizer_path: Path) -> None:
    tok = FastBPETokenizer(tokenizer_path)
    shards = list((drive_root / "shards").glob("train_*.bin"))
    random.shuffle(shards)
    for shard in shards[:3]:
        data = np.fromfile(shard, dtype=np.uint16)
        for _ in range(3):
            offset = random.randint(0, len(data) - 200)
            sample = data[offset:offset + 200].tolist()
            text = tok.decode(sample)
            log.info(f"  [{shard.name} @ {offset}]: {text[:120]}...")


def check_dataset_loader(drive_root: Path) -> None:
    """Use Phase 0's ShardedTokenDataset to verify the loader works."""
    from dataset import ShardedTokenDataset
    glob = str(drive_root / "shards" / "train_*.bin")
    ds = ShardedTokenDataset(glob_pattern=glob, block_size=512, seed=42)
    x, y = ds.get_batch(batch_size=4)
    assert x.shape == (4, 512), f"unexpected shape: {x.shape}"
    assert y.shape == (4, 512)
    log.info(f"loader OK; x.shape={x.shape}, x[0,:10]={x[0,:10].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/cognitive-kernel/data",
    )
    parser.add_argument("--tokenizer", default="tokenizer_fast.json")
    args = parser.parse_args()

    drive_root = Path(args.drive_root)

    log.info("=== DoD #3: files present ===")
    assert check_files_present(drive_root), "shards or manifest missing"

    log.info("=== DoD #2: manifest content ===")
    manifest = json.loads((drive_root / "manifest.json").read_text())
    log.info(json.dumps(manifest, indent=2))

    log.info("=== DoD #4: sanity decode samples ===")
    sanity_decode_samples(drive_root, Path(args.tokenizer))

    log.info("=== DoD #5: dataset loader ===")
    check_dataset_loader(drive_root)

    log.info("All Phase 1A validation checks passed (DoD #6 signal test runs separately).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.3: Commit**

```bash
git add configs/tiny_signal_test.yaml scripts/data_pipeline/validate_phase_1a.py
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "feat(phase-1a): tiny_signal_test config + validate_phase_1a.py DoD runner"
git push
```

---

## Task 11: Execute the pipeline (manual, multi-session)

**This task is the user's runbook. No subagent can do this — it requires HuggingFace login, Kaggle session, Colab session, and Drive auth.** Instructions below are precise so you can execute them one session at a time.

- [ ] **Step 11.1: HuggingFace setup**

In any browser:
1. https://huggingface.co/join — create account if needed, verify email
2. https://huggingface.co/settings/tokens — create **WRITE** token, save it
3. Apply for access (one-click "Agree") to each gated dataset (some may be ungated):
   - bigcode/the-stack-v2-dedup
   - HuggingFaceFW/fineweb-edu
   - EleutherAI/proof-pile-2
   - cc100
   - Others as needed

- [ ] **Step 11.2: Kaggle Session 1 — Stages 1, 2, 3**

In Kaggle Notebooks:
```python
!git clone https://github.com/Jeova-Luks/cognitive-kernel.git
%cd cognitive-kernel
!pip install -r requirements.txt

import os
os.environ["HF_USER"]  = "Jeova-Luks"
os.environ["HF_TOKEN"] = "hf_xxxYOURTOKENxxx"
!huggingface-cli login --token $HF_TOKEN

# Stage 1
!python -m scripts.data_pipeline.stage_1_download
# Stage 2
!python -m scripts.data_pipeline.stage_2_normalize
# Stage 3
!python -m scripts.data_pipeline.stage_3_heuristics
```

Expected: ~10h. End state: `Jeova-Luks/ck-stage-3-filtered` exists on HF Hub.

- [ ] **Step 11.3: Kaggle Session 2 — Stage 4 (MinHash)**

```python
!git clone https://github.com/Jeova-Luks/cognitive-kernel.git
%cd cognitive-kernel
!pip install -r requirements.txt
import os; os.environ["HF_USER"]="Jeova-Luks"; os.environ["HF_TOKEN"]="hf_..."
!huggingface-cli login --token $HF_TOKEN
!python -m scripts.data_pipeline.stage_4_minhash
```

Expected: ~8h. End state: `Jeova-Luks/ck-stage-4-deduped`.

- [ ] **Step 11.4: Codespaces Session — Stage 5 (FastText train + apply)**

In your Codespace terminal:
```bash
git pull
export HF_USER=Jeova-Luks
export HF_TOKEN=hf_...
huggingface-cli login --token $HF_TOKEN
mkdir -p artifacts
python -m scripts.data_pipeline.stage_5_fasttext_train --out-dir ./artifacts
python -m scripts.data_pipeline.stage_5_fasttext_apply --model ./artifacts/quality_classifier.bin
```

Expected: ~3h. End state: `Jeova-Luks/ck-stage-5-quality`.

- [ ] **Step 11.5: Codespaces Session — Stage 6a (train tokenizer)**

```bash
python -m scripts.data_pipeline.stage_6a_train_tokenizer
# tokenizer_fast.json now exists at repo root
python -m pytest tests/test_tokenizer_fast.py -v
```

Expected: 3 PASSED.

Commit:
```bash
git add tokenizer_fast.json
git commit -m "data(phase-1a): trained tokenizer_fast.json (vocab 32k)"
git push
```

- [ ] **Step 11.6: Kaggle Session 3 — Stage 6b (tokenize)**

```python
!git pull
!python -m scripts.data_pipeline.stage_6b_tokenize
```

Expected: ~6h. End state: `Jeova-Luks/ck-stage-6-tokenized`.

- [ ] **Step 11.7: Colab Session — Stage 7 (shard + upload)**

In a new Colab notebook (GPU not needed):
```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/Jeova-Luks/cognitive-kernel.git
%cd cognitive-kernel
!pip install -r requirements.txt
import os; os.environ["HF_USER"]="Jeova-Luks"; os.environ["HF_TOKEN"]="hf_..."
!huggingface-cli login --token $HF_TOKEN

!python -m scripts.data_pipeline.stage_7_shard_upload \
    --drive-root /content/drive/MyDrive/cognitive-kernel/data
```

Expected: ~3h. End state: `MyDrive/cognitive-kernel/data/shards/*.bin` + `manifest.json`.

---

## Task 12: Run validation + DoD signal test

- [ ] **Step 12.1: Run validate_phase_1a.py in Colab**

```python
!python -m scripts.data_pipeline.validate_phase_1a \
    --drive-root /content/drive/MyDrive/cognitive-kernel/data \
    --tokenizer tokenizer_fast.json
```

Expected output (abbreviated):
```
=== DoD #3: files present ===
shards: train=95, val=5
manifest present: True
=== DoD #2: manifest content ===
{ ... total_tokens: ~10000000000 ... }
=== DoD #4: sanity decode samples ===
  [train_007.bin @ 12345]: def search(arr, target): for i in range(len(arr))...
  ...
=== DoD #5: dataset loader ===
loader OK; x.shape=torch.Size([4, 512]), x[0,:10]=[...]
All Phase 1A validation checks passed
```

If any check fails, fix and re-run.

- [ ] **Step 12.2: Run DoD #6 signal test in Colab**

```python
from pathlib import Path
from config import load_config
from trainer import Trainer
cfg = load_config(Path("configs/tiny_signal_test.yaml"))
trainer = Trainer(cfg, output_dir=Path("/content/drive/MyDrive/cognitive-kernel/checkpoints/signal_test"))
print(f"params: {sum(p.numel() for p in trainer.model.parameters()):,}")
trainer.fit()
```

Expected: model has ~10M params, trains for 2000 steps, **final train_loss ≤ 7.0**.

If loss does not drop below 7.0 by step 2000, the data has insufficient signal — investigate the manifest or sample more docs from un-filtered sources.

- [ ] **Step 12.3: Commit the manifest.json contents in the repo for posterity**

In Codespaces:
```bash
mkdir -p data
gsutil cp gs://... data/manifest.json 2>/dev/null || \
   curl -L "https://drive.google.com/uc?id=..." -o data/manifest.json || \
   echo "manifest.json must be manually downloaded from Drive and placed at data/manifest.json"
# (Whichever way works; the manifest is small.)
git add data/manifest.json
git commit -m "data(phase-1a): record manifest.json"
git push
```

(Alternative: just download `manifest.json` from Drive in your browser, paste into the repo, commit.)

---

## Task 13: Final wrap-up

- [ ] **Step 13.1: Update README**

Edit `README.md`. Find the line `🟡 **Phase 1** — Pre-train 100M base on curated data` and replace it with:

```markdown
- ✅ **Phase 1A** — Data pipeline complete; ~10 B tokens shard on Drive
- 🟡 **Phase 1B** — Pre-train 100M base (next; spec TBD)
```

Also add a "Phase 1A status" section:

```markdown
## Phase 1A — Data Pipeline Status

Manifest: [data/manifest.json](data/manifest.json)

Final corpus: ~10 B tokens across 5 categories (30% Python / 20% math /
30% English prose / 10% sexp / 10% PT-BR), all natural data, filtered
through L1 heuristics + L2 MinHash near-dedup + L3 selective FastText
quality classifier. Tokenized with custom BPE vocab 32 000 including
five CDSL special tokens. Sharded as nanoGPT-format uint16 binary files
on `MyDrive/cognitive-kernel/data/shards/`.

DoD #6 signal validation: 10 M-param model trained on the real shards
reached `train_loss ≤ 7.0` within 2 000 steps.
```

- [ ] **Step 13.2: Commit + tag the release**

```bash
git add README.md
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" commit -m "docs: Phase 1A complete; manifest summary in README"
git -c user.email="j.eudesmdm@hotmail.com" -c user.name="LLMPessoal" tag -a phase-1a-complete -m "Phase 1A: Data pipeline complete. ~10B tokens, custom BPE 32k, full L3 filtering. Manifest in data/manifest.json. Ready for Phase 1B pre-training."
git push --tags
```

---

## Phase 1A Definition of Done (recap)

All 8 criteria from the spec, mapped to tasks:

1. ✅ `tokenizer_fast.json` committed + tests passing — Task 7 + Task 11.5
2. ✅ `data/manifest.json` documents counts — Task 12.3
3. ✅ ~95 train + ~5 val shards on Drive — Task 11.7
4. ✅ Sanity decode samples coherent — Task 12.1 DoD #4
5. ✅ `ShardedTokenDataset` loads production shards — Task 12.1 DoD #5
6. ✅ 10M model train_loss ≤ 7.0 within 2 000 steps — Task 12.2
7. ✅ Pipeline scripts in `scripts/data_pipeline/` — Tasks 0-10
8. ✅ README updated — Task 13.1

**Next:** Phase 1B plan — pre-training the 100M base model. Separate spec and plan to be written when Phase 1A is complete.

---

**END OF PHASE 1A IMPLEMENTATION PLAN**
