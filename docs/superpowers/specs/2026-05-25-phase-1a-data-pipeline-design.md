# Phase 1A — Data Pipeline Design Document

**Status:** DRAFT — awaiting user review
**Date:** 2026-05-25
**Author:** User (Brazilian undergraduate, federal university AI student). *Replace with your name when publishing.*
**Parent project:** [Cognitive Kernel v0.1](2026-05-24-cognitive-kernel-v0.1-design.md) — this is the data half of Phase 1.
**Predecessor:** [Phase 0 — Foundation Hardening](../plans/2026-05-24-phase-0-foundation-hardening.md) (complete, tag `phase-0-complete`)
**Successor:** Phase 1B — Pre-training the 100M base model (separate spec, to be written after 1A complete)
**Target environment:** GitHub Codespaces (dev) + Kaggle Notebooks (heavy lift) + HuggingFace Datasets (intermediate storage) + Google Drive 5 TB (final shards)
**Estimated timeline:** 1-2 weeks calendar, ~30-50 h hands-on

---

## 1. Context

Phase 0 produced a robust training infrastructure but the model has only ever seen random tokens. To produce a useful base model in Phase 1B, we need a **clean, deduplicated, quality-filtered, tokenized corpus** living in Google Drive in nanoGPT-style `.bin` shards.

Phase 1 was originally specified as monolithic (data + training). It naturally decomposes into:

- **Phase 1A (this document):** the data pipeline that produces `MyDrive/cognitive-kernel/data/shards/*.bin`
- **Phase 1B (future):** the pre-training run that consumes those shards

The two have a single clean interface (sharded binary files + a tokenizer JSON), and 1A can be fully validated before 1B begins.

---

## 2. Scope

### What this project IS

A multi-stage pipeline that:

1. Downloads ~60 GB of raw text from curated HuggingFace datasets
2. Normalizes Unicode and basic formatting
3. Applies heuristic quality filters (L1)
4. Applies MinHash-based near-deduplication (L2)
5. Applies a FastText quality classifier where appropriate (L3)
6. Trains a custom BPE tokenizer (vocab 32 000) on a representative sample
7. Tokenizes the entire filtered corpus
8. Splits 95% train / 5% val, shards into ~200 MB `.bin` files
9. Uploads final shards to Google Drive

Each stage is checkpoint-able via HuggingFace Datasets (private), so a failed Kaggle session does not waste prior work.

### What this project is NOT

- ❌ NOT the pre-training of the 100M model (that's Phase 1B)
- ❌ NOT a synthetic data generation effort (decided: 100% natural data; no API spend)
- ❌ NOT an instruction-tuning / chat-format dataset (that's a later fine-tuning phase)
- ❌ NOT a generic crawler (no Common Crawl raw download from scratch; we only consume curated HuggingFace-hosted datasets)
- ❌ NOT a PII-removal pipeline (decided: heuristics + MinHash + FastText is sufficient quality control for the project's research aims)
- ❌ NOT multilingual beyond PT-BR + English (no Chinese, Spanish, French, etc.)

### Token budget and mix (decided)

- **Total:** ~10 billion tokens (Phi-style 100:1 ratio for the 100M base model)
- **Mix C — Generalist with focus:**
  - 30% Python idiomatic code (~3.0 B)
  - 20% Math and step-by-step reasoning (~2.0 B)
  - 30% English technical prose (~3.0 B)
  - 10% S-expressions / structured text (~1.0 B)
  - 10% Portuguese (~1.0 B)

The rationale for skewing toward English prose (30%) over Python (30%) is that the Cognitive Kernel's bottleneck in Phase 3 will be natural-language understanding of inputs to compile into CDSL — not code generation per se. A model that reads English well is more valuable than one that writes more Python.

### Central claim (carry-over from parent spec)

This data pipeline does not, by itself, validate any thesis. It is the necessary **substrate** for the Phase 1B base model, which is in turn the substrate for the Cognitive Kernel proper. The phrasing of claims about the Cognitive Kernel remains as specified in the parent document:

> *"for verifiable domains, cognitive efficiency should be measured at the system level rather than the parameter level."*

---

## 3. Architecture — Seven-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: DOWNLOAD                                              │
│  HuggingFace datasets API → ~60 GB raw text                     │
│  Where: Kaggle (73 GB disk, ~10h)                               │
│  Output: HF Dataset Jeova-Luks/ck-stage-1-raw (parquet)         │
└─────────────────────────────────────┬───────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: NORMALIZE                                             │
│  Unicode NFC, strip HTML/excess markdown, language detection,   │
│  filter <50 or >1 M chars                                       │
│  Where: Kaggle, ~3h                                             │
│  Output: HF Dataset Jeova-Luks/ck-stage-2-normalized            │
└─────────────────────────────────────┬───────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: HEURISTIC FILTERS (L1)                                │
│  Length/composition/repetition/category-specific checks         │
│  Where: Kaggle, ~4h                                             │
│  Output: HF Dataset Jeova-Luks/ck-stage-3-filtered (~70% pass)  │
└─────────────────────────────────────┬───────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: MinHash NEAR-DEDUP (L2)                               │
│  datasketch MinHashLSH; threshold 0.85, num_perm 256            │
│  Where: Kaggle (30 GB RAM needed), ~8h                          │
│  Output: HF Dataset Jeova-Luks/ck-stage-4-deduped (~70% pass)   │
└─────────────────────────────────────┬───────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: FASTTEXT QUALITY CLASSIFIER (L3)                      │
│  Applied only to non-pre-filtered sources (CC100-PT, BrWac, JSON│
│  raw, Lisp/Scheme raw). Pre-filtered sources pass straight      │
│  through.                                                       │
│  Where: Codespaces, ~3h                                         │
│  Output: HF Dataset Jeova-Luks/ck-stage-5-quality (~85% pass)   │
└─────────────────────────────────────┬───────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 6a: TRAIN TOKENIZER                                      │
│  HF tokenizers BPE, vocab 32 000, on 1 GB representative sample │
│  Where: Codespaces, ~1h                                         │
│  Output: tokenizer_fast.json committed to repo                  │
└─────────────────────────────────────┬───────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 6b: TOKENIZE ALL                                         │
│  Apply tokenizer to all ~24 GB of filtered text                 │
│  Where: Kaggle (parallel via Rust), ~6h                         │
│  Output: HF Dataset Jeova-Luks/ck-stage-6-tokenized (uint16)    │
└─────────────────────────────────────┬───────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 7: SHARD + UPLOAD                                        │
│  Split 95% train / 5% val. Shard ~200 MB (100 M tokens) each.   │
│  Upload to Drive via google.colab.drive.mount                   │
│  Where: Colab Free, ~3h                                         │
│  Output: MyDrive/cognitive-kernel/data/shards/{train,val}_*.bin │
└─────────────────────────────────────┬───────────────────────────┘
                                      ↓
                          ✅ Phase 1A complete
```

**Total CPU-time: 25-40 hours.** Distributed across ~5-6 sessions over 1-2 weeks calendar.

---

## 4. Data sources (specific HuggingFace datasets)

### Python — 3.0 B target

| Dataset | HuggingFace ID | Raw size | Take |
|---|---|---|---|
| The Stack v2 dedup (Python subset) | `bigcode/the-stack-v2-dedup` | ~50 B Python | 2.5 B |
| CodeParrot clean | `codeparrot/codeparrot-clean` | ~10 B | 0.3 B |
| Python-edu (curated tutorials) | `bigcode/python-edu` | ~2 B | 0.2 B |

Extra filter for The Stack: `stars >= 5`, `lines >= 10`, drop docs containing markers of AI-generated code (`# AI-generated`, `# Copilot`, etc.).

### Math — 2.0 B target

| Dataset | HuggingFace ID | Take |
|---|---|---|
| Proof-Pile-2 (arXiv math + textbooks) | `EleutherAI/proof-pile-2` | 1.2 B |
| OpenMathInstruct | `nvidia/OpenMathInstruct-1` | 0.5 B |
| GSM8K + MetaMath augmentations | `gsm8k` + `meta-math/MetaMathQA` | 0.3 B |

### English technical prose — 3.0 B target

| Dataset | HuggingFace ID | Take |
|---|---|---|
| FineWeb-Edu (already classifier-filtered) | `HuggingFaceFW/fineweb-edu` (10 BT sample) | 2.5 B |
| Wikipedia STEM articles | `wikimedia/wikipedia` (en, category-filtered) | 0.5 B |

### S-expressions / structured — 1.0 B target

| Dataset | HuggingFace ID | Take |
|---|---|---|
| Lisp/Scheme/Racket/Clojure from The Stack | `bigcode/the-stack-v2-dedup` (lang filter) | 0.3 B |
| LaTeX source from arXiv | `EleutherAI/proof-pile-2` (latex subset) | 0.4 B |
| Lean theorem prover workbook | `internlm/Lean-Workbook` | 0.05 B |
| Curated JSON config (high-signal subset) | `bigcode/the-stack-v2-dedup` (lang=json, filtered) | 0.25 B |

### Portuguese — 1.0 B target

| Dataset | HuggingFace ID | Take |
|---|---|---|
| Wikipedia PT-BR | `wikimedia/wikipedia` (lang=pt) | 0.7 B |
| BrWac corpus | `nilc-nlp/BrWac` | 0.2 B |
| CC-100 PT-BR (heavily filtered Stage 5) | `cc100` (pt subset) | 0.1 B |

All datasets require accepting terms of use on HuggingFace Hub. Most are open / CC-BY-SA / MIT; user must verify license compatibility before any publication beyond a research preprint.

---

## 5. Tokenizer

### Configuration

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=[
        "<|endoftext|>",
        "<|cdsl_start|>",
        "<|cdsl_end|>",
        "<|tool_call|>",
        "<|tool_result|>",
    ],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    min_frequency=2,
)
```

### Training corpus (sampled, NOT the full 10 B)

A representative ~1 GB sample, proportional to mix C:
- ~300 MB Python
- ~200 MB math
- ~300 MB English prose
- ~100 MB sexp/structured
- ~100 MB PT-BR

Train time: ~30-60 min on Codespaces (4 cores). Output: `tokenizer_fast.json` (~10 MB), committed to the repo.

### Pure-Python BPE retained as documentation

The existing [`tokenizer.py`](../../../tokenizer.py) (pure-Python BPE, slow but pedagogically transparent) stays in the repository as a documentation artifact. Production use goes through a new module:

```python
# tokenizer_fast.py
from tokenizers import Tokenizer
from pathlib import Path

class FastBPETokenizer:
    def __init__(self, path: Path = Path("tokenizer_fast.json")):
        self.tokenizer = Tokenizer.from_file(str(path))
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids
    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
```

### Validation

Add `tests/test_tokenizer_fast.py` with:

1. **Roundtrip:** encode → decode on ~100 strings per category equals the original
2. **Special tokens:** the five CDSL markers tokenize as single token IDs in the 60 000 band as expected by the model config
3. **Compression ratio:** measure bytes/token on a held-out validation slice. Target ~3.5-4.0 on Python. If >5.0, the BPE training sample was poorly chosen — re-balance.

---

## 6. Filters (L1 + L2 + L3)

### Stage 3 — L1 heuristics

Applied per document, parallelized via `multiprocessing.Pool`. Reference implementation:

```python
def passes_heuristics(doc: str, category: str) -> bool:
    if not (50 <= len(doc) <= 1_000_000):
        return False
    lines = doc.split('\n')
    if not lines:
        return False
    mean_line_len = sum(len(l) for l in lines) / len(lines)
    if not (1 < mean_line_len <= 1000):
        return False
    if max(len(l) for l in lines) > 100_000:
        return False
    n_chars = len(doc)
    n_special = sum(1 for c in doc if not c.isalnum() and not c.isspace())
    n_whitespace = sum(1 for c in doc if c.isspace())
    if n_special / n_chars > 0.5:
        return False
    if n_whitespace / n_chars > 0.5:
        return False
    if len(lines) > 10:
        unique_lines_ratio = len(set(lines)) / len(lines)
        if unique_lines_ratio < 0.7:
            return False
    if category == "python":
        if not any(kw in doc for kw in ["def ", "class ", "import ", "= "]):
            return False
    elif category in ("english_prose", "pt_br"):
        from langdetect import detect_langs
        target = "en" if category == "english_prose" else "pt"
        try:
            langs = detect_langs(doc[:5000])
            if not any(l.lang == target and l.prob > 0.95 for l in langs):
                return False
        except Exception:
            return False
    return True
```

Plus exact deduplication via MD5 hash of the entire document.

Expected survival rate: ~70%.

### Stage 4 — L2 MinHash near-dedup

Library: `datasketch.MinHashLSH`. Parameters:
- `threshold=0.85` (Jaccard similarity above which two docs are considered "near-duplicates")
- `num_perm=256` (signature length)
- 5-gram word shingles

Memory: ~5-8 GB RAM for ~100 M documents. Kaggle's 30 GB is comfortable.

For each cluster of near-duplicates, retain the **longest** document (proxy for quality).

Expected survival rate: ~70% (varies by source; FineWeb-Edu loses ~5%, raw CC loses ~50%).

### Stage 5 — L3 FastText quality classifier

**Selective application** (hybrid approach):

| Source | Pre-filtered upstream? | Apply Stage 5? |
|---|---|---|
| FineWeb-Edu | Yes (HF Llama-3 classifier) | No |
| Proof-Pile-2 | Yes (curated) | No |
| The Stack v2 dedup | Partial | Yes, light threshold |
| python-edu | Yes | No |
| Wikipedia | Yes (editorial) | No |
| BrWac, CC100-PT | No | Yes, aggressive |
| Lisp/Scheme/JSON raw | No | Yes, medium |

Classifier training:
- ~50 K positives: Wikipedia FA articles, top-starred GitHub Python, high-score StackOverflow answers, open math textbooks
- ~50 K negatives: random Common Crawl docs, GitHub repos with 0 stars and no README, regex-detected spam
- FastText supervised: 50 epochs, lr=0.5, dim=100, ~30 min training

Inference threshold: score ≥ 0.5 passes. Tunable.

Expected survival rate on the non-pre-filtered fraction: ~85%.

### Combined funnel

| Stage | Input | Survival | Output cumulative |
|---|---|---|---|
| 1. Download | — | — | ~60 GB raw |
| 2. Normalize | 60 GB | ~95% | ~57 GB |
| 3. L1 heuristics | 57 GB | ~70% | ~40 GB |
| 4. L2 MinHash | 40 GB | ~70% | ~28 GB |
| 5. L3 FastText (partial) | 28 GB | ~85% | ~24 GB curated text |
| 6. Tokenize | 24 GB text | uint16 packing | ~10 B tokens uint16 |
| 7. Shard + upload | 10 B tokens | shards ~100 M each | ~100 shards on Drive |

---

## 7. Orchestration

Six sessions distributed across three platforms:

| Session | Platform | Duration | Stages | Output |
|---|---|---|---|---|
| 1 | Kaggle | ~10 h | 1, 2, 3 | `ck-stage-3-filtered` on HF |
| 2 | Kaggle | ~10 h | 4 | `ck-stage-4-deduped` on HF |
| 3 | Codespaces | ~3 h | 5 (classifier train + apply) | `ck-stage-5-quality` on HF |
| 4 | Codespaces | ~1 h | 6a (train tokenizer) | `tokenizer_fast.json` in git |
| 5 | Kaggle | ~6 h | 6b (tokenize all) | `ck-stage-6-tokenized` on HF |
| 6 | Colab Free | ~3 h | 7 (shard + upload) | `MyDrive/.../data/shards/*.bin` |

Each session is **idempotent and checkpoint-able**: if it fails halfway, re-running starts from the prior stage's HuggingFace dataset, not from scratch.

### Final layout

```
MyDrive/cognitive-kernel/
├── data/
│   ├── shards/
│   │   ├── train_000.bin    # ~200 MB, 100 M tokens uint16
│   │   ├── train_001.bin
│   │   ├── ...              # ~95 train shards = 9.5 B tokens
│   │   ├── train_094.bin
│   │   ├── val_000.bin
│   │   ├── ...              # ~5 val shards = 0.5 B tokens
│   │   └── val_004.bin
│   └── manifest.json        # per-shard + per-category token counts, filter stats
└── tokenizer_fast.json      # mirror of the repo file for reproducibility
```

`manifest.json` schema:
```json
{
  "total_tokens": 10000000000,
  "train_tokens": 9500000000,
  "val_tokens":   500000000,
  "category_breakdown": {
    "python":        3000000000,
    "math":          2000000000,
    "english_prose": 3000000000,
    "sexp":          1000000000,
    "pt_br":         1000000000
  },
  "filter_stats": {
    "stage_1_input_docs":   …,
    "stage_3_l1_survived":  …,
    "stage_4_l2_survived":  …,
    "stage_5_l3_survived":  …
  },
  "tokenizer_sha256": "…",
  "produced_at": "2026-06-XX",
  "git_sha": "…"
}
```

---

## 8. Definition of Done

Phase 1A is complete when **all** of the following are true:

1. ✅ `tokenizer_fast.json` committed to the repo with three unit tests passing in `tests/test_tokenizer_fast.py` (roundtrip, special tokens, compression ratio)
2. ✅ `data/manifest.json` on Drive documents token counts per shard, per category, and per filter stage
3. ✅ ~95 train shards + ~5 val shards present in `MyDrive/cognitive-kernel/data/shards/`
4. ✅ **Sanity smoke test:** load three random shards, decode ten random samples each, confirm visually that text is coherent (not corrupted bytes or repeating garbage)
5. ✅ **Loader smoke test:** instantiate `ShardedTokenDataset` (from Phase 0) pointing at the production shards; verify batches have correct shapes and seed-determinism still works
6. ✅ **Signal validation:** train a tiny 10 M-parameter model (using a modified `configs/test_toy.yaml` pointing at the production shards) for 2 000 steps. **Loss must fall from ~10 to ≤7.** This proves the data contains learnable signal — contrast with Phase 0's smoke test where loss stayed at 10.4 on random data.
7. ✅ Pipeline scripts versioned in `scripts/data_pipeline/` and committed
8. ✅ README updated to reflect Phase 1A completion and link to manifest

---

## 9. Phases and timeline

| Phase | Duration | Outcome |
|---|---|---|
| Day 1-2 | Stage 1+2 setup, dataset download begin | First HF dataset upload |
| Day 3-4 | Stage 3 heuristics + dedup development | Code reviewed, tested on subset |
| Day 5-7 | Stage 4 MinHash full run (slow) | `ck-stage-4-deduped` ready |
| Day 8 | Stage 5 FastText classifier train + apply | `ck-stage-5-quality` ready |
| Day 9 | Stage 6a tokenizer training + tests | `tokenizer_fast.json` committed |
| Day 10-11 | Stage 6b tokenization | `ck-stage-6-tokenized` ready |
| Day 12 | Stage 7 sharding + Drive upload | Final shards online |
| Day 13-14 | DoD validation (sanity + signal test) | Phase 1A signed off |

**Total: ~2 weeks at ~3-5 h/day, or one focused week at 8 h/day.**

---

## 10. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kaggle session disconnects during MinHash | 30% | Medium | Process in 4 chunks of ~25%; checkpoint each chunk to HF Datasets |
| HuggingFace Datasets quota exhausted | 10% | Low | Verified accounts get effectively unlimited storage for sub-1 TB; we're well under |
| Tokenizer trains poorly (unbalanced vocab) | 20% | High | Validate compression ratio < 5.0 bytes/token; if higher, rebalance training sample |
| FastText classifier has high false-positive rate | 30% | Medium | Threshold is tunable post-training; manually inspect 100 borderline docs to calibrate |
| Stage 5 lets junk PT-BR through | 25% | Low | If validation set looks bad, halve PT-BR allocation to 0.5 B |
| Final corpus < 10 B (filters too aggressive) | 25% | Medium | Plan B: ingest additional ~20 GB upstream (FineWeb-Edu has 1.3 T to spare) |
| Disk on Kaggle fills during Stage 4 | 15% | Medium | Process in 5 GB chunks; clean intermediate before next chunk |
| Tokenization produces a different total than 10 B | 50% | Low | Acceptable — anywhere 8-12 B is fine; manifest reflects actual count |
| Drive upload from Colab is slow (~10 MB/s) | 60% | Low | Upload final shards is ~20 GB → ~30-40 min. Acceptable. |
| Some dataset requires HF access approval that's slow | 40% | Low | Pre-apply to all gated datasets on Day 1; check approvals on Day 3 |

---

## 11. Repository layout after Phase 1A

```
LLMPessoal/
├── README.md                          # updated for Phase 1A
├── requirements.txt                   # adds tokenizers, datasketch, fasttext, langdetect, ...
├── docs/superpowers/
│   ├── specs/
│   │   ├── 2026-05-24-cognitive-kernel-v0.1-design.md
│   │   └── 2026-05-25-phase-1a-data-pipeline-design.md   # this document
│   └── plans/
│       ├── 2026-05-24-phase-0-foundation-hardening.md
│       └── 2026-05-25-phase-1a-data-pipeline.md          # to be written next
├── model.py
├── tokenizer.py                       # pure-Python BPE, didactic
├── tokenizer_fast.py                  # NEW: HF tokenizers wrapper
├── tokenizer_fast.json                # NEW: trained BPE vocab + merges
├── config.py
├── dataset.py
├── trainer.py
├── resume.py
├── server.py
├── static/
├── configs/
│   ├── base_100m.yaml
│   ├── test_toy.yaml
│   ├── smoke_100m.yaml
│   └── tiny_signal_test.yaml          # NEW: ~10M model config for DoD #6
├── scripts/
│   ├── _gen_notebook.py
│   ├── train_colab.ipynb
│   ├── make_smoke_shards.py
│   └── data_pipeline/                 # NEW directory
│       ├── stage_1_download.py
│       ├── stage_2_normalize.py
│       ├── stage_3_heuristics.py
│       ├── stage_4_minhash.py
│       ├── stage_5_fasttext_train.py
│       ├── stage_5_fasttext_apply.py
│       ├── stage_6a_train_tokenizer.py
│       ├── stage_6b_tokenize.py
│       ├── stage_7_shard_upload.py
│       └── validate_phase_1a.py       # runs DoD checks
├── tests/
│   ├── ... (Phase 0 tests)
│   └── test_tokenizer_fast.py         # NEW
└── data/                              # gitignored, ephemeral
    └── shards/                        # only on Kaggle/Colab; mirrors to Drive
```

---

## 12. Out-of-scope follow-ups

Explicitly **deferred**, not abandoned:

- **PII redaction:** if Cognitive Kernel becomes a public-facing system in Phase 4+, we add scrubbing for emails / SSN-like strings / phone numbers. For research base, current filtering is sufficient.
- **Profanity / toxicity filter:** category-specific, not a base-model concern.
- **Multilingual expansion** beyond PT-BR: Spanish, French, etc. could be added later if the Cognitive Kernel proves useful enough to justify localization.
- **Synthetic data augmentation:** could be added between Stage 5 and Stage 6 if API budget materializes (academic credit grants).
- **Continual data updates:** Phase 1A produces a one-shot corpus. A "data refresh" mechanism for periodic re-training is out of scope.

---

## 13. References

- HuggingFace `datasets` library — https://huggingface.co/docs/datasets
- HuggingFace `tokenizers` library — https://github.com/huggingface/tokenizers
- `datasketch` (MinHash LSH) — https://ekzhu.com/datasketch/
- `fasttext` — https://fasttext.cc/
- The Stack v2 paper — Lozhkov et al., 2024
- FineWeb-Edu — HuggingFace Smol team, 2024
- Proof-Pile-2 — Azerbayev et al., 2023
- Phi-1 — Gunasekar et al., 2023, *Textbooks Are All You Need*
- nanoGPT (data format reference) — Karpathy, 2023

---

## 14. Acknowledgements

Pipeline design and stage decomposition by [User] in dialog with Claude. Quality filter strategy follows Phi-1 with adaptations for the project's data sources and budget. The token mix proportions and corpus targets are project-specific and reflect Mix C (Generalist with focus) selected during brainstorming.

---

**END OF PHASE 1A DESIGN DOCUMENT (v0.1 DRAFT)**
