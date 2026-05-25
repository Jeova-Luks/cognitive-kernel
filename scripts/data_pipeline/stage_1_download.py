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
