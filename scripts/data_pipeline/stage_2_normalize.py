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
