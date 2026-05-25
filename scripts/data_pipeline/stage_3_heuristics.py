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
