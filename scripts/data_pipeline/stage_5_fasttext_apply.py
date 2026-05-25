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
