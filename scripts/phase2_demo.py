"""Phase 2 demo: one problem -> 7-stage Cognitive Kernel pipeline -> answer, with full trace.

Definition of done for Phase 2 (spec §9): "input GSM8K problem -> 7-stage
pipeline -> answer (any quality)". The 100M base model is not trained yet, so
a local OpenAI-compatible LLM (default: LM Studio + Llama 3.1 8B Instruct)
stands in as classifier and compiler.

Usage
-----
    lms server start && lms load meta-llama-3.1-8b-instruct
    python scripts/phase2_demo.py "Tom has 3 boxes of 12 pens. He gives away 7. How many are left?"
    python scripts/phase2_demo.py --target python "..."     # emit Python instead of CDSL
    python scripts/phase2_demo.py --no-plan --no-cross-check "..."   # the bare compile-run-repair loop

For benchmarks see scripts/phase2_eval.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel import CognitiveKernel, LLMError, LocalLLM, TypedMemory, format_trace  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("problem")
    p.add_argument("--target", choices=("cdsl", "python"), default="cdsl")
    p.add_argument("--no-plan", dest="plan", action="store_false", help="compile straight from the problem")
    p.add_argument("--no-cross-check", dest="cross_check", action="store_false",
                   help="skip the independent CoT derivation and the referee")
    p.add_argument("--self-checks", action="store_true", help="require assert/verify inside programs")
    p.add_argument("--referee", choices=("rule", "choose", "resolve"), default="rule")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--no-embed", action="store_true", help="bag-of-words memory instead of embeddings")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--embed-model")
    args = p.parse_args()
    try:
        llm = LocalLLM(base_url=args.base_url, model=args.model, embed_model=args.embed_model)
        if llm.model not in llm.models():
            sys.exit(f"Model '{llm.model}' is not served at {llm.base_url}")
        nomic = "nomic" in llm.embed_model
        memory = TypedMemory(embedder=None if args.no_embed else llm.embed,
                             query_prefix="search_query: " if nomic else "",
                             doc_prefix="search_document: " if nomic else "")
        kernel = CognitiveKernel(llm, memory=memory, max_attempts=args.max_attempts,
                                 target=args.target, plan=args.plan, self_checks=args.self_checks,
                                 cross_check=args.cross_check, referee=args.referee)
        r = kernel.solve(args.problem)
        if r.plan:
            print("--- plan ---\n" + r.plan)
        for a in r.attempts:
            print(f"\n--- attempt {a.index} ({a.compiled.mode}) program ---\n"
                  f"{a.compiled.program_text or a.compiled.raw}")
            if a.run is not None and a.run.trace:
                print("--- runtime trace ---\n" + "\n".join("  " + t for t in a.run.trace))
            if a.verdict.error_trace:
                print("--- verifier ---\n" + a.verdict.error_trace)
        print("\n=== pipeline ===\n" + format_trace(r))
        if r.arbitration is not None and r.arbitration.referee:
            print("\n--- third attempt (referee) ---\n" + r.arbitration.referee.text.strip())
        print(f"\nLLM calls: {llm.calls}, completion tokens: {llm.completion_tokens}")
    except LLMError as e:
        sys.exit(f"LLM error: {e}")


if __name__ == "__main__":
    main()
