"""Experiment 2026-11 (twin test and second model): collect the raw pieces.

This script only asks the model and stores what it wrote. It keeps nothing
but the id and the text of each problem, so nothing it prints or saves
depends on a label. The conditions
of the experiment are computed afterwards from these pieces by
``scripts/exp_twin_report.py``. The design, the primary comparison and the
predictions are in ``experiments/2026-11-twin/PREREGISTRATION.md``.

Pieces collected for every problem, with one model:

    cot      plain chain of thought on the problem
    read     classifier, memory notes and the reader's plan
    first    the plan compiled to CDSL and executed (today's kernel)
    calc     chain of thought with a calculator (kernel/calc_cot.py)      --calc
    twins    the two fabricated small-number twins (kernel/twin.py) and a
             chain of thought on each
    second   a second reading: a plan sampled at temperature 0.7, compiled
             to Python and executed                                       --second
    on each twin, the value the first and the second program return when
    their large numbers are replaced by the twin's numbers

Runs of the experiment (the model must be loaded in LM Studio first):

    python scripts/exp_twin_run.py --model meta-llama-3.1-8b-instruct --part main  --calc --second
    python scripts/exp_twin_run.py --model meta-llama-3.1-8b-instruct --part small --calc
    python scripts/exp_twin_run.py --model qwen2.5-7b-instruct        --part main  --second
    python scripts/exp_twin_run.py --model qwen2.5-7b-instruct        --part small

``--part main`` is the 250 problems with large numbers, ``--part small`` the
GSM8K version of the first 100 of them. A run refuses to start unless the
code is exactly the frozen version (tag ``exp-twin-frozen``). An interrupted
run continues where it stopped when the same command is given again.

Development (already used problems only, no freeze needed):

    python scripts/exp_twin_run.py --dev paired --dataset gsmhard --limit 40 --calc --second
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import gsm8k  # noqa: E402
from kernel import (CognitiveKernel, EpisodeLog, LLMError, LocalLLM, Reading, TypedMemory,  # noqa: E402
                    solve_cot)
from kernel import twin  # noqa: E402
from kernel.calc_cot import solve_calc_cot  # noqa: E402

FREEZE_TAG = "exp-twin-frozen"
CODE_PATHS = ("cdsl", "kernel", "eval", "scripts", "experiments", "tests")
SECOND_TEMPERATURE, SECOND_SEED = 0.7, 1


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def frozen_state() -> dict:
    """Commit and tree of the code, after checking it is the frozen version."""
    dirty = git("status", "--porcelain", "--", *CODE_PATHS)
    if dirty:
        sys.exit("The code differs from the last commit. The experiment runs only on the frozen "
                 "version.\n" + dirty)
    try:
        frozen_tree = git("rev-parse", f"{FREEZE_TAG}^{{tree}}")
    except subprocess.CalledProcessError:
        sys.exit(f"Tag {FREEZE_TAG} does not exist: the code has not been frozen yet.")
    tree = git("rev-parse", "HEAD^{tree}")
    if tree != frozen_tree:
        sys.exit(f"HEAD is not the frozen version (tree {tree[:12]}, frozen {frozen_tree[:12]}).")
    return {"commit": git("rev-parse", "HEAD"), "tree": tree}


def make_kernel(llm_args: dict, run_dir: Path, name: str, no_embed: bool, **options) -> CognitiveKernel:
    llm = LocalLLM(**llm_args)
    nomic = "nomic" in llm.embed_model
    memory = TypedMemory(embedder=None if no_embed else llm.embed,
                         query_prefix="search_query: " if nomic else "",
                         doc_prefix="search_document: " if nomic else "")
    log = EpisodeLog(run_dir / f"episode_log_{name}.db")
    for ep in log.successes():                       # resume: rebuild episodic memory
        memory.add("episodic", ep["problem_text"],
                   meta={"program": ep["final_program"], "plan": ep["plan"]})
    return CognitiveKernel(llm, memory=memory, episode_log=log, **options)


def attempt_record(a) -> dict:
    return {"index": a.index, "mode": a.compiled.mode, "temperature": a.compiled.temperature,
            "program": a.compiled.program_text, "parse_error": a.compiled.parse_error,
            "value": a.run.value if a.run and a.run.returned else None,
            "passed": a.verdict.passed, "error_trace": a.verdict.error_trace,
            "completion_tokens": a.compiled.completion_tokens}


def run_program(kernel: CognitiveKernel, question: str, reading: Reading):
    tok0 = kernel.llm.completion_tokens
    result = kernel.solve_program(question, reading)
    return result, kernel.llm.completion_tokens - tok0


def program_record(kernel: CognitiveKernel, result, tokens: int) -> dict:
    return {"target": kernel.target.name, "answer": result.program_answer,
            "program": result.program, "plan": result.plan,
            "self_verified": result.success, "attempts": len(result.attempts),
            "completion_tokens": tokens,
            "attempt_details": [attempt_record(a) for a in result.attempts]}


def cot_record(cot) -> dict:
    return {"text": cot.text, "answer": cot.answer, "completion_tokens": cot.completion_tokens,
            "seconds": round(cot.seconds, 2)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--part", choices=("main", "small"), help="the experiment: 250 large-number problems, or 100 small")
    p.add_argument("--dev", choices=gsm8k.SPLITS, help="development: an already used split instead of the experiment")
    p.add_argument("--dataset", choices=gsm8k.DATASETS, default="gsmhard", help="with --dev")
    p.add_argument("--limit", type=int, help="with --dev: only the first N problems")
    p.add_argument("--calc", action="store_true", help="also collect the calculator chain of thought")
    p.add_argument("--second", action="store_true", help="also collect the second reading")
    p.add_argument("--no-twins", action="store_true", help="do not solve the fabricated twins")
    p.add_argument("--out", help="run directory (default: results/exp-twin/<model>-<problems>)")
    p.add_argument("--workers", type=int, default=3, help="concurrent requests to the server")
    p.add_argument("--no-embed", action="store_true", help="bag-of-words memory instead of embeddings")
    p.add_argument("--base-url")
    p.add_argument("--embed-model")
    args = p.parse_args()
    if bool(args.part) == bool(args.dev):
        p.error("give exactly one of --part (the experiment) or --dev (development)")

    if args.part:
        state = frozen_state()
        if args.limit:
            p.error("--limit is for development runs only")
        problems = gsm8k.experiment_split(args.part, "gsm8k" if args.part == "small" else "gsmhard")
        label = args.part
    else:
        state = {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "dev": True}
        problems = gsm8k.split(args.dev, args.dataset)[: args.limit]
        label = f"dev-{args.dev}-{args.dataset}"

    problems = [{"idx": r["idx"], "question": r["question"]} for r in problems]   # no label gets past here
    llm_args = {"base_url": args.base_url, "model": args.model, "embed_model": args.embed_model}
    cot_llm = LocalLLM(**llm_args)
    run_dir = Path(args.out).resolve() if args.out else ROOT / "results" / "exp-twin" / f"{cot_llm.model}-{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["idx"] for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}

    options = {"calc": args.calc, "second": args.second, "twins": not args.no_twins}
    meta = {"model": cot_llm.model, "problems": label, "n": len(problems), **options, **state,
            "second_temperature": SECOND_TEMPERATURE, "second_seed": SECOND_SEED}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        keys = ["model", "problems", "n", *options] + (["tree"] if args.part else [])
        changed = [k for k in keys if saved.get(k) != meta[k]]
        if changed:
            sys.exit(f"{run_dir} was started with different settings ({', '.join(changed)}). "
                     "Use another --out or the same command.")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    try:
        if cot_llm.model not in cot_llm.models():
            sys.exit(f"Model '{cot_llm.model}' is not served at {cot_llm.base_url}")
        first = make_kernel(llm_args, run_dir, "first", args.no_embed, target="cdsl")
        second = make_kernel(llm_args, run_dir, "second", args.no_embed, target="python",
                             plan_temperature=SECOND_TEMPERATURE, plan_seed=SECOND_SEED) if args.second else None
        calc_llm = LocalLLM(**llm_args)
        print(f"Model: {cot_llm.model} | {label}: {len(problems)} problems ({len(done)} already done) | "
              + ", ".join(f"{k}={v}" for k, v in options.items())
              + f" | output: {run_dir.relative_to(ROOT)}\n", flush=True)

        t_start = time.perf_counter()
        with results_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
            for n, row in enumerate(problems, 1):
                if row["idx"] in done:
                    continue
                q = row["question"]
                f_cot = pool.submit(solve_cot, cot_llm, q)
                made = [] if args.no_twins else twin.fabricate(q)
                f_twins = [pool.submit(solve_cot, cot_llm, t.text) for t in made]
                f_calc = pool.submit(solve_calc_cot, calc_llm, q) if args.calc else None

                tok0 = first.llm.completion_tokens
                reading = first.read(q)
                read_tokens = first.llm.completion_tokens - tok0
                f_first = pool.submit(run_program, first, q, reading)
                f_second = None
                if second is not None:
                    plan2 = second.planner.plan(q, reading.notes)
                    reading2 = Reading(reading.category, reading.notes, plan2.text, reading.query_vector,
                                       {"planner": plan2.seconds}, plan2.completion_tokens)
                    f_second = pool.submit(run_program, second, q, reading2)

                cot = f_cot.result()
                r_first, tok_first = f_first.result()
                rec = {"idx": row["idx"], "question": q, "model": cot_llm.model,
                       "cot": cot_record(cot),
                       "read": {"category": reading.category, "plan": reading.plan,
                                "notes": reading.notes, "completion_tokens": read_tokens},
                       "first": program_record(first, r_first, tok_first),
                       "second": None, "calc": None, "twins": []}
                r_second = None
                if f_second is not None:
                    r_second, tok_second = f_second.result()
                    rec["second"] = program_record(second, r_second, plan2.completion_tokens + tok_second)
                if f_calc is not None:
                    calc = f_calc.result()
                    rec["calc"] = {**cot_record(calc), "rounds": calc.rounds,
                                   "calculations": calc.calculations, "corrected": calc.corrected}
                for t, f in zip(made, f_twins):
                    rec["twins"].append({
                        "kind": t.kind, "text": t.text, "mapping": sorted(t.mapping.items()),
                        "cot": cot_record(f.result()),
                        "first": twin.run_with(first.target, r_first.program, t.mapping),
                        "second": twin.run_with(second.target, r_second.program, t.mapping)
                        if r_second is not None else None})

                # Today's kernel, as it runs: cross-check, referee, episode log. Confirmed
                # episodes become worked examples for later problems, as in phase2_eval.py.
                arb = first.cross_check(r_first, cot)
                first.commit(r_first)
                if second is not None:
                    second.cross_check(r_second, cot, referee=False)
                    second.commit(r_second)
                rec["kernel"] = {"answer": r_first.answer, "source": arb.source, "confidence": arb.confidence}

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                twins_txt = " ".join(f"{t['kind']}:cot={t['cot']['answer']!r}/p1={t['first']!r}"
                                     for t in rec["twins"])
                print(f"[{n:>3}/{len(problems)}] idx={row['idx']} cot={cot.answer!r} "
                      f"p1={rec['first']['answer']!r}"
                      + (f" p2={rec['second']['answer']!r}" if rec["second"] else "")
                      + (f" calc={rec['calc']['answer']!r}" if rec["calc"] else "")
                      + (f" | {twins_txt}" if twins_txt else "")
                      + f" | [{arb.source}] {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    except LLMError as e:
        sys.exit(f"LLM error: {e}\nGive the same command again to continue.")
    print(f"\nDone: {results_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
