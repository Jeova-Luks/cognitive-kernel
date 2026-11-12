"""Phase 2 evaluation: four systems on the same problems, same local LLM.

    cot      plain chain-of-thought, no program (baseline 1, spec §8.2)
    cdsl     plan -> CDSL program, executed (the program's own answer, no cross-check)
    python   the same reading and plan compiled to Python instead (baseline 2, PoT)
    arbiter  the full kernel: the cdsl answer cross-checked against cot; agreement
             is "high confidence", a disagreement goes to a referee

Usage
-----
    lms server start && lms load meta-llama-3.1-8b-instruct
    python scripts/phase2_eval.py --split dev                      # 50 GSM8K development problems
    python scripts/phase2_eval.py --split paired                   # 200 held-out problems, GSM8K version
    python scripts/phase2_eval.py --split paired --dataset gsmhard # the same 200 with large numbers
    python scripts/phase2_eval.py --resume results/phase2/<run>    # continue an interrupted run

Ablation switches: --no-plan, --self-checks, --max-attempts N, --referee choose|resolve,
--no-episodic. Splits and datasets are defined in eval/gsm8k.py.

For each problem the CoT baseline runs while the problem is read (classifier,
memory, plan); the plan is then compiled to CDSL and to Python concurrently.
Problems are processed in order, so episodic memory grows deterministically.
Seconds are measured under contention: compare systems by tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import gsm8k  # noqa: E402
from kernel import CognitiveKernel, EpisodeLog, LLMError, LocalLLM, TypedMemory, solve_cot  # noqa: E402

SYSTEMS = ("cot", "cdsl", "python", "arbiter")
OPTIONS = ("dataset", "split", "limit", "max_attempts", "plan", "self_checks", "referee", "episodic")


def make_llm(args) -> LocalLLM:
    return LocalLLM(base_url=args.base_url, model=args.model, embed_model=args.embed_model)


def make_kernel(target: str, args, run_dir: Path) -> CognitiveKernel:
    llm = make_llm(args)
    nomic = "nomic" in llm.embed_model
    memory = TypedMemory(embedder=None if args.no_embed else llm.embed,
                         query_prefix="search_query: " if nomic else "",
                         doc_prefix="search_document: " if nomic else "")
    log = EpisodeLog(run_dir / f"episode_log_{target}.db")
    for ep in log.successes():                       # resume: rebuild episodic memory
        memory.add("episodic", ep["problem_text"],
                   meta={"program": ep["final_program"], "plan": ep["plan"]})
    return CognitiveKernel(llm, memory=memory, episode_log=log, max_attempts=args.max_attempts,
                           episodes_k=1 if args.episodic else 0, target=target, plan=args.plan,
                           self_checks=args.self_checks, referee=args.referee)


def attempt_record(a) -> dict:
    return {
        "index": a.index, "mode": a.compiled.mode, "temperature": a.compiled.temperature,
        "raw": a.compiled.raw, "program": a.compiled.program_text,
        "parse_error": a.compiled.parse_error,
        "executed": bool(a.run and a.run.ok),
        "value": a.run.value if a.run and a.run.returned else None,
        "trace": a.run.trace if a.run else [],
        "checks": a.verdict.checks, "passed": a.verdict.passed,
        "error_trace": a.verdict.error_trace,
        "completion_tokens": a.compiled.completion_tokens,
    }


def run_program(kernel: CognitiveKernel, question: str, reading) -> tuple[object, int, float]:
    tok0, t0 = kernel.llm.completion_tokens, time.perf_counter()
    result = kernel.solve_program(question, reading)
    return result, kernel.llm.completion_tokens - tok0, time.perf_counter() - t0


def program_record(result, gold: float, tokens: int, seconds: float) -> dict:
    return {
        "answer": result.program_answer, "correct": gsm8k.is_correct(result.program_answer, gold),
        "self_verified": result.success, "attempts": len(result.attempts),
        "seconds": round(seconds, 2), "completion_tokens": tokens,
        "episodes_recalled": len(result.episodes_recalled),
        "attempt_details": [attempt_record(a) for a in result.attempts],
    }


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}%" if n else "n/a"


def kernel_summary(records: list[dict], name: str) -> dict:
    n = len(records)
    rs = [r[name] for r in records]
    firsts = [r["attempt_details"][0] for r in rs]
    attempts = [a for r in rs for a in r["attempt_details"]]
    ran = [r for r in rs if r["self_verified"]]
    agree = [r for r in records if r[name]["agrees_with_cot"]]
    by_mode = {}
    for mode in ("first", "repair", "fallback"):
        ms = [a for a in attempts if a["mode"] == mode]
        by_mode[mode] = {"attempts": len(ms), "passed": sum(a["passed"] for a in ms)}
    return {
        "self_verified": len(ran),
        "verified_and_correct": sum(r["correct"] for r in ran),
        "no_answer": sum(r["answer"] is None for r in rs),
        "agrees_with_cot": {"n": len(agree), "correct": sum(r[name]["correct"] for r in agree)},
        "first_attempt": {
            "parsed": sum(a["parse_error"] is None for a in firsts),
            "executed": sum(a["executed"] for a in firsts),
            "passed": sum(a["passed"] for a in firsts),
            "correct": sum(gsm8k.is_correct(a["value"], rec["gold"]) for a, rec in zip(firsts, records)),
        },
        "all_attempts": {"count": len(attempts),
                         "parsed": sum(a["parse_error"] is None for a in attempts),
                         "executed": sum(a["executed"] for a in attempts)},
        "by_mode": by_mode,
        "avg_attempts": round(sum(r["attempts"] for r in rs) / n, 2),
        "solved_by_repair": sum(r["self_verified"] and r["attempts"] > 1 for r in rs),
    }


def summarize(records: list[dict], meta: dict) -> dict:
    n = len(records)
    out = {**meta, "n": n, "systems": {}}
    for name in SYSTEMS:
        k = sum(r[name]["correct"] for r in records)
        lo, hi = gsm8k.wilson(k, n)
        s = {"correct": k, "accuracy": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
             "avg_completion_tokens": round(sum(r[name]["completion_tokens"] for r in records) / n)}
        if name != "cot":
            only_sys = sum(r[name]["correct"] and not r["cot"]["correct"] for r in records)
            only_cot = sum(r["cot"]["correct"] and not r[name]["correct"] for r in records)
            s["vs_cot"] = {"only_this": only_sys, "only_cot": only_cot,
                           "mcnemar_p": round(gsm8k.mcnemar_p(only_sys, only_cot), 4)}
        if name in ("cdsl", "python"):
            s.update(kernel_summary(records, name))
        if name == "arbiter":
            sources: dict[str, dict] = {}
            confidence: dict[str, dict] = {}
            for r in records:
                for table, key in ((sources, r["arbiter"]["source"]),
                                   (confidence, r["arbiter"]["confidence"])):
                    d = table.setdefault(key, {"n": 0, "correct": 0})
                    d["n"] += 1
                    d["correct"] += r["arbiter"]["correct"]
            s["sources"], s["confidence"] = sources, confidence
            dis = [r for r in records if r["arbiter"]["source"] != "agree"]
            s["disagreements"] = {
                "n": len(dis),
                "cdsl_correct": sum(r["cdsl"]["correct"] for r in dis),
                "cot_correct": sum(r["cot"]["correct"] for r in dis),
                "arbiter_correct": sum(r["arbiter"]["correct"] for r in dis),
            }
        out["systems"][name] = s
    a = sum(r["cdsl"]["correct"] and not r["python"]["correct"] for r in records)
    b = sum(r["python"]["correct"] and not r["cdsl"]["correct"] for r in records)
    out["cdsl_vs_python"] = {"only_cdsl": a, "only_python": b,
                             "mcnemar_p": round(gsm8k.mcnemar_p(a, b), 4)}
    return out


def print_summary(s: dict) -> None:
    n = s["n"]
    print(f"\n======== Phase 2 summary: {s['dataset']} / {s['split']}, n={n}, model={s['model']} ========")
    print("options: " + ", ".join(f"{k}={s[k]}" for k in OPTIONS[3:]))
    print(f"{'system':<9}{'accuracy':>18}{'95% CI':>18}{'tok/problem':>13}"
          f"{'  vs CoT (only this / only CoT, McNemar p)'}")
    for name, d in s["systems"].items():
        vs = d.get("vs_cot")
        vs_txt = f"  {vs['only_this']} / {vs['only_cot']}, p={vs['mcnemar_p']}" if vs else ""
        print(f"{name:<9}{d['correct']:>6}/{n} = {pct(d['correct'], n):>6}"
              f"{'[%.1f%%, %.1f%%]' % (100 * d['ci95'][0], 100 * d['ci95'][1]):>18}"
              f"{d['avg_completion_tokens']:>13}{vs_txt}")
    for name in ("cdsl", "python"):
        d = s["systems"][name]
        fa, aa, bm, ag = d["first_attempt"], d["all_attempts"], d["by_mode"], d["agrees_with_cot"]
        print(f"\n[{name}] program ran and returned a number: {d['self_verified']}/{n}, "
              f"of which correct {d['verified_and_correct']} "
              f"({pct(d['verified_and_correct'], d['self_verified'])}); no answer {d['no_answer']}")
        print(f"[{name}] agrees with CoT on {ag['n']}/{n}; when they agree, correct {ag['correct']}/{ag['n']} "
              f"({pct(ag['correct'], ag['n'])})")
        print(f"[{name}] first attempt: parse {pct(fa['parsed'], n)} | exec {pct(fa['executed'], n)} | "
              f"correct {pct(fa['correct'], n)}")
        print(f"[{name}] all {aa['count']} attempts: parse {pct(aa['parsed'], aa['count'])} | "
              f"exec {pct(aa['executed'], aa['count'])} | avg attempts {d['avg_attempts']} | "
              f"solved by repair {d['solved_by_repair']} | by mode: " + " | ".join(
                  f"{m} {v['passed']}/{v['attempts']}" for m, v in bm.items()))
    c = s["cdsl_vs_python"]
    print(f"\ncdsl vs python: only cdsl {c['only_cdsl']} / only python {c['only_python']}, "
          f"McNemar p={c['mcnemar_p']}")
    arb = s["systems"]["arbiter"]
    d = arb["disagreements"]
    print("\n[arbiter] confidence: " + " | ".join(
        f"{k} {v['correct']}/{v['n']} ({pct(v['correct'], v['n'])})"
        for k, v in sorted(arb["confidence"].items())))
    print("[arbiter] decisions: " + " | ".join(
        f"{k} {v['correct']}/{v['n']}" for k, v in sorted(arb["sources"].items())))
    print(f"[arbiter] on the {d['n']} non-agreeing problems: cdsl right {d['cdsl_correct']}, "
          f"cot right {d['cot_correct']}, arbiter right {d['arbiter_correct']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=gsm8k.DATASETS, default="gsm8k")
    p.add_argument("--split", choices=gsm8k.SPLITS, default="dev")
    p.add_argument("--limit", type=int, help="only the first N problems of the split")
    p.add_argument("--resume", help="run directory to continue")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--no-plan", dest="plan", action="store_false", help="compile straight from the problem")
    p.add_argument("--self-checks", action="store_true", help="require assert/verify inside programs")
    p.add_argument("--referee", choices=("rule", "choose", "resolve"), default="rule")
    p.add_argument("--no-episodic", dest="episodic", action="store_false", help="do not recall past episodes")
    p.add_argument("--no-embed", action="store_true", help="bag-of-words memory instead of embeddings")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--embed-model")
    args = p.parse_args()

    if args.resume:
        run_dir = Path(args.resume).resolve()
        saved = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        for key in OPTIONS:
            setattr(args, key, saved[key])
    else:
        run_dir = ROOT / "results" / "phase2" / (
            f"{args.dataset}-{args.split}-{time.strftime('%Y%m%d-%H%M%S')}")
        run_dir.mkdir(parents=True, exist_ok=True)
    problems = gsm8k.split(args.split, args.dataset)[: args.limit]
    results_path = run_dir / "results.jsonl"
    records = ([json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if results_path.exists() else [])
    done = {r["idx"] for r in records}

    try:
        cot_llm = make_llm(args)
        if cot_llm.model not in cot_llm.models():
            sys.exit(f"Model '{cot_llm.model}' is not served at {cot_llm.base_url}")
        cdsl, python = (make_kernel(t, args, run_dir) for t in ("cdsl", "python"))
        meta = {"model": cot_llm.model, **{k: getattr(args, k) for k in OPTIONS}}
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Model: {cot_llm.model} | {args.dataset} / {args.split}: {len(problems)} problems "
              f"({len(done)} already done) | " + ", ".join(f"{k}={meta[k]}" for k in OPTIONS[3:]) +
              f" | output: {run_dir.relative_to(ROOT)}\n", flush=True)

        t_start = time.perf_counter()
        with results_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(3) as pool:
            for i, row in enumerate(problems, 1):
                if row["idx"] in done:
                    continue
                q, gold = row["question"], row["gold"]
                f_cot = pool.submit(solve_cot, cot_llm, q)
                tok0 = cdsl.llm.completion_tokens
                reading = cdsl.read(q)                      # shared by both targets
                read_tokens = cdsl.llm.completion_tokens - tok0
                f_cdsl = pool.submit(run_program, cdsl, q, reading)
                f_py = pool.submit(run_program, python, q, reading)
                cot = f_cot.result()
                (r_cdsl, tok_cdsl, sec_cdsl), (r_py, tok_py, sec_py) = f_cdsl.result(), f_py.result()

                rec = {"idx": row["idx"], "question": q, "gold": gold, "category": reading.category,
                       "plan": reading.plan, "memory_notes": reading.notes,
                       "cot": {"answer": cot.answer, "correct": gsm8k.is_correct(cot.answer, gold),
                               "seconds": round(cot.seconds, 2),
                               "completion_tokens": cot.completion_tokens, "text": cot.text},
                       "cdsl": program_record(r_cdsl, gold, read_tokens + tok_cdsl, sec_cdsl),
                       "python": program_record(r_py, gold, read_tokens + tok_py, sec_py)}

                tok0 = cdsl.llm.completion_tokens
                arb = cdsl.cross_check(r_cdsl, cot)          # full kernel: cross-check + referee
                referee_tokens = cdsl.llm.completion_tokens - tok0
                py_check = python.cross_check(r_py, cot, referee=False)   # confirmation only
                cdsl.commit(r_cdsl)
                python.commit(r_py)
                rec["cdsl"]["agrees_with_cot"] = arb.source == "agree"
                rec["python"]["agrees_with_cot"] = py_check.source == "agree"
                rec["arbiter"] = {
                    "answer": r_cdsl.answer, "correct": gsm8k.is_correct(r_cdsl.answer, gold),
                    "source": arb.source, "confidence": arb.confidence,
                    "referee_text": arb.referee.text if arb.referee else None,
                    # cost of the full kernel = reading + program + CoT + the referee call, if any
                    "completion_tokens": rec["cdsl"]["completion_tokens"]
                    + cot.completion_tokens + referee_tokens,
                }
                records.append(rec)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                mark = lambda name: "OK" if rec[name]["correct"] else "X "  # noqa: E731
                print(f"[{i:>3}/{len(problems)}] gold={gold:g} | "
                      f"cot={cot.answer!r} {mark('cot')} | "
                      f"cdsl={rec['cdsl']['answer']!r} {mark('cdsl')} ({rec['cdsl']['attempts']} att) | "
                      f"py={rec['python']['answer']!r} {mark('python')} ({rec['python']['attempts']} att) | "
                      f"arb={rec['arbiter']['answer']!r} {mark('arbiter')} [{arb.source}] | "
                      f"{(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    except LLMError as e:
        sys.exit(f"LLM error: {e}\nResume with: python scripts/phase2_eval.py --resume {run_dir}")

    summary = summarize(records, meta)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)
    print(f"\nSaved in {run_dir.relative_to(ROOT)}: results.jsonl, summary.json, episode_log_*.db")


if __name__ == "__main__":
    main()
