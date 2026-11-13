"""Experiment 2026-11 (twin test and second model): compute the conditions and the tests.

Reads the pieces saved by ``scripts/exp_twin_run.py`` and the answer key, and
prints every condition of ``experiments/2026-11-twin/PREREGISTRATION.md``:
accuracy with a 95% Wilson interval, the share of answers that were confirmed,
how often a confirmed answer is right, and tokens per problem. Then the
pre-registered comparisons (exact McNemar) and the kill criteria, evaluated
mechanically. No LLM call.

    python scripts/exp_twin_report.py                 # the experiment, both parts
    python scripts/exp_twin_report.py --dev-a results/exp-twin/<run> [--dev-b <run>] --dataset gsmhard

In a development report the answer key is the GSM-Hard target (problems with a
safe label only) or the GSM8K answer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import gsm8k  # noqa: E402
from eval import twin_experiment as tx  # noqa: E402

LLAMA, QWEN = "meta-llama-3.1-8b-instruct", "qwen2.5-7b-instruct"
EXPERIMENT_DIR = ROOT / "experiments" / "2026-11-twin"
GOLD_FILE = EXPERIMENT_DIR / "gold.json"
PRIMARY = ("7 twin test", "3 kernel today")


def load_run(path: Path | None) -> dict[int, dict]:
    if path is None or not (path / "results.jsonl").exists():
        return {}
    rows = (json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip())
    return {r["idx"]: r for r in rows}


def systems(a: dict, b: dict | None) -> dict[str, tx.Outcome]:
    """Every condition for one problem. ``a`` is the Llama record, ``b`` the Qwen record."""
    out = {"1 cot": tx.cot_only(a), "2 program": tx.program_only(a),
           "3 kernel today": tx.kernel_today(a), "4 kernel + audit": tx.kernel_audit(a)}
    if a.get("calc"):
        out["5 calculator cot"] = tx.calc_only(a)
        out["6a rival, program wins"] = tx.rival(a, "program")
        out["6b rival, calculator wins"] = tx.rival(a, "calc")
    out["7 twin test"] = tx.verified(a)
    out["x7 round twin only"] = tx.verified(a, kinds=("round",))
    out["x7 odd twin only"] = tx.verified(a, kinds=("odd",))
    out["x7 no twin (audit + second reading)"] = tx.verified(a, kinds=())
    out["x7 + programs agree"] = tx.verified(a, programs_agree=True)
    if b is not None:
        out["Q1 cot (Qwen)"] = tx.cot_only(b)
        out["Q2 program (Qwen)"] = tx.program_only(b)
        out["Q3 kernel today (Qwen)"] = tx.kernel_today(b)
        out["Q4 kernel + audit (Qwen)"] = tx.kernel_audit(b)
        out["Q7 twin test (Qwen)"] = tx.verified(b)
        out["8 two models, Llama base"] = tx.verified(a, b)
        out["8Q two models, Qwen base"] = tx.verified(b, a)
        out["x8 no twin"] = tx.verified(a, b, kinds=())
        out["x8 + programs agree"] = tx.verified(a, b, programs_agree=True)
        out["x8Q + programs agree"] = tx.verified(b, a, programs_agree=True)
    return out


def is_right(answer: object, accepted: list[float], whole_ok: bool = False) -> bool:
    """``accepted``: the values the answer key accepts for a problem (usually one).

    ``whole_ok`` is the sensitivity check: an answer that is a fractional key
    rounded to a whole number also counts.
    """
    if any(gsm8k.is_correct(answer, g) for g in accepted):
        return True
    return whole_ok and isinstance(answer, (int, float)) and float(answer).is_integer() and any(
        not float(g).is_integer() and abs(answer - g) <= 0.5 for g in accepted)


def table(rows: dict[int, dict[str, tx.Outcome]], gold: dict[int, list[float]]) -> dict[str, dict]:
    names = list(next(iter(rows.values())))
    out = {}
    for name in names:
        n = len(rows)
        right = {i: is_right(rows[i][name].answer, gold[i]) for i in rows}
        conf = [i for i in rows if rows[i][name].confirmed]
        lo, hi = gsm8k.wilson(sum(right.values()), n)
        out[name] = {"n": n, "correct": sum(right.values()), "accuracy": sum(right.values()) / n,
                     "ci95": [lo, hi], "confirmed": len(conf),
                     "confirmed_right": sum(right[i] for i in conf),
                     "whole_ok": sum(is_right(rows[i][name].answer, gold[i], whole_ok=True) for i in rows) / n,
                     "tokens": sum(rows[i][name].tokens for i in rows) / n, "right": right}
    return out


def compare(t: dict, x: str, y: str) -> dict:
    rx, ry = t[x]["right"], t[y]["right"]
    only_x = sum(rx[i] and not ry[i] for i in rx)
    only_y = sum(ry[i] and not rx[i] for i in rx)
    return {"x": x, "y": y, "only_x": only_x, "only_y": only_y,
            "difference_points": 100 * (t[x]["accuracy"] - t[y]["accuracy"]),
            "mcnemar_p": gsm8k.mcnemar_p(only_x, only_y)}


def pct(k: float, n: float) -> str:
    return f"{100 * k / n:5.1f}%" if n else "  n/a"


def print_table(title: str, t: dict) -> None:
    print(f"\n{title}")
    print(f"  {'system':38s} {'accuracy':>8s}  {'95% CI':>14s}  {'confirmed':>9s}  {'right if confirmed':>18s}  "
          f"{'tokens':>6s}  {'whole ok':>8s}")
    for name, s in t.items():
        ci = f"[{100 * s['ci95'][0]:.1f}, {100 * s['ci95'][1]:.1f}]"
        conf = pct(s["confirmed"], s["n"]) if s["confirmed"] else "       "
        prec = f"{s['confirmed_right']}/{s['confirmed']} = {pct(s['confirmed_right'], s['confirmed'])}" \
            if s["confirmed"] else ""
        print(f"  {name:38s} {pct(s['correct'], s['n']):>8s}  {ci:>14s}  {conf:>9s}  {prec:>18s}  "
              f"{s['tokens']:6.0f}  {100 * s['whole_ok']:7.1f}%")


def print_compare(c: dict) -> None:
    print(f"  {c['x']}  vs  {c['y']}: {c['difference_points']:+.1f} points | right only in the first "
          f"{c['only_x']}, only in the second {c['only_y']} | McNemar p = {c['mcnemar_p']:.4f}")


def kill_criteria(t: dict) -> dict:
    """The criteria written before the run, evaluated mechanically."""
    seven, audit = t["7 twin test"], t["4 kernel + audit"]
    primary = compare(t, *PRIMARY)
    cover = lambda s: 100 * s["confirmed"] / s["n"]                                   # noqa: E731
    precision = lambda s: 100 * s["confirmed_right"] / s["confirmed"] if s["confirmed"] else 0.0   # noqa: E731
    out = {
        "twin: confirmed answers right at least 95%": precision(seven) >= 95,
        "twin: confirms at least 10 points more than the audit alone": cover(seven) - cover(audit) >= 10,
        "twin: beats today's kernel by at least 5 points, p < 0.05":
            primary["difference_points"] >= 5 and primary["mcnemar_p"] < 0.05,
    }
    out["twin survives"] = all(out.values())
    if "6a rival, program wins" in t:
        best = max(("6a rival, program wins", "6b rival, calculator wins"), key=lambda k: t[k]["accuracy"])
        r = t[best]
        out["rival (best variant)"] = best
        out["rival wins"] = (100 * (seven["accuracy"] - r["accuracy"]) <= 2 and r["tokens"] <= seven["tokens"]
                             and precision(r) >= 95 and cover(seven) - cover(r) <= 5)
    return out


def report(title: str, a_run: dict, b_run: dict, gold: dict[int, list[float]], small: bool = False) -> dict:
    ids = [i for i in a_run if i in gold and (not b_run or i in b_run)]
    if not ids:
        print(f"\n{title}: no problems with both a run and a label")
        return {}
    rows = {i: systems(a_run[i], b_run.get(i)) for i in ids}
    t = table(rows, gold)
    print_table(f"{title}: {len(ids)} problems", t)
    out = {"n": len(ids), "systems": {k: {x: v[x] for x in v if x != "right"} for k, v in t.items()}}
    comparisons = [PRIMARY]
    if "8 two models, Llama base" in t:
        comparisons += [("8 two models, Llama base", "7 twin test"),
                        ("8Q two models, Qwen base", "Q7 twin test (Qwen)"),
                        ("8 two models, Llama base", "3 kernel today"),
                        ("8Q two models, Qwen base", "Q3 kernel today (Qwen)")]
    if "6a rival, program wins" in t:
        comparisons += [("7 twin test", "6a rival, program wins"), ("7 twin test", "6b rival, calculator wins"),
                        ("5 calculator cot", "1 cot")]
    if small:
        comparisons = [("4 kernel + audit", "3 kernel today")]
        if "5 calculator cot" in t:
            comparisons.append(("5 calculator cot", "1 cot"))
        if "8 two models, Llama base" in t:
            comparisons += [("8 two models, Llama base", "3 kernel today"),
                            ("8Q two models, Qwen base", "Q3 kernel today (Qwen)")]
    print("\n  comparisons (the first one is the primary comparison)" if not small else "\n  comparisons")
    out["comparisons"] = [compare(t, x, y) for x, y in comparisons]
    for c in out["comparisons"]:
        print_compare(c)
    if not small:
        out["kill_criteria"] = kill_criteria(t)
        print("\n  kill criteria")
        for k, v in out["kill_criteria"].items():
            print(f"    {k}: {v}")
        how = {}
        for name in ("7 twin test", "8 two models, Llama base", "8Q two models, Qwen base"):
            if name in t:
                counts: dict[str, list[int]] = {}
                for i in ids:
                    c = rows[i][name].confirmed
                    if c:
                        counts.setdefault(c, [0, 0])
                        counts[c][0] += 1
                        counts[c][1] += t[name]["right"][i]
                how[name] = counts
                print(f"\n  how '{name}' confirmed: " + ", ".join(
                    f"{k} {v[1]}/{v[0]}" for k, v in sorted(counts.items())))
        out["confirmed_by"] = how
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(ROOT / "results" / "exp-twin"), help="folder with the run directories")
    p.add_argument("--dev-a", help="development: run directory of the base model")
    p.add_argument("--dev-b", help="development: run directory of the second model")
    p.add_argument("--dataset", choices=gsm8k.DATASETS, default="gsmhard", help="with --dev-a")
    p.add_argument("--save", help="write the summary as JSON to this file")
    args = p.parse_args()

    summary = {}
    if args.dev_a:
        a, b = load_run(Path(args.dev_a)), load_run(Path(args.dev_b) if args.dev_b else None)
        rows = gsm8k.load()
        if args.dataset == "gsm8k":
            gold = {i: [rows[i]["gold"]] for i in a}
        else:
            twins = gsm8k.load_hard()
            gold = {i: [twins[i]["gold"]] for i in a if i in twins
                    and gsm8k.label_safe(rows[i]["question"], twins[i]["question"], twins[i]["code"])}
        summary["dev"] = report(f"Development, {args.dataset}", a, b, gold, small=args.dataset == "gsm8k")
    else:
        runs = Path(args.runs)
        labels = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["labels"]
        settled = {int(i): v["gold"] for i, v in labels.items() if v["status"] != "excluded"}   # lists
        a, b = load_run(runs / f"{LLAMA}-main"), load_run(runs / f"{QWEN}-main")
        summary["main"] = report("Large numbers, settled labels (primary analysis)", a, b, settled)
        by_status = {}
        for i, v in labels.items():
            by_status.setdefault(v["status"], {})[int(i)] = (
                v["gold"] if v["status"] != "excluded" else [v["reference"]])
        for status, gold in sorted(by_status.items()):
            if status != "agree" or len(by_status) > 1:
                summary[f"main:{status}"] = report(
                    f"Large numbers, label status '{status}'"
                    + (" (scored against the reference program)" if status == "excluded" else ""), a, b, gold)
        rows = gsm8k.load()
        a, b = load_run(runs / f"{LLAMA}-small"), load_run(runs / f"{QWEN}-small")
        summary["small"] = report("Small numbers (GSM8K version)", a, b,
                                  {i: [rows[i]["gold"]] for i in a}, small=True)
    if args.save:
        Path(args.save).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved {args.save}")


if __name__ == "__main__":
    main()
