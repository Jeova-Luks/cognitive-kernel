"""Build the answer key of the experiment (gold.json) from two independent sources.

Source 1, the benchmark: the value the GSM-Hard reference program returns. It
is re-executed here, because the target stored in the dataset is rounded to 10
decimals and that destroys answers such as 5.158e-07.

Source 2, independent labels: every problem was solved again from its text
alone by a labeller that never saw the reference program or the target
(``labels/main_*.json``). Problems where the two sources differ got a second
independent labeller (``labels/second_*.json``).

Rules, fixed before any model output existed:

    agree      the first label equals the reference value                  -> gold = that value
    rounding   the label equals the reference value except for a rounding
               the story implies (whole buses, full boxes)                 -> both values accepted
    second     the second label equals the reference value                 -> gold = reference value
    ambiguous  a labeller gives the reference value as a second reasonable
               reading of the text                                         -> both values accepted
    relabeled  the two independent labels agree with each other, not with
               the reference                                               -> gold = the label
    excluded   no two sources agree, or the agreed answer is negative      -> not in the primary analysis

Usage: python experiments/2026-11-twin/build_gold.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from eval import gsm8k  # noqa: E402
from kernel.arbiter import numbers_match  # noqa: E402
from kernel.targets import PythonTarget  # noqa: E402
from kernel.verifier import is_number  # noqa: E402

_ALT = re.compile(r"alt\s*=\s*(-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)")


def load_labels(pattern: str) -> dict[int, dict]:
    out = {}
    for path in sorted(glob.glob(str(HERE / "labels" / pattern))):
        for item in json.loads(Path(path).read_text(encoding="utf-8")):
            out[item["id"]] = item
    return out


def alternatives(label: dict) -> list[float]:
    values = [float(m.replace(",", "")) for m in _ALT.findall(label.get("note") or "")]
    values += [a["value"] for a in label.get("alternatives") or [] if is_number(a.get("value"))]
    return values


def settle(reference: float, first: dict, second: dict | None) -> tuple[str, list[float]]:
    """(status, accepted values) for one problem."""
    if numbers_match(first["answer"], reference):
        return "agree", [reference]
    if "rounding" in (first.get("flags") or []) and numbers_match(first.get("exact"), reference):
        return "rounding", [reference, first["answer"]]
    if second is None:
        return "pending", []
    if numbers_match(second["answer"], reference) or numbers_match(second.get("exact"), reference):
        return "second", [reference]
    if any(numbers_match(v, reference) for label in (first, second) for v in alternatives(label)):
        accepted = [reference] + [x["answer"] for x in (first, second)
                                  if is_number(x["answer"]) and x["answer"] >= 0]
        return "ambiguous", accepted
    if numbers_match(first["answer"], second["answer"]) and first["answer"] >= 0:
        return "relabeled", [first["answer"]]
    return "excluded", []


def main() -> None:
    rows, twins = gsm8k.load(), gsm8k.load_hard()
    first, second = load_labels("main_*.json"), load_labels("second_*.json")
    py = PythonTarget()
    labels, counts = {}, {}
    for i in gsm8k.experiment_indices()["main"]:
        run = py.run(twins[i]["code"] + "\nanswer = solution()\n")
        reference = float(run.value) if run.returned and is_number(run.value) else twins[i]["gold"]
        status, accepted = settle(reference, first[i], second.get(i))
        counts[status] = counts.get(status, 0) + 1
        labels[str(i)] = {
            "status": status, "gold": accepted, "reference": reference,
            "label": first[i]["answer"], "label_flags": first[i].get("flags") or [],
            "second_label": second[i]["answer"] if i in second else None,
            "gsm8k_answer": rows[i]["gold"],
        }
    body = json.dumps(labels, indent=1, sort_keys=True)
    out = {"sha256": hashlib.sha256(body.encode()).hexdigest(), "counts": counts, "labels": labels}
    (HERE / "gold.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"gold.json: {len(labels)} problems, {counts}, sha256 {out['sha256'][:16]}")
    for i, v in labels.items():
        if v["status"] not in ("agree", "rounding"):
            print(f"  [{i}] {v['status']}: gold={v['gold']} reference={v['reference']!r} "
                  f"label={v['label']!r} second={v['second_label']!r}")


if __name__ == "__main__":
    main()
