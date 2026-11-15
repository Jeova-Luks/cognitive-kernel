"""GSM8K and GSM-Hard loading, fixed splits and small-sample statistics.

Datasets
- ``gsm8k``    the 1319-problem GSM8K test file.
- ``gsmhard``  GSM-Hard (Gao et al., PAL, 2022): the same problems with one
               number replaced by a large one, so the reasoning is unchanged
               but mental arithmetic is not enough. Rows are matched to GSM8K
               rows by text. Dropped: twins with a negative target (the
               substitution made the problem meaningless) and twins whose target
               equals the original (unchanged copies, or a changed number with
               the old label). 764 usable twins. GSM-Hard has further label
               noise, which ``label_safe`` filters out (see below).

Splits (GSM8K test row indices; a GSM-Hard split is the twins of those rows):

- ``dev``      50 problems, ``random.Random(0).sample(range(1319), 50)``. Used
               to develop and tune the Phase 2 pipeline (prompts, runtime
               leniencies, repair loop). Never report it as a result.
- ``heldout``  200 problems drawn with seed 2026 from the rows that are not in
               dev and are not row 0 (row 0 was the manual smoke test). First
               evaluated on 2026-11-11 with the pipeline frozen. Anything tuned
               after looking at heldout results needs a fresh split
               (``fresh_split``) to be reported honestly.
- ``paired``   200 problems drawn with seed 2027 from rows that are in neither
               dev nor heldout and that have a GSM-Hard twin. Evaluated in both
               datasets, so every problem is seen with small and large numbers.

Experiment split (2026-11, twin test and second model). The pool is every
GSM-Hard twin that no earlier split used and whose label is safe: exactly one
number changed, and that value occurs once in the original text and once in
the reference program, so the GSM-Hard target is the reference function
applied to the text. 345 problems. ``experiment_indices`` draws 250 of them
with seed 2028 (``main``; its first 100 are also run in their GSM8K version,
``small``) and keeps the other 95 locked (``reserve``) for a later
confirmation: nothing may be run or looked at there before that.

Caveat for every number produced here: GSM8K test has been public since 2021,
so any modern open-weights LLM may have seen it during training. That inflates
the plain-CoT baseline more than it inflates program-emitting pipelines.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel.arbiter import numbers_match  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
CACHE = ROOT / "results" / "gsm8k" / "test.jsonl"

HARD_URL = "https://huggingface.co/datasets/reasoning-machines/gsm-hard/resolve/main/gsmhardv2.jsonl"
HARD_CACHE = ROOT / "results" / "gsmhard" / "gsmhardv2.jsonl"

DEV_SEED, DEV_N = 0, 50
HELDOUT_SEED, HELDOUT_N = 2026, 200
PAIRED_SEED, PAIRED_N = 2027, 200
SMOKE_ROWS = {0}
EXPERIMENT_SEED, EXPERIMENT_N, EXPERIMENT_SMALL_N = 2028, 250, 100
DATASETS = ("gsm8k", "gsmhard")
SPLITS = ("dev", "heldout", "paired")


def _gold(answer_field: str) -> float:
    return float(answer_field.split("####")[-1].strip().replace(",", ""))


def load() -> list[dict]:
    """All test rows as {idx, question, gold}."""
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading GSM8K test split -> {CACHE.relative_to(ROOT)}")
        with urllib.request.urlopen(URL, timeout=60) as resp:
            CACHE.write_bytes(resp.read())
    rows = []
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows.append({"idx": len(rows), "question": d["question"], "gold": _gold(d["answer"])})
    return rows


def _text_key(question: str) -> str:
    """Problem text with numbers and punctuation removed: equal for a GSM8K row and its twin."""
    return re.sub(r"[^a-z]+", "", re.sub(r"\d[\d,]*\.?\d*", "", question.lower()))


def load_hard() -> dict[int, dict]:
    """GSM-Hard twins keyed by GSM8K row index: {idx, question, gold}."""
    if not HARD_CACHE.exists():
        HARD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading GSM-Hard -> {HARD_CACHE.relative_to(ROOT)}")
        req = urllib.request.Request(HARD_URL, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            HARD_CACHE.write_bytes(resp.read())
    by_key: dict[str, list[dict]] = {}
    for line in HARD_CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            by_key.setdefault(_text_key(d["input"]), []).append(d)
    twins = {}
    for row in load():
        matches = by_key.get(_text_key(row["question"]), [])
        if len(matches) != 1:
            continue
        gold = float(matches[0]["target"])
        # Keep only real twins: about 30% of GSM-Hard rows are unchanged copies, and
        # some changed a number but kept the old target (a wrong label).
        if gold >= 0 and abs(gold - row["gold"]) > 1e-9:
            twins[row["idx"]] = {"idx": row["idx"], "question": matches[0]["input"],
                                 "gold": float(matches[0]["target"]), "code": matches[0].get("code", "")}
    return twins


_TEXT_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_CODE_LITERAL = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w]|\.\d)")


def text_numbers(text: str) -> list[float]:
    """Every number written in a problem text, in order ("1,500" -> 1500.0)."""
    out = []
    for m in _TEXT_NUMBER.finditer(text):
        try:
            out.append(float(m.group().rstrip(",").replace(",", "")))
        except ValueError:
            pass
    return out


def changed_number(original: str, twin: str) -> tuple[float, float] | None:
    """(old value, new value) when exactly one value differs between the two texts."""
    a, b = text_numbers(original), text_numbers(twin)
    if len(a) != len(b):
        return None
    pairs = {(x, y) for x, y in zip(a, b) if abs(x - y) > 1e-9}
    return next(iter(pairs)) if len(pairs) == 1 else None


def label_safe(original: str, twin: str, reference_code: str) -> bool:
    """True when the GSM-Hard target is certainly the answer to the twin text.

    GSM-Hard was built by text replacement, which fails in two ways: the new
    number can land inside another number of the text, and a reference program
    that uses the old number twice may have only one use replaced. Neither can
    happen when one value changed and that value occurs once in the original
    text and once in the body of the reference program.
    """
    change = changed_number(original, twin)
    if change is None or not reference_code:
        return False
    old, new = change
    body = [float(x) for x in _CODE_LITERAL.findall(reference_code.split('"""')[-1])]
    uses = sum(abs(v - old) < 1e-9 or abs(v - new) <= 1e-9 * max(1.0, abs(new)) for v in body)
    return text_numbers(original).count(old) == 1 and uses == 1


def _used_before_experiment(n_rows: int) -> set[int]:
    base = split_indices(n_rows)
    return set(base["dev"]) | set(base["heldout"]) | set(paired_indices()) | SMOKE_ROWS


def experiment_pool() -> list[int]:
    """GSM8K rows never used by dev, heldout or paired whose GSM-Hard twin has a safe label."""
    rows = load()
    used = _used_before_experiment(len(rows))
    twins = load_hard()
    return sorted(i for i, t in twins.items()
                  if i not in used and label_safe(rows[i]["question"], t["question"], t["code"]))


def experiment_indices() -> dict[str, list[int]]:
    pool = experiment_pool()
    main = random.Random(EXPERIMENT_SEED).sample(pool, EXPERIMENT_N)
    return {"main": main, "small": main[:EXPERIMENT_SMALL_N], "reserve": sorted(set(pool) - set(main))}


def experiment_split(part: str = "main", dataset: str = "gsmhard", unlock_reserve: bool = False) -> list[dict]:
    """Problems of the 2026-11 experiment. ``part``: main (250), small (100) or reserve (95, locked)."""
    indices = experiment_indices()
    if part not in indices or dataset not in DATASETS:
        raise ValueError(f"part must be one of {sorted(indices)} and dataset one of {DATASETS}")
    if part == "reserve" and not unlock_reserve:
        raise PermissionError("the reserve is locked until the confirmation run")
    source = load() if dataset == "gsm8k" else load_hard()
    return [source[i] for i in indices[part]]


def paired_indices() -> list[int]:
    rows = load()
    base = split_indices(len(rows))
    used = set(base["dev"]) | set(base["heldout"]) | SMOKE_ROWS
    twins = load_hard()
    rest = [i for i in range(len(rows)) if i not in used and i in twins]
    return random.Random(PAIRED_SEED).sample(rest, PAIRED_N)


def split_indices(n_rows: int) -> dict[str, list[int]]:
    dev = random.Random(DEV_SEED).sample(range(n_rows), DEV_N)
    used = set(dev) | SMOKE_ROWS
    rest = [i for i in range(n_rows) if i not in used]
    heldout = random.Random(HELDOUT_SEED).sample(rest, HELDOUT_N)
    return {"dev": dev, "heldout": heldout}


def split(name: str, dataset: str = "gsm8k") -> list[dict]:
    if name not in SPLITS or dataset not in DATASETS:
        raise ValueError(f"split must be one of {SPLITS} and dataset one of {DATASETS}")
    rows = load()
    indices = paired_indices() if name == "paired" else split_indices(len(rows))[name]
    if dataset == "gsm8k":
        return [rows[i] for i in indices]
    twins = load_hard()
    return [twins[i] for i in indices if i in twins]


def fresh_split(n: int, seed: int) -> list[dict]:
    """A new GSM8K sample that avoids dev, heldout, paired, the experiment pool and the smoke-test row."""
    rows = load()
    used = _used_before_experiment(len(rows)) | set(experiment_pool())
    rest = [i for i in range(len(rows)) if i not in used]
    return [rows[i] for i in random.Random(seed).sample(rest, n)]


def is_correct(pred: object, gold: float) -> bool:
    """Numeric match, tolerant of float noise and of a prediction rounded to cents."""
    return numbers_match(pred, gold)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
