# Experiment 2026-11: twin test, calculator and a second model

**Status: draft, not frozen.** The checks on used problems listed under
"Development record" are still to be done. Until the tag `exp-twin-frozen`
exists, `scripts/exp_twin_run.py` refuses to run the experiment.

This file is written before any model sees the problems of the experiment. It
is committed together with the code, the list of problems and the answer key,
and frozen under the tag `exp-twin-frozen`.

## The question

Today's kernel checks its program against a plain chain of thought (CoT).
With large numbers that check almost never fires, because the CoT gets the
arithmetic wrong: on the problems I had already used, only about 20% of the
answers came out confirmed.

The idea under test is that an answer should count as confirmed when two
derivations that can fail in different ways give the same number after the
machine has done all the arithmetic. Three ways of getting there are compared:

1. **Audit.** The machine redoes the arithmetic the CoT wrote down. Free.
2. **Twin test.** The problem is rewritten with small numbers, the CoT solves
   that twin, and the program is run with the twin's numbers. If they agree,
   the program read the problem the way the CoT did.
3. **A second model.** The check and the second reading come from a model of
   another family (Qwen 2.5 7B next to Llama 3.1 8B), so the two derivations
   do not share the same misreadings.

A simpler rival is included: no program checking at all, just a CoT that has
a calculator.

## Problems

- **Pool.** GSM-Hard twins that no earlier split used (not in `dev`,
  `heldout` or `paired`) and whose label passes the mechanical filter of
  `eval/gsm8k.py` (`label_safe`): 345 problems.
- **Main set.** 250 of them, drawn with seed 2028 (`experiment_indices`).
- **Small-number set.** The GSM8K version of the first 100 problems of the
  main set. It is there to check that nothing gets worse where the kernel
  already works. With 100 problems it can only detect a drop of about 8
  points or more.
- **Reserve.** The other 95 problems of the pool. Nothing is run on them and
  nobody looks at them until a confirmation run. The code refuses to load
  them without an explicit flag.

No model had seen any of the 345 problems when this was written. My own
contact with them before today was one aggregate count of code features in
the reference programs.

## Answer key

Every main problem has two independent sources (see `build_gold.py`):

1. the value its GSM-Hard reference program returns, re-executed here because
   the stored target is rounded to 10 decimals;
2. a label produced from the problem text alone by a large language model
   that is not under test (Claude), with Python for the arithmetic. It never
   saw the reference program or the target.

Where the two differ, a second independent labeller solved the problem again.
The rules that turn this into a key are in `build_gold.py` and were fixed
before any model output existed. The outcome is in `gold.json`:

| Status | Problems | Key |
|---|---|---|
| agree | 237 | the reference value |
| rounding | 5 | the reference value and the value rounded as the story implies, both accepted |
| second | 1 | the reference value (the second labeller agreed with it) |
| ambiguous | 1 | the reference value (a labeller lists it as a reasonable reading) |
| relabeled | 5 | the value both independent labellers gave |
| excluded | 1 | none (the agreed answer is negative) |

So 249 problems are in the primary analysis. The five relabeled problems show
a kind of label noise the mechanical filter cannot see: three reference
programs are wrong in general and were right on the original numbers by
coincidence or by hard-coding a derived value, one floors a fraction the text
does not floor, and one follows the GSM8K human label that reads "8:00 AM
until 11:00 PM" as 3 hours.

The same labelling recovered 83 problems outside the pool (their GSM-Hard
label had been dropped or filtered out, but two independent sources agree on
an answer). They are in `reserve_extra.json` and are locked like the reserve,
which now has 95 + 83 = 178 problems.

The primary analysis uses every problem that is not `excluded`. Results are
also reported per label status.

For the small-number set the key is the human answer of GSM8K.

## Scoring

An answer is right when `kernel.arbiter.numbers_match` says it equals an
accepted value: equal up to float noise (relative 1e-9), or one number is the
other rounded to cents or finer without changing it by more than 5%. A system
that gives no number is wrong.

As a sensitivity check, accuracy is also reported counting as right an answer
that is the key rounded to a whole number.

## Three fixes made before the freeze

They apply to every condition, including "today's kernel", and to scoring.
Each one was found on problems already used.

1. `numbers_match` had a relative tolerance of 1e-6. For answers in the tens
   of millions that accepted values 10 or more apart, enough to hide a misread
   small term. On the old `paired` run it had counted 9 wrong CoT answers and
   5 wrong program answers as right (of 200). Now 1e-9.
2. `numbers_match` also accepted any two answers below half a cent as equal,
   zero included. 9 of the 250 main problems have an answer below 0.01.
3. `normalize_answer` rounded program answers to 6 decimals, turning a correct
   5.158e-07 into 1e-06. Now 12 significant digits.

So the numbers published for the `paired` split (README) were measured with a
looser ruler. With the ruler above and the reference programs re-executed, on
the 159 `paired` problems with a safe label, large numbers: CoT 23.9% (the
README says 31.4%), program 66.7% (70.4%), kernel 66.0% (69.8%). Confirmed by
agreement 19.5%, with the audit 34.0%. The small-number results do not change.

## Systems

All are computed from the same saved model outputs by
`eval/twin_experiment.py`. Llama 3.1 8B Instruct (Q4) is the base model, as
in Phase 2. Qwen 2.5 7B Instruct (Q4) is the second model.

| # | System | What it does |
|---|---|---|
| 1 | CoT | plain chain of thought |
| 2 | Program | plan to CDSL, executed, no check |
| 3 | Kernel today | program checked against the CoT; on a disagreement the rule decides (program if the problem has a number with 5 or more significant digits, else CoT) |
| 4 | Kernel + audit | as 3, and a program that matches the CoT with its arithmetic redone is confirmed |
| 5 | Calculator CoT | a CoT that writes each calculation in brackets; the machine fills in the result before the model goes on (`kernel/calc_cot.py`) |
| 6 | Rival | program checked against the calculator CoT. 6a: on a disagreement the program stands. 6b: the calculator CoT stands |
| 7 | Twin test | as 4; a program that is still unconfirmed is run on two fabricated twins and confirmed if it matches the CoT on both; if not, a second reading of the problem (plan sampled at temperature 0.7, compiled to Python) gets the same checks; if nothing is confirmed, the answer of 3 |
| Q1, Q2, Q3, Q7 | Qwen alone | systems 1, 2, 3 and 7 with Qwen in place of Llama |
| 8 | Two models, Llama base | as 7, but the Llama program is checked by Qwen's CoT (on the problem and on the twins), and the second reading is Qwen's program, checked by Llama's CoT. If nothing is confirmed, the answer of 3 |
| 8Q | Two models, Qwen base | as 8 with the roles swapped. If nothing is confirmed, the answer of Q3 |

The twins are fabricated by `kernel/twin.py` from the problem text alone, with
no LLM call and no access to labels: each number with 5 or more significant
digits is replaced by a small value that keeps its order among the other
numbers of the text. One twin uses a multiple of 60 ("round"), the other a
prime ("odd"). A program must match on both.

On the small-number set there are no twins, and the second reading is not
collected, because it cannot change the answer there: 7 gives the answer of 4.

## Primary comparison

**System 7 against system 3: accuracy on the main set.** Two-sided exact
McNemar test, alpha 0.05.

This is the comparison the experiment was designed for, and both systems use
the same model, so the strength of the second model cannot explain a
difference. It decides whether the twin test goes into the kernel.

## Secondary comparisons, fixed in advance

They are reported whatever the primary comparison says, but with 14 systems
on the table only the primary comparison counts as a confirmed result. A
secondary result that looks good is a lead to confirm on the reserve.

1. **8 against 7**, and **8Q against Q7**: does a model of another family, as
   checker and second reader, add something to a model that checks itself?
   Each comparison keeps the base model fixed, so "Qwen is simply better" is
   not an explanation. I will say a second family helps only if both go the
   same way.
2. **7 against 6a and 6b**: the twin test against the simpler rival.
3. **4 against 3, share of confirmed answers**: what the audit buys for free.
4. **Small-number set**: 4 against 3, 5 against 1, 8 against 3, 8Q against Q3.

## Kill criteria

**The twin test is dropped** if any of these holds on the main set:

- its confirmed answers are right less than 95% of the time;
- it confirms less than 10 points more of the problems than the audit alone
  (system 4);
- it does not beat today's kernel by at least 5 points of accuracy with
  p < 0.05 (the primary comparison).

**The rival wins and the twin test is dropped** if the better of 6a and 6b is
within 2 points of system 7 in accuracy, uses no more tokens, and confirms at
most 5 points less of the problems with at least 95% of them right.

## Predictions

Written before the run. Intervals are what I would be surprised to see
violated. Main set, 250 problems:

| System | Accuracy | Confirmed | Right when confirmed |
|---|---|---|---|
| 1 CoT | 24% (18 to 31) | | |
| 2 Program | 68% (60 to 75) | | |
| 3 Kernel today | 68% (60 to 75) | 20% (14 to 27) | 96% (90 to 100) |
| 4 Kernel + audit | same answers as 3 | 34% (27 to 42) | 95% (90 to 100) |
| 5 Calculator CoT | 55% (40 to 68) | | |
| 6a, 6b Rival | 68% (60 to 75), 60% (48 to 70) | 42% (30 to 52) | 96% (90 to 100) |
| 7 Twin test | 71% (63 to 78) | 50% (40 to 62) | 95% (90 to 99) |
| Q1 Qwen CoT | 35% (20 to 50) | | |
| Q2 Qwen program | 66% (52 to 78) | | |
| Q3 Qwen kernel today | 66% (52 to 78) | 30% (15 to 45) | 96% (90 to 100) |
| Q7 Qwen twin test | 69% (55 to 80) | 55% (40 to 68) | 95% (90 to 99) |
| 8 Two models, Llama base | 76% (68 to 83) | 58% (46 to 70) | 97% (93 to 100) |
| 8Q Two models, Qwen base | 75% (65 to 84) | 58% (46 to 70) | 97% (93 to 100) |

- **Primary comparison.** I expect 7 to beat 3 by about 3 points (0 to 7).
  That is below the 5 points the kill criteria ask for, so my forecast is
  that the twin test with a single model is dropped: it should pass the
  coverage criterion, sit on the edge of the 95% criterion, and fail the
  accuracy criterion. I put the chance that it passes all three at 1 in 4.
  The reason is what I saw on used problems: when the same model writes the
  program and the chain of thought, they sometimes share a misreading, and a
  second reading of the same text by the same model rarely differs.
- **Second model.** I expect 8 to beat 7 by about 5 points (0 to 10) and 8Q
  to beat Q7 by about 5 (0 to 10). This is the result I am least sure of,
  because I have no measurement with Qwen yet.
- **Rival.** I expect the calculator CoT to recover a large part of the
  arithmetic failures of CoT, but to confirm fewer answers than the twin test.
- **Small-number set.** CoT 86% (78 to 93), kernel today 86% (78 to 93).
  System 4 differs from 3 in at most 1 problem of 100. System 5 is within
  4 points of system 1. System 8 is within 3 points of system 3.

Where the numbers come from: the old `paired` runs rescored with the ruler
above (CoT 23.9%, program 66.7%, confirmed 19.5% and 34.0% with the audit),
and a dry run of system 7 on those runs with the original small-number
problem standing in for the fabricated twin (accuracy +2.5 points, confirmed
67%, 96% of them right once label artifacts are set aside).

## Runs

With the Llama model loaded in LM Studio:

    python scripts/exp_twin_run.py --model meta-llama-3.1-8b-instruct --part main  --calc --second
    python scripts/exp_twin_run.py --model meta-llama-3.1-8b-instruct --part small --calc

With the Qwen model loaded:

    python scripts/exp_twin_run.py --model qwen2.5-7b-instruct --part main  --second
    python scripts/exp_twin_run.py --model qwen2.5-7b-instruct --part small

Then, only after the four runs are complete:

    python scripts/exp_twin_report.py --save results/exp-twin/summary.json

A run that stops continues where it stopped when the same command is given
again. The runs never load the answer key. Expected GPU time on an RX 7600:
7 to 12 hours in total.

## Exploratory

Everything else `scripts/exp_twin_report.py` prints is exploratory: each twin
kind alone, the pipeline without twins, confirmation by two programs agreeing,
results per label status and per way of confirming, tokens.

## Development record

Not written yet. Before the freeze, the calculator CoT prompt, the fabricated
twins and the Qwen prompts are checked on used problems only (the `paired`
split). Anything changed after those checks is listed here, and the
predictions above may be revised with what those checks show.

Decided offline so far, on used problems: the twin fabricator's margin (a
replacement must exceed twice the largest smaller number of the text). With
it, on 272 used problems the round twin has a positive reference answer in
270 and the odd twin in 267.
