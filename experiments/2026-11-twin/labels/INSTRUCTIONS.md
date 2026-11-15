# Independent answer labels for math word problems

You are producing answer labels for a research benchmark. Your labels will be
compared with labels from another source, so accuracy matters far more than
speed, and you must work from the problem text alone.

## Input

A JSONL file, one problem per line: `{"id": <int>, "question": "<text>"}`.

The problems are grade-school word problems in which ONE number was replaced,
by a script, with a large or odd value (often 7 digits). So answers can be
huge, fractional, negative or unrealistic, and a few texts may have become
nonsense. That is expected. Do not "fix" the numbers: use them exactly as
written.

## Hard rules

- Use ONLY the batch file you were given. Do not open any other data file on
  this machine (in particular nothing under a `results` folder), do not search
  the web, and do not try to find reference answers. If you recognise a
  problem, ignore what you remember: the numbers are different.
- Do no arithmetic in your head. Every answer must come out of Python
  (integers or `fractions.Fraction`, converted at the end).
- Solve each problem on its own. Read it twice: once to decide the method,
  once after computing, checking each number is used in the right role
  (which is the rate, which the count, "more than" / "less than", "each",
  "per", "twice", "half", "the rest", unit conversions, what exactly is asked).

## What to produce

Write a Python script in your work folder that holds one expression per
problem and writes the output file. Then run it. The output is a JSON list,
one object per problem, same order as the input:

```json
{"id": 123, "answer": 105954, "exact": 105953.075, "flags": ["rounding"],
 "method": "ceil(students / per_bus)", "note": "needs whole buses"}
```

- `answer`: the single number a careful teacher would accept as the answer,
  in the unit the question asks for (a percentage as the number of percent,
  money in the currency unit asked). Use `null` only if the problem cannot
  be computed at all.
- `exact`: the unrounded value of the same method. Equal to `answer` unless
  you rounded.
- `method`: one short line with the formula, using names, not just numbers.
- `note`: optional, one line.
- `flags`: a list, usually empty. Use only these:
  - `rounding`: the story itself forces a whole number and the exact value is
    not whole (whole buses needed, full boxes that can be filled, complete
    trips). Do NOT round just because a count of people or objects came out
    fractional: in that case keep the exact value and use `unrealistic`.
  - `unrealistic`: the arithmetic is well defined but the result is absurd in
    the story (fractional people, a negative amount left, more removed than
    there was, a speed or price that makes no sense). Still give the exact
    result of what the text says.
  - `ambiguous`: two reasonable readings give different answers. Give the
    more natural one and put the other value in `note` as `alt=<value>`.
  - `ill_posed`: the text contradicts itself or lacks information, so no
    answer can be computed. `answer` may be `null`.

Give non-terminating decimals with at least 8 significant digits. Do not
round money to cents unless the problem asks for it.

## When you finish

Check that the output file parses as JSON, has one object for every id in
the input, and that every `answer` is a number or null. Reply with: the
output path, how many problems carry each flag, and the ids you were least
sure about (with one line each on why).
