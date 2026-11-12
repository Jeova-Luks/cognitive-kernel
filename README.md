# Cognitive Kernel

This is the first project of a personal research program I plan to work on
for several years. The full design and the thesis behind it are in
[docs/superpowers/specs/2026-05-24-cognitive-kernel-v0.1-design.md](docs/superpowers/specs/2026-05-24-cognitive-kernel-v0.1-design.md).

The idea in one paragraph: I am not claiming that a small model is more
capable than a huge frontier model. The claim is that, in domains where
answers can be checked, efficiency should be measured for the whole system
and not only by parameter count. A small model that writes an executable
intermediate program, and has memory, tools and a verification loop around
it, can beat much larger models used on their own, because the program turns
vague natural language into something that can be executed, checked and
corrected cheaply.

## Phase 0 - Foundation hardening (done)

A trainer that survives Google Colab disconnects and resumes from a
checkpoint with exactly the same results. The base is a 100M-parameter
Transformer with RMSNorm, RoPE, SwiGLU, Flash Attention, Grouped-Query
Attention and gradient checkpointing. It also has a YAML config, a streaming
data loader that works from Google Drive, 8-bit AdamW and Weights & Biases
logging.

The implementation plan and acceptance criteria are in
[docs/superpowers/plans/2026-05-24-phase-0-foundation-hardening.md](docs/superpowers/plans/2026-05-24-phase-0-foundation-hardening.md).

## Running the tests

In a Python environment with the dependencies installed (GitHub Codespaces is
the easiest way):

```bash
pip install -r requirements.txt
python -m pytest -v
```

The most important Phase 0 test is
`tests/test_trainer.py::test_resume_is_bit_identical`. It trains N steps
without interruption, then trains N/2 steps, saves, loads in a new process
and trains the other N/2. The two loss curves must be identical bit for bit.

## Smoke test of the training pipeline in Colab

1. Open [scripts/train_colab.ipynb](scripts/train_colab.ipynb) in Google Colab.
2. In Cell 5, set `CONFIG = 'configs/smoke_100m.yaml'`.
3. Before Cell 6, add a cell with `!python scripts/make_smoke_shards.py`.
4. Run all cells. The first checkpoint shows up in
   `/content/drive/MyDrive/cognitive-kernel/checkpoints/smoke_100m/` after step 25.
5. Disconnect the Colab runtime on purpose, reconnect and run all cells again.
   Training should print `Resuming from ckpt_step_NNNNN.pt` and continue.

## Phase 2 - Vertical slice with a local LLM as stand-in

All 7 components exist in a minimal form. The pipeline is in `kernel/`
(classifier, typed memory, reader/planner, compiler, verifier, reviser,
arbiter, episode log) and the language is in `cdsl/` (parser, runtime, tools).

The 100M base model from Phase 1 is not trained yet. Until it is, a local LLM
with an OpenAI-compatible API plays the roles of classifier, reader and
compiler. By default that is Llama 3.1 8B Instruct running in LM Studio, and
the memory uses LM Studio's nomic embedding model.

How one problem flows through the system:

```
problem -> [1] classifier -> [3] memory -> [2a] reader writes a plan (no arithmetic)
        -> [2b] compiler turns the plan into a CDSL program -> [4] runtime executes it
        -> [5a] structural verifier -> [6] reviser (fixes syntax/runtime errors, up to 3 tries)
        -> [5b] cross-check against a separate chain-of-thought answer
               same answer: confidence "high" | different: a referee picks one, confidence "low"
               (the default referee is a rule: trust the program if the numbers are large, else CoT)
        -> [7] episode log (confirmed episodes become worked examples)
```

How to run it:

```bash
lms server start
lms load meta-llama-3.1-8b-instruct
python scripts/phase2_demo.py "Janet's ducks lay 16 eggs per day. ..."   # one problem, full trace
python scripts/phase2_demo.py --target python "..."                      # emit Python instead of CDSL
python scripts/phase2_eval.py --split dev                                # 50 GSM8K development problems
python scripts/phase2_eval.py --split paired                             # 200 held-out problems
python scripts/phase2_eval.py --split paired --dataset gsmhard           # the same 200 with large numbers
python -m pytest tests/test_kernel.py                                    # offline tests, no LLM needed
```

`phase2_eval.py` runs four systems on the same problems with the same LLM:

- `cot`: plain chain-of-thought, no program
- `cdsl`: plan, then a CDSL program
- `python`: the same plan compiled to Python
- `arbiter`: the full kernel, where the CDSL answer is cross-checked against
  CoT and a referee decides when they disagree

There are switches to turn parts off and measure what each one is worth:
`--no-plan`, `--self-checks`, `--max-attempts`, `--referee choose|resolve`
and `--no-episodic`. The splits are fixed in `eval/gsm8k.py`. I tune on `dev`
and report on `heldout` or `paired`. Once a split has been looked at, new
claims need a fresh one.

Where this differs from the design document (each point is explained in the
docstring of the module):

- The classifier is a prompt to the LLM. There is no linear probe yet.
- Memory uses brute-force cosine similarity instead of FAISS.
- The runtime has arithmetic builtins and does not implement `decompose`,
  `solve`, `loop`, `case`, `simulate`, `prove` or `request_input`.
- A plan in natural language comes before the CDSL program. In the design,
  CDSL replaces natural-language reasoning. The plan is not allowed to do
  arithmetic, and `--no-plan` measures how much it helps.
- "Verified" means the answer matches a separate derivation. It does not mean
  the program passed its own `assert` or `verify`.
- `python_exec` runs code written by the model in a `python -I` subprocess
  with a timeout. That contains crashes and infinite loops. It is not a
  security sandbox.

### Phase 2 results with the stand-in LLM (2026-11-11)

Model: Llama 3.1 8B Instruct (Q4, LM Studio). I used 200 held-out problems
(the `paired` split) and evaluated each one twice: as it appears in GSM8K,
and as its GSM-Hard twin, which has the same text with one number replaced by
a large one. The table shows accuracy with 95% Wilson intervals.

| System | GSM8K | GSM-Hard | Both (400) | Tokens/problem |
|---|---|---|---|---|
| CoT (no program) | 86.5% [81.1, 90.5] | 26.0% [20.4, 32.5] | 56.2% | 242 / 309 |
| Plan to CDSL | 72.5% [65.9, 78.2] | 60.5% [53.6, 67.0] | 66.5% | 267 / 278 |
| Plan to Python | 75.0% [68.6, 80.5] | 63.0% [56.1, 69.4] | 69.0% | 244 / 256 |
| Full kernel (CDSL, cross-check, rule referee) | 86.5% [81.1, 90.5] | 60.0% [53.1, 66.5] | 73.2% | 509 / 586 |

What I take from these numbers, and what I do not:

- With small numbers the hard part is reading the problem, and plain CoT
  reads better than plan plus program (McNemar p = 0.0001 for CDSL). With
  large numbers the hard part is the arithmetic, and executed programs win
  (p < 0.0001).
- The full kernel is the only system that does well in both cases. Part of
  that is by construction: its referee trusts the program when the problem
  has a number with 5 or more significant digits, and number size is exactly
  what separates the two benchmarks. On GSM8K it never beats CoT, it only
  ties.
- When the program and CoT agree, the answer is very likely right: 134 of 136
  on GSM8K and 41 of 45 on GSM-Hard (175 of 181, or 97%). When they disagree,
  accuracy is 61% and 51%.
- CDSL and Python give statistically the same results (p = 0.23 and 0.18).
  With a compiler that was never trained on CDSL, the DSL adds nothing over
  Python. Whether a compiler trained on CDSL changes this is the question for
  Phases 3 to 5.
- Some GSM-Hard labels do not match their text. The twins were made by text
  replacement, so in 13 of these 200 problems the new number also landed
  inside another number, and in others the reference program had only one of
  two uses of the number replaced. The label is safe in 159 of the 200: one
  number changed, and its value occurs once in the text and once in the
  reference program. On those 159:

  | System | GSM8K | GSM-Hard |
  |---|---|---|
  | CoT (no program) | 86.8% | 31.4% |
  | Plan to CDSL | 70.4% | 70.4% |
  | Plan to Python | 73.0% | 74.2% |
  | Full kernel | 86.8% | 69.8% |

  So the drop of the program systems in the first table (72.5% to 60.5% for
  CDSL) comes from the labels, not from the size of the numbers. A program
  that reads the problem right gives the right answer at any size. The CoT
  drop is real.
- The GSM8K test set has been public since 2021 and may be in the LLM's
  training data. That helps the CoT baseline.

I also evaluated an earlier version of the pipeline (no plan, checks written
by the program itself, up to 5 repair attempts, and an LLM referee that
solves the problem again) on a different set of 200 held-out GSM8K problems,
the `heldout` split. It got CoT 79.0%, CDSL 61.5%, Python 65.0% and
arbitration 75.5%. Compared with CoT on the same problems, adding the plan
reduced the CDSL gap from 17.5 to 14.0 points, raised first-attempt execution
from 88% to 98%, and cut attempts per problem from 2.16 to 1.03. The gain is
smaller than what I saw on the 50 development problems (64% to 76%), which is
normal for anything tuned on a development set.

The raw outputs are in `results/phase2/`, which is git-ignored. Each run has
a `results.jsonl` with the plan, program, trace and error of every problem,
and a `summary.json`.

## Roadmap

- Phase 0: foundation hardening (done)
- Phase 1: pre-train the 100M base on curated data (in progress)
- Phase 2: the 7-component pipeline in minimal form (runs end to end with a stand-in LLM)
- Phase 3: compiler from natural language to CDSL, supervised bootstrap
- Phase 4: compiler self-bootstrap from execution feedback
- Phase 5: DSL expressiveness ablation
- Phase 6: full evaluation, ablations and technical report
