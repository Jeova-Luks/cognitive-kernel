# Cognitive Kernel

First subproject of a multi-year personal research program. See
[docs/superpowers/specs/2026-05-24-cognitive-kernel-v0.1-design.md](docs/superpowers/specs/2026-05-24-cognitive-kernel-v0.1-design.md)
for the full design and central thesis.

> The claim is not that a small model is globally more capable than a
> trillion-parameter frontier model. The claim is that, for verifiable
> domains, cognitive efficiency should be measured at the system level
> rather than the parameter level. A small model coupled to an intermediate
> executable representation, typed memory, tool runtime, and verification
> loop can outperform much larger vanilla models because the architecture
> converts ambiguous natural language into structured states where search,
> execution, and correction are cheap.

## Phase 0 - Foundation Hardening (current)

Trainer that survives Google Colab session disconnects with bit-identical
checkpoint resume. Modern 100M-parameter Transformer base (RMSNorm + RoPE
+ SwiGLU + Flash Attention + Grouped-Query Attention + gradient
checkpointing). YAML-backed config, streaming Drive-friendly data loader,
8-bit AdamW, Weights & Biases logging.

See [docs/superpowers/plans/2026-05-24-phase-0-foundation-hardening.md](docs/superpowers/plans/2026-05-24-phase-0-foundation-hardening.md)
for the implementation plan and acceptance criteria.

## Run the tests

In a Python env with deps installed (GitHub Codespaces is the easiest):

```bash
pip install -r requirements.txt
python -m pytest -v
```

The critical Phase 0 test is `tests/test_trainer.py::test_resume_is_bit_identical`,
training N steps uninterrupted must produce a bit-identical loss trajectory
to training N/2 steps + checkpoint + load + N/2 more steps in a fresh process.

## Smoke-test the full pipeline in Colab

1. Open [scripts/train_colab.ipynb](scripts/train_colab.ipynb) in Google Colab.
2. In Cell 5, set `CONFIG = 'configs/smoke_100m.yaml'`.
3. Before Cell 6, add a cell: `!python scripts/make_smoke_shards.py`
4. Run all cells. The first checkpoint appears in
   `/content/drive/MyDrive/cognitive-kernel/checkpoints/smoke_100m/` after step 25.
5. Force-disconnect the Colab runtime, reconnect, re-run all cells. Training
   should print `Resuming from ckpt_step_NNNNN.pt` and continue without error.

## Phase 2 - Vertical slice with a local stand-in LLM

All 7 components exist in cru form: `kernel/` (classifier, typed memory,
reader/planner, compiler, verifier, reviser, arbiter, episode log, pipeline)
and `cdsl/` (parser, runtime, tools). Until the Phase 1 base model is trained,
a local OpenAI-compatible LLM stands in as classifier, reader and compiler
(default: LM Studio with Llama 3.1 8B Instruct; memory uses its nomic
embedding model).

```
problem -> [1] classifier -> [3] memory -> [2a] reader writes a plan (no arithmetic)
        -> [2b] compiler renders the plan as a CDSL program -> [4] runtime executes it
        -> [5a] structural verifier -> [6] reviser (repairs syntax/runtime errors, max 3 tries)
        -> [5b] cross-check against an independent chain-of-thought answer
               agree: confidence "high" | disagree: a referee picks one, confidence "low"
               (default referee is a rule: trust the program if the numbers are large, else CoT)
        -> [7] episode log (confirmed episodes become worked examples)
```

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
plain chain-of-thought (`cot`), plan -> CDSL (`cdsl`), the same plan compiled
to Python (`python`), and the full kernel (`arbiter`: the CDSL answer
cross-checked against CoT, with a referee on disagreement). Ablation switches:
`--no-plan`, `--self-checks`, `--max-attempts`, `--referee choose|resolve`,
`--no-episodic`. Splits are fixed in `eval/gsm8k.py`: tune on `dev`, report on
`heldout` / `paired`, and draw a fresh split after looking at either.

Deviations from the spec, all explained in module docstrings: the classifier
prompts the LLM (no linear probe yet); memory uses brute-force cosine instead
of FAISS; the runtime adds arithmetic builtins and does not implement
`decompose`, `solve`, `loop`, `case`, `simulate`, `prove` or `request_input`;
a natural-language plan precedes the CDSL program (the spec has CDSL replace
natural-language reasoning; the plan may not contain arithmetic, and
`--no-plan` measures what it buys); and "verified" means agreement with an
independent derivation, not the program's own `assert` / `verify`.
`python_exec` runs model-written code in a `python -I` subprocess with a
timeout: crash and loop containment, not a security sandbox.

### Phase 2 results with the stand-in LLM (2026-11-11)

Llama 3.1 8B Instruct (Q4, LM Studio), 200 held-out problems (`paired` split),
each evaluated twice: as in GSM8K, and as its GSM-Hard twin (same text, one
number replaced by a large one). Accuracy with 95% Wilson intervals.

| System | GSM8K | GSM-Hard | Both (400) | Tokens/problem |
|---|---|---|---|---|
| CoT (no program) | 86.5% [81.1, 90.5] | 26.0% [20.4, 32.5] | 56.2% | 242 / 309 |
| Plan -> CDSL | 72.5% [65.9, 78.2] | 60.5% [53.6, 67.0] | 66.5% | 267 / 278 |
| Plan -> Python | 75.0% [68.6, 80.5] | 63.0% [56.1, 69.4] | 69.0% | 244 / 256 |
| Full kernel (CDSL + cross-check + rule referee) | 86.5% [81.1, 90.5] | 60.0% [53.1, 66.5] | 73.2% | 509 / 586 |

What these numbers do and do not show:

- With small numbers the bottleneck is reading the problem, and plain CoT reads
  better than plan -> program (McNemar p = 0.0001 for CDSL). With large numbers
  the bottleneck is arithmetic, and executed programs win (p < 0.0001).
- The full kernel is the only system that holds up in both regimes, but part of
  that is by construction: its referee trusts the program when the problem has
  a number with 5+ significant digits, which is exactly what separates the two
  benchmarks. On GSM8K it never beats CoT; it only matches it.
- Agreement between the program and CoT is a strong confidence signal: correct
  in 134/136 cases on GSM8K and 41/45 on GSM-Hard (175/181 = 97% overall),
  against 61% and 51% when they disagree.
- CDSL and Python are statistically indistinguishable as targets (p = 0.23 and
  0.18). With an untrained compiler the DSL buys nothing over Python; whether a
  compiler trained on CDSL changes that is the question for Phases 3-5.
- GSM-Hard labels are noisy (11 of these 200 targets are below 0.001 where the
  answer should be in the millions), so absolute GSM-Hard numbers are
  underestimates; the noise is the same for every system.
- GSM8K test is public since 2021 and may be in the LLM's training data, which
  favours the CoT baseline.

Earlier version of the pipeline (no plan, self-written assert/verify, up to 5
repair attempts, LLM referee that re-solves), on a different set of 200
held-out GSM8K problems (`heldout` split): CoT 79.0%, CDSL 61.5%, Python
65.0%, arbitration 75.5%. Relative to CoT on the same problems, the plan
narrowed the CDSL gap from 17.5 to 14.0 points, raised first-attempt execution
from 88% to 98% and cut attempts per problem from 2.16 to 1.03. The gain is
smaller than on the 50 development problems (64% -> 76%), as expected of
anything tuned on a development set.

Raw outputs are in `results/phase2/` (git-ignored): one `results.jsonl` with
every plan, program, trace and error per problem, plus `summary.json`.

## Roadmap (multi-year program)

- ✅ **Phase 0** - Foundation hardening (this branch)
- 🟡 **Phase 1** - Pre-train 100M base on curated data
- 🟡 **Phase 2** - Build the 7-component Cognitive Kernel pipeline (cru) - runs end to end with a stand-in LLM
- ⚪ **Phase 3** - Compiler NL → CDSL (deep dive, supervised bootstrap)
- ⚪ **Phase 4** - Compiler self-bootstrap via execution feedback
- ⚪ **Phase 5** - DSL expressiveness ablation
- ⚪ **Phase 6** - Full evaluation, ablations, technical report
