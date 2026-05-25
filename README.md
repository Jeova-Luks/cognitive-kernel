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

## Phase 0 — Foundation Hardening (current)

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

The critical Phase 0 test is `tests/test_trainer.py::test_resume_is_bit_identical`
— training N steps uninterrupted must produce a bit-identical loss trajectory
to training N/2 steps + checkpoint + load + N/2 more steps in a fresh process.

## Smoke-test the full pipeline in Colab

1. Open [scripts/train_colab.ipynb](scripts/train_colab.ipynb) in Google Colab.
2. In Cell 5, set `CONFIG = 'configs/smoke_100m.yaml'`.
3. Before Cell 6, add a cell: `!python scripts/make_smoke_shards.py`
4. Run all cells. The first checkpoint appears in
   `/content/drive/MyDrive/cognitive-kernel/checkpoints/smoke_100m/` after step 25.
5. Force-disconnect the Colab runtime, reconnect, re-run all cells. Training
   should print `Resuming from ckpt_step_NNNNN.pt` and continue without error.

## Roadmap (multi-year program)

- ✅ **Phase 0** — Foundation hardening (this branch)
- 🟡 **Phase 1** — Pre-train 100M base on curated data
- ⚪ **Phase 2** — Build the 7-component Cognitive Kernel pipeline (cru)
- ⚪ **Phase 3** — Compiler NL → CDSL (deep dive, supervised bootstrap)
- ⚪ **Phase 4** — Compiler self-bootstrap via execution feedback
- ⚪ **Phase 5** — DSL expressiveness ablation
- ⚪ **Phase 6** — Full evaluation, ablations, technical report
