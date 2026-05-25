# Cognitive Kernel v0.1 — Design Document

**Status:** DRAFT — awaiting user review
**Date:** 2026-05-24
**Author (vision):** User (Brazilian undergraduate, federal university AI student) — articulated during brainstorming. *Replace with your name when publishing.*
**Author (technical scaffolding):** drafted with Claude Opus 4.7
**Target environment:** Free Colab/Kaggle + 5 TB Google Drive + local PC as editor
**Estimated timeline:** 5–6 months solo, ~10–15 h/week

---

## 1. Vision (multi-year program — context, not scope)

This project is the **first concrete contribution** to a multi-year personal research program — informally named **Cognitive Kernel** — whose long-term thesis is:

> Parameter count measures internal raw capacity, but **does not measure cognitive efficiency of a system that knows how to convert ambiguous natural language into verifiable state**. Comparing brains alone — ignoring paper, calculator, language, laboratory, code, tools and feedback — has never been a fair measure of practical intelligence. The Cognitive Kernel applies this idea to small language models, treating them as **controllers of an externalized cognitive architecture** rather than as encyclopedias compressed into weights.

The full Cognitive Kernel vision includes, at minimum, the following principles (each backed by active research in 2024–2026):

- **Externalized memory of multiple semantic types** (factual, procedural, episodic, causal, structural, error history) instead of monolithic parametric memory.
- **Explicit planning and decomposition** before generation.
- **Adaptive computation** (cheap thinking for easy tasks, expensive thinking for hard ones).
- **Modular specialization** (different components for language, math, code, simulation, planning).
- **Internal simulable model of the problem** rather than text-only reasoning.
- **Native verification** (self-critique, tool execution, contradiction search).
- **Continual learning via memory updates** rather than retraining all weights.
- **Intermediate executable representation** (NL → DSL/AST/graph → execution → text) — *structure as source of truth, text as projection*.

**The full system is a research program likely spanning 5–10 years and multiple people.** This document scopes a single tractable first project that establishes a foundation and tests the **single sharpest sub-claim** of the program empirically.

---

## 2. Scope of this project

### What this project IS

A **vertical slice** of the Cognitive Kernel — every architectural component exists in **minimum viable form**, end-to-end, runnable — combined with **deep development of one component** (the **Compiler NL → CDSL**, see §5) which becomes the empirical centerpiece and the primary publishable contribution.

### What this project is NOT

- ❌ NOT a general-purpose conversational LLM.
- ❌ NOT a claim that a small model is "more intelligent" than GPT-4 / Claude / Gemini in any general sense. **The frontier trillion-parameter models dominate in world knowledge, fluency, multilingualism, creativity, breadth of transfer, and robustness, and we make no contrary claim.**
- ❌ NOT a fine-tune of an existing open-weights model (e.g. Llama, Qwen, Mistral). The base model is trained from scratch — this is a methodological commitment to keep the experiment "pure" and the contribution clearly architectural.
- ❌ NOT a claim resolved by "we added tools to a small model". Adding tools to a small model is well established. The contribution is the **intermediate executable representation** (the CDSL) that makes verification cheap and correction structured; tools are downstream of that representation, not the headline.
- ❌ NOT the full Cognitive Kernel. The other 5–6 components beyond the Compiler exist in "cru" form only and are explicitly out of scope for deep optimization in this project.

### Domain of demonstration

**Verifiable Python code + mathematical/logical reasoning**, in English (international focus for visibility), with optional secondary PT-BR contribution.

### The honest central claim (verbatim from user's articulation, lightly polished)

> The core claim is **not** that a small model is globally more capable than a trillion-parameter frontier model. The claim is that, **for verifiable domains, cognitive efficiency should be measured at the system level rather than the parameter level**. A small model coupled to an intermediate executable representation, typed memory, tool runtime, and verification loop can outperform much larger vanilla models because the architecture converts ambiguous natural language into structured states where search, execution, and correction are cheap. The contribution is therefore not "small model beats GPT-4"; it is **"structured cognitive scaffolding shifts the bottleneck from parametric memorization to state transformation and verification."**

This is the framing used in the abstract and throughout the project's communication.

---

## 3. System architecture (the 7-component vertical slice)

```
                    ┌───────────────────────────────────────────┐
                    │            INPUT (NL problem)             │
                    └────────────────────┬──────────────────────┘
                                         ↓
                    ┌───────────────────────────────────────────┐
   [1] CLASSIFIER   │  Classify problem type                    │
                    │  (math / code / logic / mixed)            │
                    └────────────────────┬──────────────────────┘
                                         ↓
                    ┌───────────────────────────────────────────┐
   [2] COMPILER ★   │  Compile NL → CDSL program (AST)          │
                    │  ★ DEEP COMPONENT ★                       │
                    └────────────────────┬──────────────────────┘
                                         ↓
                    ┌───────────────────────────────────────────┐
   [3] MEMORY       │  Lookup typed memory                      │
                    │  (factual / procedural / causal / ep.)    │
                    └────────────────────┬──────────────────────┘
                                         ↓
                    ┌───────────────────────────────────────────┐
   [4] RUNTIME      │  Execute CDSL program                     │
                    │  (Python / SymPy / search / search tools) │
                    └────────────────────┬──────────────────────┘
                                         ↓
                    ┌───────────────────────────────────────────┐
   [5] VERIFIER     │  Verify outputs against constraints       │
                    │  (exec passes? types correct? sat?)       │
                    └────────────────────┬──────────────────────┘
                                         ↓
                    ┌───────────────────────────────────────────┐
   [6] REVISER      │  If failed: re-prompt with error trace    │
                    │  → loop back to [2] up to N times         │
                    └────────────────────┬──────────────────────┘
                                         ↓
                    ┌───────────────────────────────────────────┐
   [7] EPISODE LOG  │  Persist (problem, program, result,       │
                    │  success/failure) into episodic memory    │
                    └────────────────────┬──────────────────────┘
                                         ↓
                    ┌───────────────────────────────────────────┐
                    │       OUTPUT (answer + reasoning trace)   │
                    └───────────────────────────────────────────┘
```

### Component status

| # | Component | Form | Approx. LOC | Reuses existing |
|---|---|---|---|---|
| 0 | Base model (100M from-scratch) | full | uses `model.py` + extensions | ✅ `model.py` |
| 1 | Classifier | cru (minimal) | ~150 | partial |
| 2 | **Compiler NL→CDSL** ★ | **deep** | ~1500 (incl. DSL + bootstrap) | new |
| 3 | Memory (typed) | cru | ~300 | new |
| 4 | Runtime | cru | ~400 | new |
| 5 | Verifier | cru | ~300 | new |
| 6 | Reviser | cru | ~150 | new |
| 7 | Episode log | cru | ~150 | new |
| — | Eval harness | minimal | ~400 | new |
| — | Notebook / training scripts | — | ~500 | new |
| — | Web demo (existing UI) | refactor | ~150 modifications | ✅ `server.py` + `static/` |

**Total new code: ~4000 LOC** over 5–6 months. Existing `model.py`, `tokenizer.py`, `trainer.py`, `server.py` are extended, not replaced.

---

## 4. Base model (component 0)

### Architecture

Reuse [model.py](../../../model.py) as-is (modern LLaMA-style: RMSNorm + RoPE + SwiGLU + weight-tied embedding) with 4 surgical modifications:

1. **Flash Attention**: replace manual attention computation (lines 112–121) with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. Mathematically identical; ~40% VRAM savings on T4 / A100.
2. **Optional Grouped-Query Attention (GQA)**: parameterizable `n_kv_head < n_head`, default `n_kv_head = n_head // 4`. Reduces KV cache memory in inference (relevant for the verifier loop with many rollouts).
3. **Gradient checkpointing hook**: wrap each `TransformerBlock` forward in `torch.utils.checkpoint.checkpoint` when `config.grad_checkpoint = True`.
4. **`<|endoftext|>`, `<|cdsl_start|>`, `<|cdsl_end|>`, `<|tool_call|>`, `<|tool_result|>`** added as special tokens for structured outputs.

### Dimensions (~100 M parameters)

```python
n_embd       = 768     # embedding dim
n_head       = 12      # query heads
n_kv_head    = 4       # K/V heads (GQA)
n_layer      = 12      # transformer blocks
max_seq_len  = 2048    # context window
vocab_size   = 32000   # BPE vocab (extended for CDSL tokens)
```

Final parameter count: **~100 M** (embedding 24.6 M shared with `lm_head`, transformer blocks ~75 M). Chosen as "small enough that the Cognitive Kernel thesis is interesting (we want a model that is clearly small compared to gigantic baselines) and large enough that the base model can plausibly learn to emit syntactically valid CDSL programs."

### Tokenizer

Hybrid strategy:
- Keep [tokenizer.py](../../../tokenizer.py) as documented didactic artifact (pure-Python BPE, slow but pedagogically clear).
- Add `tokenizer_fast.py` using HuggingFace `tokenizers` (Rust, ~100× faster). Same BPE algorithm, different implementation.
- Vocab size: **32 000** (room for code idioms, CDSL primitives, common math symbols).
- Special tokens reserved for CDSL boundary markers, tool-call markers, and chain-of-execution markers.

---

## 5. The Cognitive DSL (CDSL) — the deep component

This is the project's empirical centerpiece and primary publishable contribution. **The thesis-testing artifact.**

### 5.1 Design principles

CDSL is a small, typed, functional-with-effects intermediate language that the base model **emits** in place of natural-language chains-of-thought. Programs in CDSL are:

- **Deterministic when executed** — no randomness in the language itself; randomness only in the *generation* by the model.
- **Verifiable by construction** — every primitive has explicit pre/post conditions that the verifier can check.
- **Composable** — programs can call other programs (recursion bounded by depth limit).
- **Reflectable** — the program AST is itself first-class data, so the reviser can edit a generated program without regenerating from scratch.

### 5.2 Primitives (v0.1 — to be ablated, see §7)

```
(program  :type <category> <body>)
(let       <var> <expr>)
(call_tool <tool_id> <args>)         ; python_exec | sympy | calculator | grep | search
(lookup    <memory_type> <query>)    ; factual | procedural | causal | episodic
(decompose <problem> → [<sub>...])
(solve     <subproblem>)              ; recursive call to compiler
(assert    <condition>)
(compare   <a> <b> :op <op>)
(if        <cond> <then> <else>)
(cond      [<test> <branch>] ...)
(loop      <init> <test> <step> <body>)
(case      <var> [<pattern> <branch>] ...)
(simulate  <rules> <initial-state> <steps>)
(prove     <statement> :using <axioms>)   ; calls Z3 (later: Lean)
(verify    <result> :against <expected>)
(return    <value>)
(request_input <prompt>)              ; rare; for interactive demos
```

Roughly **15 primitives**. The ablation study in §7 tests 8 vs 15 vs 25 primitive variants to identify the trade-off between expressiveness and learnability for a 100 M model.

### 5.3 Example program

NL input: *"Quantos múltiplos de 7 existem entre 1 e 100?"* (or English equivalent)

CDSL program emitted by compiler:

```lisp
(program :type counting-with-constraint
  (let n (call_tool python_exec "list(range(1, 101))"))
  (let multiples (call_tool python_exec
                   "[x for x in n if x % 7 == 0]" :env {n n}))
  (let count (call_tool python_exec "len(multiples)" :env {multiples multiples}))
  (assert (compare count :op > 0))
  (verify count :against (call_tool python_exec
                           "len([x for x in range(1,101) if x%7==0])"))
  (return count))
```

Runtime executes deterministically. Verifier confirms `count > 0` and that the verification call matches. Output: `14`.

### 5.4 Compiler training pipeline (the deep work)

**Stage 1 — Pre-train base model** (no CDSL yet)
- Pre-train the 100 M base on a curated mix focused on Python + math + structured text.
- Token budget: ~5 B tokens (Chinchilla-relaxed for 100 M).
- Mix: 35% Python (The Stack v2, filtered), 20% math/reasoning (GSM8K-expanded + MATH + synthetic), 20% English technical prose, 15% Lisp/Scheme/sexp-style text (so model gets used to bracket-heavy syntax), 10% misc.
- Output: a coherent base model that can predict tokens reasonably in code + math contexts but knows nothing about CDSL specifically.

**Stage 2 — Hand-bootstrap CDSL** (supervised)
- Construct ~5 000 hand-curated pairs `(problem_NL, program_CDSL)` semi-automatically:
  - Take problems from GSM8K, MBPP, HumanEval, ARC.
  - Use a free-tier API (Gemini, OpenRouter free models) as a **drafting assistant** to propose CDSL programs from NL problems.
  - **Human-curate** the drafts (manual quality control on at least 1 000; automated execution-based filter on the rest).
- Fine-tune the base model on this set with standard supervised loss. Output: a model that can emit syntactically valid CDSL programs ~30–60% of the time and semantically correct programs ~10–25% of the time on held-out problems.

**Stage 3 — Self-bootstrap by execution feedback** (the AlphaZero-style loop)
- For each problem in a large unlabeled pool: model emits N candidate programs (N = 64), runtime executes all, verifier filters those that pass.
- Successful programs → new training data.
- Re-train (fine-tune) the model on the union of original Stage-2 data and harvested successes.
- Iterate ~10 rounds.
- Risks (with mitigations) explicitly tracked: difficulty collapse, mode collapse, regression on original distribution. See §10.

**Stage 4 — DSL expressiveness ablation** (the headline experiment)
- Train three compiler variants on three DSL sizes:
  - **CDSL-mini**: 8 primitives (minimal)
  - **CDSL-mid**: 15 primitives (baseline)
  - **CDSL-rich**: 25 primitives (extended with `match`, `unify`, `try`, `chain`, etc.)
- Measure: parse rate, execution success, transferability across benchmarks, training compute cost.
- **The headline finding** is the trade-off curve. Whether it's monotonic, U-shaped, or saturating is itself the publishable result.

---

## 6. The cru components (vertical slice)

Each gets a **minimum-viable** implementation. Out of scope for deep optimization in v0.1; explicitly named as future work.

### 6.1 Classifier (cru)
Single linear probe on top of frozen base model embeddings of the input. 4 classes: `math`, `code`, `logic`, `mixed`. Trained on ~500 hand-labeled examples from benchmarks. Used to bias the prompt template passed to the compiler.

### 6.2 Memory — typed (cru)
4 banks of FAISS vector indices over `sentence-transformers/all-MiniLM-L6-v2` embeddings (frozen, ~22 M parameters, used only for retrieval):
- **factual** — populated with key facts extracted from base-model pretraining corpus (snippets of fact-rich text).
- **procedural** — populated with "how-to" snippets (idioms, recipes, algorithmic patterns).
- **causal** — populated with cause-effect statements mined heuristically from text.
- **episodic** — initially empty; populated by component 7 as the system runs.

CDSL `(lookup <type> <query>)` calls the appropriate index. Returns top-k snippets joined into the compiler's context.

### 6.3 Runtime (cru)
A small Python interpreter for CDSL. Each primitive maps to a Python implementation:
- `call_tool python_exec` → `subprocess` to sandboxed Python (timeout + memory cap).
- `call_tool sympy` → `sympy.sympify`, `simplify`, `solve`, etc.
- `call_tool search` → optional, uses DuckDuckGo HTML scraping (free, no API key).
- `prove` → Z3 (`z3-solver` library).
- Control flow (`if`, `loop`, `case`) → straightforward interpreter.

### 6.4 Verifier (cru)
Set of deterministic checkers:
- `exec_passes(program, test_cases)` — runs program against test cases.
- `output_matches(actual, expected, op)` — equality with tolerance.
- `type_correct(value, type)` — runtime type check.
- `constraint_satisfied(assertion)` — eval boolean assertion.

Returns `(passed: bool, error_trace: str | None)`. The error trace is used by the reviser to construct a corrective prompt.

### 6.5 Reviser (cru)
A loop with maximum 5 iterations. Each iteration:
1. If previous attempt failed, build a corrective prompt: original problem + failed program + error trace + instruction "fix the program".
2. Call compiler again to get a new program.
3. Run runtime → verifier.
4. If passes → return. If fails → iterate.

### 6.6 Episode log (cru)
SQLite database (`episode_log.db`). Each row: `(timestamp, problem_text, problem_type, final_program, num_attempts, success, error_trace, embedding)`. The `embedding` column is the input-problem embedding (for retrieval by future runs as episodic memory).

---

## 7. Training and data strategy

### 7.1 Base-model pre-training data

**Total budget:** ~5 B tokens (Chinchilla-relaxed for 100 M).

| Source | % | Notes |
|---|---|---|
| The Stack v2 (Python, ≥10 stars, dedup) | 35 % | Code substrate |
| GSM8K + MATH + extended | 20 % | Math reasoning |
| English technical prose (OpenStax + OpenWebText filtered) | 20 % | General reasoning + fluency |
| Lisp/Scheme/JSON/S-expr text | 15 % | Bracket-heavy syntax familiarity (helps CDSL emission later) |
| Misc (PT-BR Wikipedia tech subset, README files) | 10 % | Diversification + PT-BR bonus |

Data lives on Google Drive. Pre-processing pipeline (dedupe → filter → tokenize → shard) produces `.bin` files (nanoGPT format, 2 bytes/token) read via streaming `DataLoader`.

### 7.2 CDSL bootstrap data (Stage 2)

- ~5 000 hand-curated `(NL, CDSL)` pairs.
- Sources of NL: GSM8K (40 %), MBPP (30 %), HumanEval (10 %), ARC (10 %), custom hand-written (10 %).
- Drafting via free-tier API; human curation on a sample; execution-based filter on the rest.

### 7.3 Self-bootstrap (Stage 3)

- Pool of ~50 000 unlabeled problems (mix of synthetic generated by the model itself + benchmark questions held out from Stage 2).
- 10 rounds of: emit-execute-filter-retrain.
- Per round: ~10 hours of T4 time (Colab session).

### 7.4 Infrastructure

- **Code editor:** local PC (228 MB free on C: is enough for editing + git; everything heavy lives in Drive or Colab).
- **Training:** Colab Free (T4, primary) + Kaggle (P100, backup + parallel eval) + Lightning AI Studio Free (prototyping).
- **Persistence:** Google Drive 5 TB.
  - `MyDrive/cognitive-kernel/checkpoints/` — model checkpoints (2–3 GB each, keep last 3 + best).
  - `MyDrive/cognitive-kernel/data/` — preprocessed shards.
  - `MyDrive/cognitive-kernel/logs/` — training logs.
- **Cross-session resume:** trainer saves complete state (weights + optimizer + scheduler + RNG + step) every 500 steps. Next session detects and resumes bit-identically.
- **Live monitoring:** existing [server.py](../../../server.py) runs locally; Colab notebook opens an ngrok / Cloudflare Tunnel to its training-side WebSocket; local browser connects.
- **Metrics:** Weights & Biases (free tier for academic use) for all standard logging.

### 7.5 Memory efficiency stack (mandatory for 100 M on free T4)

- BF16 mixed precision via `torch.amp`.
- Gradient checkpointing on `TransformerBlock`.
- 8-bit AdamW via `bitsandbytes`.
- Gradient accumulation (physical batch 16 × accumulation 16 = effective batch 256 sequences × 2048 tokens ≈ 524 k tokens/step).
- Flash Attention via `F.scaled_dot_product_attention`.

### 7.6 Optimizer

AdamW as default (proven stable). **Muon optimizer** (Keller Jordan, 2024) as opt-in flag for 2-D matrices (attention + MLP weights), keeping AdamW for embeddings and normalizations. Expected ~25–35 % wall-clock speed-up if it works on this scale; falls back to AdamW on instability.

---

## 8. Evaluation plan

### 8.1 Benchmarks

| Benchmark | What it measures | Why we include it |
|---|---|---|
| **HumanEval** (pass@1, pass@10 with N=64) | Python code generation correctness | Primary code metric, most-cited |
| **MBPP** (pass@1) | Basic Python correctness | Secondary, broader Python distribution |
| **GSM8K** (accuracy with N=64 rollouts) | Grade-school math reasoning | Primary math metric |
| **MATH** subset (level 1–3) | Harder math | Stretch goal |
| **ARC-AGI** small subset | Abstract reasoning | Tests transfer, low expectations |
| **Custom CDSL parse rate** | % of emitted programs that parse | Compiler health |
| **Custom CDSL exec rate** | % that execute without error | Pipeline health |
| **PT-BR mini-bench** (optional) | Tests transfer to PT | Brazilian visibility bonus |

### 8.2 Baselines

For every benchmark, report **at least three baselines** alongside the Cognitive Kernel:

1. **CoT vanilla on base model** — same 100 M model, plain chain-of-thought, no CDSL.
2. **PoT (Program-of-Thought) on base model** — same model, emits Python directly (no CDSL).
3. **CoT on GPT-3.5 / GPT-4o / Claude Haiku** (via free or cheap API quotas) — to anchor scale comparison.

The headline plots compare **(base model + CDSL pipeline)** to **(GPT-3.5/4o vanilla CoT)**.

### 8.3 Target outcomes (3 scenarios, honest)

#### Pessimistic (~30 % probability)
- HumanEval pass@1 ≥ 10 %, pass@10 ≥ 25 %.
- GSM8K acc ≥ 8 %.
- Compiler parse rate ≥ 70 %, exec rate ≥ 40 %.
- Outcome: workshop-paper-worthy; thesis qualitatively supported; quantitative gap to bigger models remains.

#### Realistic (~50 % probability)
- HumanEval pass@1 ≥ 25 %, pass@10 ≥ 50 %.
- GSM8K acc ≥ 20 %.
- Compiler exec rate ≥ 65 %.
- **The headline runs:** the 100 M Cognitive Kernel beats GPT-3.5 vanilla CoT and matches Llama-2-7B vanilla CoT on HumanEval pass@10 and GSM8K. Workshop-strong, possibly conference-quality with a strong writeup.

#### Optimistic (~20 % probability)
- HumanEval pass@1 ≥ 40 %, pass@10 ≥ 70 %.
- GSM8K acc ≥ 35 %.
- **The headline runs:** the 100 M Cognitive Kernel beats GPT-4o vanilla on HumanEval pass@10 in pass-style metrics, while remaining honest that GPT-4o would dominate again under equivalent scaffolding. Genuine conference-quality result; press potential.

### 8.4 Honest claims permitted (and disallowed) in writeup

**Allowed:**
- "The 100 M Cognitive Kernel system outperforms [X] (parameter-count Y) on [benchmark Z] under standard vanilla inference."
- "Structured intermediate representations shift the cognitive bottleneck from parametric memorization to state transformation and verification, as evidenced by [ablation results]."
- "DSL primitive count exhibits a [shape] trade-off between expressiveness and learnability at the 100 M scale."

**Disallowed:**
- ❌ "Our 100 M model is more intelligent than GPT-4."
- ❌ "We have compressed a trillion-parameter model into 100 M."
- ❌ "Parameter count does not matter."

The point throughout: **claims should be defensible against a hostile reviewer in 30 seconds.**

---

## 9. Phases and timeline

| Phase | Weeks | Goal | Definition of done |
|---|---|---|---|
| **Phase 0 — Foundation hardening** | 1–2 | Make trainer survive Colab + add mixed precision + checkpoint/resume + wandb + Drive integration | Train [model.py](../../../model.py) at 100 M for 1 000 steps across 3 simulated Colab disconnects with bit-identical resume |
| **Phase 1 — Pre-train base 100 M** | 3–6 | Train base model from scratch on the data mix | Base model evals: perplexity ≤ 4.0 on held-out Python; coherent generation in code + math contexts |
| **Phase 2 — Vertical slice (cru, all 7 components)** | 7–10 | Build every Cognitive Kernel component in minimal form; end-to-end runs | Demo notebook: input GSM8K problem → 7-stage pipeline → answer (any quality) |
| **Phase 3 — Compiler deep dive: Stage 2 supervised bootstrap** | 11–14 | Hand-bootstrap dataset of ~5 000 pairs; fine-tune compiler | Parse rate ≥ 50 %, exec rate ≥ 20 % on held-out problems |
| **Phase 4 — Compiler deep dive: Stage 3 self-bootstrap** | 15–20 | 10 rounds of self-improvement | Exec rate doubles from Stage 2; no mode collapse |
| **Phase 5 — DSL ablation experiments** | 21–22 | Train 8/15/25-primitive variants | Three trained variants + comparative table |
| **Phase 6 — Full evaluation and writeup** | 23–24 | Run all benchmarks; write technical report | Report (~15–25 pages) + public GitHub repo + Hugging Face Spaces demo |

**Total: ~24 weeks (~6 months) at 10–15 h/week.**

---

## 10. Risks and mitigations

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| **Self-bootstrap collapses to easy problems** (compiler-generator generates only trivial programs because they succeed easily) | 40 % | High | Explicit difficulty-stratified curriculum in self-bootstrap pool; require difficulty diversification per round; track and reject if entropy of program AST shapes drops below threshold |
| **Base model too weak after 5 B tokens** (100 M is small; not enough capacity to even emit syntactically valid CDSL reliably) | 30 % | High | Stage 2 hand-bootstrap mitigates; if parse rate < 30 % after Stage 2 → escalate base model to 175–200 M (3 weeks added, fits on T4 with stricter optimizations) |
| **Self-bootstrap diverges / mode collapse** (model converges to one program template that "always passes" trivially) | 25 % | Medium | Keep original Stage-2 distribution as anchor in every round; diversity penalty in selection of harvested data |
| **DSL design needs iteration** (initial 15 primitives turn out to be wrong; some are dead weight, others are missing) | 70 % | Low | Iteration is expected and planned. Ablation in Stage 4 IS the iteration framed as an experiment |
| **Free-tier compute gives out** (Colab Free repeatedly disconnects mid-session; Kaggle quota exhausted) | 25 % | Medium | Lightning AI Studio (third backup); strategically schedule heavy runs for late-night UTC (Colab availability higher); checkpoint every 500 steps means lost work is bounded |
| **Free API quota for synthetic data exhausted** (Gemini free, OpenRouter free run out during Stage 2 drafting) | 50 % | Low | Spread across multiple free accounts (within ToS); use only as drafting aid (human curation is mandatory anyway); fall back to fully manual curation for smaller dataset |
| **Drive bandwidth bottleneck during training data load** | 15 % | Low | Pre-shard data into ~256 MB chunks; sequential streaming is bandwidth-OK for Drive (this only fails for random small reads, which our DataLoader doesn't do) |
| **Project loses momentum / pivots midway** | 50 % | (Personal — meta-risk) | Each phase produces a self-contained artifact: even abandoning at end of Phase 1 leaves a usable 100 M base model + lessons. Phase boundaries are natural pause points |

---

## 11. Repository structure (target end-state)

```
LLMPessoal/
├── README.md                       # project overview + headline results
├── requirements.txt
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-05-24-cognitive-kernel-v0.1-design.md   # this document
├── model/
│   ├── model.py                    # existing, extended (Flash Attn, GQA, grad ckpt)
│   ├── tokenizer.py                # existing didactic BPE (kept)
│   └── tokenizer_fast.py           # NEW: HuggingFace tokenizers wrapper
├── data/
│   ├── prepare_pretrain.py
│   ├── prepare_cdsl_bootstrap.py
│   ├── dataset.py                  # streaming DataLoader from Drive
│   └── synthesize_with_api.py      # NEW: API-assisted draft generation
├── training/
│   ├── trainer.py                  # existing, refactored (AMP, ckpt, accum, muon)
│   ├── optim_muon.py               # NEW: Muon optimizer
│   ├── schedule.py                 # LR scheduling
│   └── resume.py                   # checkpoint/resume across sessions
├── cdsl/
│   ├── grammar.py                  # CDSL grammar definition
│   ├── parser.py                   # NL→AST validation
│   ├── runtime.py                  # interpreter
│   ├── primitives.py               # primitive implementations
│   └── examples/                   # canonical example programs
├── kernel/                         # the 7-component pipeline
│   ├── classifier.py
│   ├── compiler.py                 # the deep component; wraps base model + CDSL grammar
│   ├── memory.py
│   ├── verifier.py
│   ├── reviser.py
│   └── episode_log.py
├── eval/
│   ├── humaneval.py
│   ├── gsm8k.py
│   ├── mbpp.py
│   ├── compiler_metrics.py         # parse rate, exec rate
│   └── ablation_dsl_size.py
├── serve/
│   ├── server.py                   # existing, extended
│   ├── tunnel.py                   # NEW: ngrok / Cloudflare bootstrap
│   └── static/                     # existing
├── scripts/
│   ├── train_colab.ipynb
│   ├── self_bootstrap_round.py
│   └── prepare_data_colab.ipynb
├── configs/
│   ├── base_110m.yaml
│   ├── compiler_finetune.yaml
│   └── self_bootstrap.yaml
└── results/                        # generated; gitignored
    ├── runs/
    └── reports/
```

---

## 12. Out-of-scope follow-ups (the multi-year program)

Listed explicitly so we know what we're **deferring**, not abandoning:

- **Deep work on Memory** (component 3): semantic-type-aware retrieval routing learned end-to-end (not hand-routed).
- **Deep work on Verifier** (component 5): learned process reward models, MCTS over CDSL program space.
- **World-model component**: full internal simulator for physical/procedural domains; closer to LeCun's JEPA.
- **Continual-learning component**: memory-edit-based weight updates (ROME-style) for incorporating new knowledge without retraining.
- **Multimodal extension**: image/diagram inputs feeding the classifier and compiler.
- **Scaling-laws study**: how do the same architectural choices transfer to 500 M, 1 B, 7 B base models?

Each of these is candidate material for the **next** project in the program (Cognitive Kernel v0.2, v0.3, …).

---

## 13. References (initial)

Papers, blog posts, and systems the design draws from. Not exhaustive; will grow during execution.

- LeCun, Y. (2022–2025). *A Path Towards Autonomous Machine Intelligence* (JEPA position paper and subsequent talks).
- Kojima et al. (2022). *Large Language Models are Zero-Shot Reasoners* (CoT baseline).
- Chen et al. (2022). *Program of Thoughts Prompting* (PoT baseline).
- Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting* (tool-augmented baseline).
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (self-bootstrap inspiration).
- Gunasekar et al. (2023). *Textbooks Are All You Need* (Phi-1, curated data thesis).
- DreamCoder: Ellis et al. (2021). *DreamCoder: bootstrapping inductive program synthesis with wake-sleep library learning* (DSL bootstrap precedent).
- AlphaProof / AlphaGeometry (DeepMind, 2024). *Solving IMO problems with neuro-symbolic systems* (neural-policy-over-formal-language precedent).
- Keller Jordan et al. (2024). *Muon: A New Optimizer for nanoGPT-Speedrun*.
- Karpathy, A. (2023). *nanoGPT* (training-loop reference).
- bitsandbytes (Dettmers et al.).
- Flash Attention 2 (Dao et al., 2023).

---

## 14. Acknowledgements

The vision in §1 and the precise framing of the central claim in §2 (verbatim adopted) are entirely the user's articulation, developed during brainstorming. The technical scaffolding is a joint product of that vision with engineering structure proposed by the AI assistant. This is a personal research project; no institutional affiliation is claimed.

---

**END OF DESIGN DOCUMENT (v0.1 DRAFT)**
